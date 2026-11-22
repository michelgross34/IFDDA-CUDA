#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
using cuda_utils::check_cuda;
using cuda_utils::dotc;
using cuda_utils::get_cublas_handle;
using cuda_utils::norm2;

struct Term {
    int column;
    CUCOMPLEX coefficient;
};

//-----------------------------------------------------------
CUCOMPLEX divv(CUCOMPLEX a, CUCOMPLEX b) {
    return CUFFT_DIV(a, b);
}

//-----------------------------------------------------------
CUCOMPLEX neg(CUCOMPLEX a) {
    return CUFFT_SUB(CUFFT_ZERO, a);
}

//-----------------------------------------------------------
bool combine(CUCOMPLEX *work, CUCOMPLEX *xsol, int lda, int n, int dst,
             const std::vector<Term> &terms, bool accumulate = false) {
    int cols[10];
    CUCOMPLEX coeff[10];
    if (terms.empty() || terms.size() > 10)
        return false;
    for (size_t i = 0; i < terms.size(); ++i) {
        cols[i] = terms[i].column;
        coeff[i] = terms[i].coefficient;
    }
    return check_cuda(gpbicgstabl_combine_wrapper(work, xsol, lda, n, dst, cols, coeff,
                                                  static_cast<int>(terms.size()),
                                                  accumulate ? 1 : 0),
                      "GPBICGSTABL CUDA", "vector combination");
}

//-----------------------------------------------------------
bool copy_column(CUCOMPLEX *work, const CUCOMPLEX *xr, int col, int lda, int n) {
    return check_cuda(idrs_copy_xr_wrapper(work, xr, col, lda, n), "GPBICGSTABL CUDA",
                      "save requested Ax vector");
}

//-----------------------------------------------------------
bool run_gpbicgstabl(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, COMPLEX *xsol_host,
                     CU_REAL *norm, COMPLEX *work_host, CUCOMPLEX *alpha, CUCOMPLEX *beta,
                     CUCOMPLEX *sigma, CUCOMPLEX *rho, const int *lp, const int *ldap,
                     const int *ndimp, const int *nlarp, int *nou, int *nrp, int *npp, int *nsp,
                     int *nqp, int *jp, int *nstat, int *steperr, int *itno, const int *maxit,
                     const CU_REAL *tole, const CU_REAL *tol) {
    if (!xi || !xr || !b || !xsol_host || !norm || !work_host || !alpha || !beta || !sigma ||
        !rho || !lp || !ldap || !ndimp || !nlarp || !nou || !nrp || !npp || !nsp || !nqp || !jp ||
        !nstat || !steperr || !itno || !maxit || !tole || !tol)
        return false;
    const int l = *lp, lda = *ldap, n = *ndimp, nlar = *nlarp;
    if ((l != 2 && l != 4 && l != 8) || n <= 0 || lda < n || nlar < 4 * l + 8) {
        *nstat = -1;
        *steperr = 10;
        return true;
    }
    const size_t count = size_t(n), work_count = size_t(lda) * size_t(nlar);
    const bool initial = (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4 && *nou != 5);
    CUCOMPLEX *gxi = nullptr, *gxr = nullptr, *gb = nullptr, *gw = nullptr;
    if (!get_gpbicg_device_arrays_resident(xi, xr, b, work_host, count, work_count, initial, &gxi,
                                           &gxr, &gb, &gw)) {
        *nstat = -1;
        *steperr = 13;
        return true;
    }
    CUCOMPLEX *gxsol = get_tfqmr_device_x_from_device(xsol_host, gxi, count, initial);
    cublasHandle_t handle = get_cublas_handle();
    if (!gxsol || !handle) {
        *nstat = -1;
        *steperr = 14;
        return true;
    }
    auto finish_xi = [&]() {
        return cuda_data::copy_device_to_host(xi, gxi, count * sizeof(CUCOMPLEX),
                                              cuda_utils::TransferCategory::Other) == cudaSuccess;
    };
    auto fail = [&](int code) {
        *nstat = -1;
        *steperr = code;
        finish_xi();
    };
    auto dc = [&](int c1, int c2, CUCOMPLEX *out) {
        return dotc(handle, n, gw + size_t(c1) * lda, gw + size_t(c2) * lda, out,
                    "GPBICGSTABL CUDA");
    };
    auto publish_xi = [&](int col) {
        return check_cuda(idrs_copy_work_to_xi_wrapper(gxi, gw, col, lda, n), "GPBICGSTABL CUDA",
                          "publish reverse-communication vector");
    };

    *nstat = 0;
    *steperr = -1;
    if (initial) {
        if (!norm2(handle, n, gb, norm, "GPBICGSTABL CUDA") || !std::isfinite(*norm) ||
            *norm <= CU_REAL(0)) {
            fail(11);
            return true;
        }
        *nou = 1;
        if (!finish_xi()) {
            fail(15);
        }
        return true;
    }

    if (*nou == 1) {
        *nrp = 8;
        *npp = *nrp + l + 1;
        *nqp = *npp + l + 1;
        *nsp = *nqp + l;
        *rho = CUFFT_ZERO;
        // r0tilde=b-Ax, r0=p0=r0tilde and z=0.
        // Form column 1 from b-xr using a tiny dedicated CUDA expression.
        // Reuse the combine kernel after copying XR to the first work column.
        if (cuda_utils::tracked_memcpy(gw + size_t(*nrp - 1) * lda, gxr, count * sizeof(CUCOMPLEX),
                                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            fail(15);
            return true;
        }
        // w(:,1)=b-xr is assembled as b plus -xr into column 1.
        // The first term is copied through an already available column, then combined.
        if (cuda_utils::tracked_memcpy(gw + size_t(1) * lda, gb, count * sizeof(CUCOMPLEX),
                                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            fail(15);
            return true;
        }
        // col 1 currently contains b; subtract xr through col nr (temporarily restored below).
        if (!combine(gw, gxsol, lda, n, 0,
                     {{1, CUFFT_REAL(CU_REAL(1))}, {*nrp - 1, CUFFT_REAL(CU_REAL(-1))}})) {
            fail(16);
            return true;
        }
        // r0 and p0 receive r0tilde; initialize z.
        if (!combine(gw, gxsol, lda, n, *nrp - 1, {{0, CUFFT_REAL(CU_REAL(1))}}) ||
            !combine(gw, gxsol, lda, n, *npp - 1, {{0, CUFFT_REAL(CU_REAL(1))}}) ||
            !combine(gw, gxsol, lda, n, 2, {{0, CUFFT_ZERO}}) || !dc(0, *nrp - 1, rho)) {
            fail(16);
            return true;
        }
        *jp = 0;
        ++(*jp);
        if (!publish_xi(*npp + *jp - 2)) {
            fail(16);
            return true;
        }
        *nou = 2;
        if (!finish_xi()) {
            fail(15);
        }
        return true;
    }

    if (*nou == 2) {
        if (!copy_column(gw, gxr, *npp + *jp - 1, lda, n) || !dc(0, *npp + *jp - 1, sigma)) {
            fail(16);
            return true;
        }
        if (std::hypot(sigma->x, sigma->y) == 0.0) {
            fail(1);
            return true;
        }
        *alpha = divv(*rho, *sigma);
        if (!combine(gw, gxsol, lda, n, -1, {{*npp - 1, *alpha}}, true)) {
            fail(16);
            return true;
        }
        for (int i = 0; i < *jp; ++i)
            if (!combine(gw, gxsol, lda, n, *nrp + i - 1,
                         {{*npp + i, neg(*alpha)}, {*nrp + i - 1, CUFFT_REAL(CU_REAL(1))}})) {
                fail(16);
                return true;
            }
        if (!publish_xi(*nrp + *jp - 2)) {
            fail(16);
            return true;
        }
        *nou = 3;
        if (!finish_xi()) {
            fail(15);
        }
        return true;
    }

    if (*nou == 3) {
        if (!copy_column(gw, gxr, *nrp + *jp - 1, lda, n) || !dc(0, *nrp + *jp - 1, rho) ||
            std::hypot(sigma->x, sigma->y) == 0.0) {
            fail(16);
            return true;
        }
        *beta = divv(*rho, *sigma);
        for (int i = 0; i <= *jp; ++i)
            if (!combine(gw, gxsol, lda, n, *npp + i - 1,
                         {{*nrp + i - 1, CUFFT_REAL(CU_REAL(1))}, {*npp + i - 1, neg(*beta)}})) {
                fail(16);
                return true;
            }
        if (*jp <= l - 1) {
            ++(*jp);
            if (!publish_xi(*npp + *jp - 2)) {
                fail(16);
                return true;
            }
            *nou = 2;
            if (!finish_xi()) {
                fail(15);
            }
            return true;
        }
        // Initialize r', p', s_i and q_i before the first stabilization.
        if (!combine(gw, gxsol, lda, n, 4, {{*nrp - 1, CUFFT_REAL(CU_REAL(1))}}) ||
            !combine(gw, gxsol, lda, n, 5, {{*npp - 1, CUFFT_REAL(CU_REAL(1))}})) {
            fail(16);
            return true;
        }
        for (int i = 0; i < l - 1; ++i)
            if (!combine(gw, gxsol, lda, n, *nsp + i - 1, {{*nrp + i, CUFFT_REAL(CU_REAL(1))}})) {
                fail(16);
                return true;
            }
        for (int i = 0; i < l; ++i)
            if (!combine(gw, gxsol, lda, n, *nqp + i - 1, {{*npp + i, CUFFT_REAL(CU_REAL(1))}})) {
                fail(16);
                return true;
            }
        // Dense least-squares system M^H M zeta=M^H r0.
        CUCOMPLEX mat[9][9]{};
        CUCOMPLEX zeta[9]{};
        for (int i = 0; i < l; ++i)
            for (int j = 0; j < l; ++j)
                if (!dc(*nrp + i, *nrp + j, &mat[i][j])) {
                    fail(16);
                    return true;
                }
        for (int i = 0; i < l; ++i)
            if (!dc(*nrp + i, *nrp - 1, &zeta[i]) || !dc(*nrp + i, 1, &mat[i][l])) {
                fail(16);
                return true;
            }
        int dense_info = 0;
        if (!check_cuda(cuda_data::solve_small_dense_system_host(&mat[0][0], zeta, l, &dense_info),
                        "GPBICGSTABL CUDA", "GPU dense coefficient solve")) {
            fail(17);
            return true;
        }
        if (dense_info != 0) {
            fail(17);
            return true;
        }
        for (int i = 0; i < l; ++i)
            if (!combine(gw, gxsol, lda, n, 2, {{*nrp + i - 1, zeta[i]}}, true)) {
                fail(16);
                return true;
            }
        if (!combine(gw, gxsol, lda, n, -1, {{2, CUFFT_REAL(CU_REAL(1))}}, true)) {
            fail(16);
            return true;
        }
        for (int i = 0; i < l; ++i) {
            if (!combine(gw, gxsol, lda, n, *nrp - 1,
                         {{*nrp + i, neg(zeta[i])}, {*nrp - 1, CUFFT_REAL(CU_REAL(1))}}) ||
                !combine(gw, gxsol, lda, n, *npp - 1,
                         {{*npp + i, neg(zeta[i])}, {*npp - 1, CUFFT_REAL(CU_REAL(1))}})) {
                fail(16);
                return true;
            }
        }
        *itno = -1;
        ++(*itno);
        // Shared transition into the large loop starts by returning p_0 for A*p_0.
        if (!combine(gw, gxsol, lda, n, 1,
                     {{4, CUFFT_REAL(CU_REAL(1))}, {*nrp - 1, CUFFT_REAL(CU_REAL(-1))}}) ||
            !combine(gw, gxsol, lda, n, 3,
                     {{5, CUFFT_REAL(CU_REAL(1))}, {*npp - 1, CUFFT_REAL(CU_REAL(-1))}}) ||
            !dc(0, *nrp - 1, rho)) {
            fail(16);
            return true;
        }
        *jp = 1;
        if (!publish_xi(*npp - 1)) {
            fail(16);
            return true;
        }
        *nou = 4;
        if (!finish_xi()) {
            fail(15);
        }
        return true;
    }

    if (*nou == 4) {
        if (!copy_column(gw, gxr, *npp + *jp - 1, lda, n) ||
            !combine(gw, gxsol, lda, n, 6,
                     {{*nqp - 1, CUFFT_REAL(CU_REAL(1))}, {*npp, CUFFT_REAL(CU_REAL(-1))}}) ||
            !dc(0, *npp + *jp - 1, sigma)) {
            fail(16);
            return true;
        }
        if (std::hypot(sigma->x, sigma->y) == 0.0) {
            fail(4);
            return true;
        }
        *alpha = divv(*rho, *sigma);
        if (!combine(gw, gxsol, lda, n, -1, {{*npp - 1, *alpha}}, true) ||
            !combine(gw, gxsol, lda, n, 2, {{3, neg(*alpha)}, {2, CUFFT_REAL(CU_REAL(1))}}) ||
            !combine(gw, gxsol, lda, n, 1, {{6, neg(*alpha)}, {1, CUFFT_REAL(CU_REAL(1))}})) {
            fail(16);
            return true;
        }
        for (int i = 0; i < *jp; ++i)
            if (!combine(gw, gxsol, lda, n, *nrp + i - 1,
                         {{*npp + i, neg(*alpha)}, {*nrp + i - 1, CUFFT_REAL(CU_REAL(1))}})) {
                fail(16);
                return true;
            }
        if (!publish_xi(*nrp + *jp - 2)) {
            fail(16);
            return true;
        }
        *nou = 5;
        if (!finish_xi()) {
            fail(15);
        }
        return true;
    }

    if (*nou == 5) {
        if (!copy_column(gw, gxr, *nrp + *jp - 1, lda, n) || !dc(0, *nrp + *jp - 1, rho) ||
            std::hypot(sigma->x, sigma->y) == 0.0) {
            fail(16);
            return true;
        }
        *beta = divv(*rho, *sigma);
        for (int i = 0; i <= *jp; ++i)
            if (!combine(gw, gxsol, lda, n, *npp + i - 1,
                         {{*nrp + i - 1, CUFFT_REAL(CU_REAL(1))}, {*npp + i - 1, neg(*beta)}})) {
                fail(16);
                return true;
            }
        if (!combine(gw, gxsol, lda, n, 3, {{1, CUFFT_REAL(CU_REAL(1))}, {3, neg(*beta)}})) {
            fail(16);
            return true;
        }
        if (*jp <= l - 1) {
            ++(*jp);
            if (*jp > 1) {
                for (int i = 0; i <= l - *jp; ++i)
                    if (!combine(
                            gw, gxsol, lda, n, *nsp + i - 1,
                            {{*nqp + i, neg(*alpha)}, {*nsp + i - 1, CUFFT_REAL(CU_REAL(1))}})) {
                        fail(16);
                        return true;
                    }
                for (int i = 0; i <= l - *jp; ++i)
                    if (!combine(
                            gw, gxsol, lda, n, *nqp + i - 1,
                            {{*nsp + i - 1, CUFFT_REAL(CU_REAL(1))}, {*nqp + i - 1, neg(*beta)}})) {
                        fail(16);
                        return true;
                    }
            }
            if (!publish_xi(*npp + *jp - 2)) {
                fail(16);
                return true;
            }
            *nou = 4;
            if (!finish_xi()) {
                fail(15);
            }
            return true;
        }

        if (!combine(gw, gxsol, lda, n, 4, {{*nrp - 1, CUFFT_REAL(CU_REAL(1))}}) ||
            !combine(gw, gxsol, lda, n, 5, {{*npp - 1, CUFFT_REAL(CU_REAL(1))}})) {
            fail(16);
            return true;
        }
        for (int i = 0; i < l - 1; ++i)
            if (!combine(gw, gxsol, lda, n, *nsp + i - 1, {{*nrp + i, CUFFT_REAL(CU_REAL(1))}})) {
                fail(16);
                return true;
            }
        for (int i = 0; i < l; ++i)
            if (!combine(gw, gxsol, lda, n, *nqp + i - 1, {{*npp + i, CUFFT_REAL(CU_REAL(1))}})) {
                fail(16);
                return true;
            }

        CUCOMPLEX mat[9][9]{};
        CUCOMPLEX vect[9]{};
        const int dense = l + 1;
        for (int i = 0; i < l; ++i)
            for (int j = 0; j < l; ++j)
                if (!dc(*nrp + i, *nrp + j, &mat[i][j])) {
                    fail(16);
                    return true;
                }
        for (int i = 0; i < l; ++i) {
            if (!dc(*nrp + i, *nrp - 1, &vect[i]) || !dc(*nrp + i, 1, &mat[i][l])) {
                fail(16);
                return true;
            }
            mat[l][i] = CUFFT_CONJ(mat[i][l]);
        }
        if (!dc(1, 1, &mat[l][l]) || !dc(1, *nrp - 1, &vect[l])) {
            fail(16);
            return true;
        }
        int dense_info = 0;
        if (!check_cuda(
                cuda_data::solve_small_dense_system_host(&mat[0][0], vect, dense, &dense_info),
                "GPBICGSTABL CUDA", "GPU dense least-squares solve")) {
            fail(17);
            return true;
        }
        if (dense_info != 0) {
            fail(17);
            return true;
        }
        CUCOMPLEX zeta[8];
        for (int i = 0; i < l; ++i)
            zeta[i] = vect[i];
        const CUCOMPLEX eta = vect[l];
        if (l == 8) {
            std::vector<Term> zterms{{2, eta}};
            for (int i = 0; i < l; ++i)
                zterms.push_back({*nrp + i - 1, zeta[i]});
            if (!combine(gw, gxsol, lda, n, 2, zterms) ||
                !combine(gw, gxsol, lda, n, -1, {{2, CUFFT_REAL(CU_REAL(1))}}, true)) {
                fail(16);
                return true;
            }
            std::vector<Term> rterms{{1, neg(eta)}};
            for (int i = 0; i < l; ++i)
                rterms.push_back({*nrp + i, neg(zeta[i])});
            if (!combine(gw, gxsol, lda, n, *nrp - 1, rterms, true)) {
                fail(16);
                return true;
            }
            std::vector<Term> pterms{{3, neg(eta)}};
            for (int i = 0; i < l; ++i)
                pterms.push_back({*npp + i, neg(zeta[i])});
            if (!combine(gw, gxsol, lda, n, *npp - 1, pterms, true)) {
                fail(16);
                return true;
            }
        } else {
            if (!combine(gw, gxsol, lda, n, 2, {{2, eta}})) {
                fail(16);
                return true;
            }
            for (int i = 0; i < l; ++i)
                if (!combine(gw, gxsol, lda, n, 2, {{*nrp + i - 1, zeta[i]}}, true)) {
                    fail(16);
                    return true;
                }
            if (!combine(gw, gxsol, lda, n, -1, {{2, CUFFT_REAL(CU_REAL(1))}}, true) ||
                !combine(gw, gxsol, lda, n, *nrp - 1,
                         {{1, neg(eta)}, {*nrp - 1, CUFFT_REAL(CU_REAL(1))}}) ||
                !combine(gw, gxsol, lda, n, *npp - 1,
                         {{3, neg(eta)}, {*npp - 1, CUFFT_REAL(CU_REAL(1))}})) {
                fail(16);
                return true;
            }
            for (int i = 0; i < l; ++i)
                if (!combine(gw, gxsol, lda, n, *nrp - 1,
                             {{*nrp + i, neg(zeta[i])}, {*nrp - 1, CUFFT_REAL(CU_REAL(1))}}) ||
                    !combine(gw, gxsol, lda, n, *npp - 1,
                             {{*npp + i, neg(zeta[i])}, {*npp - 1, CUFFT_REAL(CU_REAL(1))}})) {
                    fail(16);
                    return true;
                }
        }
        CUCOMPLEX rr{};
        if (!dc(*nrp - 1, *nrp - 1, &rr)) {
            fail(16);
            return true;
        }
        const CU_REAL resid = std::sqrt(std::hypot(rr.x, rr.y)) / (*norm);
        std::printf("RESIDUE %.16g iteration %d tolerance %.8g\n", double(resid), *itno,
                    double(*tol));
        std::fflush(stdout);
        if (resid <= *tol) {
            *nstat = 1;
            *nou = 6;
            if (cuda_data::copy_device_to_host(xsol_host, gxsol, count * sizeof(CUCOMPLEX),
                                               cuda_utils::TransferCategory::Other) !=
                    cudaSuccess ||
                !finish_xi()) {
                fail(15);
                return true;
            }
            return true;
        }
        if (*itno > *maxit) {
            fail(7);
            return true;
        }
        ++(*itno);
        // Begin next stabilized loop: y=r'-r0, u=p'-p0, rho=r0tilde^H r0.
        if (!combine(gw, gxsol, lda, n, 1,
                     {{4, CUFFT_REAL(CU_REAL(1))}, {*nrp - 1, CUFFT_REAL(CU_REAL(-1))}}) ||
            !combine(gw, gxsol, lda, n, 3,
                     {{5, CUFFT_REAL(CU_REAL(1))}, {*npp - 1, CUFFT_REAL(CU_REAL(-1))}}) ||
            !dc(0, *nrp - 1, rho)) {
            fail(16);
            return true;
        }
        *jp = 1;
        if (!publish_xi(*npp - 1)) {
            fail(16);
            return true;
        }
        *nou = 4;
        if (!finish_xi()) {
            fail(15);
        }
        return true;
    }
    fail(18);
    return true;
}
} // namespace

//-----------------------------------------------------------
extern "C" void gpbicgstabl_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, COMPLEX *xsol,
                             CU_REAL *norm, COMPLEX *work, CUCOMPLEX *alpha, CUCOMPLEX *beta,
                             CUCOMPLEX *sigma, CUCOMPLEX *rho, const int *l, const int *lda,
                             const int *ndim, const int *nlar, int *nou, int *nr, int *np, int *ns,
                             int *nq, int *j, int *nstat, int *steperr, int *itno, const int *maxit,
                             CU_REAL *tole, const CU_REAL *tol) {
    if (!run_gpbicgstabl(xi, xr, b, xsol, norm, work, alpha, beta, sigma, rho, l, lda, ndim, nlar,
                         nou, nr, np, ns, nq, j, nstat, steperr, itno, maxit, tole, tol)) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 19;
    }
}
