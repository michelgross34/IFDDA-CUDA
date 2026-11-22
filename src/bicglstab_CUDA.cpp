#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using cuda_utils::check_cuda;
using cuda_utils::dotc;
using cuda_utils::dotu;
using cuda_utils::get_cublas_handle;
using cuda_utils::norm2;

struct Term {
    int column;
    CUCOMPLEX coefficient;
};

//-----------------------------------------------------------
CUCOMPLEX add(CUCOMPLEX a, CUCOMPLEX b) {
    return CUFFT_ADD(a, b);
}

//-----------------------------------------------------------
CUCOMPLEX sub(CUCOMPLEX a, CUCOMPLEX b) {
    return CUFFT_SUB(a, b);
}

//-----------------------------------------------------------
CUCOMPLEX mul(CUCOMPLEX a, CUCOMPLEX b) {
    return CUFFT_MUL(a, b);
}

//-----------------------------------------------------------
CUCOMPLEX divv(CUCOMPLEX a, CUCOMPLEX b) {
    return CUFFT_DIV(a, b);
}

//-----------------------------------------------------------
CUCOMPLEX neg(CUCOMPLEX a) {
    return CUFFT_SUB(CUFFT_ZERO, a);
}

//-----------------------------------------------------------
double magnitude(CUCOMPLEX a) {
    return std::hypot(double(a.x), double(a.y));
}

//-----------------------------------------------------------
bool combine(CUCOMPLEX *work, CUCOMPLEX *xsol, int lda, int n, int dst,
             const std::vector<Term> &terms, bool accumulate = false) {
    if (terms.empty() || terms.size() > 10)
        return false;
    int columns[10];
    CUCOMPLEX coefficients[10];
    for (size_t i = 0; i < terms.size(); ++i) {
        columns[i] = terms[i].column;
        coefficients[i] = terms[i].coefficient;
    }
    return check_cuda(gpbicgstabl_combine_wrapper(work, xsol, lda, n, dst, columns, coefficients,
                                                  int(terms.size()), accumulate ? 1 : 0),
                      "BICGLSTAB CUDA", "vector update");
}

//-----------------------------------------------------------
bool publish(COMPLEX *host_xi, CUCOMPLEX *device_xi, CUCOMPLEX *device_work, int column, int lda,
             int n) {
    if (!check_cuda(idrs_copy_work_to_xi_wrapper(device_xi, device_work, column, lda, n),
                    "BICGLSTAB CUDA", "publish XI"))
        return false;
    return cuda_data::copy_device_to_host(host_xi, device_xi, size_t(n) * sizeof(CUCOMPLEX),
                                          cuda_utils::TransferCategory::Other) == cudaSuccess;
}

//-----------------------------------------------------------
bool inner(cublasHandle_t handle, int n, CUCOMPLEX *work, int lda, int c1, int c2,
           CUCOMPLEX *result, bool conjugate_first = true) {
    const CUCOMPLEX *a = work + size_t(c1) * lda;
    const CUCOMPLEX *b = work + size_t(c2) * lda;
    return conjugate_first ? dotc(handle, n, a, b, result, "BICGLSTAB CUDA")
                           : dotu(handle, n, a, b, result, "BICGLSTAB CUDA");
}

//-----------------------------------------------------------
void trace_xsol(const char *stage, const CUCOMPLEX *device_xsol, int n) {
    if (!std::getenv("IFDDA_TRACE_CHAN_PRODUCT"))
        return;
    std::vector<CUCOMPLEX> host_xsol(static_cast<size_t>(n));
    if (cuda_data::copy_device_to_host(host_xsol.data(), device_xsol, size_t(n) * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Diagnostic) != cudaSuccess)
        return;
    double norm_squared = 0.0;
    for (const CUCOMPLEX value : host_xsol)
        norm_squared += double(value.x) * double(value.x) + double(value.y) * double(value.y);
    std::fprintf(stderr, "[BICGLSTAB CUDA Xsol %s] (%.17g,%.17g) (%.17g,%.17g) %.17g\n", stage,
                 double(host_xsol[0].x), double(host_xsol[0].y), double(host_xsol[1].x),
                 double(host_xsol[1].y), std::sqrt(norm_squared));
}

//-----------------------------------------------------------
bool run_bicglstab(COMPLEX *xi, COMPLEX *xr, const COMPLEX *b, COMPLEX *xsol_host,
                   CU_REAL *norm, COMPLEX *work_host, CUCOMPLEX *alpha, CUCOMPLEX *rho,
                   CUCOMPLEX *omega, const int *lp, int *nrp, int *nup, int *jp, const int *ldap,
                   const int *ndimp, const int *nlarp, int *nou, int *nstat, int *steperr,
                   int *itno, const int *maxit, const CU_REAL *tole, const CU_REAL *tol) {
    if (!xi || !xr || !b || !xsol_host || !norm || !work_host || !alpha || !rho || !omega || !lp ||
        !nrp || !nup || !jp || !ldap || !ndimp || !nlarp || !nou || !nstat || !steperr || !itno ||
        !maxit || !tole || !tol) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 12;
        return false;
    }

    const int l = *lp, lda = *ldap, n = *ndimp, nlar = *nlarp;
    if ((l != 2 && l != 4 && l != 8) || n <= 0 || lda < n || nlar < 2 * l + 3) {
        *nstat = -1;
        *steperr = 10;
        return true;
    }
    const size_t count = size_t(n);
    const size_t work_count = size_t(lda) * size_t(nlar);
    const bool initial = (*nou != 1 && *nou != 2 && *nou != 3);
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
    auto fail = [&](int code) {
        *nstat = -1;
        *steperr = code;
    };
    auto finish_xsol = [&]() { return copy_tfqmr_x_to_cpu(xsol_host, gxsol, count); };

    *nstat = 0;
    *steperr = -1;
    if (initial) {
        if (!norm2(handle, n, gb, norm, "BICGLSTAB CUDA") || !std::isfinite(double(*norm)) ||
            *norm <= CU_REAL(0)) {
            fail(11);
            return true;
        }
        if (!check_cuda(cudaMemset(gw, 0, work_count * sizeof(CUCOMPLEX)), "BICGLSTAB CUDA",
                        "initialize work buffer")) {
            fail(15);
            return true;
        }
        *nup = 2;
        *nrp = l + 3;
        *jp = 0;
        *nou = 1;
        return true;
    }

    const int nr = *nrp;
    const int nu = *nup;
    const int j = *jp;
    if (*nou == 1) {
        // Fortran columns are one-based: r0=b-Ax is column 1, r is column nr.
        // Save XR in column 2, then form the initial residual b-XR in column 1.
        if (!check_cuda(idrs_copy_xr_wrapper(gw, gxr, 1, lda, n), "BICGLSTAB CUDA", "save XR") ||
            cuda_utils::tracked_memcpy(gw, gb, count * sizeof(CUCOMPLEX),
                                       cudaMemcpyDeviceToDevice) != cudaSuccess ||
            !combine(gw, gxsol, lda, n, 0,
                     {{1, CUFFT_REAL(CU_REAL(-1))}, {0, CUFFT_REAL(CU_REAL(1))}})) {
            fail(16);
            return true;
        }
        if (!combine(gw, gxsol, lda, n, nr - 1, {{0, CUFFT_REAL(CU_REAL(1))}})) {
            fail(16);
            return true;
        }
        *rho = CUFFT_REAL(CU_REAL(1));
        *alpha = CUFFT_ZERO;
        *omega = CUFFT_REAL(CU_REAL(1));
        *itno = -1;
        ++(*itno);
        *jp = 0;
        *rho = neg(mul(*omega, *rho));
        CUCOMPLEX rho1{};
        if (!inner(handle, n, gw, lda, 0, nr - 1, &rho1) || magnitude(*rho) < 1.0e-30) {
            fail(1);
            return true;
        }
        const CUCOMPLEX beta = divv(mul(*alpha, rho1), *rho);
        *rho = rho1;
        if (magnitude(*rho) < 1.0e-30 ||
            !combine(gw, gxsol, lda, n, nu - 1,
                     {{nr - 1, CUFFT_REAL(CU_REAL(1))}, {nu - 1, neg(beta)}}) ||
            !publish(xi, gxi, gw, nu - 1, lda, n)) {
            fail(16);
            return true;
        }
        *nou = 2;
        return true;
    }

    if (*nou == 2) {
        if (j < 0 || j >= l) {
            fail(10);
            return true;
        }
        const int ap_col = nu + j; // Fortran nu+j+1, converted to zero-based.
        if (cuda_utils::tracked_memcpy(gw + size_t(ap_col) * lda, gxr, count * sizeof(CUCOMPLEX),
                                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            fail(16);
            return true;
        }
        CUCOMPLEX gamma0{};
        if (!dotc(handle, n, gw, gxr, &gamma0, "BICGLSTAB CUDA") || magnitude(gamma0) < 1.0e-30) {
            fail(2);
            return true;
        }
        *alpha = divv(*rho, gamma0);
        for (int i = 0; i <= j; ++i) {
            if (!combine(gw, gxsol, lda, n, nr + i - 1,
                         {{nr + i - 1, CUFFT_REAL(CU_REAL(1))}, {nu + i, neg(*alpha)}})) {
                fail(16);
                return true;
            }
        }
        if (!publish(xi, gxi, gw, nr + j - 1, lda, n)) {
            fail(15);
            return true;
        }
        *nou = 3;
        return true;
    }

    if (*nou != 3) {
        fail(10);
        return true;
    }
    if (j < 0 || j >= l) {
        fail(10);
        return true;
    }
    const int next_r_col = nr + j; // Fortran nr+j+1, zero-based.
    if (std::getenv("IFDDA_TRACE_CHAN_PRODUCT")) {
        CUCOMPLEX sample[4]{};
        if (cuda_data::copy_device_to_host(sample, gxsol, 2 * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Diagnostic) ==
                cudaSuccess &&
            cuda_data::copy_device_to_host(sample + 2, gw + size_t(nu - 1) * lda,
                                           2 * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Diagnostic) == cudaSuccess)
            std::fprintf(stderr,
                         "[BICGLSTAB CUDA Xsol input] (%.17g,%.17g)(%.17g,%.17g) "
                         "alpha=(%.17g,%.17g) u=(%.17g,%.17g)(%.17g,%.17g)\n",
                         double(sample[0].x), double(sample[0].y), double(sample[1].x),
                         double(sample[1].y), double(alpha->x), double(alpha->y),
                         double(sample[2].x), double(sample[2].y), double(sample[3].x),
                         double(sample[3].y));
    }
    if (cuda_utils::tracked_memcpy(gw + size_t(next_r_col) * lda, gxr, count * sizeof(CUCOMPLEX),
                                   cudaMemcpyDeviceToDevice) != cudaSuccess ||
        !combine(gw, gxsol, lda, n, -1, {{nu - 1, *alpha}}, true)) {
        fail(16);
        return true;
    }
    trace_xsol("BiCG", gxsol, n);

    if (j < l - 1) {
        ++(*jp);
        const int new_j = *jp;
        CUCOMPLEX rho1{};
        if (!inner(handle, n, gw, lda, 0, nr + new_j - 1, &rho1) || magnitude(*rho) < 1.0e-30) {
            fail(1);
            return true;
        }
        const CUCOMPLEX beta = divv(mul(*alpha, rho1), *rho);
        *rho = rho1;
        if (magnitude(*rho) < 1.0e-30) {
            fail(1);
            return true;
        }
        for (int i = 0; i <= new_j; ++i) {
            if (!combine(gw, gxsol, lda, n, nu + i - 1,
                         {{nr + i - 1, CUFFT_REAL(CU_REAL(1))}, {nu + i - 1, neg(beta)}})) {
                fail(16);
                return true;
            }
        }
        if (!publish(xi, gxi, gw, nu + new_j - 1, lda, n)) {
            fail(16);
            return true;
        }
        *nou = 2;
        return true;
    }

    // Minimum-residual projection, matching zbicgstabl.f90's default
    // Hermitian orientation (IFDDA_BICG_LEGACY_MR=1 selects the old one).
    const char *legacy_env = std::getenv("IFDDA_BICG_LEGACY_MR");
    const bool mr_reference = !(legacy_env && legacy_env[0] == '1' && legacy_env[1] == '\0');
    CUCOMPLEX gamma[8]{}, gammap[8]{}, gammapp[8]{}, sigma[8]{};
    CUCOMPLEX tau[8][8]{};
    for (int mj = 0; mj < l; ++mj) {
        for (int mi = 0; mi < mj; ++mi) {
            CUCOMPLEX value{};
            if (!inner(handle, n, gw, lda, nr + mi, nr + mj, &value, mr_reference) ||
                magnitude(sigma[mi]) < 1.0e-30) {
                fail(3);
                return true;
            }
            tau[mi][mj] = divv(value, sigma[mi]);
            if (!combine(gw, gxsol, lda, n, nr + mj,
                         {{nr + mj, CUFFT_REAL(CU_REAL(1))}, {nr + mi, neg(tau[mi][mj])}})) {
                fail(16);
                return true;
            }
        }
        if (!inner(handle, n, gw, lda, nr + mj, nr + mj, &sigma[mj])) {
            fail(16);
            return true;
        }
        CUCOMPLEX gp{};
        if (!inner(handle, n, gw, lda, mr_reference ? nr + mj : nr - 1,
                   mr_reference ? nr - 1 : nr + mj, &gp) ||
            magnitude(sigma[mj]) < 1.0e-30) {
            fail(3);
            return true;
        }
        gammap[mj] = divv(gp, sigma[mj]);
    }
    gamma[l - 1] = gammap[l - 1];
    *omega = gamma[l - 1];
    for (int mj = l - 2; mj >= 0; --mj) {
        gamma[mj] = gammap[mj];
        for (int mi = mj + 1; mi < l; ++mi)
            gamma[mj] = sub(gamma[mj], mul(tau[mj][mi], gamma[mi]));
    }
    for (int mj = 0; mj < l - 1; ++mj) {
        gammapp[mj] = gamma[mj + 1];
        for (int mi = mj + 1; mi < l - 1; ++mi)
            gammapp[mj] = add(gammapp[mj], mul(tau[mj][mi], gamma[mi + 1]));
    }
    if (std::getenv("IFDDA_TRACE_CHAN_PRODUCT")) {
        std::fprintf(stderr, "[BICGLSTAB CUDA MR]");
        for (int i = 0; i < l; ++i)
            std::fprintf(stderr, " G%d=(%.17g,%.17g) GP%d=(%.17g,%.17g)", i + 1, double(gamma[i].x),
                         double(gamma[i].y), i + 1, double(gammap[i].x), double(gammap[i].y));
        if (l > 1)
            std::fprintf(stderr, " TAU12=(%.17g,%.17g)", double(tau[0][1].x), double(tau[0][1].y));
        std::fprintf(stderr, "\n");
    }
    if (!combine(gw, gxsol, lda, n, -1, {{nr - 1, gamma[0]}}, true) ||
        !combine(gw, gxsol, lda, n, nr - 1,
                 {{nr - 1, CUFFT_REAL(CU_REAL(1))}, {nr + l - 1, neg(gammap[l - 1])}}) ||
        !combine(gw, gxsol, lda, n, nu - 1,
                 {{nu - 1, CUFFT_REAL(CU_REAL(1))}, {nu + l - 1, neg(gamma[l - 1])}})) {
        fail(16);
        return true;
    }
    for (int mj = 0; mj < l - 1; ++mj) {
        if (!combine(gw, gxsol, lda, n, nu - 1,
                     {{nu - 1, CUFFT_REAL(CU_REAL(1))}, {nu + mj, neg(gamma[mj])}}) ||
            !combine(gw, gxsol, lda, n, -1, {{nr + mj, gammapp[mj]}}, true) ||
            !combine(gw, gxsol, lda, n, nr - 1,
                     {{nr - 1, CUFFT_REAL(CU_REAL(1))}, {nr + mj, neg(gammap[mj])}})) {
            fail(16);
            return true;
        }
    }
    trace_xsol("MR", gxsol, n);

    CUCOMPLEX residual2{};
    if (!inner(handle, n, gw, lda, nr - 1, nr - 1, &residual2)) {
        fail(16);
        return true;
    }
    if (!std::isfinite(double(*norm)) || *norm < 1.0e-30) {
        fail(4);
        return true;
    }
    const double residual = std::sqrt(magnitude(residual2)) / double(*norm);
    std::printf("BICGLSTAB CUDA residue %.16g iteration %d\n", residual, *itno);
    if (!std::isfinite(residual)) {
        fail(5);
        return true;
    }
    if (residual <= double(*tol)) {
        if (cuda_utils::tracked_memcpy(gxi, gxsol, count * sizeof(CUCOMPLEX),
                                       cudaMemcpyDeviceToDevice) != cudaSuccess ||
            cuda_data::copy_device_to_host(xi, gxi, count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Other) != cudaSuccess ||
            !finish_xsol()) {
            fail(15);
            return true;
        }
        *nstat = 1;
        *nou = 4;
        return true;
    }

    ++(*itno);
    *rho = neg(mul(*omega, *rho));
    if (magnitude(*rho) < 1.0e-30) {
        fail(1);
        return true;
    }
    CUCOMPLEX rho1{};
    if (!inner(handle, n, gw, lda, 0, nr - 1, &rho1) || magnitude(*rho) < 1.0e-30) {
        fail(1);
        return true;
    }
    const CUCOMPLEX beta = divv(mul(*alpha, rho1), *rho);
    *rho = rho1;
    if (magnitude(*rho) < 1.0e-30 ||
        !combine(gw, gxsol, lda, n, nu - 1,
                 {{nr - 1, CUFFT_REAL(CU_REAL(1))}, {nu - 1, neg(beta)}}) ||
        !publish(xi, gxi, gw, nu - 1, lda, n)) {
        fail(16);
        return true;
    }
    *jp = 0;
    *nou = 2;
    return true;
}
} // namespace

//-----------------------------------------------------------
extern "C" void zbicgstabl_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, COMPLEX *xsol,
                            CU_REAL *norm, COMPLEX *work, CUCOMPLEX *alpha, CUCOMPLEX *rho,
                            CUCOMPLEX *omega, const int *l, int *nr, int *nu, int *j,
                            const int *lda, const int *ndim, const int *nlar, int *nou, int *nstat,
                            int *steperr, int *itno, const int *maxit, const CU_REAL *tole,
                            const CU_REAL *tol) {
    (void)maxit;
    (void)tole;
    run_bicglstab(xi, xr, b, xsol, norm, work, alpha, rho, omega, l, nr, nu, j, lda, ndim, nlar,
                  nou, nstat, steperr, itno, maxit, tole, tol);
}
