#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cmath>
#include <cstdio>

namespace {
using cuda_utils::check_cuda;
using cuda_utils::device_value;
using cuda_utils::dotc;
using cuda_utils::get_cublas_handle;
using cuda_utils::host_value;
using cuda_utils::HostComplex;
using cuda_utils::norm2;

//-----------------------------------------------------------
bool publish_xi(COMPLEX *cpu_xi, const CUCOMPLEX *gpu_xi, size_t count) {
    return copy_gpbicg_results_to_cpu(cpu_xi, nullptr, gpu_xi, nullptr, count, 0, true, false);
}

//-----------------------------------------------------------
bool combine(CUCOMPLEX *work, CUCOMPLEX *xi, int lda, int n, int destination, const int *columns,
             const CUCOMPLEX *coefficients, int terms) {
    return check_cuda(
        gpbicgstabl_combine_wrapper(work, xi, lda, n, destination, columns, coefficients, terms, 0),
        "SSBICGSAFE2 CUDA", "combine vectors");
}

//-----------------------------------------------------------
void ssbicgsafe2_impl(COMPLEX *xi, COMPLEX *xr, const COMPLEX *b, const int *lda,
                      const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                      const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                      CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *ffa, int *nstat,
                      int *steperr) {
    if (!xi || !xr || !b || !lda || !ndim || !nlar || !nou || !work || !itno || !maxit || !tol ||
        !norm || !alpha || !beta || !eta || !dzeta || !ffa || !nstat || !steperr || *ndim <= 0 ||
        *lda < *ndim || *nlar < 12 || *maxit < 0) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 5;
        return;
    }

    const int n = *ndim;
    const size_t count = static_cast<size_t>(n);
    size_t work_count = 0;
    if (!cuda_utils::checked_multiply(static_cast<size_t>(*lda), static_cast<size_t>(*nlar),
                                      &work_count)) {
        *nstat = -1;
        *steperr = 5;
        return;
    }
    CUCOMPLEX *gxi = nullptr, *gxr = nullptr, *gb = nullptr, *gwork = nullptr;
    const bool begin_solve = (*nou == 0);
    if (!get_gpbicg_device_arrays_resident(xi, xr, b, work, count, work_count, begin_solve, &gxi,
                                           &gxr, &gb, &gwork)) {
        *nstat = -1;
        *steperr = 5;
        return;
    }
    cublasHandle_t handle = get_cublas_handle();
    if (!handle) {
        *nstat = -1;
        *steperr = 5;
        return;
    }

    const auto launch = [&](cudaError_t status, const char *operation) {
        return check_cuda(status, "SSBICGSAFE2 CUDA", operation);
    };
    const auto fail_cuda = [&]() {
        *nstat = -1;
        *steperr = 5;
        (void)publish_xi(xi, gxi, count);
    };
    const auto dot = [&](const CUCOMPLEX *x, const CUCOMPLEX *y, CUCOMPLEX *out) {
        return dotc(handle, n, x, y, out, "SSBICGSAFE2 CUDA");
    };
    const auto publish_column = [&](int column) {
        return launch(idrs_copy_work_to_xi_wrapper(gxi, gwork, column, *lda, n),
                      "publish reverse-communication vector");
    };
    const auto set_failure = [&](int code) {
        *nstat = -1;
        *steperr = code;
    };

    *nstat = 0;
    *steperr = -1;
    if (*nou == 0) {
        if (!norm2(handle, n, gb, norm, "SSBICGSAFE2 CUDA") ||
            !launch(gpbicg_save_initial_wrapper(gxi, gwork, *lda, n), "save initial solution")) {
            fail_cuda();
            return;
        }
        *nou = 1;
        (void)publish_xi(xi, gxi, count);
        return;
    }

    if (*nou == 1) {
        if (!launch(ssbicgsafe2_initialize_wrapper(gxr, gb, gwork, *lda, n),
                    "initialize residual and work vectors")) {
            fail_cuda();
            return;
        }
        CU_REAL initial_residual = 0;
        if (!norm2(handle, n, gwork, &initial_residual, "SSBICGSAFE2 CUDA")) {
            fail_cuda();
            return;
        }
        const CU_REAL initial =
            (*norm > CU_REAL(0)) ? initial_residual / (*norm) : initial_residual;
        std::printf("residu initial %.16g\n", static_cast<double>(initial));
        *itno = 0;
        if (!publish_column(2)) {
            fail_cuda();
            return;
        }
        *nou = 2;
        (void)publish_xi(xi, gxi, count);
        return;
    }

    if (*nou == 2) {
        if (!launch(idrs_copy_xr_wrapper(gwork, gxr, 5, *lda, n), "save Ar")) {
            fail_cuda();
            return;
        }
        CUCOMPLEX ff = CUFFT_ZERO, dd = CUFFT_ZERO, gg = CUFFT_ZERO, aa = CUFFT_ZERO;
        CUCOMPLEX bb = CUFFT_ZERO, cc = CUFFT_ZERO, ee = CUFFT_ZERO, hh = CUFFT_ZERO;
        if (!dot(gwork, gwork + size_t(2) * (*lda), &ff) ||
            !dot(gwork + size_t(5) * (*lda), gwork + size_t(2) * (*lda), &dd) ||
            !dot(gwork, gwork + size_t(5) * (*lda), &gg) ||
            !dot(gwork + size_t(5) * (*lda), gwork + size_t(5) * (*lda), &aa)) {
            fail_cuda();
            return;
        }

        if (*itno == 0) {
            if (host_value(gg) == HostComplex(0, 0) || host_value(aa) == HostComplex(0, 0)) {
                set_failure(1);
                (void)publish_xi(xi, gxi, count);
                return;
            }
            *alpha = device_value(host_value(ff) / host_value(gg));
            *dzeta = device_value(host_value(dd) / host_value(aa));
            *beta = CUFFT_ZERO;
            *eta = CUFFT_ZERO;
            *ffa = ff;
        } else {
            if (!dot(gwork + size_t(3) * (*lda), gwork + size_t(3) * (*lda), &bb) ||
                !dot(gwork + size_t(5) * (*lda), gwork + size_t(3) * (*lda), &cc) ||
                !dot(gwork + size_t(3) * (*lda), gwork + size_t(2) * (*lda), &ee) ||
                !dot(gwork, gwork + size_t(4) * (*lda), &hh)) {
                fail_cuda();
                return;
            }
            HostComplex denominator = host_value(*dzeta) * host_value(*ffa);
            if (denominator == HostComplex(0, 0)) {
                set_failure(2);
                (void)publish_xi(xi, gxi, count);
                return;
            }
            *beta = device_value(host_value(*alpha) * host_value(ff) / denominator);
            denominator = host_value(gg) + host_value(*beta) * host_value(hh);
            if (denominator == HostComplex(0, 0)) {
                set_failure(3);
                (void)publish_xi(xi, gxi, count);
                return;
            }
            *alpha = device_value(host_value(ff) / denominator);
            denominator =
                host_value(aa) * host_value(bb) - std::conj(host_value(cc)) * host_value(cc);
            if (denominator == HostComplex(0, 0)) {
                set_failure(4);
                (void)publish_xi(xi, gxi, count);
                return;
            }
            *dzeta = device_value(
                (host_value(bb) * host_value(dd) - host_value(cc) * host_value(ee)) / denominator);
            *eta = device_value(
                (host_value(aa) * host_value(ee) - std::conj(host_value(cc)) * host_value(dd)) /
                denominator);
            *ffa = ff;
        }

        const int pcols[3] = {2, 1, 7};
        const CUCOMPLEX pcoef[3] = {device_value(HostComplex(1, 0)), *beta,
                                    device_value(-host_value(*beta))};
        const int ocols[2] = {5, 4};
        const CUCOMPLEX ocoef[2] = {device_value(HostComplex(1, 0)), *beta};
        const int ucols[3] = {6, 3, 7};
        const CUCOMPLEX ucoef[3] = {*dzeta, *eta,
                                    device_value(host_value(*eta) * host_value(*beta))};
        if (!combine(gwork, gxi, *lda, n, 1, pcols, pcoef, 3) ||
            !combine(gwork, gxi, *lda, n, 6, ocols, ocoef, 2) ||
            !combine(gwork, gxi, *lda, n, 7, ucols, ucoef, 3) || !publish_column(7)) {
            fail_cuda();
            return;
        }
        *nou = 3;
        (void)publish_xi(xi, gxi, count);
        return;
    }

    if (*nou == 3) {
        if (!launch(idrs_copy_xr_wrapper(gwork, gxr, 8, *lda, n), "save Ap")) {
            fail_cuda();
            return;
        }
        const int tcols[2] = {6, 8};
        const CUCOMPLEX tcoef[2] = {device_value(HostComplex(1, 0)),
                                    device_value(HostComplex(-1, 0))};
        const int zcols[3] = {2, 9, 7};
        const CUCOMPLEX zcoef[3] = {*dzeta, *eta, device_value(-host_value(*alpha))};
        const int ycols[3] = {5, 3, 8};
        const CUCOMPLEX ycoef[3] = {*dzeta, *eta, device_value(-host_value(*alpha))};
        const int xcols[3] = {10, 1, 9};
        const CUCOMPLEX xcoef[3] = {device_value(HostComplex(1, 0)), *alpha,
                                    device_value(HostComplex(1, 0))};
        const int rcols[3] = {2, 6, 3};
        const CUCOMPLEX rcoef[3] = {device_value(HostComplex(1, 0)),
                                    device_value(-host_value(*alpha)),
                                    device_value(HostComplex(-1, 0))};
        if (!combine(gwork, gxi, *lda, n, 4, tcols, tcoef, 2) ||
            !combine(gwork, gxi, *lda, n, 9, zcols, zcoef, 3) ||
            !combine(gwork, gxi, *lda, n, 3, ycols, ycoef, 3) ||
            !combine(gwork, gxi, *lda, n, 10, xcols, xcoef, 3) ||
            !combine(gwork, gxi, *lda, n, 2, rcols, rcoef, 3)) {
            fail_cuda();
            return;
        }
        CU_REAL residual = 0;
        if (!norm2(handle, n, gwork + size_t(2) * (*lda), &residual, "SSBICGSAFE2 CUDA")) {
            fail_cuda();
            return;
        }
        residual = (*norm > CU_REAL(0)) ? residual / (*norm) : residual;
        std::printf("RESIDU %.16g iteration %d\n", static_cast<double>(residual), *itno);
        if (residual <= *tol) {
            *nstat = 1;
            *nou = 4;
            if (!publish_column(10)) {
                fail_cuda();
                return;
            }
            (void)publish_xi(xi, gxi, count);
            return;
        }
        if (*itno <= *maxit) {
            ++(*itno);
            if (!publish_column(2)) {
                fail_cuda();
                return;
            }
            *nou = 2;
        } else {
            *nstat = 1;
            if (!publish_column(10)) {
                fail_cuda();
                return;
            }
        }
        (void)publish_xi(xi, gxi, count);
        return;
    }

    if (*nou == 4) {
        if (*itno <= *maxit) {
            ++(*itno);
            if (!publish_column(2)) {
                fail_cuda();
                return;
            }
            *nou = 2;
        } else {
            *nstat = 1;
            if (!publish_column(10)) {
                fail_cuda();
                return;
            }
        }
        (void)publish_xi(xi, gxi, count);
        return;
    }

    *nstat = -1;
    *steperr = 5;
}
} // namespace

//-----------------------------------------------------------
extern "C" void ssbicgsafe2_(COMPLEX *xi, COMPLEX *xr, const COMPLEX *b, const int *lda,
                             const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                             const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                             CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *ffa,
                             int *nstat, int *steperr) {
    ssbicgsafe2_impl(xi, xr, b, lda, ndim, nlar, nou, work, itno, maxit, tol, norm, alpha, beta,
                     eta, dzeta, ffa, nstat, steperr);
}
