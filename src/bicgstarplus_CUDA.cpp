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
using cuda_utils::dotu;
using cuda_utils::get_cublas_handle;
using cuda_utils::host_value;
using cuda_utils::HostComplex;
using cuda_utils::norm2;
using cuda_utils::scalar_is_zero;
constexpr const char *solver_name = "BICGSTARPLUS CUDA";

//-----------------------------------------------------------
bool copy_xi(COMPLEX *cpu_xi, CUCOMPLEX *gpu_xi, size_t n, int *nstat, int *steperr) {
    if (copy_gpbicg_results_to_cpu(cpu_xi, nullptr, gpu_xi, nullptr, n, 0, true, false))
        return true;
    *nstat = -1;
    *steperr = 15;
    return false;
}

//-----------------------------------------------------------
bool launch(cudaError_t status, const char *operation, int *nstat, int *steperr) {
    if (check_cuda(status, solver_name, operation))
        return true;
    *nstat = -1;
    *steperr = 16;
    return false;
}

//-----------------------------------------------------------
void run_bicgstarplus(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda, const int *ndim,
                      const int *nlar, int *nou, COMPLEX *work, int *itno, const int *maxit,
                      const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *beta,
                      CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn, int *nstat, int *steperr) {
    if (!xi || !xr || !b || !lda || !ndim || !nlar || !nou || !work || !itno || !maxit || !tol ||
        !norm || !alpha || !beta || !eta || !dzeta || !r0rn || !nstat || !steperr || *ndim <= 0 ||
        *lda < *ndim || *nlar < 12 || *maxit < 0) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 12;
        return;
    }

    const int n = *ndim;
    const size_t count = static_cast<size_t>(n);
    const size_t work_count = static_cast<size_t>(*lda) * static_cast<size_t>(*nlar);
    const bool first = (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4);
    CUCOMPLEX *gxi = nullptr, *gxr = nullptr, *gb = nullptr, *gw = nullptr;
    if (!get_gpbicg_device_arrays_resident(xi, xr, b, work, count, work_count, first, &gxi, &gxr,
                                           &gb, &gw)) {
        *nstat = -1;
        *steperr = 13;
        return;
    }
    cublasHandle_t handle = get_cublas_handle();
    if (!handle) {
        *nstat = -1;
        *steperr = 14;
        return;
    }
    const auto fail = [&](int code) {
        *nstat = -1;
        *steperr = code;
        (void)copy_xi(xi, gxi, count, nstat, steperr);
    };

    if (first) {
        *nstat = 0;
        *steperr = -1;
        if (!norm2(handle, n, gb, norm, solver_name) || !std::isfinite(*norm) ||
            *norm <= CU_REAL(0)) {
            fail(1);
            return;
        }
        if (!launch(bicgstarplus_save_solution_wrapper(gxi, gw, *lda, n), "save initial solution",
                    nstat, steperr)) {
            fail(16);
            return;
        }
        *nou = 1;
        (void)copy_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    if (*nou == 1) {
        if (!launch(bicgstarplus_initialize_residual_wrapper(gxi, gxr, gb, gw, *lda, n),
                    "initialize residual", nstat, steperr)) {
            fail(16);
            return;
        }
        CUCOMPLEX q = CUFFT_ZERO;
        if (!dotu(handle, n, gw, gw + static_cast<size_t>(2) * *lda, &q, solver_name)) {
            fail(1);
            return;
        }
        *r0rn = q;
        const HostComplex rh = host_value(q);
        const HostComplex initial = std::sqrt(rh);
        std::printf("Initial Residue for iterative method %.16g Norm %.16g\n",
                    std::abs(initial) / *norm, double(*norm));
        *itno = 0; // Fortran sets -1 then enters label 100.
        *nou = 2;
        (void)copy_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    if (*nou == 2) {
        if (!launch(gpbicg_save_xr_column_wrapper(gxr, gw, *lda, n, 1), "save Ax column", nstat,
                    steperr)) {
            fail(16);
            return;
        }
        CUCOMPLEX q1 = CUFFT_ZERO, q2 = CUFFT_ZERO, q3 = CUFFT_ZERO, q4 = CUFFT_ZERO;
        CUCOMPLEX q5 = CUFFT_ZERO, q6 = CUFFT_ZERO, q7 = CUFFT_ZERO, q8 = CUFFT_ZERO;
        if (*itno == 0) {
            if (!dotu(handle, n, gw, gw + static_cast<size_t>(2) * *lda, &q1, solver_name) ||
                !dotc(handle, n, gw + static_cast<size_t>(1) * *lda,
                      gw + static_cast<size_t>(2) * *lda, &q2, solver_name) ||
                !dotc(handle, n, gw + static_cast<size_t>(1) * *lda,
                      gw + static_cast<size_t>(1) * *lda, &q3, solver_name) ||
                !dotu(handle, n, gw, gw + static_cast<size_t>(1) * *lda, &q4, solver_name)) {
                fail(1);
                return;
            }
            *r0rn = q1;
            if (scalar_is_zero(q3)) {
                *nstat = -1;
                *steperr = 2;
                return;
            }
            if (scalar_is_zero(q4)) {
                *nstat = -1;
                *steperr = 3;
                return;
            }
            *dzeta = device_value(host_value(q2) / host_value(q3));
            *alpha = device_value(host_value(q1) / host_value(q4));
            *beta = CUFFT_ZERO;
            *eta = CUFFT_ZERO;
        } else {
            if (!dotu(handle, n, gw, gw + static_cast<size_t>(2) * *lda, &q1, solver_name) ||
                !dotu(handle, n, gw, gw + static_cast<size_t>(1) * *lda, &q2, solver_name) ||
                !dotc(handle, n, gw + static_cast<size_t>(4) * *lda,
                      gw + static_cast<size_t>(4) * *lda, &q3, solver_name) ||
                !dotc(handle, n, gw + static_cast<size_t>(1) * *lda,
                      gw + static_cast<size_t>(2) * *lda, &q4, solver_name) ||
                !dotc(handle, n, gw + static_cast<size_t>(1) * *lda,
                      gw + static_cast<size_t>(4) * *lda, &q5, solver_name) ||
                !dotc(handle, n, gw + static_cast<size_t>(4) * *lda,
                      gw + static_cast<size_t>(2) * *lda, &q6, solver_name) ||
                !dotc(handle, n, gw + static_cast<size_t>(1) * *lda,
                      gw + static_cast<size_t>(1) * *lda, &q7, solver_name) ||
                !dotu(handle, n, gw, gw + static_cast<size_t>(11) * *lda, &q8, solver_name)) {
                fail(1);
                return;
            }
            if (scalar_is_zero(*dzeta)) {
                *nstat = -1;
                *steperr = 4;
                return;
            }
            if (scalar_is_zero(*r0rn)) {
                *nstat = -1;
                *steperr = 5;
                return;
            }
            const HostComplex betah =
                -host_value(*alpha) / host_value(*dzeta) * host_value(q1) / host_value(*r0rn);
            *beta = device_value(betah);
            const HostComplex c = host_value(q2) - betah * host_value(q8);
            if (c == HostComplex(0, 0)) {
                *nstat = -1;
                *steperr = 6;
                return;
            }
            *alpha = device_value(host_value(q1) / c);
            const HostComplex gram =
                host_value(q7) * host_value(q3) - host_value(q5) * std::conj(host_value(q5));
            if (gram == HostComplex(0, 0)) {
                *nstat = -1;
                *steperr = 7;
                return;
            }
            const HostComplex dz =
                (host_value(q3) * host_value(q4) - host_value(q5) * host_value(q6)) / gram;
            const HostComplex et =
                (host_value(q7) * host_value(q6) - std::conj(host_value(q5)) * host_value(q4)) /
                gram;
            *dzeta = device_value(dz);
            *eta = device_value(et);
            *r0rn = q1;
        }
        if (!launch(bicgstarplus_update_direction_wrapper(gxi, gw, *alpha, *beta, *eta, *dzeta,
                                                          *lda, n),
                    "update direction", nstat, steperr)) {
            fail(16);
            return;
        }
        *nou = 3;
        (void)copy_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    if (*nou == 3 || *nou == 4) {
        if (!launch(bicgstarplus_update_residual_wrapper(gxi, gxr, gw, *alpha, *beta, *eta, *dzeta,
                                                         *lda, n),
                    "update residual", nstat, steperr)) {
            fail(16);
            return;
        }
        CU_REAL rnorm = 0;
        if (!norm2(handle, n, gw + static_cast<size_t>(2) * *lda, &rnorm, solver_name)) {
            fail(1);
            return;
        }
        const CU_REAL residual = rnorm / *norm;
        if (residual <= *tol) {
            *nstat = 1;
            *nou = 4;
            if (!launch(gpbicg_publish_solution_wrapper(gxi, gw, *lda, n), "publish solution",
                        nstat, steperr)) {
                fail(16);
                return;
            }
            (void)copy_xi(xi, gxi, count, nstat, steperr);
            return;
        }
        if (*itno <= *maxit) {
            ++(*itno);
            if (!launch(bicgstarplus_publish_residual_wrapper(gxi, gw, *lda, n), "publish residual",
                        nstat, steperr)) {
                fail(16);
                return;
            }
            *nou = 2;
            *nstat = 0;
            *steperr = -1;
            (void)copy_xi(xi, gxi, count, nstat, steperr);
            return;
        }
        *nstat = 1;
        *steperr = 0;
        if (!launch(gpbicg_publish_solution_wrapper(gxi, gw, *lda, n), "publish final solution",
                    nstat, steperr)) {
            fail(16);
            return;
        }
        (void)copy_xi(xi, gxi, count, nstat, steperr);
        return;
    }
}
} // namespace

//-----------------------------------------------------------
extern "C" void gpbicgstarplus_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                                const int *ndim, const int *nlar, int *nou, COMPLEX *work,
                                int *itno, const int *maxit, const CU_REAL *tol, CU_REAL *norm,
                                CUCOMPLEX *alpha, CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta,
                                CUCOMPLEX *r0rn, int *nstat, int *steperr) {
    run_bicgstarplus(xi, xr, b, lda, ndim, nlar, nou, work, itno, maxit, tol, norm, alpha, beta,
                     eta, dzeta, r0rn, nstat, steperr);
}
