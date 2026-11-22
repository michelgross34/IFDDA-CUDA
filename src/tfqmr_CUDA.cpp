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
using cuda_utils::scalar_is_zero;

constexpr const char *solver_name = "TFQMR CUDA";

//-----------------------------------------------------------
bool publish_xi(COMPLEX *cpu_xi, CUCOMPLEX *gpu_xi, size_t count, int *nstat, int *steperr) {
    if (copy_gpbicg_results_to_cpu(cpu_xi, nullptr, gpu_xi, nullptr, count, 0, true, false))
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
void tfqmr_impl(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                CUCOMPLEX *beta, CUCOMPLEX *rho, CUCOMPLEX *tau, CUCOMPLEX *theta, CUCOMPLEX *eta,
                int *nstat, int *steperr) {
    if (!x || !xi || !xr || !b || !lda || !ndim || !nlar || !nou || !work || !itno || !maxit ||
        !tol || !norm || !alpha || !beta || !rho || !tau || !theta || !eta || !nstat || !steperr ||
        *ndim <= 0 || *lda < *ndim || *nlar < 9 || *maxit < 0) {
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
    CUCOMPLEX *gx = get_tfqmr_device_x_from_device(x, gxi, count, first);
    if (!gx) {
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
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        (void)copy_tfqmr_x_to_cpu(x, gx, count);
    };

    if (first) {
        *nstat = 0;
        *steperr = -1;
        if (!norm2(handle, n, gb, norm, solver_name) || !std::isfinite(*norm) ||
            *norm <= CU_REAL(0)) {
            fail(11);
            return;
        }
        *nou = 1;
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    if (*nou == 1) {
        if (!launch(tfqmr_initialize_residual_wrapper(gxi, gxr, gb, gw, *lda, n),
                    "initialize residual", nstat, steperr)) {
            fail(16);
            return;
        }
        *nou = 2;
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    bool start_halfstep = false;
    if (*nou == 2) {
        if (!launch(tfqmr_initialize_vectors_wrapper(gxr, gw, *lda, n), "initialize vectors", nstat,
                    steperr)) {
            fail(16);
            return;
        }
        CUCOMPLEX q = CUFFT_ZERO;
        if (!dotc(handle, n, gw + static_cast<size_t>(5) * *lda, gw + static_cast<size_t>(2) * *lda,
                  &q, solver_name)) {
            fail(16);
            return;
        }
        *rho = q;
        *tau = CUFFT_REAL(*norm);
        *theta = CUFFT_ZERO;
        *eta = CUFFT_ZERO;
        *itno = -1;
        ++(*itno); // Fortran label 100: first half-step is iteration zero.
        start_halfstep = true;
    } else if (*nou == 3) {
        const int odd = ((*itno % 2) != 0);
        if (!launch(tfqmr_continue_wrapper(gxi, gxr, gw, *beta, odd, *lda, n),
                    "continue recurrence", nstat, steperr)) {
            fail(16);
            return;
        }
        if (*itno > *maxit) {
            *nstat = -1;
            *steperr = 1;
            (void)publish_xi(xi, gxi, count, nstat, steperr);
            (void)copy_tfqmr_x_to_cpu(x, gx, count);
            return;
        }
        ++(*itno);
        start_halfstep = true;
    } else // nou == 4 follows the original Fortran label 40.
    {
        start_halfstep = false;
    }

    if (start_halfstep && ((*itno % 2) == 0)) {
        CUCOMPLEX denominator = CUFFT_ZERO;
        if (!dotc(handle, n, gw + static_cast<size_t>(5) * *lda, gw + static_cast<size_t>(3) * *lda,
                  &denominator, solver_name) ||
            scalar_is_zero(denominator) || scalar_is_zero(*rho)) {
            fail(1);
            return;
        }
        *alpha = device_value(host_value(*rho) / host_value(denominator));
        if (!launch(tfqmr_prepare_uplus_wrapper(gxi, gw, *alpha, *lda, n), "prepare u-plus", nstat,
                    steperr)) {
            fail(16);
            return;
        }
    }

    // Use the old theta/eta values to form d before updating either scalar.
    const HostComplex alpha_h = host_value(*alpha);
    const HostComplex theta_h = host_value(*theta);
    const HostComplex eta_h = host_value(*eta);
    if (alpha_h == HostComplex(0, 0)) {
        fail(1);
        return;
    }
    const HostComplex dcoef_h = theta_h * theta_h / alpha_h * eta_h;
    if (!launch(tfqmr_update_w_wrapper(gw, *alpha, *lda, n), "update w", nstat, steperr)) {
        fail(16);
        return;
    }
    CUCOMPLEX thtmp = CUFFT_ZERO;
    if (!norm2(handle, n, gw + static_cast<size_t>(1) * *lda, &thtmp.x, solver_name)) {
        fail(16);
        return;
    }
    // TFQMR's Fortran reduction is the Hermitian norm; cuBLAS returns its sqrt.
    const HostComplex tau_h = host_value(*tau);
    if (tau_h == HostComplex(0, 0)) {
        fail(1);
        return;
    }
    const HostComplex theta_new = HostComplex(static_cast<CU_REAL>(thtmp.x), 0) / tau_h;
    const HostComplex cc = HostComplex(1, 0) / std::sqrt(HostComplex(1, 0) + theta_new * theta_new);
    *tau = device_value(tau_h * theta_new * cc);
    *theta = device_value(theta_new);
    *eta = device_value(cc * cc * alpha_h);
    if (!launch(tfqmr_update_halfstep_wrapper(gx, gw, device_value(dcoef_h), *eta, *lda, n),
                "update half-step", nstat, steperr)) {
        fail(16);
        return;
    }

    const CU_REAL residual = std::sqrt(static_cast<CU_REAL>(*itno + 1)) *
                             static_cast<CU_REAL>(std::hypot((*tau).x, (*tau).y)) / *norm;
    if (residual <= *tol) {
        *nstat = 1;
        *nou = 4;
        if (!copy_tfqmr_x_to_cpu(x, gx, count)) {
            *nstat = -1;
            *steperr = 15;
        }
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    const int odd = ((*itno % 2) != 0);
    if (odd) {
        const CUCOMPLEX rho0 = *rho;
        CUCOMPLEX next_rho = CUFFT_ZERO;
        if (!dotc(handle, n, gw + static_cast<size_t>(5) * *lda, gw + static_cast<size_t>(1) * *lda,
                  &next_rho, solver_name) ||
            scalar_is_zero(rho0)) {
            fail(1);
            return;
        }
        *rho = next_rho;
        *beta = device_value(host_value(next_rho) / host_value(rho0));
    }
    if (!launch(tfqmr_end_halfstep_wrapper(gxi, gw, odd ? *beta : CUFFT_ZERO, odd, *lda, n),
                "finish half-step", nstat, steperr)) {
        fail(16);
        return;
    }
    *nou = 3;
    (void)publish_xi(xi, gxi, count, nstat, steperr);
}
} // namespace

//-----------------------------------------------------------
extern "C" void tfqmr_(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                       const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                       const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                       CUCOMPLEX *beta, CUCOMPLEX *rho, CUCOMPLEX *tau, CUCOMPLEX *theta,
                       CUCOMPLEX *eta, int *nstat, int *steperr) {
    tfqmr_impl(x, xi, xr, b, lda, ndim, nlar, nou, work, itno, maxit, tol, norm, alpha, beta, rho,
               tau, theta, eta, nstat, steperr);
}
