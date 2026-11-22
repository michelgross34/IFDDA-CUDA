#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cmath>

namespace {
using cuda_utils::check_cuda;
using cuda_utils::device_value;
using cuda_utils::dotc;
using cuda_utils::get_cublas_handle;
using cuda_utils::host_value;
using cuda_utils::norm2;
using cuda_utils::scalar_is_zero;
constexpr const char *solver_name = "PIMZBICGSTAB CUDA";

//-----------------------------------------------------------
bool publish_xi(COMPLEX *cpu, CUCOMPLEX *gpu, size_t n, int *ns, int *se) {
    if (copy_gpbicg_results_to_cpu(cpu, nullptr, gpu, nullptr, n, 0, true, false))
        return true;
    *ns = -1;
    *se = 15;
    return false;
}

//-----------------------------------------------------------
bool publish_x(COMPLEX *cpu, const CUCOMPLEX *gpu, size_t n, int *ns, int *se) {
    if (copy_tfqmr_x_to_cpu(cpu, gpu, n))
        return true;
    *ns = -1;
    *se = 15;
    return false;
}

//-----------------------------------------------------------
bool launch(cudaError_t status, const char *operation, int *ns, int *se) {
    if (check_cuda(status, solver_name, operation))
        return true;
    *ns = -1;
    *se = 16;
    return false;
}

//-----------------------------------------------------------
void pimzbicgstab_impl(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, const COMPLEX *b, const int *lda,
                       const int *nlar, const int *ndim, int *nou, COMPLEX *work,
                       COMPLEX *alpha, COMPLEX *omega, COMPLEX *rho, CU_REAL *norm,
                       const CU_REAL *tole, int *itno, const int *maxit, int *nstat, int *steperr) {
    if (!x || !xi || !xr || !b || !lda || !nlar || !ndim || !nou || !work || !alpha || !omega ||
        !rho || !norm || !tole || !itno || !maxit || !nstat || !steperr || *ndim <= 0 ||
        *lda < *ndim || *nlar < 10 || *maxit <= 0) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 12;
        return;
    }
    const int n = *ndim;
    const size_t count = static_cast<size_t>(n),
                 nw = static_cast<size_t>(*lda) * static_cast<size_t>(*nlar);
    const bool first = (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4);
    CUCOMPLEX *gxi = nullptr, *gxr = nullptr, *gb = nullptr, *gw = nullptr;
    if (!get_gpbicg_device_arrays_resident(xi, xr, b, work, count, nw, first, &gxi, &gxr, &gb,
                                           &gw)) {
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
    cublasHandle_t h = get_cublas_handle();
    if (!h) {
        *nstat = -1;
        *steperr = 14;
        return;
    }
    const auto fail = [&](int code) {
        *nstat = -1;
        *steperr = code;
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        (void)publish_x(x, gx, count, nstat, steperr);
    };

    if (first) {
        *nstat = 0;
        *steperr = -1;
        if (!norm2(h, n, gb, norm, solver_name) || !std::isfinite(*norm) || *norm <= CU_REAL(0)) {
            fail(1);
            return;
        }
        if (!launch(pimzbicgstab_save_rhs_wrapper(gb, gw, *lda, n), "save right hand side", nstat,
                    steperr)) {
            fail(16);
            return;
        }
        *nou = 1;
        return;
    }

    bool enter_loop = false;
    if (*nou == 1) {
        if (!launch(pimzbicgstab_initialize_residual_wrapper(gxi, gxr, gw, *lda, n),
                    "initialize residual", nstat, steperr)) {
            fail(16);
            return;
        }
        CU_REAL rnorm = 0;
        if (!norm2(h, n, gw, &rnorm, solver_name)) {
            fail(1);
            return;
        }
        std::printf("INIT RESIDU %.16g NORM %.16g\n", double(rnorm / *norm), double(*norm));
        *rho = COMPLEX(CU_REAL(1.0), CU_REAL(0.0));
        *alpha = COMPLEX(CU_REAL(1.0), CU_REAL(0.0));
        *omega = COMPLEX(CU_REAL(1.0), CU_REAL(0.0));
        *itno = 0;
        enter_loop = true;
    } else if (*nou == 4)
        enter_loop = true;

    if (enter_loop) {
        ++(*itno);
        if (*itno >= *maxit) {
            *nstat = -3;
            *steperr = 6;
            *itno = *maxit;
            (void)publish_x(x, gx, count, nstat, steperr);
            return;
        }
        const CUCOMPLEX rho0 = device_value(*rho);
        CUCOMPLEX next_rho = CUFFT_ZERO;
        if (!dotc(h, n, gw + static_cast<size_t>(1) * *lda, gw, &next_rho, solver_name)) {
            fail(1);
            return;
        }
        const CUCOMPLEX denom = CUFFT_MUL(rho0, device_value(*omega));
        if (scalar_is_zero(denom)) {
            *nstat = -3;
            *steperr = 6;
            return;
        }
        const CUCOMPLEX beta =
            device_value(host_value(next_rho) * host_value(device_value(*alpha)) / host_value(denom));
        *rho = host_value(next_rho);
        if (!launch(pimzbicgstab_prepare_direction_wrapper(gxi, gw, beta, device_value(*omega), *lda, n),
                    "prepare p", nstat, steperr)) {
            fail(16);
            return;
        }
        *nou = 2;
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    if (*nou == 2) {
        if (!launch(gpbicg_save_xr_column_wrapper(gxr, gw, *lda, n, 3), "save v", nstat, steperr)) {
            fail(16);
            return;
        }
        CUCOMPLEX xxi = CUFFT_ZERO;
        if (!dotc(h, n, gw + static_cast<size_t>(1) * *lda, gw + static_cast<size_t>(3) * *lda,
                  &xxi, solver_name)) {
            fail(1);
            return;
        }
        if (scalar_is_zero(xxi)) {
            *nstat = -3;
            *steperr = 10;
            (void)publish_x(x, gx, count, nstat, steperr);
            return;
        }
        *alpha = *rho / host_value(xxi);
        if (!launch(pimzbicgstab_prepare_s_wrapper(gxi, gxr, gw, device_value(*alpha), *lda, n), "prepare s",
                    nstat, steperr)) {
            fail(16);
            return;
        }
        CU_REAL snorm = 0;
        if (!norm2(h, n, gw + static_cast<size_t>(4) * *lda, &snorm, solver_name)) {
            fail(1);
            return;
        }
        if (snorm < CU_REAL(1.0e-15)) {
            *nstat = -2;
            *steperr = 12;
            if (!launch(pimzbicgstab_soft_breakdown_wrapper(gx, gw, *lda, n),
                        "soft breakdown solution", nstat, steperr)) {
                fail(16);
                return;
            }
            (void)publish_x(x, gx, count, nstat, steperr);
            return;
        }
        *nou = 3;
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    if (*nou == 3) {
        if (!launch(gpbicg_save_xr_column_wrapper(gxr, gw, *lda, n, 5), "save t", nstat, steperr)) {
            fail(16);
            return;
        }
        CUCOMPLEX tt = CUFFT_ZERO, ts = CUFFT_ZERO;
        if (!dotc(h, n, gw + static_cast<size_t>(5) * *lda, gw + static_cast<size_t>(5) * *lda, &tt,
                  solver_name) ||
            !dotc(h, n, gw + static_cast<size_t>(5) * *lda, gw + static_cast<size_t>(4) * *lda, &ts,
                  solver_name)) {
            fail(1);
            return;
        }
        if (scalar_is_zero(tt)) {
            *nstat = -3;
            *steperr = 14;
            goto finalize;
        }
        *omega = host_value(ts) / host_value(tt);
        if (!launch(pimzbicgstab_update_xr_wrapper(gx, gw, device_value(*alpha), device_value(*omega), *lda, n),
                    "update x and r", nstat, steperr)) {
            fail(16);
            return;
        }
        {
            CU_REAL rnorm = 0;
            if (!norm2(h, n, gw, &rnorm, solver_name)) {
                fail(1);
                return;
            }
            const CU_REAL residual = rnorm / *norm;
            std::printf("RESIDU %.16g\n", double(residual));
            if (*itno > *maxit) {
                *nstat = -1;
                *itno = *maxit;
                goto finalize;
            }
            if (residual <= *tole) {
                *nstat = 1;
                *nou = 4;
                goto finalize;
            }
        }
        enter_loop = true;
        if (enter_loop) {
            ++(*itno);
            if (*itno >= *maxit) {
                *nstat = -3;
                *steperr = 6;
                *itno = *maxit;
                goto finalize;
            }
            const CUCOMPLEX rho0 = device_value(*rho);
            CUCOMPLEX next_rho = CUFFT_ZERO;
            if (!dotc(h, n, gw + static_cast<size_t>(1) * *lda, gw, &next_rho, solver_name)) {
                fail(1);
                return;
            }
            const CUCOMPLEX denom = CUFFT_MUL(rho0, device_value(*omega));
            if (scalar_is_zero(denom)) {
                *nstat = -3;
                *steperr = 6;
                goto finalize;
            }
            const CUCOMPLEX beta =
                device_value(host_value(next_rho) * host_value(device_value(*alpha)) / host_value(denom));
            *rho = host_value(next_rho);
            if (!launch(pimzbicgstab_prepare_direction_wrapper(gxi, gw, beta, device_value(*omega), *lda, n),
                        "prepare p", nstat, steperr)) {
                fail(16);
                return;
            }
            *nou = 2;
            (void)publish_xi(xi, gxi, count, nstat, steperr);
            return;
        }
    }
finalize:
    (void)publish_x(x, gx, count, nstat, steperr);
}
} // namespace

//-----------------------------------------------------------
extern "C" void pimzbicgstab_(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, const COMPLEX *b,
                              const int *lda, const int *nlar, const int *ndim, int *nou,
                              COMPLEX *work, COMPLEX *alpha, COMPLEX *omega, COMPLEX *rho,
                              CU_REAL *norm, const CU_REAL *tole, int *itno, const int *maxit,
                              int *nstat, int *steperr) {
    pimzbicgstab_impl(x, xi, xr, b, lda, nlar, ndim, nou, work, alpha, omega, rho, norm, tole, itno,
                      maxit, nstat, steperr);
}
