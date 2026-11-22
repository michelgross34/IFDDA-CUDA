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

//-----------------------------------------------------------
bool publish_xi(COMPLEX *cpu_xi, const CUCOMPLEX *gpu_xi, size_t count, int *nstat,
                int *steperr) {
    if (copy_gpbicg_results_to_cpu(cpu_xi, nullptr, gpu_xi, nullptr, count, 0, true, false))
        return true;
    *nstat = -1;
    *steperr = 15;
    return false;
}

//-----------------------------------------------------------
void qmr_bicgstab_impl(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                       const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                       const int *maxit, const CU_REAL *tol, CU_REAL *residu, CU_REAL *norm,
                       CUCOMPLEX *alpha, CUCOMPLEX *rho, CUCOMPLEX *omega, CUCOMPLEX *tau,
                       CUCOMPLEX *taut, CUCOMPLEX *theta, CUCOMPLEX *thetat, CUCOMPLEX *eta,
                       CUCOMPLEX *etat, const int *methode, int *nstat, int *steperr) {
    if (!x || !xi || !xr || !b || !lda || !ndim || !nlar || !nou || !work || !itno || !maxit ||
        !tol || !residu || !norm || !alpha || !rho || !omega || !tau || !taut || !theta ||
        !thetat || !eta || !etat || !methode || !nstat || !steperr || *ndim <= 0 || *lda < *ndim ||
        *nlar < 9 || *maxit < 0) {
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
    const auto launch = [&](cudaError_t status, const char *op) {
        if (check_cuda(status, "QMRBICGSTAB CUDA", op))
            return true;
        *nstat = -1;
        *steperr = 16;
        return false;
    };
    const auto dot = [&](const CUCOMPLEX *a, const CUCOMPLEX *c, CUCOMPLEX *out) {
        return dotc(handle, n, a, c, out, "QMRBICGSTAB CUDA");
    };
    bool initial_step = false;
    *nstat = 0;
    *steperr = -1;

    if (first) {
        if (!norm2(handle, n, gb, norm, "QMRBICGSTAB CUDA") || !std::isfinite(*norm)) {
            fail(11);
            return;
        }
        *nou = 1;
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    if (*nou == 1) {
        if (!launch(qmrbicgstab_initialize_wrapper(gxi, gxr, gb, gw, *lda, n),
                    "initialize residual and work vectors")) {
            fail(16);
            return;
        }
        CU_REAL rr = 0;
        if (!norm2(handle, n, gw, &rr, "QMRBICGSTAB CUDA")) {
            fail(16);
            return;
        }
        const CU_REAL initial = (*norm > CU_REAL(0)) ? rr / (*norm) : rr;
        std::printf("Initial Residue for QMRBICGSTAB %.16g Norm %.16g\n",
                    static_cast<double>(initial), static_cast<double>(*norm));
        *rho = device_value(HostComplex(1, 0));
        *alpha = device_value(HostComplex(1, 0));
        *omega = device_value(HostComplex(1, 0));
        *tau = device_value(HostComplex(rr, 0));
        *theta = CUFFT_ZERO;
        *eta = CUFFT_ZERO;
        *itno = 1;
        initial_step = true;
    } else if (*nou == 2) {
        CUCOMPLEX a = CUFFT_ZERO;
        if (!dot(gw + size_t(4) * (*lda), gxr, &a)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(a)) {
            fail(1);
            return;
        }
        *alpha = device_value(host_value(*rho) / host_value(a));
        CU_REAL snorm = 0;
        if (!launch(qmrbicgstab_prepare_s_wrapper(gxi, gxr, gw, *alpha, *lda, n), "prepare s") ||
            !norm2(handle, n, gw + size_t(5) * (*lda), &snorm, "QMRBICGSTAB CUDA")) {
            fail(16);
            return;
        }
        if (scalar_is_zero(*tau)) {
            fail(2);
            return;
        }
        const HostComplex tau_h = host_value(*tau);
        const CU_REAL tt = snorm / std::abs(tau_h);
        const CU_REAL cc = CU_REAL(1) / std::sqrt(CU_REAL(1) + tt * tt);
        *thetat = device_value(HostComplex(tt, 0));
        *taut = device_value(HostComplex(std::abs(tau_h) * tt * cc, 0));
        *etat = device_value(HostComplex(cc * cc, 0) * host_value(*alpha));
        if (!launch(qmrbicgstab_prepare_x_wrapper(gx, gw, *alpha, *theta, *eta, *etat, *lda, n),
                    "prepare x tilde")) {
            fail(16);
            return;
        }
        *nou = 3;
        if (!publish_xi(xi, gxi, count, nstat, steperr))
            return;
        return;
    }

    // QMR's initial setup falls through to label 100, as does each
    // nonconverged NOU=3 return in the Fortran reverse-communication routine.
    if (*nou == 3) {
        if (!launch(qmrbicgstab_save_t_wrapper(gxr, gw, *lda, n), "save A*s")) {
            fail(16);
            return;
        }
        CUCOMPLEX den = CUFFT_ZERO, num = CUFFT_ZERO;
        const CUCOMPLEX *s = gw + size_t(5) * (*lda);
        const CUCOMPLEX *t = gw + size_t(8) * (*lda);
        if (*methode == 1) {
            if (!dot(t, t, &den) || !dot(s, t, &num)) {
                fail(16);
                return;
            }
        } else {
            if (!dot(s, t, &den) || !dot(s, s, &num)) {
                fail(16);
                return;
            }
        }
        if (scalar_is_zero(den)) {
            fail(3);
            return;
        }
        *omega = device_value(host_value(num) / host_value(den));
        if (!launch(qmrbicgstab_update_xr_wrapper(gw, *omega, *lda, n), "update residual")) {
            fail(16);
            return;
        }
        CU_REAL rnorm = 0;
        if (!norm2(handle, n, gw, &rnorm, "QMRBICGSTAB CUDA")) {
            fail(16);
            return;
        }
        if (scalar_is_zero(*taut)) {
            fail(4);
            return;
        }
        const CU_REAL th = rnorm / std::abs(host_value(*taut));
        const CU_REAL cc = CU_REAL(1) / std::sqrt(CU_REAL(1) + th * th);
        *theta = device_value(HostComplex(th, 0));
        *tau = device_value(HostComplex(std::abs(host_value(*taut)) * th * cc, 0));
        *eta = device_value(HostComplex(cc * cc, 0) * host_value(*omega));
        if (scalar_is_zero(*omega)) {
            fail(5);
            return;
        }
        if (!launch(qmrbicgstab_finalize_x_wrapper(gx, gw, *omega, *eta, *thetat, *etat, *lda, n),
                    "update solution")) {
            fail(16);
            return;
        }
        *residu = std::sqrt(static_cast<CU_REAL>(*itno + 1)) * std::abs(host_value(*tau)) / (*norm);
        std::printf("RESIDU %.16g iteration %d\n", static_cast<double>(*residu), *itno);
        if (*residu <= *tol) {
            if (!launch(qmrbicgstab_publish_x_wrapper(gxi, gx, n), "publish solution")) {
                fail(16);
                return;
            }
            *nstat = 1;
            *nou = 4;
            if (!publish_xi(xi, gxi, count, nstat, steperr) || !copy_tfqmr_x_to_cpu(x, gx, count)) {
                *nstat = -1;
                *steperr = 15;
            }
            return;
        }
        if (*itno > *maxit) {
            return;
        }
        ++(*itno);
    }

    // Label 100: start a new BiCG direction and request its product Ax.
    const CUCOMPLEX rho0 = initial_step ? device_value(HostComplex(1, 0)) : *rho;
    CUCOMPLEX rhon = CUFFT_ZERO;
    if (!dot(gw + size_t(4) * (*lda), gw, &rhon)) {
        fail(16);
        return;
    }
    if (scalar_is_zero(rho0) || scalar_is_zero(*omega)) {
        fail(6);
        return;
    }
    *rho = rhon;
    const HostComplex beta =
        host_value(*rho) * host_value(*alpha) / host_value(rho0) / host_value(*omega);
    if (!launch(qmrbicgstab_update_p_wrapper(gxi, gw, device_value(beta), *omega, *lda, n),
                "update direction")) {
        fail(16);
        return;
    }
    *nou = 2;
    (void)publish_xi(xi, gxi, count, nstat, steperr);
}
} // namespace

//-----------------------------------------------------------
extern "C" void qmrbicgstab_(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, COMPLEX *b,
                             const int *lda, const int *ndim, const int *nlar, int *nou,
                             COMPLEX *work, int *itno, const int *maxit, const CU_REAL *tol,
                             CU_REAL *residu, CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *rho,
                             CUCOMPLEX *omega, CUCOMPLEX *tau, CUCOMPLEX *taut, CUCOMPLEX *theta,
                             CUCOMPLEX *thetat, CUCOMPLEX *eta, CUCOMPLEX *etat, const int *methode,
                             int *nstat, int *steperr) {
    qmr_bicgstab_impl(x, xi, xr, b, lda, ndim, nlar, nou, work, itno, maxit, tol, residu, norm,
                      alpha, rho, omega, tau, taut, theta, thetat, eta, etat, methode, nstat,
                      steperr);
}
