#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
using cuda_utils::check_cuda;
using cuda_utils::device_value;
using cuda_utils::dotc;
using cuda_utils::dotu;
using cuda_utils::get_cublas_handle;
using cuda_utils::host_value;
using cuda_utils::HostComplex;
using cuda_utils::norm2;

//-----------------------------------------------------------
bool publish_xi(COMPLEX *cpu_xi, const CUCOMPLEX *gpu_xi, size_t count) {
    return copy_gpbicg_results_to_cpu(cpu_xi, nullptr, gpu_xi, nullptr, count, 0, true, false);
}

//-----------------------------------------------------------
bool combine(CUCOMPLEX *work, CUCOMPLEX *x, int lda, int n, int destination, const int *columns,
             const CUCOMPLEX *coefficients, int terms, bool add = false) {
    return check_cuda(gpbicgstabl_combine_wrapper(work, x, lda, n, destination, columns,
                                                  coefficients, terms, add ? 1 : 0),
                      "PIMZQMR CUDA", "combine vectors");
}

//-----------------------------------------------------------
void pimzqmr_impl(COMPLEX *xs, COMPLEX *xi, COMPLEX *xr, const COMPLEX *b, COMPLEX *work,
                  CU_REAL *norm, const int *lda, const int *ndim, const int *nlar,
                  CUCOMPLEX *lambda, CUCOMPLEX *kappa, CUCOMPLEX *theta, CUCOMPLEX *gamma,
                  CUCOMPLEX *ksi, CUCOMPLEX *rho, CUCOMPLEX *epsilon, CUCOMPLEX *mu, CUCOMPLEX *tau,
                  CUCOMPLEX *dots1, CUCOMPLEX *dots2, CUCOMPLEX *dots3, CUCOMPLEX *dots4, int *nou,
                  int *nt, int *itno, const int *maxit, CU_REAL *tole, const CU_REAL *tol,
                  int *nstat, int *steperr) {
    if (!xs || !xi || !xr || !b || !work || !norm || !lda || !ndim || !nlar || !lambda || !kappa ||
        !theta || !gamma || !ksi || !rho || !epsilon || !mu || !tau || !dots1 || !dots2 || !dots3 ||
        !dots4 || !nou || !nt || !itno || !maxit || !tole || !tol || !nstat || !steperr ||
        *ndim <= 0 || *lda < *ndim || *nlar < 9 || *maxit < 0) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 16;
        return;
    }

    const int n = *ndim;
    const size_t vector_count = static_cast<size_t>(n);
    size_t work_count = 0;
    if (!cuda_utils::checked_multiply(static_cast<size_t>(*lda), static_cast<size_t>(*nlar),
                                      &work_count)) {
        *nstat = -1;
        *steperr = 16;
        return;
    }

    CUCOMPLEX *gxi = nullptr, *gxr = nullptr, *gb = nullptr, *gwork = nullptr;
    const bool first_call = (*nou == 0);
    if (!get_gpbicg_device_arrays_resident(xi, xr, b, work, vector_count, work_count, first_call,
                                           &gxi, &gxr, &gb, &gwork)) {
        *nstat = -1;
        *steperr = 16;
        return;
    }
    CUCOMPLEX *gxs = get_tfqmr_device_x_from_device(xs, gxi, vector_count, first_call);
    cublasHandle_t handle = get_cublas_handle();
    if (!gxs || !handle) {
        *nstat = -1;
        *steperr = 16;
        return;
    }
    const auto launch = [&](cudaError_t status, const char *operation) {
        return check_cuda(status, "PIMZQMR CUDA", operation);
    };
    const auto fail_cuda = [&]() {
        *nstat = -1;
        *steperr = 16;
        (void)publish_xi(xi, gxi, vector_count);
    };
    const auto set_error = [&](int code) {
        *nstat = -3;
        *steperr = code;
    };
    const auto hermitian_dot = [&](const CUCOMPLEX *a, const CUCOMPLEX *c, CUCOMPLEX *result) {
        return dotc(handle, n, a, c, result, "PIMZQMR CUDA");
    };
    const auto bilinear_dot = [&](const CUCOMPLEX *a, const CUCOMPLEX *c, CUCOMPLEX *result) {
        return dotu(handle, n, a, c, result, "PIMZQMR CUDA");
    };
    const auto copy_column_to_xi = [&](int column) {
        return launch(idrs_copy_work_to_xi_wrapper(gxi, gwork, column, *lda, n),
                      "publish reverse-communication vector");
    };
    const auto begin_iteration = [&]() -> bool {
        ++(*itno);
        if (*itno > *maxit) {
            *nstat = -1;
            *steperr = 15;
            *itno = *maxit;
            return false;
        }
        if (std::abs(host_value(*gamma)) == CU_REAL(0)) {
            set_error(6);
            return false;
        }
        if (std::abs(host_value(*ksi)) == CU_REAL(0)) {
            set_error(7);
            return false;
        }
        const HostComplex g = host_value(*gamma);
        const HostComplex k = host_value(*ksi);
        const HostComplex m = host_value(*mu);
        const HostComplex gm = g * m;
        const int pcols[2] = {6, 1};
        const CUCOMPLEX pcoef[2] = {device_value(HostComplex(1, 0) / g), device_value(-m)};
        const int qcols[2] = {8, 3};
        const CUCOMPLEX qcoef[2] = {device_value(HostComplex(1, 0) / k), device_value(-gm / k)};
        if (!combine(gwork, gxs, *lda, n, 1, pcols, pcoef, 2) ||
            !combine(gwork, gxs, *lda, n, 3, qcols, qcoef, 2) || !copy_column_to_xi(1))
            return false;
        *nt = 1;
        *nou = 3;
        return true;
    };

    *nstat = 0;
    *steperr = -1;

    if (*nou == 0) {
        if (!norm2(handle, n, gb, norm, "PIMZQMR CUDA") ||
            !std::isfinite(static_cast<double>(*norm))) {
            fail_cuda();
            return;
        }
        *lambda = device_value(HostComplex(1, 0));
        *kappa = device_value(HostComplex(-1, 0));
        *theta = device_value(HostComplex(-1, 0));
        *itno = 0;
        *nt = 1;
        *nou = 1;
        (void)publish_xi(xi, gxi, vector_count);
        return;
    }

    if (*nou == 1) {
        if (!launch(pimzqmr_initialize_wrapper(gxr, gb, gwork, *lda, n),
                    "initialize residual and QMR vectors")) {
            fail_cuda();
            return;
        }
        if (!hermitian_dot(gwork + size_t(6) * *lda, gwork + size_t(6) * *lda, dots1) ||
            !hermitian_dot(gwork + size_t(7) * *lda, gwork + size_t(7) * *lda, dots2) ||
            !bilinear_dot(gwork + size_t(6) * *lda, gwork + size_t(7) * *lda, dots3)) {
            fail_cuda();
            return;
        }

        char legacy_shadow[8] = {};
        const bool use_legacy = std::getenv("IFDDA_QMR_LEGACY_SHADOW") &&
                                std::getenv("IFDDA_QMR_LEGACY_SHADOW")[0] == '1';
        if (!use_legacy && host_value(*dots1).real() > CU_REAL(0) &&
            host_value(*dots2).real() > CU_REAL(0)) {
            const CU_REAL denominator =
                std::sqrt(host_value(*dots1).real()) * std::sqrt(host_value(*dots2).real());
            const CU_REAL ratio = std::abs(host_value(*dots3)) / denominator;
            if (ratio < CU_REAL(64) * std::numeric_limits<CU_REAL>::epsilon()) {
                if (!launch(pimzqmr_shadow_wrapper(gxi, gwork, *lda, n),
                            "select nondegenerate shadow vector") ||
                    !hermitian_dot(gwork + size_t(7) * *lda, gwork + size_t(7) * *lda, dots2) ||
                    !bilinear_dot(gwork + size_t(6) * *lda, gwork + size_t(7) * *lda, dots3)) {
                    fail_cuda();
                    return;
                }
            } else if (!copy_column_to_xi(7)) {
                fail_cuda();
                return;
            }
        } else if (!copy_column_to_xi(7)) {
            fail_cuda();
            return;
        }

        if (host_value(*dots1).real() == CU_REAL(0)) {
            *nou = 5;
            *nt = 1;
            *nstat = 1;
            if (!copy_tfqmr_x_to_cpu(xs, gxs, vector_count)) {
                *nstat = -1;
                *steperr = 16;
            }
            (void)publish_xi(xi, gxi, vector_count);
            return;
        }
        *nt = 2;
        *nou = 2;
        (void)publish_xi(xi, gxi, vector_count);
        return;
    }

    if (*nou == 2) {
        if (!launch(idrs_copy_xr_wrapper(gwork, gxr, 8, *lda, n), "save transpose product") ||
            !bilinear_dot(gwork + size_t(6) * *lda, gwork + size_t(8) * *lda, dots4)) {
            fail_cuda();
            return;
        }
        *gamma = device_value(std::sqrt(host_value(*dots1)));
        *ksi = device_value(std::sqrt(host_value(*dots2)));
        *rho = *dots3;
        *epsilon = *dots4;
        *mu = CUFFT_ZERO;
        if (std::abs(host_value(*rho)) == CU_REAL(0)) {
            set_error(5);
            (void)publish_xi(xi, gxi, vector_count);
            return;
        }
        *tau = device_value(host_value(*epsilon) / host_value(*rho));
        if (!begin_iteration()) {
            if (*nstat == 0)
                fail_cuda();
            (void)publish_xi(xi, gxi, vector_count);
            return;
        }
        (void)publish_xi(xi, gxi, vector_count);
        return;
    }

    if (*nou == 3) {
        const HostComplex tau_gamma = host_value(*tau) / host_value(*gamma);
        const HostComplex tau_ksi = host_value(*tau) / host_value(*ksi);
        const int first_cols[2] = {2, 6};
        const CUCOMPLEX first_coeffs[2] = {device_value(HostComplex(1, 0)),
                                           device_value(-tau_gamma)};
        const int second_cols[2] = {3, 7};
        const CUCOMPLEX second_coeffs[2] = {device_value(HostComplex(1, 0)),
                                            device_value(-tau_ksi)};
        if (!launch(idrs_copy_xr_wrapper(gwork, gxr, 2, *lda, n), "save product Ap") ||
            !combine(gwork, gxs, *lda, n, 6, first_cols, first_coeffs, 2) ||
            !combine(gwork, gxs, *lda, n, 7, second_cols, second_coeffs, 2) ||
            !hermitian_dot(gwork + size_t(6) * *lda, gwork + size_t(6) * *lda, dots1) ||
            !hermitian_dot(gwork + size_t(7) * *lda, gwork + size_t(7) * *lda, dots2) ||
            !bilinear_dot(gwork + size_t(6) * *lda, gwork + size_t(7) * *lda, dots3) ||
            !copy_column_to_xi(7)) {
            fail_cuda();
            return;
        }
        *nt = 2;
        *nou = 4;
        (void)publish_xi(xi, gxi, vector_count);
        return;
    }

    if (*nou == 4) {
        if (!launch(idrs_copy_xr_wrapper(gwork, gxr, 8, *lda, n), "save transpose product") ||
            !bilinear_dot(gwork + size_t(6) * *lda, gwork + size_t(8) * *lda, dots4)) {
            fail_cuda();
            return;
        }
        const HostComplex gamma0 = host_value(*gamma);
        const HostComplex ksi0 = host_value(*ksi);
        const HostComplex rho0 = host_value(*rho);
        const HostComplex tau0 = host_value(*tau);
        *gamma = device_value(std::sqrt(host_value(*dots1)));
        *ksi = device_value(std::sqrt(host_value(*dots2)));
        *rho = *dots3;
        *epsilon = *dots4;
        const HostComplex den_mu = host_value(*gamma) * tau0 * rho0;
        if (std::abs(den_mu) == CU_REAL(0)) {
            set_error(12);
            (void)publish_xi(xi, gxi, vector_count);
            return;
        }
        *mu = device_value(gamma0 * ksi0 * host_value(*rho) / den_mu);
        if (std::abs(host_value(*rho)) == CU_REAL(0)) {
            set_error(13);
            (void)publish_xi(xi, gxi, vector_count);
            return;
        }
        *tau = device_value(host_value(*epsilon) / host_value(*rho) -
                            host_value(*gamma) * host_value(*mu));
        const CU_REAL abs_tau0_squared = std::norm(tau0);
        const CU_REAL abs_gamma_squared = std::norm(host_value(*gamma));
        const HostComplex den = host_value(*lambda) * abs_tau0_squared + abs_gamma_squared;
        if (std::abs(den) == CU_REAL(0)) {
            set_error(14);
            (void)publish_xi(xi, gxi, vector_count);
            return;
        }
        const HostComplex lambda0 = host_value(*lambda);
        const HostComplex kappa0 = host_value(*kappa);
        *theta = device_value(abs_tau0_squared * (HostComplex(1, 0) - lambda0) / den);
        *kappa = device_value(-(gamma0 * std::conj(tau0) * kappa0) / den);
        *lambda = device_value(lambda0 * abs_tau0_squared / den);

        const int dcols[2] = {4, 1};
        const CUCOMPLEX dcoeffs[2] = {*theta, *kappa};
        const int scols[2] = {5, 2};
        const CUCOMPLEX scoeffs[2] = {*theta, *kappa};
        const int xcols[1] = {4};
        const CUCOMPLEX xcoeffs[1] = {device_value(HostComplex(1, 0))};
        const int rcols[1] = {5};
        const CUCOMPLEX rcoeffs[1] = {device_value(HostComplex(-1, 0))};
        if (!combine(gwork, gxs, *lda, n, 4, dcols, dcoeffs, 2) ||
            !combine(gwork, gxs, *lda, n, 5, scols, scoeffs, 2) ||
            !combine(gwork, gxs, *lda, n, -1, xcols, xcoeffs, 1, true) ||
            !combine(gwork, gxs, *lda, n, 0, rcols, rcoeffs, 1, true)) {
            fail_cuda();
            return;
        }
        CU_REAL residual_norm = 0;
        if (!norm2(handle, n, gwork, &residual_norm, "PIMZQMR CUDA")) {
            fail_cuda();
            return;
        }
        *tole = (*norm > CU_REAL(0)) ? residual_norm / *norm : residual_norm;
        if ((*itno % 50) == 0)
            std::printf("RESIDUE %.16g iteration %d\n", static_cast<double>(*tole), *itno);
        if (*tole <= *tol) {
            *nou = 5;
            *nstat = 1;
            if (!copy_tfqmr_x_to_cpu(xs, gxs, vector_count)) {
                *nstat = -1;
                *steperr = 16;
            }
            (void)publish_xi(xi, gxi, vector_count);
            return;
        }
        if (!begin_iteration()) {
            if (*nstat == 0)
                fail_cuda();
            (void)publish_xi(xi, gxi, vector_count);
            return;
        }
        (void)publish_xi(xi, gxi, vector_count);
        return;
    }

    if (*nou == 5) {
        if (!begin_iteration()) {
            if (*nstat == 0)
                fail_cuda();
            return;
        }
        (void)publish_xi(xi, gxi, vector_count);
        return;
    }

    *nstat = -1;
    *steperr = 16;
}
} // namespace

//-----------------------------------------------------------
extern "C" void pimzqmr_(COMPLEX *xs, COMPLEX *xi, COMPLEX *xr, const COMPLEX *b, COMPLEX *work,
                         CU_REAL *norm, const int *lda, const int *ndim, const int *nlar,
                         CUCOMPLEX *lambda, CUCOMPLEX *kappa, CUCOMPLEX *theta, CUCOMPLEX *gamma,
                         CUCOMPLEX *ksi, CUCOMPLEX *rho, CUCOMPLEX *epsilon, CUCOMPLEX *mu,
                         CUCOMPLEX *tau, CUCOMPLEX *dots1, CUCOMPLEX *dots2, CUCOMPLEX *dots3,
                         CUCOMPLEX *dots4, int *nou, int *nt, int *itno, const int *maxit,
                         CU_REAL *tole, const CU_REAL *tol, int *nstat, int *steperr) {
    pimzqmr_impl(xs, xi, xr, b, work, norm, lda, ndim, nlar, lambda, kappa, theta, gamma, ksi, rho,
                 epsilon, mu, tau, dots1, dots2, dots3, dots4, nou, nt, itno, maxit, tole, tol,
                 nstat, steperr);
}
