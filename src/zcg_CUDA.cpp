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
using cuda_utils::norm2;
using cuda_utils::scalar_is_zero;
constexpr const char *solver_name = "ZCG CUDA";

//-----------------------------------------------------------
bool copy_xi(COMPLEX *cpu, CUCOMPLEX *gpu, size_t n, int *ns, int *se) {
    if (copy_gpbicg_results_to_cpu(cpu, nullptr, gpu, nullptr, n, 0, true, false))
        return true;
    *ns = -1;
    *se = 15;
    return false;
}

//-----------------------------------------------------------
bool launch(cudaError_t status, const char *what, int *ns) {
    if (check_cuda(status, solver_name, what))
        return true;
    *ns = -1;
    return false;
}

//-----------------------------------------------------------
void zcg_impl(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, CU_REAL *norm, COMPLEX *work,
              CUCOMPLEX *alpha, CUCOMPLEX *rho0, CUCOMPLEX *xxi, const int *lda, const int *ndim,
              const int *nlar, int *nou, int *nstat, int *nloop, const int *maxit, CU_REAL *tole,
              const CU_REAL *tol) {
    if (!xi || !xr || !b || !norm || !work || !alpha || !rho0 || !xxi || !lda || !ndim || !nlar ||
        !nou || !nstat || !nloop || !maxit || !tole || !tol || *ndim <= 0 || *lda < *ndim ||
        *nlar < 5 || *maxit <= 0) {
        if (nstat)
            *nstat = -1;
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
        return;
    }
    cublasHandle_t h = get_cublas_handle();
    if (!h) {
        *nstat = -1;
        return;
    }
    const auto fail = [&](int code) {
        *nstat = -1;
        *nou = 0;
        (void)copy_xi(xi, gxi, count, nstat, &code);
    };
    CUCOMPLEX q = CUFFT_ZERO;

    if (first) {
        *nstat = 0;
        if (!launch(zcg_save_solution_wrapper(gxi, gw, *lda, n), "save initial solution", nstat)) {
            return;
        }
        *nou = 1;
        {
            int err = 0;
            (void)copy_xi(xi, gxi, count, nstat, &err);
        }
        return;
    }
    if (*nou == 1) {
        if (!norm2(h, n, gb, norm, solver_name) || !std::isfinite(*norm) || *norm <= CU_REAL(0)) {
            fail(1);
            return;
        }
        if (!launch(zcg_initialize_residual_wrapper(gxi, gxr, gb, gw, *lda, n),
                    "initialize residual", nstat))
            return;
        if (!dotc(h, n, gw + static_cast<size_t>(1) * *lda, gw + static_cast<size_t>(1) * *lda, &q,
                  solver_name)) {
            fail(1);
            return;
        }
        *rho0 = q;
        *nou = 2;
        {
            int err = 0;
            (void)copy_xi(xi, gxi, count, nstat, &err);
        }
        return;
    }
    if (*nou == 4) {
        *nou = 3;
        if (!launch(zcg_publish_residual_wrapper(gxi, gw, *lda, n), "publish residual", nstat))
            return;
        {
            int err = 0;
            (void)copy_xi(xi, gxi, count, nstat, &err);
        }
        return;
    }

    if (*nou == 2) {
        if (!launch(zcg_save_initial_w_wrapper(gxr, gw, *lda, n), "save initial product", nstat))
            return;
        if (!dotc(h, n, gw + static_cast<size_t>(2) * *lda, gxr, &q, solver_name)) {
            fail(1);
            return;
        }
        *xxi = q;
    } else if (*nou == 3) {
        CUCOMPLEX rho = CUFFT_ZERO, delta = CUFFT_ZERO;
        if (!dotc(h, n, gw + static_cast<size_t>(1) * *lda, gw + static_cast<size_t>(1) * *lda,
                  &rho, solver_name) ||
            !dotc(h, n, gw + static_cast<size_t>(1) * *lda, gxr, &delta, solver_name)) {
            fail(1);
            return;
        }
        if (scalar_is_zero(*rho0)) {
            fail(2);
            return;
        }
        const CUCOMPLEX beta = device_value(host_value(rho) / host_value(*rho0));
        *rho0 = rho;
        *xxi = device_value(host_value(delta) -
                            host_value(beta) * host_value(beta) * host_value(*xxi));
        if (!launch(zcg_update_directions_wrapper(gxr, gw, beta, *lda, n), "update directions",
                    nstat))
            return;
    }

    if (scalar_is_zero(*xxi)) {
        fail(3);
        return;
    }
    ++(*nloop);
    if (*nloop >= *maxit) {
        *nstat = -4;
        *nou = 0;
        if (!launch(zcg_publish_solution_wrapper(gxi, gw, *lda, n), "restore initial solution",
                    nstat))
            return;
        {
            int err = 0;
            (void)copy_xi(xi, gxi, count, nstat, &err);
        }
        return;
    }
    *alpha = device_value(host_value(*rho0) / host_value(*xxi));
    if (!launch(zcg_step_wrapper(gw, *alpha, *lda, n), "CG step", nstat))
        return;
    CU_REAL rnorm = 0;
    if (!norm2(h, n, gw + static_cast<size_t>(1) * *lda, &rnorm, solver_name)) {
        fail(1);
        return;
    }
    *tole = rnorm / *norm;
    if (*tole <= *tol) {
        *nstat = 1;
        *nou = 4;
        if (!launch(zcg_publish_solution_wrapper(gxi, gw, *lda, n), "publish solution", nstat))
            return;
        {
            int err = 0;
            (void)copy_xi(xi, gxi, count, nstat, &err);
        }
        return;
    }
    *nstat = 0;
    *nou = 3;
    if (!launch(zcg_publish_residual_wrapper(gxi, gw, *lda, n), "publish residual", nstat))
        return;
    {
        int err = 0;
        (void)copy_xi(xi, gxi, count, nstat, &err);
    }
}
} // namespace

//-----------------------------------------------------------
extern "C" void zcg_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, CU_REAL *norm, COMPLEX *work,
                     CUCOMPLEX *alpha, CUCOMPLEX *rho0, CUCOMPLEX *xxi, const int *lda,
                     const int *ndim, const int *nlar, int *nou, int *nstat, int *nloop,
                     const int *maxit, CU_REAL *tole, const CU_REAL *tol) {
    zcg_impl(xi, xr, b, norm, work, alpha, rho0, xxi, lda, ndim, nlar, nou, nstat, nloop, maxit,
             tole, tol);
}
