#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cmath>
#include <cstdio>

namespace {
using cuda_utils::check_cuda;
using cuda_utils::device_value;
using cuda_utils::dotu;
using cuda_utils::get_cublas_handle;
using cuda_utils::host_value;
using cuda_utils::norm2;
using cuda_utils::scalar_is_zero;
constexpr const char *solver_name = "CORS CUDA";

//-----------------------------------------------------------
bool copy_xi(COMPLEX *xi, COMPLEX *work, CUCOMPLEX *gx, CUCOMPLEX *gw, size_t n, size_t nw,
             int *ns, int *se) {
    if (copy_gpbicg_results_to_cpu(xi, work, gx, gw, n, nw, true, false))
        return true;
    *ns = -1;
    *se = 15;
    return false;
}

//-----------------------------------------------------------
bool dot(cublasHandle_t h, int n, const CUCOMPLEX *x, const CUCOMPLEX *y, CUCOMPLEX *q) {
    return dotu(h, n, x, y, q, solver_name);
}
} // namespace

//-----------------------------------------------------------
extern "C" void cors_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda, const int *ndim,
                      const int *nlar, int *nou, COMPLEX *work, int *itno, const int *maxit,
                      const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *rho,
                      CUCOMPLEX *rhoa, CUCOMPLEX *beta, int *nstat, int *steperr) {
    if (!xi || !xr || !b || !lda || !ndim || !nlar || !nou || !work || !itno || !maxit || !tol ||
        !norm || !alpha || !rho || !rhoa || !beta || !nstat || !steperr || *ndim <= 0 ||
        *lda < *ndim || *nlar < 10) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 12;
        return;
    }
    const int n = *ndim;
    const size_t count = size_t(n), nw = size_t(*lda) * size_t(*nlar);
    const bool begin_solve = (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4);
    CUCOMPLEX *gx = nullptr, *gr = nullptr, *gb = nullptr, *gw = nullptr;
    if (!get_gpbicg_device_arrays_resident(xi, xr, b, work, count, nw, begin_solve, &gx, &gr, &gb,
                                           &gw)) {
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
    CUCOMPLEX q = CUFFT_ZERO;
    bool changed = false;
    const auto fail = [&](int e) {
        *nstat = -1;
        *steperr = e;
        copy_xi(xi, work, gx, gw, count, nw, nstat, steperr);
    };

    if (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4) {
        *nstat = 0;
        *steperr = -1;
        cudaError_t st = cors_save_initial_wrapper(gx, gw, *lda, n);
        if (!check_cuda(st, solver_name, "save initial solution") ||
            !norm2(h, n, gb, norm, solver_name)) {
            fail(16);
            return;
        }
        if (!std::isfinite(*norm) || *norm <= CU_REAL(0)) {
            fail(11);
            return;
        }
        *nou = 1;
        copy_xi(xi, work, gx, gw, count, nw, nstat, steperr);
        return;
    }
    if (*nou == 1) {
        cudaError_t st = cors_initialize_residual_wrapper(gx, gr, gb, gw, *lda, n);
        if (!check_cuda(st, solver_name, "initialize residual") ||
            !norm2(h, n, gw + size_t(1) * *lda, &q.x, solver_name)) {
            fail(16);
            return;
        }
        std::printf("Initial Residue for iterative method %.16g %.16g\n", double(q.x / *norm),
                    double(*norm));
        *itno = -1;
        ++(*itno);
        *nou = 2;
        changed = true;
        copy_xi(xi, work, gx, gw, count, nw, nstat, steperr);
        return;
    }
    if (*nou == 2) {
        cudaError_t st;
        if (*itno == 0) {
            st = cors_initialize_direction_wrapper(gx, gr, gw, *lda, n);
            if (!check_cuda(st, solver_name, "initialize shadow residual") ||
                !dot(h, n, gw, gw + size_t(2) * *lda, rho)) {
                fail(16);
                return;
            }
            if (scalar_is_zero(*rho)) {
                *nstat = -1;
                *steperr = 1;
                return;
            }
            *beta = CUFFT_ZERO;
        } else {
            if (!dot(h, n, gw, gr, rho)) {
                fail(16);
                return;
            }
            if (scalar_is_zero(*rho) || scalar_is_zero(*rhoa)) {
                *nstat = -1;
                *steperr = 1;
                return;
            }
            *beta = device_value(host_value(*rho) / host_value(*rhoa));
            st = cors_update_direction_wrapper(gx, gr, gw, *beta, *lda, n);
            if (!check_cuda(st, solver_name, "update directions")) {
                fail(16);
                return;
            }
        }
        *nou = 3;
        changed = true;
        copy_xi(xi, work, gx, gw, count, nw, nstat, steperr);
        return;
    }
    if (*nou == 3) {
        if (!dot(h, n, gw, gr, &q)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(q)) {
            *nstat = -1;
            *steperr = 2;
            return;
        }
        *alpha = device_value(host_value(*rho) / host_value(q));
        cudaError_t st = cors_update_stage_wrapper(gw, gr, *alpha, *lda, n);
        if (!check_cuda(st, solver_name, "update solution and residual") ||
            !norm2(h, n, gw + size_t(1) * *lda, &q.x, solver_name)) {
            fail(16);
            return;
        }
        const CU_REAL residue = q.x / *norm;
        if (!std::isfinite(residue)) {
            fail(11);
            return;
        }
        if (*itno % 50 == 0)
            std::printf("CORS residue %.16g iteration %d\n", double(residue), *itno);
        if (residue <= *tol) {
            st = cors_publish_solution_wrapper(gx, gw, *lda, n);
            if (!check_cuda(st, solver_name, "publish solution")) {
                fail(16);
                return;
            }
            *nstat = 1;
            *nou = 4;
            changed = true;
            copy_xi(xi, work, gx, gw, count, nw, nstat, steperr);
            return;
        }
        *rhoa = *rho;
        if (*itno <= *maxit) {
            ++(*itno);
            st = cors_publish_residual_wrapper(gx, gw, *lda, n);
            if (!check_cuda(st, solver_name, "publish residual")) {
                fail(16);
                return;
            }
            *nou = 2;
            changed = true;
            copy_xi(xi, work, gx, gw, count, nw, nstat, steperr);
            return;
        }
        *nstat = 1;
    }
}
