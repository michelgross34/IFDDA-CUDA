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
constexpr const char *solver_name = "GPBICGAR2 CUDA";

//-----------------------------------------------------------
bool done(COMPLEX *xi, COMPLEX *work, CUCOMPLEX *gx, CUCOMPLEX *gw, size_t n, size_t nw,
          bool changed, int *ns, int *se) {
    if (copy_gpbicg_results_to_cpu(xi, work, gx, gw, n, nw, changed, false))
        return true;
    *ns = -1;
    *se = 15;
    return false;
}

//-----------------------------------------------------------
bool dot(cublasHandle_t h, int n, const CUCOMPLEX *x, const CUCOMPLEX *y, CUCOMPLEX *q) {
    return dotc(h, n, x, y, q, solver_name);
}
} // namespace

//-----------------------------------------------------------
extern "C" void gpbicgar2_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                           const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                           const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                           CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn,
                           int *nstat, int *steperr) {
    if (!xi || !xr || !b || !lda || !ndim || !nlar || !nou || !work || !itno || !maxit || !tol ||
        !norm || !alpha || !beta || !eta || !dzeta || !r0rn || !nstat || !steperr || *ndim <= 0 ||
        *lda < *ndim || *nlar < 12) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 12;
        return;
    }
    const int n = *ndim;
    const size_t count = size_t(n), nw = size_t(*lda) * size_t(*nlar);
    const bool begin_solve =
        (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4 && *nou != 5 && *nou != 6);
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
    CUCOMPLEX q = CUFFT_ZERO, q1 = CUFFT_ZERO, q2 = CUFFT_ZERO, q3 = CUFFT_ZERO, q4 = CUFFT_ZERO,
              q5 = CUFFT_ZERO;
    bool changed = false;
    const auto fail = [&](int e) {
        *nstat = -1;
        *steperr = e;
        done(xi, work, gx, gw, count, nw, changed, nstat, steperr);
    };

    if (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4 && *nou != 5 && *nou != 6) {
        *nstat = 0;
        *steperr = -1;
        cudaError_t st = gpbicg_save_initial_wrapper(gx, gw, *lda, n);
        if (!check_cuda(st, solver_name, "save x0") || !norm2(h, n, gb, norm, solver_name)) {
            fail(16);
            return;
        }
        if (!std::isfinite(*norm) || *norm <= CU_REAL(0)) {
            fail(11);
            return;
        }
        *nou = 1;
        done(xi, work, gx, gw, count, nw, false, nstat, steperr);
        return;
    }
    if (*nou == 1) {
        cudaError_t st = gpbicgar_initialize_wrapper(gr, gb, gx, gw, *lda, n);
        if (!check_cuda(st, solver_name, "initialize residual") ||
            !dot(h, n, gw, gw + size_t(2) * *lda, r0rn)) {
            fail(16);
            return;
        }
        std::printf("Initial Residue for iterative method %.16g\n",
                    double(std::sqrt(std::hypot(r0rn->x, r0rn->y)) / *norm));
        changed = true;
        *nou = 2;
        done(xi, work, gx, gw, count, nw, true, nstat, steperr);
        return;
    }
    if (*nou == 2) {
        *beta = CUFFT_ZERO;
        *itno = -1;
        cudaError_t st = gpbicgar_update_ap_wrapper(gr, gw, *beta, *lda, n);
        if (!check_cuda(st, solver_name, "update initial p and Ap") ||
            !dot(h, n, gw, gw + size_t(5) * *lda, &q)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(q)) {
            fail(1);
            return;
        }
        *alpha = device_value(host_value(*r0rn) / host_value(q));
        if (!dot(h, n, gw + size_t(6) * *lda, gw + size_t(2) * *lda, &q) ||
            !dot(h, n, gw + size_t(6) * *lda, gw + size_t(6) * *lda, &q1)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(q1)) {
            fail(2);
            return;
        }
        *dzeta = device_value(host_value(q) / host_value(q1));
        *eta = CUFFT_ZERO;
        *itno = 0;
        st = gpbicgar_update_u_wrapper(gx, gw, *beta, *eta, *dzeta, *lda, n);
        if (!check_cuda(st, solver_name, "update first u")) {
            fail(16);
            return;
        }
        changed = true;
        *nou = 3;
        done(xi, work, gx, gw, count, nw, true, nstat, steperr);
        return;
    }
    if (*nou == 3 || *nou == 4) {
        cudaError_t st = gpbicgar_update_stage_wrapper(gw, gr, *alpha, *eta, *dzeta, *lda, n);
        if (!check_cuda(st, solver_name, "update stage") ||
            !norm2(h, n, gw + size_t(2) * *lda, &q.x, solver_name)) {
            fail(16);
            return;
        }
        const CU_REAL residue = q.x / *norm;
        if (!std::isfinite(residue)) {
            fail(11);
            return;
        }
        if (*itno % 50 == 0)
            std::printf("GPBICGAR2 residue %.16g iteration %d\n", double(residue), *itno);
        if (residue <= *tol) {
            st = gpbicg_publish_solution_wrapper(gx, gw, *lda, n);
            if (!check_cuda(st, solver_name, "publish solution")) {
                fail(16);
                return;
            }
            *nstat = 1;
            *nou = 6;
            changed = true;
            done(xi, work, gx, gw, count, nw, true, nstat, steperr);
            return;
        }
        st = gpbicgar_publish_residual_wrapper(gx, gw, *lda, n);
        if (!check_cuda(st, solver_name, "publish residual")) {
            fail(16);
            return;
        }
        *nou = 5;
        changed = true;
        done(xi, work, gx, gw, count, nw, true, nstat, steperr);
        return;
    }
    if (*nou == 5) {
        cudaError_t st = gpbicgar_copy_ar_wrapper(gr, gw, *lda, n);
        if (!check_cuda(st, solver_name, "store Ar") || !dot(h, n, gw, gw + size_t(2) * *lda, &q)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(*r0rn)) {
            fail(5);
            return;
        }
        if (scalar_is_zero(*dzeta)) {
            fail(6);
            return;
        }
        *beta = device_value(host_value(*alpha) * host_value(q) / host_value(*dzeta) /
                             host_value(*r0rn));
        *r0rn = q;
        if (*itno > *maxit) {
            st = gpbicg_publish_solution_wrapper(gx, gw, *lda, n);
            if (!check_cuda(st, solver_name, "publish final solution")) {
                fail(16);
                return;
            }
            *nstat = 1;
            changed = true;
            done(xi, work, gx, gw, count, nw, true, nstat, steperr);
            return;
        }
        ++(*itno);
        st = gpbicgar_update_ap_wrapper(gr, gw, *beta, *lda, n);
        if (!check_cuda(st, solver_name, "update directions") ||
            !dot(h, n, gw, gw + size_t(5) * *lda, &q)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(q)) {
            fail(1);
            return;
        }
        *alpha = device_value(host_value(*r0rn) / host_value(q));
        const CUCOMPLEX *ar = gw + size_t(6) * *lda;
        const CUCOMPLEX *r = gw + size_t(2) * *lda;
        if ((*itno % 2) == 0) {
            if (!dot(h, n, ar, r, &q) || !dot(h, n, ar, ar, &q1)) {
                fail(16);
                return;
            }
            if (scalar_is_zero(q1)) {
                fail(2);
                return;
            }
            *dzeta = device_value(host_value(q) / host_value(q1));
            *eta = CUFFT_ZERO;
            *nou = 3;
        } else {
            const CUCOMPLEX *z = gw + size_t(8) * *lda;
            if (!dot(h, n, z, z, &q1) || !dot(h, n, ar, r, &q2) || !dot(h, n, z, r, &q3) ||
                !dot(h, n, ar, z, &q4) || !dot(h, n, ar, ar, &q5)) {
                fail(16);
                return;
            }
            const HostComplex a = host_value(q1), b2 = host_value(q2), c = host_value(q3),
                              d = host_value(q4), e = host_value(q5);
            const HostComplex den = e * a - d * std::conj(d);
            if (den == HostComplex(0, 0)) {
                fail(3);
                return;
            }
            *dzeta = device_value((a * b2 - c * d) / den);
            *eta = device_value((e * c - std::conj(d) * b2) / den);
            *nou = 4;
        }
        st = gpbicgar_update_u_wrapper(gx, gw, *beta, *eta, *dzeta, *lda, n);
        if (!check_cuda(st, solver_name, "update u")) {
            fail(16);
            return;
        }
        changed = true;
        done(xi, work, gx, gw, count, nw, true, nstat, steperr);
    }
}
