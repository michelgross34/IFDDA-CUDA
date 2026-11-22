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
constexpr const char *solver_name = "GPBICOR CUDA";

//-----------------------------------------------------------
bool copy_out(COMPLEX *xi, COMPLEX *f, CUCOMPLEX *gx, CUCOMPLEX *gf, size_t n, bool field,
              int *ns, int *se) {
    if (copy_gpbicor_results_to_cpu(xi, f, gx, gf, n, field))
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
extern "C" void gpbicor_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, COMPLEX *ffloc,
                         const int *lda, const int *ndim, const int *nlar, int *nou,
                         COMPLEX *work, int *itno, const int *maxit, const CU_REAL *tol,
                         CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *beta, CUCOMPLEX *eta,
                         CUCOMPLEX *dzeta, CUCOMPLEX *rho, CUCOMPLEX *rhoa, int *nstat,
                         int *steperr) {
    if (!xi || !xr || !b || !ffloc || !lda || !ndim || !nlar || !nou || !work || !itno || !maxit ||
        !tol || !norm || !alpha || !beta || !eta || !dzeta || !rho || !rhoa || !nstat || !steperr ||
        *ndim <= 0 || *lda < *ndim || *nlar < 12) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 12;
        return;
    }
    const int n = *ndim;
    const size_t count = size_t(n), nw = size_t(*lda) * size_t(*nlar);
    const bool first = (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4);
    CUCOMPLEX *gx = nullptr, *gr = nullptr, *gb = nullptr, *gf = nullptr, *gw = nullptr;
    if (!get_gpbicor_device_arrays(xi, xr, b, ffloc, work, count, nw, first, &gx, &gr, &gb, &gf,
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
    const auto fail = [&](int e) {
        *nstat = -1;
        *steperr = e;
        copy_out(xi, ffloc, gx, gf, count, false, nstat, steperr);
    };

    if (first) {
        *nstat = 0;
        *steperr = -1;
        if (!norm2(h, n, gb, norm, solver_name)) {
            fail(16);
            return;
        }
        if (!std::isfinite(*norm) || *norm <= CU_REAL(0)) {
            fail(11);
            return;
        }
        *nou = 1;
        copy_out(xi, ffloc, gx, gf, count, false, nstat, steperr);
        return;
    }
    if (*nou == 1) {
        cudaError_t st = gpbicor_initialize_wrapper(gx, gr, gb, gw, *lda, n);
        if (!check_cuda(st, solver_name, "initialize residual") ||
            !norm2(h, n, gw, &q.x, solver_name)) {
            fail(16);
            return;
        }
        std::printf("Initial Residue for iterative method %.16g\n", double(q.x / *norm));
        *itno = 0;
        ++(*itno);
        st = gpbicor_publish_residual_wrapper(gx, gw, *lda, n);
        if (!check_cuda(st, solver_name, "publish initial residual")) {
            fail(16);
            return;
        }
        *nou = 2;
        copy_out(xi, ffloc, gx, gf, count, false, nstat, steperr);
        return;
    }
    if (*nou == 2) {
        if (!dot(h, n, gw, gr, rho)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(*rho)) {
            *nstat = -1;
            *steperr = 1;
            return;
        }
        cudaError_t st;
        if (*itno == 1) {
            *beta = CUFFT_ZERO;
            st = gpbicor_first_direction_wrapper(gx, gw, gr, *lda, n);
        } else {
            if (scalar_is_zero(*rhoa) || scalar_is_zero(*dzeta)) {
                *nstat = -1;
                *steperr = 1;
                return;
            }
            *beta = device_value(host_value(*rho) / host_value(*rhoa) * host_value(*alpha) /
                                 host_value(*dzeta));
            st = gpbicor_update_directions_wrapper(gx, gr, gw, *beta, *lda, n);
        }
        if (!check_cuda(st, solver_name, "update BiCG directions")) {
            fail(16);
            return;
        }
        *nou = 3;
        copy_out(xi, ffloc, gx, gf, count, false, nstat, steperr);
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
        cudaError_t st = gpbicor_prepare_stage_wrapper(gx, gr, gw, *alpha, *lda, n);
        if (!check_cuda(st, solver_name, "prepare MR stage")) {
            fail(16);
            return;
        }
        const CUCOMPLEX *r0 = gw;
        const CUCOMPLEX *t = gw + size_t(6) * *lda;
        if (*itno == 1) {
            if (!dot(h, n, gr, t, &q) || !dot(h, n, gr, gr, &q1)) {
                fail(16);
                return;
            }
            if (scalar_is_zero(q1)) {
                *nstat = -1;
                *steperr = 3;
                return;
            }
            *eta = CUFFT_ZERO;
            *dzeta = device_value(host_value(q) / host_value(q1));
        } else {
            const CUCOMPLEX *y = gx;
            const CUCOMPLEX *v = gw + size_t(9) * *lda;
            if (!dot(h, n, gr, gr, &q1) || !dot(h, n, y, t, &q2) || !dot(h, n, y, gr, &q3) ||
                !dot(h, n, gr, t, &q4) || !dot(h, n, y, y, &q5)) {
                fail(16);
                return;
            }
            const HostComplex a = host_value(q1), b2 = host_value(q2), c = host_value(q3),
                              d = host_value(q4), e = host_value(q5);
            const HostComplex den = a * e - c * std::conj(c);
            if (den == HostComplex(0, 0)) {
                *nstat = -1;
                *steperr = 4;
                return;
            }
            *eta = device_value((a * b2 - c * d) / den);
            *dzeta = device_value((e * d - b2 * std::conj(c)) / den);
            (void)v;
            (void)r0;
        }
        st = gpbicor_finish_stage_wrapper(gf, gw, gx, gr, *alpha, *beta, *eta, *dzeta, *lda, n);
        if (!check_cuda(st, solver_name, "finish MR stage") ||
            !norm2(h, n, gw + size_t(9) * *lda, &q.x, solver_name)) {
            fail(16);
            return;
        }
        const CU_REAL residue = q.x / *norm;
        if (!std::isfinite(residue)) {
            fail(11);
            return;
        }
        if (*itno % 50 == 0)
            std::printf("GPBICOR residue %.16g iteration %d\n", double(residue), *itno);
        if (residue <= *tol) {
            st = gpbicor_publish_solution_wrapper(gx, gf, n);
            if (!check_cuda(st, solver_name, "publish solution")) {
                fail(16);
                return;
            }
            *nstat = 1;
            *nou = 4;
            copy_out(xi, ffloc, gx, gf, count, true, nstat, steperr);
            return;
        }
        *rhoa = *rho;
        if (*itno <= *maxit) {
            ++(*itno);
            st = gpbicor_publish_residual_wrapper(gx, gw, *lda, n);
            if (!check_cuda(st, solver_name, "publish residual")) {
                fail(16);
                return;
            }
            *nou = 2;
            copy_out(xi, ffloc, gx, gf, count, false, nstat, steperr);
            return;
        }
        *nstat = 1;
        copy_out(xi, ffloc, gx, gf, count, true, nstat, steperr);
    }
}
