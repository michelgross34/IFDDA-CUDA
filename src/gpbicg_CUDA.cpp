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

//-----------------------------------------------------------
bool inner_product(cublasHandle_t handle, int n, const CUCOMPLEX *x, const CUCOMPLEX *y,
                   CUCOMPLEX *result, bool conjugate_first) {
    return conjugate_first ? dotc(handle, n, x, y, result, "GPBICG CUDA")
                           : dotu(handle, n, x, y, result, "GPBICG CUDA");
}

//-----------------------------------------------------------
bool finish_call(COMPLEX *cpu_xi, COMPLEX *cpu_work, CUCOMPLEX *gpu_xi, CUCOMPLEX *gpu_work,
                 size_t n, size_t work_count, bool xi_modified, int *nstat, int *steperr) {
    if (!copy_gpbicg_results_to_cpu(cpu_xi, cpu_work, gpu_xi, gpu_work, n, work_count, xi_modified,
                                    false)) {
        *nstat = -1;
        *steperr = 15;
        return false;
    }
    return true;
}

} // namespace

// Fortran-compatible reverse-communication GPBICG implementation.
// Vector updates are parallel kernels in kernel.cu; inner products and norms
// are cuBLAS reductions. The original cdmlib/gpbicg.f90 remains untouched.
//-----------------------------------------------------------
static void run_gpbicg(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda, const int *ndim,
                       const int *nlar, int *nou, COMPLEX *work, int *itno, const int *maxit,
                       const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *beta,
                       CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn, int *nstat, int *steperr,
                       bool gpbicg2) {
    if (!xi || !xr || !b || !lda || !ndim || !nlar || !nou || !work || !itno || !maxit || !tol ||
        !norm || !alpha || !beta || !eta || !dzeta || !r0rn || !nstat || !steperr || *lda <= 0 ||
        *ndim <= 0 || *nlar < 12 || *lda < *ndim) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 12;
        std::fprintf(stderr, "gpbicg CUDA: invalid Fortran arguments\n");
        return;
    }

    const int n = *ndim;
    const bool begin_solve = (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4);
    const size_t vector_count = static_cast<size_t>(n);
    const size_t work_count = static_cast<size_t>(*lda) * static_cast<size_t>(*nlar);
    CUCOMPLEX *gpu_xi = nullptr, *gpu_xr = nullptr;
    CUCOMPLEX *gpu_b = nullptr, *gpu_work = nullptr;
    if (!get_gpbicg_device_arrays_resident(xi, xr, b, work, vector_count, work_count, begin_solve,
                                           &gpu_xi, &gpu_xr, &gpu_b, &gpu_work)) {
        *nstat = -1;
        *steperr = 13;
        std::fprintf(stderr, "gpbicg CUDA: unable to prepare GPU arrays\n");
        return;
    }

    cublasHandle_t handle = get_cublas_handle();
    if (!handle) {
        *nstat = -1;
        *steperr = 14;
        return;
    }
    bool xi_modified = false;
    const auto fail = [&](int error_code) {
        *nstat = -1;
        *steperr = error_code;
        finish_call(xi, work, gpu_xi, gpu_work, vector_count, work_count, xi_modified, nstat,
                    steperr);
    };
    CUCOMPLEX ctmp = CUFFT_ZERO, ctmp1 = CUFFT_ZERO, ctmp2 = CUFFT_ZERO;
    CUCOMPLEX ctmp3 = CUFFT_ZERO, ctmp4 = CUFFT_ZERO, ctmp5 = CUFFT_ZERO;
    cudaError_t kernel_status = cudaSuccess;

    // Initial call: retain Xi in W(:,11), compute ||B|| with cuBLAS.
    if (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4) {
        *nstat = 0;
        *steperr = -1;
        kernel_status = gpbicg_save_initial_wrapper(gpu_xi, gpu_work, *lda, n);
        if (!check_cuda(kernel_status, "gpbicg CUDA", "save initial solution") ||
            !norm2(handle, n, gpu_b, norm, "gpbicg CUDA")) {
            fail(16);
            return;
        }
        if (!std::isfinite(*norm) || *norm <= CU_REAL(0)) {
            fail(11);
            return;
        }
        *nou = 1;
        finish_call(xi, work, gpu_xi, gpu_work, vector_count, work_count, false, nstat, steperr);
        return;
    }

    if (*nou == 1) {
        kernel_status = gpbicg2 ? gpbicg2_initialize_wrapper(gpu_xr, gpu_b, gpu_work, *lda, n)
                                : gpbicg_initialize_wrapper(gpu_xr, gpu_b, gpu_work, *lda, n);
        if (!check_cuda(kernel_status, "gpbicg CUDA", "initialize residual") ||
            !inner_product(handle, n, gpu_work, gpu_work + size_t(2) * *lda, r0rn, gpbicg2)) {
            fail(16);
            return;
        }
        const CU_REAL residue = std::sqrt(std::hypot(r0rn->x, r0rn->y)) / *norm;
        std::printf("Initial Residue for iterative method %.16g Norm %.16g\n",
                    static_cast<double>(residue), static_cast<double>(*norm));
        std::fflush(stdout);
        if (!std::isfinite(residue)) {
            fail(11);
            return;
        }
        if (residue <= *tol) {
            *nstat = 1;
            kernel_status = gpbicg_publish_solution_wrapper(gpu_xi, gpu_work, *lda, n);
            xi_modified = kernel_status == cudaSuccess;
            if (!check_cuda(kernel_status, "gpbicg CUDA", "publish converged solution")) {
                fail(16);
                return;
            }
            finish_call(xi, work, gpu_xi, gpu_work, vector_count, work_count, xi_modified, nstat,
                        steperr);
            return;
        }
        *beta = CUFFT_ZERO;
        *itno = -1;
        ++(*itno);
        kernel_status = gpbicg_prepare_direction_wrapper(gpu_xi, gpu_work, *beta, *lda, n);
        if (!check_cuda(kernel_status, "gpbicg CUDA", "prepare initial direction")) {
            fail(16);
            return;
        }
        xi_modified = true;
        *nou = 2;
        finish_call(xi, work, gpu_xi, gpu_work, vector_count, work_count, xi_modified, nstat,
                    steperr);
        return;
    } else if (*nou == 2) {
        kernel_status = gpbicg_save_xr_column_wrapper(gpu_xr, gpu_work, *lda, n, 5);
        if (!check_cuda(kernel_status, "gpbicg CUDA", "save A*p") ||
            !inner_product(handle, n, gpu_work, gpu_work + size_t(5) * *lda, &ctmp, gpbicg2)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(ctmp)) {
            fail(1);
            return;
        }
        *alpha = device_value(host_value(*r0rn) / host_value(ctmp));
        kernel_status = gpbicg_update_stage2_wrapper(gpu_xi, gpu_work, *alpha, *lda, n);
        if (!check_cuda(kernel_status, "gpbicg CUDA", "update stage 2")) {
            fail(16);
            return;
        }
        xi_modified = true;
        *nou = 3;
        finish_call(xi, work, gpu_xi, gpu_work, vector_count, work_count, xi_modified, nstat,
                    steperr);
        return;
    } else if (*nou == 3) {
        kernel_status = gpbicg_save_xr_column_wrapper(gpu_xr, gpu_work, *lda, n, 6);
        if (!check_cuda(kernel_status, "gpbicg CUDA", "save A*t")) {
            fail(16);
            return;
        }
        const CUCOMPLEX *w7 = gpu_work + size_t(6) * *lda;
        const CUCOMPLEX *w4 = gpu_work + size_t(3) * *lda;
        const CUCOMPLEX *w5 = gpu_work + size_t(4) * *lda;
        if ((!gpbicg2 && *itno == 0) || (gpbicg2 && *itno % 2 == 0)) {
            if (!dotc(handle, n, w7, w5, &ctmp, "gpbicg CUDA") ||
                !dotc(handle, n, w7, w7, &ctmp1, "gpbicg CUDA")) {
                fail(16);
                return;
            }
            if (scalar_is_zero(ctmp1)) {
                fail(2);
                return;
            }
            *dzeta = device_value(host_value(ctmp) / host_value(ctmp1));
            *eta = CUFFT_ZERO;
        } else {
            if (!dotc(handle, n, w7, w7, &ctmp1, "gpbicg CUDA") ||
                !dotc(handle, n, w4, w4, &ctmp2, "gpbicg CUDA") ||
                !dotc(handle, n, w7, w4, &ctmp3, "gpbicg CUDA") ||
                !dotc(handle, n, w7, w5, &ctmp4, "gpbicg CUDA") ||
                !dotc(handle, n, w4, w5, &ctmp5, "gpbicg CUDA")) {
                fail(16);
                return;
            }
            const HostComplex h1 = host_value(ctmp1), h2 = host_value(ctmp2);
            const HostComplex h3 = host_value(ctmp3), h4 = host_value(ctmp4);
            const HostComplex h5 = host_value(ctmp5);
            const HostComplex denominator = h1 * h2 - h3 * std::conj(h3);
            if (denominator == HostComplex(0, 0)) {
                fail(3);
                return;
            }
            *dzeta = device_value((h2 * h4 - h5 * h3) / denominator);
            *eta = device_value((h1 * h5 - std::conj(h3) * h4) / denominator);
        }
        kernel_status =
            gpbicg_update_stage3_wrapper(gpu_work, *alpha, *beta, *eta, *dzeta, *lda, n);
        if (!check_cuda(kernel_status, "gpbicg CUDA", "update stage 3") ||
            !norm2(handle, n, gpu_work + size_t(2) * *lda, &ctmp.x, "gpbicg CUDA")) {
            fail(16);
            return;
        }
        const CU_REAL residue = ctmp.x / *norm;
        std::printf("RESIDUE %.16g iteration %d tolerance %.16g\n", static_cast<double>(residue),
                    *itno, static_cast<double>(*tol));
        std::fflush(stdout);
        if (!std::isfinite(residue)) {
            fail(11);
            return;
        }
        if (residue <= *tol) {
            *nstat = 1;
            *nou = 4;
            kernel_status = gpbicg_publish_solution_wrapper(gpu_xi, gpu_work, *lda, n);
            xi_modified = kernel_status == cudaSuccess;
            if (!check_cuda(kernel_status, "gpbicg CUDA", "publish converged solution")) {
                fail(16);
                return;
            }
            finish_call(xi, work, gpu_xi, gpu_work, vector_count, work_count, xi_modified, nstat,
                        steperr);
            return;
        }
        if (residue >= CU_REAL(100)) {
            *nstat = 1;
            *steperr = 0;
            finish_call(xi, work, gpu_xi, gpu_work, vector_count, work_count, false, nstat,
                        steperr);
            return;
        }
    }

    // Fortran label 40: beta, W(:,9), then next p or final solution.
    if (!inner_product(handle, n, gpu_work, gpu_work + size_t(2) * *lda, &ctmp, gpbicg2)) {
        fail(16);
        return;
    }
    if (scalar_is_zero(*r0rn)) {
        fail(4);
        return;
    }
    const HostComplex hbeta =
        host_value(*alpha) * host_value(ctmp) / host_value(*dzeta) / host_value(*r0rn);
    *beta = device_value(hbeta);
    *r0rn = ctmp;
    kernel_status = gpbicg_update_stage4_wrapper(gpu_work, *beta, *lda, n);
    if (!check_cuda(kernel_status, "gpbicg CUDA", "update stage 4")) {
        fail(16);
        return;
    }

    if (*itno <= *maxit) {
        ++(*itno);
        kernel_status = gpbicg_prepare_direction_wrapper(gpu_xi, gpu_work, *beta, *lda, n);
        if (!check_cuda(kernel_status, "gpbicg CUDA", "prepare direction")) {
            fail(16);
            return;
        }
        xi_modified = true;
        *nou = 2;
    } else {
        *nstat = 1;
        *steperr = 0;
        kernel_status = gpbicg_publish_solution_wrapper(gpu_xi, gpu_work, *lda, n);
        xi_modified = kernel_status == cudaSuccess;
        if (!check_cuda(kernel_status, "gpbicg CUDA", "publish final solution")) {
            fail(16);
            return;
        }
    }
    finish_call(xi, work, gpu_xi, gpu_work, vector_count, work_count, xi_modified, nstat, steperr);
}

//-----------------------------------------------------------
extern "C" void gpbicg_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda, const int *ndim,
                        const int *nlar, int *nou, COMPLEX *work, int *itno, const int *maxit,
                        const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *beta,
                        CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn, int *nstat,
                        int *steperr) {
    run_gpbicg(xi, xr, b, lda, ndim, nlar, nou, work, itno, maxit, tol, norm, alpha, beta, eta,
               dzeta, r0rn, nstat, steperr, false);
}

//-----------------------------------------------------------
extern "C" void gpbicg2_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                         const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                         const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                         CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn,
                         int *nstat, int *steperr) {
    run_gpbicg(xi, xr, b, lda, ndim, nlar, nou, work, itno, maxit, tol, norm, alpha, beta, eta,
               dzeta, r0rn, nstat, steperr, true);
}
