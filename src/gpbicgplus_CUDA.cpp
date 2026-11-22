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

constexpr const char *solver_name = "GPBICGplus CUDA";

//-----------------------------------------------------------
bool copy_result(COMPLEX *xi, COMPLEX *work, CUCOMPLEX *gpu_xi, CUCOMPLEX *gpu_work, size_t n,
                 size_t work_count, bool xi_modified, int *nstat, int *steperr) {
    if (copy_gpbicg_results_to_cpu(xi, work, gpu_xi, gpu_work, n, work_count, xi_modified, false))
        return true;
    *nstat = -1;
    *steperr = 15;
    return false;
}

//-----------------------------------------------------------
bool dot(cublasHandle_t handle, int n, const CUCOMPLEX *x, const CUCOMPLEX *y, CUCOMPLEX *result) {
    return dotc(handle, n, x, y, result, solver_name);
}

} // namespace

// Fortran-compatible reverse-communication GPBICGplus. Per-element updates
// are CUDA kernels; vector reductions are cuBLAS operations.
//-----------------------------------------------------------
extern "C" void gpbicgplus_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                            const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                            const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                            CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn,
                            int *nstat, int *steperr) {
    if (!xi || !xr || !b || !lda || !ndim || !nlar || !nou || !work || !itno || !maxit || !tol ||
        !norm || !alpha || !beta || !eta || !dzeta || !r0rn || !nstat || !steperr || *lda <= 0 ||
        *ndim <= 0 || *nlar < 12 || *lda < *ndim) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 12;
        std::fprintf(stderr, "%s: invalid Fortran arguments\n", solver_name);
        return;
    }

    const int n = *ndim;
    const size_t vector_count = static_cast<size_t>(n);
    const size_t work_count = static_cast<size_t>(*lda) * static_cast<size_t>(*nlar);
    const bool begin_solve = (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4);
    CUCOMPLEX *gpu_xi = nullptr, *gpu_xr = nullptr;
    CUCOMPLEX *gpu_b = nullptr, *gpu_work = nullptr;
    if (!get_gpbicg_device_arrays_resident(xi, xr, b, work, vector_count, work_count, begin_solve,
                                           &gpu_xi, &gpu_xr, &gpu_b, &gpu_work)) {
        *nstat = -1;
        *steperr = 13;
        std::fprintf(stderr, "%s: unable to prepare GPU arrays\n", solver_name);
        return;
    }

    cublasHandle_t handle = get_cublas_handle();
    if (!handle) {
        *nstat = -1;
        *steperr = 14;
        return;
    }

    CUCOMPLEX ctmp = CUFFT_ZERO, ctmp1 = CUFFT_ZERO, ctmp2 = CUFFT_ZERO;
    CUCOMPLEX ctmp3 = CUFFT_ZERO, ctmp4 = CUFFT_ZERO, ctmp5 = CUFFT_ZERO;
    bool xi_modified = false;
    const auto fail = [&](int error_code) {
        *nstat = -1;
        *steperr = error_code;
        copy_result(xi, work, gpu_xi, gpu_work, vector_count, work_count, xi_modified, nstat,
                    steperr);
    };

    // Initial call: save X0 and calculate ||B||.
    if (*nou != 1 && *nou != 2 && *nou != 3 && *nou != 4) {
        *nstat = 0;
        *steperr = -1;
        cudaError_t status = gpbicg_save_initial_wrapper(gpu_xi, gpu_work, *lda, n);
        if (!check_cuda(status, solver_name, "save initial solution") ||
            !norm2(handle, n, gpu_b, norm, solver_name)) {
            fail(16);
            return;
        }
        if (!std::isfinite(*norm) || *norm <= CU_REAL(0)) {
            fail(11);
            return;
        }
        *nou = 1;
        copy_result(xi, work, gpu_xi, gpu_work, vector_count, work_count, false, nstat, steperr);
        return;
    }

    if (*nou == 1) {
        const cudaError_t status = gpbicgplus_initialize_wrapper(gpu_xr, gpu_b, gpu_work, *lda, n);
        if (!check_cuda(status, solver_name, "initialize residual") ||
            !dot(handle, n, gpu_work, gpu_work + size_t(2) * *lda, r0rn)) {
            fail(16);
            return;
        }
        const CU_REAL initial_residue = std::sqrt(std::hypot(r0rn->x, r0rn->y)) / *norm;
        std::printf("GPBICGplus initial residue %.16g\n", static_cast<double>(initial_residue));
        std::fflush(stdout);
        if (!std::isfinite(initial_residue)) {
            fail(11);
            return;
        }

        *beta = CUFFT_ZERO;
        *itno = 0;
        const cudaError_t update =
            gpbicgplus_prepare_direction_wrapper(gpu_xi, gpu_work, *beta, *lda, n);
        if (!check_cuda(update, solver_name, "prepare initial direction")) {
            fail(16);
            return;
        }
        xi_modified = true;
        *nou = 2;
        copy_result(xi, work, gpu_xi, gpu_work, vector_count, work_count, true, nstat, steperr);
        return;
    }

    if (*nou == 2) {
        cudaError_t status = gpbicgplus_update_ap_wrapper(gpu_xr, gpu_xi, gpu_work, *beta, *lda, n);
        if (!check_cuda(status, solver_name, "update Ar and Ap") ||
            !dot(handle, n, gpu_work, gpu_work + size_t(3) * *lda, &ctmp)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(ctmp)) {
            fail(1);
            return;
        }
        *alpha = device_value(host_value(*r0rn) / host_value(ctmp));

        const CUCOMPLEX *ar = gpu_work + size_t(5) * *lda;
        const CUCOMPLEX *r = gpu_work + size_t(2) * *lda;
        if (*itno == 0) {
            if (!dot(handle, n, ar, r, &ctmp) || !dot(handle, n, ar, ar, &ctmp1)) {
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
            const CUCOMPLEX *az = gpu_work + size_t(8) * *lda;
            if (!dot(handle, n, az, az, &ctmp1) || !dot(handle, n, ar, ar, &ctmp2) ||
                !dot(handle, n, ar, r, &ctmp3) || !dot(handle, n, az, r, &ctmp4) ||
                !dot(handle, n, az, ar, &ctmp5)) {
                fail(16);
                return;
            }
            const HostComplex h1 = host_value(ctmp1), h2 = host_value(ctmp2);
            const HostComplex h3 = host_value(ctmp3), h4 = host_value(ctmp4);
            const HostComplex h5 = host_value(ctmp5);
            const HostComplex denominator = h1 * h2 - h5 * std::conj(h5);
            if (denominator == HostComplex(0, 0)) {
                fail(3);
                return;
            }
            *dzeta = device_value((h1 * h3 - h4 * std::conj(h5)) / denominator);
            *eta = device_value((h2 * h4 - h5 * h3) / denominator);
        }

        status = gpbicgplus_update_stage2_wrapper(gpu_xi, gpu_work, *alpha, *beta, *eta, *dzeta,
                                                  *lda, n);
        if (!check_cuda(status, solver_name, "update u, t and z")) {
            fail(16);
            return;
        }
        xi_modified = true;
        *nou = 3;
        copy_result(xi, work, gpu_xi, gpu_work, vector_count, work_count, true, nstat, steperr);
        return;
    }

    if (*nou == 3 || *nou == 4) {
        const cudaError_t status =
            gpbicgplus_update_stage3_wrapper(gpu_work, gpu_xr, *alpha, *eta, *dzeta, *lda, n);
        if (!check_cuda(status, solver_name, "update Az, X and residual") ||
            !norm2(handle, n, gpu_work + size_t(2) * *lda, &ctmp.x, solver_name)) {
            fail(16);
            return;
        }
        const CU_REAL residue = ctmp.x / *norm;
        if (!std::isfinite(residue)) {
            fail(11);
            return;
        }
        if (*itno % 50 == 0)
            std::printf("GPBICGplus residue %.16g iteration %d\n", static_cast<double>(residue),
                        *itno);
        if (residue <= *tol) {
            const cudaError_t publish = gpbicg_publish_solution_wrapper(gpu_xi, gpu_work, *lda, n);
            if (!check_cuda(publish, solver_name, "publish converged solution")) {
                fail(16);
                return;
            }
            xi_modified = true;
            *nstat = 1;
            *nou = 4;
            copy_result(xi, work, gpu_xi, gpu_work, vector_count, work_count, true, nstat, steperr);
            return;
        }

        if (!dot(handle, n, gpu_work, gpu_work + size_t(2) * *lda, &ctmp)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(*r0rn)) {
            fail(4);
            return;
        }
        *beta = device_value(host_value(*alpha) * host_value(ctmp) / host_value(*dzeta) /
                             host_value(*r0rn));
        *r0rn = ctmp;
        if (*itno <= *maxit) {
            ++(*itno);
            const cudaError_t prepare =
                gpbicgplus_prepare_direction_wrapper(gpu_xi, gpu_work, *beta, *lda, n);
            if (!check_cuda(prepare, solver_name, "prepare next direction")) {
                fail(16);
                return;
            }
            xi_modified = true;
            *nou = 2;
        } else {
            const cudaError_t publish = gpbicg_publish_solution_wrapper(gpu_xi, gpu_work, *lda, n);
            if (!check_cuda(publish, solver_name, "publish final solution")) {
                fail(16);
                return;
            }
            xi_modified = true;
            *nstat = 1;
        }
        copy_result(xi, work, gpu_xi, gpu_work, vector_count, work_count, xi_modified, nstat,
                    steperr);
    }
}
