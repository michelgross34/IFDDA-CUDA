#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cmath>
#include <cstdio>
#include <vector>



namespace {
using cuda_utils::check_cuda;
using cuda_utils::copy;
using cuda_utils::device_value;
using cuda_utils::dotc;
using cuda_utils::get_cublas_handle;
using cuda_utils::host_value;
using cuda_utils::HostComplex;
using cuda_utils::norm2;
using cuda_utils::scalar_is_zero;

constexpr const char *context = "GCORS2 CUDA";

//-----------------------------------------------------------
bool launch(cudaError_t status, const char *action) {
    return check_cuda(status, context, action);
}

//-----------------------------------------------------------
bool combine(CUCOMPLEX *work, CUCOMPLEX *xsol, int lda, int n, int destination, const int *columns,
             const CUCOMPLEX *coefficients, int terms, bool accumulate = false) {
    return launch(gpbicgstabl_combine_wrapper(work, xsol, lda, n, destination, columns,
                                              coefficients, terms, accumulate ? 1 : 0),
                  "combine work vectors");
}

//-----------------------------------------------------------
bool copy_out(COMPLEX *host_xi, COMPLEX *host_xsol, CUCOMPLEX *gpu_xi, CUCOMPLEX *gpu_xsol,
              size_t n, bool solution, int *nstat, int *steperr) {
    if (copy_gpbicor_results_to_cpu(host_xi, host_xsol, gpu_xi, gpu_xsol, n, solution))
        return true;
    *nstat = -1;
    *steperr = 15;
    return false;
}

//-----------------------------------------------------------
bool save_xr(CUCOMPLEX *work, const CUCOMPLEX *xr, int column, int lda, int n) {
    return launch(idrs_copy_xr_wrapper(work, xr, column, lda, n), "save Ax vector");
}

//-----------------------------------------------------------
bool publish_work(CUCOMPLEX *xi, const CUCOMPLEX *work, int column, int lda, int n) {
    return launch(idrs_copy_work_to_xi_wrapper(xi, work, column, lda, n), "publish work vector");
}

//-----------------------------------------------------------
bool stage_iteration(CUCOMPLEX *xi, CUCOMPLEX *xsol, CUCOMPLEX *work, int lda, int n,
                     CUCOMPLEX *alpha, CUCOMPLEX *alphat, CUCOMPLEX rho, CUCOMPLEX rhot,
                     CU_REAL tolerance, CU_REAL norm, int *nstat, int *steperr, int *nou,
                     cublasHandle_t handle) {
    CUCOMPLEX sigma = CUFFT_ZERO, sigmat = CUFFT_ZERO;
    // Fortran sums work(:,14)*conjg(work(:,4)) and work(:,13)*conjg(work(:,4)).
    if (!dotc(handle, n, work + size_t(3) * lda, work + size_t(13) * lda, &sigma, context) ||
        !dotc(handle, n, work + size_t(3) * lda, work + size_t(12) * lda, &sigmat, context)) {
        *nstat = -1;
        *steperr = 16;
        return false;
    }
    if (scalar_is_zero(sigma)) {
        *nstat = -1;
        *steperr = 1;
        return false;
    }
    if (scalar_is_zero(sigmat)) {
        *nstat = -1;
        *steperr = 2;
        return false;
    }

    *alpha = device_value(host_value(rho) / host_value(sigma));
    *alphat = device_value(host_value(rhot) / host_value(sigmat));
    if (scalar_is_zero(*alpha)) {
        *nstat = -1;
        *steperr = 3;
        return false;
    }
    if (scalar_is_zero(*alphat)) {
        *nstat = -1;
        *steperr = 4;
        return false;
    }

    const CUCOMPLEX one = CUFFT_REAL(CU_REAL(1));
    const int col_s[2] = {4, 2};
    const CUCOMPLEX coef_s[2] = {one, CUFFT_SUB(CUFFT_ZERO, *alpha)};
    const int col_st[2] = {5, 3};
    const CUCOMPLEX coef_st[2] = {one, CUFFT_SUB(CUFFT_ZERO, *alpha)};
    const int col_h[2] = {8, 2};
    const CUCOMPLEX coef_h[2] = {one, CUFFT_SUB(CUFFT_ZERO, *alphat)};
    const int col_ht[2] = {9, 3};
    const CUCOMPLEX coef_ht[2] = {one, CUFFT_SUB(CUFFT_ZERO, *alphat)};
    const int col_r[3] = {0, 9, 7};
    const CUCOMPLEX coef_r[3] = {one, CUFFT_SUB(CUFFT_ZERO, *alpha),
                                 CUFFT_SUB(CUFFT_ZERO, *alphat)};
    const int col_x[2] = {8, 6};
    const CUCOMPLEX coef_x[2] = {*alpha, *alphat};
    if (!combine(work, xsol, lda, n, 6, col_s, coef_s, 2) ||
        !combine(work, xsol, lda, n, 7, col_st, coef_st, 2) ||
        !combine(work, xsol, lda, n, 10, col_h, coef_h, 2) ||
        !combine(work, xsol, lda, n, 11, col_ht, coef_ht, 2) ||
        !combine(work, xsol, lda, n, 0, col_r, coef_r, 3) ||
        !combine(work, xsol, lda, n, -1, col_x, coef_x, 2, true) ||
        !publish_work(xi, work, 0, lda, n)) {
        *nstat = -1;
        *steperr = 16;
        return false;
    }
    CU_REAL residual = 0;
    if (!norm2(handle, n, xi, &residual, context)) {
        *nstat = -1;
        *steperr = 16;
        return false;
    }
    residual /= norm;
    std::printf("GCORS2 residue %.16g\n", double(residual));
    if (residual <= tolerance) {
        if (!launch(gcors2_publish_solution_wrapper(xi, xsol, n), "publish solution")) {
            *nstat = -1;
            *steperr = 16;
            return false;
        }
        *nstat = 1;
        *nou = 6;
    } else
        *nou = 4;
    return true;
}

} // namespace

//-----------------------------------------------------------
extern "C" void gcors2_(COMPLEX *xi, COMPLEX *xr, COMPLEX *xsol_host, COMPLEX *b,
                        const int *ldap, const int *ndimp, const int *nlarp, int *nou,
                        COMPLEX *work_host, int *itno, const int *maxitp,
                        const CU_REAL *tolerancep, CU_REAL *norm, CUCOMPLEX *alpha,
                        CUCOMPLEX *alphat, CUCOMPLEX *rho, CUCOMPLEX *rhot, int *nstat,
                        int *steperr) {
    if (!xi || !xr || !xsol_host || !b || !ldap || !ndimp || !nlarp || !nou || !work_host ||
        !itno || !maxitp || !tolerancep || !norm || !alpha || !alphat || !rho || !rhot || !nstat ||
        !steperr || *ndimp <= 0 || *ldap < *ndimp || *nlarp < 14) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 12;
        return;
    }

    const int n = *ndimp, lda = *ldap, nlar = *nlarp;
    const size_t vector_count = size_t(n), work_count = size_t(lda) * size_t(nlar);
    const bool first = (*nou < 1 || *nou > 6);
    CUCOMPLEX *gxi = nullptr, *gxr = nullptr, *gb = nullptr, *gxsol = nullptr, *gwork = nullptr;
    if (!get_gpbicor_device_arrays(xi, xr, b, xsol_host, work_host, vector_count, work_count, first,
                                   &gxi, &gxr, &gb, &gxsol, &gwork)) {
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
    const auto finish = [&](bool solution = false) {
        return copy_out(xi, xsol_host, gxi, gxsol, vector_count, solution, nstat, steperr);
    };
    const auto fail = [&](int code) {
        *nstat = -1;
        *steperr = code;
        finish(false);
    };

    if (first) {
        *nstat = 0;
        *steperr = -1;
        // Fortran's initial call sets xsol = Xi before returning with nou=1.
        // Keep that state transition on-device; FFloc is not guaranteed to
        // contain the initial iterate (especially for a preconditioned run).
        if (!copy(handle, n, gxi, gxsol, context) || !norm2(handle, n, gb, norm, context) ||
            !std::isfinite(*norm) || *norm <= 0) {
            fail(11);
            return;
        }
        *nou = 1;
        finish(false);
        return;
    }
    if (*nou == 1) {
        if (!launch(gcors2_initialize_residual_wrapper(gxi, gxr, gb, gwork, lda, n),
                    "initialize residual")) {
            fail(16);
            return;
        }
        CU_REAL initial = 0;
        if (!norm2(handle, n, gxi, &initial, context)) {
            fail(16);
            return;
        }
        std::printf("GCORS2 initial residue %.16g norm %.16g\n", double(initial / *norm),
                    double(*norm));
        *nou = 2;
        finish(false);
        return;
    }
    if (*nou == 2) {
        std::vector<CU_REAL> random_imag(static_cast<size_t>(n));
        std::vector<CU_REAL> random_real(static_cast<size_t>(n));
        gcors2_random_values(n, random_imag.data(), random_real.data());
        std::vector<CUCOMPLEX> packed(static_cast<size_t>(n), CUFFT_ZERO);
        for (int i = 0; i < n; ++i)
            packed[size_t(i)] =
                device_value(HostComplex(random_real[size_t(i)], random_imag[size_t(i)]));
        CUCOMPLEX *random_gpu = get_device_vectors(size_t(n));
        // Keep the Fortran-compatible RNG on the CPU; this one-time upload is
        // solver input, not an unclassified auxiliary transfer.
        if (!random_gpu ||
            cuda_data::copy_host_to_device(random_gpu, packed.data(), size_t(n) * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::SolverVector) !=
                cudaSuccess ||
            !launch(gcors2_setup_wrapper(gxi, gxr, gwork, random_gpu, lda, n),
                    "initialize shadow vectors")) {
            fail(16);
            return;
        }
        *itno = -1;
        *nou = 3;
        finish(false);
        return;
    }
    if (*nou == 3) {
        if (!save_xr(gwork, gxr, 3, lda, n)) {
            fail(16);
            return;
        }
        CUCOMPLEX local_rho = CUFFT_ZERO, local_rhot = CUFFT_ZERO;
        if (!dotc(handle, n, gwork + size_t(1) * lda, gwork + size_t(13) * lda, &local_rho,
                  context) ||
            !dotc(handle, n, gwork + size_t(1) * lda, gwork + size_t(12) * lda, &local_rhot,
                  context)) {
            fail(16);
            return;
        }
        *rho = local_rho;
        *rhot = local_rhot;
        *itno = -1;
    }
    if (*nou == 3) {
        ++(*itno);
        if (!stage_iteration(gxi, gxsol, gwork, lda, n, alpha, alphat, *rho, *rhot, *tolerancep,
                             *norm, nstat, steperr, nou, handle)) {
            finish(false);
            return;
        }
        finish(*nstat == 1);
        return;
    }
    if (*nou == 4) {
        if (!save_xr(gwork, gxr, 1, lda, n)) {
            fail(16);
            return;
        }
        CUCOMPLEX rhoa = CUFFT_ZERO, rhota = CUFFT_ZERO;
        if (!dotc(handle, n, gwork + size_t(1) * lda, gwork + size_t(13) * lda, &rhoa, context) ||
            !dotc(handle, n, gwork + size_t(1) * lda, gwork + size_t(12) * lda, &rhota, context)) {
            fail(16);
            return;
        }
        if (scalar_is_zero(*rho) || scalar_is_zero(*rhot) || scalar_is_zero(*alpha) ||
            scalar_is_zero(*alphat)) {
            fail(5);
            return;
        }
        const HostComplex beta =
            host_value(rhoa) / host_value(*rho) * host_value(*alpha) / host_value(*alphat);
        const HostComplex betat =
            host_value(rhota) / host_value(*rhot) * host_value(*alphat) / host_value(*alpha);
        const CUCOMPLEX cbeta = device_value(beta), cbetat = device_value(betat);
        *rho = rhoa;
        const CUCOMPLEX one = CUFFT_REAL(CU_REAL(1));
        const int c0[2] = {0, 6};
        const CUCOMPLEX a0[2] = {one, cbetat};
        const int c1[2] = {1, 7};
        const CUCOMPLEX a1[2] = {one, cbetat};
        const int c2[2] = {0, 10};
        const CUCOMPLEX a2[2] = {one, cbeta};
        const int c3[2] = {1, 11};
        const CUCOMPLEX a3[2] = {one, cbeta};
        const int cq[3] = {5, 11, 2};
        const CUCOMPLEX aq[3] = {one, cbeta, device_value(beta * betat)};
        if (!combine(gwork, gxsol, lda, n, 4, c0, a0, 2) ||
            !combine(gwork, gxsol, lda, n, 5, c1, a1, 2) ||
            !combine(gwork, gxsol, lda, n, 8, c2, a2, 2) ||
            !combine(gwork, gxsol, lda, n, 9, c3, a3, 2) ||
            !combine(gwork, gxsol, lda, n, 2, cq, aq, 3) || !publish_work(gxi, gwork, 2, lda, n)) {
            fail(16);
            return;
        }
        *nou = 5;
        finish(false);
        return;
    }
    if (*nou == 5) {
        if (!save_xr(gwork, gxr, 3, lda, n)) {
            fail(16);
            return;
        }
        if (*itno <= *maxitp) {
            *nou = 5;
            ++(*itno);
            if (!stage_iteration(gxi, gxsol, gwork, lda, n, alpha, alphat, *rho, *rhot, *tolerancep,
                                 *norm, nstat, steperr, nou, handle)) {
                finish(false);
                return;
            }
            finish(*nstat == 1);
            return;
        }
        if (!launch(gcors2_publish_solution_wrapper(gxi, gxsol, n), "publish final solution")) {
            fail(16);
            return;
        }
        *nstat = 1;
        finish(true);
        return;
    }
    if (*nou == 6) {
        *nou = 4;
        finish(false);
        return;
    }
    fail(12);
}
