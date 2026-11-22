#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>



namespace {
using cuda_utils::check_cuda;
using cuda_utils::device_value;
using cuda_utils::dotc;
using cuda_utils::dotu;
using cuda_utils::get_cublas_handle;
using cuda_utils::host_value;
using cuda_utils::norm2;
using cuda_utils::scalar_is_zero;

//-----------------------------------------------------------
bool publish_xi(COMPLEX *cpu_xi, const CUCOMPLEX *gpu_xi, size_t n, int *nstat, int *steperr) {
    if (copy_gpbicg_results_to_cpu(cpu_xi, nullptr, gpu_xi, nullptr, n, 0, true, false))
        return true;
    *nstat = -1;
    *steperr = 15;
    return false;
}

//-----------------------------------------------------------
void idrs_impl(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
               const int *ndim, const int *nlar, int *nou, int *ns, COMPLEX *work, int *itno,
               const int *maxit, const CU_REAL *tol, CU_REAL *residu, CU_REAL *norm,
               CUCOMPLEX *omega, int *is, int *nstat, int *steperr) {
    if (!x || !xi || !xr || !b || !lda || !ndim || !nlar || !nou || !ns || !work || !itno ||
        !maxit || !tol || !residu || !norm || !omega || !is || !nstat || !steperr || *ndim <= 0 ||
        *lda < *ndim || *ns < 1 || *ns > 8 || *nlar < 5 + 3 * (*ns) || *maxit < 0) {
        if (nstat)
            *nstat = -1;
        if (steperr)
            *steperr = 1;
        return;
    }

    const int n = *ndim, s = *ns, ld = *lda;
    const size_t count = static_cast<size_t>(n);
    const size_t work_count = static_cast<size_t>(ld) * static_cast<size_t>(*nlar);
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
    *nstat = 0;
    *steperr = -1;
    const auto fail = [&](int code) {
        *nstat = -1;
        *steperr = code;
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        (void)copy_tfqmr_x_to_cpu(x, gx, count);
    };
    const auto launch = [&](cudaError_t status, const char *op) {
        if (check_cuda(status, "IDRS CUDA", op))
            return true;
        *nstat = -1;
        *steperr = 16;
        return false;
    };
    const auto norm_col = [&](int col, CU_REAL *result) {
        return norm2(handle, n, gw + size_t(col) * ld, result, "IDRS CUDA");
    };
    const auto inner = [&](const CUCOMPLEX *u, const CUCOMPLEX *v, CUCOMPLEX *result) {
        return dotc(handle, n, u, v, result, "IDRS CUDA");
    };

    if (first) {
        if (!norm2(handle, n, gb, norm, "IDRS CUDA") || !std::isfinite(*norm)) {
            fail(1);
            return;
        }
        *nou = 1;
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    if (*nou == 1) {
        if (!launch(idrs_initialize_wrapper(gxr, gb, gw, ld, n), "initialize residual")) {
            fail(1);
            return;
        }
        CU_REAL rnorm = 0;
        if (!norm_col(0, &rnorm)) {
            fail(1);
            return;
        }
        std::printf("IDRS initial residue %.16g\n", static_cast<double>(rnorm / (*norm)));
        std::vector<CUCOMPLEX> random_values(count * static_cast<size_t>(s));
        idrs_fill_random_(&n, &s, random_values.data());
        for (int j = 0; j < s; ++j) {
            CUCOMPLEX *q = gw + size_t(4 + s + j + 1) * ld;
            const CUCOMPLEX *source = random_values.data() + size_t(j) * n;
            const cudaError_t copied = cuda_data::copy_host_to_device(
                q, source, count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Other);
            if (!check_cuda(copied, "IDRS CUDA", "copy random shadow vector")) {
                fail(1);
                return;
            }
            const int pcol = 4 + j + 1, qcol = 4 + s + j + 1;
            if (!launch(idrs_copy_column_wrapper(gw, pcol, qcol, ld, n), "copy initial basis")) {
                fail(1);
                return;
            }
            for (int ic = 0; ic < j; ++ic) {
                CUCOMPLEX coefficient = CUFFT_ZERO;
                if (!inner(gw + size_t(4 + ic + 1) * ld, q, &coefficient)) {
                    fail(1);
                    return;
                }
                if (!launch(idrs_axpy_wrapper(gw, pcol, 4 + ic + 1,
                                              device_value(-host_value(coefficient)), ld, n),
                            "orthogonalize initial basis")) {
                    fail(1);
                    return;
                }
            }
            CU_REAL pnorm = 0;
            if (!norm_col(pcol, &pnorm) || pnorm <= CU_REAL(0)) {
                fail(1);
                return;
            }
            if (!launch(idrs_normalize_basis_wrapper(gw, pcol, pnorm, ld, n),
                        "normalize initial basis")) {
                fail(1);
                return;
            }
        }
        *is = 0;
        if (!launch(idrs_copy_work_to_xi_wrapper(gxi, gw, 0, ld, n),
                    "request initial residual product")) {
            fail(1);
            return;
        }
        *nou = 2;
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        return;
    }

    std::vector<CUCOMPLEX> coeff(static_cast<size_t>(s), CUFFT_ZERO);
    const auto load_coefficients = [&]() -> bool {
        return cuda_data::copy_device_to_host(coeff.data(), gw + size_t(2) * ld,
                                              size_t(s) * sizeof(CUCOMPLEX),
                                              cuda_utils::TransferCategory::Other) == cudaSuccess;
    };
    const auto save_coefficients = [&]() -> bool {
        return cuda_data::copy_host_to_device(gw + size_t(2) * ld, coeff.data(),
                                              size_t(s) * sizeof(CUCOMPLEX),
                                              cuda_utils::TransferCategory::Other) == cudaSuccess;
    };
    bool need_loop = false;

    if (*nou == 2) {
        CUCOMPLEX numerator = CUFFT_ZERO, denominator = CUFFT_ZERO;
        if (!launch(idrs_copy_xr_wrapper(gw, gxr, 1, ld, n), "save A*r")) {
            fail(2);
            return;
        }
        if (!inner(gxr, gw, &numerator) || !inner(gxr, gxr, &denominator)) {
            fail(2);
            return;
        }
        if (scalar_is_zero(denominator)) {
            fail(2);
            return;
        }
        *omega = device_value(host_value(numerator) / host_value(denominator));
        if (!launch(idrs_initial_pair_wrapper(gw, gx, *omega, *is, s, ld, n),
                    "update initial residual")) {
            fail(2);
            return;
        }
        if (*is < s - 1) {
            ++(*is);
            if (!launch(idrs_copy_work_to_xi_wrapper(gxi, gw, 0, ld, n),
                        "request next initial product")) {
                fail(2);
                return;
            }
            (void)publish_xi(xi, gxi, count, nstat, steperr);
            return;
        }
        *itno = s;
        need_loop = true;
    } else if (*nou == 3 || *nou == 4) {
        if (!load_coefficients()) {
            fail(3);
            return;
        }
        if (*nou == 3) {
            CUCOMPLEX numerator = CUFFT_ZERO, denominator = CUFFT_ZERO;
            if (!inner(gxr, gw + size_t(1) * ld, &numerator) || !inner(gxr, gxr, &denominator)) {
                fail(3);
                return;
            }
            if (scalar_is_zero(denominator)) {
                fail(3);
                return;
            }
            *omega = device_value(host_value(numerator) / host_value(denominator));
            if (!launch(idrs_scale_copy_wrapper(gw, 3, 1, *omega, ld, n), "form q") ||
                !launch(idrs_scale_xr_wrapper(gw, gxr, 4, device_value(-host_value(*omega)), ld, n),
                        "form e")) {
                fail(3);
                return;
            }
            for (int j = 0; j < s; ++j) {
                const CUCOMPLEX negative_coefficient = device_value(-host_value(coeff[j]));
                if (!launch(idrs_axpy_wrapper(gw, 3, 4 + s + j + 1, negative_coefficient, ld, n),
                            "orthogonalize q") ||
                    !launch(
                        idrs_axpy_wrapper(gw, 4, 4 + 2 * s + j + 1, negative_coefficient, ld, n),
                        "orthogonalize e")) {
                    fail(3);
                    return;
                }
            }
        } else {
            if (!launch(idrs_scale_copy_wrapper(gw, 3, 1, *omega, ld, n), "form q") ||
                !launch(idrs_negate_xr_wrapper(gw, gxr, ld, n), "form e")) {
                fail(4);
                return;
            }
            for (int j = 0; j < s; ++j) {
                if (!launch(idrs_axpy_wrapper(gw, 3, 4 + s + j + 1,
                                              device_value(-host_value(coeff[j])), ld, n),
                            "orthogonalize q")) {
                    fail(4);
                    return;
                }
            }
        }
        if (!launch(idrs_apply_pair_wrapper(gw, gx, ld, n), "update solution and residual")) {
            fail(4);
            return;
        }
        if (*nou == 4) {
            CU_REAL rnorm = 0;
            if (!norm_col(0, &rnorm)) {
                fail(4);
                return;
            }
            *residu = rnorm / (*norm);
            std::printf("IDRS residue %.16g iteration %d\n", static_cast<double>(*residu), *itno);
            if (*residu <= *tol) {
                *nstat = 1;
                *nou = 5;
                if (!publish_xi(xi, gxi, count, nstat, steperr) ||
                    !copy_tfqmr_x_to_cpu(x, gx, count)) {
                    *nstat = -1;
                    *steperr = 15;
                }
                return;
            }
            if (*residu >= CU_REAL(100)) {
                fail(0);
                return;
            }
        }
        if (!launch(idrs_shift_qe_wrapper(gw, s, ld, n), "shift Q and E basis")) {
            fail(4);
            return;
        }
        if (*itno > *maxit) {
            fail(0);
            return;
        }
        ++(*itno);
        need_loop = true;
    }

    if (!need_loop)
        return;
    for (;;) {
        // Dot products return this small coefficient system to the host. Use
        // the wrapper's fixed 9-column row-major layout; solve it on the GPU.
        CUCOMPLEX matrix[81]{};
        CUCOMPLEX rhs[9]{};
        coeff.assign(static_cast<size_t>(s), CUFFT_ZERO);
        for (int row = 0; row < s; ++row) {
            if (!dotu(handle, n, gw + size_t(4 + row + 1) * ld, gw, &coeff[row], "IDRS CUDA")) {
                std::fprintf(stderr, "IDRS dotu(P,r) failed row=%d\\n", row);
                fail(5);
                return;
            }
            for (int col = 0; col < s; ++col) {
                CUCOMPLEX value = CUFFT_ZERO;
                if (!dotu(handle, n, gw + size_t(4 + row + 1) * ld,
                          gw + size_t(4 + 2 * s + col + 1) * ld, &value, "IDRS CUDA")) {
                    std::fprintf(stderr, "IDRS dotu(P,E) failed row=%d col=%d\\n", row, col);
                    fail(5);
                    return;
                }
                matrix[size_t(row) * 9 + col] = value;
            }
        }
        for (int j = 0; j < s; ++j)
            rhs[j] = coeff[j];
        int dense_info = 0;
        const cudaError_t dense_status =
            cuda_data::solve_small_dense_system_host(matrix, rhs, s, &dense_info);
        if (!check_cuda(dense_status, "IDRS CUDA", "GPU dense coefficient solve") ||
            dense_info != 0) {
            std::fprintf(stderr, "IDRS GPU dense solve failed, ns=%d info=%d\\n", s, dense_info);
            fail(5);
            return;
        }
        for (int j = 0; j < s; ++j)
            coeff[j] = rhs[j];
        if (!save_coefficients()) {
            fail(5);
            return;
        }

        if (!launch(idrs_copy_column_wrapper(gw, 1, 0, ld, n), "initialize v")) {
            fail(5);
            return;
        }
        for (int j = 0; j < s; ++j) {
            if (!launch(idrs_axpy_wrapper(gw, 1, 4 + 2 * s + j + 1,
                                          device_value(-host_value(rhs[j])), ld, n),
                        "form v")) {
                fail(5);
                return;
            }
        }

        if ((*itno % (s + 1)) == s) {
            if (!launch(idrs_copy_work_to_xi_wrapper(gxi, gw, 1, ld, n), "request A*v")) {
                fail(5);
                return;
            }
            *nou = 3;
        } else {
            if (!launch(idrs_scale_copy_wrapper(gw, 3, 1, *omega, ld, n), "form q")) {
                fail(5);
                return;
            }
            for (int j = 0; j < s; ++j) {
                if (!launch(idrs_axpy_wrapper(gw, 3, 4 + s + j + 1,
                                              device_value(-host_value(rhs[j])), ld, n),
                            "form q")) {
                    fail(5);
                    return;
                }
            }
            if (!launch(idrs_copy_work_to_xi_wrapper(gxi, gw, 3, ld, n), "request A*q")) {
                fail(5);
                return;
            }
            *nou = 4;
        }
        (void)publish_xi(xi, gxi, count, nstat, steperr);
        return;
    }
}
} // namespace

//-----------------------------------------------------------
extern "C" void idrs_(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                      const int *ndim, const int *nlar, int *nou, int *ns, COMPLEX *work,
                      int *itno, const int *maxit, const CU_REAL *tol, CU_REAL *residu,
                      CU_REAL *norm, CUCOMPLEX *omega, int *is, int *nstat, int *steperr) {
    idrs_impl(x, xi, xr, b, lda, ndim, nlar, nou, ns, work, itno, maxit, tol, residu, norm, omega,
              is, nstat, steperr);
}
