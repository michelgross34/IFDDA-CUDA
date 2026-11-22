#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <climits>
#include <cmath>
#include <cstddef>

//-----------------------------------------------------------
extern "C" int ifdda_correlated_material_field_cuda(
    int nx, int ny, int nz, CU_REAL spacing, CU_REAL coherence_length,
    CU_REAL standard_deviation, CU_REAL epsilon_real, CU_REAL epsilon_imag,
    const CU_REAL *phases, CU_REAL *real_field, CU_REAL *mean,
    CU_REAL *standard_deviation_result) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || !phases || !real_field || !mean ||
        !standard_deviation_result || spacing <= CU_REAL(0) ||
        coherence_length <= CU_REAL(0) || standard_deviation <= CU_REAL(0) ||
        nx > INT_MAX / ny || nx * ny > INT_MAX / nz || !initialize_cuda_device())
        return 1;

    const int count = nx * ny * nz;
    const CU_REAL pi = CU_REAL(3.141592653589793238462643383279502884L);
    const CU_REAL lx = CU_REAL(nx) * spacing;
    const CU_REAL ly = CU_REAL(ny) * spacing;
    const CU_REAL lz = CU_REAL(nz) * spacing;
    const CU_REAL dkx = CU_REAL(2) * pi / lx;
    const CU_REAL dky = CU_REAL(2) * pi / ly;
    const CU_REAL dkz = CU_REAL(2) * pi / lz;
    const CU_REAL coefficient1 = CU_REAL(0.125) * standard_deviation *
        standard_deviation * coherence_length * coherence_length * coherence_length /
        (pi * std::sqrt(pi));
    const CU_REAL coefficient2 = coherence_length * coherence_length;
    const CU_REAL coefficient3 = CU_REAL(4) * dkx * dky * dkz;

    cuda_data::InhomogeneousMaterialDeviceBuffers buffers{};
    if (!cuda_data::get_device_inhomogeneous_material_buffers(nx, ny, nz, &buffers))
        return 2;
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.phases, phases, size_t(count) * sizeof(CU_REAL),
        cuda_utils::TransferCategory::Material);
    if (status != cudaSuccess)
        return 3;
    if (!cuda_utils::check_cuda(
            inhomogeneous_material_spectrum_wrapper(
                buffers.spectrum, buffers.phases, nx, ny, nz, dkx, dky, dkz,
                coefficient1, coefficient2, coefficient3),
            "inhomogeneous material", "build random spectrum"))
        return 4;

#if PRODUITMATVECTFFT_SINGLE_PRECISION
    if (!cuda_utils::check_cufft(
            cufftExecC2C(buffers.fft_plan, buffers.spectrum, buffers.spectrum,
                         CUFFT_INVERSE),
            "inhomogeneous material", "execute inverse cuFFT"))
        return 5;
#else
    if (!cuda_utils::check_cufft(
            cufftExecZ2Z(buffers.fft_plan, buffers.spectrum, buffers.spectrum,
                         CUFFT_INVERSE),
            "inhomogeneous material", "execute inverse cuFFT"))
        return 5;
#endif

    if (!cuda_utils::check_cuda(
            inhomogeneous_material_finalize_wrapper(
                buffers.spectrum, buffers.real_values, buffers.magnitude_squares,
                count, epsilon_real, epsilon_imag),
            "inhomogeneous material", "form permittivity field") ||
        !cuda_utils::check_cuda(fill_real_ones_wrapper(buffers.ones, count),
                                "inhomogeneous material", "prepare reduction vector"))
        return 6;

    const cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle)
        return 7;
    CU_REAL sum = CU_REAL(0);
    CU_REAL sum_squares = CU_REAL(0);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    cublasStatus_t blas_status = cublasSdot(
        handle, count, buffers.real_values, 1, buffers.ones, 1, &sum);
    if (blas_status == CUBLAS_STATUS_SUCCESS)
        blas_status = cublasSdot(handle, count, buffers.magnitude_squares, 1,
                                 buffers.ones, 1, &sum_squares);
#else
    cublasStatus_t blas_status = cublasDdot(
        handle, count, buffers.real_values, 1, buffers.ones, 1, &sum);
    if (blas_status == CUBLAS_STATUS_SUCCESS)
        blas_status = cublasDdot(handle, count, buffers.magnitude_squares, 1,
                                 buffers.ones, 1, &sum_squares);
#endif
    if (!cuda_utils::check_cublas(blas_status, "inhomogeneous material",
                                  "reduce mean and variance"))
        return 8;

    status = cuda_data::copy_device_to_host(
        real_field, buffers.real_values, size_t(count) * sizeof(CU_REAL),
        cuda_utils::TransferCategory::Material);
    if (status != cudaSuccess)
        return 9;
    *mean = sum / CU_REAL(count);
    const CU_REAL mean_square = sum_squares / CU_REAL(count);
    *standard_deviation_result = std::sqrt(mean_square - *mean * *mean);
    return 0;
}
