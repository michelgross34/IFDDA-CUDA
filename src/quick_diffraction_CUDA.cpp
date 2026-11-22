#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <climits>
#include <cmath>
#include <limits>

//-----------------------------------------------------------
extern "C" void ifdda_quick_diffraction_cuda_(
    const int *nx, const int *ny, const int *nz, const int *nfft, int *fft_map, const REAL *k0,
    const REAL *xs, const REAL *ys, const REAL *zs, const REAL *spacing,
    const COMPLEX *field, const int *ntheta, const int *nphi, const REAL *intensity,
    const REAL *theta, const REAL *phi, const REAL *const_poynting, int *imax,
    REAL *deltakx, REAL *deltaky, REAL *positive_spectrum, REAL *negative_spectrum,
    REAL *kx_values, REAL *angular_poynting, REAL *theta_values, REAL *phi_values,
    REAL *asymmetry, REAL *scattering, int *status_out) {
    if (status_out)
        *status_out = 1;
    if (!nx || !ny || !nz || !nfft || !fft_map || !k0 || !xs || !ys || !zs || !spacing || !field ||
        !ntheta || !nphi || !intensity || !theta || !phi || !const_poynting || !imax || !deltakx ||
        !deltaky || !positive_spectrum || !negative_spectrum || !kx_values || !angular_poynting ||
        !theta_values || !phi_values || !asymmetry || !scattering || !status_out || *nx <= 0 ||
        *ny <= 0 || *nz <= 0 || *nfft < 2 || *nfft > 16384 || *k0 <= CU_REAL(0) ||
        *spacing <= CU_REAL(0) || *intensity <= CU_REAL(0) || *ntheta <= 0 || *nphi <= 0)
        return;

    auto fail = [&](int status, const char *message) {
        (void)message;
        *status_out = status;
    };

    const size_t nxny = size_t(*nx) * size_t(*ny);
    if (nxny > size_t(INT_MAX) ||
        size_t(*nfft) > std::numeric_limits<size_t>::max() / size_t(*nfft)) {
        fail(1, "quick diffraction dimensions overflow");
        return;
    }
    const size_t plane = size_t(*nfft) * size_t(*nfft);
    if ((size_t(*ntheta) + size_t(1)) > std::numeric_limits<size_t>::max() / size_t(*nphi)) {
        fail(1, "quick diffraction angle dimensions overflow");
        return;
    }
    const size_t angle_count = (size_t(*ntheta) + size_t(1)) * size_t(*nphi);
    if (plane > size_t(INT_MAX) || nxny > std::numeric_limits<size_t>::max() / size_t(*nz) ||
        //-----------------------------------------------------------
        nxny * size_t(*nz) > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX))) {
        fail(1, "quick diffraction dimensions overflow");
        return;
    }

    const CU_REAL pi = CU_REAL(3.14159265358979323846);
    *deltakx = CU_REAL(2) * pi / (CU_REAL(*nfft) * *spacing);
    *deltaky = *deltakx;
    if (*deltakx >= *k0) {
        fail(1, "In FFT Poynting nfft2d too small");
        return;
    }
    const int half = *nfft / 2;
    *imax = int(std::floor(*k0 / *deltakx + CU_REAL(0.5))) + 1;
    if (2 * *imax + 1 > *nfft || 2 * *imax + 1 < 7) {
        fail(1, "In FFT diffract nfft2d too small");
        return;
    }
    const int np = 2 * *imax + 1;
    const size_t spectrum_count = size_t(np) * size_t(np);
    if (spectrum_count > plane || spectrum_count > size_t(INT_MAX) ||
        angle_count > size_t(INT_MAX)) {
        fail(1, "quick diffraction output dimensions overflow");
        return;
    }
    if (!initialize_cuda_device()) {
        fail(2, "CUDA device initialization failed");
        return;
    }

    cuda_data::ScatteringResultDeviceBuffers buffers{};
    if (!cuda_data::get_device_scattering_result_buffers(nxny * size_t(*nz), angle_count, plane,
                                                         plane, size_t(*nfft), &buffers)) {
        fail(3, "cannot acquire quick diffraction GPU buffers");
        return;
    }

    const size_t field_count = 3 * nxny * size_t(*nz);
    const size_t field_bytes = field_count * sizeof(COMPLEX);
    const size_t coord_bytes = nxny * size_t(*nz) * sizeof(REAL);
    const CUCOMPLEX* resident_field = get_device_result_vector_alias(field, field_count);
    const CUCOMPLEX* device_field = resident_field ? resident_field : buffers.dipole_field;
    const CU_REAL *device_xs = nullptr, *device_ys = nullptr, *device_zs = nullptr;
    const bool coordinates_are_resident = cuda_data::get_device_geometry_coordinate_aliases(
        xs, ys, zs, nxny * size_t(*nz), &device_xs, &device_ys, &device_zs);
    (void)device_xs;
    (void)device_ys;
    if ((!resident_field &&
         cuda_data::copy_host_to_device(buffers.dipole_field, field, field_bytes,
                                        cuda_utils::TransferCategory::Result) != cudaSuccess) ||
        (!coordinates_are_resident &&
         cuda_data::copy_host_to_device(buffers.zs, zs, coord_bytes,
                                        cuda_utils::TransferCategory::Geometry) != cudaSuccess) ||
        quick_diffraction_build_fft_map_wrapper(buffers.quick_fft_map, *nfft) != cudaSuccess ||
        cuda_data::copy_device_to_host(fft_map, buffers.quick_fft_map,
                                       size_t(*nfft) * sizeof(int),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cudaMemset(buffers.quick_positive_field, 0, 3 * plane * sizeof(CUCOMPLEX)) != cudaSuccess ||
        cudaMemset(buffers.quick_negative_field, 0, 3 * plane * sizeof(CUCOMPLEX)) != cudaSuccess) {
        fail(4, "quick diffraction buffer transfer/initialization failed");
        return;
    }
    if (!coordinates_are_resident)
        device_zs = buffers.zs;

    const cufftHandle plan = get_speckle_fft_plan(*nfft);
    if (!plan) {
        fail(5, "cannot acquire quick diffraction FFT plan");
        return;
    }

    const CU_REAL phase_x = (xs[0] + CU_REAL(half) * *spacing) * *deltakx;
    const CU_REAL phase_y = (ys[0] + CU_REAL(half) * *spacing) * *deltaky;
    cudaError_t kernel_status = cudaSuccess;
    cufftResult fft_status = CUFFT_SUCCESS;
    for (int k = 0; k < *nz && kernel_status == cudaSuccess && fft_status == CUFFT_SUCCESS; ++k) {
        kernel_status =
            quick_diffraction_gather_wrapper(device_field, buffers.quick_fft_images,
            buffers.quick_fft_map, *nx, *ny, int(nxny), *nfft, k);
        if (kernel_status != cudaSuccess)
            break;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        fft_status =
            cufftExecC2C(plan, buffers.quick_fft_images, buffers.quick_fft_images, CUFFT_FORWARD);
#else
        fft_status =
            cufftExecZ2Z(plan, buffers.quick_fft_images, buffers.quick_fft_images, CUFFT_FORWARD);
#endif
        if (fft_status != CUFFT_SUCCESS)
            break;
        kernel_status = quick_diffraction_accumulate_wrapper(
            buffers.quick_fft_images, buffers.quick_fft_map, device_zs, int(nxny), *nfft, *imax, k,
            *k0, *deltakx, phase_x, phase_y, buffers.quick_positive_field,
            buffers.quick_negative_field);
    }
    if (kernel_status == cudaSuccess && fft_status == CUFFT_SUCCESS)
        kernel_status = quick_diffraction_postprocess_wrapper(
            buffers.quick_positive_field, buffers.quick_negative_field, *nfft, *imax, *deltakx,
            *deltaky, *k0, *theta * pi / CU_REAL(180), *phi * pi / CU_REAL(180), *const_poynting,
            int(angle_count > 0 ? *ntheta : 0), *nphi, buffers.positive_spectrum,
            buffers.negative_spectrum, buffers.kx_values, buffers.angular_poynting,
            buffers.theta_values, buffers.phi_values, buffers.weighted_scattering,
            buffers.weighted_asymmetry, buffers.reduction_ones);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (kernel_status != cudaSuccess || fft_status != CUFFT_SUCCESS || sync_status != cudaSuccess) {
        fail(6, "quick diffraction GPU FFT/kernel execution failed");
        return;
    }

    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle) {
        fail(7, "cannot acquire cuBLAS reduction handle");
        return;
    }
    cublasPointerMode_t previous_mode = CUBLAS_POINTER_MODE_HOST;
    if (!cuda_utils::check_cublas(cublasGetPointerMode(handle, &previous_mode), "quick diffraction",
                                  "get reduction pointer mode") ||
        !cuda_utils::check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                                  "quick diffraction", "select host reduction results")) {
        fail(7, "cannot configure quick diffraction reductions");
        return;
    }
    CU_REAL scattering_sum = CU_REAL(0), asymmetry_sum = CU_REAL(0);
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const cublasStatus_t scattering_status =
        cublasSdot(handle, int(spectrum_count), buffers.reduction_ones, 1,
                   buffers.weighted_scattering, 1, &scattering_sum);
    const cublasStatus_t asymmetry_status =
        cublasSdot(handle, int(spectrum_count), buffers.reduction_ones, 1,
                   buffers.weighted_asymmetry, 1, &asymmetry_sum);
#else
    const cublasStatus_t scattering_status =
        cublasDdot(handle, int(spectrum_count), buffers.reduction_ones, 1,
                   buffers.weighted_scattering, 1, &scattering_sum);
    const cublasStatus_t asymmetry_status =
        cublasDdot(handle, int(spectrum_count), buffers.reduction_ones, 1,
                   buffers.weighted_asymmetry, 1, &asymmetry_sum);
#endif
    const cublasStatus_t restore_status = cublasSetPointerMode(handle, previous_mode);
    if (!cuda_utils::check_cublas(scattering_status, "quick diffraction", "reduce scattering") ||
        !cuda_utils::check_cublas(asymmetry_status, "quick diffraction", "reduce asymmetry") ||
        !cuda_utils::check_cublas(restore_status, "quick diffraction",
                                  "restore reduction pointer mode")) {
        fail(7, "quick diffraction cuBLAS reduction failed");
        return;
    }

    const size_t output_k_count = size_t(np);
    if (cuda_data::copy_device_to_host(positive_spectrum, buffers.positive_spectrum,
                                       //-----------------------------------------------------------
                                       spectrum_count * sizeof(REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(negative_spectrum, buffers.negative_spectrum,
                                       //-----------------------------------------------------------
                                       spectrum_count * sizeof(REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(kx_values, buffers.kx_values,
                                       //-----------------------------------------------------------
                                       output_k_count * sizeof(REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(angular_poynting, buffers.angular_poynting,
                                       //-----------------------------------------------------------
                                       angle_count * sizeof(REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(theta_values, buffers.theta_values,
                                       //-----------------------------------------------------------
                                       angle_count * sizeof(REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(phi_values, buffers.phi_values,
                                       //-----------------------------------------------------------
                                       angle_count * sizeof(REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess) {
        fail(8, "cannot retrieve quick diffraction result arrays");
        return;
    }
    *scattering = scattering_sum * *k0 * *k0 * *k0 / *intensity;
    *asymmetry = scattering_sum != CU_REAL(0) ? asymmetry_sum / scattering_sum : CU_REAL(0);
    *status_out = 0;
}
