#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <climits>
#include <cstdio>
#include <limits>

//-----------------------------------------------------------
extern "C" void fouriertoimage_(const CU_REAL *deltakx, const CU_REAL *deltaky,
                                  const CU_REAL *gross, CUCOMPLEX *image_x,
                                  CUCOMPLEX *image_y, CUCOMPLEX *image_z,
                                  CUCOMPLEX *incident_x, CUCOMPLEX *incident_y,
                                  CUCOMPLEX *incident_z, const int *nfft,
                                  const int *nfft_half, const long long *plan_backward,
                                  const long long *plan_forward) {
    if (!deltakx || !deltaky || !gross || !image_x || !image_y || !image_z ||
        !incident_x || !incident_y || !incident_z || !nfft || !nfft_half ||
        !plan_backward || !plan_forward || *nfft <= 0 || (*nfft & 1) != 0 ||
        *nfft > INT_MAX / *nfft || *nfft_half != *nfft / 2 || *gross == CU_REAL(0)) {
        std::fprintf(stderr, "fouriertoimage CUDA: invalid input\n");
        return;
    }
    if (!initialize_cuda_device())
        return;

    const size_t plane = size_t(*nfft) * size_t(*nfft);
    if (plane > std::numeric_limits<size_t>::max() / (6 * sizeof(CUCOMPLEX)))
        return;
    cuda_data::FourierImageDeviceBuffers buffers{};
    if (!cuda_data::get_device_fourier_image_buffers(plane, &buffers))
        return;

    const CUCOMPLEX *host_fields[6] = {incident_x, incident_y, incident_z,
                                       image_x, image_y, image_z};
    for (size_t component = 0; component < 6; ++component) {
        const cuda_utils::TransferCategory category = component < 3
            ? cuda_utils::TransferCategory::IncidentField
            : cuda_utils::TransferCategory::Result;
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(buffers.fields + component * plane,
                                               host_fields[component],
                                               plane * sizeof(CUCOMPLEX), category),
                "fouriertoimage CUDA", "upload image plane"))
            return;
    }

    const cufftHandle plan = get_speckle_fft_plan(*nfft);
    if (plan == 0) {
        std::fprintf(stderr, "fouriertoimage CUDA: failed to create cuFFT plan\n");
        return;
    }
    if (!cuda_utils::check_cufft(CUFFT_EXECUTE(plan, buffers.fields, buffers.fields,
                                                CUFFT_FORWARD),
                                 "fouriertoimage CUDA", "transform incident image planes") ||
        !cuda_utils::check_cufft(CUFFT_EXECUTE(plan, buffers.fields + 3 * plane,
                                                buffers.fields + 3 * plane, CUFFT_FORWARD),
                                 "fouriertoimage CUDA", "transform scattered image planes") ||
        !cuda_utils::check_cuda(fourier_to_image_rearrange_wrapper(
                                    buffers.fields, *nfft,
                                    (*deltakx * *deltaky) / *gross),
                                "fouriertoimage CUDA", "rearrange and scale image planes"))
        return;

    CUCOMPLEX *host_outputs[6] = {incident_x, incident_y, incident_z,
                                  image_x, image_y, image_z};
    for (size_t component = 0; component < 6; ++component) {
        const cuda_utils::TransferCategory category = component < 3
            ? cuda_utils::TransferCategory::IncidentField
            : cuda_utils::TransferCategory::Result;
        if (!cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(host_outputs[component],
                                               buffers.fields + component * plane,
                                               plane * sizeof(CUCOMPLEX), category),
                "fouriertoimage CUDA", "download image plane"))
            return;
    }
    (void)plan_backward;
    (void)plan_forward;
}

//-----------------------------------------------------------
extern "C" void fouriertoimage2_cuda_(const CU_REAL *deltakx, const CU_REAL *deltaky,
                                  const CU_REAL *gross, CUCOMPLEX *image_x,
                                  CUCOMPLEX *image_y, CUCOMPLEX *image_z,
                                  const int *nfft, const int *nfft_half,
                                  const long long *plan_backward,
                                  const long long *plan_forward) {
    if (!deltakx || !deltaky || !gross || !image_x || !image_y || !image_z ||
        !nfft || !nfft_half || !plan_backward || !plan_forward || *nfft <= 0 ||
        (*nfft & 1) != 0 || *nfft > INT_MAX / *nfft ||
        *nfft_half != *nfft / 2 || *gross == CU_REAL(0)) {
        std::fprintf(stderr, "fouriertoimage2 CUDA: invalid input\\n");
        return;
    }
    if (!initialize_cuda_device())
        return;

    const size_t plane = size_t(*nfft) * size_t(*nfft);
    if (plane > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)))
        return;
    cuda_data::FourierImageDeviceBuffers buffers{};
    if (!cuda_data::get_device_fourier_image_buffers(plane, &buffers) || !buffers.fields)
        return;

    const CUCOMPLEX *host_fields[3] = {image_x, image_y, image_z};
    for (size_t component = 0; component < 3; ++component) {
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(buffers.fields + component * plane,
                                               host_fields[component],
                                               plane * sizeof(CUCOMPLEX),
                                               cuda_utils::TransferCategory::Result),
                "fouriertoimage2 CUDA", "upload image plane"))
            return;
    }

    const cufftHandle plan = get_speckle_fft_plan(*nfft);
    if (!plan || !cuda_utils::check_cufft(
                     CUFFT_EXECUTE(plan, buffers.fields, buffers.fields, CUFFT_FORWARD),
                     "fouriertoimage2 CUDA", "transform three Fourier planes") ||
        !cuda_utils::check_cuda(
            fourier_to_image_rearrange_wrapper(
                buffers.fields, *nfft, (*deltakx * *deltaky) / *gross),
            "fouriertoimage2 CUDA", "shift and scale image planes"))
        return;

    CUCOMPLEX *host_outputs[3] = {image_x, image_y, image_z};
    for (size_t component = 0; component < 3; ++component) {
        if (!cuda_utils::check_cuda(
                cuda_data::copy_device_to_host(host_outputs[component],
                                               buffers.fields + component * plane,
                                               plane * sizeof(CUCOMPLEX),
                                               cuda_utils::TransferCategory::Result),
                "fouriertoimage2 CUDA", "download image plane"))
            return;
    }
    (void)plan_backward;
    (void)plan_forward;
}

//-----------------------------------------------------------
extern "C" void ifdda_direct_scattering_cuda_(
    const COMPLEX *field, const REAL *xs, const REAL *ys, const REAL *zs,
    const int *dipole_count, const int *ntheta, const int *nphi, const int *nfft, const REAL *k0,
    const REAL *theta_degrees, const REAL *phi_degrees, const REAL *intensity,
    const REAL *const_poynting, REAL *angular_poynting, REAL *theta_values,
    REAL *phi_values, REAL *positive_spectrum, REAL *negative_spectrum, REAL *kx_values,
    REAL *asymmetry, REAL *scattering, int *status_out) {
    if (status_out)
        *status_out = 1;
    if (!field || !xs || !ys || !zs || !dipole_count || !ntheta || !nphi || !nfft || !k0 ||
        !theta_degrees || !phi_degrees || !intensity || !const_poynting || !angular_poynting ||
        !theta_values || !phi_values || !positive_spectrum || !negative_spectrum || !kx_values ||
        !asymmetry || !scattering || !status_out || *dipole_count <= 0 || *ntheta <= 10 ||
        *nphi <= 20 || *nfft < 2 || *k0 <= CU_REAL(0) || *intensity <= CU_REAL(0) ||
        *ntheta == INT_MAX || *nphi > INT_MAX / (*ntheta + 1))
        return;

    const int half_denominator = ((*ntheta - 1) < (*nphi - 1) ? (*ntheta - 1) : (*nphi - 1)) / 2;
    if (half_denominator <= 0)
        return;
    const CU_REAL delta = *k0 / CU_REAL(half_denominator);
    const int ll = int(*k0 / delta);
    const int np = 2 * ll + 1;
    if (np > *nfft)
        return;

    const size_t dipoles = size_t(*dipole_count);
    const size_t angles = (size_t(*ntheta) + size_t(1)) * size_t(*nphi);
    const size_t spectrum = size_t(np) * size_t(np);
    if (dipoles > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        angles > size_t(INT_MAX) || spectrum > size_t(INT_MAX))
        return;

    if (!initialize_cuda_device()) {
        *status_out = 2;
        return;
    }
    cuda_data::ScatteringResultDeviceBuffers buffers{};
    if (!cuda_data::get_device_scattering_result_buffers(dipoles, angles, spectrum, 0, 0,
                                                         &buffers)) {
        *status_out = 3;
        return;
    }

    const size_t field_count = 3 * dipoles;
    const size_t field_bytes = field_count * sizeof(CUCOMPLEX);
    const size_t coordinate_bytes = dipoles * sizeof(CU_REAL);
    const CUCOMPLEX *device_field =
        get_device_result_vector_alias(field, field_count);
    const CU_REAL *device_xs = nullptr, *device_ys = nullptr, *device_zs = nullptr;
    if (!device_field &&
        cuda_data::copy_host_to_device(buffers.dipole_field, field, field_bytes,
                                       cuda_utils::TransferCategory::Result) != cudaSuccess) {
        *status_out = 4;
        return;
    }
    if (!device_field)
        device_field = buffers.dipole_field;
    const bool have_coordinate_aliases = cuda_data::get_device_geometry_coordinate_aliases(
        xs, ys, zs, dipoles, &device_xs, &device_ys, &device_zs);
    if (!have_coordinate_aliases) {
        if (cuda_data::copy_host_to_device(buffers.xs, xs, coordinate_bytes,
                                           cuda_utils::TransferCategory::Geometry) != cudaSuccess ||
            cuda_data::copy_host_to_device(buffers.ys, ys, coordinate_bytes,
                                           cuda_utils::TransferCategory::Geometry) != cudaSuccess ||
            cuda_data::copy_host_to_device(buffers.zs, zs, coordinate_bytes,
                                           cuda_utils::TransferCategory::Geometry) != cudaSuccess) {
            *status_out = 4;
            return;
        }
        device_xs = buffers.xs;
        device_ys = buffers.ys;
        device_zs = buffers.zs;
    }

    const CU_REAL pi = CU_REAL(3.14159265358979323846);
    const CU_REAL dtheta = pi / CU_REAL(*ntheta);
    const CU_REAL dphi = CU_REAL(2) * pi / CU_REAL(*nphi);
    const CU_REAL theta0 = *theta_degrees * pi / CU_REAL(180);
    const CU_REAL phi0 = *phi_degrees * pi / CU_REAL(180);
    cudaError_t launch = direct_scattering_angles_wrapper(
        device_field, device_xs, device_ys, device_zs, *dipole_count, *ntheta, *nphi,
        *k0, theta0, phi0, dtheta, dphi, *const_poynting, buffers.angular_poynting,
        buffers.theta_values, buffers.phi_values, buffers.weighted_scattering,
        buffers.weighted_asymmetry, buffers.reduction_ones);
    if (launch == cudaSuccess)
        launch = direct_scattering_spectrum_wrapper(device_field, device_xs, device_ys,
                                                    device_zs, *dipole_count, ll, np, *k0, delta,
                                                    *const_poynting, buffers.positive_spectrum,
                                                    buffers.negative_spectrum, buffers.kx_values);
    if (!cuda_utils::check_cuda(launch, "direct scattering", "launch Poynting kernels")) {
        *status_out = 5;
        return;
    }

    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle) {
        *status_out = 6;
        return;
    }
    cublasPointerMode_t previous_mode = CUBLAS_POINTER_MODE_HOST;
    if (!cuda_utils::check_cublas(cublasGetPointerMode(handle, &previous_mode), "direct scattering",
                                  "get reduction pointer mode") ||
        !cuda_utils::check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                                  "direct scattering", "select host reduction results")) {
        *status_out = 6;
        return;
    }
    CU_REAL scatter_integral = CU_REAL(0);
    CU_REAL asymmetry_integral = CU_REAL(0);
    bool reduction_ok = false;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const cublasStatus_t scatter_status =
        cublasSdot(handle, int(angles), buffers.reduction_ones, 1, buffers.weighted_scattering, 1,
                   &scatter_integral);
    const cublasStatus_t asymmetry_status =
        cublasSdot(handle, int(angles), buffers.reduction_ones, 1, buffers.weighted_asymmetry, 1,
                   &asymmetry_integral);
#else
    const cublasStatus_t scatter_status =
        cublasDdot(handle, int(angles), buffers.reduction_ones, 1, buffers.weighted_scattering, 1,
                   &scatter_integral);
    const cublasStatus_t asymmetry_status =
        cublasDdot(handle, int(angles), buffers.reduction_ones, 1, buffers.weighted_asymmetry, 1,
                   &asymmetry_integral);
#endif
    reduction_ok =
        cuda_utils::check_cublas(scatter_status, "direct scattering", "reduce cross section") &&
        cuda_utils::check_cublas(asymmetry_status, "direct scattering", "reduce asymmetry");
    const cublasStatus_t restore_mode = cublasSetPointerMode(handle, previous_mode);
    if (!reduction_ok || !cuda_utils::check_cublas(restore_mode, "direct scattering",
                                                   "restore reduction pointer mode")) {
        *status_out = 7;
        return;
    }
    if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "direct scattering",
                                "synchronize GPU results")) {
        *status_out = 8;
        return;
    }

    if (cuda_data::copy_device_to_host(angular_poynting, buffers.angular_poynting,
                                       //-----------------------------------------------------------
                                       angles * sizeof(CU_REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(theta_values, buffers.theta_values, angles * sizeof(CU_REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(phi_values, buffers.phi_values, angles * sizeof(CU_REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(positive_spectrum, buffers.positive_spectrum,
                                       //-----------------------------------------------------------
                                       spectrum * sizeof(CU_REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(negative_spectrum, buffers.negative_spectrum,
                                       //-----------------------------------------------------------
                                       spectrum * sizeof(CU_REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess ||
        cuda_data::copy_device_to_host(kx_values, buffers.kx_values, size_t(np) * sizeof(CU_REAL),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess) {
        *status_out = 9;
        return;
    }

    *asymmetry =
        scatter_integral != CU_REAL(0) ? asymmetry_integral / scatter_integral : CU_REAL(0);
    *scattering = scatter_integral * (*k0) * (*k0) * (*k0) * (*k0) / (*intensity);
    *status_out = 0;
}
