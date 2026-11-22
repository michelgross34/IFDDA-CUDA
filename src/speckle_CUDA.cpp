#include "cuda_data.h"
#include "cuda_utils.h"
#define IFDDA_CUDA_CALLERS_CALCULWIDEFIELD_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_CALCULWIDEFIELD_CUDA
#include "kernel.h"

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

//-----------------------------------------------------------
static void ifdda_speckle_field_cuda_impl(
    const int *nx, const int *ny, const int *nz, const int *dipole_count, const int *nfft,
    const CU_REAL *k0, const CU_REAL *xx0, const CU_REAL *yy0, const CU_REAL *zz0,
    const CU_REAL *psi_degrees, const CU_REAL *spacing, const CU_REAL *power,
    const CU_REAL *aperture, COMPLEX *e0, const COMPLEX *phase_positive_cpu,
    const COMPLEX *phase_negative_cpu, const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    const int *bidirectional, COMPLEX *output, CU_REAL *irradiance, int *status_out,
    CUCOMPLEX **resident_output = nullptr,
    const CU_REAL *grid_origin_x = nullptr, const CU_REAL *grid_origin_y = nullptr,
    const CU_REAL *grid_origin_z = nullptr,
    const CUCOMPLEX *phase_positive_device = nullptr,
    const CUCOMPLEX *phase_negative_device = nullptr) {
    const bool phases_are_device = phase_positive_device != nullptr;
    const bool trace = std::getenv("IFDDA_TRACE_SPECKLE") != nullptr;
    if (resident_output)
        *resident_output = nullptr;
    if (status_out)
        *status_out = 1;
    if (!nx || !ny || !nz || !dipole_count || !nfft || !k0 || !xx0 || !yy0 || !zz0 ||
        !psi_degrees || !spacing || !power || !aperture || !e0 ||
        (phases_are_device ? !phase_positive_device : !phase_positive_cpu) ||
        !bidirectional || ((*bidirectional != 0) &&
            (phases_are_device ? !phase_negative_device : !phase_negative_cpu)) ||
        (!xs && !grid_origin_x) || (!ys && !grid_origin_y) || (!zs && !grid_origin_z) ||
        (!output && !resident_output) || !irradiance ||
        !status_out || *nx <= 0 || *ny <= 0 || *nz <= 0 || *dipole_count <= 0 || *nfft <= 0 ||
        *k0 <= CU_REAL(0) || *spacing <= CU_REAL(0) || *power <= CU_REAL(0) ||
        *aperture <= CU_REAL(0) || (*bidirectional != 0 && *bidirectional != 1))
    {
        if (trace)
            std::fprintf(stderr,
                "SPECKLE_TRACE invalid-args bidir=%d phase+null=%d phase-null=%d xs=%d ys=%d zs=%d origin=%d%d%d\n",
                bidirectional ? *bidirectional : -1,
                phases_are_device ? phase_positive_device == nullptr : phase_positive_cpu == nullptr,
                phases_are_device ? phase_negative_device == nullptr : phase_negative_cpu == nullptr,
                xs == nullptr, ys == nullptr, zs == nullptr,
                grid_origin_x != nullptr, grid_origin_y != nullptr, grid_origin_z != nullptr);
        return;
    }

    size_t grid_xy = 0, grid_size = 0, plane_size = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx), size_t(*ny), &grid_xy) ||
        !cuda_utils::checked_multiply(grid_xy, size_t(*nz), &grid_size) ||
        !cuda_utils::checked_multiply(size_t(*nfft), size_t(*nfft), &plane_size) ||
        grid_size != size_t(*dipole_count) || grid_size > size_t(INT_MAX) / 3 ||
        plane_size > size_t(INT_MAX) || grid_xy > size_t(INT_MAX)) {
        *status_out = 5;
        if (trace)
            std::fprintf(stderr, "SPECKLE_TRACE invalid-grid nx=%d ny=%d nz=%d count=%d nfft=%d\n",
                *nx, *ny, *nz, *dipole_count, *nfft);
        return;
    }
    const CU_REAL deltak =
        CU_REAL(2) * CU_REAL(3.1415926535897932384626433832795) / (CU_REAL(*nfft) * *spacing);
    const CU_REAL kmax = *k0 * *aperture;
    if (deltak >= kmax / CU_REAL(2)) {
        *status_out = 2;
        if (trace)
            std::fprintf(stderr, "SPECKLE_TRACE invalid-resolution deltak=%g kmax=%g\n",
                double(deltak), double(kmax));
        return;
    }
    if (*nx >= *nfft) {
        *status_out = 3;
        return;
    }
    if (*ny == *nfft) {
        *status_out = 4;
        return;
    }
    if (!initialize_cuda_device())
    {
        if (trace)
            std::fprintf(stderr, "SPECKLE_TRACE initialize-device failed\n");
        return;
    }

    cuda_data::SpectralFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_spectral_field_buffers(plane_size, grid_size, &buffers))
    {
        if (trace)
            std::fprintf(stderr, "SPECKLE_TRACE get-buffers failed\n");
        return;
    }
    CUCOMPLEX *device_field = output ? cuda_data::get_ff0(grid_size) : buffers.field;
    if (!device_field)
        return;
    cufftHandle plan = get_speckle_fft_plan(*nfft);
    if (!plan)
    {
        if (trace)
            std::fprintf(stderr, "SPECKLE_TRACE get-fft-plan failed nfft=%d\n", *nfft);
        return;
    }

    using cuda_utils::TransferCategory;
    cudaError_t status = cudaSuccess;
    const CUCOMPLEX* device_phase_positive = phases_are_device
        ? phase_positive_device : buffers.phase_positive;
    const CUCOMPLEX* device_phase_negative = phases_are_device
        ? phase_negative_device : buffers.phase_negative;
    if (!phases_are_device)
        status = cuda_data::copy_host_complex_to_device(buffers.phase_positive,
            phase_positive_cpu, plane_size, TransferCategory::IncidentField);
    if (status == cudaSuccess && *bidirectional == 1 && !phases_are_device)
        status = cuda_data::copy_host_complex_to_device(buffers.phase_negative,
            phase_negative_cpu, plane_size, TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = cudaMemset(device_field, 0, grid_size * 3 * sizeof(CUCOMPLEX));
    if (status == cudaSuccess)
        status = speckle_flux_normalization_wrapper(
            buffers.normalization, *nfft, *k0, deltak, kmax,
            *psi_degrees * CU_REAL(3.1415926535897932384626433832795) / CU_REAL(180), *e0, *power,
            *bidirectional == 1 ? CU_REAL(2) : CU_REAL(1));
    if (trace)
        std::fprintf(stderr, "SPECKLE_TRACE setup status=%d plan=%d\n", int(status), int(plan));

    const CU_REAL psi = *psi_degrees * CU_REAL(3.1415926535897932384626433832795) / CU_REAL(180);
    const CU_REAL first_x = xs ? xs[0] : *grid_origin_x;
    const CU_REAL first_y = ys ? ys[0] : *grid_origin_y;
    const CU_REAL translated_x0 = *xx0 - (first_x + CU_REAL(*nfft / 2) * *spacing);
    const CU_REAL translated_y0 = *yy0 - (first_y + CU_REAL(*nfft / 2) * *spacing);
    const CU_REAL factor = deltak * deltak;
    for (int z = 0; status == cudaSuccess && z < *nz; ++z) {
        const CU_REAL z_plane = zs ? zs[size_t(z) * grid_xy] :
            *grid_origin_z + CU_REAL(z) * *spacing;
        {
            status = speckle_spectrum_wrapper(device_phase_positive, buffers.spectrum, *nfft, *k0,
                                              deltak, kmax, psi, *e0, translated_x0, translated_y0,
                                              *zz0, z_plane, 1);
            if (trace)
                std::fprintf(stderr, "SPECKLE_TRACE z=%d positive-spectrum=%d\n", z, int(status));
            if (status == cudaSuccess) {
                const cufftResult fft_status =
                    CUFFT_EXECUTE(plan, buffers.spectrum, buffers.spectrum, CUFFT_INVERSE);
                if (fft_status != CUFFT_SUCCESS)
                    status = cudaErrorUnknown;
                if (trace)
                    std::fprintf(stderr, "SPECKLE_TRACE z=%d positive-fft=%d\n", z,
                                 int(fft_status));
            }
            if (status == cudaSuccess)
                status = speckle_extract_plane_wrapper(buffers.spectrum, device_field, *nx, *ny,
                                                       *nfft, z, factor, 0);
            if (trace)
                std::fprintf(stderr, "SPECKLE_TRACE z=%d positive-gather=%d\n", z, int(status));
        }
        if (status == cudaSuccess && *bidirectional) {
            status = speckle_spectrum_wrapper(device_phase_negative, buffers.spectrum, *nfft, *k0,
                                              deltak, kmax, psi, *e0, translated_x0, translated_y0,
                                              *zz0, z_plane, -1);
            if (trace)
                std::fprintf(stderr, "SPECKLE_TRACE z=%d negative-spectrum=%d\n", z, int(status));
            if (status == cudaSuccess) {
                const cufftResult fft_status =
                    CUFFT_EXECUTE(plan, buffers.spectrum, buffers.spectrum, CUFFT_INVERSE);
                if (fft_status != CUFFT_SUCCESS)
                    status = cudaErrorUnknown;
                if (trace)
                    std::fprintf(stderr, "SPECKLE_TRACE z=%d negative-fft=%d\n", z,
                                 int(fft_status));
            }
            if (status == cudaSuccess)
                status = speckle_extract_plane_wrapper(buffers.spectrum, device_field, *nx, *ny,
                                                       *nfft, z, factor, 1);
            if (trace)
                std::fprintf(stderr, "SPECKLE_TRACE z=%d negative-gather=%d\n", z, int(status));
        }
    }
    if (status == cudaSuccess)
        status =
            speckle_scale_field_wrapper(device_field, int(grid_size * 3), buffers.normalization);
    CU_REAL normalization = CU_REAL(0);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(&normalization, buffers.normalization,
                                                sizeof(CU_REAL), TransferCategory::Result);
    if (status == cudaSuccess) {
        *e0 *= normalization;
    }
    if (status == cudaSuccess && resident_output)
        *resident_output = device_field;
    else if (status == cudaSuccess && output)
        status = cuda_data::copy_device_complex_to_host(
            output, device_field, grid_size * 3,
            TransferCategory::IncidentField);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA speckle illumination failed: %s\n", cudaGetErrorString(status));
        return;
    }
    *irradiance = CU_REAL(0);
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_speckle_field_cuda_(
    const int *nx, const int *ny, const int *nz, const int *dipole_count, const int *nfft,
    const REAL *k0, const REAL *xx0, const REAL *yy0, const REAL *zz0,
    const REAL *psi_degrees, const REAL *spacing, const REAL *power,
    const REAL *aperture, COMPLEX *e0, const COMPLEX *phase_positive,
    const COMPLEX *phase_negative, const REAL *xs, const REAL *ys, const REAL *zs,
    const int *bidirectional, COMPLEX *output, REAL *irradiance, int *status_out)
{
    if (!nfft || *nfft <= 0 || !dipole_count || *dipole_count < 0 || !e0 ||
        !phase_positive || !output ||
        size_t(*nfft) > std::numeric_limits<size_t>::max() / size_t(*nfft)) {
        if (status_out) *status_out = 1;
        return;
    }
    ifdda_speckle_field_cuda_impl(nx, ny, nz, dipole_count, nfft, k0, xx0, yy0, zz0,
        psi_degrees, spacing, power, aperture, e0,
        phase_positive, phase_negative,
        xs, ys, zs, bidirectional, output, irradiance, status_out);
}

//-----------------------------------------------------------
extern "C" void ifdda_speckle_field_cuda_resident_grid_origin_(
    const int *nx, const int *ny, const int *nz, const int *dipole_count,
    const int *nfft, const CU_REAL *k0, const CU_REAL *xx0, const CU_REAL *yy0,
    const CU_REAL *zz0, const CU_REAL *psi_degrees, const CU_REAL *spacing,
    const CU_REAL *power, const CU_REAL *aperture, COMPLEX *e0,
    const CUCOMPLEX *phase_positive, const CUCOMPLEX *phase_negative,
    const CU_REAL *grid_origin_x, const CU_REAL *grid_origin_y,
    const CU_REAL *grid_origin_z, const int *bidirectional,
    CUCOMPLEX **output_device, CU_REAL *irradiance, int *status_out) {
    if (!output_device) {
        if (status_out)
            *status_out = 1;
        return;
    }
    ifdda_speckle_field_cuda_impl(
        nx, ny, nz, dipole_count, nfft, k0, xx0, yy0, zz0, psi_degrees,
        spacing, power, aperture, e0, nullptr, nullptr,
        nullptr, nullptr, nullptr, bidirectional, nullptr, irradiance,
        status_out, output_device, grid_origin_x, grid_origin_y,
        grid_origin_z, phase_positive, phase_negative);
}
