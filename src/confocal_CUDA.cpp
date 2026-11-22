#include "cuda_data.h"
#include "cuda_utils.h"
#define IFDDA_CUDA_CALLERS_CALCULWIDEFIELD_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_CALCULWIDEFIELD_CUDA
#include "kernel.h"

#include <complex>
#include <cstdio>
#include <limits>
#include <vector>

//-----------------------------------------------------------
static void ifdda_confocal_field_cuda_impl(
    const int *nx, const int *ny, const int *nz, const int *dipole_count, const int *nfft,
    const CU_REAL *k0, const CU_REAL *xx0, const CU_REAL *yy0, const CU_REAL *zz0,
    const CU_REAL *psi_degrees, const CU_REAL *spacing, const CU_REAL *power,
    const CU_REAL *aperture, COMPLEX *e0, const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    const int *bidirectional, COMPLEX *output, CUCOMPLEX **resident_output,
    CU_REAL *irradiance, int *status_out, const CU_REAL *grid_origin_x = nullptr,
    const CU_REAL *grid_origin_y = nullptr, const CU_REAL *grid_origin_z = nullptr) {
    if (resident_output)
        *resident_output = nullptr;
    if (status_out)
        *status_out = 1;
    if (!nx || !ny || !nz || !dipole_count || !nfft || !k0 || !xx0 || !yy0 || !zz0 ||
        !psi_degrees || !spacing || !power || !aperture || !e0 ||
        (!xs && !grid_origin_x) || (!ys && !grid_origin_y) ||
        (!zs && !grid_origin_z) ||
        !bidirectional || (!output && !resident_output) || !irradiance ||
        !status_out || *nx <= 0 || *ny <= 0 ||
        *nz <= 0 || *dipole_count <= 0 || *nfft <= 0 || *k0 <= CU_REAL(0) ||
        *spacing <= CU_REAL(0) || *power <= CU_REAL(0) || *aperture <= CU_REAL(0) ||
        (*bidirectional != 0 && *bidirectional != 1))
        return;

    size_t grid_xy = 0, grid_size = 0, plane_size = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx), size_t(*ny), &grid_xy) ||
        !cuda_utils::checked_multiply(grid_xy, size_t(*nz), &grid_size) ||
        !cuda_utils::checked_multiply(size_t(*nfft), size_t(*nfft), &plane_size) ||
        grid_size != size_t(*dipole_count) || grid_size > size_t(INT_MAX) / 3 ||
        plane_size > size_t(INT_MAX) || grid_xy > size_t(INT_MAX)) {
        *status_out = 5;
        return;
    }
    const CU_REAL deltak =
        CU_REAL(2) * CU_REAL(3.1415926535897932384626433832795) / (CU_REAL(*nfft) * *spacing);
    const CU_REAL kmax = *k0 * *aperture;
    if (deltak >= kmax / CU_REAL(2)) {
        *status_out = 2;
        return;
    }
    if (*nx >= *nfft) {
        *status_out = 3;
        return;
    }
    if (*ny >= *nfft) {
        *status_out = 4;
        return;
    }
    if (!initialize_cuda_device())
        return;

    cuda_data::SpectralFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_spectral_field_buffers(plane_size, grid_size, &buffers))
        return;
    CUCOMPLEX *device_field = output ? cuda_data::get_ff0(grid_size) : buffers.field;
    if (!device_field)
        return;
    const cufftHandle plan = get_speckle_fft_plan(*nfft);
    if (!plan)
        return;

    using cuda_utils::TransferCategory;
    cudaError_t status = cudaMemset(device_field, 0,
                                    grid_size * 3 * sizeof(CUCOMPLEX));
    const CU_REAL psi = *psi_degrees * CU_REAL(3.1415926535897932384626433832795) / CU_REAL(180);
    if (status == cudaSuccess)
        status = speckle_flux_normalization_wrapper(buffers.normalization, *nfft, *k0, deltak, kmax,
                                                    psi, *e0, *power,
                                                    *bidirectional ? CU_REAL(2) : CU_REAL(1));

    const CU_REAL first_x = xs ? xs[0] : *grid_origin_x;
    const CU_REAL first_y = ys ? ys[0] : *grid_origin_y;
    const CU_REAL x0 = *xx0 - (first_x + CU_REAL(*nfft / 2) * *spacing);
    const CU_REAL y0 = *yy0 - (first_y + CU_REAL(*nfft / 2) * *spacing);
    const CU_REAL factor = deltak * deltak;
    for (int z = 0; status == cudaSuccess && z < *nz; ++z) {
        const CU_REAL z_plane = zs ? zs[size_t(z) * grid_xy] :
            *grid_origin_z + CU_REAL(z) * *spacing;
        status = speckle_spectrum_wrapper(nullptr, buffers.spectrum, *nfft, *k0, deltak, kmax, psi,
                                          *e0, x0, y0, *zz0, z_plane, 1);
        if (status == cudaSuccess) {
            const cufftResult fft_status =
                CUFFT_EXECUTE(plan, buffers.spectrum, buffers.spectrum, CUFFT_INVERSE);
            if (fft_status != CUFFT_SUCCESS)
                status = cudaErrorUnknown;
        }
        if (status == cudaSuccess)
        status = speckle_extract_plane_wrapper(buffers.spectrum, device_field, *nx, *ny, *nfft,
                                                   z, factor, 0);

        if (status == cudaSuccess && *bidirectional) {
            status = speckle_spectrum_wrapper(nullptr, buffers.spectrum, *nfft, *k0, deltak, kmax,
                                              psi, *e0, x0, y0, *zz0, z_plane, -1);
            if (status == cudaSuccess) {
                const cufftResult fft_status =
                    CUFFT_EXECUTE(plan, buffers.spectrum, buffers.spectrum, CUFFT_INVERSE);
                if (fft_status != CUFFT_SUCCESS)
                    status = cudaErrorUnknown;
            }
            if (status == cudaSuccess)
                status = speckle_extract_plane_wrapper(buffers.spectrum, device_field, *nx, *ny,
                                                       *nfft, z, factor, 1);
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
        if (resident_output)
            *resident_output = device_field;
        else
            status = cuda_data::copy_device_complex_to_host(
                output, device_field, grid_size * 3,
                TransferCategory::IncidentField);
    }
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA confocal illumination failed: %s\n", cudaGetErrorString(status));
        return;
    }
    *irradiance = *power / CU_REAL(3.1415926535897932384626433832795) * *k0 * *k0;
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_confocal_field_cuda_(
    const int *nx, const int *ny, const int *nz, const int *dipole_count, const int *nfft,
    const REAL *k0, const REAL *xx0, const REAL *yy0, const REAL *zz0,
    const REAL *psi_degrees, const REAL *spacing, const REAL *power,
    const REAL *aperture, COMPLEX *e0, const REAL *xs, const REAL *ys,
    const REAL *zs, const int *bidirectional, COMPLEX *output,
    REAL *irradiance, int *status_out) {
    if (!dipole_count || *dipole_count < 0 || !e0 || !output) {
        if (status_out) *status_out = 1;
        return;
    }
    ifdda_confocal_field_cuda_impl(
        nx, ny, nz, dipole_count, nfft, k0, xx0, yy0, zz0, psi_degrees,
        spacing, power, aperture, e0, xs, ys, zs, bidirectional, output,
        nullptr, irradiance, status_out);
}

//-----------------------------------------------------------
extern "C" void ifdda_confocal_field_cuda_resident_grid_origin_(
    const int *nx, const int *ny, const int *nz, const int *dipole_count,
    const int *nfft, const REAL *k0, const REAL *xx0, const REAL *yy0,
    const REAL *zz0, const REAL *psi_degrees, const REAL *spacing,
    const REAL *power, const REAL *aperture, COMPLEX *e0,
    const REAL *grid_origin_x, const REAL *grid_origin_y,
    const REAL *grid_origin_z, const int *bidirectional,
    CUCOMPLEX **output_device, REAL *irradiance, int *status_out) {
    if (!output_device) {
        if (status_out)
            *status_out = 1;
        return;
    }
    ifdda_confocal_field_cuda_impl(
        nx, ny, nz, dipole_count, nfft, k0, xx0, yy0, zz0, psi_degrees,
        spacing, power, aperture, e0, nullptr, nullptr, nullptr, bidirectional,
        nullptr, output_device, irradiance, status_out, grid_origin_x,
        grid_origin_y, grid_origin_z);
}
