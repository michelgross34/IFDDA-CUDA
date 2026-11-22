#include "cuda_data.h"
#include "cuda_utils.h"
#include "cuda_interfaces.h"
#include "kernel.h"

#include <climits>
#include <cstdio>
#include <limits>

namespace {
//-----------------------------------------------------------
bool derive_field_cuda(const int* nx, const int* ny, const int* nz, const int* nmax,
                       const REAL* spacing, const COMPLEX* field, const int* axis,
                       COMPLEX* derivative) {
    if (!nx || !ny || !nz || !nmax || !spacing || !field || !axis || !derivative ||
        *nx < 5 || *ny < 5 || *nz < 5 || *nmax <= 0 || *axis < 1 || *axis > 3 ||
        *spacing <= CU_REAL(0))
        return false;

    const size_t nx_size = size_t(*nx);
    const size_t ny_size = size_t(*ny);
    const size_t nz_size = size_t(*nz);
    if (nx_size > std::numeric_limits<size_t>::max() / ny_size)
        return false;
    const size_t plane_size = nx_size * ny_size;
    if (plane_size > std::numeric_limits<size_t>::max() / nz_size)
        return false;
    const size_t site_count = plane_size * nz_size;
    if (site_count > size_t(*nmax) || site_count > size_t(INT_MAX / 3))
        return false;
    if (!initialize_cuda_device())
        return false;

    cuda_data::OpticalForceBufferRequest request{};
    request.dipole_count = site_count;
    request.mesh_size = site_count;
    request.incident_derivative = true;
    cuda_data::OpticalForceDeviceBuffers buffers{};
    if (!cuda_data::get_device_optical_force_buffers(request, &buffers))
        return false;

    // The original caller computes the x, y, and z derivatives in sequence.
    // Keep the incident grid resident across that sequence and refresh it when
    // the host source or grid dimensions change.
    static const COMPLEX* cached_host_field = nullptr;
    static int cached_nx = 0;
    static int cached_ny = 0;
    static int cached_nz = 0;
    static bool cache_ready = false;
    const bool refresh = *axis == 1 || !cache_ready || cached_host_field != field ||
                         cached_nx != *nx || cached_ny != *ny || cached_nz != *nz;
    if (refresh) {
        const cudaError_t copy_status = cuda_data::copy_host_complex_to_device(
            buffers.incident_grid, field, site_count * 3,
            cuda_utils::TransferCategory::Other);
        if (!cuda_utils::check_cuda(copy_status, "Fortran derivechamp CUDA",
                                    "upload incident field"))
            return false;
        cached_host_field = field;
        cached_nx = *nx;
        cached_ny = *ny;
        cached_nz = *nz;
        cache_ready = true;
    }

    if (!cuda_utils::check_cuda(
            optical_field_derivative_wrapper(buffers.incident_grid,
                                             buffers.incident_derivative, *nx, *ny, *nz,
                                             *axis - 1, *spacing),
            "Fortran derivechamp CUDA", "differentiate field") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_complex_to_host(derivative, buffers.incident_derivative,
                                           site_count * 3,
                                           cuda_utils::TransferCategory::Result),
            "Fortran derivechamp CUDA", "download derivative"))
        return false;
    return true;
}

//-----------------------------------------------------------
bool derive_field_device_cuda(const int* nx, const int* ny, const int* nz,
                              const int* nmax, const CU_REAL* spacing,
                              const CUCOMPLEX* field, const int* axis,
                              CUCOMPLEX* derivative) {
    if (!nx || !ny || !nz || !nmax || !spacing || !field || !axis || !derivative ||
        *nx < 5 || *ny < 5 || *nz < 5 || *nmax <= 0 || *axis < 1 || *axis > 3 ||
        *spacing <= CU_REAL(0) || !initialize_cuda_device())
        return false;
    return cuda_utils::check_cuda(
        optical_field_derivative_wrapper(field, derivative, *nx, *ny, *nz,
                                         *axis - 1, *spacing),
        "incident field derivative CUDA", "differentiate resident field");
}
} // namespace

//-----------------------------------------------------------
extern "C" void derivechamp_(const int* nx, const int* ny, const int* nz, const int* nmax,
                             const REAL* spacing, const COMPLEX* field, const int* axis,
                             COMPLEX* derivative) {
    if (!derive_field_cuda(nx, ny, nz, nmax, spacing, field, axis, derivative))
        std::fprintf(stderr, "derivechamp_: CUDA derivative failed (axis=%d)\n",
                     axis ? *axis : -1);
}

//-----------------------------------------------------------
extern "C" void ifdda_derive_incident_field_cuda_(
    const CUCOMPLEX* field, const int* nx, const int* ny, const int* nz, const int* nmax,
    const CU_REAL* spacing, const int* axis, CUCOMPLEX* derivative, int* status_out) {
    if (status_out)
        *status_out = 1;
    if (!status_out || !derive_field_device_cuda(nx, ny, nz, nmax, spacing,
                                                 field, axis, derivative))
        return;
    *status_out = 0;
}
