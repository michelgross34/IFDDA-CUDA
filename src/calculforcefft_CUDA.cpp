#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <limits>

// Prepare the padded field spectrum on the GPU. The field may already have a
// device alias from the CUDA solver; otherwise it is uploaded from its CPU
// COMPLEX array. The transformed components are copied back because the
// surrounding legacy force routine still consumes vx/vy/vz on the host.
static bool force_fft(const int *nx, const int *ny, const int *nz,
                      const int *nx2, const int *ny2, const int *nz2,
                      const int *nxy2, const int *tabdip, COMPLEX *vx,
                      COMPLEX *vy, COMPLEX *vz, const COMPLEX *ff, bool masked) {
    if (!nx || !ny || !nz || !nx2 || !ny2 || !nz2 || !nxy2 || !vx || !vy || !vz || !ff ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 || *nx2 < *nx || *ny2 < *ny || *nz2 < *nz ||
        *nxy2 != *nx2 * *ny2 || (masked && !tabdip))
        return false;

    size_t dipole_count = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx), size_t(*ny), &dipole_count) ||
        !cuda_utils::checked_multiply(dipole_count, size_t(*nz), &dipole_count) ||
        dipole_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)))
        return false;
    // These are host outputs of padded size; initialize them so an error never
    // leaves stale spectra visible to the legacy caller.
    size_t padded_count = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx2), size_t(*ny2), &padded_count) ||
        !cuda_utils::checked_multiply(padded_count, size_t(*nz2), &padded_count))
        return false;
    std::fill_n(vx, padded_count, COMPLEX{});
    std::fill_n(vy, padded_count, COMPLEX{});
    std::fill_n(vz, padded_count, COMPLEX{});

    if (!cuda_utils::checked_multiply(size_t(*nx2), size_t(*ny2), &padded_count) ||
        !cuda_utils::checked_multiply(padded_count, size_t(*nz2), &padded_count) ||
        dipole_count > size_t(std::numeric_limits<int>::max()) ||
        padded_count > size_t(std::numeric_limits<int>::max()) ||
        padded_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;

    cuda_data::OpticalForceBufferRequest request{};
    request.dipole_count = dipole_count;
    request.mesh_size = dipole_count;
    request.padded_count = padded_count;
    request.force_fft = true;
    cuda_data::OpticalForceDeviceBuffers buffers{};
    if (!cuda_data::get_device_optical_force_buffers(request, &buffers))
        return false;

    const CUCOMPLEX *device_field = get_device_result_vector_alias(ff, 3 * dipole_count);
    if (!device_field) {
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(buffers.polarization, ff,
                    3 * dipole_count * sizeof(CUCOMPLEX),
                    cuda_utils::TransferCategory::SolverVector),
                "optical force FFT", "upload CPU field"))
            return false;
        device_field = buffers.polarization;
    }

    const int use_map = masked ? 1 : 0;
    CU_INT *device_map = masked ? get_device_tabdip(tabdip, dipole_count) : nullptr;
    if ((masked && !device_map) ||
        !cuda_utils::check_cuda(optical_force_gather_fft_wrapper(
                                    device_field, device_map, buffers.fft.vectors, *nx, *ny,
                                    *nz, *nx2, *ny2, *nz2, int(padded_count), use_map),
                                "optical force FFT", "pack field components"))
        return false;

    const cufftHandle plan = get_fft_plan(*nx2, *ny2, *nz2);
    if (!plan)
        return false;
    for (int component = 0; component < 3; ++component) {
        if (!cuda_utils::check_cufft(
                CUFFT_EXECUTE(plan, buffers.fft.vectors + size_t(component) * padded_count,
                              buffers.fft.vectors + size_t(component) * padded_count,
                              CUFFT_INVERSE),
                "optical force FFT", "transform field component"))
            return false;
    }

    const size_t component_bytes = padded_count * sizeof(CUCOMPLEX);
    return cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                      vx, buffers.fft.vectors, component_bytes,
                                      cuda_utils::TransferCategory::Result),
                                  "optical force FFT", "download x spectrum") &&
           cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                      vy, buffers.fft.vectors + padded_count, component_bytes,
                                      cuda_utils::TransferCategory::Result),
                                  "optical force FFT", "download y spectrum") &&
           cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                      vz, buffers.fft.vectors + 2 * padded_count, component_bytes,
                                      cuda_utils::TransferCategory::Result),
                                  "optical force FFT", "download z spectrum");
}

extern "C" void calculforcefft_(const int *nx, const int *ny, const int *nz,
                                const int *nx2, const int *ny2, const int *nz2,
                                const int *nxy2, const int *, const int *, const int *,
                                const int *tabdip, COMPLEX *vx, COMPLEX *vy, COMPLEX *vz,
                                const COMPLEX *ff, const int64_t *) {
    if (!force_fft(nx, ny, nz, nx2, ny2, nz2, nxy2, tabdip, vx, vy, vz, ff, true))
        std::fprintf(stderr, "calculforcefft_: CUDA/cuFFT transform failed\n");
}

extern "C" void calculforcefftopt_(const int *nx, const int *ny, const int *nz,
                                   const int *nx2, const int *ny2, const int *nz2,
                                   const int *nxy2, const int *, const int *, const int *,
                                   COMPLEX *vx, COMPLEX *vy, COMPLEX *vz,
                                   const COMPLEX *ff, const int64_t *) {
    if (!force_fft(nx, ny, nz, nx2, ny2, nz2, nxy2, nullptr, vx, vy, vz, ff, false))
        std::fprintf(stderr, "calculforcefftopt_: CUDA/cuFFT transform failed\n");
}
