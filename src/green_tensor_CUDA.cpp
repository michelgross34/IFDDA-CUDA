#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"

#include <cstdio>
#include <limits>

//-----------------------------------------------------------
extern "C" void ifdda_generate_vector_green_cuda(
    COMPLEX *gxx, COMPLEX *gxy, COMPLEX *gxz, COMPLEX *gyy, COMPLEX *gyz, COMPLEX *gzz,
    const int *nx, const int *ny, const int *nz, const int *nxm, const int *nym, const int *nzm,
    const int *nx2, const int *ny2, const int *nz2, const REAL *aretecube, const REAL *k0,
    const COMPLEX *eps0, const int *medium, const int *nquad, const int *filtered, int *status) {
    if (status)
        *status = 1;
    if (!gxx || !gxy || !gxz || !gyy || !gyz || !gzz || !nx || !ny || !nz || !nxm || !nym || !nzm ||
        !nx2 || !ny2 || !nz2 || !aretecube || !k0 || !eps0 || !medium || !nquad || !filtered ||
        !status)
        return;
    if (*nquad < 0 || *nquad > 5 || *nx <= 0 || *ny <= 0 || *nz <= 0 ||
        *nxm < *nx || *nym < *ny || *nzm < *nz ||
        *nx2 < 2 * *nx || *ny2 < 2 * *ny || *nz2 < 2 * *nz)
        return;

    const size_t sx = size_t(*nxm) + 1, sy = size_t(*nym) + 1, sz = size_t(*nzm) + 1;
    size_t xy = 0, tensor_count = 0;
    if (!cuda_utils::checked_multiply(sx, sy, &xy) ||
        !cuda_utils::checked_multiply(xy, sz, &tensor_count) ||
        tensor_count > std::numeric_limits<size_t>::max() / (6 * sizeof(CUCOMPLEX))) {
        std::fprintf(stderr, "CUDA vector Green: tensor extent overflow\n");
        return;
    }

    size_t fft_xy = 0, fft_count = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx2), size_t(*ny2), &fft_xy) ||
        !cuda_utils::checked_multiply(fft_xy, size_t(*nz2), &fft_count) ||
        fft_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)))
        return;
    const void *keys[6] = {gxx, gxy, gxz, gyy, gyz, gzz};
    GreenTensorDevicePointers gpu_green = get_device_green_output_for_host_keys(keys, tensor_count);
    cuda_data::GreenTensorDeviceBuffers green_buffers{};
    if (!gpu_green ||
        !cuda_data::get_device_vector_green_buffers(tensor_count, fft_count, &green_buffers)) {
        (void)cuda_data::release_device_vector_green_buffers();
        std::fprintf(stderr, "CUDA vector Green: unable to allocate tensor or temporary buffers\n");
        return;
    }
    const cufftHandle fft_plan = get_fft_plan(*nx2, *ny2, *nz2);
    if (!fft_plan) {
        std::fprintf(stderr, "CUDA vector Green: unable to acquire persistent FFT plan\n");
        (void)cuda_data::release_device_vector_green_buffers();
        return;
    }
    const cudaError_t generated = generate_vector_green_tensor_wrapper(
        gpu_green, green_buffers.spatial, green_buffers.spectra, *nx, *ny, *nz, *nxm, *nym, *nzm,
        *nx2, *ny2, *nz2, fft_plan, *aretecube, *k0, *eps0, *medium, *nquad, *filtered);
    const bool scratch_released = cuda_data::release_device_vector_green_buffers();
    if (!cuda_utils::check_cuda(generated, "CUDA vector Green", "generate tensor") ||
        !scratch_released)
        return;

    cuda_data::mark_green_generated();

    // The host arrays are ABI keys passed to product wrappers; CUDA code uses
    // the persistent device tensor directly. No Green tensor H2D/D2H is done
    // here; temporary spatial/FFT allocations have already been released.
    *status = 0;
}
