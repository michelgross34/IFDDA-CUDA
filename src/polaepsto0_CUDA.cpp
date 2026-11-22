#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <climits>
#include <cstdio>
#include <limits>

//-----------------------------------------------------------
extern "C" void polaepsto0_(COMPLEX* polarizability, COMPLEX* epsilon,
                             int* dipole_map, const int* isotropy, const int* dipole_count) {
    if (!polarizability || !epsilon || !dipole_map || !isotropy || !dipole_count ||
        *dipole_count <= 0 || (*isotropy != 1 && *isotropy != 3) ||
        *dipole_count > INT_MAX / 3) {
        std::fprintf(stderr, "polaepsto0_: invalid arguments\n");
        return;
    }

    size_t tensor_count = 0;
    const size_t components = size_t(*isotropy) * size_t(*isotropy);
    if (!cuda_utils::checked_multiply(size_t(*dipole_count), components, &tensor_count) ||
        tensor_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX)) {
        std::fprintf(stderr, "polaepsto0_: tensor size overflow\n");
        return;
    }
    if (!initialize_cuda_device())
        return;

    cuda_data::PolarizabilityInitDeviceBuffers buffers{};
    if (!cuda_data::get_device_polarizability_init_buffers(
            size_t(*dipole_count), tensor_count, &buffers)) {
        std::fprintf(stderr, "polaepsto0_: could not acquire CUDA buffers\n");
        return;
    }
    if (!cuda_utils::check_cuda(
            initialize_polarizability_arrays_wrapper(buffers.dipole_map,
                                                     buffers.polarizability, buffers.epsilon,
                                                     *dipole_count, *isotropy),
            "polaepsto0 CUDA", "initialize material arrays"))
        return;

    const size_t dipole_bytes = size_t(*dipole_count) * sizeof(int);
    const size_t tensor_bytes = tensor_count * sizeof(CUCOMPLEX);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(dipole_map, buffers.dipole_map, dipole_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "polaepsto0 CUDA", "download dipole map") ||
        !cuda_utils::check_cuda(
        cuda_data::copy_device_complex_to_host(polarizability, buffers.polarizability, tensor_count,
                                           cuda_utils::TransferCategory::Material),
            "polaepsto0 CUDA", "download polarizability") ||
        !cuda_utils::check_cuda(
        cuda_data::copy_device_complex_to_host(epsilon, buffers.epsilon, tensor_count,
                                           cuda_utils::TransferCategory::Material),
            "polaepsto0 CUDA", "download permittivity"))
        return;
}
