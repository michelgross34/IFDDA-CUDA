#include "cuda_data.h"
#define IFDDA_CUDA_CALLERS_PRODUITPOLAVECT_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_PRODUITPOLAVECT_CUDA
#include "cuda_utils.h"
#include "cdmlib_cpp_abi.hpp"
#include <cstdio>
#include <limits>
#include <vector>



//-----------------------------------------------------------
extern "C" void produitpolavect_(const COMPLEX *polarizability, const int *niso, const int *nmax,
                                 const int *ndipoles, const COMPLEX *input, COMPLEX *output) {
    if (!polarizability || !niso || !nmax || !ndipoles || !input || !output ||
        (*niso != 1 && *niso != 3) || *nmax <= 0 || *ndipoles <= 0 || *ndipoles > *nmax ||
        size_t(*ndipoles) > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        size_t(*nmax) > std::numeric_limits<size_t>::max() /
                            (size_t(*niso) * size_t(*niso) * sizeof(CUCOMPLEX))) {
        std::fprintf(stderr, "produitpolavect CUDA: invalid input\n");
        return;
    }
    if (!initialize_cuda_device())
        return;

    const size_t field_count = size_t(3) * size_t(*ndipoles);
    const size_t polarizability_count = size_t(*nmax) * size_t(*niso) * size_t(*niso);
    CUCOMPLEX *resident_local = nullptr;
    CUCOMPLEX *resident_polarizability = nullptr;
    const bool resident = ifdda_get_multilayer_resident_polarization_inputs_cuda(
                              input, polarizability, niso, nmax, ndipoles, &resident_local,
                              &resident_polarizability) != 0;
    CUCOMPLEX *device_input =
        resident ? get_device_auxiliary_vector(field_count) : get_device_vectors(field_count);
    CUCOMPLEX *device_polarizability =
        resident ? resident_polarizability
                 : copy_polarizability_to_device(polarizability, polarizability_count);
    if (!device_input || !device_polarizability)
        return;

    cudaError_t status =
        resident
            ? cudaMemcpy(device_input, resident_local, field_count * sizeof(CUCOMPLEX),
                         cudaMemcpyDeviceToDevice)
            : cuda_data::copy_host_to_device(device_input, input, field_count * sizeof(CUCOMPLEX),
                                             cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = polarization_inplace_wrapper(device_polarizability, device_input, *niso, *nmax,
                                              *ndipoles);
    if (status == cudaSuccess)
        status =
            cuda_data::copy_device_to_host(output, device_input, field_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Result);
    if (status != cudaSuccess)
        std::fprintf(stderr, "produitpolavect CUDA: %s\n", cudaGetErrorString(status));
}

//-----------------------------------------------------------
extern "C" int ifdda_produitpolavect_device_input_c99(
    const CUCOMPLEX* device_input, const COMPLEX* host_polarizability,
    const int* niso, const int* nmax, const int* ndipoles, COMPLEX* host_output) {
    if (!device_input || !host_polarizability || !niso || !nmax || !ndipoles ||
        !host_output || (*niso != 1 && *niso != 3) || *nmax <= 0 ||
        *ndipoles <= 0 || *ndipoles > *nmax)
        return 1;

    const size_t dipole_count = size_t(*ndipoles);
    const size_t isotropy = size_t(*niso);
    const size_t maximum = size_t(*nmax);
    if (dipole_count > std::numeric_limits<size_t>::max() / 3u ||
        maximum > std::numeric_limits<size_t>::max() / isotropy / isotropy)
        return 2;

    const size_t field_count = 3u * dipole_count;
    const size_t polarizability_count = maximum * isotropy * isotropy;
    try {
        std::vector<CUCOMPLEX> packed_polarizability(polarizability_count);
        std::vector<CUCOMPLEX> packed_output(field_count);
        for (size_t index = 0; index < polarizability_count; ++index) {
            packed_polarizability[index] = CUCOMPLEX{
                CU_REAL(std::real(host_polarizability[index])),
                CU_REAL(std::imag(host_polarizability[index]))};
        }

        cuda_data::DarkFieldMicroscopeDeviceBuffers buffers{};
        if (!cuda_data::get_device_dark_field_microscope_buffers(
                0, polarizability_count, 0, 0, 0, 0, field_count, 0, 0,
                &buffers) || !buffers.polarization || !buffers.solver_fields)
            return 3;

        cudaError_t status = cuda_data::copy_host_to_device(
            buffers.polarization, packed_polarizability.data(),
            polarizability_count * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::Material);
        if (status == cudaSuccess)
            status = cudaMemcpy(buffers.solver_fields, device_input,
                field_count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice);
        if (status == cudaSuccess)
            status = polarization_inplace_wrapper(buffers.polarization,
                buffers.solver_fields, *niso, *nmax, *ndipoles);
        if (status == cudaSuccess)
            status = cuda_data::copy_device_to_host(packed_output.data(),
                buffers.solver_fields, field_count * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::Result);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "produitpolavect CUDA device input: %s\n",
                cudaGetErrorString(status));
            return 4;
        }
        for (size_t index = 0; index < field_count; ++index)
            host_output[index] = COMPLEX(packed_output[index].x,
                                         packed_output[index].y);
        return 0;
    } catch (...) {
        return 5;
    }
}
