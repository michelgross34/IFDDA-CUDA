#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cstdio>
#include <limits>

//-----------------------------------------------------------
extern "C" void ifdda_rytov_vector_field_cuda_(const CUCOMPLEX *eref, const CUCOMPLEX *delta,
                                               CUCOMPLEX *result, const int *dipole_count,
                                               int *nstop) {
    if (!nstop)
        return;
    *nstop = 1;
    if (!eref || !delta || !result || !dipole_count || *dipole_count <= 0) {
        std::fprintf(stderr, "Rytov CUDA: invalid input\n");
        return;
    }
    if (!initialize_cuda_device()) {
        std::fprintf(stderr, "Rytov CUDA: CUDA device initialization failed\n");
        return;
    }

    const size_t dipoles = static_cast<size_t>(*dipole_count);
    if (dipoles > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX))) {
        std::fprintf(stderr, "Rytov CUDA: vector size overflow\n");
        return;
    }
    const size_t bytes = dipoles * 3 * sizeof(CUCOMPLEX);
    const size_t vector_count = dipoles * 3;
    cuda_data::RytovFieldDeviceBuffers buffers{};
    if (!cuda_data::get_device_rytov_field_buffers(vector_count, &buffers)) {
        std::fprintf(stderr, "Rytov CUDA: reusable device workspace unavailable\n");
        return;
    }
    CUCOMPLEX *device_eref = buffers.reference;
    CUCOMPLEX *device_delta = buffers.delta;
    CUCOMPLEX *device_result = buffers.result;
    CU_INT *device_error = buffers.error;
    if (!device_eref || !device_delta || !device_result || !device_error) {
        std::fprintf(stderr, "Rytov CUDA: reusable device workspace unavailable\n");
        return;
    }

    cudaError_t status = cuda_data::copy_host_to_device(device_eref, eref, bytes,
                                                        cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(device_delta, delta, bytes,
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess)
        status = cudaMemset(device_error, 0, sizeof(int));
    if (status == cudaSuccess)
        status = rytov_vector_field_wrapper(device_eref, device_delta, device_result, device_error,
                                            *dipole_count);
    int device_error_value = 1;
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(&device_error_value, device_error, sizeof(int),
                                                cuda_utils::TransferCategory::Other);
    if (status == cudaSuccess && device_error_value == 0)
        status = cuda_data::copy_device_to_host(result, device_result, bytes,
                                                cuda_utils::TransferCategory::Other);

    if (status != cudaSuccess || device_error_value != 0) {
        std::fprintf(stderr, "Rytov CUDA: %s\n",
                     status == cudaSuccess ? "non-finite field detected"
                                           : cudaGetErrorString(status));
        return;
    }
    *nstop = 0;
}

//-----------------------------------------------------------
// Fortran entry point. Keep the historical ABI while routing the operation
// through the persistent CUDA buffers and the kernel wrapper above.
extern "C" void rytov_vector_field_cuda_(const CUCOMPLEX *eref, const CUCOMPLEX *delta,
                                    CUCOMPLEX *result, const int *dipole_count,
                                    int *nstop) {
    ifdda_rytov_vector_field_cuda_(eref, delta, result, dipole_count, nstop);
}
