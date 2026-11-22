#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"
#include "cuda_interfaces.h"

#include <algorithm>
#include <cstddef>
#include <mutex>

namespace {
std::mutex polarizability_mutex;

//-----------------------------------------------------------
CUCOMPLEX make_cuda_complex(CU_REAL real, CU_REAL imag) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real, imag);
#else
    return make_cuDoubleComplex(real, imag);
#endif
}

//-----------------------------------------------------------
int polarizability_method_code(const char method[2]) {
    if (method[0] == 'C' && method[1] == 'M') return 1;
    if (method[0] == 'R' && method[1] == 'R') return 2;
    if (method[0] == 'P' && method[1] == 'S') return 3;
    if (method[0] == 'G' && method[1] == 'B') return 4;
    if (method[0] == 'L' && method[1] == 'A') return 5;
    if (method[0] == 'L' && method[1] == 'R') return 6;
    if (method[0] == 'L' && method[1] == 'S') return 7;
    if (method[0] == 'F' && method[1] == 'G') return 8;
    return 0;
}
} // namespace

//-----------------------------------------------------------
void poladiff_cuda(
    const CU_REAL* radius, const COMPLEX* epsilon,
    const COMPLEX* epsilon0, const CU_REAL* k0,
    const int* discretized, const char* method, CU_REAL* sldr,
    COMPLEX* polarizability, std::size_t) {
    if (!radius || !epsilon || !epsilon0 || !k0 || !discretized || !method ||
        !sldr || !polarizability) {
        return;
    }
    const int method_code = polarizability_method_code(method);
    if (!method_code || (*discretized != 0 && *discretized != 1) ||
        *radius <= CU_REAL(0) || *k0 <= CU_REAL(0) || !initialize_cuda_device()) {
        *polarizability = COMPLEX{};
        return;
    }
    std::lock_guard<std::mutex> lock(polarizability_mutex);

    cuda_data::ScalarPolarizabilityDeviceBuffers buffers{};
    if (!cuda_data::get_device_scalar_polarizability_buffers(&buffers)) {
        *polarizability = {CU_REAL(0), CU_REAL(0)};
        return;
    }
    CU_REAL sldr_value = *sldr;
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.sldr, &sldr_value, sizeof(sldr_value),
        cuda_utils::TransferCategory::Material);
    if (status == cudaSuccess) {
        status = scalar_polarizability_wrapper(
            buffers.value, buffers.sldr, *radius,
            make_cuda_complex(epsilon->real(), epsilon->imag()),
            make_cuda_complex(epsilon0->real(), epsilon0->imag()), *k0,
            *discretized, method_code);
    }
    CUCOMPLEX result{};
    if (status == cudaSuccess) {
        status = cuda_data::copy_device_to_host(
            &result, buffers.value, sizeof(result),
            cuda_utils::TransferCategory::Material);
    }
    if (status == cudaSuccess && method_code == 6) {
        status = cuda_data::copy_device_to_host(
            &sldr_value, buffers.sldr, sizeof(sldr_value),
            cuda_utils::TransferCategory::Material);
    }
    if (status != cudaSuccess) {
        *polarizability = COMPLEX{};
        return;
    }
    *polarizability = COMPLEX(result.x, result.y);
    *sldr = sldr_value;
}

//-----------------------------------------------------------
void polaepstens_cuda(
    const CU_REAL* radius, const COMPLEX* epsilon,
    const COMPLEX* epsilon0, const CU_REAL* k0,
    const int* discretized, const char* method, const int* invert_result,
    COMPLEX* polarizability, std::size_t) {
    if (!radius || !epsilon || !epsilon0 || !k0 || !discretized || !method ||
        !invert_result || !polarizability) {
        return;
    }
    std::fill(polarizability, polarizability + 9, COMPLEX{});
    const int method_code = method[0] == 'C' && method[1] == 'M'
                                ? 1
                                : (method[0] == 'R' && method[1] == 'R' ? 2 : 0);
    if (!method_code || (*discretized != 0 && *discretized != 1) ||
        (*invert_result != 1 && *invert_result != -1) ||
        *radius <= CU_REAL(0) || *k0 <= CU_REAL(0) || !initialize_cuda_device()) {
        return;
    }
    std::lock_guard<std::mutex> lock(polarizability_mutex);
    cuda_data::TensorPolarizabilityDeviceBuffers buffers{};
    if (!cuda_data::get_device_tensor_polarizability_buffers(&buffers))
        return;
    CUCOMPLEX input[9];
    for (int i = 0; i < 9; ++i)
        input[i] = make_cuda_complex(epsilon[i].real(), epsilon[i].imag());
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.epsilon, input, sizeof(input), cuda_utils::TransferCategory::Material);
    if (status == cudaSuccess) {
        status = tensor_polarizability_wrapper(
            buffers.epsilon, buffers.value, buffers.status,
            make_cuda_complex(epsilon0->real(), epsilon0->imag()), *radius, *k0,
            *discretized, method_code, *invert_result);
    }
    CUCOMPLEX output[9]{};
    int kernel_status = 0;
    if (status == cudaSuccess) {
        status = cuda_data::copy_device_to_host(
            output, buffers.value, sizeof(output), cuda_utils::TransferCategory::Material);
    }
    if (status == cudaSuccess) {
        status = cuda_data::copy_device_to_host(
            &kernel_status, buffers.status, sizeof(kernel_status),
            cuda_utils::TransferCategory::Material);
    }
    if (status != cudaSuccess || kernel_status != 0)
        return;
    for (int i = 0; i < 9; ++i) {
        polarizability[i] = COMPLEX(output[i].x, output[i].y);
    }
}

//-----------------------------------------------------------
extern "C" void ifdda_poladiff_cuda_abi_(
    const CU_REAL* radius, const IfddaCudaPolarizabilityComplex* epsilon,
    const IfddaCudaPolarizabilityComplex* epsilon0, const CU_REAL* k0,
    const int* discretized, const char* method, CU_REAL* sldr,
    IfddaCudaPolarizabilityComplex* polarizability, std::size_t method_length) {
    if (!radius || !epsilon || !epsilon0 || !k0 || !discretized || !method ||
        !sldr || !polarizability)
        return;
    const COMPLEX eps(epsilon->real, epsilon->imag);
    const COMPLEX eps0(epsilon0->real, epsilon0->imag);
    COMPLEX alpha{};
    poladiff_cuda(radius, &eps, &eps0, k0, discretized, method,
                  sldr, &alpha, method_length);
    polarizability->real = alpha.real();
    polarizability->imag = alpha.imag();
}

//-----------------------------------------------------------
extern "C" void ifdda_polaepstens_cuda_abi_(
    const CU_REAL* radius, const IfddaCudaPolarizabilityComplex* epsilon,
    const IfddaCudaPolarizabilityComplex* epsilon0, const CU_REAL* k0,
    const int* discretized, const char* method, const int* invert_result,
    IfddaCudaPolarizabilityComplex* polarizability, std::size_t method_length) {
    if (!radius || !epsilon || !epsilon0 || !k0 || !discretized || !method ||
        !invert_result || !polarizability)
        return;
    COMPLEX eps[9];
    for (int i = 0; i < 9; ++i)
        eps[i] = COMPLEX(epsilon[i].real, epsilon[i].imag);
    const COMPLEX eps0(epsilon0->real, epsilon0->imag);
    COMPLEX alpha[9]{};
    polaepstens_cuda(radius, eps, &eps0, k0, discretized, method,
                     invert_result, alpha, method_length);
    for (int i = 0; i < 9; ++i) {
        polarizability[i].real = alpha[i].real();
        polarizability[i].imag = alpha[i].imag();
    }
}

// gfortran calls these two symbols from cdmlib.f90 when the microscope
// recomputes polarizability. Keep the Fortran ABI without introducing a
// second polarizability implementation (or reinterpretation as a GPU pointer).
extern "C" void poladiff_(
    const REAL* radius, const COMPLEX* epsilon,
    const COMPLEX* epsilon0, const REAL* k0,
    const int* discretized, const char* method, REAL* sldr,
    COMPLEX* polarizability, std::size_t method_length) {
    poladiff_cuda(radius, epsilon, epsilon0, k0, discretized,
                  method, sldr, polarizability, method_length);
}

extern "C" void polaepstens_(
    const REAL* radius, const COMPLEX* epsilon,
    const COMPLEX* epsilon0, const REAL* k0,
    const int* discretized, const char* method, const int* invert_result,
    COMPLEX* polarizability, std::size_t method_length) {
    polaepstens_cuda(radius, epsilon, epsilon0, k0, discretized,
                     method, invert_result, polarizability, method_length);
}
