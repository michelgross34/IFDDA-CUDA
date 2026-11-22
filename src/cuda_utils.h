#ifndef IFDDA_CUDA_UTILS_H
#define IFDDA_CUDA_UTILS_H

#include "kernel.h"

#include <complex>
#include <cstddef>
#include <cublas_v2.h>

namespace cuda_utils {

using HostComplex = std::complex<CU_REAL>;

// cuBLAS uses int* for GPU pivot arrays. Expose the first .dat member of
// a contiguous CU_INT allocation only at this external-library boundary.
// This accepts no CPU int* and performs no pointer reinterpretation.
inline int* device_integer_library_view(CU_INT* device) {
    return device ? &device->dat : nullptr;
}
inline const int* device_integer_library_view(const CU_INT* device) {
    return device ? &device->dat : nullptr;
}

enum class TransferCategory : int {
    GreenTensor,
    SolverVector,
    FftWorkspace,
    IncidentField,
    Material,
    Geometry,
    Preconditioner,
    Result,
    Control,
    Other,
    Diagnostic,
    Count
};
enum class TransferDirection : int {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

cudaError_t tracked_memcpy(void* destination,
                           const void* source,
                           size_t bytes,
                           cudaMemcpyKind direction,
                           TransferCategory category = TransferCategory::Other);
void record_host_transfer(size_t bytes, TransferDirection direction, TransferCategory category);
void reset_transfer_stats();
void print_transfer_stats();
extern "C" void ifdda_reset_cuda_transfer_stats();
extern "C" void ifdda_print_cuda_transfer_stats();

// Process-wide cuBLAS handle, initialized lazily and reused by CUDA solvers.
cublasHandle_t get_cublas_handle();
// Forget the cuBLAS handle after cudaDeviceReset invalidates its context.
extern "C" void reset_after_device_reset();

bool checked_multiply(size_t a, size_t b, size_t* result);
bool check_cuda(cudaError_t status, const char* context, const char* operation);
bool check_cufft(cufftResult status, const char* context, const char* operation);
bool check_cublas(cublasStatus_t status, const char* context, const char* operation);

HostComplex host_value(CUCOMPLEX value);
CUCOMPLEX device_value(HostComplex value);
bool scalar_is_zero(CUCOMPLEX value);

bool dotu(cublasHandle_t handle,
          int n,
          const CUCOMPLEX* x,
          const CUCOMPLEX* y,
          CUCOMPLEX* result,
          const char* context);
bool dotc(cublasHandle_t handle,
          int n,
          const CUCOMPLEX* x,
          const CUCOMPLEX* y,
          CUCOMPLEX* result,
          const char* context);
bool norm2(cublasHandle_t handle, int n, const CUCOMPLEX* x, CU_REAL* result, const char* context);
bool sum_nonnegative(
    cublasHandle_t handle, int n, const CU_REAL* values, CU_REAL* result, const char* context);
bool copy(cublasHandle_t handle,
          int n,
          const CUCOMPLEX* source,
          CUCOMPLEX* destination,
          const char* context);
bool axpy(cublasHandle_t handle,
          int n,
          CUCOMPLEX alpha,
          const CUCOMPLEX* x,
          CUCOMPLEX* y,
          const char* context);
bool scale(cublasHandle_t handle, int n, CUCOMPLEX alpha, CUCOMPLEX* x, const char* context);

} // namespace cuda_utils

#endif // IFDDA_CUDA_UTILS_H
