#include "cuda_utils.h"
#include "cuda_data.h"

#include <cstdio>
#include <limits>
#include <mutex>

namespace cuda_utils {

namespace {
cublasHandle_t cublas_handle = nullptr;
std::mutex cublas_handle_mutex;

struct TransferTotals {
    unsigned long long calls[3]{};
    unsigned long long bytes[3]{};
};

TransferTotals transfer_totals[static_cast<int>(TransferCategory::Count)]{};
std::mutex transfer_stats_mutex;
const char *const transfer_names[] = {
    "GreenTensor",    "SolverVector", "FftWorkspace", "IncidentField", "Material",  "Geometry",
    "Preconditioner", "Result",       "Control",      "Other",         "Diagnostic"};
} // namespace

//-----------------------------------------------------------
cudaError_t tracked_memcpy(void *destination, const void *source, size_t bytes,
                           cudaMemcpyKind direction, TransferCategory category) {
    (void)category;
    // Host transfers must pass through cuda_data so there is one auditable
    // place for transfer policy and accounting. This helper remains for D2D.
    if (direction != cudaMemcpyDeviceToDevice)
        return cudaErrorInvalidMemcpyDirection;
    return cudaMemcpy(destination, source, bytes, direction);
}

//-----------------------------------------------------------
void record_host_transfer(size_t bytes, TransferDirection direction, TransferCategory category) {
    if (bytes == 0 || direction == TransferDirection::DeviceToDevice)
        return;
    const int dir = direction == TransferDirection::HostToDevice ? 0 : 1;
    std::lock_guard<std::mutex> lock(transfer_stats_mutex);
    TransferTotals &totals = transfer_totals[static_cast<int>(category)];
    ++totals.calls[dir];
    totals.bytes[dir] += bytes;
}

//-----------------------------------------------------------
void reset_transfer_stats() {
    {
        std::lock_guard<std::mutex> lock(transfer_stats_mutex);
        for (TransferTotals &totals : transfer_totals)
            totals = TransferTotals{};
    }
    cuda_data::clear_device_geometry_coordinate_aliases();
}

//-----------------------------------------------------------
void print_transfer_stats() {
    std::lock_guard<std::mutex> lock(transfer_stats_mutex);
    std::fprintf(stderr, "CUDA_COPY_STATS_BEGIN\n");
    for (int category = 0; category < static_cast<int>(TransferCategory::Count); ++category) {
        const TransferTotals &totals = transfer_totals[category];
        const unsigned long long count = totals.calls[0] + totals.calls[1];
        if (!count)
            continue;
        std::fprintf(stderr, "CUDA_COPY_STATS %s H2D=%llu/%llu D2H=%llu/%llu\n",
                     transfer_names[category], totals.calls[0], totals.bytes[0], totals.calls[1],
                     totals.bytes[1]);
    }
    std::fprintf(stderr, "CUDA_COPY_STATS_END\n");
}

//-----------------------------------------------------------
extern "C" void ifdda_reset_cuda_transfer_stats() {
    reset_transfer_stats();
}

//-----------------------------------------------------------
extern "C" void ifdda_print_cuda_transfer_stats() {
    print_transfer_stats();
}

//-----------------------------------------------------------
cublasHandle_t get_cublas_handle() {
    std::lock_guard<std::mutex> lock(cublas_handle_mutex);
    if (cublas_handle)
        return cublas_handle;

    if (!initialize_cuda_device())
        return nullptr;

    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublas_handle = nullptr;
        std::fprintf(stderr, "get_cublas_handle: cublasCreate failed (status %d)\n",
                     static_cast<int>(status));
        return nullptr;
    }

    status = cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "get_cublas_handle: cublasSetPointerMode failed (status %d)\n",
                     static_cast<int>(status));
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
        return nullptr;
    }

    return cublas_handle;
}

//-----------------------------------------------------------
extern "C" void reset_after_device_reset() {
    std::lock_guard<std::mutex> lock(cublas_handle_mutex);
    cublas_handle = nullptr;
}

//-----------------------------------------------------------
bool checked_multiply(size_t a, size_t b, size_t *result) {
    if (!result || (b != 0 && a > std::numeric_limits<size_t>::max() / b))
        return false;
    *result = a * b;
    return true;
}

//-----------------------------------------------------------
bool check_cuda(cudaError_t status, const char *context, const char *operation) {
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s: %s failed: %s\n", context, operation, cudaGetErrorString(status));
    return false;
}

//-----------------------------------------------------------
bool check_cufft(cufftResult status, const char *context, const char *operation) {
    if (status == CUFFT_SUCCESS)
        return true;
    std::fprintf(stderr, "%s: %s failed (cuFFT status %d)\n", context, operation,
                 static_cast<int>(status));
    return false;
}

//-----------------------------------------------------------
bool check_cublas(cublasStatus_t status, const char *context, const char *operation) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr, "%s: %s failed (cuBLAS status %d)\n", context, operation,
                 static_cast<int>(status));
    return false;
}

//-----------------------------------------------------------
HostComplex host_value(CUCOMPLEX value) {
    return HostComplex(static_cast<CU_REAL>(value.x), static_cast<CU_REAL>(value.y));
}

//-----------------------------------------------------------
CUCOMPLEX device_value(HostComplex value) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(value.real(), value.imag());
#else
    return make_cuDoubleComplex(value.real(), value.imag());
#endif
}

//-----------------------------------------------------------
bool scalar_is_zero(CUCOMPLEX value) {
    return value.x == CU_REAL(0) && value.y == CU_REAL(0);
}

//-----------------------------------------------------------
bool dotu(cublasHandle_t handle, int n, const CUCOMPLEX *x, const CUCOMPLEX *y, CUCOMPLEX *result,
          const char *context) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return check_cublas(cublasCdotu(handle, n, x, 1, y, 1, result), context, "cublasCdotu");
#else
    return check_cublas(cublasZdotu(handle, n, x, 1, y, 1, result), context, "cublasZdotu");
#endif
}

//-----------------------------------------------------------
bool dotc(cublasHandle_t handle, int n, const CUCOMPLEX *x, const CUCOMPLEX *y, CUCOMPLEX *result,
          const char *context) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return check_cublas(cublasCdotc(handle, n, x, 1, y, 1, result), context, "cublasCdotc");
#else
    return check_cublas(cublasZdotc(handle, n, x, 1, y, 1, result), context, "cublasZdotc");
#endif
}

//-----------------------------------------------------------
bool norm2(cublasHandle_t handle, int n, const CUCOMPLEX *x, CU_REAL *result, const char *context) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return check_cublas(cublasScnrm2(handle, n, x, 1, result), context, "cublasScnrm2");
#else
    return check_cublas(cublasDznrm2(handle, n, x, 1, result), context, "cublasDznrm2");
#endif
}

//-----------------------------------------------------------
bool sum_nonnegative(cublasHandle_t handle, int n, const CU_REAL *values,
                     CU_REAL *result, const char *context) {
    if (!handle || n < 0 || !values || !result)
        return false;
    if (n == 0) {
        *result = CU_REAL(0);
        return true;
    }
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return check_cublas(cublasSasum(handle, n, values, 1, result), context, "cublasSasum");
#else
    return check_cublas(cublasDasum(handle, n, values, 1, result), context, "cublasDasum");
#endif
}

//-----------------------------------------------------------
bool copy(cublasHandle_t handle, int n, const CUCOMPLEX *source, CUCOMPLEX *destination,
          const char *context) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return check_cublas(cublasCcopy(handle, n, source, 1, destination, 1), context, "cublasCcopy");
#else
    return check_cublas(cublasZcopy(handle, n, source, 1, destination, 1), context, "cublasZcopy");
#endif
}

//-----------------------------------------------------------
bool axpy(cublasHandle_t handle, int n, CUCOMPLEX alpha, const CUCOMPLEX *x, CUCOMPLEX *y,
          const char *context) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return check_cublas(cublasCaxpy(handle, n, &alpha, x, 1, y, 1), context, "cublasCaxpy");
#else
    return check_cublas(cublasZaxpy(handle, n, &alpha, x, 1, y, 1), context, "cublasZaxpy");
#endif
}

//-----------------------------------------------------------
bool scale(cublasHandle_t handle, int n, CUCOMPLEX alpha, CUCOMPLEX *x, const char *context) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return check_cublas(cublasCscal(handle, n, &alpha, x, 1), context, "cublasCscal");
#else
    return check_cublas(cublasZscal(handle, n, &alpha, x, 1), context, "cublasZscal");
#endif
}

} // namespace cuda_utils
