

#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <cstddef>
#include <mutex>

namespace {
std::mutex gaussian_power_mutex;
}

//-----------------------------------------------------------
extern "C" void gaussianpuissancepara_(
    const CU_REAL* waist, const CU_REAL* power,
    CU_REAL* irradiance, COMPLEX* field) {
    if (!waist || !power || !irradiance || !field)
        return;
    *irradiance = CU_REAL(0);
    *field = COMPLEX(0, 0);
    if (*waist <= CU_REAL(0) || *power < CU_REAL(0) || !initialize_cuda_device())
        return;
    std::lock_guard<std::mutex> lock(gaussian_power_mutex);
    cuda_data::GaussianPowerDeviceBuffers buffers{};
    if (!cuda_data::get_device_gaussian_power_buffers(&buffers))
        return;
    cudaError_t status = gaussian_power_wrapper(
        buffers.field, buffers.irradiance, *waist, *power);
    CUCOMPLEX cuda_field{};
    CU_REAL cuda_irradiance = CU_REAL(0);
    if (status == cudaSuccess) {
        status = cuda_data::copy_device_to_host(
            &cuda_field, buffers.field, sizeof(cuda_field),
            cuda_utils::TransferCategory::IncidentField);
    }
    if (status == cudaSuccess) {
        status = cuda_data::copy_device_to_host(
            &cuda_irradiance, buffers.irradiance, sizeof(cuda_irradiance),
            cuda_utils::TransferCategory::IncidentField);
    }
    if (status != cudaSuccess)
        return;
    *field = COMPLEX(cuda_field.x, cuda_field.y);
    *irradiance = cuda_irradiance;
}

//-----------------------------------------------------------
extern "C" void gaussianpuissance_(
    CU_REAL* power, CU_REAL* irradiance, CU_REAL* waist, CU_REAL* k0,
    COMPLEX* field) {
    if (!power || !irradiance || !waist || !k0 || !field)
        return;
    *irradiance = CU_REAL(0);
    *field = COMPLEX(0, 0);
    if (*waist <= CU_REAL(0) || *k0 <= CU_REAL(0) ||
        *power < CU_REAL(0) || !initialize_cuda_device())
        return;
    std::lock_guard<std::mutex> lock(gaussian_power_mutex);
    cuda_data::GaussianPowerDeviceBuffers buffers{};
    if (!cuda_data::get_device_gaussian_power_buffers(&buffers))
        return;
    cudaError_t status = rigorous_gaussian_power_wrapper(
        buffers.field, buffers.irradiance, *waist, *k0, *power);
    CUCOMPLEX cuda_field{};
    CU_REAL cuda_irradiance = CU_REAL(0);
    if (status == cudaSuccess) {
        status = cuda_data::copy_device_to_host(
            &cuda_field, buffers.field, sizeof(cuda_field),
            cuda_utils::TransferCategory::IncidentField);
    }
    if (status == cudaSuccess) {
        status = cuda_data::copy_device_to_host(
            &cuda_irradiance, buffers.irradiance, sizeof(cuda_irradiance),
            cuda_utils::TransferCategory::IncidentField);
    }
    if (status != cudaSuccess)
        return;
    *field = COMPLEX(cuda_field.x, cuda_field.y);
    *irradiance = cuda_irradiance;
}
