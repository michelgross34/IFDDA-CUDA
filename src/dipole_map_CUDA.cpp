#include "dipole_map_CUDA.h"

#include "cuda_utils.h"
#include "kernel.h"

#include <chrono>
#include <climits>
#include <cstdio>
#include <limits>
#include <vector>

namespace cuda_geometry {

//--------------------------------------------
/*
bool build_dipole_map_from_inside(
    const int* device_inside_map,
    const int* host_inside_reference,
    size_t count,
    int* host_tabdip,
    int* host_occupied_count,
    bool compare_cpu,
    cuda_data::DipoleMapCompactionDeviceBuffers* device_buffers) {
    if (!device_inside_map || count == 0 || count > size_t(INT_MAX) ||
        !host_tabdip || !host_occupied_count || !device_buffers ||
        (compare_cpu && !host_inside_reference))
        return false;

    std::vector<int> cpu_tabdip;
    size_t cpu_occupied = 0;
    double cpu_milliseconds = 0.0;
    if (compare_cpu) {
        const auto cpu_start = std::chrono::steady_clock::now();
        cpu_tabdip.resize(count);
        for (size_t index = 0; index < count; ++index) {
            if (host_inside_reference[index] != 0)
                cpu_tabdip[index] = int(++cpu_occupied);
            else
                cpu_tabdip[index] = 0;
        }
        cpu_milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpu_start).count();
    }

    if (count > size_t(std::numeric_limits<int>::max()))
        return false;
    size_t temporary_bytes = 0;
    if (!cuda_utils::check_cuda(dipole_map_compaction_workspace_wrapper(
                                    int(count), &temporary_bytes),
                                "dipole map", "query CUB scan workspace") ||
        !cuda_data::get_device_dipole_map_compaction_buffers(
            count, temporary_bytes, device_buffers))
        return false;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (compare_cpu &&
        (cudaEventCreate(&start_event) != cudaSuccess ||
         cudaEventCreate(&stop_event) != cudaSuccess)) {
        if (start_event)
            cudaEventDestroy(start_event);
        if (stop_event)
            cudaEventDestroy(stop_event);
        return false;
    }
    if (compare_cpu && cudaEventRecord(start_event) != cudaSuccess)
        return false;

    const bool kernel_ok = cuda_utils::check_cuda(dipole_map_from_inside_wrapper(
        device_inside_map, int(count), device_buffers->prefix,
        device_buffers->tabdip, device_buffers->occupied_count,
        device_buffers->scan_temporary, device_buffers->scan_temporary_bytes),
        "dipole map", "scan occupancy and build indices");
    if (compare_cpu && kernel_ok && cudaEventRecord(stop_event) != cudaSuccess) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        return false;
    }
    if (!kernel_ok ||
        !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                    host_occupied_count, device_buffers->occupied_count,
                                    sizeof(*host_occupied_count),
                                    cuda_utils::TransferCategory::Control),
                                "dipole map", "download occupied count") ||
        !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                    host_tabdip, device_buffers->tabdip,
                                    count * sizeof(*host_tabdip),
                                    cuda_utils::TransferCategory::Geometry),
                                "dipole map", "publish host ABI map")) {
        if (start_event)
            cudaEventDestroy(start_event);
        if (stop_event)
            cudaEventDestroy(stop_event);
        return false;
    }

    bool matches_cpu = true;
    if (compare_cpu) {
        if (cudaEventSynchronize(stop_event) != cudaSuccess) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            return false;
        }
        float gpu_milliseconds = 0.0F;
        if (cudaEventElapsedTime(&gpu_milliseconds, start_event, stop_event) != cudaSuccess) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            return false;
        }
        matches_cpu = size_t(*host_occupied_count) == cpu_occupied;
        size_t mismatch_count = 0;
        size_t first_mismatch = count;
        for (size_t index = 0; index < count; ++index) {
            if (host_tabdip[index] != cpu_tabdip[index]) {
                ++mismatch_count;
                if (first_mismatch == count)
                    first_mismatch = index;
            }
        }
        matches_cpu = matches_cpu && mismatch_count == 0;
        std::fprintf(stderr,
                     "DIPOLE_MAP_COMPARE cells=%zu occupied_cpu=%zu occupied_gpu=%d "
                     "mismatches=%zu first_mismatch=%zu cpu_ms=%.6f gpu_ms=%.6f %s\n",
                     count, cpu_occupied, *host_occupied_count, mismatch_count,
                     first_mismatch == count ? 0 : first_mismatch + 1,
                     cpu_milliseconds, double(gpu_milliseconds),
                     matches_cpu ? "PASS" : "FAIL");
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    return matches_cpu;
}
*/
} // namespace cuda_geometry
