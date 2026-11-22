#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuda_data {

namespace {

std::mutex cuda_device_state_mutex;
bool cuda_device_initialized = false;
int cuda_device_id = 0;
struct NamedScratchBuffer {
    void *device = nullptr;
    size_t capacity = 0;
};

std::mutex workspace_buffers_mutex;
std::unordered_map<std::string, NamedScratchBuffer> workspace_buffers;
// Five fixed-size GPU arrays owned by one IF-DDA calculation. They cannot grow
// while a calculation is running: other kernels may retain their addresses.
std::mutex persistent_arrays_mutex;
CU_INT* dipole_map = nullptr;
size_t dipole_map_capacity = 0;
CU_INT* tapdip = nullptr;
size_t tapdip_capacity = 0;
std::mutex tapdip_upload_mutex;
std::vector<int> uploaded_tabdip_host_values;
CUCOMPLEX* polarisa = nullptr;
size_t polarisa_capacity = 0;
CUCOMPLEX* epsilon = nullptr;
size_t epsilon_capacity = 0;
CUCOMPLEX* ff0 = nullptr;
size_t ff0_capacity = 0;
std::unordered_map<int, cufftHandle> lens_fft_plans;
std::unordered_map<int, cufftHandle> diffraction_energy_fft_plans;
std::unordered_map<int, cufftHandle> diffraction_incident_fft_plans;
std::unordered_map<int, cufftHandle> bright_field_fft_plans;
std::mutex bright_field_plan_mutex;
std::mutex inhomogeneous_material_plan_mutex;
std::unordered_map<std::string, cufftHandle> inhomogeneous_material_fft_plans;
std::mutex chan_preconditioner_plan_mutex;
std::unordered_map<std::string, cufftHandle> chan_preconditioner_fft_plans;
std::mutex multilayer_plan_mutex;
std::unordered_map<int, cufftHandle> multilayer_layer_fft_plans;
std::unordered_map<int, cufftHandle> multilayer_image_fft_plans;
struct GeometryCoordinateAlias {
    const CU_REAL* device = nullptr;
    size_t count = 0;
};
std::mutex geometry_coordinate_alias_mutex;
std::unordered_map<const REAL*, GeometryCoordinateAlias> geometry_coordinate_aliases;
std::mutex host_regular_geometry_mutex;
HostRegularGeometry current_host_regular_geometry{};
} // namespace

namespace {
template <typename T>
bool check_fixed_gpu_array_size(const T* pointer, size_t capacity,
                                size_t count, const char* label) {
    if (!count || count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        std::fprintf(stderr, "%s: invalid GPU array size %zu\n", label, count);
        return false;
    }
    if (pointer && count > capacity) {
        std::fprintf(stderr,
                     "%s: requested %zu elements, persistent capacity is %zu; "
                     "restart IF-DDA with a larger allocation\n",
                     label, count, capacity);
        return false;
    }
    return true;
}

template <typename T>
void get_fixed_gpu_array(T*& pointer, size_t& capacity, size_t count,
                         const char* label) {
    const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&pointer),
                                          count * sizeof(T));
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: cudaMalloc failed: %s\n", label,
                     cudaGetErrorString(status));
        pointer = nullptr;
        return;
    }
    capacity = count;
}
} // namespace

//--------------------------------------------
cudaError_t copy_host_to_device(void *device_destination, const void *host_source, size_t bytes,
                                cuda_utils::TransferCategory category) {
    const cudaError_t status =
        cudaMemcpy(device_destination, host_source, bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess)
        cuda_utils::record_host_transfer(bytes, cuda_utils::TransferDirection::HostToDevice,
                                         category);
    return status;
}

//--------------------------------------------
cudaError_t copy_device_to_host(void *host_destination, const void *device_source, size_t bytes,
                                cuda_utils::TransferCategory category) {
    const cudaError_t status =
        cudaMemcpy(host_destination, device_source, bytes, cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
        cuda_utils::record_host_transfer(bytes, cuda_utils::TransferDirection::DeviceToHost,
                                         category);
    return status;
}

//--------------------------------------------
cudaError_t copy_host_complex_to_device(
    CUCOMPLEX *device_destination, const std::complex<CU_REAL> *host_source,
    size_t count, cuda_utils::TransferCategory category) {
    static_assert(sizeof(std::complex<CU_REAL>) == sizeof(CUCOMPLEX),
                  "Host and CUDA complex element sizes must match");
    if (count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return cudaErrorInvalidValue;
    return copy_host_to_device(device_destination, host_source,
        count * sizeof(CUCOMPLEX), category);
}

//--------------------------------------------
cudaError_t copy_device_complex_to_host(
    std::complex<CU_REAL> *host_destination, const CUCOMPLEX *device_source,
    size_t count, cuda_utils::TransferCategory category) {
    static_assert(sizeof(std::complex<CU_REAL>) == sizeof(CUCOMPLEX),
                  "Host and CUDA complex element sizes must match");
    if (count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return cudaErrorInvalidValue;
    return copy_device_to_host(host_destination, device_source,
        count * sizeof(CUCOMPLEX), category);
}

//--------------------------------------------
cudaError_t copy_device_to_device(void *device_destination, const void *device_source,
                                  size_t bytes) {
    return cudaMemcpy(device_destination, device_source, bytes, cudaMemcpyDeviceToDevice);
}

//--------------------------------------------
void *get_device_workspace_buffer_bytes(const char *name, size_t bytes) {
    if (!name || !*name || bytes == 0 || !initialize_cuda_device())
        return nullptr;

    std::lock_guard<std::mutex> lock(workspace_buffers_mutex);
    NamedScratchBuffer &entry = workspace_buffers[name];
    if (entry.device && entry.capacity >= bytes)
        return entry.device;

    if (entry.device) {
        const cudaError_t free_status = cudaFree(entry.device);
        if (free_status != cudaSuccess) {
            std::fprintf(stderr, "CUDA preconditioner buffer '%s': cudaFree failed: %s\n", name,
                         cudaGetErrorString(free_status));
            return nullptr;
        }
        entry.device = nullptr;
        entry.capacity = 0;
    }

    const cudaError_t alloc_status = cudaMalloc(&entry.device, bytes);
    if (alloc_status != cudaSuccess) {
        entry.device = nullptr;
        std::fprintf(stderr, "CUDA preconditioner buffer '%s': cudaMalloc failed: %s\n", name,
                     cudaGetErrorString(alloc_status));
        return nullptr;
    }
    entry.capacity = bytes;
    return entry.device;
}

//--------------------------------------------
bool get_device_mie_reference_buffers(size_t recurrence_count,
                                      MieReferenceDeviceBuffers *buffers) {
    if (!buffers || recurrence_count == 0 ||
        recurrence_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;
    MieReferenceDeviceBuffers acquired{};
    acquired.recurrence = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "mie.reference.recurrence", recurrence_count * sizeof(CUCOMPLEX)));
    acquired.outputs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "mie.reference.outputs", 5 * sizeof(CU_REAL)));
    if (!acquired.recurrence || !acquired.outputs)
        return false;
    *buffers = acquired;
    return true;
}

namespace {
//--------------------------------------------
bool request_named_buffer(CUCOMPLEX **target, const char *name, size_t count) {
    if (!target || (count && count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX)))
        return false;
    *target = count ? static_cast<CUCOMPLEX *>(
                          get_device_workspace_buffer_bytes(name, count * sizeof(CUCOMPLEX)))
                    : nullptr;
    return count == 0 || *target != nullptr;
}

//--------------------------------------------
bool request_named_buffer(CU_INT **target, const char *name, size_t count) {
    if (!target || (count && count > std::numeric_limits<size_t>::max() / sizeof(CU_INT)))
        return false;
    *target = count
                  ? static_cast<CU_INT *>(get_device_workspace_buffer_bytes(name, count * sizeof(CU_INT)))
                  : nullptr;
    return count == 0 || *target != nullptr;
}

//--------------------------------------------
bool request_named_buffer(CU_REAL **target, const char *name, size_t count) {
    if (!target || (count && count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL)))
        return false;
    *target = count ? static_cast<CU_REAL *>(
                          get_device_workspace_buffer_bytes(name, count * sizeof(CU_REAL)))
                    : nullptr;
    return count == 0 || *target != nullptr;
}

//--------------------------------------------
bool request_named_buffer(CUCOMPLEX ***target, const char *name, size_t count) {
    if (!target || (count && count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX *)))
        return false;
    *target = count ? static_cast<CUCOMPLEX **>(
                          get_device_workspace_buffer_bytes(name, count * sizeof(CUCOMPLEX *)))
                    : nullptr;
    return count == 0 || *target != nullptr;
}
} // namespace

//--------------------------------------------
bool get_device_cube_geometry_buffers(size_t count, CubeGeometryDeviceBuffers *buffers,
                                      size_t shape_parameter_count) {
    if (!buffers)
        return false;
    CubeGeometryDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.xs, "geometry.cube.xs", count) ||
        !request_named_buffer(&acquired.ys, "geometry.cube.ys", count) ||
        !request_named_buffer(&acquired.zs, "geometry.cube.zs", count) ||
        (count && !(acquired.dipole_map = get_dipole_map(count))) ||
        !request_named_buffer(&acquired.shape_parameters, "geometry.shape.parameters",
                              shape_parameter_count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
CU_INT* get_dipole_map(size_t grid_count) {
    if (!initialize_cuda_device()) return nullptr;
    std::lock_guard<std::mutex> lock(persistent_arrays_mutex);
    if (!check_fixed_gpu_array_size(dipole_map, dipole_map_capacity, grid_count,
                                    "get_dipole_map")) return nullptr;
    if (dipole_map == nullptr)
        get_fixed_gpu_array(dipole_map, dipole_map_capacity, grid_count, "get_dipole_map");
    return dipole_map;
}

//--------------------------------------------
CUCOMPLEX* get_polarisa(size_t tensor_count) {
    if (!initialize_cuda_device()) return nullptr;
    std::lock_guard<std::mutex> lock(persistent_arrays_mutex);
    if (!check_fixed_gpu_array_size(polarisa, polarisa_capacity, tensor_count,
                                    "get_polarisa")) return nullptr;
    if (polarisa == nullptr)
        get_fixed_gpu_array(polarisa, polarisa_capacity, tensor_count, "get_polarisa");
    return polarisa;
}

CUCOMPLEX* get_epsilon(size_t tensor_count) {
    if (!initialize_cuda_device()) return nullptr;
    std::lock_guard<std::mutex> lock(persistent_arrays_mutex);
    if (!check_fixed_gpu_array_size(epsilon, epsilon_capacity, tensor_count,
                                    "get_epsilon")) return nullptr;
    if (epsilon == nullptr)
        get_fixed_gpu_array(epsilon, epsilon_capacity, tensor_count, "get_epsilon");
    return epsilon;
}

//--------------------------------------------
CUCOMPLEX* get_ff0(size_t nbsphere) {
    if (!initialize_cuda_device() ||
        nbsphere > std::numeric_limits<size_t>::max() / 3)
        return nullptr;
    const size_t count = 3 * nbsphere;
    std::lock_guard<std::mutex> lock(persistent_arrays_mutex);
    if (!check_fixed_gpu_array_size(ff0, ff0_capacity, count, "get_ff0"))
        return nullptr;
    if (ff0 == nullptr)
        get_fixed_gpu_array(ff0, ff0_capacity, count, "get_ff0");
    return ff0;
}

CU_INT* get_tapdip(size_t grid_count) {
    if (!initialize_cuda_device()) return nullptr;
    std::lock_guard<std::mutex> lock(persistent_arrays_mutex);
    if (!check_fixed_gpu_array_size(tapdip, tapdip_capacity, grid_count,
                                    "get_tapdip")) return nullptr;
    if (tapdip == nullptr)
        get_fixed_gpu_array(tapdip, tapdip_capacity, grid_count, "get_tapdip");
    return tapdip;
}

//--------------------------------------------
bool get_device_sphere_geometry_compact_buffers(
    size_t count, size_t scan_temporary_bytes,
    SphereGeometryCompactDeviceBuffers *buffers) {
    if (!buffers || count == 0 || count > size_t(INT_MAX) ||
        count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
        count > std::numeric_limits<size_t>::max() / sizeof(int))
        return false;
    SphereGeometryCompactDeviceBuffers acquired{};
    acquired.xs = static_cast<CU_REAL*>(get_device_workspace_buffer_bytes(
        "geometry.sphere.compact.xs", count * sizeof(CU_REAL)));
    acquired.ys = static_cast<CU_REAL*>(get_device_workspace_buffer_bytes(
        "geometry.sphere.compact.ys", count * sizeof(CU_REAL)));
    acquired.zs = static_cast<CU_REAL*>(get_device_workspace_buffer_bytes(
        "geometry.sphere.compact.zs", count * sizeof(CU_REAL)));
    acquired.prefix = static_cast<CU_INT*>(get_device_workspace_buffer_bytes(
        "geometry.sphere.compact.prefix", count * sizeof(int)));
    acquired.packed_count = static_cast<CU_INT*>(get_device_workspace_buffer_bytes(
        "geometry.sphere.compact.count", sizeof(int)));
    acquired.scan_temporary_bytes = scan_temporary_bytes;
    acquired.scan_temporary = scan_temporary_bytes
        ? get_device_workspace_buffer_bytes("geometry.sphere.compact.scan_temporary",
                                           scan_temporary_bytes)
        : nullptr;
    if (!acquired.xs || !acquired.ys || !acquired.zs || !acquired.prefix ||
        !acquired.packed_count || !acquired.scan_temporary)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool register_device_geometry_coordinate_aliases(
    const REAL *host_xs, const REAL *host_ys, const REAL *host_zs,
    size_t count, const CU_REAL *device_xs, const CU_REAL *device_ys,
    const CU_REAL *device_zs) {
    if (!host_xs || !host_ys || !host_zs || !device_xs || !device_ys ||
        !device_zs || count == 0)
        return false;
    std::lock_guard<std::mutex> lock(geometry_coordinate_alias_mutex);
    geometry_coordinate_aliases[host_xs] = {device_xs, count};
    geometry_coordinate_aliases[host_ys] = {device_ys, count};
    geometry_coordinate_aliases[host_zs] = {device_zs, count};
    return true;
}


//--------------------------------------------
bool get_device_geometry_coordinate_aliases(
    const REAL *host_xs, const REAL *host_ys, const REAL *host_zs,
    size_t count, const CU_REAL **device_xs, const CU_REAL **device_ys,
    const CU_REAL **device_zs) {
    if (!host_xs || !host_ys || !host_zs || !device_xs || !device_ys ||
        !device_zs || count == 0)
        return false;
    std::lock_guard<std::mutex> lock(geometry_coordinate_alias_mutex);
    const auto x = geometry_coordinate_aliases.find(host_xs);
    const auto y = geometry_coordinate_aliases.find(host_ys);
    const auto z = geometry_coordinate_aliases.find(host_zs);
    if (x == geometry_coordinate_aliases.end() || y == geometry_coordinate_aliases.end() ||
        z == geometry_coordinate_aliases.end() || x->second.count < count ||
        y->second.count < count || z->second.count < count)
        return false;
    *device_xs = x->second.device;
    *device_ys = y->second.device;
    *device_zs = z->second.device;
    return *device_xs && *device_ys && *device_zs;
}

//--------------------------------------------
void clear_device_geometry_coordinate_aliases() {
    std::lock_guard<std::mutex> lock(geometry_coordinate_alias_mutex);
    geometry_coordinate_aliases.clear();
}

//--------------------------------------------
bool register_host_regular_geometry(const HostRegularGeometry& geometry) {
    if (!geometry.xs || !geometry.ys || !geometry.zs || !geometry.tabdip ||
        geometry.nx <= 0 || geometry.ny <= 0 || geometry.nz <= 0 ||
        geometry.compact_count <= 0 || geometry.cell_size <= REAL(0) ||
        geometry.nx > INT_MAX / geometry.ny ||
        geometry.nx * geometry.ny > INT_MAX / geometry.nz ||
        geometry.compact_count > geometry.nx * geometry.ny * geometry.nz)
        return false;
    std::lock_guard<std::mutex> lock(host_regular_geometry_mutex);
    current_host_regular_geometry = geometry;
    return true;
}

//--------------------------------------------
bool get_host_regular_geometry(const REAL* xs, const REAL* ys, const REAL* zs,
                               int compact_count, HostRegularGeometry* geometry) {
    if (!xs || !ys || !zs || !geometry || compact_count <= 0)
        return false;
    std::lock_guard<std::mutex> lock(host_regular_geometry_mutex);
    if (current_host_regular_geometry.xs != xs ||
        current_host_regular_geometry.ys != ys ||
        current_host_regular_geometry.zs != zs ||
        current_host_regular_geometry.compact_count != compact_count)
        return false;
    *geometry = current_host_regular_geometry;
    return true;
}

//--------------------------------------------
void clear_host_regular_geometry() {
    std::lock_guard<std::mutex> lock(host_regular_geometry_mutex);
    current_host_regular_geometry = {};
}

//--------------------------------------------
bool get_device_inhomogeneous_material_buffers(
    int nx, int ny, int nz, InhomogeneousMaterialDeviceBuffers *buffers) {
    if (!buffers || nx <= 0 || ny <= 0 || nz <= 0)
        return false;
    size_t plane = 0;
    size_t count = 0;
    if (!cuda_utils::checked_multiply(size_t(nx), size_t(ny), &plane) ||
        !cuda_utils::checked_multiply(plane, size_t(nz), &count))
        return false;

    InhomogeneousMaterialDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.spectrum, "material.inhomogeneous.spectrum", count) ||
        !request_named_buffer(&acquired.phases, "material.inhomogeneous.phases", count) ||
        !request_named_buffer(&acquired.ones, "material.inhomogeneous.ones", count) ||
        !request_named_buffer(&acquired.real_values, "material.inhomogeneous.real", count) ||
        !request_named_buffer(&acquired.magnitude_squares,
                              "material.inhomogeneous.magnitude_squares", count))
        return false;

    const std::string key = std::to_string(nx) + "x" + std::to_string(ny) + "x" +
                            std::to_string(nz);
    std::lock_guard<std::mutex> lock(inhomogeneous_material_plan_mutex);
    auto found = inhomogeneous_material_fft_plans.find(key);
    if (found == inhomogeneous_material_fft_plans.end()) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
        const cufftType transform_type = CUFFT_C2C;
#else
        const cufftType transform_type = CUFFT_Z2Z;
#endif
        cufftHandle plan = 0;
        if (!cuda_utils::check_cufft(cufftPlan3d(&plan, nz, ny, nx, transform_type),
                                     "inhomogeneous material", "create inverse FFT plan"))
            return false;
        found = inhomogeneous_material_fft_plans.emplace(key, plan).first;
    }
    acquired.fft_plan = found->second;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_dipole_inside_buffers(size_t count,
                                      DipoleInsideDeviceBuffers *buffers) {
    if (!buffers || count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL))
        return false;
    DipoleInsideDeviceBuffers acquired{};
    if (count > 0) {
        acquired.xs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "geometry.dipole_inside.xs", count * sizeof(CU_REAL)));
        acquired.ys = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "geometry.dipole_inside.ys", count * sizeof(CU_REAL)));
        acquired.zs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "geometry.dipole_inside.zs", count * sizeof(CU_REAL)));
    }
    acquired.first_match = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
        "geometry.dipole_inside.first_match", sizeof(int)));
    if ((count > 0 && (!acquired.xs || !acquired.ys || !acquired.zs)) ||
        !acquired.first_match)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_polarizability_init_buffers(
    size_t dipole_count, size_t tensor_count, PolarizabilityInitDeviceBuffers *buffers) {
    if (!buffers || tensor_count < dipole_count)
        return false;
    PolarizabilityInitDeviceBuffers acquired{};
    // The initializer needs a separate scratch map; the geometry map is kept
    // intact for later material and illumination kernels.
    if (!request_named_buffer(&acquired.dipole_map, "material.init.dipole_map", dipole_count) ||
        !(acquired.polarizability = get_polarisa(tensor_count)) ||
        !(acquired.epsilon = get_epsilon(tensor_count)))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_scalar_polarizability_buffers(
    ScalarPolarizabilityDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    ScalarPolarizabilityDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.value, "material.scalar_polarizability.value", 1) ||
        !request_named_buffer(&acquired.sldr, "material.scalar_polarizability.sldr", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_polarizability_batch_buffers(
    size_t count, PolarizabilityBatchDeviceBuffers *buffers) {
    if (!buffers || count == 0)
        return false;
    PolarizabilityBatchDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.epsilon, "material.polarizability_batch.epsilon", count) ||
        !request_named_buffer(&acquired.values, "material.polarizability_batch.values", count) ||
        !request_named_buffer(&acquired.inside, "material.polarizability_batch.inside", count) ||
        !request_named_buffer(&acquired.sldr, "material.polarizability_batch.sldr", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_tensor_polarizability_buffers(
    TensorPolarizabilityDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    TensorPolarizabilityDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.epsilon, "material.tensor_polarizability.epsilon", 9) ||
        !request_named_buffer(&acquired.value, "material.tensor_polarizability.value", 9) ||
        !request_named_buffer(&acquired.status, "material.tensor_polarizability.status", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_gaussian_power_buffers(GaussianPowerDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    GaussianPowerDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.field, "incident.gaussian_power.field", 1) ||
        !request_named_buffer(&acquired.irradiance, "incident.gaussian_power.irradiance", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_material_interpolation_buffers(
    MaterialInterpolationDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    MaterialInterpolationDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.result, "material.interpolation.result", 2))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_microscope_mask_buffers(size_t coordinate_count, size_t mask_capacity,
                                        MicroscopeMaskDeviceBuffers *buffers) {
    if (!buffers || coordinate_count == 0 || mask_capacity == 0)
        return false;
    MicroscopeMaskDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.xs, "microscopy.mask.xs", coordinate_count) ||
        !request_named_buffer(&acquired.ys, "microscopy.mask.ys", coordinate_count) ||
        !request_named_buffer(&acquired.extrema, "microscopy.mask.extrema", 4) ||
        !request_named_buffer(&acquired.mask, "microscopy.mask.output", mask_capacity) ||
        !request_named_buffer(&acquired.selected_count, "microscopy.mask.selected_count", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_dark_field_microscope_buffers(
    size_t dipole_count, size_t polarization_count, size_t scalar_local_count,
    size_t image_value_count, size_t spectrum_value_count,
    size_t detector_value_count, size_t solver_field_count,
    size_t scalar_solution_count, size_t image_scratch_count,
    DarkFieldMicroscopeDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    DarkFieldMicroscopeDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.dipole_map,
                              "microscopy.dark_field.dipole_map", dipole_count) ||
        !request_named_buffer(&acquired.polarization,
                              "microscopy.dark_field.polarization", polarization_count) ||
        !request_named_buffer(&acquired.scalar_local,
                              "microscopy.dark_field.scalar_local", scalar_local_count) ||
        !request_named_buffer(&acquired.image_planes,
                              "microscopy.dark_field.image_planes", image_value_count) ||
        !request_named_buffer(&acquired.spectrum_planes,
                              "microscopy.dark_field.spectrum_planes", spectrum_value_count) ||
        !request_named_buffer(&acquired.detector_planes,
                              "microscopy.dark_field.detector_planes", detector_value_count) ||
        !request_named_buffer(&acquired.solver_fields,
                              "microscopy.dark_field.solver_fields", solver_field_count) ||
        !request_named_buffer(&acquired.scalar_solution,
                              "microscopy.dark_field.scalar_solution", scalar_solution_count) ||
        !request_named_buffer(&acquired.image_scratch,
                              "microscopy.dark_field.image_scratch", image_scratch_count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_microscope_incident_buffers(
    size_t dipole_count, MicroscopeIncidentDeviceBuffers* buffers) {
    if (!buffers || dipole_count == 0)
        return false;
    size_t field_count = 0;
    if (!cuda_utils::checked_multiply(dipole_count, size_t(3), &field_count))
        return false;
    MicroscopeIncidentDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.xs, "microscopy.incident.xs", dipole_count) ||
        !request_named_buffer(&acquired.ys, "microscopy.incident.ys", dipole_count) ||
        !request_named_buffer(&acquired.zs, "microscopy.incident.zs", dipole_count) ||
        !request_named_buffer(&acquired.incident, "microscopy.incident.ff0", field_count) ||
        !request_named_buffer(&acquired.initial, "microscopy.incident.xi", field_count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_bright_field_microscope_buffers(
    size_t dipole_count, size_t fft_plane_count, int nfft,
    BrightFieldMicroscopeDeviceBuffers* buffers) {
    if (!buffers || dipole_count == 0 || fft_plane_count == 0 ||
        nfft <= 0 || size_t(nfft) > size_t(INT_MAX) / size_t(nfft) ||
        size_t(nfft) * size_t(nfft) != fft_plane_count ||
        dipole_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        fft_plane_count > std::numeric_limits<size_t>::max() / (6 * sizeof(CUCOMPLEX)) ||
        fft_plane_count > std::numeric_limits<size_t>::max() / (6 * sizeof(CU_REAL)))
        return false;
    BrightFieldMicroscopeDeviceBuffers acquired{};
    acquired.scattered_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "microscopy.bright_field.scattered_field",
        3 * dipole_count * sizeof(CUCOMPLEX)));
    acquired.diffraction_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "microscopy.bright_field.diffraction_field",
        3 * fft_plane_count * sizeof(CUCOMPLEX)));
    acquired.fourier_fields = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "microscopy.bright_field.fourier_fields",
        6 * fft_plane_count * sizeof(CUCOMPLEX)));
    acquired.image_intensity = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "microscopy.bright_field.image_intensity",
        6 * fft_plane_count * sizeof(CU_REAL)));
    if (!acquired.scattered_field || !acquired.diffraction_field ||
        !acquired.fourier_fields || !acquired.image_intensity)
        return false;
    std::lock_guard<std::mutex> plan_lock(bright_field_plan_mutex);
    const auto plan = bright_field_fft_plans.find(nfft);
    if (plan != bright_field_fft_plans.end()) {
        acquired.fft_plan = plan->second;
    } else {
        int dimensions[2] = {nfft, nfft};
        cufftHandle handle = 0;
        const cufftResult plan_status = cufftPlanMany(
            &handle, 2, dimensions, nullptr, 1, int(fft_plane_count),
            nullptr, 1, int(fft_plane_count), CUFFT_TRANSFORM, 6);
        if (plan_status != CUFFT_SUCCESS) {
            std::fprintf(stderr,
                "CUDA bright-field FFT plan creation failed (status %d)\\n",
                int(plan_status));
            return false;
        }
        bright_field_fft_plans.emplace(nfft, handle);
        acquired.fft_plan = handle;
    }
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_confocal_fluorescence_buffers(
    size_t field_value_count, size_t intensity_value_count,
    size_t diffraction_value_count, size_t spectrum_value_count,
    ConfocalFluorescenceDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    ConfocalFluorescenceDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.field_planes,
                              "microscopy.confocal_fluorescence.fields",
                              field_value_count) ||
        !request_named_buffer(&acquired.intensity_stack,
                              "microscopy.confocal_fluorescence.intensity",
                              intensity_value_count) ||
        !request_named_buffer(&acquired.diffraction,
                              "microscopy.confocal_fluorescence.diffraction",
                              diffraction_value_count) ||
        !request_named_buffer(&acquired.spectrum_planes,
                              "microscopy.confocal_fluorescence.spectra",
                              spectrum_value_count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_confocal_microscope_buffers(
    size_t dipole_count, size_t diffraction_value_count,
    size_t dense_field_value_count,
    ConfocalMicroscopeDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    ConfocalMicroscopeDeviceBuffers acquired{};
    size_t field_value_count = 0;
    if (!cuda_utils::checked_multiply(dipole_count, size_t(3), &field_value_count) ||
        !request_named_buffer(&acquired.dipole_field,
                              "microscopy.confocal.dipole_field",
                              field_value_count) ||
        !request_named_buffer(&acquired.xs, "microscopy.confocal.xs", dipole_count) ||
        !request_named_buffer(&acquired.ys, "microscopy.confocal.ys", dipole_count) ||
        !request_named_buffer(&acquired.zs, "microscopy.confocal.zs", dipole_count) ||
        !request_named_buffer(&acquired.diffraction,
                              "microscopy.confocal.diffraction",
                              diffraction_value_count) ||
        !request_named_buffer(&acquired.dense_field,
                              "microscopy.confocal.dense_field",
                              dense_field_value_count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_confocal_detector_buffers(
    size_t plane_value_count, size_t image_value_count,
    ConfocalDetectorDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    ConfocalDetectorDeviceBuffers acquired{};
    size_t six_plane_values = 0;
    size_t three_plane_values = 0;
    size_t three_image_values = 0;
    if (!cuda_utils::checked_multiply(plane_value_count, size_t(3),
                                      &three_plane_values) ||
        !cuda_utils::checked_multiply(plane_value_count, size_t(6),
                                      &six_plane_values) ||
        !cuda_utils::checked_multiply(image_value_count, size_t(3),
                                      &three_image_values) ||
        !request_named_buffer(&acquired.diffraction,
                              "microscopy.confocal_detector.diffraction",
                              three_plane_values) ||
        !request_named_buffer(&acquired.fourier_fields,
                              "microscopy.confocal_detector.fourier_fields",
                              six_plane_values) ||
        !request_named_buffer(&acquired.image_stack,
                              "microscopy.confocal_detector.image_stack",
                              three_image_values) ||
        !request_named_buffer(&acquired.status,
                              "microscopy.confocal_detector.status", size_t(1)))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_widefield_buffers(const WideFieldBufferRequest& request,
                                  WideFieldDeviceBuffers* buffers) {
    if (!buffers)
        return false;
    WideFieldDeviceBuffers acquired{};
    size_t field_count = 0;
    if (!cuda_utils::checked_multiply(request.grid_count, size_t(3),
                                      &field_count))
        return false;
    if (!request_named_buffer(&acquired.dipole_x, "widefield.dipole_x",
                              request.dipole_count) ||
        !request_named_buffer(&acquired.dipole_y, "widefield.dipole_y",
                              request.dipole_count) ||
        !request_named_buffer(&acquired.dipole_z, "widefield.dipole_z",
                              request.dipole_count) ||
        !request_named_buffer(&acquired.grid_x, "widefield.grid_x",
                              request.grid_count) ||
        !request_named_buffer(&acquired.grid_y, "widefield.grid_y",
                              request.grid_count) ||
        !request_named_buffer(&acquired.grid_z, "widefield.grid_z",
                              request.grid_count) ||
        !request_named_buffer(&acquired.incident, "widefield.incident",
                              field_count) ||
        !request_named_buffer(&acquired.incident_magnitude,
                              "widefield.incident_magnitude",
                              request.grid_count) ||
        !request_named_buffer(&acquired.initial_field,
                              "widefield.initial_field", field_count) ||
        !request_named_buffer(&acquired.scattered_field,
                              "widefield.scattered_field", field_count) ||
        !request_named_buffer(&acquired.local_field, "widefield.local_field",
                              field_count) ||
        !request_named_buffer(&acquired.polarization,
                              "widefield.polarization",
                              request.grid_material_count) ||
        !request_named_buffer(&acquired.epsilon, "widefield.epsilon",
                              request.grid_material_count) ||
        !request_named_buffer(&acquired.dipole_scattered,
                              "widefield.dipole_scattered",
                              request.dipole_field_count) ||
        !request_named_buffer(&acquired.dipole_incident,
                              "widefield.dipole_incident",
                              request.dipole_incident_count) ||
        !request_named_buffer(&acquired.dipole_epsilon,
                              "widefield.dipole_epsilon",
                              request.dipole_material_count) ||
        !request_named_buffer(&acquired.dipole_epsilon_source,
                              "widefield.dipole_epsilon_source",
                              request.dipole_material_source_count) ||
        !request_named_buffer(&acquired.fft_vectors,
                              "widefield.fft_vectors",
                              request.fft_vector_count) ||
        !request_named_buffer(&acquired.green_tensors,
                              "widefield.green_tensors",
                              request.green_tensor_count) ||
        !request_named_buffer(&acquired.dipole_map, "widefield.dipole_map",
                              request.grid_count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_abcchi_buffers(AbcChiDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    AbcChiDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.epsilon_output, "geometry.abcchi.epsilon", 9) ||
        !request_named_buffer(&acquired.contrast_output, "geometry.abcchi.contrast", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_small_dense_system_buffers(SmallDenseSystemDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    SmallDenseSystemDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.matrix, "small_dense_system.matrix", 81) ||
        !request_named_buffer(&acquired.rhs, "small_dense_system.rhs", 9) ||
        !request_named_buffer(&acquired.info, "small_dense_system.info", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_local_macro_buffers(size_t tensor_count, size_t dipole_count,
                                    LocalMacroDeviceBuffers *buffers) {
    if (!buffers || tensor_count == 0 || dipole_count == 0 ||
        dipole_count > std::numeric_limits<size_t>::max() / 3)
        return false;
    LocalMacroDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.epsilon, "local_macro.epsilon", tensor_count) ||
        !request_named_buffer(&acquired.input_field, "local_macro.input_field", 3 * dipole_count) ||
        !request_named_buffer(&acquired.output_field, "local_macro.output_field", 3 * dipole_count) ||
        !request_named_buffer(&acquired.field_magnitude, "local_macro.field_magnitude", dipole_count) ||
        !request_named_buffer(&acquired.status, "local_macro.status", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_chan_preconditioner_buffers(const ChanPreconditionerBufferSizes &sizes,
                                            ChanPreconditionerDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    ChanPreconditionerDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.scalar.channel, "chan.scalar.channel",
                              sizes.scalar_mesh_count) ||
        !request_named_buffer(&acquired.scalar.vectors, "chan.scalar.chan_vectors",
                              sizes.scalar_mesh_count) ||
        !request_named_buffer(&acquired.scalar.field, "chan.scalar.field",
                              sizes.scalar_field_count) ||
        !request_named_buffer(&acquired.scalar.matrices, "chan.scalar.matrices",
                              sizes.scalar_matrix_count) ||
        !request_named_buffer(&acquired.scalar.fft_planes, "chan.scalar.fft_planes",
                              sizes.scalar_mesh_count) ||
        !request_named_buffer(&acquired.scalar.rhs, "chan.scalar.rhs", sizes.scalar_mesh_count) ||
        !request_named_buffer(&acquired.scalar.rhs_output, "chan.scalar.rhs_output",
                              sizes.scalar_mesh_count) ||
        !request_named_buffer(&acquired.scalar.polarizabilities, "chan.scalar.polarizabilities",
                              sizes.scalar_polarizability_count) ||
        !request_named_buffer(&acquired.scalar.polarizability_sum, "chan.scalar.polarizability_sum",
                              sizes.scalar_polarizability_count ? 1 : 0) ||
        !request_named_buffer(&acquired.scalar.nonzero_count, "chan.scalar.nonzero_count",
                              sizes.scalar_polarizability_count ? 1 : 0) ||
        !request_named_buffer(&acquired.scalar.pivots, "chan.scalar.pivots",
                              sizes.scalar_pivot_count) ||
        !request_named_buffer(&acquired.scalar.factor_info, "chan.scalar.factor_info",
                              sizes.scalar_block_count) ||
        !request_named_buffer(&acquired.scalar.matrix_ptrs, "chan.scalar.matrix_ptrs",
                              sizes.scalar_block_count) ||
        !request_named_buffer(&acquired.scalar.rhs_ptrs, "chan.scalar.rhs_ptrs",
                              sizes.scalar_block_count) ||
        !request_named_buffer(&acquired.vector.field, "chan.vector.field",
                              sizes.vector_field_count) ||
        !request_named_buffer(&acquired.vector.fft_planes, "chan.vector.fft_planes",
                              sizes.vector_fft_count) ||
        !request_named_buffer(&acquired.vector.rhs, "chan.vector.rhs", sizes.vector_rhs_count) ||
        !request_named_buffer(&acquired.vector.rhs_output, "chan.vector.rhs_output",
                              sizes.vector_rhs_count) ||
        !request_named_buffer(&acquired.vector.matrices, "chan.vector.matrices",
                              sizes.vector_matrix_count) ||
        !request_named_buffer(&acquired.vector.channel_vectors, "chan.vector.channel_vectors",
                              sizes.vector_channel_count) ||
        !request_named_buffer(&acquired.vector.green_components, "chan.vector.green_components",
                              sizes.vector_green_count) ||
        !request_named_buffer(&acquired.vector.fft_components, "chan.vector.green_fft_components",
                              sizes.vector_green_count) ||
        !request_named_buffer(&acquired.vector.chan_vectors, "chan.vector.chan_vectors",
                              sizes.vector_chan_vectors_count) ||
        !request_named_buffer(&acquired.vector.polarizabilities, "chan.vector.polarizabilities",
                              sizes.vector_polarizability_count) ||
        !request_named_buffer(&acquired.vector.polarizability_sum, "chan.vector.polarizability_sum",
                              sizes.vector_polarizability_count ? 1 : 0) ||
        !request_named_buffer(&acquired.vector.mean_polarizability,
                              "chan.vector.mean_polarizability",
                              sizes.vector_polarizability_count ? 1 : 0) ||
        !request_named_buffer(&acquired.vector.identity_map, "chan.vector.identity_map",
                              sizes.vector_map_count) ||
        !request_named_buffer(&acquired.vector.nonzero_count, "chan.vector.nonzero_count",
                              sizes.vector_polarizability_count ? 1 : 0) ||
        !request_named_buffer(&acquired.vector.pivots, "chan.vector.pivots",
                              sizes.vector_pivot_count) ||
        !request_named_buffer(&acquired.vector.factor_info, "chan.vector.factor_info",
                              sizes.vector_batch_count) ||
        !request_named_buffer(&acquired.vector.matrix_ptrs, "chan.vector.matrix_ptrs",
                              sizes.vector_batch_count) ||
        !request_named_buffer(&acquired.vector.rhs_ptrs, "chan.vector.rhs_ptrs",
                              sizes.vector_batch_count))
        return false;

    auto acquire_plan = [](const char *prefix, int nx, int ny, int nz, int batch_factor,
                           cufftHandle *target) {
        if (!target || nx <= 0 || ny <= 0 || nz <= 0 ||
            nx > std::numeric_limits<int>::max() / ny ||
            nz > std::numeric_limits<int>::max() / batch_factor)
            return false;
        const std::string key = std::string(prefix) + "." + std::to_string(nx) + "." +
                                std::to_string(ny) + "." + std::to_string(nz);
        std::lock_guard<std::mutex> lock(chan_preconditioner_plan_mutex);
        auto plan = chan_preconditioner_fft_plans.find(key);
        if (plan == chan_preconditioner_fft_plans.end()) {
            int dimensions[2] = {ny, nx};
            cufftHandle handle = 0;
            if (cufftPlanMany(&handle, 2, dimensions, nullptr, 1, nx * ny, nullptr, 1,
                              nx * ny, CUFFT_TRANSFORM, batch_factor * nz) != CUFFT_SUCCESS)
                return false;
            plan = chan_preconditioner_fft_plans.emplace(key, handle).first;
        }
        *target = plan->second;
        return true;
    };
    if (sizes.scalar_mesh_count &&
        !acquire_plan("chan.scalar", sizes.scalar_nx, sizes.scalar_ny, sizes.scalar_nz, 1,
                      &acquired.scalar.fft_plan))
        return false;
    if (sizes.vector_fft_count &&
        !acquire_plan("chan.vector", sizes.vector_nx, sizes.vector_ny, sizes.vector_nz, 3,
                      &acquired.vector.fft_plan))
        return false;
    if (sizes.vector_green_count &&
        !acquire_plan("chan.vector.green", sizes.vector_nx, sizes.vector_ny,
                      sizes.vector_nz, 6, &acquired.vector.green_fft_plan))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_scalar_driver_buffers(size_t scalar_count, size_t fft_count,
                                      size_t vector_capacity_count,
                                      ScalarDriverDeviceBuffers *buffers) {
    if (!buffers || scalar_count > std::numeric_limits<size_t>::max() / 3 ||
        vector_capacity_count > std::numeric_limits<size_t>::max() / 3)
        return false;
    ScalarDriverDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.incident, "scalar_driver.incident",
                              scalar_count ? 3 * scalar_count : 0) ||
        !request_named_buffer(&acquired.polarizability, "scalar_driver.polarizability",
                              scalar_count) ||
        !request_named_buffer(&acquired.fft_work, "scalar_driver.fft_work", fft_count) ||
        !request_named_buffer(&acquired.scalar_solution, "scalar_driver.scalar_solution",
                              scalar_count) ||
        !request_named_buffer(&acquired.local_solution, "scalar_driver.local_solution",
                              scalar_count) ||
        !request_named_buffer(&acquired.vector_field, "scalar_driver.vector_field",
                              scalar_count ? 3 * scalar_count : 0) ||
        !request_named_buffer(&acquired.vector_local_field, "scalar_driver.vector_local_field",
                              scalar_count ? 3 * scalar_count : 0))
        return false;

    if (vector_capacity_count > scalar_count &&
        !request_named_buffer(&acquired.vector_field, "scalar_driver.vector_field",
                              3 * vector_capacity_count))
        return false;

    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_inverse_driver_gamma_buffers(size_t count, InverseDriverGammaBuffers *buffers) {
    if (!buffers)
        return false;
    InverseDriverGammaBuffers acquired{};
    if (!request_named_buffer(&acquired.input, "inverserig.gamma.input", count) ||
        !request_named_buffer(&acquired.output, "inverserig.gamma.output", count) ||
        !request_named_buffer(&acquired.incident, "inverserig.gamma.incident", count) ||
        !request_named_buffer(&acquired.solution, "inverserig.gamma.solution", count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_inverse_driver_born_buffers(size_t count, InverseDriverBornBuffers *buffers) {
    if (!buffers)
        return false;
    InverseDriverBornBuffers acquired{};
    if (!request_named_buffer(&acquired.input, "inverserig.born.input", count) ||
        !request_named_buffer(&acquired.output, "inverserig.born.output", count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_inverse_driver_rytov_buffers(InverseDriverRytovBuffers *buffers) {
    if (!buffers)
        return false;
    InverseDriverRytovBuffers acquired{};
    if (!request_named_buffer(&acquired.error, "inverserig.rytov.error", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_rytov_field_buffers(size_t vector_count, RytovFieldDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    RytovFieldDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.reference, "rytov.vector.eref", vector_count) ||
        !request_named_buffer(&acquired.delta, "rytov.vector.delta", vector_count) ||
        !request_named_buffer(&acquired.result, "rytov.vector.result", vector_count) ||
        !request_named_buffer(&acquired.error, "rytov.vector.error", 1))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_incident_field_buffers(const IncidentFieldBufferRequest &request,
                                       IncidentFieldDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    IncidentFieldDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.xs, "incident.plane.x", request.coordinate_count) ||
        !request_named_buffer(&acquired.ys, "incident.plane.y", request.coordinate_count) ||
        !request_named_buffer(&acquired.zs, "incident.plane.z", request.coordinate_count) ||
        !request_named_buffer(&acquired.field, "incident.shared.field", request.field_count) ||
        !request_named_buffer(&acquired.parameters, "incident.multi.parameters",
                              request.parameter_count) ||
        !request_named_buffer(&acquired.real_parameters, "incident.real.parameters",
                              request.real_parameter_count) ||
        !request_named_buffer(&acquired.antenna_xs, "incident.antenna.x",
                              request.antenna_coordinate_count) ||
        !request_named_buffer(&acquired.antenna_ys, "incident.antenna.y",
                              request.antenna_coordinate_count) ||
        !request_named_buffer(&acquired.antenna_zs, "incident.antenna.z",
                              request.antenna_coordinate_count) ||
        !request_named_buffer(&acquired.antenna_field, "incident.antenna.field",
                              request.antenna_field_count) ||
        !request_named_buffer(&acquired.antenna_reference, "incident.antenna.reference",
                              request.antenna_reference_count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_dipole_derivative_buffers(DipoleDerivativeDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    DipoleDerivativeDeviceBuffers acquired{};
    acquired.derivative = static_cast<CUCOMPLEX *>(
        get_device_workspace_buffer_bytes("incident.dipole.derivative",
                                          3 * sizeof(CUCOMPLEX)));
    if (!acquired.derivative)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_beam_propagation_buffers(const BeamPropagationBufferRequest &request,
                                         BeamPropagationDeviceBuffers *buffers) {
    if (!buffers)
        return false;
    BeamPropagationDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.xs, "bpm.xs", request.coordinate_count) ||
        !request_named_buffer(&acquired.ys, "bpm.ys", request.coordinate_count) ||
        !request_named_buffer(&acquired.zs, "bpm.zs", request.coordinate_count) ||
        !request_named_buffer(&acquired.epsilon, "bpm.epsilon", request.epsilon_count) ||
        !request_named_buffer(&acquired.incident, "bpm.incident", request.incident_count) ||
        !request_named_buffer(&acquired.images, "bpm.images", request.image_count) ||
        !request_named_buffer(&acquired.map, "bpm.map", request.map_count) ||
        !request_named_buffer(&acquired.local_field, "bpm.local_field", request.local_field_count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_beam_propagation_gaussian_buffers(size_t plane_count,
                                                  BeamPropagationGaussianBuffers *buffers) {
    if (!buffers || plane_count == 0 ||
        plane_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        plane_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
        plane_count > std::numeric_limits<size_t>::max() / sizeof(int))
        return false;
    const size_t partial_count = (plane_count + 255) / 256;
    if (partial_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL))
        return false;

    BeamPropagationGaussianBuffers acquired{};
    acquired.xs = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("bpm.gaussian.xs", plane_count * sizeof(CU_REAL)));
    acquired.ys = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("bpm.gaussian.ys", plane_count * sizeof(CU_REAL)));
    acquired.zs = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("bpm.gaussian.zs", plane_count * sizeof(CU_REAL)));
    acquired.paraxial_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "bpm.gaussian.paraxial", 3 * plane_count * sizeof(CUCOMPLEX)));
    acquired.rigorous_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "bpm.gaussian.rigorous", 3 * plane_count * sizeof(CUCOMPLEX)));
    acquired.max_partials = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "bpm.gaussian.max_partials", partial_count * sizeof(CU_REAL)));
    acquired.max_intensity = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("bpm.gaussian.max_intensity", sizeof(CU_REAL)));
    acquired.convergence = static_cast<CU_INT *>(
        get_device_workspace_buffer_bytes("bpm.gaussian.convergence", plane_count * sizeof(int)));
    acquired.gkp_nodes = static_cast<double *>(
        get_device_workspace_buffer_bytes("bpm.gaussian.gkp_nodes", 511 * sizeof(double)));
    acquired.gkp_weights = static_cast<double *>(
        get_device_workspace_buffer_bytes("bpm.gaussian.gkp_weights", 1013 * sizeof(double)));
    if (!acquired.xs || !acquired.ys || !acquired.zs || !acquired.paraxial_field ||
        !acquired.rigorous_field || !acquired.max_partials || !acquired.max_intensity ||
        !acquired.convergence || !acquired.gkp_nodes || !acquired.gkp_weights)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_multilayer_fft_plans(int layer_side, int image_side,
                                     MultilayerFftPlans *plans) {
    if (!plans || layer_side <= 0 || image_side <= 0 || !initialize_cuda_device())
        return false;
    std::lock_guard<std::mutex> lock(multilayer_plan_mutex);
    auto acquire = [](std::unordered_map<int, cufftHandle> &cache, int side,
                      cufftHandle *target) {
        auto plan = cache.find(side);
        if (plan == cache.end()) {
            cufftHandle handle = 0;
            if (cufftPlan2d(&handle, side, side, CUFFT_TRANSFORM) != CUFFT_SUCCESS)
                return false;
            plan = cache.emplace(side, handle).first;
        }
        *target = plan->second;
        return true;
    };
    MultilayerFftPlans acquired{};
    if (!acquire(multilayer_layer_fft_plans, layer_side, &acquired.layer) ||
        !acquire(multilayer_image_fft_plans, image_side, &acquired.image))
        return false;
    *plans = acquired;
    return true;
}

//--------------------------------------------
bool get_device_vector_green_buffers(size_t tensor_count, size_t fft_count,
                                     GreenTensorDeviceBuffers *buffers) {
    if (!buffers || tensor_count == 0 || fft_count == 0 ||
        tensor_count > size_t(2147483647) / sizeof(CUCOMPLEX) ||
        fft_count > size_t(2147483647) / sizeof(CUCOMPLEX))
        return false;
    GreenTensorDeviceBuffers acquired{};
    for (int i = 0; i < 6; ++i) {
        const std::string name = "green.vector.spatial." + std::to_string(i);
        acquired.spatial.data[i] = static_cast<CUCOMPLEX *>(
            get_device_workspace_buffer_bytes(name.c_str(), tensor_count * sizeof(CUCOMPLEX)));
        if (!acquired.spatial.data[i])
            return false;  // The producer releases any partially acquired scratch.
    }
    for (int i = 0; i < 3; ++i) {
        const std::string name = "green.vector.spectra." + std::to_string(i);
        acquired.spectra.data[i] = static_cast<CUCOMPLEX *>(
            get_device_workspace_buffer_bytes(name.c_str(), fft_count * sizeof(CUCOMPLEX)));
        if (!acquired.spectra.data[i])
            return false;
    }
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool release_device_vector_green_buffers() {
    // The final Green kernel and both cuFFT passes must finish before any
    // temporary allocation is released. Never release persistent tensors.
    const cudaError_t synchronized = cudaDeviceSynchronize();
    if (synchronized != cudaSuccess) {
        std::fprintf(stderr, "Green temporary buffers: CUDA synchronize failed: %s\n",
                     cudaGetErrorString(synchronized));
        return false;
    }
    std::lock_guard<std::mutex> lock(workspace_buffers_mutex);
    bool success = true;
    for (int index = 0; index < 9; ++index) {
        const std::string name = (index < 6 ? "green.vector.spatial." :
                                  "green.vector.spectra.") +
                                 std::to_string(index < 6 ? index : index - 6);
        const auto it = workspace_buffers.find(name);
        if (it == workspace_buffers.end())
            continue;
        if (it->second.device) {
            const cudaError_t freed = cudaFree(it->second.device);
            if (freed != cudaSuccess) {
                std::fprintf(stderr, "Green temporary buffers: cudaFree(%s) failed: %s\n",
                             name.c_str(), cudaGetErrorString(freed));
                success = false;
                continue;
            }
        }
        workspace_buffers.erase(it);
    }
    return success;
}

//--------------------------------------------
bool get_device_pml_buffers(size_t dipole_count, PmlDeviceBuffers *buffers,
                            size_t material_count) {
    if (!buffers || dipole_count == 0 ||
        dipole_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
        dipole_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        material_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL))
        return false;
    PmlDeviceBuffers acquired{};
    acquired.xs = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("pml.xs", dipole_count * sizeof(CU_REAL)));
    acquired.ys = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("pml.ys", dipole_count * sizeof(CU_REAL)));
    acquired.zs = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("pml.zs", dipole_count * sizeof(CU_REAL)));
    acquired.inside_map = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
        "pml.inside_map", dipole_count * sizeof(int)));
    if (material_count != 0)
        acquired.material = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "pml.material", material_count * sizeof(CU_REAL)));
    acquired.mask = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("pml.mask", dipole_count * sizeof(CU_REAL)));
    acquired.z_candidates = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("pml.z_candidates", dipole_count * sizeof(CU_REAL)));
    acquired.zinf =
        static_cast<CU_REAL *>(get_device_workspace_buffer_bytes("pml.zinf", sizeof(CU_REAL)));
    acquired.field = static_cast<CUCOMPLEX *>(
        get_device_workspace_buffer_bytes("pml.field", dipole_count * 3 * sizeof(CUCOMPLEX)));
    if (material_count != 0) {
        acquired.epsilon_output = static_cast<CUCOMPLEX *>(
            get_device_workspace_buffer_bytes("pml.epsilon_output",
                                              dipole_count * sizeof(CUCOMPLEX)));
        acquired.contrast_output = static_cast<CUCOMPLEX *>(
            get_device_workspace_buffer_bytes("pml.contrast_output",
                                              dipole_count * sizeof(CUCOMPLEX)));
    }
    if (!acquired.xs || !acquired.ys || !acquired.zs || !acquired.inside_map ||
        !acquired.mask || !acquired.z_candidates ||
        !acquired.zinf || !acquired.field ||
        (material_count != 0 && (!acquired.material || !acquired.epsilon_output ||
                                 !acquired.contrast_output)))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_speckle_phase_buffers(
    size_t phase_count, bool bidirectional,
    SpecklePhaseDeviceBuffers *buffers) {
    if (!buffers || phase_count == 0 ||
        phase_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        phase_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL))
        return false;
    SpecklePhaseDeviceBuffers acquired{};
    acquired.phase_positive = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "spectral_field.phase_positive", phase_count * sizeof(CUCOMPLEX)));
    acquired.uniforms_positive = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "spectral_field.uniforms_positive", phase_count * sizeof(CU_REAL)));
    if (bidirectional) {
        acquired.phase_negative = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "spectral_field.phase_negative", phase_count * sizeof(CUCOMPLEX)));
        acquired.uniforms_negative = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "spectral_field.uniforms_negative", phase_count * sizeof(CU_REAL)));
    }
    if (!acquired.phase_positive || !acquired.uniforms_positive ||
        (bidirectional && (!acquired.phase_negative || !acquired.uniforms_negative)))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_spectral_field_buffers(size_t plane_size, size_t field_count,
                                       SpectralFieldDeviceBuffers *buffers) {
    if (!buffers || plane_size == 0 || field_count == 0 ||
        plane_size > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        field_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)))
        return false;
    SpectralFieldDeviceBuffers acquired{};
    acquired.phase_positive = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "spectral_field.phase_positive", plane_size * sizeof(CUCOMPLEX)));
    acquired.phase_negative = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "spectral_field.phase_negative", plane_size * sizeof(CUCOMPLEX)));
    acquired.spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "spectral_field.spectrum", plane_size * 3 * sizeof(CUCOMPLEX)));
    acquired.field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "spectral_field.field", field_count * 3 * sizeof(CUCOMPLEX)));
    acquired.normalization = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("spectral_field.normalization", sizeof(CU_REAL)));
    if (!acquired.phase_positive || !acquired.phase_negative || !acquired.spectrum ||
        !acquired.field || !acquired.normalization)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_negative_kz_diffraction_buffers(
    size_t plane_size, NegativeKzDiffractionDeviceBuffers *buffers) {
    if (!buffers || plane_size == 0 ||
        plane_size > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)))
        return false;
    NegativeKzDiffractionDeviceBuffers acquired{};
    acquired.random_phases = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.negative_kz.random_phases", plane_size * sizeof(CUCOMPLEX)));
    acquired.spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.negative_kz.spectrum", plane_size * 3 * sizeof(CUCOMPLEX)));
    if (!acquired.random_phases || !acquired.spectrum)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_lens_diffraction_buffers(size_t plane_size, size_t mesh_count,
                                         LensDiffractionDeviceBuffers *buffers) {
    if (!buffers || plane_size == 0 || mesh_count == 0 ||
        plane_size > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        mesh_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)))
        return false;
    LensDiffractionDeviceBuffers acquired{};
    acquired.green_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.lens.green_field", mesh_count * 3 * sizeof(CUCOMPLEX)));
    acquired.layer_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.lens.layer_spectrum", plane_size * 3 * sizeof(CUCOMPLEX)));
    acquired.accumulated_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.lens.accumulated_spectrum", plane_size * 3 * sizeof(CUCOMPLEX)));
    acquired.incident_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.lens.incident_spectrum", plane_size * 3 * sizeof(CUCOMPLEX)));
    acquired.dipole_x = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.lens.direct.x", mesh_count * sizeof(CU_REAL)));
    acquired.dipole_y = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.lens.direct.y", mesh_count * sizeof(CU_REAL)));
    acquired.dipole_z = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.lens.direct.z", mesh_count * sizeof(CU_REAL)));
    acquired.fft_indices = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
        "diffraction.lens.fft_indices", size_t(plane_size ? std::sqrt(double(plane_size)) : 0) * sizeof(int)));
    if (!acquired.green_field || !acquired.layer_spectrum || !acquired.accumulated_spectrum ||
        !acquired.incident_spectrum || !acquired.dipole_x || !acquired.dipole_y ||
        !acquired.dipole_z ||
        !acquired.fft_indices)
        return false;
    const int nfft = int(std::sqrt(double(plane_size)));
    if (size_t(nfft) * size_t(nfft) != plane_size)
        return false;
    {
        std::lock_guard<std::mutex> lock(workspace_buffers_mutex);
        auto plan = lens_fft_plans.find(nfft);
        if (plan == lens_fft_plans.end()) {
            int dimensions[2] = {nfft, nfft};
            cufftHandle handle = 0;
            if (cufftPlanMany(&handle, 2, dimensions, nullptr, 1, int(plane_size),
                              nullptr, 1, int(plane_size), CUFFT_TRANSFORM, 3) != CUFFT_SUCCESS)
                return false;
            plan = lens_fft_plans.emplace(nfft, handle).first;
        }
        acquired.fft_plan = plan->second;
    }
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_diffraction_energy_buffers(
    size_t plane_size, size_t mesh_count, DiffractionEnergyDeviceBuffers *buffers) {
    if (!buffers || plane_size == 0 ||
        mesh_count == 0 ||
        plane_size > std::numeric_limits<size_t>::max() / (6 * sizeof(CUCOMPLEX)) ||
        plane_size > std::numeric_limits<size_t>::max() / (6 * sizeof(CU_REAL)) ||
        mesh_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        mesh_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
        plane_size > size_t(std::numeric_limits<int>::max()))
        return false;
    const size_t nfft_size = size_t(std::sqrt(double(plane_size)));
    if (nfft_size == 0 || nfft_size * nfft_size != plane_size ||
        nfft_size > size_t(std::numeric_limits<int>::max()))
        return false;

    DiffractionEnergyDeviceBuffers acquired{};
    const size_t spectrum_bytes = 3 * plane_size * sizeof(CUCOMPLEX);
    acquired.green_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.green_field", 3 * mesh_count * sizeof(CUCOMPLEX)));
    acquired.incident_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.incident_spectrum", spectrum_bytes));
    acquired.incident_positive_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.incident_positive_spectrum", spectrum_bytes));
    acquired.incident_negative_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.incident_negative_spectrum", spectrum_bytes));
    acquired.positive_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.positive_spectrum", spectrum_bytes));
    acquired.negative_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.negative_spectrum", spectrum_bytes));
    acquired.layer_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.layer_spectrum", spectrum_bytes));
    acquired.phase_positive = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.phase_positive", plane_size * sizeof(CU_REAL)));
    acquired.phase_negative = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.phase_negative", plane_size * sizeof(CU_REAL)));
    acquired.z_coordinates = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.z_coordinates", mesh_count * sizeof(CU_REAL)));
    acquired.reduction_partials = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.reduction_partials", 6 * plane_size * sizeof(CU_REAL)));
    acquired.fft_indices = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.fft_indices", nfft_size * sizeof(int)));
    if (!acquired.green_field || !acquired.incident_spectrum ||
        !acquired.incident_positive_spectrum || !acquired.incident_negative_spectrum ||
        !acquired.positive_spectrum || !acquired.negative_spectrum || !acquired.layer_spectrum ||
        !acquired.phase_positive || !acquired.phase_negative ||
        !acquired.z_coordinates || !acquired.reduction_partials || !acquired.fft_indices)
        return false;

    const int nfft = int(nfft_size);
    {
        std::lock_guard<std::mutex> lock(workspace_buffers_mutex);
        auto plan = diffraction_energy_fft_plans.find(nfft);
        if (plan == diffraction_energy_fft_plans.end()) {
            int dimensions[2] = {nfft, nfft};
            cufftHandle handle = 0;
            if (cufftPlanMany(&handle, 2, dimensions, nullptr, 1, int(plane_size),
                              nullptr, 1, int(plane_size), CUFFT_TRANSFORM, 3) != CUFFT_SUCCESS)
                return false;
            plan = diffraction_energy_fft_plans.emplace(nfft, handle).first;
        }
        acquired.fft_plan = plan->second;
    }
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_diffraction_gaussian_buffers(
    size_t plane_size, bool include_rigorous, DiffractionGaussianDeviceBuffers *buffers) {
    if (!buffers || plane_size == 0 ||
        plane_size > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        plane_size > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
        plane_size > size_t(std::numeric_limits<int>::max()) ||
        plane_size > std::numeric_limits<size_t>::max() / sizeof(int))
        return false;

    DiffractionGaussianDeviceBuffers acquired{};
    acquired.x_coordinates = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.gaussian.x_coordinates", plane_size * sizeof(CU_REAL)));
    acquired.y_coordinates = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.gaussian.y_coordinates", plane_size * sizeof(CU_REAL)));
    acquired.z_coordinates = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.gaussian.z_coordinates", plane_size * sizeof(CU_REAL)));
    acquired.paraxial_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.energy.gaussian.paraxial_field", 3 * plane_size * sizeof(CUCOMPLEX)));
    if (!acquired.x_coordinates || !acquired.y_coordinates || !acquired.z_coordinates ||
        !acquired.paraxial_field)
        return false;
    if (include_rigorous) {
        const size_t partial_count = (plane_size + 255) / 256;
        acquired.rigorous_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "diffraction.energy.gaussian.rigorous_field", 3 * plane_size * sizeof(CUCOMPLEX)));
        acquired.maximum_intensity = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "diffraction.energy.gaussian.maximum_intensity", sizeof(CU_REAL)));
        acquired.maximum_partials = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "diffraction.energy.gaussian.maximum_partials", partial_count * sizeof(CU_REAL)));
        acquired.convergence = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
            "diffraction.energy.gaussian.convergence", plane_size * sizeof(int)));
        acquired.gkp_nodes = static_cast<double *>(get_device_workspace_buffer_bytes(
            "diffraction.energy.gaussian.gkp_nodes", 511 * sizeof(double)));
        acquired.gkp_weights = static_cast<double *>(get_device_workspace_buffer_bytes(
            "diffraction.energy.gaussian.gkp_weights", 1013 * sizeof(double)));
        if (!acquired.rigorous_field || !acquired.maximum_intensity ||
            !acquired.maximum_partials || !acquired.convergence ||
            !acquired.gkp_nodes || !acquired.gkp_weights)
            return false;
    }
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_diffraction_incident_buffers(
    size_t plane_size, size_t plane_count, bool gaussian_path, bool antenna_path,
    DiffractionIncidentDeviceBuffers *buffers) {
    if (!buffers || plane_size == 0 || plane_count == 0 || plane_count > 10 ||
        plane_size > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        plane_count > std::numeric_limits<size_t>::max() / (5 * sizeof(CUCOMPLEX)) ||
        plane_size > size_t(std::numeric_limits<int>::max()))
        return false;

    DiffractionIncidentDeviceBuffers acquired{};
    acquired.spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.incident.spectrum", 3 * plane_size * sizeof(CUCOMPLEX)));
    acquired.plane_parameters = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.incident.plane_parameters", 5 * plane_count * sizeof(CUCOMPLEX)));
    acquired.random_phases = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "diffraction.incident.random_phases", plane_size * sizeof(CUCOMPLEX)));
    if (!acquired.spectrum || !acquired.plane_parameters || !acquired.random_phases)
        return false;

    if (gaussian_path || antenna_path) {
        if (plane_size > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
            plane_size > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
            plane_size > std::numeric_limits<size_t>::max() / sizeof(int))
            return false;
        acquired.xs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.fft.xs", plane_size * sizeof(CU_REAL)));
        acquired.ys = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.fft.ys", plane_size * sizeof(CU_REAL)));
        acquired.zs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.fft.zs", plane_size * sizeof(CU_REAL)));
        acquired.fft_planes = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.fft.planes", 3 * plane_size * sizeof(CUCOMPLEX)));
        if (!acquired.xs || !acquired.ys || !acquired.zs || !acquired.fft_planes)
            return false;
        if (antenna_path) {
            acquired.antenna_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
                "diffraction.incident.antenna.field", 3 * plane_size * sizeof(CUCOMPLEX)));
            acquired.antenna_reference = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
                "diffraction.incident.antenna.reference", 3 * sizeof(CUCOMPLEX)));
            if (!acquired.antenna_field || !acquired.antenna_reference)
                return false;
        }
    }

    if (gaussian_path) {
        if (plane_size > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
            plane_size > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
            plane_size > std::numeric_limits<size_t>::max() / sizeof(int))
            return false;
        acquired.paraxial_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.gaussian.paraxial", 3 * plane_size * sizeof(CUCOMPLEX)));
        acquired.rigorous_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.gaussian.rigorous", 3 * plane_size * sizeof(CUCOMPLEX)));
        acquired.max_intensity = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.gaussian.max_intensity", sizeof(CU_REAL)));
        const size_t partial_count = (plane_size + 255) / 256;
        acquired.max_partials = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.gaussian.max_partials", partial_count * sizeof(CU_REAL)));
        acquired.convergence = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.gaussian.convergence", plane_size * sizeof(int)));
        acquired.gkp_nodes = static_cast<double *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.gaussian.gkp_nodes", 511 * sizeof(double)));
        acquired.gkp_weights = static_cast<double *>(get_device_workspace_buffer_bytes(
            "diffraction.incident.gaussian.gkp_weights", 1013 * sizeof(double)));
        if (!acquired.xs || !acquired.ys || !acquired.zs || !acquired.paraxial_field ||
            !acquired.rigorous_field || !acquired.max_intensity ||
            !acquired.max_partials || !acquired.convergence || !acquired.gkp_nodes ||
            !acquired.gkp_weights)
            return false;

    }

    if (gaussian_path || antenna_path) {
        const size_t nfft_size = size_t(std::sqrt(double(plane_size)));
        if (nfft_size * nfft_size != plane_size ||
            nfft_size > size_t(std::numeric_limits<int>::max()))
            return false;
        const int nfft = int(nfft_size);
        std::lock_guard<std::mutex> lock(workspace_buffers_mutex);
        auto plan = diffraction_incident_fft_plans.find(nfft);
        if (plan == diffraction_incident_fft_plans.end()) {
            int dimensions[2] = {nfft, nfft};
            cufftHandle handle = 0;
            if (cufftPlanMany(&handle, 2, dimensions, nullptr, 1, int(plane_size),
                              nullptr, 1, int(plane_size), CUFFT_TRANSFORM, 3) != CUFFT_SUCCESS)
                return false;
            plan = diffraction_incident_fft_plans.emplace(nfft, handle).first;
        }
        acquired.fft_plan = plan->second;
    }

    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_rigorous_gaussian_buffers(size_t dipole_count,
                                          RigorousGaussianDeviceBuffers *buffers) {
    return get_device_rigorous_gaussian_buffers(dipole_count, true, true, buffers);
}

//--------------------------------------------
bool get_device_rigorous_gaussian_buffers(size_t dipole_count,
                                          bool need_coordinates,
                                          bool need_field,
                                          RigorousGaussianDeviceBuffers *buffers) {
    constexpr size_t gkp_node_count = 511;
    constexpr size_t gkp_weight_count = 1013;
    if (!buffers || dipole_count == 0 ||
        dipole_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        dipole_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
        dipole_count > std::numeric_limits<size_t>::max() / sizeof(int))
        return false;
    RigorousGaussianDeviceBuffers acquired{};
    if (need_coordinates) {
        acquired.xs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "incident.rigorous_gaussian.xs", dipole_count * sizeof(CU_REAL)));
        acquired.ys = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "incident.rigorous_gaussian.ys", dipole_count * sizeof(CU_REAL)));
        acquired.zs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "incident.rigorous_gaussian.zs", dipole_count * sizeof(CU_REAL)));
    }
    acquired.gkp_nodes = static_cast<double *>(get_device_workspace_buffer_bytes(
        "incident.rigorous_gaussian.gkp_nodes", gkp_node_count * sizeof(double)));
    acquired.gkp_weights = static_cast<double *>(get_device_workspace_buffer_bytes(
        "incident.rigorous_gaussian.gkp_weights", gkp_weight_count * sizeof(double)));
    if (need_field)
        acquired.field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "incident.rigorous_gaussian.field", 3 * dipole_count * sizeof(CUCOMPLEX)));
    acquired.convergence = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
        "incident.rigorous_gaussian.convergence", dipole_count * sizeof(int)));
    if ((need_coordinates && (!acquired.xs || !acquired.ys || !acquired.zs)) ||
        !acquired.gkp_nodes || !acquired.gkp_weights ||
        (need_field && !acquired.field) || !acquired.convergence)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_gaussian_fft_buffers(size_t plane_size, size_t mesh_count,
                                     GaussianFftDeviceBuffers *buffers,
                                     bool need_dipole_coordinates) {
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (!buffers || mesh_count == 0 || plane_size == 0 || plane_size > max_size / sizeof(CU_REAL) ||
        plane_size > max_size / sizeof(int) || plane_size > max_size / (3 * sizeof(CUCOMPLEX)) ||
        mesh_count > max_size / (3 * sizeof(CUCOMPLEX)))
        return false;

    GaussianFftDeviceBuffers acquired{};
    acquired.plane_map = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.plane_map", plane_size * sizeof(int)));
    if (need_dipole_coordinates) {
        acquired.dipole_xs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "incident.gaussian_fft.dipole_xs", mesh_count * sizeof(CU_REAL)));
        acquired.dipole_ys = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "incident.gaussian_fft.dipole_ys", mesh_count * sizeof(CU_REAL)));
    }
    acquired.image_planes = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.image_planes", 3 * plane_size * sizeof(CUCOMPLEX)));
    acquired.dipole_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.dipole_field", 3 * mesh_count * sizeof(CUCOMPLEX)));
    acquired.xs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.xs", plane_size * sizeof(CU_REAL)));
    acquired.ys = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.ys", plane_size * sizeof(CU_REAL)));
    acquired.zs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.zs", plane_size * sizeof(CU_REAL)));
    acquired.paraxial_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.paraxial_field", 3 * plane_size * sizeof(CUCOMPLEX)));
    acquired.rigorous_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.rigorous_field", 3 * plane_size * sizeof(CUCOMPLEX)));
    acquired.max_intensity = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("incident.gaussian_fft.max_intensity", sizeof(CU_REAL)));
    const size_t partial_count = (plane_size + 255) / 256;
    acquired.max_partials = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.max_partials", partial_count * sizeof(CU_REAL)));
    acquired.convergence = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.convergence", plane_size * sizeof(int)));
    acquired.gkp_nodes = static_cast<double *>(
        get_device_workspace_buffer_bytes("incident.gaussian_fft.gkp_nodes", 511 * sizeof(double)));
    acquired.gkp_weights = static_cast<double *>(get_device_workspace_buffer_bytes(
        "incident.gaussian_fft.gkp_weights", 1013 * sizeof(double)));
    if (!acquired.plane_map || !acquired.image_planes || !acquired.dipole_field ||
        (need_dipole_coordinates && (!acquired.dipole_xs || !acquired.dipole_ys)))
        return false;
    if (!acquired.xs || !acquired.ys || !acquired.zs || !acquired.paraxial_field ||
        !acquired.rigorous_field || !acquired.max_intensity || !acquired.max_partials ||
        !acquired.convergence || !acquired.gkp_nodes || !acquired.gkp_weights)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_arbitrary_illumination_buffers(size_t grid_count, size_t dipole_count,
                                               ArbitraryIlluminationDeviceBuffers *buffers) {
    return get_device_arbitrary_illumination_buffers(
        grid_count, dipole_count, true, true, buffers);
}

//--------------------------------------------
bool get_device_arbitrary_illumination_buffers(size_t grid_count, size_t dipole_count,
                                               bool need_coordinates, bool need_field,
                                               ArbitraryIlluminationDeviceBuffers *buffers) {
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (!buffers || grid_count == 0 || dipole_count == 0 ||
        grid_count > max_size / sizeof(CUCOMPLEX) || dipole_count > max_size / sizeof(CU_REAL) ||
        dipole_count > max_size / (3 * sizeof(CUCOMPLEX)))
        return false;
    ArbitraryIlluminationDeviceBuffers acquired{};
    if (need_coordinates) {
        acquired.xs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "incident.arbitrary.xs", dipole_count * sizeof(CU_REAL)));
        acquired.ys = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "incident.arbitrary.ys", dipole_count * sizeof(CU_REAL)));
        acquired.zs = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "incident.arbitrary.zs", dipole_count * sizeof(CU_REAL)));
    }
    acquired.grid_x = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.arbitrary.grid_x", grid_count * sizeof(CUCOMPLEX)));
    acquired.grid_y = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.arbitrary.grid_y", grid_count * sizeof(CUCOMPLEX)));
    acquired.grid_z = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.arbitrary.grid_z", grid_count * sizeof(CUCOMPLEX)));
    if (need_field)
        acquired.field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "incident.arbitrary.field", 3 * dipole_count * sizeof(CUCOMPLEX)));
    if ((need_coordinates && (!acquired.xs || !acquired.ys || !acquired.zs)) ||
        !acquired.grid_x || !acquired.grid_y || !acquired.grid_z ||
        (need_field && !acquired.field))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_arbitrary_derivative_buffers(
    size_t grid_count, size_t dipole_count, ArbitraryDerivativeDeviceBuffers *buffers) {
    if (!buffers || grid_count == 0 ||
        grid_count > std::numeric_limits<size_t>::max() / (9 * sizeof(CUCOMPLEX)) ||
        dipole_count == 0 ||
        dipole_count > std::numeric_limits<size_t>::max() / (9 * sizeof(CUCOMPLEX)))
        return false;
    ArbitraryDerivativeDeviceBuffers acquired{};
    acquired.xs = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("incident.arbitrary_derivative.x", dipole_count * sizeof(CU_REAL)));
    acquired.ys = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("incident.arbitrary_derivative.y", dipole_count * sizeof(CU_REAL)));
    acquired.zs = static_cast<CU_REAL *>(
        get_device_workspace_buffer_bytes("incident.arbitrary_derivative.z", dipole_count * sizeof(CU_REAL)));
    acquired.field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.arbitrary_derivative.field", 3 * grid_count * sizeof(CUCOMPLEX)));
    acquired.derivative_scratch = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.arbitrary_derivative.scratch", 3 * grid_count * sizeof(CUCOMPLEX)));
    acquired.derivative_grids = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.arbitrary_derivative.grids", 9 * grid_count * sizeof(CUCOMPLEX)));
    acquired.result = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "incident.arbitrary_derivative.result", 9 * dipole_count * sizeof(CUCOMPLEX)));
    if (!acquired.xs || !acquired.ys || !acquired.zs || !acquired.field ||
        !acquired.derivative_scratch || !acquired.derivative_grids ||
        !acquired.result)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_scattering_result_buffers(size_t dipole_count, size_t angular_count,
                                          size_t spectrum_count, size_t fft_plane_count,
                                          size_t fft_map_count,
                                          ScatteringResultDeviceBuffers *buffers) {
    if (!buffers || dipole_count == 0 ||
        dipole_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        angular_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
        spectrum_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL) ||
        fft_plane_count > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX)) ||
        fft_map_count > std::numeric_limits<size_t>::max() / sizeof(int))
        return false;

    ScatteringResultDeviceBuffers acquired{};
    if (!request_named_buffer(&acquired.dipole_field, "scattering_result.dipole_field",
                              3 * dipole_count) ||
        !request_named_buffer(&acquired.xs, "scattering_result.xs", dipole_count) ||
        !request_named_buffer(&acquired.ys, "scattering_result.ys", dipole_count) ||
        !request_named_buffer(&acquired.zs, "scattering_result.zs", dipole_count) ||
        !request_named_buffer(&acquired.angular_poynting, "scattering_result.angular_poynting",
                              angular_count) ||
        !request_named_buffer(&acquired.theta_values, "scattering_result.theta", angular_count) ||
        !request_named_buffer(&acquired.phi_values, "scattering_result.phi", angular_count) ||
        !request_named_buffer(&acquired.positive_spectrum, "scattering_result.positive_spectrum",
                              spectrum_count) ||
        !request_named_buffer(&acquired.negative_spectrum, "scattering_result.negative_spectrum",
                              spectrum_count) ||
        !request_named_buffer(&acquired.kx_values, "scattering_result.kx", spectrum_count) ||
        !request_named_buffer(&acquired.weighted_scattering,
                              "scattering_result.weighted_scattering",
                              angular_count > spectrum_count ? angular_count : spectrum_count) ||
        !request_named_buffer(&acquired.weighted_asymmetry, "scattering_result.weighted_asymmetry",
                              angular_count > spectrum_count ? angular_count : spectrum_count) ||
        !request_named_buffer(&acquired.reduction_ones, "scattering_result.reduction_ones",
                              angular_count > spectrum_count ? angular_count : spectrum_count) ||
        !request_named_buffer(&acquired.quick_fft_images, "scattering_result.quick_fft_images",
                              3 * fft_plane_count) ||
        !request_named_buffer(&acquired.quick_positive_field,
                              "scattering_result.quick_positive_field", 3 * fft_plane_count) ||
        !request_named_buffer(&acquired.quick_negative_field,
                              "scattering_result.quick_negative_field", 3 * fft_plane_count) ||
        !request_named_buffer(&acquired.quick_fft_map, "scattering_result.quick_fft_map",
                              fft_map_count))
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_quick_diffraction_fft_buffers(
    size_t dipole_count, int nfft, QuickDiffractionFftDeviceBuffers *buffers) {
    if (!buffers || dipole_count == 0 || nfft <= 0 ||
        size_t(nfft) > std::numeric_limits<size_t>::max() / size_t(nfft))
        return false;
    const size_t plane_size = size_t(nfft) * size_t(nfft);
    ScatteringResultDeviceBuffers scattering{};
    if (!get_device_scattering_result_buffers(dipole_count, 1, 1, plane_size,
                                               size_t(nfft), &scattering))
        return false;
    const cufftHandle plan = get_speckle_fft_plan(nfft);
    if (!plan)
        return false;
    QuickDiffractionFftDeviceBuffers acquired{};
    acquired.dipole_field = scattering.dipole_field;
    acquired.fft_images = scattering.quick_fft_images;
    acquired.fft_map = scattering.quick_fft_map;
    acquired.fft_plan = plan;
    if (!acquired.dipole_field || !acquired.fft_images || !acquired.fft_map)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_fourier_image_buffers(size_t plane_count,
                                      FourierImageDeviceBuffers *buffers) {
    if (!buffers || plane_count == 0 ||
        plane_count > std::numeric_limits<size_t>::max() / (6 * sizeof(CUCOMPLEX)))
        return false;
    FourierImageDeviceBuffers acquired{};
    acquired.fields = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "fourier_image.fields", 6 * plane_count * sizeof(CUCOMPLEX)));
    if (!acquired.fields)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_energy_conservation_buffers(size_t dipole_count, size_t fft_plane_count,
                                            size_t fft_map_count,
                                            EnergyConservationDeviceBuffers *buffers) {
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (!buffers || dipole_count == 0 || fft_plane_count == 0 ||
        dipole_count > max_size / (3 * sizeof(CUCOMPLEX)) ||
        fft_plane_count > max_size / (3 * sizeof(CUCOMPLEX)) ||
        fft_plane_count > max_size / sizeof(CU_REAL) || fft_map_count > max_size / sizeof(int))
        return false;

    EnergyConservationDeviceBuffers acquired{};
    acquired.dipole_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "energy_conservation.dipole_field", 3 * dipole_count * sizeof(CUCOMPLEX)));
    acquired.z_planes = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "energy_conservation.z_planes", dipole_count * sizeof(CU_REAL)));
    acquired.fft_map = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
        "energy_conservation.fft_map", fft_map_count * sizeof(int)));
    acquired.fft_images = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "energy_conservation.fft_images", 3 * fft_plane_count * sizeof(CUCOMPLEX)));
    acquired.incident_spectrum = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "energy_conservation.incident_spectrum", 3 * fft_plane_count * sizeof(CUCOMPLEX)));
    acquired.positive_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "energy_conservation.positive_field", 3 * fft_plane_count * sizeof(CUCOMPLEX)));
    acquired.negative_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
        "energy_conservation.negative_field", 3 * fft_plane_count * sizeof(CUCOMPLEX)));
    acquired.flux_incident = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "energy_conservation.flux_incident", fft_plane_count * sizeof(CU_REAL)));
    acquired.flux_reflected = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "energy_conservation.flux_reflected", fft_plane_count * sizeof(CU_REAL)));
    acquired.flux_transmitted = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "energy_conservation.flux_transmitted", fft_plane_count * sizeof(CU_REAL)));
    acquired.energy_inside_na = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "energy_conservation.energy_inside_na", fft_plane_count * sizeof(CU_REAL)));
    acquired.energy_total = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "energy_conservation.energy_total", fft_plane_count * sizeof(CU_REAL)));
    acquired.reduction_ones = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
        "energy_conservation.reduction_ones", fft_plane_count * sizeof(CU_REAL)));
    if (!acquired.dipole_field || !acquired.z_planes || !acquired.fft_map || !acquired.fft_images ||
        !acquired.incident_spectrum || !acquired.positive_field || !acquired.negative_field ||
        !acquired.flux_incident || !acquired.flux_reflected || !acquired.flux_transmitted ||
        !acquired.energy_inside_na || !acquired.energy_total || !acquired.reduction_ones)
        return false;
    *buffers = acquired;
    return true;
}

//--------------------------------------------
bool get_device_optical_force_buffers(const OpticalForceBufferRequest &request,
                                      OpticalForceDeviceBuffers *buffers) {
    const size_t dipole_count = request.dipole_count;
    const size_t mesh_size = request.mesh_size;
    const size_t padded_count = request.padded_count;
    const size_t object_count = request.object_count;
    const bool need_force_output = request.force_density || request.group_force;
    if (!buffers || dipole_count == 0 || mesh_size < dipole_count ||
        dipole_count > std::numeric_limits<size_t>::max() / 3 ||
        (request.force_density &&
         (mesh_size > std::numeric_limits<size_t>::max() / 9 ||
          mesh_size * 9 > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))) ||
        (request.force_fft && padded_count == 0) ||
        (request.force_fft && padded_count > std::numeric_limits<size_t>::max() / 6) ||
        (object_count > 1 && object_count > std::numeric_limits<size_t>::max() / 3) ||
        ((need_force_output || request.torque_density || request.group_torque) &&
         dipole_count > std::numeric_limits<size_t>::max() / sizeof(CU_REAL)) ||
        ((request.force_density || request.force_fft || request.torque_density) &&
         dipole_count * 3 > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX)) ||
        (request.force_fft &&
         padded_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) / 6))
        return false;

    OpticalForceDeviceBuffers acquired{};
    if (request.incident_derivative) {
        if (mesh_size > std::numeric_limits<size_t>::max() / 3 / sizeof(CUCOMPLEX))
            return false;
        acquired.incident_grid = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_force.incident_grid", mesh_size * 3 * sizeof(CUCOMPLEX)));
        acquired.incident_derivative = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_force.incident_derivative", mesh_size * 3 * sizeof(CUCOMPLEX)));
        if (!acquired.incident_grid || !acquired.incident_derivative)
            return false;
    }
    if (request.force_density) {
        acquired.polarization = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_force.polarization", dipole_count * 3 * sizeof(CUCOMPLEX)));
        acquired.field_derivatives = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_force.derivatives", mesh_size * 9 * sizeof(CUCOMPLEX)));
    }
    if (need_force_output) {
        acquired.force_x = static_cast<CU_REAL *>(
            get_device_workspace_buffer_bytes("optical_force.x", dipole_count * sizeof(CU_REAL)));
        acquired.force_y = static_cast<CU_REAL *>(
            get_device_workspace_buffer_bytes("optical_force.y", dipole_count * sizeof(CU_REAL)));
        acquired.force_z = static_cast<CU_REAL *>(
            get_device_workspace_buffer_bytes("optical_force.z", dipole_count * sizeof(CU_REAL)));
        if (!acquired.force_x || !acquired.force_y || !acquired.force_z)
            return false;
    }
    if (request.torque_density) {
        acquired.polarization = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_force.polarization", dipole_count * 3 * sizeof(CUCOMPLEX)));
        acquired.local_field = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_torque.local_field", dipole_count * 3 * sizeof(CUCOMPLEX)));
        acquired.xs = static_cast<CU_REAL *>(
            get_device_workspace_buffer_bytes("optical_torque.xs", dipole_count * sizeof(CU_REAL)));
        acquired.ys = static_cast<CU_REAL *>(
            get_device_workspace_buffer_bytes("optical_torque.ys", dipole_count * sizeof(CU_REAL)));
        acquired.zs = static_cast<CU_REAL *>(
            get_device_workspace_buffer_bytes("optical_torque.zs", dipole_count * sizeof(CU_REAL)));
        if (!acquired.polarization || !acquired.local_field || !acquired.xs || !acquired.ys ||
            !acquired.zs)
            return false;
    }
    if (request.torque_density || request.group_torque) {
        acquired.torque_x = static_cast<CU_REAL *>(
            get_device_workspace_buffer_bytes("optical_torque.x", dipole_count * sizeof(CU_REAL)));
        acquired.torque_y = static_cast<CU_REAL *>(
            get_device_workspace_buffer_bytes("optical_torque.y", dipole_count * sizeof(CU_REAL)));
        acquired.torque_z = static_cast<CU_REAL *>(
            get_device_workspace_buffer_bytes("optical_torque.z", dipole_count * sizeof(CU_REAL)));
        if (!acquired.torque_x || !acquired.torque_y || !acquired.torque_z)
            return false;
    }
    if (request.reduction) {
        acquired.reduction_ones = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "optical_force.reduction_ones", dipole_count * sizeof(CU_REAL)));
        acquired.reduction_results = static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
            "optical_force.reduction_results", size_t(3) * sizeof(CU_REAL)));
        if (!acquired.reduction_ones || !acquired.reduction_results)
            return false;
    }
    if ((request.group_force || request.group_torque) && object_count > 1) {
        if (object_count * 3 > std::numeric_limits<size_t>::max() / sizeof(CU_REAL))
            return false;
        acquired.multi_object.object_ids = static_cast<CU_INT *>(get_device_workspace_buffer_bytes(
            "optical_force.multi_object_ids", dipole_count * sizeof(int)));
        if (!acquired.multi_object.object_ids)
            return false;
        if (request.group_force) {
            acquired.multi_object.force_sums =
                static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
                    "optical_force.multi_force_sums", object_count * 3 * sizeof(CU_REAL)));
            if (!acquired.multi_object.force_sums)
                return false;
        }
        // Couple-specific grouped output is conditional just like the per-dipole
        // torque buffers above: force-only calculations never acquire it.
        if (request.group_torque) {
            acquired.multi_object.torque_sums =
                static_cast<CU_REAL *>(get_device_workspace_buffer_bytes(
                    "optical_torque.multi_torque_sums", object_count * 3 * sizeof(CU_REAL)));
            if (!acquired.multi_object.torque_sums)
                return false;
        }
    }
    // The FFT workspaces are acquired only by the force path that computes
    // Green-tensor derivatives. Force-only runs do not reserve torque state.
    if (request.force_fft) {
        if (mesh_size > std::numeric_limits<size_t>::max() / 3 / sizeof(CUCOMPLEX))
            return false;
        acquired.polarization = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_force.polarization", dipole_count * 3 * sizeof(CUCOMPLEX)));
        acquired.fft.vectors = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_force.fft_vectors", padded_count * 3 * sizeof(CUCOMPLEX)));
        acquired.fft.green_derivatives = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_force.green_derivatives", padded_count * 6 * sizeof(CUCOMPLEX)));
        acquired.fft.output = static_cast<CUCOMPLEX *>(get_device_workspace_buffer_bytes(
            "optical_force.derivative_output", mesh_size * 3 * sizeof(CUCOMPLEX)));
        if (!acquired.polarization || !acquired.fft.vectors || !acquired.fft.green_derivatives ||
            !acquired.fft.output)
            return false;
    }
    *buffers = acquired;
    return true;
}

//--------------------------------------------
cudaError_t solve_small_dense_system_host(CUCOMPLEX *matrix_host, CUCOMPLEX *rhs_host, int order,
                                          int *info_host) {
    if (!matrix_host || !rhs_host || !info_host || order < 1 || order > 9)
        return cudaErrorInvalidValue;
    if (!initialize_cuda_device())
        return cudaErrorNoDevice;

    static std::mutex buffers_mutex;
    std::lock_guard<std::mutex> lock(buffers_mutex);

    SmallDenseSystemDeviceBuffers buffers{};
    if (!get_device_small_dense_system_buffers(&buffers))
        return cudaErrorMemoryAllocation;

    cudaError_t status = copy_host_to_device(buffers.matrix, matrix_host, 81 * sizeof(CUCOMPLEX),
                                             cuda_utils::TransferCategory::Preconditioner);
    if (status == cudaSuccess)
        status = copy_host_to_device(buffers.rhs, rhs_host, 9 * sizeof(CUCOMPLEX),
                                     cuda_utils::TransferCategory::Preconditioner);
    if (status == cudaSuccess)
        status = solve_small_dense_system_device_wrapper(buffers.matrix, buffers.rhs, order,
                                                         buffers.info);
    if (status == cudaSuccess)
        status = copy_device_to_host(rhs_host, buffers.rhs, 9 * sizeof(CUCOMPLEX),
                                     cuda_utils::TransferCategory::Preconditioner);
    if (status == cudaSuccess)
        status = copy_device_to_host(info_host, buffers.info, sizeof(int),
                                     cuda_utils::TransferCategory::Preconditioner);
    return status;
}

} // namespace cuda_data

namespace {

//--------------------------------------------
// Required diagnostic D2H callback, registered by initialize_cuda_device().
cudaError_t diagnostic_device_to_host(void *host_destination, const void *device_source,
                                      size_t bytes) {
    return cuda_data::copy_device_to_host(host_destination, device_source, bytes,
                                          cuda_utils::TransferCategory::Diagnostic);
}

CUCOMPLEX *device_vectors = nullptr;
size_t device_vectors_capacity = 0;
CUCOMPLEX *device_auxiliary_vector = nullptr;
size_t device_auxiliary_vector_capacity = 0;
CUCOMPLEX *device_result_vectors = nullptr;
size_t device_result_vectors_capacity = 0;
GreenTensorDevicePointers device_green{};
size_t device_green_capacity = 0;
size_t device_green_tensor_count = 0;
bool device_green_generated = false;
// Widefield can regenerate a physically different Green tensor on a distinct
// grid. Its six components must never overwrite the main solver's tensor.
GreenTensorDevicePointers device_widefield_green{};
size_t device_widefield_green_capacity = 0;
size_t device_widefield_green_tensor_count = 0;
CUCOMPLEX *device_multilayer_green = nullptr;
size_t device_multilayer_green_capacity = 0;
CUCOMPLEX *device_scalar_green = nullptr;
size_t device_scalar_green_capacity = 0;
const COMPLEX *device_scalar_green_source = nullptr;
CUCOMPLEX *device_polarizability = nullptr;
size_t device_polarizability_capacity = 0;
std::mutex fft_plan_mutex;
std::map<std::tuple<int, int, int>, cufftHandle> fft_plans;
std::map<int, cufftHandle> speckle_fft_plans;

struct ComplexBufferEntry {
    CUCOMPLEX *device = nullptr;
    size_t capacity = 0;
    bool initialized_from_host = false;
    bool device_current = false;
};

struct GeneratedGreenAlias {
    CUCOMPLEX *component_device = nullptr;
    size_t component_index = 0;
    size_t count = 0;
};

std::unordered_map<const COMPLEX *, ComplexBufferEntry> ffx_buffers;
std::unordered_map<const COMPLEX *, ComplexBufferEntry> ffb_buffers;
std::unordered_map<const COMPLEX *, ComplexBufferEntry> gpbicg_xi_buffers;
std::unordered_map<const COMPLEX *, ComplexBufferEntry> gpbicg_xr_buffers;
std::unordered_map<const COMPLEX *, ComplexBufferEntry> gpbicg_b_buffers;
std::unordered_map<const COMPLEX *, ComplexBufferEntry> gpbicg_work_buffers;
std::unordered_map<const COMPLEX *, ComplexBufferEntry> gpbicor_ffloc_buffers;
std::unordered_map<const COMPLEX *, ComplexBufferEntry> tfqmr_x_buffers;

struct DeviceBufferAlias {
    CUCOMPLEX *device = nullptr;
    size_t capacity = 0;
};

struct DeviceBufferSource {
    const CUCOMPLEX *device = nullptr;
    size_t capacity = 0;
};

std::unordered_map<const void *, DeviceBufferAlias> gpbicg_device_b_aliases;
std::unordered_map<const void *, DeviceBufferSource> gpbicg_device_b_sources;
// Residency is keyed by host address identity and shared by legacy solver APIs.
std::unordered_map<const void *, bool> resident_host_buffers;
std::unordered_map<const COMPLEX *, bool> resident_gpbicg_rhs_buffers;
std::unordered_map<const void *, GeneratedGreenAlias> generated_green_aliases;

struct ResultVectorAlias {
    const CUCOMPLEX *device = nullptr;
    size_t count = 0;
};

std::unordered_map<const void *, ResultVectorAlias> result_vector_aliases;
std::unordered_set<const void *> device_only_result_vector_requests;

//--------------------------------------------
template <typename HostComplex>
CUCOMPLEX *get_or_create_buffer(std::unordered_map<const HostComplex *, ComplexBufferEntry> &buffers,
                                const HostComplex *cpu_pointer, size_t count, const char *label) {
    if (!cpu_pointer || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return nullptr;

    ComplexBufferEntry &entry = buffers[cpu_pointer];
    if (entry.device && entry.capacity >= count)
        return entry.device;

    if (entry.device) {
        const cudaError_t free_status = cudaFree(entry.device);
        if (free_status != cudaSuccess) {
            std::fprintf(stderr, "%s: cudaFree failed: %s\n", label,
                         cudaGetErrorString(free_status));
            return nullptr;
        }
        entry.device = nullptr;
        entry.capacity = 0;
        entry.initialized_from_host = false;
    }

    const cudaError_t alloc_status = cudaMalloc(&entry.device, count * sizeof(CUCOMPLEX));
    if (alloc_status != cudaSuccess) {
        entry.device = nullptr;
        std::fprintf(stderr, "%s: cudaMalloc failed: %s\n", label,
                     cudaGetErrorString(alloc_status));
        return nullptr;
    }

    entry.capacity = count;
    return entry.device;
}

} // namespace

//--------------------------------------------
extern "C" bool initialize_cuda_device() {
    std::lock_guard<std::mutex> lock(cuda_data::cuda_device_state_mutex);
    if (cuda_data::cuda_device_initialized)
        return true;

    int device = 0;
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr, "CUDA device initialization failed: %s\n",
                     status == cudaSuccess ? "no CUDA device found" : cudaGetErrorString(status));
        return false;
    }

    status = cudaGetDevice(&device);
    if (status != cudaSuccess || device < 0 || device >= device_count)
        device = 0;
    status = cudaSetDevice(device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA device initialization failed: cudaSetDevice: %s\n",
                     cudaGetErrorString(status));
        return false;
    }
    cuda_data::cuda_device_id = device;

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA device initialization failed: cudaGetDeviceProperties: %s\n",
                     cudaGetErrorString(status));
        return false;
    }

    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);

    std::fprintf(stderr,
                 "CUDA device initialized for IF-DDA\n"
                 "  GPU                    %s\n"
                 "  Selected device        %d of %d\n"
                 "  Compute capability     %d.%d\n"
                 "  Global memory          %.2f GiB\n"
                 "  Multiprocessors        %d\n"
                 "  CUDA runtime           %d.%d\n"
                 "  CUDA driver            %d.%d\n"
                 "  Solver precision       %s\n",
                 properties.name, device, device_count, properties.major, properties.minor,
                 double(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                 properties.multiProcessorCount, runtime_version / 1000,
                 (runtime_version % 1000) / 10, driver_version / 1000, (driver_version % 1000) / 10,
#if PRODUITMATVECTFFT_SINGLE_PRECISION
                 "single"
#else
                 "double"
#endif
    );
    ifdda_set_host_download_callback(&diagnostic_device_to_host);
    cuda_data::cuda_device_initialized = true;
    return true;
}

//--------------------------------------------
extern "C" bool ifdda_reset_cuda_device_state() {
    std::lock_guard<std::mutex> lock(cuda_data::cuda_device_state_mutex);
    if (cuda_data::cuda_device_initialized) {
        cudaError_t status = cudaSetDevice(cuda_data::cuda_device_id);
        if (status == cudaSuccess)
            status = cudaDeviceReset();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "ifdda_reset_cuda_device_state: cudaDeviceReset failed: %s\n",
                         cudaGetErrorString(status));
            return false;
        }
    }

    cuda_data::cuda_device_initialized = false;
    cuda_data::cuda_device_id = 0;
    device_vectors = nullptr;
    device_vectors_capacity = 0;
    device_auxiliary_vector = nullptr;
    device_auxiliary_vector_capacity = 0;
    device_result_vectors = nullptr;
    device_result_vectors_capacity = 0;
    device_green = {};
    device_green_capacity = 0;
    device_green_tensor_count = 0;
    device_green_generated = false;
    device_widefield_green = {};
    device_widefield_green_capacity = 0;
    device_widefield_green_tensor_count = 0;
    device_multilayer_green = nullptr;
    device_multilayer_green_capacity = 0;
    device_scalar_green = nullptr;
    device_scalar_green_capacity = 0;
    device_scalar_green_source = nullptr;
    // cudaDeviceReset has invalidated these five calculation-owned allocations.
    // Clearing the host pointers and capacities permits exactly one new
    // allocation when the next IF-DDA calculation starts.
    {
        std::lock_guard<std::mutex> persistent_lock(cuda_data::persistent_arrays_mutex);
        cuda_data::dipole_map = nullptr;
        cuda_data::dipole_map_capacity = 0;
        cuda_data::tapdip = nullptr;
        cuda_data::tapdip_capacity = 0;
        cuda_data::polarisa = nullptr;
        cuda_data::polarisa_capacity = 0;
        cuda_data::epsilon = nullptr;
        cuda_data::epsilon_capacity = 0;
        cuda_data::ff0 = nullptr;
        cuda_data::ff0_capacity = 0;
    }
    {
        std::lock_guard<std::mutex> upload_lock(cuda_data::tapdip_upload_mutex);
        cuda_data::uploaded_tabdip_host_values.clear();
    }
    device_polarizability = nullptr;
    device_polarizability_capacity = 0;

    cuda_data::workspace_buffers.clear();
    cuda_data::geometry_coordinate_aliases.clear();
    cuda_data::clear_host_regular_geometry();
    cuda_data::lens_fft_plans.clear();
    cuda_data::diffraction_energy_fft_plans.clear();
    cuda_data::diffraction_incident_fft_plans.clear();
    cuda_data::bright_field_fft_plans.clear();
    cuda_data::inhomogeneous_material_fft_plans.clear();
    cuda_data::chan_preconditioner_fft_plans.clear();
    cuda_data::multilayer_layer_fft_plans.clear();
    cuda_data::multilayer_image_fft_plans.clear();
    fft_plans.clear();
    speckle_fft_plans.clear();
    ffx_buffers.clear();
    ffb_buffers.clear();
    gpbicg_xi_buffers.clear();
    gpbicg_xr_buffers.clear();
    gpbicg_b_buffers.clear();
    gpbicg_work_buffers.clear();
    gpbicor_ffloc_buffers.clear();
    tfqmr_x_buffers.clear();
    gpbicg_device_b_aliases.clear();
    gpbicg_device_b_sources.clear();
    resident_host_buffers.clear();
    resident_gpbicg_rhs_buffers.clear();
    generated_green_aliases.clear();
    result_vector_aliases.clear();
    device_only_result_vector_requests.clear();
    ifdda_reset_incident_gaussian_gkp_state();
    ifdda_reset_gaussianfft_gkp_state();
    ifdda_reset_beampropagation_gkp_state();
    cuda_utils::reset_after_device_reset();
    return true;
}


//--------------------------------------------
extern "C" CUCOMPLEX *get_device_vectors(size_t count) {
    if (!initialize_cuda_device())
        return nullptr;

    if (count == 0 || count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return nullptr;

    if (device_vectors && device_vectors_capacity >= count)
        return device_vectors;

    if (device_vectors) {
        const cudaError_t free_status = cudaFree(device_vectors);
        if (free_status != cudaSuccess) {
            std::fprintf(stderr, "get_device_vectors: cudaFree failed: %s\n",
                         cudaGetErrorString(free_status));
            return nullptr;
        }
        device_vectors = nullptr;
        device_vectors_capacity = 0;
    }

    const size_t bytes = count * sizeof(CUCOMPLEX);
    const cudaError_t alloc_status = cudaMalloc(&device_vectors, bytes);
    if (alloc_status != cudaSuccess) {
        device_vectors = nullptr;
        std::fprintf(stderr, "get_device_vectors: cudaMalloc failed: %s\n",
                     cudaGetErrorString(alloc_status));
        return nullptr;
    }

    device_vectors_capacity = count;
    return device_vectors;
}

//--------------------------------------------
extern "C" CUCOMPLEX *get_device_auxiliary_vector(size_t count) {
    if (!initialize_cuda_device() || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return nullptr;
    if (device_auxiliary_vector && device_auxiliary_vector_capacity >= count)
        return device_auxiliary_vector;
    if (device_auxiliary_vector) {
        const cudaError_t free_status = cudaFree(device_auxiliary_vector);
        if (free_status != cudaSuccess) {
            std::fprintf(stderr, "get_device_auxiliary_vector: cudaFree failed: %s\n",
                         cudaGetErrorString(free_status));
            return nullptr;
        }
        device_auxiliary_vector = nullptr;
        device_auxiliary_vector_capacity = 0;
    }
    const cudaError_t alloc_status =
        cudaMalloc(&device_auxiliary_vector, count * sizeof(CUCOMPLEX));
    if (alloc_status != cudaSuccess) {
        std::fprintf(stderr, "get_device_auxiliary_vector: cudaMalloc failed: %s\n",
                     cudaGetErrorString(alloc_status));
        return nullptr;
    }
    device_auxiliary_vector_capacity = count;
    return device_auxiliary_vector;
}

//--------------------------------------------
extern "C" CUCOMPLEX *get_device_result_vectors(size_t count) {
    if (!initialize_cuda_device() || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return nullptr;
    if (device_result_vectors && device_result_vectors_capacity >= count)
        return device_result_vectors;
    if (device_result_vectors) {
        const cudaError_t free_status = cudaFree(device_result_vectors);
        if (free_status != cudaSuccess) {
            std::fprintf(stderr, "get_device_result_vectors: cudaFree failed: %s\\n",
                         cudaGetErrorString(free_status));
            return nullptr;
        }
        device_result_vectors = nullptr;
        device_result_vectors_capacity = 0;
    }
    const cudaError_t alloc_status =
        cudaMalloc(&device_result_vectors, count * sizeof(CUCOMPLEX));
    if (alloc_status != cudaSuccess) {
        std::fprintf(stderr, "get_device_result_vectors: cudaMalloc failed: %s\\n",
                     cudaGetErrorString(alloc_status));
        return nullptr;
    }
    device_result_vectors_capacity = count;
    return device_result_vectors;
}

//--------------------------------------------
extern "C" bool register_device_result_vector_alias(const void *host_key,
                                                    const CUCOMPLEX *device_data, size_t count) {
    if (!initialize_cuda_device() || !host_key || !device_data || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;
    result_vector_aliases[host_key] = {device_data, count};
    return true;
}

//--------------------------------------------
extern "C" const CUCOMPLEX *get_device_result_vector_alias(const void *host_key,
                                                           size_t count) {
    if (!host_key || count == 0)
        return nullptr;
    const auto found = result_vector_aliases.find(host_key);
    if (found == result_vector_aliases.end() || !found->second.device ||
        found->second.count < count)
        return nullptr;
    return found->second.device;
}

//--------------------------------------------
extern "C" void clear_device_result_vector_alias(const void *host_key) {
    if (host_key)
        result_vector_aliases.erase(host_key);
}

//--------------------------------------------
extern "C" bool request_device_only_result_vector(const void *host_key) {
    if (!host_key)
        return false;
    device_only_result_vector_requests.insert(host_key);
    return true;
}

//--------------------------------------------
extern "C" bool take_device_only_result_vector_request(const void *host_key) {
    if (!host_key)
        return false;
    return device_only_result_vector_requests.erase(host_key) != 0;
}

//--------------------------------------------
extern "C" void clear_device_only_result_vector_request(const void *host_key) {
    if (host_key)
        device_only_result_vector_requests.erase(host_key);
}

//--------------------------------------------
namespace cuda_data {
void mark_green_generated() {
    if (device_green)
        device_green_generated = true;
}

GreenTensorDevicePointers get_green(size_t tensor_count) {
    if (!initialize_cuda_device() || tensor_count == 0 ||
        tensor_count > size_t(2147483647) / sizeof(CUCOMPLEX)) {
        std::fprintf(stderr, "get_green: component allocation exceeds Windows 2 GiB limit\n");
        return {};
    }
    if (device_green_tensor_count && tensor_count != device_green_tensor_count) {
        std::fprintf(stderr,
                     "get_green: tensor extent changed from %zu to %zu; "
                     "restart IF-DDA rather than altering resident pointers\n",
                     device_green_tensor_count, tensor_count);
        return {};
    }
    if (!device_green_tensor_count) {
        device_green_tensor_count = tensor_count;
        device_green_capacity = tensor_count;
    }
    for (int component = 0; component < 6; ++component) {
        if (device_green.data[component])
            continue;
        const cudaError_t allocated =
            cudaMalloc(&device_green.data[component], tensor_count * sizeof(CUCOMPLEX));
        if (allocated != cudaSuccess) {
            device_green.data[component] = nullptr;
            std::fprintf(stderr, "get_green: cudaMalloc component %d failed: %s\n",
                         component, cudaGetErrorString(allocated));
            return {}; // Keep previously reserved component pointers stable until reset.
        }
    }
    return device_green;
}

GreenTensorDevicePointers get_widefield_green(size_t tensor_count) {
    if (!initialize_cuda_device() || tensor_count == 0 ||
        tensor_count > size_t(2147483647) / sizeof(CUCOMPLEX)) {
        std::fprintf(stderr, "get_widefield_green: component exceeds Windows 2 GiB limit\n");
        return {};
    }
    if (device_widefield_green_tensor_count &&
        tensor_count != device_widefield_green_tensor_count) {
        std::fprintf(stderr, "get_widefield_green: tensor extent changed from %zu to %zu; "
                     "the widefield buffers are fixed until CUDA reset\n",
                     device_widefield_green_tensor_count, tensor_count);
        return {};
    }
    if (!device_widefield_green_tensor_count) {
        device_widefield_green_tensor_count = tensor_count;
        device_widefield_green_capacity = tensor_count;
    }
    for (int component = 0; component < 6; ++component) {
        if (device_widefield_green.data[component])
            continue;
        const cudaError_t allocated = cudaMalloc(&device_widefield_green.data[component],
                                                tensor_count * sizeof(CUCOMPLEX));
        if (allocated != cudaSuccess) {
            device_widefield_green.data[component] = nullptr;
            std::fprintf(stderr, "get_widefield_green: cudaMalloc component %d failed: %s\n",
                         component, cudaGetErrorString(allocated));
            return {};
        }
    }
    return device_widefield_green;
}
} // namespace cuda_data

//--------------------------------------------
GreenTensorDevicePointers get_device_green(const COMPLEX *const cpu_tensors[6], size_t tensor_count) {
    if (!cpu_tensors)
        return {};
    for (int component = 0; component < 6; ++component)
        if (!cpu_tensors[component])
            return {};

    const bool already_resident = bool(device_green);
    GreenTensorDevicePointers green = cuda_data::get_green(tensor_count);
    if (!green)
        return {};
    if (already_resident && device_green_generated)
        return device_green;
    if (already_resident) {
        const void *keys[6] = {
            cpu_tensors[0], cpu_tensors[1], cpu_tensors[2],
            cpu_tensors[3], cpu_tensors[4], cpu_tensors[5]};
        if (!find_device_green_for_host_keys(keys, tensor_count)) {
            std::fprintf(stderr, "get_device_green: no matching resident Green tensor keys\n");
            return {};
        }
        return green;
    }

    // Retain the host-only producer fallback for callers that supply tensors
    // before a GPU Green generator has run; never reinterpret CPU pointers.
    for (int component = 0; component < 6; ++component) {
        const cudaError_t status = cuda_data::copy_host_complex_to_device(
            green[component], cpu_tensors[component],
            tensor_count, cuda_utils::TransferCategory::GreenTensor);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "get_device_green: upload component %d failed: %s\n",
                         component, cudaGetErrorString(status));
            return {};  // Persistent allocation is freed only by CUDA reset.
        }
    }
    // A host-only producer must establish the same six identities as a GPU
    // producer; subsequent product calls must not reject its resident tensor
    // or upload it a second time.
    generated_green_aliases.clear();
    for (size_t component = 0; component < 6; ++component)
        generated_green_aliases[cpu_tensors[component]] = {
            green[component], size_t(component), tensor_count};
    return green;
}

//--------------------------------------------
GreenTensorDevicePointers find_ready_device_green(
    const COMPLEX *const cpu_tensors[6], size_t tensor_count) {
    if (!cpu_tensors || !device_green || !tensor_count ||
        tensor_count != device_green_tensor_count)
        return {};
    for (int i = 0; i < 6; ++i)
        if (!cpu_tensors[i])
            return {};
    // A GPU-generated Green can be consumed by another Fortran host mirror
    // without a host upload, just as in the inverse driver.
    if (device_green_generated)
        return device_green;
    const void *keys[6] = {
        cpu_tensors[0], cpu_tensors[1], cpu_tensors[2],
        cpu_tensors[3], cpu_tensors[4], cpu_tensors[5]};
    return find_device_green_for_host_keys(keys, tensor_count);
}

//--------------------------------------------
extern "C" int ifdda_refresh_green_tensors_cuda(const COMPLEX *tensor_xx,
                                                const COMPLEX *tensor_xy,
                                                const COMPLEX *tensor_xz,
                                                const COMPLEX *tensor_yy,
                                                const COMPLEX *tensor_yz,
                                                const COMPLEX *tensor_zz, size_t tensor_count) {
    const COMPLEX *cpu_tensors[6] = {tensor_xx, tensor_xy, tensor_xz,
                                       tensor_yy, tensor_yz, tensor_zz};
    GreenTensorDevicePointers refreshed_device_green = get_device_green(cpu_tensors, tensor_count);
    if (!refreshed_device_green)
        return 1;

    for (int component = 0; component < 6; ++component) {
        const cudaError_t status = cuda_data::copy_host_to_device(
            refreshed_device_green[component], cpu_tensors[component],
            tensor_count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::GreenTensor);
        if (status != cudaSuccess) {
            std::fprintf(stderr,
                         "ifdda_refresh_green_tensors_cuda: component %d upload failed: %s\n",
                         component, cudaGetErrorString(status));
            return 2;
        }
    }
    return 0;
}

//--------------------------------------------
GreenTensorDevicePointers get_device_green_output(const COMPLEX *const cpu_tensor_keys[6],
                                              size_t tensor_count) {
    if (!cpu_tensor_keys)
        return {};
    const void *host_keys[6] = {
        cpu_tensor_keys[0], cpu_tensor_keys[1], cpu_tensor_keys[2],
        cpu_tensor_keys[3], cpu_tensor_keys[4], cpu_tensor_keys[5]};
    return get_device_green_output_for_host_keys(host_keys, tensor_count);
}

//--------------------------------------------
GreenTensorDevicePointers get_device_green_output_for_host_keys(
    const void *const host_tensor_keys[6], size_t tensor_count) {
    if (!host_tensor_keys)
        return {};
    for (int component = 0; component < 6; ++component)
        if (!host_tensor_keys[component])
            return {};
    GreenTensorDevicePointers green = cuda_data::get_green(tensor_count);
    if (!green)
        return {};
    device_green_generated = false;
    // A new GPU production supersedes any prior host identities. Never leave
    // stale aliases pointing at components that have just been recomputed.
    generated_green_aliases.clear();
    for (size_t component = 0; component < 6; ++component)
        generated_green_aliases[host_tensor_keys[component]] = {
            green[component], component, tensor_count};
    return green;
}

//--------------------------------------------
GreenTensorDevicePointers find_device_green_for_host_keys(
    const void *const host_tensor_keys[6], size_t tensor_count) {
    if (!host_tensor_keys || !device_green || !tensor_count ||
        tensor_count != device_green_tensor_count)
        return {};
    for (size_t component = 0; component < 6; ++component) {
        if (!host_tensor_keys[component])
            return {};
        const auto it = generated_green_aliases.find(host_tensor_keys[component]);
        if (it == generated_green_aliases.end() ||
            it->second.component_device != device_green[component] ||
            it->second.component_index != component ||
            it->second.count != tensor_count)
            return {};
    }
    return device_green;
}

//--------------------------------------------
extern "C" CUCOMPLEX *get_device_multilayer_green_output(const COMPLEX *const cpu_tensor_keys[6],
                                                         size_t tensor_count) {
    if (!initialize_cuda_device() || !cpu_tensor_keys || tensor_count == 0 ||
        tensor_count > std::numeric_limits<size_t>::max() / (6 * sizeof(CUCOMPLEX)))
        return nullptr;
    for (int i = 0; i < 6; ++i)
        if (!cpu_tensor_keys[i])
            return nullptr;
    const size_t required = 6 * tensor_count;
    if (device_multilayer_green && device_multilayer_green_capacity >= required)
        return device_multilayer_green;
    if (device_multilayer_green) {
        const cudaError_t status = cudaFree(device_multilayer_green);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "get_device_multilayer_green_output: cudaFree failed: %s\n",
                         cudaGetErrorString(status));
            return nullptr;
        }
        device_multilayer_green = nullptr;
        device_multilayer_green_capacity = 0;
    }
    const cudaError_t status = cudaMalloc(&device_multilayer_green,
                                          required * sizeof(CUCOMPLEX));
    if (status != cudaSuccess) {
        device_multilayer_green = nullptr;
        std::fprintf(stderr, "get_device_multilayer_green_output: cudaMalloc failed: %s\n",
                     cudaGetErrorString(status));
        return nullptr;
    }
    device_multilayer_green_capacity = required;
    return device_multilayer_green;
}

//--------------------------------------------
extern "C" CUCOMPLEX *get_device_scalar_green(const COMPLEX *cpu_tensor, size_t tensor_count) {
    if (!initialize_cuda_device() || !cpu_tensor || tensor_count == 0 ||
        tensor_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return nullptr;

    // Some scalar microscopy solves intentionally use the xx component of the
    // vector Green tensor. It was generated into separate GPU allocations;
    // resolve its host key to the correct device component instead
    // of uploading the (uninitialized) CPU mirror.
    const auto generated = generated_green_aliases.find(cpu_tensor);
    if (generated != generated_green_aliases.end() && generated->second.component_device &&
        generated->second.count >= tensor_count) {
        return generated->second.component_device;
    }

    if (device_scalar_green_capacity < tensor_count) {
        if (device_scalar_green) {
            const cudaError_t free_status = cudaFree(device_scalar_green);
            if (free_status != cudaSuccess) {
                std::fprintf(stderr, "get_device_scalar_green: cudaFree failed: %s\n",
                             cudaGetErrorString(free_status));
                return nullptr;
            }
            device_scalar_green = nullptr;
            device_scalar_green_capacity = 0;
        }
        const cudaError_t alloc_status = cudaMalloc(&device_scalar_green,
                                                    tensor_count * sizeof(CUCOMPLEX));
        if (alloc_status != cudaSuccess) {
            device_scalar_green = nullptr;
            std::fprintf(stderr, "get_device_scalar_green: cudaMalloc failed: %s\n",
                         cudaGetErrorString(alloc_status));
            return nullptr;
        }
        device_scalar_green_capacity = tensor_count;
        device_scalar_green_source = nullptr;
    }

    if (device_scalar_green_source != cpu_tensor) {
        const cudaError_t copy_status = cuda_data::copy_host_to_device(
            device_scalar_green, cpu_tensor, tensor_count * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::GreenTensor);
        if (copy_status != cudaSuccess) {
            std::fprintf(stderr, "get_device_scalar_green: H2D copy failed: %s\n",
                         cudaGetErrorString(copy_status));
            return nullptr;
        }
        device_scalar_green_source = cpu_tensor;
    }
    return device_scalar_green;
}

//--------------------------------------------
extern "C" CUCOMPLEX *get_device_scalar_green_output(const COMPLEX *cpu_tensor_key,
                                                     size_t tensor_count) {
    if (!initialize_cuda_device() || !cpu_tensor_key || tensor_count == 0 ||
        tensor_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return nullptr;

    if (device_scalar_green_capacity < tensor_count) {
        if (device_scalar_green) {
            const cudaError_t free_status = cudaFree(device_scalar_green);
            if (free_status != cudaSuccess) {
                std::fprintf(stderr, "get_device_scalar_green_output: cudaFree failed: %s\n",
                             cudaGetErrorString(free_status));
                return nullptr;
            }
            device_scalar_green = nullptr;
            device_scalar_green_capacity = 0;
        }
        const cudaError_t alloc_status = cudaMalloc(&device_scalar_green,
                                                    tensor_count * sizeof(CUCOMPLEX));
        if (alloc_status != cudaSuccess) {
            device_scalar_green = nullptr;
            std::fprintf(stderr, "get_device_scalar_green_output: cudaMalloc failed: %s\n",
                         cudaGetErrorString(alloc_status));
            return nullptr;
        }
        device_scalar_green_capacity = tensor_count;
    }

    device_scalar_green_source = cpu_tensor_key;
    return device_scalar_green;
}

//--------------------------------------------
extern "C" CU_INT *get_device_tabdip(const int *cpu_tabdip, size_t count) {
    if (!cpu_tabdip)
        return nullptr;
    CU_INT* device = cuda_data::get_tapdip(count);
    if (!device)
        return nullptr;
    // Only the CPU-producing interface uploads tabdip. Skip repeated transfers
    // when the CPU values are unchanged, but detect in-place host map updates.
    // GPU consumers use cuda_data::get_tapdip(count) without any upload.
    std::lock_guard<std::mutex> upload_lock(cuda_data::tapdip_upload_mutex);
    if (cuda_data::uploaded_tabdip_host_values.size() == count &&
        std::memcmp(cuda_data::uploaded_tabdip_host_values.data(), cpu_tabdip,
                    count * sizeof(CU_INT)) == 0)
        return device;
    const cudaError_t status = cuda_data::copy_host_to_device(
        device, cpu_tabdip, count * sizeof(CU_INT),
        cuda_utils::TransferCategory::Geometry);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "get_device_tabdip: H2D upload failed: %s\n",
                     cudaGetErrorString(status));
        return nullptr;
    }
    cuda_data::uploaded_tabdip_host_values.assign(cpu_tabdip, cpu_tabdip + count);
    return device;
}

//--------------------------------------------
extern "C" CU_INT* get_tapdip(const int* cpu_tabdip, size_t count) {
    return get_device_tabdip(cpu_tabdip, count);
}

//--------------------------------------------
extern "C" CUCOMPLEX *copy_polarizability_to_device(const COMPLEX *cpu_polarizability,
                                                    size_t count) {
    if (!initialize_cuda_device() || !cpu_polarizability || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return nullptr;
    if (device_polarizability_capacity < count) {
        if (device_polarizability) {
            const cudaError_t free_status = cudaFree(device_polarizability);
            if (free_status != cudaSuccess) {
                std::fprintf(stderr, "polarizability GPU buffer cudaFree failed: %s\n",
                             cudaGetErrorString(free_status));
                return nullptr;
            }
            device_polarizability = nullptr;
            device_polarizability_capacity = 0;
        }
        const cudaError_t alloc_status = cudaMalloc(
            &device_polarizability, count * sizeof(CUCOMPLEX));
        if (alloc_status != cudaSuccess) {
            std::fprintf(stderr, "polarizability GPU buffer cudaMalloc failed: %s\n",
                         cudaGetErrorString(alloc_status));
            device_polarizability = nullptr;
            return nullptr;
        }
        device_polarizability_capacity = count;
    }
    const cudaError_t copy_status = cuda_data::copy_host_to_device(
        device_polarizability, cpu_polarizability, count * sizeof(CUCOMPLEX),
        cuda_utils::TransferCategory::Material);
    if (copy_status != cudaSuccess) {
        std::fprintf(stderr, "polarizability GPU buffer H2D copy failed: %s\n",
                     cudaGetErrorString(copy_status));
        return nullptr;
    }
    return device_polarizability;
}

//--------------------------------------------
extern "C" cufftHandle get_fft_plan(int nx, int ny, int nz) {
    if (!initialize_cuda_device())
        return 0;

    if (nx <= 0 || ny <= 0 || nz <= 0)
        return 0;
    std::lock_guard<std::mutex> lock(fft_plan_mutex);
    const auto key = std::make_tuple(nx, ny, nz);
    const auto found = fft_plans.find(key);
    if (found != fft_plans.end())
        return found->second;

    cufftHandle plan = 0;
    const cufftResult plan_status = cufftPlan3d(&plan, nz, ny, nx, CUFFT_TRANSFORM);
    if (plan_status != CUFFT_SUCCESS) {
        std::fprintf(stderr, "get_fft_plan: cufftPlan3d failed (status %d)\n",
                     static_cast<int>(plan_status));
        return 0;
    }
    fft_plans.emplace(key, plan);
    return plan;
}

//--------------------------------------------
extern "C" cufftHandle get_speckle_fft_plan(int nfft) {
    if (!initialize_cuda_device() || nfft <= 0)
        return 0;
    if (nfft > std::numeric_limits<int>::max() / nfft)
        return 0;
    std::lock_guard<std::mutex> lock(fft_plan_mutex);
    const auto found = speckle_fft_plans.find(nfft);
    if (found != speckle_fft_plans.end())
        return found->second;
    int dimensions[2] = {nfft, nfft};
    cufftHandle plan = 0;
    const cufftResult status =
        cufftPlanMany(&plan, 2, dimensions, nullptr, 1, nfft * nfft, nullptr, 1,
                      nfft * nfft, CUFFT_TRANSFORM, 3);
    if (status != CUFFT_SUCCESS) {
        return 0;
    }
    speckle_fft_plans.emplace(nfft, plan);
    return plan;
}

//--------------------------------------------
extern "C" bool get_ffx_ffb(const COMPLEX *cpu_ffx, COMPLEX *cpu_ffb, size_t count,
                            CUCOMPLEX **gpu_ffx, CUCOMPLEX **gpu_ffb) {
    if (!initialize_cuda_device())
        return false;

    if (!gpu_ffx || !gpu_ffb || !cpu_ffx || !cpu_ffb || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;

    *gpu_ffx = get_or_create_buffer(ffx_buffers, cpu_ffx, count, "get_ffx_ffb FFX buffer");
    *gpu_ffb = get_or_create_buffer(ffb_buffers, cpu_ffb, count, "get_ffx_ffb FFB buffer");
    if (!*gpu_ffx || !*gpu_ffb)
        return false;

    const size_t bytes = count * sizeof(CUCOMPLEX);
    cudaError_t status = cuda_data::copy_host_to_device(*gpu_ffx, cpu_ffx, bytes,
                                                        cuda_utils::TransferCategory::FftWorkspace);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "get_ffx_ffb: FFX H2D copy failed: %s\n", cudaGetErrorString(status));
        return false;
    }

    // The CUDA update kernel subtracts from the existing FFB values.
    status = cuda_data::copy_host_to_device(*gpu_ffb, cpu_ffb, bytes,
                                            cuda_utils::TransferCategory::FftWorkspace);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "get_ffx_ffb: FFB H2D initialization failed: %s\n",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

//--------------------------------------------
extern "C" bool copy_ffb_to_cpu(COMPLEX *cpu_ffb, const CUCOMPLEX *gpu_ffb, size_t count) {
    if (!cpu_ffb || !gpu_ffb || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;

    const cudaError_t status = cuda_data::copy_device_to_host(
        cpu_ffb, gpu_ffb, count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::FftWorkspace);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "copy_ffb_to_cpu: D2H copy failed: %s\n", cudaGetErrorString(status));
        return false;
    }
    return true;
}

//--------------------------------------------
static bool get_gpbicg_device_arrays_impl(COMPLEX *cpu_xi,
                                          const COMPLEX *cpu_xr,
                                          const COMPLEX *cpu_b,
                                          COMPLEX *cpu_work,
                                          size_t vector_count, size_t work_count, bool begin_solve,
                                          bool refresh_host_inputs, int initial_guess_mode,
                                          CUCOMPLEX **gpu_xi, CUCOMPLEX **gpu_xr, CUCOMPLEX **gpu_b,
                                          CUCOMPLEX **gpu_work) {
    if (!initialize_cuda_device() || !cpu_xi || !cpu_xr || !cpu_b || !cpu_work || !gpu_xi ||
        !gpu_xr || !gpu_b || !gpu_work || vector_count == 0 || work_count == 0 ||
        vector_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX) ||
        work_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;

    *gpu_xi = get_or_create_buffer(gpbicg_xi_buffers, cpu_xi, vector_count, "GPBICG xi buffer");
    *gpu_xr = get_or_create_buffer(gpbicg_xr_buffers, cpu_xr, vector_count, "GPBICG xr buffer");
    const auto b_alias = gpbicg_device_b_aliases.find(cpu_b);
    const bool has_b_alias = b_alias != gpbicg_device_b_aliases.end();
    const auto b_source = gpbicg_device_b_sources.find(cpu_b);
    const bool has_b_source = !has_b_alias &&
                              b_source != gpbicg_device_b_sources.end();
    if (has_b_source && (!b_source->second.device ||
                         b_source->second.capacity < vector_count))
        return false;
    if (has_b_alias) {
        if (!b_alias->second.device || b_alias->second.capacity < vector_count)
            return false;
        *gpu_b = b_alias->second.device;
    } else
        *gpu_b = get_or_create_buffer(gpbicg_b_buffers, cpu_b, vector_count, "GPBICG b buffer");
    *gpu_work =
        get_or_create_buffer(gpbicg_work_buffers, cpu_work, work_count, "GPBICG work buffer");
    if (!*gpu_xi || !*gpu_xr || !*gpu_b || !*gpu_work)
        return false;

    const size_t vector_bytes = vector_count * sizeof(CUCOMPLEX);
    const size_t work_bytes = work_count * sizeof(CUCOMPLEX);
    // In resident mode Xi/B/work are seeded at solve start and XR is uploaded
    // only when its host copy is newer. The direct GPU product marks XR current.
    ComplexBufferEntry &xi_entry = gpbicg_xi_buffers[cpu_xi];
    const bool xi_is_device_current =
        begin_solve && !refresh_host_inputs && xi_entry.device_current;
    const cudaError_t status_xi =
        ((begin_solve || refresh_host_inputs) && initial_guess_mode < 0 && !xi_is_device_current)
            ? cuda_data::copy_host_to_device(*gpu_xi, cpu_xi, vector_bytes,
                                             cuda_utils::TransferCategory::SolverVector)
            : cudaSuccess;
    ComplexBufferEntry &xr_entry = gpbicg_xr_buffers[cpu_xr];
    const bool xr_is_device_current = !refresh_host_inputs && xr_entry.device_current;
    const cudaError_t status_xr =
        (refresh_host_inputs || !xr_is_device_current) && !(begin_solve && initial_guess_mode >= 0)
            ? cuda_data::copy_host_to_device(*gpu_xr, cpu_xr, vector_bytes,
                                             cuda_utils::TransferCategory::SolverVector)
            : cudaSuccess;
    const bool rhs_is_device_resident =
        has_b_alias || resident_gpbicg_rhs_buffers.find(cpu_b) != resident_gpbicg_rhs_buffers.end();
    const cudaError_t status_b =
        ((begin_solve || refresh_host_inputs) && has_b_source)
            ? cuda_data::copy_device_to_device(*gpu_b, b_source->second.device,
                                               vector_bytes)
            : ((begin_solve || refresh_host_inputs) && !rhs_is_device_resident)
            ? cuda_data::copy_host_to_device(*gpu_b, cpu_b, vector_bytes,
                                             cuda_utils::TransferCategory::SolverVector)
            : cudaSuccess;
    ComplexBufferEntry &b_entry = gpbicg_b_buffers[cpu_b];
    ComplexBufferEntry &work_entry = gpbicg_work_buffers[cpu_work];
    // The solver work array is scratch storage. Its host contents are never
    // an input to the CUDA solver, so initialize it in place instead of
    // uploading the stale CPU buffer.
    const cudaError_t status_work =
        (begin_solve || !work_entry.initialized_from_host)
            ? cudaMemset(*gpu_work, 0, work_bytes)
            : cudaSuccess;
    if (status_xi != cudaSuccess || status_xr != cudaSuccess || status_b != cudaSuccess ||
        status_work != cudaSuccess) {
        const cudaError_t status = status_xi != cudaSuccess   ? status_xi
                                   : status_xr != cudaSuccess ? status_xr
                                   : status_b != cudaSuccess  ? status_b
                                                              : status_work;
        std::fprintf(stderr, "get_gpbicg_device_arrays: H2D copy failed: %s\n",
                     cudaGetErrorString(status));
        return false;
    }
    cudaError_t initialized_xi = cudaSuccess;
    if (begin_solve && initial_guess_mode == 0) {
        initialized_xi = cudaMemset(*gpu_xi, 0, vector_bytes);
        if (initialized_xi == cudaSuccess)
            initialized_xi = cudaMemset(*gpu_xr, 0, vector_bytes);
    } else if (begin_solve && initial_guess_mode == 1) {
        initialized_xi =
            cuda_utils::tracked_memcpy(*gpu_xi, *gpu_b, vector_bytes, cudaMemcpyDeviceToDevice);
        if (initialized_xi == cudaSuccess)
            initialized_xi =
                cuda_utils::tracked_memcpy(*gpu_xr, *gpu_b, vector_bytes, cudaMemcpyDeviceToDevice);
    }
    if (initialized_xi != cudaSuccess) {
        std::fprintf(stderr, "get_gpbicg_device_arrays: Xi initialization failed: %s\n",
                     cudaGetErrorString(initialized_xi));
        return false;
    }
    xi_entry.initialized_from_host = true;
    if (begin_solve || refresh_host_inputs)
        xi_entry.device_current =
            !refresh_host_inputs && (initial_guess_mode >= 0 || xi_is_device_current);
    if (begin_solve || refresh_host_inputs)
        xr_entry.device_current =
            !refresh_host_inputs && (initial_guess_mode >= 0 || xr_is_device_current);
    b_entry.initialized_from_host = true;
    work_entry.initialized_from_host = true;
    return true;
}

//--------------------------------------------
// Unused legacy host-refresh entry point. Active GPU solvers call the
// _resident / _resident_initialized variants below; keep the old body for
// reference without exporting an unused C ABI symbol.
/*
extern "C" bool get_gpbicg_device_arrays(COMPLEX *cpu_xi, const COMPLEX *cpu_xr,
                                         const COMPLEX *cpu_b, COMPLEX *cpu_work,
                                         size_t vector_count, size_t work_count, CUCOMPLEX **gpu_xi,
                                         CUCOMPLEX **gpu_xr, CUCOMPLEX **gpu_b,
                                         CUCOMPLEX **gpu_work) {
    return get_gpbicg_device_arrays_impl(cpu_xi, cpu_xr, cpu_b, cpu_work, vector_count, work_count,
                                         false, true, -1, gpu_xi, gpu_xr, gpu_b, gpu_work);
}
*/

//--------------------------------------------
extern "C" bool get_gpbicg_device_arrays_resident(COMPLEX *cpu_xi, const COMPLEX *cpu_xr,
                                                  const COMPLEX *cpu_b, COMPLEX *cpu_work,
                                                  size_t vector_count, size_t work_count,
                                                  bool begin_solve, CUCOMPLEX **gpu_xi,
                                                  CUCOMPLEX **gpu_xr, CUCOMPLEX **gpu_b,
                                                  CUCOMPLEX **gpu_work) {
    return get_gpbicg_device_arrays_impl(cpu_xi, cpu_xr, cpu_b, cpu_work, vector_count, work_count,
                                         begin_solve, false, -1, gpu_xi, gpu_xr, gpu_b, gpu_work);
}

//--------------------------------------------
extern "C" bool get_gpbicg_device_arrays_resident_initialized(
    COMPLEX *cpu_xi, const COMPLEX *cpu_xr, const COMPLEX *cpu_b, COMPLEX *cpu_work,
    size_t vector_count, size_t work_count, bool begin_solve, int initial_guess_mode,
    CUCOMPLEX **gpu_xi, CUCOMPLEX **gpu_xr, CUCOMPLEX **gpu_b, CUCOMPLEX **gpu_work) {
    if (initial_guess_mode < 0 || initial_guess_mode > 2)
        return false;
    return get_gpbicg_device_arrays_impl(cpu_xi, cpu_xr, cpu_b, cpu_work, vector_count, work_count,
                                         begin_solve, false, initial_guess_mode, gpu_xi, gpu_xr,
                                         gpu_b, gpu_work);
}

//--------------------------------------------
extern "C" bool register_gpbicg_device_b_alias(const void *cpu_b_key, CUCOMPLEX *gpu_b,
                                               size_t count) {
    if (!initialize_cuda_device() || !cpu_b_key || !gpu_b || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;
    gpbicg_device_b_aliases[cpu_b_key] = {gpu_b, count};
    return true;
}

//--------------------------------------------
extern "C" bool register_gpbicg_device_b_source(const void *cpu_b_key,
                                                 const CUCOMPLEX *gpu_b,
                                                 size_t count) {
    if (!initialize_cuda_device() || !cpu_b_key || !gpu_b || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;
    gpbicg_device_b_sources[cpu_b_key] = {gpu_b, count};
    return true;
}

extern "C" bool has_gpbicg_device_b_source(const void *cpu_b_key) {
    return cpu_b_key && gpbicg_device_b_sources.find(cpu_b_key) !=
                            gpbicg_device_b_sources.end();
}

//--------------------------------------------
extern "C" bool get_gpbicg_device_b_alias(const void *cpu_b_key,
                                           size_t required_count,
                                           CUCOMPLEX **gpu_b) {
    if (!cpu_b_key || !gpu_b || required_count == 0)
        return false;
    const auto alias = gpbicg_device_b_aliases.find(cpu_b_key);
    if (alias == gpbicg_device_b_aliases.end() || !alias->second.device ||
        alias->second.capacity < required_count)
        return false;
    *gpu_b = alias->second.device;
    return true;
}

//--------------------------------------------
extern "C" void release_gpbicg_device_b_alias(const void *cpu_b_key) {
    if (cpu_b_key)
        gpbicg_device_b_aliases.erase(cpu_b_key);
}

//--------------------------------------------
extern "C" CUCOMPLEX *get_gpbicg_device_xi_buffer(const COMPLEX *cpu_xi_key, size_t count) {
    if (!initialize_cuda_device())
        return nullptr;
    return get_or_create_buffer(gpbicg_xi_buffers, cpu_xi_key, count, "GPBICG xi buffer");
}

//--------------------------------------------
extern "C" void mark_gpbicg_device_xi_current(const COMPLEX *cpu_xi_key, bool current) {
    if (!cpu_xi_key)
        return;
    const auto it = gpbicg_xi_buffers.find(cpu_xi_key);
    if (it != gpbicg_xi_buffers.end())
        it->second.device_current = current;
}

//--------------------------------------------
extern "C" void set_gpbicg_device_resident(const COMPLEX *cpu_xi, bool enabled) {
    if (!cpu_xi)
        return;
    if (enabled)
        resident_host_buffers[cpu_xi] = true;
    else
        resident_host_buffers.erase(cpu_xi);
}

//--------------------------------------------
extern "C" void set_gpbicg_device_b_resident(const COMPLEX *cpu_b, bool enabled) {
    if (!cpu_b)
        return;
    if (enabled)
        resident_gpbicg_rhs_buffers[cpu_b] = true;
    else
        resident_gpbicg_rhs_buffers.erase(cpu_b);
}

//--------------------------------------------
extern "C" void mark_gpbicg_device_xr_current(const COMPLEX *cpu_xr, bool current) {
    if (!cpu_xr)
        return;
    const auto it = gpbicg_xr_buffers.find(cpu_xr);
    if (it != gpbicg_xr_buffers.end())
        it->second.device_current = current;
}

//--------------------------------------------
extern "C" bool copy_device_complex_to_host(COMPLEX *cpu_destination, const CUCOMPLEX *gpu_source,
                                            size_t count) {
    if (!cpu_destination || !gpu_source || count == 0 ||
        count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
        return false;
    const cudaError_t status =
        cuda_data::copy_device_to_host(cpu_destination, gpu_source, count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Result);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "copy_device_complex_to_host failed: %s\n",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

//--------------------------------------------
extern "C" bool copy_gpbicg_results_to_cpu(COMPLEX *cpu_xi, COMPLEX *cpu_work,
                                           const CUCOMPLEX *gpu_xi, const CUCOMPLEX *gpu_work,
                                           size_t vector_count, size_t work_count, bool xi_modified,
                                           bool work_modified) {
    if ((xi_modified && (!cpu_xi || !gpu_xi || vector_count == 0 ||
                         vector_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))) ||
        (work_modified && (!cpu_work || !gpu_work || work_count == 0 ||
                           work_count > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))))
        return false;

    const bool resident_solution =
        cpu_xi && resident_host_buffers.find(cpu_xi) != resident_host_buffers.end();
    if (xi_modified && !resident_solution) {
        const cudaError_t status = cuda_data::copy_device_to_host(
            cpu_xi, gpu_xi, vector_count * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "copy_gpbicg_results_to_cpu (xi): D2H copy failed: %s\n",
                         cudaGetErrorString(status));
            return false;
        }
    }
    const bool resident_work =
        cpu_work && resident_host_buffers.find(cpu_work) != resident_host_buffers.end();
    if (work_modified && !resident_work) {
        const cudaError_t status =
            cuda_data::copy_device_to_host(cpu_work, gpu_work, work_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Result);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "copy_gpbicg_results_to_cpu (work): D2H copy failed: %s\n",
                         cudaGetErrorString(status));
            return false;
        }
    }
    return true;
}

//--------------------------------------------
extern "C" bool get_gpbicor_device_arrays(COMPLEX *cpu_xi, const COMPLEX *cpu_xr,
                                          const COMPLEX *cpu_b, COMPLEX *cpu_ffloc,
                                          COMPLEX *cpu_work, size_t vector_count,
                                          size_t work_count, bool initialize_ffloc,
                                          CUCOMPLEX **gpu_xi, CUCOMPLEX **gpu_xr, CUCOMPLEX **gpu_b,
                                          CUCOMPLEX **gpu_ffloc, CUCOMPLEX **gpu_work) {
    if (!initialize_cuda_device() || !cpu_xi || !cpu_xr || !cpu_b || !cpu_ffloc || !cpu_work ||
        !gpu_xi || !gpu_xr || !gpu_b || !gpu_ffloc || !gpu_work || vector_count == 0 ||
        work_count == 0)
        return false;
    *gpu_xi = get_or_create_buffer(gpbicg_xi_buffers, cpu_xi, vector_count, "GPBICOR xi");
    *gpu_xr = get_or_create_buffer(gpbicg_xr_buffers, cpu_xr, vector_count, "GPBICOR xr");
    const auto b_alias = gpbicg_device_b_aliases.find(cpu_b);
    const bool has_b_alias = b_alias != gpbicg_device_b_aliases.end();
    const auto b_source = gpbicg_device_b_sources.find(cpu_b);
    const bool has_b_source = !has_b_alias &&
                              b_source != gpbicg_device_b_sources.end();
    if (has_b_source && (!b_source->second.device ||
                         b_source->second.capacity < vector_count))
        return false;
    if (has_b_alias) {
        if (!b_alias->second.device || b_alias->second.capacity < vector_count)
            return false;
        *gpu_b = b_alias->second.device;
    } else
        *gpu_b = get_or_create_buffer(gpbicg_b_buffers, cpu_b, vector_count, "GPBICOR b");
    *gpu_ffloc =
        get_or_create_buffer(gpbicor_ffloc_buffers, cpu_ffloc, vector_count, "GPBICOR field");
    *gpu_work = get_or_create_buffer(gpbicg_work_buffers, cpu_work, work_count, "GPBICOR work");
    if (!*gpu_xi || !*gpu_xr || !*gpu_b || !*gpu_ffloc || !*gpu_work)
        return false;
    const size_t vb = vector_count * sizeof(CUCOMPLEX), wb = work_count * sizeof(CUCOMPLEX);
    const bool resident = resident_host_buffers.find(cpu_xi) != resident_host_buffers.end();
    auto &xe = gpbicg_xi_buffers[cpu_xi];
    auto &re = gpbicg_xr_buffers[cpu_xr];
    auto &be = gpbicg_b_buffers[cpu_b];
    const bool xi_device_current = xe.device_current;
    if ((!resident && !xi_device_current &&
         cuda_data::copy_host_to_device(
             *gpu_xi, cpu_xi, vb, cuda_utils::TransferCategory::SolverVector) != cudaSuccess) ||
        ((!resident || !re.device_current) &&
         cuda_data::copy_host_to_device(
             *gpu_xr, cpu_xr, vb, cuda_utils::TransferCategory::SolverVector) != cudaSuccess) ||
        (!resident && has_b_source &&
         cuda_data::copy_device_to_device(*gpu_b,b_source->second.device,vb) != cudaSuccess) ||
        (!resident && !has_b_alias && !has_b_source &&
         cuda_data::copy_host_to_device(*gpu_b, cpu_b, vb,
                                        cuda_utils::TransferCategory::SolverVector) != cudaSuccess))
        return false;
    // Preserve a device-current Xi marker established by an earlier GPU
    // preconditioner. The reverse-communication solver consumes this marker
    // when it acquires its buffers; clearing it here would upload the stale
    // host Xi and overwrite the resident preconditioned initial iterate.
    xe.initialized_from_host = true;
    be.initialized_from_host = true;
    auto &fe = gpbicor_ffloc_buffers[cpu_ffloc];
    if (initialize_ffloc || !fe.initialized_from_host) {
        // Chan resident mode keeps the preconditioned initial iterate in gpu_xi;
        // cpu_xi may still contain its pre-preconditioner value.  Seed FFloc
        // from that authoritative device buffer on the first reverse call.
        const cudaError_t status =
            initialize_ffloc
                ? cuda_utils::tracked_memcpy(*gpu_ffloc, *gpu_xi, vb, cudaMemcpyDeviceToDevice)
                : cuda_data::copy_host_to_device(*gpu_ffloc, cpu_ffloc, vb,
                                                 cuda_utils::TransferCategory::IncidentField);
        if (status != cudaSuccess)
            return false;
        fe.initialized_from_host = true;
    }
    auto &we = gpbicg_work_buffers[cpu_work];
    if (!we.initialized_from_host) {
        if (cudaMemset(*gpu_work, 0, wb) != cudaSuccess)
            return false;
        we.initialized_from_host = true;
    }
    return true;
}

//--------------------------------------------
extern "C" bool copy_gpbicor_results_to_cpu(COMPLEX *cpu_xi, COMPLEX *cpu_ffloc,
                                            const CUCOMPLEX *gpu_xi, const CUCOMPLEX *gpu_ffloc,
                                            size_t vector_count, bool copy_ffloc) {
    if (!cpu_xi || !gpu_xi || vector_count == 0)
        return false;
    const bool resident_xi = resident_host_buffers.find(cpu_xi) != resident_host_buffers.end();
    if (!resident_xi &&
        cuda_data::copy_device_to_host(cpu_xi, gpu_xi, vector_count * sizeof(CUCOMPLEX),
                                       cuda_utils::TransferCategory::Result) != cudaSuccess)
        return false;
    if (copy_ffloc) {
        if (!cpu_ffloc || !gpu_ffloc)
            return false;
        const bool resident_ffloc =
            resident_host_buffers.find(cpu_ffloc) != resident_host_buffers.end();
        if (!resident_ffloc &&
            cuda_data::copy_device_to_host(cpu_ffloc, gpu_ffloc, vector_count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::Result) != cudaSuccess)
            return false;
    }
    return true;
}

//--------------------------------------------
extern "C" CUCOMPLEX *get_tfqmr_device_x(const COMPLEX *cpu_x, const COMPLEX *initial_x,
                                         size_t count, bool initialize) {
    if (!initialize_cuda_device() || !cpu_x || !initial_x || count == 0)
        return nullptr;
    CUCOMPLEX *device = get_or_create_buffer(tfqmr_x_buffers, cpu_x, count, "TFQMR X");
    if (!device)
        return nullptr;
    auto &entry = tfqmr_x_buffers[cpu_x];
    if (initialize || !entry.initialized_from_host) {
        if (cuda_data::copy_host_to_device(device, initial_x, count * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::SolverVector) !=
            cudaSuccess)
            return nullptr;
        entry.initialized_from_host = true;
    }
    return device;
}

//--------------------------------------------
extern "C" CUCOMPLEX *get_tfqmr_device_x_from_device(const COMPLEX *cpu_x,
                                                     const CUCOMPLEX *initial_device_x,
                                                     size_t count, bool initialize) {
    if (!initialize_cuda_device() || !cpu_x || !initial_device_x || count == 0)
        return nullptr;
    CUCOMPLEX *device = get_or_create_buffer(tfqmr_x_buffers, cpu_x, count, "TFQMR X device seed");
    if (!device)
        return nullptr;
    auto &entry = tfqmr_x_buffers[cpu_x];
    if (initialize || !entry.initialized_from_host) {
        const cudaError_t status = cuda_utils::tracked_memcpy(
            device, initial_device_x, count * sizeof(CUCOMPLEX), cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "get_tfqmr_device_x_from_device: D2D seed failed: %s\n",
                         cudaGetErrorString(status));
            return nullptr;
        }
        entry.initialized_from_host = true;
    }
    return device;
}

//--------------------------------------------
extern "C" bool copy_tfqmr_x_to_cpu(COMPLEX *cpu_x, const CUCOMPLEX *gpu_x, size_t count) {
    if (!cpu_x || !gpu_x || count == 0)
        return false;
    if (resident_host_buffers.find(cpu_x) != resident_host_buffers.end())
        return true;
    return cuda_data::copy_device_to_host(cpu_x, gpu_x, count * sizeof(CUCOMPLEX),
                                          cuda_utils::TransferCategory::Result) == cudaSuccess;
}
