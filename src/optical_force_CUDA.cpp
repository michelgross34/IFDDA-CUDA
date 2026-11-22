#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <vector>

namespace {
//-----------------------------------------------------------
struct PreparedForcePolarization {
    std::mutex mutex;
    const COMPLEX* host_pointer = nullptr;
    size_t element_count = 0;
    bool ready = false;
};

PreparedForcePolarization& prepared_force_polarization() {
    static PreparedForcePolarization state;
    return state;
}

void clear_prepared_force_polarization() {
    auto& state = prepared_force_polarization();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.host_pointer = nullptr;
    state.element_count = 0;
    state.ready = false;
}
//-------------------------------------------

void mark_prepared_force_polarization(const COMPLEX* host_pointer, size_t element_count) {
    auto& state = prepared_force_polarization();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.host_pointer = host_pointer;
    state.element_count = element_count;
    state.ready = true;
}
//-------------------------------------------

bool consume_prepared_force_polarization(const COMPLEX* host_pointer, size_t element_count) {
    auto& state = prepared_force_polarization();
    std::lock_guard<std::mutex> lock(state.mutex);
    const bool matches = state.ready && state.host_pointer == host_pointer &&
                         state.element_count == element_count;
    state.host_pointer = nullptr;
    state.element_count = 0;
    state.ready = false;
    return matches;
}

//-----------------------------------------------------------
bool reduce_force_components(int count, const CU_REAL *x, const CU_REAL *y, const CU_REAL *z,
                             const CU_REAL *ones, CU_REAL *device_results, CU_REAL *host_results,
                             const char *context) {
    cublasHandle_t handle = cuda_utils::get_cublas_handle();
    if (!handle ||
        !cuda_utils::check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE), context,
                                  "select device result mode"))
        return false;
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const cublasStatus_t status_x = cublasSdot(handle, count, x, 1, ones, 1, device_results);
    const cublasStatus_t status_y = cublasSdot(handle, count, y, 1, ones, 1, device_results + 1);
    const cublasStatus_t status_z = cublasSdot(handle, count, z, 1, ones, 1, device_results + 2);
#else
    const cublasStatus_t status_x = cublasDdot(handle, count, x, 1, ones, 1, device_results);
    const cublasStatus_t status_y = cublasDdot(handle, count, y, 1, ones, 1, device_results + 1);
    const cublasStatus_t status_z = cublasDdot(handle, count, z, 1, ones, 1, device_results + 2);
#endif
    const cublasStatus_t restore_status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    const bool dots_ok = cuda_utils::check_cublas(status_x, context, "reduce x component") &&
                         cuda_utils::check_cublas(status_y, context, "reduce y component") &&
                         cuda_utils::check_cublas(status_z, context, "reduce z component");
    if (!dots_ok || !cuda_utils::check_cublas(restore_status, context, "restore host result mode"))
        return false;
    return cuda_utils::check_cuda(
        cuda_data::copy_device_to_host(host_results, device_results, size_t(3) * sizeof(CU_REAL),
                                       cuda_utils::TransferCategory::Result),
        context, "download reduced components");
}
} // namespace

//-----------------------------------------------------------
extern "C" void ifdda_prepare_optical_force_fft_cuda_(const COMPLEX *polarization,
                                                      const int *dipole_map, const int *nx,
                                                      const int *ny, const int *nz, const int *nx2,
                                                      const int *ny2, const int *nz2,
                                                      const int *use_map, int *status_out) {
    clear_prepared_force_polarization();
    if (status_out)
        *status_out = 1;
    if (!polarization || !nx || !ny || !nz || !nx2 || !ny2 || !nz2 || !use_map || !status_out ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 || *nx2 < *nx || *ny2 < *ny || *nz2 < *nz ||
        (*use_map != 0 && *use_map != 1) || (*use_map && !dipole_map))
        return;
    size_t physical = 0, padded = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx), size_t(*ny), &physical) ||
        !cuda_utils::checked_multiply(physical, size_t(*nz), &physical) ||
        !cuda_utils::checked_multiply(size_t(*nx2), size_t(*ny2), &padded) ||
        !cuda_utils::checked_multiply(padded, size_t(*nz2), &padded) ||
        physical > size_t(INT_MAX) || padded > size_t(INT_MAX))
        return;
    if (!initialize_cuda_device())
        return;
    cuda_data::OpticalForceDeviceBuffers buffers{};
    cuda_data::OpticalForceBufferRequest request{};
    request.dipole_count = physical;
    request.mesh_size = physical;
    request.padded_count = padded;
    request.force_fft = true;
    if (!cuda_data::get_device_optical_force_buffers(request, &buffers))
        return;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.polarization, polarization,
                                           physical * 3 * sizeof(CUCOMPLEX),
                                           cuda_utils::TransferCategory::SolverVector),
            "optical force FFT", "upload polarization"))
        return;
    CU_INT *device_map = nullptr;
    if (*use_map) {
        device_map = get_device_tabdip(dipole_map, physical);
        if (!device_map)
            return;
    }
    if (!cuda_utils::check_cuda(optical_force_gather_fft_wrapper(
                                    buffers.polarization, device_map, buffers.fft.vectors, *nx, *ny,
                                    *nz, *nx2, *ny2, *nz2, int(padded), *use_map),
                                "optical force FFT", "gather polarization"))
        return;
    const cufftHandle plan = get_fft_plan(*nx2, *ny2, *nz2);
    if (!plan)
        return;
    for (int component = 0; component < 3; ++component)
        if (!cuda_utils::check_cufft(
                CUFFT_EXECUTE(plan, buffers.fft.vectors + size_t(component) * padded,
                              buffers.fft.vectors + size_t(component) * padded, CUFFT_INVERSE),
                "optical force FFT", "transform polarization"))
            return;
    mark_prepared_force_polarization(polarization, physical * 3);
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_compute_force_green_derivative_cuda_(
    const REAL *spacing, const REAL *k0, const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nz2, const int *nxy2, const int *axis,
    COMPLEX *output, int *status_out) {
    if (status_out)
        *status_out = 1;
    if (!spacing || !k0 || !nx || !ny || !nz || !nx2 || !ny2 || !nz2 || !nxy2 || !axis || !output ||
        !status_out || *nx <= 0 || *ny <= 0 || *nz <= 0 || *nx2 < *nx || *ny2 < *ny || *nz2 < *nz ||
        *nxy2 != *nx2 * *ny2 || *axis < 1 || *axis > 3 || *spacing <= CU_REAL(0))
        return;
    size_t physical = 0, padded = 0;
    if (!cuda_utils::checked_multiply(size_t(*nx), size_t(*ny), &physical) ||
        !cuda_utils::checked_multiply(physical, size_t(*nz), &physical) ||
        !cuda_utils::checked_multiply(size_t(*nxy2), size_t(*nz2), &padded) ||
        physical > size_t(INT_MAX) || padded > size_t(INT_MAX))
        return;
    if (!initialize_cuda_device())
        return;
    cuda_data::OpticalForceDeviceBuffers buffers{};
    cuda_data::OpticalForceBufferRequest request{};
    request.dipole_count = physical;
    request.mesh_size = physical;
    request.padded_count = padded;
    request.force_fft = true;
    if (!cuda_data::get_device_optical_force_buffers(request, &buffers))
        return;
    if (!cuda_utils::check_cuda(optical_force_build_green_derivative_wrapper(
                                    buffers.fft.green_derivatives, *nx, *ny, *nz, *nx2, *ny2, *nz2,
                                    *nxy2, int(padded), *axis, *spacing, *k0),
                                "optical force Green derivative", "build tensor"))
        return;
    const cufftHandle plan = get_fft_plan(*nx2, *ny2, *nz2);
    if (!plan)
        return;
    for (int component = 0; component < 6; ++component)
        if (!cuda_utils::check_cufft(
                CUFFT_EXECUTE(plan, buffers.fft.green_derivatives + size_t(component) * padded,
                              buffers.fft.green_derivatives + size_t(component) * padded,
                              CUFFT_INVERSE),
                "optical force Green derivative", "transform tensor"))
            return;
    if (!cuda_utils::check_cuda(
            optical_force_derivative_spectral_product_wrapper(
                buffers.fft.vectors, buffers.fft.green_derivatives, int(padded)),
            "optical force Green derivative", "multiply tensor by field"))
        return;
    for (int component : {0, 3, 5})
        if (!cuda_utils::check_cufft(
                CUFFT_EXECUTE(plan, buffers.fft.green_derivatives + size_t(component) * padded,
                              buffers.fft.green_derivatives + size_t(component) * padded,
                              CUFFT_FORWARD),
                "optical force Green derivative", "inverse transform"))
            return;
    if (!cuda_utils::check_cuda(optical_force_extract_green_derivative_wrapper(
                                    buffers.fft.green_derivatives, buffers.fft.output, *nx, *ny,
                                    *nz, *nx2, *nxy2, int(padded), int(physical)),
                                "optical force Green derivative", "extract physical field") ||
        !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                    output, buffers.fft.output, physical * 3 * sizeof(CUCOMPLEX),
                                    cuda_utils::TransferCategory::Result),
                                "optical force Green derivative", "download derivative field"))
        return;
    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_compute_optical_force_cuda_(const COMPLEX *polarization,
                                                  const COMPLEX *field_derivatives,
                                                  const int *dipole_count, const int *mesh_size,
                                                  const REAL *scale, REAL *force_x,
                                                  REAL *force_y, REAL *force_z,
                                                  REAL *force_total, int *status_out) {
    if (status_out)
        *status_out = 1;
    if (!polarization || !field_derivatives || !dipole_count || !mesh_size || !scale || !force_x ||
        !force_y || !force_z || !force_total || !status_out || *dipole_count <= 0 ||
        *mesh_size < *dipole_count || size_t(*mesh_size) > std::numeric_limits<size_t>::max() / 9 ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / 3)
        return;

    if (!initialize_cuda_device())
        return;
    const size_t vectors = size_t(*dipole_count) * 3;
    const size_t derivative_count = size_t(*mesh_size) * 9;
    cuda_data::OpticalForceDeviceBuffers buffers{};
    cuda_data::OpticalForceBufferRequest request{};
    request.dipole_count = size_t(*dipole_count);
    request.mesh_size = size_t(*mesh_size);
    request.force_density = true;
    request.reduction = true;
    if (!cuda_data::get_device_optical_force_buffers(request, &buffers))
        return;

    if ((!consume_prepared_force_polarization(polarization, vectors) &&
         !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
                                     buffers.polarization, polarization,
                                     vectors * sizeof(COMPLEX),
                                     cuda_utils::TransferCategory::Other),
                                 "optical force", "upload dipole polarization")) ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(buffers.field_derivatives,
                                                               field_derivatives,
                                                               derivative_count * sizeof(COMPLEX),
                                                               cuda_utils::TransferCategory::Other),
                                "optical force", "upload local-field derivatives"))
        return;

    if (!cuda_utils::check_cuda(
            optical_force_density_wrapper(buffers.polarization, buffers.field_derivatives,
                                          buffers.force_x, buffers.force_y, buffers.force_z,
                                          *dipole_count, *mesh_size, *scale),
            "optical force", "calculate force density") ||
        !cuda_utils::check_cuda(
            optical_force_fill_reduction_ones_wrapper(buffers.reduction_ones, *dipole_count),
            "optical force", "prepare force reduction"))
        return;

    if (!reduce_force_components(*dipole_count, buffers.force_x, buffers.force_y, buffers.force_z,
                                 buffers.reduction_ones, buffers.reduction_results, force_total,
                                 "optical force"))
        return;

    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(force_x, buffers.force_x,
                                           size_t(*dipole_count) * sizeof(REAL),
                                           cuda_utils::TransferCategory::Result),
            "optical force", "download x force density") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(force_y, buffers.force_y,
                                           size_t(*dipole_count) * sizeof(REAL),
                                           cuda_utils::TransferCategory::Result),
            "optical force", "download y force density") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(force_z, buffers.force_z,
                                           size_t(*dipole_count) * sizeof(REAL),
                                           cuda_utils::TransferCategory::Result),
            "optical force", "download z force density"))
        return;

    *status_out = 0;
}

//-----------------------------------------------------------
extern "C" void ifdda_compute_optical_torque_cuda_(const COMPLEX *local_field, const REAL *xs,
                                                   const REAL *ys, const REAL *zs,
                                                   const int *dipole_count, const CU_REAL *k03,
                                                   const REAL *scale, REAL *center,
                                                   REAL *torque_x, REAL *torque_y,
                                                   REAL *torque_z, REAL *torque_total,
                                                   int *status_out) {
    if (status_out)
        *status_out = 1;
    if (!local_field || !xs || !ys || !zs || !dipole_count || !k03 || !scale || !center ||
        !torque_x || !torque_y || !torque_z || !torque_total || !status_out || *dipole_count <= 0 ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / 3)
        return;

    if (!initialize_cuda_device())
        return;
    const size_t dipoles = size_t(*dipole_count);
    const size_t vectors = dipoles * 3;
    cuda_data::OpticalForceDeviceBuffers buffers{};
    cuda_data::OpticalForceBufferRequest request{};
    request.dipole_count = dipoles;
    request.mesh_size = dipoles;
    request.force_density = true;
    request.torque_density = true;
    request.reduction = true;
    if (!cuda_data::get_device_optical_force_buffers(request, &buffers))
        return;

    const CUCOMPLEX *device_local_field =
        get_device_result_vector_alias(local_field, vectors);
    if (!device_local_field) {
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(
                    buffers.local_field, local_field, vectors * sizeof(COMPLEX),
                    cuda_utils::TransferCategory::Other),
                "optical torque", "upload local field"))
            return;
        device_local_field = buffers.local_field;
    }

    const CU_REAL *device_xs = nullptr;
    const CU_REAL *device_ys = nullptr;
    const CU_REAL *device_zs = nullptr;
    const bool coordinates_are_resident =
        cuda_data::get_device_geometry_coordinate_aliases(
            xs, ys, zs, dipoles, &device_xs, &device_ys, &device_zs);
    if (!coordinates_are_resident &&
        (!cuda_utils::check_cuda(
             cuda_data::copy_host_to_device(
                 buffers.xs, xs, dipoles * sizeof(CU_REAL),
                 cuda_utils::TransferCategory::Geometry),
             "optical torque", "upload x coordinates") ||
         !cuda_utils::check_cuda(
             cuda_data::copy_host_to_device(
                 buffers.ys, ys, dipoles * sizeof(CU_REAL),
                 cuda_utils::TransferCategory::Geometry),
             "optical torque", "upload y coordinates") ||
         !cuda_utils::check_cuda(
             cuda_data::copy_host_to_device(
                 buffers.zs, zs, dipoles * sizeof(CU_REAL),
                 cuda_utils::TransferCategory::Geometry),
             "optical torque", "upload z coordinates")))
        return;
    if (!coordinates_are_resident) {
        device_xs = buffers.xs;
        device_ys = buffers.ys;
        device_zs = buffers.zs;
    }

    if (!cuda_utils::check_cuda(
            optical_force_fill_reduction_ones_wrapper(buffers.reduction_ones, *dipole_count),
            "optical torque", "prepare coordinate reduction") ||
        !reduce_force_components(*dipole_count, device_xs, device_ys, device_zs,
                                 buffers.reduction_ones, buffers.reduction_results, center,
                                 "optical torque center"))
        return;
    const CU_REAL inverse_count = CU_REAL(1) / CU_REAL(*dipole_count);
    center[0] *= inverse_count;
    center[1] *= inverse_count;
    center[2] *= inverse_count;

    if (!cuda_utils::check_cuda(
            optical_torque_density_wrapper(buffers.polarization, device_local_field, device_xs,
                                           device_ys, device_zs, buffers.force_x, buffers.force_y,
                                           buffers.force_z, buffers.torque_x, buffers.torque_y,
                                           buffers.torque_z, *dipole_count, *k03, center[0],
                                           center[1], center[2], *scale),
            "optical torque", "calculate torque density"))
        return;

    if (!reduce_force_components(*dipole_count, buffers.torque_x, buffers.torque_y,
                                 buffers.torque_z, buffers.reduction_ones,
                                 buffers.reduction_results, torque_total, "optical torque"))
        return;

    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(torque_x, buffers.torque_x, dipoles * sizeof(CU_REAL),
                                           cuda_utils::TransferCategory::Result),
            "optical torque", "download x torque density") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(torque_y, buffers.torque_y, dipoles * sizeof(CU_REAL),
                                           cuda_utils::TransferCategory::Result),
            "optical torque", "download y torque density") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(torque_z, buffers.torque_z, dipoles * sizeof(CU_REAL),
                                           cuda_utils::TransferCategory::Result),
            "optical torque", "download z torque density"))
        return;

    *status_out = 0;
}

namespace {
//-----------------------------------------------------------
bool sum_optical_density_by_object(const int *object_ids, const int *dipole_count,
                                   const int *object_count, CU_REAL *sum_x, CU_REAL *sum_y,
                                   CU_REAL *sum_z, int include_torque, int *status_out) {
    if (status_out)
        *status_out = 1;
    if (!object_ids || !dipole_count || !object_count || !sum_x || !sum_y || !sum_z ||
        !status_out || *dipole_count <= 0 || *object_count <= 1 ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / sizeof(int) ||
        size_t(*object_count) > std::numeric_limits<size_t>::max() / (3 * sizeof(CU_REAL)))
        return false;
    if (!initialize_cuda_device())
        return false;

    cuda_data::OpticalForceDeviceBuffers buffers{};
    cuda_data::OpticalForceBufferRequest request{};
    request.dipole_count = size_t(*dipole_count);
    request.mesh_size = size_t(*dipole_count);
    request.object_count = size_t(*object_count);
    request.group_force = include_torque == 0;
    request.group_torque = include_torque != 0;
    if (!cuda_data::get_device_optical_force_buffers(request, &buffers))
        return false;
    CU_REAL *device_sums =
        include_torque ? buffers.multi_object.torque_sums : buffers.multi_object.force_sums;
    const CU_REAL *x = include_torque ? buffers.torque_x : buffers.force_x;
    const CU_REAL *y = include_torque ? buffers.torque_y : buffers.force_y;
    const CU_REAL *z = include_torque ? buffers.torque_z : buffers.force_z;
    if (!device_sums || !x || !y || !z ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.multi_object.object_ids, object_ids,
                                           size_t(*dipole_count) * sizeof(int),
                                           cuda_utils::TransferCategory::Geometry),
            "optical density grouping", "upload object ids"))
        return false;

    const cudaError_t launch_status =
        include_torque ? optical_torque_group_sum_wrapper(buffers.multi_object.object_ids, x, y, z,
                                                          device_sums, *dipole_count, *object_count)
                       : optical_force_group_sum_wrapper(buffers.multi_object.object_ids, x, y, z,
                                                         device_sums, *dipole_count, *object_count);
    if (!cuda_utils::check_cuda(launch_status, "optical density grouping",
                                "sum dipole densities by object"))
        return false;

    std::vector<CU_REAL> host_sums(size_t(*object_count) * 3);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(host_sums.data(), device_sums,
                                           host_sums.size() * sizeof(CU_REAL),
                                           cuda_utils::TransferCategory::Result),
            "optical density grouping", "download object sums"))
        return false;
    for (int object = 0; object < *object_count; ++object) {
        sum_x[object] = host_sums[size_t(object) * 3];
        sum_y[object] = host_sums[size_t(object) * 3 + 1];
        sum_z[object] = host_sums[size_t(object) * 3 + 2];
    }
    *status_out = 0;
    return true;
}
} // namespace

//-----------------------------------------------------------
extern "C" void ifdda_sum_optical_force_by_object_cuda_(const int *object_ids,
                                                        const int *dipole_count,
                                                        const int *object_count, CU_REAL *sum_x,
                                                        CU_REAL *sum_y, CU_REAL *sum_z,
                                                        int *status_out) {
    sum_optical_density_by_object(object_ids, dipole_count, object_count, sum_x, sum_y, sum_z, 0,
                                  status_out);
}

//-----------------------------------------------------------
extern "C" void ifdda_sum_optical_torque_by_object_cuda_(const int *object_ids,
                                                         const int *dipole_count,
                                                         const int *object_count, CU_REAL *sum_x,
                                                         CU_REAL *sum_y, CU_REAL *sum_z,
                                                         int *status_out) {
    sum_optical_density_by_object(object_ids, dipole_count, object_count, sum_x, sum_y, sum_z, 1,
                                  status_out);
}
