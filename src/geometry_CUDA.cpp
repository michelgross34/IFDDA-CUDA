#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

//-----------------------------------------------------------
extern "C" int ifdda_cube_geometry_cuda(int subdivisions, CU_REAL side, CU_REAL cell_size,
                                        CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
                                        CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, int *dipole_map) {
    if (subdivisions <= 0 || subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions || !xs || !ys || !zs || !dipole_map)
        return 1;

    const size_t count = size_t(subdivisions) * size_t(subdivisions) * size_t(subdivisions);
    cuda_data::CubeGeometryDeviceBuffers buffers{};
    if (!cuda_data::get_device_cube_geometry_buffers(count, &buffers))
        return 2;
    if (!cuda_utils::check_cuda(cube_geometry_wrapper(buffers.xs, buffers.ys, buffers.zs,
                                                      buffers.dipole_map, subdivisions, side,
                                                      cell_size, center_x, center_y, center_z),
                                "cube geometry", "launch kernel"))
        return 3;
    if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "cube geometry",
                                "complete coordinate generation"))
        return 4;

    const size_t real_bytes = count * sizeof(CU_REAL);
    const size_t map_bytes = count * sizeof(CU_INT);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(xs, buffers.xs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cube geometry", "download x coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(ys, buffers.ys, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cube geometry", "download y coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(zs, buffers.zs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cube geometry", "download z coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(dipole_map, buffers.dipole_map, map_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cube geometry", "download dipole map"))
        return 5;
    return cuda_data::register_device_geometry_coordinate_aliases(
               xs, ys, zs, count, buffers.xs, buffers.ys, buffers.zs)
               ? 0
               : 6;
}

//-----------------------------------------------------------
namespace {

bool find_dipole_inside(CU_REAL *xdip, CU_REAL *ydip, CU_REAL *zdip,
                        const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
                        const CU_REAL *cell_size, const int *nmax,
                        const int *dipole_count, int *match_index) {
    if (!xdip || !ydip || !zdip || !xs || !ys || !zs || !cell_size ||
        !nmax || !dipole_count || !match_index || *nmax <= 0 ||
        *dipole_count < 0 || *dipole_count > *nmax || *cell_size == CU_REAL(0))
        return false;
    *match_index = INT_MAX;
    if (*dipole_count == 0)
        return true;

    const size_t count = size_t(*dipole_count);
    // The shared geometry stage has already compacted these Fortran host
    // coordinates. Searching them once on CPU avoids allocating/uploading
    // three GPU vectors solely for the antenna's source-point lookup.
    // Use the exact round-to-nearest-away-from-zero predicate of the kernel
    // and return its first (smallest) matching compact index.
    cuda_data::HostRegularGeometry host_grid{};
    if (cuda_data::get_host_regular_geometry(
            xs, ys, zs, *dipole_count, &host_grid)) {
        const CU_REAL scale = CU_REAL(0.9999) / *cell_size;
        for (size_t index = 0; index < count; ++index) {
            if (std::round((*xdip - xs[index]) * scale) == CU_REAL(0) &&
                std::round((*ydip - ys[index]) * scale) == CU_REAL(0) &&
                std::round((*zdip - zs[index]) * scale) == CU_REAL(0)) {
                *match_index = int(index);
                break;
            }
        }
        return true;
    }
    const CU_REAL *device_xs = nullptr;
    const CU_REAL *device_ys = nullptr;
    const CU_REAL *device_zs = nullptr;
    CU_REAL *upload_xs = nullptr;
    CU_REAL *upload_ys = nullptr;
    CU_REAL *upload_zs = nullptr;
    const bool resident_coordinates = cuda_data::get_device_geometry_coordinate_aliases(
        xs, ys, zs, count, &device_xs, &device_ys, &device_zs);
    cuda_data::DipoleInsideDeviceBuffers buffers{};
    if (!cuda_data::get_device_dipole_inside_buffers(
            resident_coordinates ? 0 : count, &buffers))
        return false;
    if (!resident_coordinates) {
        upload_xs = buffers.xs;
        upload_ys = buffers.ys;
        upload_zs = buffers.zs;
        device_xs = upload_xs;
        device_ys = upload_ys;
        device_zs = upload_zs;
    }
    cudaError_t status = cudaSuccess;
    if (!resident_coordinates)
        status = cuda_data::copy_host_to_device(
            upload_xs, xs, count * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess && !resident_coordinates)
        status = cuda_data::copy_host_to_device(
            upload_ys, ys, count * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess && !resident_coordinates)
        status = cuda_data::copy_host_to_device(
            upload_zs, zs, count * sizeof(CU_REAL), cuda_utils::TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cudaMemset(buffers.first_match, 0x7f, sizeof(*buffers.first_match));
    if (status == cudaSuccess)
        status = dipole_inside_find_wrapper(device_xs, device_ys, device_zs,
                                            *dipole_count, *xdip, *ydip, *zdip,
                                            *cell_size, buffers.first_match);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            match_index, buffers.first_match, sizeof(*match_index),
            cuda_utils::TransferCategory::Control);
    return cuda_utils::check_cuda(status, "dipole inside geometry", "search dipole coordinates");
}

} // namespace

//-----------------------------------------------------------
extern "C" void dipoleinside_(CU_REAL *xdip, CU_REAL *ydip, CU_REAL *zdip,
                                const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
                                const CU_REAL *cell_size, const int *nmax,
                                const int *dipole_count) {
    int match_index = INT_MAX;
    if (!find_dipole_inside(xdip, ydip, zdip, xs, ys, zs, cell_size,
                            nmax, dipole_count, &match_index))
        return;
    if (match_index >= 0 && match_index < *dipole_count) {
        *xdip = xs[match_index];
        *ydip = ys[match_index];
        *zdip = zs[match_index];
        std::fprintf(stderr,
                     "antenna inside the near field domain %.9g %.9g %.9g\n",
                     double(*xdip), double(*ydip), double(*zdip));
    } else {
        std::fprintf(stderr, "antenna outside the near field domain\n");
    }
}

//-----------------------------------------------------------
extern "C" void dipoleinsidep_(CU_REAL *xdip, CU_REAL *ydip, CU_REAL *zdip,
                                 const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
                                 const CU_REAL *cell_size, const int *nmax,
                                 const int *dipole_count, int *test) {
    if (!test)
        return;
    *test = 1;
    int match_index = INT_MAX;
    if (!find_dipole_inside(xdip, ydip, zdip, xs, ys, zs, cell_size,
                            nmax, dipole_count, &match_index))
        return;
    if (match_index != INT_MAX) {
        *xdip = xs[match_index];
        *ydip = ys[match_index];
        *zdip = zs[match_index];
        *test = 0;
    }
}

//-----------------------------------------------------------
extern "C" int ifdda_cuboid_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size, CU_REAL side_x,
                                          CU_REAL side_y, CU_REAL side_z, CU_REAL center_x,
                                          CU_REAL center_y, CU_REAL center_z, CU_REAL *xs,
                                          CU_REAL *ys, CU_REAL *zs, int *dipole_map) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny || nx * ny > INT_MAX / nz || !xs ||
        !ys || !zs || !dipole_map)
        return 1;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    cuda_data::CubeGeometryDeviceBuffers buffers{};
    if (!cuda_data::get_device_cube_geometry_buffers(count, &buffers))
        return 2;
    if (!cuda_utils::check_cuda(cuboid_geometry_wrapper(buffers.xs, buffers.ys, buffers.zs,
                                                        buffers.dipole_map, nx, ny, nz, cell_size,
                                                        side_x, side_y, side_z, center_x, center_y,
                                                        center_z),
                                "cuboid geometry", "launch kernel"))
        return 3;
    if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "cuboid geometry",
                                "complete coordinate generation"))
        return 4;
    const size_t real_bytes = count * sizeof(CU_REAL);
    const size_t map_bytes = count * sizeof(CU_INT);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(xs, buffers.xs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cuboid geometry", "download x coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(ys, buffers.ys, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cuboid geometry", "download y coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(zs, buffers.zs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cuboid geometry", "download z coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(dipole_map, buffers.dipole_map, map_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cuboid geometry", "download dipole map"))
        return 5;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_rotated_cuboid_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size,
                                                  CU_REAL extent_x, CU_REAL extent_y,
                                                  CU_REAL extent_z, CU_REAL side_x, CU_REAL side_y,
                                                  CU_REAL side_z, CU_REAL center_x,
                                                  CU_REAL center_y, CU_REAL center_z,
                                                  const CU_REAL *inverse_rotation, CU_REAL *xs,
                                                  CU_REAL *ys, CU_REAL *zs, int *inside_map) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny || nx * ny > INT_MAX / nz ||
        !inverse_rotation || !xs || !ys || !zs || !inside_map)
        return 1;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    cuda_data::CubeGeometryDeviceBuffers buffers{};
    if (!cuda_data::get_device_cube_geometry_buffers(count, &buffers))
        return 2;
    if (!cuda_utils::check_cuda(rotated_cuboid_geometry_wrapper(
                                    buffers.xs, buffers.ys, buffers.zs, buffers.dipole_map, nx, ny,
                                    nz, cell_size, extent_x, extent_y, extent_z, side_x, side_y,
                                    side_z, center_x, center_y, center_z, inverse_rotation[0],
                                    inverse_rotation[1], inverse_rotation[2], inverse_rotation[3],
                                    inverse_rotation[4], inverse_rotation[5], inverse_rotation[6],
                                    inverse_rotation[7], inverse_rotation[8]),
                                "rotated cuboid geometry", "launch kernel"))
        return 3;
    if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "rotated cuboid geometry",
                                "complete coordinate generation"))
        return 4;
    const size_t real_bytes = count * sizeof(CU_REAL);
    const size_t map_bytes = count * sizeof(int);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(xs, buffers.xs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "rotated cuboid geometry", "download x coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(ys, buffers.ys, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "rotated cuboid geometry", "download y coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(zs, buffers.zs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "rotated cuboid geometry", "download z coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(inside_map, buffers.dipole_map, map_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "rotated cuboid geometry", "download inside map"))
        return 5;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_sphere_geometry_cuda(int subdivisions, CU_REAL radius, CU_REAL cell_size,
                                          CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
                                          CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, int *inside_map) {
    if (subdivisions <= 0 || subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions || !xs || !ys || !zs || !inside_map)
        return 1;
    const size_t count = size_t(subdivisions) * size_t(subdivisions) * size_t(subdivisions);
    cuda_data::CubeGeometryDeviceBuffers buffers{};
    if (!cuda_data::get_device_cube_geometry_buffers(count, &buffers))
        return 2;
    if (!cuda_utils::check_cuda(sphere_geometry_wrapper(buffers.xs, buffers.ys, buffers.zs,
                                                        buffers.dipole_map, subdivisions, radius,
                                                        cell_size, center_x, center_y, center_z),
                                "sphere geometry", "launch kernel"))
        return 3;
    if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "sphere geometry",
                                "complete coordinate generation"))
        return 4;
    const size_t real_bytes = count * sizeof(CU_REAL);
    const size_t map_bytes = count * sizeof(int);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(xs, buffers.xs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "sphere geometry", "download x coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(ys, buffers.ys, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "sphere geometry", "download y coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(zs, buffers.zs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "sphere geometry", "download z coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(inside_map, buffers.dipole_map, map_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "sphere geometry", "download inside map"))
        return 5;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_sphere_geometry_compact_cuda(
    int subdivisions, CU_REAL radius, CU_REAL cell_size,
    CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
    CU_REAL* mesh_x, CU_REAL* mesh_y, CU_REAL* mesh_z, int* inside_map,
    CU_REAL* packed_x, CU_REAL* packed_y, CU_REAL* packed_z,
    int* packed_count) {
    if (!mesh_x || !mesh_y || !mesh_z || !inside_map ||
        !packed_x || !packed_y || !packed_z || !packed_count ||
        subdivisions <= 0 || subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions)
        return 1;
    const size_t count = size_t(subdivisions) * size_t(subdivisions) * size_t(subdivisions);
    cuda_data::CubeGeometryDeviceBuffers geometry{};
    cuda_data::SphereGeometryCompactDeviceBuffers compact{};
    if (!cuda_data::get_device_cube_geometry_buffers(count, &geometry))
        return 2;
    if (!cuda_utils::check_cuda(sphere_geometry_wrapper(
                                    geometry.xs, geometry.ys, geometry.zs,
                                    geometry.dipole_map, subdivisions, radius,
                                    cell_size, center_x, center_y, center_z),
                                "sphere geometry", "generate compact source grid") ||
        !cuda_utils::check_cuda(cudaDeviceSynchronize(), "sphere geometry",
                                "complete compact source grid"))
        return 3;
    size_t scan_temporary_bytes = 0;
    if (!cuda_utils::check_cuda(sphere_geometry_compaction_workspace_wrapper(
                                    int(count), &scan_temporary_bytes),
                                "sphere geometry", "query stable compaction workspace") ||
        !cuda_data::get_device_sphere_geometry_compact_buffers(
            count, scan_temporary_bytes, &compact))
        return 4;
    if (!cuda_utils::check_cuda(sphere_geometry_compact_wrapper(
                                    geometry.xs, geometry.ys, geometry.zs,
                                    geometry.dipole_map, int(count), compact.prefix,
                                    compact.xs, compact.ys, compact.zs,
                                    compact.packed_count, compact.scan_temporary,
                                    compact.scan_temporary_bytes),
                                "sphere geometry", "stable GPU coordinate compaction") ||
        !cuda_utils::check_cuda(cudaDeviceSynchronize(), "sphere geometry",
                                "complete stable coordinate compaction"))
        return 5;
    if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                   inside_map, geometry.dipole_map,
                                   count * sizeof(*inside_map),
                                   cuda_utils::TransferCategory::Geometry),
                                "sphere geometry", "download occupancy map"))
        return 6;
    if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                   packed_count, compact.packed_count, sizeof(*packed_count),
                                   cuda_utils::TransferCategory::Control),
                                "sphere geometry", "download packed dipole count") ||
        *packed_count < 0 || size_t(*packed_count) > count)
        return 7;
    if (*packed_count > 0 &&
        (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                     packed_x, compact.xs, size_t(*packed_count) * sizeof(CU_REAL),
                                     cuda_utils::TransferCategory::Geometry),
                                 "sphere geometry", "download compact x coordinates") ||
         !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                     packed_y, compact.ys, size_t(*packed_count) * sizeof(CU_REAL),
                                     cuda_utils::TransferCategory::Geometry),
                                 "sphere geometry", "download compact y coordinates") ||
         !cuda_utils::check_cuda(cuda_data::copy_device_to_host(
                                     packed_z, compact.zs, size_t(*packed_count) * sizeof(CU_REAL),
                                     cuda_utils::TransferCategory::Geometry),
                                 "sphere geometry", "download compact z coordinates")))
        return 8;
    return cuda_data::register_device_geometry_coordinate_aliases(
               packed_x, packed_y, packed_z, size_t(*packed_count),
               compact.xs, compact.ys, compact.zs)
               ? 0
               : 9;
}

//-----------------------------------------------------------
extern "C" int ifdda_ellipsoid_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size,
                                             CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
                                             CU_REAL radius_a, CU_REAL radius_b, CU_REAL radius_c,
                                             CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3,
                                             CU_REAL m4, CU_REAL m5, CU_REAL m6, CU_REAL m7,
                                             CU_REAL m8, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
                                             int *inside_map) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny || nx * ny > INT_MAX / nz || !xs ||
        !ys || !zs || !inside_map)
        return 1;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    cuda_data::CubeGeometryDeviceBuffers buffers{};
    if (!cuda_data::get_device_cube_geometry_buffers(count, &buffers))
        return 2;
    if (!cuda_utils::check_cuda(
            ellipsoid_geometry_wrapper(buffers.xs, buffers.ys, buffers.zs, buffers.dipole_map, nx,
                                       ny, nz, cell_size, extent_x, extent_y, extent_z, radius_a,
                                       radius_b, radius_c, m0, m1, m2, m3, m4, m5, m6, m7, m8),
            "ellipsoid geometry", "launch kernel"))
        return 3;
    if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "ellipsoid geometry",
                                "complete coordinate generation"))
        return 4;
    const size_t real_bytes = count * sizeof(CU_REAL);
    const size_t map_bytes = count * sizeof(int);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(xs, buffers.xs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "ellipsoid geometry", "download x coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(ys, buffers.ys, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "ellipsoid geometry", "download y coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(zs, buffers.zs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "ellipsoid geometry", "download z coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(inside_map, buffers.dipole_map, map_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "ellipsoid geometry", "download inside map"))
        return 5;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_cylinder_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size,
                                            CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
                                            CU_REAL radius, CU_REAL height, CU_REAL m0, CU_REAL m1,
                                            CU_REAL m2, CU_REAL m3, CU_REAL m4, CU_REAL m5,
                                            CU_REAL m6, CU_REAL m7, CU_REAL m8, CU_REAL *xs,
                                            CU_REAL *ys, CU_REAL *zs, int *inside_map) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny || nx * ny > INT_MAX / nz || !xs ||
        !ys || !zs || !inside_map)
        return 1;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    cuda_data::CubeGeometryDeviceBuffers buffers{};
    if (!cuda_data::get_device_cube_geometry_buffers(count, &buffers))
        return 2;
    if (!cuda_utils::check_cuda(
            cylinder_geometry_wrapper(buffers.xs, buffers.ys, buffers.zs, buffers.dipole_map, nx,
                                      ny, nz, cell_size, extent_x, extent_y, extent_z, radius,
                                      height, m0, m1, m2, m3, m4, m5, m6, m7, m8),
            "cylinder geometry", "launch kernel"))
        return 3;
    if (!cuda_utils::check_cuda(cudaDeviceSynchronize(), "cylinder geometry",
                                "complete coordinate generation"))
        return 4;
    const size_t real_bytes = count * sizeof(CU_REAL);
    const size_t map_bytes = count * sizeof(int);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(xs, buffers.xs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cylinder geometry", "download x coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(ys, buffers.ys, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cylinder geometry", "download y coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(zs, buffers.zs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cylinder geometry", "download z coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(inside_map, buffers.dipole_map, map_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "cylinder geometry", "download inside map"))
        return 5;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_nspheres_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size,
                                            CU_REAL x_min, CU_REAL y_min, CU_REAL z_min,
                                            int sphere_count, int fixed_sphere_count,
                                            const CU_REAL *sphere_parameters, CU_REAL *xs,
                                            CU_REAL *ys, CU_REAL *zs, int *sphere_map) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || nx > INT_MAX / ny || nx * ny > INT_MAX / nz ||
        sphere_count <= 0 || fixed_sphere_count < 0 || fixed_sphere_count > sphere_count ||
        size_t(sphere_count) > std::numeric_limits<size_t>::max() / 4 || !sphere_parameters ||
        !xs || !ys || !zs || !sphere_map)
        return 1;
    const size_t count = size_t(nx) * size_t(ny) * size_t(nz);
    const size_t parameter_count = size_t(sphere_count) * 4;
    cuda_data::CubeGeometryDeviceBuffers buffers{};
    if (!cuda_data::get_device_cube_geometry_buffers(count, &buffers, parameter_count))
        return 2;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.shape_parameters, sphere_parameters,
                                           parameter_count * sizeof(CU_REAL),
                                           cuda_utils::TransferCategory::Geometry),
            "multiple-sphere geometry", "upload sphere parameters"))
        return 3;
    if (!cuda_utils::check_cuda(multi_sphere_geometry_wrapper(
                                    buffers.xs, buffers.ys, buffers.zs, buffers.dipole_map, nx, ny,
                                    nz, cell_size, x_min, y_min, z_min, buffers.shape_parameters,
                                    sphere_count, fixed_sphere_count),
                                "multiple-sphere geometry", "launch kernel") ||
        !cuda_utils::check_cuda(cudaDeviceSynchronize(), "multiple-sphere geometry",
                                "complete coordinate generation"))
        return 4;
    const size_t real_bytes = count * sizeof(CU_REAL);
    const size_t map_bytes = count * sizeof(int);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(xs, buffers.xs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "multiple-sphere geometry", "download x coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(ys, buffers.ys, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "multiple-sphere geometry", "download y coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(zs, buffers.zs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "multiple-sphere geometry", "download z coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(sphere_map, buffers.dipole_map, map_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "multiple-sphere geometry", "download sphere map"))
        return 5;
    return 0;
}

//-----------------------------------------------------------
extern "C" int ifdda_concentric_sphere_geometry_cuda(int subdivisions, CU_REAL cell_size,
                                                     CU_REAL center, int sphere_count,
                                                     const CU_REAL *radii, CU_REAL *xs, CU_REAL *ys,
                                                     CU_REAL *zs, int *layer_map) {
    if (subdivisions <= 0 || subdivisions > INT_MAX / subdivisions ||
        subdivisions * subdivisions > INT_MAX / subdivisions || sphere_count <= 0 ||
        sphere_count > 20 || !radii || !xs || !ys || !zs || !layer_map)
        return 1;
    const size_t count = size_t(subdivisions) * size_t(subdivisions) * size_t(subdivisions);
    cuda_data::CubeGeometryDeviceBuffers buffers{};
    if (!cuda_data::get_device_cube_geometry_buffers(count, &buffers, size_t(sphere_count)))
        return 2;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.shape_parameters, radii,
                                           size_t(sphere_count) * sizeof(CU_REAL),
                                           cuda_utils::TransferCategory::Geometry),
            "concentric-sphere geometry", "upload radii") ||
        !cuda_utils::check_cuda(
            concentric_sphere_geometry_wrapper(buffers.xs, buffers.ys, buffers.zs,
                                               buffers.dipole_map, subdivisions, cell_size, center,
                                               buffers.shape_parameters, sphere_count),
            "concentric-sphere geometry", "launch kernel") ||
        !cuda_utils::check_cuda(cudaDeviceSynchronize(), "concentric-sphere geometry",
                                "complete mesh generation"))
        return 3;
    const size_t real_bytes = count * sizeof(CU_REAL);
    const size_t map_bytes = count * sizeof(int);
    if (!cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(xs, buffers.xs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "concentric-sphere geometry", "download x coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(ys, buffers.ys, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "concentric-sphere geometry", "download y coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(zs, buffers.zs, real_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "concentric-sphere geometry", "download z coordinates") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(layer_map, buffers.dipole_map, map_bytes,
                                           cuda_utils::TransferCategory::Geometry),
            "concentric-sphere geometry", "download layer map"))
        return 4;
    return 0;
}

//-----------------------------------------------------------
//-----------------------------------------------------------
namespace {
void set_cube_fortran_message(char* destination, size_t length, const char* message) {
    if (!destination)
        return;
    std::memset(destination, ' ', length);
    if (message && length)
        std::memcpy(destination, message, std::min(length, std::strlen(message)));
}
}

//-----------------------------------------------------------
extern "C" void objetcube_(
    const char* trope, const COMPLEX* epsilon_object, const COMPLEX* epsilon_anisotropic,
    const COMPLEX* epsilon0, CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
    const CU_REAL* k0, CU_REAL* cell_size, int* dipole_map,
    const int* nnnr, const int* nmax, int* sphere_count, int* dipole_count,
    int* nx, int* ny, int* nz, const char* method, COMPLEX* epsilon,
    COMPLEX* polarizability, const int* isotropy, CU_REAL* side,
    CU_REAL* center_x, CU_REAL* center_y, CU_REAL* center_z, const CU_REAL* sldr,
    const int* material_mode, const int64_t* file_id, const int64_t* group_iddip,
    char* infostr, int* nstop, size_t trope_length, size_t method_length,
    size_t infostr_length) {
    ifdda_object_all_cube_cuda(2, trope, epsilon_object, epsilon_anisotropic,
                               epsilon0, xs, ys, zs, k0, cell_size, dipole_map,
                               nnnr, nmax, sphere_count, dipole_count, nx, ny, nz,
                               method, epsilon, polarizability, isotropy, side,
                               center_x, center_y, center_z, sldr, material_mode,
                               file_id, group_iddip, infostr, nstop, trope_length,
                               method_length, infostr_length);
}
