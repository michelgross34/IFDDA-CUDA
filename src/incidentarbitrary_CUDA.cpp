#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
//-----------------------------------------------------------
void set_fortran_message(char *destination, size_t count, const char *message) {
    if (!destination)
        return;
    std::memset(destination, ' ', count);
    if (message && count)
        std::memcpy(destination, message, std::min(count, std::strlen(message)));
}

//-----------------------------------------------------------
double parse_real_field(char *field, bool *ok) {
    for (int i = 0; i < 22; ++i)
        if (field[i] == 'D' || field[i] == 'd')
            field[i] = 'E';
    field[22] = '\0';
    char *end = nullptr;
    errno = 0;
    const double value = std::strtod(field, &end);
    *ok = (end != field && errno == 0);
    return value;
}

//-----------------------------------------------------------
bool read_grid_component(FILE *file, std::vector<COMPLEX> *output, int component) {
    char record[23];
    bool ok = false;
    const size_t count = output->size();
    for (size_t i = 0; i < count; ++i) {
        if (std::fread(record, 1, 22, file) != 22)
            return false;
        const CU_REAL value = CU_REAL(parse_real_field(record, &ok));
        if (!ok)
            return false;
        const COMPLEX old = (*output)[i];
        (*output)[i] = component == 0 ? COMPLEX(value, old.imag())
                                      : COMPLEX(old.real(), value);
    }
    return true;
}

//-----------------------------------------------------------
bool parse_metadata(char *name, int *nx, int *ny, int *nz, CU_REAL *dx, CU_REAL *dy, CU_REAL *dz,
                    CU_REAL *xmin, CU_REAL *ymin, CU_REAL *zmin) {
    FILE *file = std::fopen(name, "r");
    if (!file)
        return false;
    char lines[3][512] = {};
    bool read_ok = std::fgets(lines[0], sizeof(lines[0]), file) &&
                   std::fgets(lines[1], sizeof(lines[1]), file) &&
                   std::fgets(lines[2], sizeof(lines[2]), file);
    std::fclose(file);
    if (!read_ok)
        return false;
    if (std::sscanf(lines[0], "%d %d %d", nx, ny, nz) != 3)
        return false;
    CU_REAL *values[6] = {dx, dy, dz, xmin, ymin, zmin};
    for (int row = 1; row < 3; ++row) {
        char *cursor = lines[row];
        for (int column = 0; column < 3; ++column) {
            while (*cursor == ' ' || *cursor == '\t' || *cursor == ',' || *cursor == '\r' ||
                   *cursor == '\n')
                ++cursor;
            char *token = cursor;
            while (*cursor && *cursor != ' ' && *cursor != '\t' && *cursor != ',' &&
                   *cursor != '\r' && *cursor != '\n') {
                if (*cursor == 'D' || *cursor == 'd')
                    *cursor = 'E';
                ++cursor;
            }
            if (token == cursor)
                return false;
            const char saved = *cursor;
            *cursor = '\0';
            char *end = nullptr;
            errno = 0;
            const double parsed = std::strtod(token, &end);
            *cursor = saved;
            if (end == token || errno != 0)
                return false;
            *values[(row - 1) * 3 + column] = CU_REAL(parsed);
        }
    }
    return true;
}

//-----------------------------------------------------------
void incident_arbitrary_cuda(const REAL *xs, const REAL *ys, const REAL *zs,
                             const REAL *cube_size, const int *nxm,
                             const int *nym, const int *nzm, const int *dipole_count,
                             int *stop, const char *namefile,
                             char *info, const CU_REAL *device_x,
                             const CU_REAL *device_y, const CU_REAL *device_z,
                             CUCOMPLEX *device_output,
                             const REAL *validation_bounds = nullptr) {
    constexpr size_t info_length = 64;
    static const char *const filenames[6] = {"Exr.mat", "Exi.mat", "Eyr.mat",
                                             "Eyi.mat", "Ezr.mat", "Ezi.mat"};
    if (stop)
        *stop = 1;
    if (!xs || !ys || !zs || !cube_size || !device_output ||
        !nxm || !nym || !nzm || !dipole_count ||
        !stop || !namefile || !info || *dipole_count <= 0 || *nxm <= 0 ||
        *nym <= 0 || *nzm <= 0) {
        set_fortran_message(info, info_length, "arbitrary incident: invalid CUDA arguments");
        return;
    }

    char path[65];
    std::memcpy(path, namefile, 64);
    path[64] = '\0';
    for (int i = 63; i >= 0 && (path[i] == ' ' || path[i] == '\0'); --i)
        path[i] = '\0';
    int nx = 0, ny = 0, nz = 0;
    CU_REAL dx_nm = 0, dy_nm = 0, dz_nm = 0;
    CU_REAL xmin_nm = 0, ymin_nm = 0, zmin_nm = 0;
    if (!parse_metadata(path, &nx, &ny, &nz, &dx_nm, &dy_nm, &dz_nm, &xmin_nm, &ymin_nm,
                        &zmin_nm) ||
        nx <= 0 || ny <= 0 || nz <= 0 ||
        size_t(nx) > std::numeric_limits<size_t>::max() / size_t(ny) ||
        size_t(nx) * size_t(ny) > std::numeric_limits<size_t>::max() / size_t(nz)) {
        set_fortran_message(info, info_length, "arbitrary incident: invalid incident grid file");
        return;
    }
    const size_t grid_count = size_t(nx) * size_t(ny) * size_t(nz);
    const CU_REAL scale_nm = CU_REAL(1.0e-9);
    const CU_REAL dx = CU_REAL(dx_nm) * scale_nm;
    const CU_REAL dy = CU_REAL(dy_nm) * scale_nm;
    const CU_REAL dz = CU_REAL(dz_nm) * scale_nm;
    const CU_REAL xmin = CU_REAL(xmin_nm) * scale_nm;
    const CU_REAL ymin = CU_REAL(ymin_nm) * scale_nm;
    const CU_REAL zmin = CU_REAL(zmin_nm) * scale_nm;
    const CU_REAL dmax = std::max(dx, std::max(dy, dz));
    const CU_REAL mesh_tolerance =
        CU_REAL(10) * std::numeric_limits<CU_REAL>::epsilon() *
        std::max(std::abs(*cube_size), std::abs(dmax));
    if (dx <= CU_REAL(0) || dy <= CU_REAL(0) || dz <= CU_REAL(0) ||
        *cube_size + mesh_tolerance < dmax) {
        set_fortran_message(info, info_length, "object mesh size < mesh size incident field");
        return;
    }
    const CU_REAL xmax = xmin + CU_REAL(nx - 1) * dx;
    const CU_REAL ymax = ymin + CU_REAL(ny - 1) * dy;
    const CU_REAL zmax = zmin + CU_REAL(nz - 1) * dz;
    const CU_REAL validation_xmin = validation_bounds ? validation_bounds[0] : xs[0];
    const CU_REAL validation_xmax = validation_bounds ? validation_bounds[1] : xs[*dipole_count - 1];
    const CU_REAL validation_ymin = validation_bounds ? validation_bounds[2] : ys[0];
    const CU_REAL validation_ymax = validation_bounds ? validation_bounds[3] : ys[*dipole_count - 1];
    const CU_REAL validation_zmin = validation_bounds ? validation_bounds[4] : zs[0];
    const CU_REAL validation_zmax = validation_bounds ? validation_bounds[5] : zs[*dipole_count - 1];
    if (validation_xmin < xmin || validation_xmax > xmax ||
        validation_ymin < ymin || validation_ymax > ymax ||
        validation_zmin < zmin || validation_zmax > zmax) {
        set_fortran_message(info, info_length, "objet not inside the incident box");
        *stop = 1;
        std::fprintf(stderr, "objet not inside the incident box (arbitrary illumination)\n");
        return;
    }

    FILE *files[6] = {};
    for (int i = 0; i < 6; ++i) {
        files[i] = std::fopen(filenames[i], "rb");
        if (!files[i]) {
            char message[96];
            std::snprintf(message, sizeof(message), "arbitrary incident: %.40s does not exist",
                          filenames[i]);
            set_fortran_message(info, info_length, message);
            for (FILE *file : files)
                if (file)
                    std::fclose(file);
            return;
        }
    }
    std::vector<COMPLEX> grid_x(grid_count, COMPLEX{});
    std::vector<COMPLEX> grid_y(grid_count, COMPLEX{});
    std::vector<COMPLEX> grid_z(grid_count, COMPLEX{});
    const bool loaded =
        read_grid_component(files[0], &grid_x, 0) && read_grid_component(files[1], &grid_x, 1) &&
        read_grid_component(files[2], &grid_y, 0) && read_grid_component(files[3], &grid_y, 1) &&
        read_grid_component(files[4], &grid_z, 0) && read_grid_component(files[5], &grid_z, 1);
    for (FILE *file : files)
        if (file)
            std::fclose(file);
    if (!loaded) {
        set_fortran_message(info, info_length, "arbitrary incident: error reading field files");
        return;
    }

    if (!initialize_cuda_device()) {
        set_fortran_message(info, info_length,
                            "CUDA device initialization failed for arbitrary illumination");
        return;
    }
    cuda_data::ArbitraryIlluminationDeviceBuffers buffers{};
    const CU_REAL *coords_x = device_x;
    const CU_REAL *coords_y = device_y;
    const CU_REAL *coords_z = device_z;
    if (!coords_x || !coords_y || !coords_z)
        cuda_data::get_device_geometry_coordinate_aliases(
            xs, ys, zs, size_t(*dipole_count), &coords_x, &coords_y, &coords_z);
    const bool resident_coordinates = coords_x && coords_y && coords_z;
    if (!cuda_data::get_device_arbitrary_illumination_buffers(
            grid_count, size_t(*dipole_count), !resident_coordinates,
            false, &buffers)) {
        set_fortran_message(info, info_length, "CUDA arbitrary illumination buffers unavailable");
        return;
    }
    using cuda_utils::TransferCategory;
    cudaError_t status = cudaSuccess;
    if (!resident_coordinates)
        status = cuda_data::copy_host_to_device(
            buffers.xs, xs, size_t(*dipole_count) * sizeof(CU_REAL), TransferCategory::Geometry);
    if (status == cudaSuccess)
        if (!resident_coordinates)
            status = cuda_data::copy_host_to_device(
                buffers.ys, ys, size_t(*dipole_count) * sizeof(CU_REAL),
                TransferCategory::Geometry);
    if (status == cudaSuccess && !resident_coordinates)
        status = cuda_data::copy_host_to_device(
            buffers.zs, zs, size_t(*dipole_count) * sizeof(CU_REAL), TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_complex_to_device(buffers.grid_x, grid_x.data(),
                                                grid_count,
                                                TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_complex_to_device(buffers.grid_y, grid_y.data(),
                                                grid_count,
                                                TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_complex_to_device(buffers.grid_z, grid_z.data(),
                                                grid_count,
                                                TransferCategory::IncidentField);
    if (status == cudaSuccess)
        status = arbitrary_illumination_interpolate_wrapper(
            resident_coordinates ? coords_x : buffers.xs,
            resident_coordinates ? coords_y : buffers.ys,
            resident_coordinates ? coords_z : buffers.zs,
            *dipole_count, buffers.grid_x, buffers.grid_y,
            buffers.grid_z, nx, ny, nz, xmin, ymin, zmin, dx, dy, dz, device_output);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "arbitrary CUDA illumination failed: %s\n",
                     cudaGetErrorString(status));
        set_fortran_message(info, info_length, "CUDA arbitrary illumination calculation failed");
        return;
    }
    set_fortran_message(info, info_length, "");
    *stop = 0;
}
} // namespace

//-----------------------------------------------------------
extern "C" void ifdda_incident_arbitrary_cuda_(const REAL *xs, const REAL *ys,
                                               const REAL *zs, const REAL *cube_size,
                                               COMPLEX *output, const int *nxm, const int *nym,
                                               const int *nzm, const int *dipole_count,
                                               const int *direct_device_consumer, int *stop,
                                               const char *namefile, char *info) {
    (void)direct_device_consumer; // Public Fortran ABI retained; solver uses immutable FF0.
    if (!dipole_count || *dipole_count < 0 || !output) {
        if (stop) *stop = 1;
        return;
    }
    CUCOMPLEX *device_ff0 = cuda_data::get_ff0(size_t(*dipole_count));
    if (!device_ff0) {
        if (stop) *stop = 1;
        return;
    }
    incident_arbitrary_cuda(xs, ys, zs, cube_size, nxm, nym, nzm, dipole_count,
                            stop, namefile, info,
                            nullptr, nullptr, nullptr, device_ff0);
    if (stop && *stop == 0)
        if (cuda_data::copy_device_complex_to_host(output, device_ff0,
                size_t(3) * size_t(*dipole_count),
                cuda_utils::TransferCategory::IncidentField) != cudaSuccess)
            *stop = 1;
}

//-----------------------------------------------------------
extern "C" void ifdda_incident_arbitrary_cuda_device_grid_bounds_(
    const REAL *grid_xmin, const REAL *grid_xmax,
    const REAL *grid_ymin, const REAL *grid_ymax,
    const REAL *grid_zmin, const REAL *grid_zmax,
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const REAL *cube_size,
    CUCOMPLEX *output_device, const int *nxm, const int *nym,
    const int *nzm, const int *dipole_count, int *stop,
    const char *namefile, char *info) {
    if (!grid_xmin || !grid_xmax || !grid_ymin || !grid_ymax ||
        !grid_zmin || !grid_zmax) {
        if (stop)
            *stop = 1;
        set_fortran_message(info, 64, "arbitrary incident: invalid grid bounds");
        return;
    }
    const REAL bounds[6] = {
        *grid_xmin, *grid_xmax, *grid_ymin, *grid_ymax, *grid_zmin, *grid_zmax};
    incident_arbitrary_cuda(
        bounds, bounds + 2, bounds + 4, cube_size, nxm, nym, nzm,
        dipole_count, stop, namefile, info, xs_device, ys_device,
        zs_device, output_device, bounds);
}
