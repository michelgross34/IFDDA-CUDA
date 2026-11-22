#include "cuda_data.h"
#include "cuda_utils.h"
#include "cuda_interfaces.h"
#include "kernel.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace {
//-----------------------------------------------------------
struct ArbitraryDerivativeGrid {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    CU_REAL dx = CU_REAL(0);
    CU_REAL dy = CU_REAL(0);
    CU_REAL dz = CU_REAL(0);
    CU_REAL xmin = CU_REAL(0);
    CU_REAL ymin = CU_REAL(0);
    CU_REAL zmin = CU_REAL(0);
    size_t sites = 0;
    std::string key;
    bool uploaded = false;
};

ArbitraryDerivativeGrid& derivative_grid() {
    static ArbitraryDerivativeGrid grid;
    return grid;
}
//-------------------------------------------

std::mutex& derivative_mutex() {
    static std::mutex mutex;
    return mutex;
}

//-----------------------------------------------------------
bool read_grid_header(const std::string& filename, ArbitraryDerivativeGrid& grid) {
    std::ifstream input(filename);
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double xmin = 0.0;
    double ymin = 0.0;
    double zmin = 0.0;
    if (!(input >> grid.nx >> grid.ny >> grid.nz >> dx >> dy >> dz >> xmin >> ymin >> zmin))
        return false;
    if (grid.nx < 2 || grid.ny < 2 || grid.nz < 2 ||
        dx <= 0.0 || dy <= 0.0 || dz <= 0.0)
        return false;
    const size_t nx = size_t(grid.nx);
    const size_t ny = size_t(grid.ny);
    const size_t nz = size_t(grid.nz);
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz)
        return false;
    grid.sites = nx * ny * nz;
    if (grid.sites > std::numeric_limits<size_t>::max() / (9 * sizeof(CUCOMPLEX)))
        return false;
    constexpr double nm_to_m = 1.0e-9;
    grid.dx = CU_REAL(dx * nm_to_m);
    grid.dy = CU_REAL(dy * nm_to_m);
    grid.dz = CU_REAL(dz * nm_to_m);
    grid.xmin = CU_REAL(xmin * nm_to_m);
    grid.ymin = CU_REAL(ymin * nm_to_m);
    grid.zmin = CU_REAL(zmin * nm_to_m);
    return true;
}

//-----------------------------------------------------------
bool read_derivative_file(const char* filename, size_t sites,
                          std::vector<CUCOMPLEX>& values) {
    constexpr size_t record_size = 2 * sizeof(CU_REAL);
    std::ifstream input(filename, std::ios::binary);
    if (!input || sites > std::numeric_limits<size_t>::max() / record_size)
        return false;
    values.resize(sites);
    std::array<char, record_size> record{};
    for (size_t i = 0; i < sites; ++i) {
        input.read(record.data(), std::streamsize(record.size()));
        if (!input)
            return false;
        CU_REAL real_part = CU_REAL(0);
        CU_REAL imag_part = CU_REAL(0);
        std::memcpy(&real_part, record.data(), sizeof(real_part));
        std::memcpy(&imag_part, record.data() + sizeof(real_part), sizeof(imag_part));
        values[i] = CUCOMPLEX{real_part, imag_part};
    }
    return true;
}

//-----------------------------------------------------------
bool read_field_grid(std::vector<CUCOMPLEX>& field, size_t sites) {
    constexpr std::array<const char*, 6> filenames = {
        "Exr.mat", "Exi.mat", "Eyr.mat", "Eyi.mat", "Ezr.mat", "Ezi.mat"};
    constexpr size_t record_size = 22;
    std::array<std::vector<CU_REAL>, 6> components;
    std::array<char, record_size + 1> record{};
    for (size_t component = 0; component < filenames.size(); ++component) {
        std::ifstream input(filenames[component], std::ios::binary);
        if (!input || sites > std::numeric_limits<size_t>::max() / record_size)
            return false;
        components[component].resize(sites);
        for (size_t site = 0; site < sites; ++site) {
            input.read(record.data(), std::streamsize(record_size));
            if (!input)
                return false;
            for (char& value : record) {
                if (value == 'D' || value == 'd')
                    value = 'E';
            }
            char* end = nullptr;
            errno = 0;
            const double parsed = std::strtod(record.data(), &end);
            if (end == record.data() || errno == ERANGE || !std::isfinite(parsed))
                return false;
            components[component][site] = CU_REAL(parsed);
        }
    }
    field.resize(3 * sites);
    for (size_t site = 0; site < sites; ++site) {
        field[3 * site] = CUCOMPLEX{components[0][site], components[1][site]};
        field[3 * site + 1] = CUCOMPLEX{components[2][site], components[3][site]};
        field[3 * site + 2] = CUCOMPLEX{components[4][site], components[5][site]};
    }
    return true;
}

//-----------------------------------------------------------
bool upload_derivative_grid(const std::string& header_name,
                            ArbitraryDerivativeGrid& grid,
                            cuda_data::ArbitraryDerivativeDeviceBuffers& buffers) {
    constexpr std::array<const char*, 9> filenames = {
        "Exdx.mat", "Eydx.mat", "Ezdx.mat", "Exdy.mat", "Eydy.mat",
        "Ezdy.mat", "Exdz.mat", "Eydz.mat", "Ezdz.mat"};
    std::vector<CUCOMPLEX> host_grid(9 * grid.sites);
    std::vector<CUCOMPLEX> component;
    for (size_t field = 0; field < filenames.size(); ++field) {
        if (!read_derivative_file(filenames[field], grid.sites, component))
            return false;
        std::copy(component.begin(), component.end(),
                  host_grid.begin() + std::ptrdiff_t(field * grid.sites));
    }
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.derivative_grids, host_grid.data(),
                host_grid.size() * sizeof(CUCOMPLEX),
                cuda_utils::TransferCategory::IncidentField),
            "arbitrary incident derivative", "upload derivative grids"))
        return false;
    grid.key = header_name;
    grid.uploaded = true;
    return true;
}

//-----------------------------------------------------------
bool interpolate_derivative(const int* test, const CU_REAL* x, const CU_REAL* y,
                            const CU_REAL* z, const std::string& header_name,
                            CUCOMPLEX* output) {
    if (!test || !x || !y || !z || !output || *test < 1 || *test > 4)
        return false;
    std::lock_guard<std::mutex> lock(derivative_mutex());
    ArbitraryDerivativeGrid& grid = derivative_grid();
    if (grid.key != header_name || grid.sites == 0) {
        grid = ArbitraryDerivativeGrid{};
        if (!read_grid_header(header_name, grid))
            return false;
    }
    const CU_REAL xmax = grid.xmin + CU_REAL(grid.nx - 1) * grid.dx;
    const CU_REAL ymax = grid.ymin + CU_REAL(grid.ny - 1) * grid.dy;
    const CU_REAL zmax = grid.zmin + CU_REAL(grid.nz - 1) * grid.dz;
    if (*x < grid.xmin || *x > xmax || *y < grid.ymin || *y > ymax ||
        *z < grid.zmin || *z > zmax) {
        std::fprintf(stderr,
            "incidentarbitraryder1_: object coordinate lies outside arbitrary-field grid\n");
        return false;
    }
    if (!initialize_cuda_device())
        return false;
    cuda_data::ArbitraryDerivativeDeviceBuffers buffers{};
    if (!cuda_data::get_device_arbitrary_derivative_buffers(grid.sites, 1, &buffers))
        return false;
    if (!grid.uploaded && !upload_derivative_grid(header_name, grid, buffers))
        return false;

    const CU_REAL host_x = *x;
    const CU_REAL host_y = *y;
    const CU_REAL host_z = *z;
    if (!cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.xs, &host_x, sizeof(host_x),
                cuda_utils::TransferCategory::Control),
            "arbitrary incident derivative", "upload x coordinate") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.ys, &host_y, sizeof(host_y),
                cuda_utils::TransferCategory::Control),
            "arbitrary incident derivative", "upload y coordinate") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_host_to_device(buffers.zs, &host_z, sizeof(host_z),
                cuda_utils::TransferCategory::Control),
            "arbitrary incident derivative", "upload z coordinate") ||
        !cuda_utils::check_cuda(cudaMemset(buffers.result, 0, 9 * sizeof(CUCOMPLEX)),
            "arbitrary incident derivative", "clear interpolation output"))
        return false;

    const int first_axis = *test == 4 ? 0 : *test - 1;
    const int last_axis = *test == 4 ? 2 : *test - 1;
    for (int axis = first_axis; axis <= last_axis; ++axis) {
        const CUCOMPLEX* axis_grid = buffers.derivative_grids +
            size_t(axis) * 3 * grid.sites;
        if (!cuda_utils::check_cuda(
                arbitrary_illumination_interpolate_wrapper(
                    buffers.xs, buffers.ys, buffers.zs, 1,
                    axis_grid, axis_grid + grid.sites, axis_grid + 2 * grid.sites,
                    grid.nx, grid.ny, grid.nz, grid.xmin, grid.ymin, grid.zmin,
                    grid.dx, grid.dy, grid.dz, buffers.result + 3 * axis),
                "arbitrary incident derivative", "interpolate derivative on GPU"))
            return false;
    }
    return cuda_utils::check_cuda(
        cuda_data::copy_device_to_host(output, buffers.result, 9 * sizeof(CUCOMPLEX),
            cuda_utils::TransferCategory::Result),
        "arbitrary incident derivative", "download interpolated derivative");
}
} // namespace

//-----------------------------------------------------------
extern "C" void incidentarbitraryder1_(const int* test, const REAL* x,
                                         const REAL* y, const REAL* z,
                                         const char* namefile, COMPLEX* derivative,
                                         size_t namefile_length) {
    if (!namefile || !derivative || !x || !y || !z)
        return;
    std::string header_name(namefile, namefile_length);
    while (!header_name.empty() &&
           (header_name.back() == ' ' || header_name.back() == '\0'))
        header_name.pop_back();
    CUCOMPLEX device_result[9]{};
    const CU_REAL cx = static_cast<CU_REAL>(*x);
    const CU_REAL cy = static_cast<CU_REAL>(*y);
    const CU_REAL cz = static_cast<CU_REAL>(*z);
    if (!interpolate_derivative(test, &cx, &cy, &cz, header_name, device_result)) {
        for (int i = 0; i < 9; ++i)
            derivative[i] = COMPLEX(0, 0);
        std::fprintf(stderr, "incidentarbitraryder1_: CUDA interpolation failed\n");
        return;
    }
    for (int i = 0; i < 9; ++i)
        derivative[i] = COMPLEX(device_result[i].x, device_result[i].y);
}

//-----------------------------------------------------------
extern "C" void ifdda_arbitrary_derivatives_cuda_(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, const int* tabdip,
    const int* coordinate_count, const int* dipole_count,
    const char* filename, const int* filename_length,
    CUCOMPLEX* output, int* status_out) {
    if (status_out)
        *status_out = 1;
    if (!xs || !ys || !zs || !tabdip || !coordinate_count || !dipole_count ||
        !filename || !filename_length || !output || !status_out ||
        *coordinate_count <= 0 || *dipole_count <= 0 || *dipole_count > INT_MAX ||
        *filename_length <= 0 ||
        size_t(*dipole_count) > std::numeric_limits<size_t>::max() / 9 / sizeof(CUCOMPLEX))
        return;

    std::string header_name(filename, size_t(*filename_length));
    while (!header_name.empty() &&
           (header_name.back() == ' ' || header_name.back() == '\0'))
        header_name.pop_back();
    if (header_name.empty())
        return;

    std::lock_guard<std::mutex> lock(derivative_mutex());
    ArbitraryDerivativeGrid& grid = derivative_grid();
    if (grid.key != header_name || grid.sites == 0) {
        grid = ArbitraryDerivativeGrid{};
        if (!read_grid_header(header_name, grid))
            return;
    }
    const size_t count = size_t(*dipole_count);
    std::vector<CU_REAL> point_x(count, CU_REAL(0));
    std::vector<CU_REAL> point_y(count, CU_REAL(0));
    std::vector<CU_REAL> point_z(count, CU_REAL(0));
    const CU_REAL xmax = grid.xmin + CU_REAL(grid.nx - 1) * grid.dx;
    const CU_REAL ymax = grid.ymin + CU_REAL(grid.ny - 1) * grid.dy;
    const CU_REAL zmax = grid.zmin + CU_REAL(grid.nz - 1) * grid.dz;
    const CU_REAL safe_x = grid.xmin + CU_REAL(grid.nx / 2) * grid.dx;
    const CU_REAL safe_y = grid.ymin + CU_REAL(grid.ny / 2) * grid.dy;
    const CU_REAL safe_z = grid.zmin + CU_REAL(grid.nz / 2) * grid.dz;
    std::vector<unsigned char> active(count, 0);
    for (size_t point = 0; point < count; ++point) {
        const int index = tabdip[point];
        if (index == 0) {
            point_x[point] = safe_x;
            point_y[point] = safe_y;
            point_z[point] = safe_z;
            continue;
        }
        if (index < 0 || index > *coordinate_count)
            return;
        const size_t coordinate = size_t(index - 1);
        const CU_REAL x = xs[coordinate];
        const CU_REAL y = ys[coordinate];
        const CU_REAL z = zs[coordinate];
        if (x < grid.xmin || x > xmax || y < grid.ymin || y > ymax ||
            z < grid.zmin || z > zmax)
            return;
        point_x[point] = x;
        point_y[point] = y;
        point_z[point] = z;
        active[point] = 1;
    }
    if (!initialize_cuda_device())
        return;
    cuda_data::ArbitraryDerivativeDeviceBuffers buffers{};
    if (!cuda_data::get_device_arbitrary_derivative_buffers(
            grid.sites, count, &buffers))
        return;
    if (!grid.uploaded) {
        std::vector<CUCOMPLEX> host_field;
        if (!read_field_grid(host_field, grid.sites))
            return;
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_to_device(
                    buffers.field, host_field.data(),
                    host_field.size() * sizeof(CUCOMPLEX),
                    cuda_utils::TransferCategory::IncidentField),
                "arbitrary incident derivative", "upload incident field grid"))
            return;
        const std::array<CU_REAL, 3> spacings = {grid.dx, grid.dy, grid.dz};
        for (int axis = 0; axis < 3; ++axis) {
            CUCOMPLEX* component_major =
                buffers.derivative_grids + size_t(axis) * 3 * grid.sites;
            if (!cuda_utils::check_cuda(
                    optical_field_derivative_wrapper(
                        buffers.field, buffers.derivative_scratch,
                        grid.nx, grid.ny, grid.nz, axis, spacings[size_t(axis)]),
                    "arbitrary incident derivative", "differentiate incident grid") ||
                !cuda_utils::check_cuda(
                    arbitrary_derivative_grid_pack_wrapper(
                        buffers.derivative_scratch, component_major, grid.sites),
                    "arbitrary incident derivative", "pack derivative components"))
                return;
        }
        grid.key = header_name;
        grid.uploaded = true;
    }

    using cuda_utils::TransferCategory;
    const size_t coordinate_bytes = count * sizeof(CU_REAL);
    cudaError_t status = cuda_data::copy_host_to_device(
        buffers.xs, point_x.data(), coordinate_bytes, TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.ys, point_y.data(), coordinate_bytes, TransferCategory::Geometry);
    if (status == cudaSuccess)
        status = cuda_data::copy_host_to_device(
            buffers.zs, point_z.data(), coordinate_bytes, TransferCategory::Geometry);
    for (int axis = 0; status == cudaSuccess && axis < 3; ++axis) {
        const CUCOMPLEX* axis_grid = buffers.derivative_grids +
            size_t(axis) * 3 * grid.sites;
        status = arbitrary_illumination_interpolate_wrapper(
            buffers.xs, buffers.ys, buffers.zs, *dipole_count,
            axis_grid, axis_grid + grid.sites, axis_grid + 2 * grid.sites,
            grid.nx, grid.ny, grid.nz, grid.xmin, grid.ymin, grid.zmin,
            grid.dx, grid.dy, grid.dz,
            buffers.result + size_t(axis) * 3 * count);
    }
    std::vector<CUCOMPLEX> host_axis_major(9 * count);
    if (status == cudaSuccess)
        status = cuda_data::copy_device_to_host(
            host_axis_major.data(), buffers.result,
            host_axis_major.size() * sizeof(CUCOMPLEX), TransferCategory::Result);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA arbitrary derivative batch failed: %s\n",
                     cudaGetErrorString(status));
        return;
    }
    for (size_t point = 0; point < count; ++point) {
        for (size_t axis = 0; axis < 3; ++axis)
            for (size_t component = 0; component < 3; ++component)
                output[9 * point + component + 3 * axis] = active[point]
                    ? host_axis_major[3 * axis * count + 3 * point + component]
                    : CUCOMPLEX{CU_REAL(0), CU_REAL(0)};
    }
    *status_out = 0;
}
