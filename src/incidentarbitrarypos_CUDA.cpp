#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <array>
#include <cerrno>
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
struct ArbitraryPointGrid {
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

ArbitraryPointGrid& cached_grid() {
    static ArbitraryPointGrid grid;
    return grid;
}

std::mutex& grid_mutex() {
    static std::mutex mutex;
    return mutex;
}

void set_message(char* output, size_t size, const char* message) {
    if (!output)
        return;
    std::memset(output, ' ', size);
    if (message && size)
        std::memcpy(output, message, std::min(size, std::strlen(message)));
}

//-----------------------------------------------------------
bool load_grid_header(const std::string& name, ArbitraryPointGrid& grid) {
    std::ifstream input(name);
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double xmin = 0.0;
    double ymin = 0.0;
    double zmin = 0.0;
    if (!(input >> grid.nx >> grid.ny >> grid.nz >> dx >> dy >> dz >> xmin >> ymin >> zmin))
        return false;
    if (grid.nx < 1 || grid.ny < 1 || grid.nz < 1 ||
        dx <= 0.0 || dy <= 0.0 || dz <= 0.0)
        return false;
    const size_t nx = size_t(grid.nx);
    const size_t ny = size_t(grid.ny);
    const size_t nz = size_t(grid.nz);
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz)
        return false;
    grid.sites = nx * ny * nz;
    if (grid.sites > std::numeric_limits<size_t>::max() / sizeof(CUCOMPLEX))
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
bool read_real_records(const char* filename, size_t count, std::vector<CU_REAL>& values) {
    constexpr size_t record_size = 22;
    if (count > std::numeric_limits<size_t>::max() / record_size)
        return false;
    std::ifstream input(filename, std::ios::binary);
    if (!input)
        return false;
    std::array<char, record_size + 1> record{};
    values.resize(count);
    for (size_t i = 0; i < count; ++i) {
        input.read(record.data(), std::streamsize(record_size));
        if (!input)
            return false;
        for (size_t j = 0; j < record_size; ++j)
            if (record[j] == 'D' || record[j] == 'd')
                record[j] = 'E';
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(record.data(), &end);
        if (end == record.data() || errno == ERANGE || !std::isfinite(value))
            return false;
        values[i] = CU_REAL(value);
    }
    return true;
}

//-----------------------------------------------------------
bool load_and_upload_grid(const std::string& name, ArbitraryPointGrid& grid,
                          cuda_data::ArbitraryIlluminationDeviceBuffers& buffers) {
    constexpr std::array<const char*, 6> filenames = {
        "Exr.mat", "Exi.mat", "Eyr.mat", "Eyi.mat", "Ezr.mat", "Ezi.mat"};
    std::array<std::vector<CU_REAL>, 6> components;
    for (size_t i = 0; i < filenames.size(); ++i)
        if (!read_real_records(filenames[i], grid.sites, components[i]))
            return false;
    std::array<std::vector<CUCOMPLEX>, 3> fields;
    for (auto& field : fields)
        field.resize(grid.sites);
    for (size_t i = 0; i < grid.sites; ++i) {
        fields[0][i] = CUCOMPLEX{components[0][i], components[1][i]};
        fields[1][i] = CUCOMPLEX{components[2][i], components[3][i]};
        fields[2][i] = CUCOMPLEX{components[4][i], components[5][i]};
    }
    const cuda_utils::TransferCategory category = cuda_utils::TransferCategory::IncidentField;
    if (!cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.grid_x, fields[0].data(), grid.sites * sizeof(CUCOMPLEX), category),
            "arbitrary point field", "upload Ex grid") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.grid_y, fields[1].data(), grid.sites * sizeof(CUCOMPLEX), category),
            "arbitrary point field", "upload Ey grid") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.grid_z, fields[2].data(), grid.sites * sizeof(CUCOMPLEX), category),
            "arbitrary point field", "upload Ez grid"))
        return false;
    grid.key = name;
    grid.uploaded = true;
    return true;
}

//-----------------------------------------------------------
bool interpolate_point(const CU_REAL* x, const CU_REAL* y, const CU_REAL* z,
                       const CU_REAL* cube_size, CUCOMPLEX* ex, CUCOMPLEX* ey,
                       CUCOMPLEX* ez, int* stop, const std::string& header_name,
                       char* info, size_t info_length) {
    if (!x || !y || !z || !cube_size || !ex || !ey || !ez || !stop) {
        set_message(info, info_length, "arbitrary incident: invalid argument");
        return false;
    }
    *stop = 0;
    *ex = CUCOMPLEX{CU_REAL(0), CU_REAL(0)};
    *ey = CUCOMPLEX{CU_REAL(0), CU_REAL(0)};
    *ez = CUCOMPLEX{CU_REAL(0), CU_REAL(0)};
    std::lock_guard<std::mutex> lock(grid_mutex());
    ArbitraryPointGrid& grid = cached_grid();
    if (grid.key != header_name || grid.sites == 0) {
        grid = ArbitraryPointGrid{};
        if (!load_grid_header(header_name, grid)) {
            set_message(info, info_length, "arbitrary incident: invalid name file");
            *stop = 1;
            return false;
        }
    }
    const CU_REAL maximum_step = std::max(grid.dx, std::max(grid.dy, grid.dz));
    if (*cube_size < maximum_step) {
        set_message(info, info_length, "object mesh size < mesh size incident field");
        *stop = 1;
        return false;
    }
    const CU_REAL xmax = grid.xmin + CU_REAL(grid.nx - 1) * grid.dx;
    const CU_REAL ymax = grid.ymin + CU_REAL(grid.ny - 1) * grid.dy;
    const CU_REAL zmax = grid.zmin + CU_REAL(grid.nz - 1) * grid.dz;
    if (*x < grid.xmin || *x > xmax || *y < grid.ymin || *y > ymax ||
        *z < grid.zmin || *z > zmax) {
        set_message(info, info_length, "near field not inside the incident box");
        *stop = 1;
        return false;
    }
    if (!initialize_cuda_device()) {
        set_message(info, info_length, "CUDA device initialization failed");
        *stop = 1;
        return false;
    }
    cuda_data::ArbitraryIlluminationDeviceBuffers buffers{};
    if (!cuda_data::get_device_arbitrary_illumination_buffers(grid.sites, 1, &buffers)) {
        set_message(info, info_length, "CUDA arbitrary interpolation buffers unavailable");
        *stop = 1;
        return false;
    }
    if (!grid.uploaded && !load_and_upload_grid(header_name, grid, buffers)) {
        set_message(info, info_length, "arbitrary incident: error reading field files");
        *stop = 1;
        return false;
    }
    const CU_REAL host_x = *x;
    const CU_REAL host_y = *y;
    const CU_REAL host_z = *z;
    const auto category = cuda_utils::TransferCategory::Geometry;
    if (!cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.xs, &host_x, sizeof(host_x), category),
            "arbitrary point field", "upload x coordinate") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.ys, &host_y, sizeof(host_y), category),
            "arbitrary point field", "upload y coordinate") ||
        !cuda_utils::check_cuda(cuda_data::copy_host_to_device(
            buffers.zs, &host_z, sizeof(host_z), category),
            "arbitrary point field", "upload z coordinate") ||
        !cuda_utils::check_cuda(arbitrary_illumination_interpolate_wrapper(
            buffers.xs, buffers.ys, buffers.zs, 1, buffers.grid_x, buffers.grid_y,
            buffers.grid_z, grid.nx, grid.ny, grid.nz, grid.xmin, grid.ymin, grid.zmin,
            grid.dx, grid.dy, grid.dz, buffers.field),
            "arbitrary point field", "interpolate field on GPU")) {
        set_message(info, info_length, "CUDA arbitrary point interpolation failed");
        *stop = 1;
        return false;
    }
    CUCOMPLEX result[3]{};
    if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
            result, buffers.field, sizeof(result), cuda_utils::TransferCategory::Result),
            "arbitrary point field", "download interpolated field")) {
        set_message(info, info_length, "CUDA arbitrary point result copy failed");
        *stop = 1;
        return false;
    }
    *ex = result[0];
    *ey = result[1];
    *ez = result[2];
    set_message(info, info_length, "");
    return true;
}
} // namespace

//-----------------------------------------------------------
extern "C" void incidentarbitrarypos_(
    const CU_REAL* x, const CU_REAL* y, const CU_REAL* z, const CU_REAL* cube_size,
    CUCOMPLEX* ex, CUCOMPLEX* ey, CUCOMPLEX* ez, int* stop, const char* namefile,
    char* info, size_t namefile_length, size_t info_length) {
    if (!namefile) {
        if (stop)
            *stop = 1;
        set_message(info, info_length, "arbitrary incident: invalid name file");
        return;
    }
    std::string header_name(namefile, namefile_length);
    while (!header_name.empty() &&
           (header_name.back() == ' ' || header_name.back() == '\0'))
        header_name.pop_back();
    interpolate_point(x, y, z, cube_size, ex, ey, ez, stop, header_name,
                      info, info_length);
}
