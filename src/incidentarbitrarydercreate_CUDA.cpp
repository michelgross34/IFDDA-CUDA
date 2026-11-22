#include "kernel.h"
#include "cuda_interfaces.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {
//-----------------------------------------------------------
bool read_header(const std::string& filename, int& nx, int& ny, int& nz,
                 CU_REAL& dx, CU_REAL& dy, CU_REAL& dz) {
    std::ifstream input(filename);
    double dx_file = 0.0;
    double dy_file = 0.0;
    double dz_file = 0.0;
    double xmin = 0.0;
    double ymin = 0.0;
    double zmin = 0.0;
    if (!(input >> nx >> ny >> nz >> dx_file >> dy_file >> dz_file >> xmin >> ymin >> zmin))
        return false;
    if (nx < 5 || ny < 5 || nz < 5 || dx_file <= 0.0 || dy_file <= 0.0 || dz_file <= 0.0)
        return false;
    dx = CU_REAL(dx_file * 1.0e-9);
    dy = CU_REAL(dy_file * 1.0e-9);
    dz = CU_REAL(dz_file * 1.0e-9);
    return true;
}

//-----------------------------------------------------------
bool read_formatted_records(const char* filename, size_t count,
                            std::vector<CU_REAL>& values) {
    constexpr size_t record_size = 22;
    if (count > std::numeric_limits<size_t>::max() / record_size)
        return false;
    std::ifstream input(filename, std::ios::binary);
    if (!input)
        return false;
    std::string records(count * record_size, '\0');
    input.read(records.data(), std::streamsize(records.size()));
    if (input.gcount() != std::streamsize(records.size()))
        return false;
    values.resize(count);
    for (size_t i = 0; i < count; ++i) {
        std::array<char, record_size + 1> text{};
        for (size_t j = 0; j < record_size; ++j) {
            char c = records[i * record_size + j];
            text[j] = (c == 'D' || c == 'd') ? 'E' : c;
        }
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(text.data(), &end);
        if (end == text.data() || errno == ERANGE || !std::isfinite(value))
            return false;
        values[i] = CU_REAL(value);
    }
    return true;
}

//-----------------------------------------------------------
bool write_derivative_file(const char* filename, const COMPLEX* values,
                           size_t count, int component) {
    std::ofstream output(filename, std::ios::binary | std::ios::trunc);
    if (!output)
        return false;
    for (size_t i = 0; i < count; ++i) {
        const COMPLEX value = values[3 * i + size_t(component)];
        constexpr size_t record_size = sizeof(CU_REAL) == sizeof(float)
                                            ? size_t(16)
                                            : size_t(2) * sizeof(CU_REAL);
        std::array<char, record_size> record{};
        const CU_REAL real_part = static_cast<CU_REAL>(value.real());
        const CU_REAL imag_part = static_cast<CU_REAL>(value.imag());
        std::memcpy(record.data(), &real_part, sizeof(real_part));
        std::memcpy(record.data() + sizeof(real_part), &imag_part, sizeof(imag_part));
        output.write(record.data(), std::streamsize(record.size()));
        if (!output)
            return false;
    }
    return true;
}

//-----------------------------------------------------------
bool create_derivatives(const std::string& header_name) {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    CU_REAL dx = CU_REAL(0);
    CU_REAL dy = CU_REAL(0);
    CU_REAL dz = CU_REAL(0);
    if (!read_header(header_name, nx, ny, nz, dx, dy, dz))
        return false;

    const size_t sx = size_t(nx);
    const size_t sy = size_t(ny);
    const size_t sz = size_t(nz);
    if (sx > std::numeric_limits<size_t>::max() / sy || sx * sy >
        std::numeric_limits<size_t>::max() / sz)
        return false;
    const size_t sites = sx * sy * sz;
    if (sites > size_t(std::numeric_limits<int>::max() / 3))
        return false;

    constexpr std::array<const char*, 6> input_names = {
        "Exr.mat", "Exi.mat", "Eyr.mat", "Eyi.mat", "Ezr.mat", "Ezi.mat"};
    constexpr std::array<const char*, 9> output_names = {
        "Exdx.mat", "Eydx.mat", "Ezdx.mat", "Exdy.mat", "Eydy.mat",
        "Ezdy.mat", "Exdz.mat", "Eydz.mat", "Ezdz.mat"};
    for (const char* name : output_names) {
        if (std::filesystem::exists(name))
            return false;
    }

    std::array<std::vector<CU_REAL>, 6> input_values;
    for (size_t i = 0; i < input_names.size(); ++i) {
        if (!read_formatted_records(input_names[i], sites, input_values[i]))
            return false;
    }

    std::vector<COMPLEX> field(3 * sites);
    std::vector<COMPLEX> derivative(3 * sites);
    for (size_t i = 0; i < sites; ++i) {
        field[3 * i] = COMPLEX(input_values[0][i], input_values[1][i]);
        field[3 * i + 1] = COMPLEX(input_values[2][i], input_values[3][i]);
        field[3 * i + 2] = COMPLEX(input_values[4][i], input_values[5][i]);
    }

    const int nmax = int(sites);
    const std::array<CU_REAL, 3> spacings = {dx, dy, dz};
    for (int axis = 1; axis <= 3; ++axis) {
        derivechamp_(&nx, &ny, &nz, &nmax, &spacings[size_t(axis - 1)],
                     field.data(), &axis, derivative.data());
        for (int component = 0; component < 3; ++component) {
            const size_t file_index = size_t((axis - 1) * 3 + component);
            if (!write_derivative_file(output_names[file_index], derivative.data(),
                                       sites, component))
                return false;
        }
    }
    return true;
}
} // namespace

//-----------------------------------------------------------
extern "C" void incidentarbitrarydercreate_(const char* namefile,
                                               size_t namefile_length) {
    if (!namefile)
        return;
    std::string header(namefile, namefile_length);
    while (!header.empty() && (header.back() == ' ' || header.back() == '\0'))
        header.pop_back();
    if (!create_derivatives(header))
        std::fprintf(stderr,
                     "incidentarbitrarydercreate_: CUDA arbitrary-field derivative failed\n");
}
