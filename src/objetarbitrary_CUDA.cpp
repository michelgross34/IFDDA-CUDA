#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <climits>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

//-----------------------------------------------------------
void set_info(char* info, const char* message) {
    if (!info)
        return;
    std::memset(info, ' ', 64);
    if (message)
        std::memcpy(info, message, std::min(std::strlen(message), size_t(64)));
}

//-----------------------------------------------------------
std::size_t array_offset(int nmax, int isotropy, int dipole, int row, int column) {
    return std::size_t(dipole) + std::size_t(row) * std::size_t(nmax) +
        std::size_t(column) * std::size_t(nmax) * std::size_t(isotropy);
}

//-----------------------------------------------------------
std::string fortran_string(const char* text, std::size_t length) {
    std::size_t used = 0;
    while (used < length && text[used] != '\0')
        ++used;
    while (used > 0 && text[used - 1] == ' ')
        --used;
    return std::string(text, used);
}

//-----------------------------------------------------------
bool read_complex(std::istream& input, COMPLEX* value) {
    COMPLEX parsed{};
    if (!value || !(input >> parsed))
        return false;
    *value = parsed;
    return true;
}

CUCOMPLEX device_complex(const COMPLEX& value) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(value.real(), value.imag());
#else
    return make_cuDoubleComplex(value.real(), value.imag());
#endif
}

//-----------------------------------------------------------
int polarizability_method_code(const char method[2]) {
    if (method[0] == 'C' && method[1] == 'M') return 1;
    if (method[0] == 'R' && method[1] == 'R') return 2;
    if (method[0] == 'P' && method[1] == 'S') return 3;
    if (method[0] == 'G' && method[1] == 'B') return 4;
    if (method[0] == 'L' && method[1] == 'A') return 5;
    if (method[0] == 'L' && method[1] == 'R') return 6;
    if (method[0] == 'L' && method[1] == 'S') return 7;
    if (method[0] == 'F' && method[1] == 'G') return 8;
    return 0;
}

struct TemporaryDeviceAllocation {
    void* ptr = nullptr;
    ~TemporaryDeviceAllocation() { if (ptr) cudaFree(ptr); }
};
} // namespace

//-----------------------------------------------------------
extern "C" void objetarbitrary_(
    const char* object_type, COMPLEX* epsilon_value, COMPLEX* epsilon_anisotropic,
    const COMPLEX* epsilon_background, CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
    const CU_REAL* k0, CU_REAL* cell_size, int* dipole_map, const int* random_count,
    const int* nmax, int* dipole_count, int* ndipole, int* nx, int* ny, int* nz,
    const char* polarizability_method, const char* filename, int* particle_count,
    COMPLEX* epsilon, COMPLEX* polarizability, const int* isotropy,
    CU_REAL* sldr, const int* output_mode, const int64_t* file_id, const int64_t* group_id,
    char* infostr, int* nstop, std::size_t object_type_length,
    std::size_t method_length, std::size_t filename_length,
    std::size_t infostr_length) {
    ifdda_object_all_arbitrary_cuda(13, object_type, epsilon_value, epsilon_anisotropic, epsilon_background, xs, ys, zs, k0, cell_size, dipole_map, random_count, nmax, dipole_count, ndipole, nx, ny, nz, polarizability_method, filename, particle_count, epsilon, polarizability, isotropy, sldr, output_mode, file_id, group_id, infostr, nstop, object_type_length, method_length, filename_length, infostr_length);
}
