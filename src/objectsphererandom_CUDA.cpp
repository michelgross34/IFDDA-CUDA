#include "kernel.h"
#include "cuda_interfaces.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using FortranComplex = COMPLEX;

namespace {

//-----------------------------------------------------------
void set_error(char *info, std::size_t length, int *stop, const char *message) {
    if (stop)
        *stop = 1;
    if (!info || length == 0)
        return;
    std::memset(info, ' ', length);
    std::memcpy(info, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
bool is_isotropic(const char *trope, std::size_t length) {
    return trope && length >= 3 && std::memcmp(trope, "iso", 3) == 0;
}

//-----------------------------------------------------------
std::size_t tensor_index(int dipole, int nmax, int niso, int row, int column) {
    return std::size_t(dipole) + std::size_t(nmax) *
           std::size_t(row + niso * column);
}

//-----------------------------------------------------------
bool write_values(const char *path, const CU_REAL *values, int count, int stride) {
    FILE *file = std::fopen(path, "w");
    if (!file)
        return false;
    for (int i = 0; i < count; ++i)
        std::fprintf(file, " %.17g\n",
                     static_cast<double>(values[std::size_t(i) * stride]));
    return std::fclose(file) == 0;
}

//-----------------------------------------------------------
bool write_geometry_files(int nx, int ny, int nz, int dipole_count,
                          const CU_REAL *mesh_x, const CU_REAL *mesh_y,
                          const CU_REAL *mesh_z, const CU_REAL *xs,
                          const CU_REAL *ys, const CU_REAL *zs) {
    const int mesh_count = nx * ny * nz;
    return write_values("x.mat", mesh_x, nx, 1) &&
           write_values("y.mat", mesh_y, ny, nx) &&
           write_values("z.mat", mesh_z, nz, nx * ny) &&
           write_values("xc.mat", xs, dipole_count, 1) &&
           write_values("yc.mat", ys, dipole_count, 1) &&
           write_values("zc.mat", zs, dipole_count, 1) && mesh_count > 0;
}

//-----------------------------------------------------------
void set_complex(FortranComplex *array, std::size_t index,
                 COMPLEX value) {
    array[index] = value;
}
} // namespace




//-----------------------------------------------------------
extern "C" void objetsphererandom_(
    const char *trope, const FortranComplex *eps,
    const FortranComplex *epsani, const FortranComplex *eps0, CU_REAL *xs,
    CU_REAL *ys, CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube,
    int *tabdip, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const int *nxm,
    const int *nym, const int *nzm, const char *methode,
    FortranComplex *epsilon, FortranComplex *polarisa, const int *niso,
    CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez, CU_REAL *xg,
    CU_REAL *yg, CU_REAL *zg, CU_REAL *xsphe, CU_REAL *ysphe,
    CU_REAL *zsphe, CU_REAL *radius, CU_REAL *density, const int *nseed,
    const int *na, CU_REAL *sldr, const int *nmat, const int64_t *file_id,
    const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length) {
    ifdda_object_all_randomsphere_cuda(11, trope, eps, epsani, eps0, xs, ys, zs, k0, aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, methode, epsilon, polarisa, niso, sidex, sidey, sidez, xg, yg, zg, xsphe, ysphe, zsphe, radius, density, nseed, na, sldr, nmat, file_id, group_iddip, infostr, nstop, trope_length, methode_length, infostr_length);
}
