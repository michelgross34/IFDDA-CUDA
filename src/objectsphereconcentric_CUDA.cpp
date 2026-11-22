#include "kernel.h"
#include "cuda_interfaces.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using FortranComplex = COMPLEX;

//-----------------------------------------------------------
void set_error(char *info, std::size_t length, int *stop, const char *message) {
    if (stop)
        *stop = 1;
    if (info && length) {
        std::memset(info, ' ', length);
        if (message)
            std::memcpy(info, message, std::min(length, std::strlen(message)));
    }
}

//-----------------------------------------------------------
void set_complex(COMPLEX *array, size_t index, FortranComplex value) {
    array[index] = value;
}

//-----------------------------------------------------------
bool write_values(const char *path, const CU_REAL *values, int count, int stride) {
    FILE *file = std::fopen(path, "w");
    if (!file)
        return false;
    for (int i = 0; i < count; ++i)
        std::fprintf(file, " %.17g\n", static_cast<double>(values[i * stride]));
    return std::fclose(file) == 0;
}

//-----------------------------------------------------------
bool write_mesh_files(int n, int dipole_count, const CU_REAL *mesh_x,
                      const CU_REAL *mesh_y, const CU_REAL *mesh_z,
                      const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs) {
    const int count = n * n * n;
    return write_values("x.mat", mesh_x, n, 1) &&
           write_values("y.mat", mesh_y, n, n) &&
           write_values("z.mat", mesh_z, n, n * n) &&
           write_values("xc.mat", xs, dipole_count, 1) &&
           write_values("yc.mat", ys, dipole_count, 1) &&
           write_values("zc.mat", zs, dipole_count, 1) && count > 0;
}
} // namespace


//-----------------------------------------------------------
extern "C" void objetsphereconcentric_(
    const char *trope, const COMPLEX *epss, const COMPLEX *epsanis,
    const int *numbersphere, const int *numberspheremax, CU_REAL *xg, CU_REAL *yg,
    CU_REAL *zg, CU_REAL *rayons, const COMPLEX *eps0, CU_REAL *xs,
    CU_REAL *ys, CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube, int *tabdip,
    int *tabnbs, const int *nnnr, const int *nmax, int *nbsphere, int *ndipole,
    int *nx, int *ny, int *nz, const char *methode, int *na,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip, char *infostr,
    int *nstop, std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length) {
    ifdda_object_all_concentricsphere_cuda(0, trope, epss, epsanis, numbersphere, numberspheremax, xg, yg, zg, rayons, eps0, xs, ys, zs, k0, aretecube, tabdip, tabnbs, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, methode, na, epsilon, polarisa, niso, sldr, nmat, file_id, group_iddip, infostr, nstop, trope_length, methode_length, infostr_length);
}
