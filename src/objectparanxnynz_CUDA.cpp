#include "kernel.h"
#include "cuda_interfaces.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>

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
bool write_values(const char *path, const CU_REAL *values, int count, int stride) {
    FILE *file = std::fopen(path, "w");
    if (!file)
        return false;
    for (int i = 0; i < count; ++i)
        std::fprintf(file, " %.17g\n", static_cast<double>(values[i * stride]));
    const bool ok = std::fclose(file) == 0;
    return ok;
}

//-----------------------------------------------------------
bool write_mesh_files(int nx, int ny, int nz, const CU_REAL *xs,
                      const CU_REAL *ys, const CU_REAL *zs) {
    const int count = nx * ny * nz;
    return write_values("x.mat", xs, nx, 1) &&
           write_values("y.mat", ys, ny, nx) &&
           write_values("z.mat", zs, nz, nx * ny) &&
           write_values("xc.mat", xs, count, 1) &&
           write_values("yc.mat", ys, count, 1) &&
           write_values("zc.mat", zs, count, 1);
}
} // namespace


//-----------------------------------------------------------
extern "C" void objetparanxnynz_(
    const char *trope, COMPLEX *eps, COMPLEX *epsani,
    COMPLEX *eps0, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, const int *nnnr,
    const int *nmax, int *nbsphere, int *ndipole, int *nx, int *ny, int *nz,
    const int *nxm, const int *nym, const int *nzm, const int *nxmp,
    const int *nymp, const int *nzmp, const char *methode, COMPLEX *epsilon,
    COMPLEX *polarisa, const int *niso, CU_REAL *sidex, CU_REAL *sidey,
    CU_REAL *sidez, CU_REAL *xg, CU_REAL *yg, CU_REAL *zg, const int *na,
    CU_REAL *sldr, const int *nmat, const int64_t *file_id, const int64_t *group_iddip,
    char *infostr, int *nstop, std::size_t trope_length,
    std::size_t methode_length, std::size_t infostr_length) {
    ifdda_object_all_cuboid2_cuda(4, trope, eps, epsani, eps0, xs, ys, zs,
                                  k0, aretecube, tabdip, nnnr, nmax,
                                  nbsphere, ndipole, nx, ny, nz,
                                  nxm, nym, nzm, nxmp, nymp, nzmp,
                                  methode, epsilon, polarisa, niso,
                                  sidex, sidey, sidez, xg, yg, zg, na,
                                  sldr, nmat, file_id, group_iddip,
                                  infostr, nstop, trope_length,
                                  methode_length, infostr_length);
}
