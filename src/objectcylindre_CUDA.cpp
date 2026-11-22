#include "kernel.h"
#include "cuda_interfaces.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

//-----------------------------------------------------------
void set_error(char *info, std::size_t length, int *stop, const char *message) {
    if (stop) *stop = 1;
    if (!info || !length) return;
    std::memset(info, ' ', length);
    std::memcpy(info, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
bool write_axis(const char *path, const CU_REAL *values, int count, int stride) {
    FILE *file = std::fopen(path, "w");
    if (!file) return false;
    for (int i = 0; i < count; ++i)
        std::fprintf(file, " %.17g\n", double(values[std::size_t(i) * stride]));
    return std::fclose(file) == 0;
}

//-----------------------------------------------------------
bool write_geometry_files(int nx, int ny, int nz, int dipoles,
                          const CU_REAL *mesh_x, const CU_REAL *mesh_y,
                          const CU_REAL *mesh_z, const CU_REAL *xs,
                          const CU_REAL *ys, const CU_REAL *zs) {
    return write_axis("x.mat", mesh_x, nx, 1) && write_axis("y.mat", mesh_y, ny, nx) &&
           write_axis("z.mat", mesh_z, nz, nx * ny) &&
           write_axis("xc.mat", xs, dipoles, 1) &&
           write_axis("yc.mat", ys, dipoles, 1) &&
           write_axis("zc.mat", zs, dipoles, 1);
}
} // namespace


//-----------------------------------------------------------
extern "C" void objetcylindre_(
    const char *trope, const COMPLEX *eps, const COMPLEX *epsani,
    const COMPLEX *eps0, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, const int *nnnr,
    const int *nmax, int *nbsphere, int *ndipole, int *nx, int *ny, int *nz,
    const int *nxm, const int *nym, const int *nzm, const char *methode,
    const int *na, COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    CU_REAL *radius, CU_REAL *height, CU_REAL *xg, CU_REAL *yg, CU_REAL *zg,
    CU_REAL *phi, CU_REAL *theta, CU_REAL *psi, CU_REAL *sldr, const int *nmat,
    const int64_t *file_id, const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length) {
    ifdda_object_all_cylinder_cuda(10, trope, eps, epsani, eps0,
                                   xs, ys, zs, k0, aretecube, tabdip, nnnr,
                                   nmax, nbsphere, ndipole, nx, ny, nz,
                                   nxm, nym, nzm, methode, na, epsilon, polarisa,
                                   niso, radius, height, xg, yg, zg,
                                   phi, theta, psi, sldr, nmat, file_id,
                                   group_iddip, infostr, nstop, trope_length,
                                   methode_length, infostr_length);
}
