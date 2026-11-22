#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "kernel.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using FortranComplex = std::complex<CU_REAL>;
static_assert(sizeof(FortranComplex) == sizeof(CUCOMPLEX),
              "Fortran and CUDA complex values must have the same layout");

namespace {
//-----------------------------------------------------------
void set_error(char *info, std::size_t length, int *stop,
               const char *message) {
    if (stop) *stop = 1;
    if (!info || length == 0) return;
    std::memset(info, ' ', length);
    std::memcpy(info, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
int polarizability_method_code(const char *method) {
    if (!method)
        return 0;
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

//-----------------------------------------------------------
CUCOMPLEX make_cuda_complex(CU_REAL real, CU_REAL imag) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real, imag);
#else
    return make_cuDoubleComplex(real, imag);
#endif
}

//-----------------------------------------------------------
bool write_values(const char *path, const CU_REAL *values, int count,
                  int stride) {
    FILE *file = std::fopen(path, "w");
    if (!file) return false;
    for (int i = 0; i < count; ++i)
        std::fprintf(file, " %.17g\n",
                     static_cast<double>(values[std::size_t(i) * stride]));
    return std::fclose(file) == 0;
}

//-----------------------------------------------------------
bool write_geometry_files(int n, int dipole_count, const CU_REAL *mesh_x,
                          const CU_REAL *mesh_y, const CU_REAL *mesh_z,
                          const CU_REAL *xs, const CU_REAL *ys,
                          const CU_REAL *zs) {
    return write_values("x.mat", mesh_x, n, 1) &&
           write_values("y.mat", mesh_y, n, n) &&
           write_values("z.mat", mesh_z, n, n * n) &&
           write_values("xc.mat", xs, dipole_count, 1) &&
           write_values("yc.mat", ys, dipole_count, 1) &&
           write_values("zc.mat", zs, dipole_count, 1);
}
} // namespace








//-----------------------------------------------------------
extern "C" void objetsphereinhomo_(
    COMPLEX *eps, const COMPLEX *eps0, CU_REAL *lnrbc,
    CU_REAL *lcos, int *ntypeabc, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, int *tabmulti,
    int *pmlwrk, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const char *methode,
    const int *na, COMPLEX *epsilon, COMPLEX *polarisa,
    const int *niso, CU_REAL *rayon, CU_REAL *lc, CU_REAL *hc,
    const int *ng, COMPLEX *epsb, CU_REAL *sldr, const int *nmat,
    const int64_t *file_id, const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t methode_length, std::size_t infostr_length) {
    ifdda_object_all_inhomosphere_cuda(5, eps, eps0, lnrbc, lcos, ntypeabc, xs, ys, zs, k0, aretecube, tabdip, tabmulti, pmlwrk, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, methode, na, epsilon, polarisa, niso, rayon, lc, hc, ng, epsb, sldr, nmat, file_id, group_iddip, infostr, nstop, methode_length, infostr_length);
}
