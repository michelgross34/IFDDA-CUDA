#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using FortranComplex = std::complex<CU_REAL>;
static_assert(sizeof(FortranComplex) == sizeof(CUCOMPLEX),
              "Fortran and CUDA complex values must have the same layout");

namespace {
//-----------------------------------------------------------
void set_error(char *destination, std::size_t length, int *stop,
               const char *message) {
    if (stop)
        *stop = 1;
    if (!destination || length == 0)
        return;
    std::memset(destination, ' ', length);
    if (message)
        std::memcpy(destination, message,
                    std::min(length, std::strlen(message)));
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
    if (!file)
        return false;
    for (int i = 0; i < count; ++i)
        std::fprintf(file, " %.17g\n",
                     static_cast<double>(values[std::size_t(i) * stride]));
    return std::fclose(file) == 0;
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
extern "C" void objetparainhomonxnynz_(
    COMPLEX *eps, COMPLEX *eps0, CU_REAL *xs, CU_REAL *ys,
    CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube, int *tabdip,
    const int *nnnr, const int *nmax, int *nbsphere, int *ndipole,
    int *nx, int *ny, int *nz, const int *nxm, const int *nym,
    const int *nzm, const int *nxmp, const int *nymp, const int *nzmp,
    const char *methode, COMPLEX *epsilon, COMPLEX *polarisa,
    const int *niso, CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez,
    CU_REAL *xg, CU_REAL *yg, CU_REAL *zg, CU_REAL *lc, const CU_REAL *hc,
    const int *ng, COMPLEX *epsb, const int *na, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip,
    char *infostr, int *nstop,
    std::size_t methode_length, std::size_t infostr_length) {
    ifdda_object_all_inhomocuboid2_cuda(7, eps, eps0, xs, ys, zs, k0, aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, nxmp, nymp, nzmp, methode, epsilon, polarisa, niso, sidex, sidey, sidez, xg, yg, zg, lc, hc, ng, epsb, na, sldr, nmat, file_id, group_iddip, infostr, nstop, methode_length, infostr_length);
}
