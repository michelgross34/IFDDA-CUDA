#include "cuda_data.h"
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

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
struct PolarizabilityComplex {
    CU_REAL real;
    CU_REAL imag;
};

//-----------------------------------------------------------
void set_error(char *info, std::size_t length, int *stop,
               const char *message) {
    if (stop)
        *stop = 1;
    if (!info || length == 0)
        return;
    std::memset(info, ' ', length);
    std::memcpy(info, message, std::min(length, std::strlen(message)));
}

//-----------------------------------------------------------
std::size_t tensor_index(int dipole, int nmax, int niso,
                         int row, int column) {
    return std::size_t(dipole) + std::size_t(nmax) *
           std::size_t(row + niso * column);
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
bool write_values(const char *path, const CU_REAL *values,
                  int count, int stride) {
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
    return write_values("x.mat", mesh_x, nx, 1) &&
           write_values("y.mat", mesh_y, ny, nx) &&
           write_values("z.mat", mesh_z, nz, nx * ny) &&
           write_values("xc.mat", xs, dipole_count, 1) &&
           write_values("yc.mat", ys, dipole_count, 1) &&
           write_values("zc.mat", zs, dipole_count, 1);
}

//-----------------------------------------------------------
bool write_shifted_geometry_files(int nx, int ny, int nz, int dipole_count,
                                  const CU_REAL *xs, const CU_REAL *ys,
                                  const CU_REAL *zs, CU_REAL x_offset,
                                  CU_REAL y_offset, CU_REAL z_offset) {
    FILE *x_file = std::fopen("x.mat", "w");
    FILE *y_file = std::fopen("y.mat", "w");
    FILE *z_file = std::fopen("z.mat", "w");
    if (!x_file || !y_file || !z_file) {
        if (x_file) std::fclose(x_file);
        if (y_file) std::fclose(y_file);
        if (z_file) std::fclose(z_file);
        return false;
    }
    bool ok = true;
    for (int i = 0; i < nx; ++i)
        ok = ok && std::fprintf(x_file, " %.17g\n",
            static_cast<double>(xs[i] - x_offset)) >= 0;
    for (int i = 0; i < ny; ++i)
        ok = ok && std::fprintf(y_file, " %.17g\n",
            static_cast<double>(ys[std::size_t(i) * nx] - y_offset)) >= 0;
    for (int i = 0; i < nz; ++i)
        ok = ok && std::fprintf(z_file, " %.17g\n",
            static_cast<double>(zs[std::size_t(i) * nx * ny] - z_offset)) >= 0;
    ok = (std::fclose(x_file) == 0) && ok;
    ok = (std::fclose(y_file) == 0) && ok;
    ok = (std::fclose(z_file) == 0) && ok;
    return ok && write_values("xc.mat", xs, dipole_count, 1) &&
           write_values("yc.mat", ys, dipole_count, 1) &&
           write_values("zc.mat", zs, dipole_count, 1);
}

//-----------------------------------------------------------
void set_complex(FortranComplex *array, std::size_t index,
                 PolarizabilityComplex value) {
    array[index] = FortranComplex(value.real, value.imag);
}
} // namespace






//-----------------------------------------------------------
extern "C" void objetparainhomo_(
    COMPLEX *eps, COMPLEX *eps0, CU_REAL *lnrbc,
    CU_REAL *lcos, int *ntypeabc, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, int *tabmulti,
    int *pmlwrk, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const int *nxm,
    const int *nym, const int *nzm, const char *methode,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez, CU_REAL *xg,
    CU_REAL *yg, CU_REAL *zg, CU_REAL *lc, const CU_REAL *hc,
    const int *ng, COMPLEX *epsb, const int *na, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip,
    char *infostr, int *nstop, std::size_t methode_length,
    std::size_t infostr_length) {
    ifdda_object_all_inhomocuboid1_cuda(6, eps, eps0, lnrbc, lcos, ntypeabc, xs, ys, zs, k0, aretecube, tabdip, tabmulti, pmlwrk, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, methode, epsilon, polarisa, niso, sidex, sidey, sidez, xg, yg, zg, lc, hc, ng, epsb, na, sldr, nmat, file_id, group_iddip, infostr, nstop, methode_length, infostr_length);
}
