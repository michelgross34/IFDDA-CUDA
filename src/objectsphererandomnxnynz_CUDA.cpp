#include "kernel.h"
#include "cuda_interfaces.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>

using FortranComplex = COMPLEX;

//-----------------------------------------------------------
extern "C" void objetsphererandomnxnynz_(
    const char *trope, const FortranComplex *eps,
    const FortranComplex *epsani, const FortranComplex *eps0, CU_REAL *xs,
    CU_REAL *ys, CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube,
    int *tabdip, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const int *nxm,
    const int *nym, const int *nzm, const int *nxmp, const int *nymp,
    const int *nzmp, const char *methode, FortranComplex *epsilon,
    FortranComplex *polarisa, const int *niso, CU_REAL *sidex,
    CU_REAL *sidey, CU_REAL *sidez, CU_REAL *xg, CU_REAL *yg, CU_REAL *zg,
    CU_REAL *xsphe, CU_REAL *ysphe, CU_REAL *zsphe, CU_REAL *radius,
    CU_REAL *density, const int *nseed, const int *na, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip,
    char *infostr, int *nstop, std::size_t trope_length,
    std::size_t methode_length, std::size_t infostr_length) {
    if (!nnnr || !nmax || !nxm || !nym || !nzm || !nxmp || !nymp || !nzmp ||
        !aretecube || !sidex || !sidey || !sidez || *aretecube <= CU_REAL(0)) {
        if (nstop)
            *nstop = 1;
        if (infostr && infostr_length) {
            std::memset(infostr, ' ', infostr_length);
            const char message[] = "Invalid random-sphere dimensions";
            std::memcpy(infostr, message,
                        std::min(infostr_length, sizeof(message) - 1));
        }
        return;
    }

    const int nx_interior = *nxm - 2 * *nxmp;
    const int ny_interior = *nym - 2 * *nymp;
    const int nz_interior = *nzm - 2 * *nzmp;
    if (nx_interior <= 0 || ny_interior <= 0 || nz_interior <= 0) {
        if (nstop)
            *nstop = 1;
        if (infostr && infostr_length) {
            std::memset(infostr, ' ', infostr_length);
            const char message[] = "Dimension Problem of the Box : Box too small!";
            std::memcpy(infostr, message,
                        std::min(infostr_length, sizeof(message) - 1));
        }
        return;
    }

    const int subdivisions = std::max({nx_interior, ny_interior, nz_interior});
    const CU_REAL cell_nm = *aretecube;
    int actual_subdivisions = subdivisions;

    // The randomsphere2 Fortran routine defines its usable box from the
    // requested cell size and the unpadded dimensions. Pass an equivalent
    // box to the shared random-sphere CUDA implementation.
    *sidex = cell_nm * CU_REAL(nx_interior);
    *sidey = cell_nm * CU_REAL(ny_interior);
    *sidez = cell_nm * CU_REAL(nz_interior);
    *aretecube = cell_nm;

    // Preserve randomsphere2's selector through the shared CUDA entry.
    ifdda_object_all_randomsphere_cuda(12,
        trope, eps, epsani, eps0, xs, ys, zs, k0, aretecube, tabdip,
        &actual_subdivisions, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym,
        nzm, methode, epsilon, polarisa, niso, sidex, sidey, sidez, xg, yg,
        zg, xsphe, ysphe, zsphe, radius, density, nseed, na, sldr, nmat,
        file_id, group_iddip, infostr, nstop, trope_length, methode_length,
        infostr_length);
    (void)nnnr;
}
