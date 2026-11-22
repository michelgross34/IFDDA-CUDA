#include "kernel.h"
#include "cuda_interfaces.h"

#include <algorithm>
#include <cstring>







namespace {

//-----------------------------------------------------------
void set_status(int *nstop, char *infostr, size_t infostr_len,
                const char *message, int status) {
    if (nstop)
        *nstop = status == 0 ? 0 : 1;
    if (!infostr || infostr_len == 0)
        return;
    std::memset(infostr, ' ', infostr_len);
    if (status != 0 && message) {
        const size_t length = std::min(std::strlen(message), infostr_len);
        std::memcpy(infostr, message, length);
    }
}

//-----------------------------------------------------------
bool valid_dimensions(const int *nx, const int *ny, const int *nz,
                      const int *nxm, const int *nym, const int *nzm) {
    return nx && ny && nz && nxm && nym && nzm &&
           *nx > 0 && *ny > 0 && *nz > 0 &&
           *nxm >= *nx && *nym >= *ny && *nzm >= *nz;
}

} // namespace

//-----------------------------------------------------------
extern "C" void preconditionneur_(
    CUCOMPLEX *ff, CUCOMPLEX *xi, CUCOMPLEX *xr, const int *tabdip,
    const CUCOMPLEX *polarisa, const int *niso,
    CUCOMPLEX *chanxxvect, CUCOMPLEX *chanxyvect, CUCOMPLEX *chanxzvect,
    CUCOMPLEX *chanyyvect, CUCOMPLEX *chanyzvect, CUCOMPLEX *chanzzvect,
    const int *nx, const int *ny, const int *nz, const int *nbsphere,
    const int *ndipole, const int *nxm, const int *nym, const int *nzm,
    const CU_REAL *aretecube, const CU_REAL *k0, const int *ntest,
    CUCOMPLEX *sdetnn, int *ipvtnn, CUCOMPLEX *vectxx,
    CUCOMPLEX *vectyy, CUCOMPLEX *vectzz, const long long *plan2b,
    const long long *plan2f, char *infostr, int *nstop,
    size_t infostr_len) {
    (void)xi;
    (void)xr;
    (void)niso;
    (void)ndipole;
    (void)chanxxvect;
    (void)chanxyvect;
    (void)chanxzvect;
    (void)chanyyvect;
    (void)chanyzvect;
    (void)chanzzvect;
    (void)sdetnn;
    (void)ipvtnn;
    (void)vectxx;
    (void)vectyy;
    (void)vectzz;
    (void)plan2b;
    (void)plan2f;

    if (!ntest || !nbsphere || !aretecube || !k0 || !polarisa ||
        !valid_dimensions(nx, ny, nz, nxm, nym, nzm)) {
        set_status(nstop, infostr, infostr_len,
                   "Invalid CUDA Chan preconditioner arguments", 1);
        return;
    }

    int status = 1;
    if (*ntest == 0) {
        status = chan_precon_pol_build_lu_cuda_(polarisa, nbsphere,
                                                aretecube, k0, nx, ny, nz,
                                                nxm, nym, nzm);
    } else if (*ntest == 1 && ff && tabdip) {
        status = chan_preconditioner_apply_cuda_(ff, tabdip, nx, ny, nz,
                                                 nxm, nym, nzm);
    }

    set_status(nstop, infostr, infostr_len,
               "CUDA Chan preconditioner operation failed", status);
}

//-----------------------------------------------------------
extern "C" void preconditionneurinit_(
    CUCOMPLEX *ff, COMPLEX *xi, CUCOMPLEX *xr,
    const CUCOMPLEX *polarisa, const int *niso,
    CUCOMPLEX *chanxxvect, CUCOMPLEX *chanxyvect, CUCOMPLEX *chanxzvect,
    CUCOMPLEX *chanyyvect, CUCOMPLEX *chanyzvect, CUCOMPLEX *chanzzvect,
    const int *nx, const int *ny, const int *nz, const int *nbsphere,
    const int *nxm, const int *nym, const int *nzm,
    const CU_REAL *aretecube, const CU_REAL *k0, const int *ntest,
    CUCOMPLEX *sdetnn, CUCOMPLEX *vectxx, CUCOMPLEX *vectyy,
    CUCOMPLEX *vectzz, const long long *plan2b, const long long *plan2f,
    char *infostr, int *nstop, size_t infostr_len) {
    (void)xr;
    (void)niso;
    (void)chanxxvect;
    (void)chanxyvect;
    (void)chanxzvect;
    (void)chanyyvect;
    (void)chanyzvect;
    (void)chanzzvect;
    (void)sdetnn;
    (void)vectxx;
    (void)vectyy;
    (void)vectzz;
    (void)plan2b;
    (void)plan2f;

    if (!ntest || !nbsphere || !aretecube || !k0 || !polarisa ||
        !valid_dimensions(nx, ny, nz, nxm, nym, nzm)) {
        set_status(nstop, infostr, infostr_len,
                   "Invalid CUDA Chan initial-preconditioner arguments", 1);
        return;
    }

    int status = 1;
    if (*ntest == 0) {
        status = chan_precon_pol_build_cuda_(polarisa, nbsphere, aretecube,
                                             k0, nx, ny, nz, nxm, nym, nzm);
    } else if (*ntest == 1 && ff && xi) {
        status = chan_preconditioner_apply_initial_dense_device_cuda_(
            ff, xi, nbsphere, nx, ny, nz, nxm, nym, nzm);
    }

    set_status(nstop, infostr, infostr_len,
               "CUDA dense Chan initial operation failed", status);
}

//-----------------------------------------------------------
extern "C" void preconditionneurinit2_(
    CUCOMPLEX *ff, COMPLEX *xi, CUCOMPLEX *xr, const int *tabdip,
    const CUCOMPLEX *polarisa, const int *niso,
    CUCOMPLEX *chanxxvect, CUCOMPLEX *chanxyvect, CUCOMPLEX *chanxzvect,
    CUCOMPLEX *chanyyvect, CUCOMPLEX *chanyzvect, CUCOMPLEX *chanzzvect,
    const int *nx, const int *ny, const int *nz, const int *nbsphere,
    const int *ndipole, const int *nxm, const int *nym, const int *nzm,
    const CU_REAL *aretecube, const CU_REAL *k0, const int *ntest,
    CUCOMPLEX *sdetnn, CUCOMPLEX *vectxx, CUCOMPLEX *vectyy,
    CUCOMPLEX *vectzz, const long long *plan2b, const long long *plan2f,
    char *infostr, int *nstop, size_t infostr_len) {
    (void)xr;
    (void)niso;
    (void)ndipole;
    (void)chanxxvect;
    (void)chanxyvect;
    (void)chanxzvect;
    (void)chanyyvect;
    (void)chanyzvect;
    (void)chanzzvect;
    (void)sdetnn;
    (void)vectxx;
    (void)vectyy;
    (void)vectzz;
    (void)plan2b;
    (void)plan2f;

    if (!ntest || !nbsphere || !aretecube || !k0 || !polarisa ||
        !valid_dimensions(nx, ny, nz, nxm, nym, nzm)) {
        set_status(nstop, infostr, infostr_len,
                   "Invalid CUDA Chan initial-preconditioner arguments", 1);
        return;
    }

    int status = 1;
    if (*ntest == 0) {
        status = chan_precon_pol_build_cuda_(polarisa, nbsphere, aretecube,
                                             k0, nx, ny, nz, nxm, nym, nzm);
    } else if (*ntest == 1 && ff && xi && tabdip) {
        status = chan_preconditioner_apply_initial_device_cuda_(
            ff, xi, nbsphere, tabdip, nx, ny, nz, nxm, nym, nzm);
    }

    set_status(nstop, infostr, infostr_len,
               "CUDA mapped Chan initial operation failed", status);
}
