#include "cdmlib_cpp_abi.hpp"
#define IFDDA_CUDA_CALLERS_MICROSBFNEG_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_MICROSBFNEG_CUDA

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>



//-----------------------------------------------------------
extern "C" void microsbfneg_(
    COMPLEX *FFTTENSORxx, COMPLEX *FFTTENSORxy, COMPLEX *FFTTENSORxz,
    COMPLEX *FFTTENSORyy, COMPLEX *FFTTENSORyz, COMPLEX *FFTTENSORzz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    int *ntotalm, int *ntotal, int *ldabi, int *nlar, int *nmax,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz,
    int *nx2, int *ny2, int *nxy2, int *nz2, int *ndipole,
    int *nbsphere, int *nbsphere3, int *nproche, int *nrig,
    int *nfft2d, int *tabfft2, int *tabdip, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *wrk, COMPLEX *FF, COMPLEX *FF0, COMPLEX *FFloc,
    COMPLEX *polarisa, int *niso, COMPLEX *epsilon,
    char *methodeit, char *trope, REAL *tolinit, REAL *tol1,
    int *nloop, int *ncompte, int *nlim,
    REAL *xs, REAL *ys, REAL *zs, REAL *aretecube,
    REAL *numaper, REAL *numaperinc, REAL *numaperinc2,
    int *npolainc_in, int *nquicklens, COMPLEX *eps0, REAL *k0,
    REAL *P0, REAL *irra, REAL *w0, REAL *gross, REAL *zlens,
    COMPLEX *Eimagex, COMPLEX *Eimagey, COMPLEX *Eimagez,
    COMPLEX *Efourierx, COMPLEX *Efouriery, COMPLEX *Efourierz,
    COMPLEX *Ediffkzpos, COMPLEX *Ediffkzneg, REAL *kxy, REAL *xy,
    int *nside, COMPLEX *masque, int *nmasque, int *ideltam,
    int *imaxk0, int *nbitermic, REAL *deltak, REAL *deltakx,
    int *ntypemic, int64_t *planXf, int64_t *planXb,
    int64_t *planYf, int64_t *planYb, int64_t *planZf, int64_t *planZb,
    int64_t *planf, int64_t *planb, int64_t *plan2f, int64_t *plan2b,
    int *nmat, int *file_id_in, int *group_idmic_in,
    int *nstop, char *infostr,
    std::size_t methodeit_len, std::size_t trope_len,
    std::size_t infostr_len)
{
    if (nfft2d == nullptr || *nfft2d <= 0 || nstop == nullptr ||
        npolainc_in == nullptr) {
        if (nstop != nullptr) {
            *nstop = 1;
        }
        return;
    }

    const std::size_t nfft = static_cast<std::size_t>(*nfft2d);
    if (nfft > std::numeric_limits<std::size_t>::max() / nfft ||
        nfft * nfft > std::numeric_limits<std::size_t>::max() / 6u) {
        *nstop = 1;
        return;
    }

    const std::size_t image_count = nfft * nfft;
    thread_local std::vector<COMPLEX> incident_scratch;
    try {
        incident_scratch.resize(6u * image_count, COMPLEX{});
    } catch (...) {
        *nstop = 1;
        if (infostr != nullptr) {
            std::fill_n(infostr, infostr_len, ' ');
            constexpr char message[] = "microsbfneg scratch allocation failed";
            const std::size_t message_size =
                std::min(infostr_len, sizeof(message) - 1u);
            std::copy_n(message, message_size, infostr);
        }
        return;
    }
    std::fill(incident_scratch.begin(), incident_scratch.end(), COMPLEX{});

    COMPLEX *const image_incident_x = incident_scratch.data();
    COMPLEX *const image_incident_y = image_incident_x + image_count;
    COMPLEX *const image_incident_z = image_incident_y + image_count;
    COMPLEX *const spectrum_incident_x = image_incident_z + image_count;
    COMPLEX *const spectrum_incident_y = spectrum_incident_x + image_count;
    COMPLEX *const spectrum_incident_z = spectrum_incident_y + image_count;
    int negative_nside = -1;
    int npolainc = *npolainc_in;

    microsbf_(
        FFTTENSORxx, FFTTENSORxy, FFTTENSORxz,
        FFTTENSORyy, FFTTENSORyz, FFTTENSORzz,
        vectx, vecty, vectz,
        ntotalm, ntotal, ldabi, nlar, nmax,
        nxm, nym, nzm, nx, ny, nz,
        nx2, ny2, nxy2, nz2, ndipole,
        nbsphere, nbsphere3, nproche, nrig,
        nfft2d, tabfft2, tabdip, XI, XR,
        wrk, FF, FF0, FFloc, polarisa, niso, epsilon,
        methodeit, trope, tolinit, tol1,
        nloop, ncompte, nlim, xs, ys, zs,
        aretecube, numaper, numaperinc, numaperinc2,
        npolainc_in, nquicklens, eps0, k0, P0, irra,
        w0, gross, zlens, Eimagex, Eimagey, Eimagez,
        Efourierx, Efouriery, Efourierz,
        image_incident_x, image_incident_y, image_incident_z,
        spectrum_incident_x, spectrum_incident_y, spectrum_incident_z,
        Ediffkzpos, Ediffkzneg, kxy, xy, &negative_nside,
        masque, nmasque, ideltam, imaxk0, nbitermic,
        deltak, deltakx, ntypemic,
        planXf, planXb, planYf, planYb, planZf, planZb,
        planf, planb, plan2f, plan2b, nmat, file_id_in,
        group_idmic_in, nstop, infostr,
        methodeit_len, trope_len, infostr_len);
}
