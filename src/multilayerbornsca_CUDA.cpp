#include "cuda_data.h"
#define IFDDA_CUDA_CDMLIB_HOST_ABI
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CDMLIB_HOST_ABI
#include "cuda_interfaces.h"
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

namespace {

constexpr CU_REAL multilayer_pi = CU_REAL(3.1415926535897932384626433832795L);

CUCOMPLEX make_cuda_complex(CU_REAL real, CU_REAL imag) {
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    return make_cuFloatComplex(real, imag);
#else
    return make_cuDoubleComplex(real, imag);
#endif
}

bool beam_is(const char beam[64], const char *name, size_t length) {
    return beam && std::memcmp(beam, name, length) == 0;
}

void set_error(int *nstop, char *infostr, size_t infostr_length, const char *message) {
    if (nstop)
        *nstop = 1;
    if (infostr && infostr_length) {
        std::memset(infostr, ' ', infostr_length);
        if (message)
            std::memcpy(infostr, message, std::min(infostr_length, std::strlen(message)));
    }
}

} // namespace











//-----------------------------------------------------------
static void run_multilayer_scalar(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const CU_REAL *ss,
    const CU_REAL *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length, int reflection, int projected) {
    (void)epsilon;
    (void)nxm;
    (void)nym;
    (void)nzm;
    (void)FF;
    (void)plan2f;
    (void)plan2b;
    if (!nstop)
        return;
    *nstop = 0;
    if (infostr && infostr_length)
        std::memset(infostr, ' ', infostr_length);

    if (!xs || !ys || !zs || !aretecube || !k0 || !w0 || !E0 || !ss || !pp || !theta || !phi ||
        !xgaus || !ygaus || !zgaus || !beam || !polarisa || !niso || !ndipole || !nx || !ny ||
        !nz || !nmax || !nfft2d || !imagex || !imagey || !imagez || !FF0 || !FFloc ||
        !infostr || *ndipole <= 0 || *nmax < *ndipole || *nfft2d <= *nx || *nfft2d <= *ny ||
        *nx <= 0 || *ny <= 0 || *nz <= 0 || *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0)) {
        set_error(nstop, infostr, infostr_length, "Invalid multilayer Born input");
        return;
    }

    // Route supported plane, paraxial, and rigorous Gaussian families through
    // the GPU first-plane implementation.
    int beam_type = -1;
    if (beam_is(beam, "pwavelinear", 11))
        beam_type = 0;
    else if (beam_is(beam, "pwavecircular", 13))
        beam_type = 1;
    else if (beam_is(beam, "gparawavelinear", 15))
        beam_type = 2;
    else if (beam_is(beam, "gparawavecircular", 17))
        beam_type = 3;
    else if (beam_is(beam, "gwavelinear", 11) || beam_is(beam, "gfftwavelinear", 14))
        beam_type = 4;
    else if (beam_is(beam, "gwavecircular", 13) || beam_is(beam, "gfftwavecircular", 16))
        beam_type = 5;
    // The Fortran scalar multilayer routine uses only polarisa(:,1,1).
    // That component is the first contiguous nmax block for both niso=1 and
    // niso=3, so the supported CUDA path can handle either layout directly.
    if (beam_type < 0) {
        set_error(nstop, infostr, infostr_length,
                  "Beam not implemented for this approximation");
        return;
    }

    const CU_REAL two_pi = CU_REAL(2) * multilayer_pi;
    const CU_REAL deltakx = two_pi / (CU_REAL(*nfft2d) * *aretecube);
    const CU_REAL deltaky = deltakx;
    if (deltakx >= *k0 / CU_REAL(2) || deltaky >= *k0 / CU_REAL(2)) {
        set_error(nstop, infostr, infostr_length, "nfft2d too small");
        return;
    }

    CU_REAL xmin = xs[0];
    CU_REAL xmax = xs[0];
    CU_REAL ymin = ys[0];
    CU_REAL ymax = ys[0];
    for (int i = 1; i < *ndipole; ++i) {
        xmin = std::min(xmin, xs[i]);
        xmax = std::max(xmax, xs[i]);
        ymin = std::min(ymin, ys[i]);
        ymax = std::max(ymax, ys[i]);
    }
    (void)xmax;
    (void)ymax;

    const int kkm = (*nfft2d - *nx) / 2;
    const int jjm = (*nfft2d - *ny) / 2;
    const COMPLEX eps0(CU_REAL(1), CU_REAL(0));
    const COMPLEX eps0_fortran(CU_REAL(1), CU_REAL(0));
    const COMPLEX E0_fortran(E0->real(), E0->imag());
    COMPLEX ux_fortran{}, uy_fortran{}, uz_fortran{};
    CU_REAL x0 = *xgaus;
    CU_REAL y0 = *ygaus;
    CU_REAL z0 = *zgaus;
    CU_REAL xx = x0;
    CU_REAL yy = y0;
    CU_REAL zz = z0;
    CU_REAL k0_value = *k0;
    CU_REAL theta_value = *theta;
    CU_REAL phi_value = *phi;
    CU_REAL ss_value = *ss;
    CU_REAL pp_value = *pp;
    int beam_status = 0;
    char beam_message[64];
    std::memset(beam_message, ' ', sizeof(beam_message));
    if (beam_type == 0) {
        ondeplane_(&xx, &yy, &zz, &k0_value, &eps0_fortran, &E0_fortran,
                   &ss_value, &pp_value, &theta_value, &phi_value,
                   &ux_fortran, &uy_fortran, &uz_fortran, &beam_status, beam_message,
                   sizeof(beam_message));
    } else if (beam_type == 1) {
        ondecirce_(&xx, &yy, &zz, k0, &E0_fortran, &ss_value, theta, phi, &ux_fortran,
                   &uy_fortran, &uz_fortran);
    } else if (beam_type == 2) {
        gaussianparalinear_(&xx, &yy, &zz, xgaus, ygaus, zgaus, theta, phi, w0, k0,
                            &ss_value, &pp_value, &eps0_fortran, &E0_fortran, &ux_fortran,
                            &uy_fortran, &uz_fortran, &beam_status, beam_message,
                            sizeof(beam_message));
    } else if (beam_type == 3) {
        gaussianparacirc_(&xx, &yy, &zz, xgaus, ygaus, zgaus, theta, phi, w0, k0,
                          &ss_value, &eps0_fortran, &E0_fortran, &ux_fortran, &uy_fortran,
                          &uz_fortran, &beam_status, beam_message, sizeof(beam_message));
    } else {
        CU_REAL tolerance = CU_REAL(1e-3);
        int nloin = 0;
        if (beam_type == 4)
            gaussianchamp_(&xx, &yy, &zz, xgaus, ygaus, zgaus, theta, phi, w0, k0,
                           &ss_value, &pp_value, &E0_fortran, &ux_fortran, &uy_fortran,
                           &uz_fortran, &tolerance, &nloin, &beam_status, beam_message,
                           sizeof(beam_message));
        else
            gaussianchampcirc_(&xx, &yy, &zz, xgaus, ygaus, zgaus, theta, phi, w0, k0,
                               &ss_value, &E0_fortran, &ux_fortran, &uy_fortran, &uz_fortran,
                               &tolerance, &nloin, &beam_status, beam_message,
                               sizeof(beam_message));
    }
    if (beam_status != 0) {
        set_error(nstop, infostr, infostr_length, "Incident field initialization failed");
        return;
    }

    using Complex = std::complex<CU_REAL>;
    Complex ux(ux_fortran.real(), ux_fortran.imag());
    Complex uy(uy_fortran.real(), uy_fortran.imag());
    Complex uz(uz_fortran.real(), uz_fortran.imag());
    const CU_REAL norm = std::sqrt(std::norm(ux) + std::norm(uy) + std::norm(uz));
    if (norm == CU_REAL(0) || !std::isfinite(norm)) {
        set_error(nstop, infostr, infostr_length, "Invalid multilayer polarization");
        return;
    }
    ux /= norm;
    uy /= norm;
    uz /= norm;
    const COMPLEX ux_host(ux.real(), ux.imag());
    const COMPLEX uy_host(uy.real(), uy.imag());
    const COMPLEX uz_host(uz.real(), uz.imag());

    // imagey is a caller-owned, stable key for the persistent device Green
    // buffer; the Green tensor is generated directly on the GPU.
    const int green_status = projected
                                 ? ifdda_generate_multilayer_projected_green_cuda_(
                                       imagey, nfft2d, aretecube, k0, &ux_host, &uy_host,
                                       &uz_host)
                                 : ifdda_generate_multilayer_scalar_green_cuda_(
                                       imagey, nfft2d, aretecube, k0);
    if (green_status != 0) {
        set_error(nstop, infostr, infostr_length, "CUDA scalar Green generation failed");
        return;
    }

    const CU_REAL first_z = zs[0];
    const int conjugate_projection = projected ? 1 : 0;
    const CU_REAL rigorous_tolerance = CU_REAL(1e-3);
    const int circular = beam_type == 5 ? 1 : 0;
    const int initialize_status = beam_type >= 4
        ? ifdda_multilayer_scalar_initialize_rigorous_plane_cuda_(
              imagex, FFloc, FF0, xs, ys, zs, ndipole, nfft2d, nmax, &kkm, &jjm,
              &xmin, &ymin, &first_z, aretecube, k0, E0, &ss_value, &pp_value,
              theta, phi, xgaus, ygaus, zgaus, w0, &rigorous_tolerance, &circular,
              &conjugate_projection, &ux_host, &uy_host, &uz_host)
        : ifdda_multilayer_scalar_initialize_plane_cuda_(
              imagex, FFloc, FF0, xs, ys, zs, ndipole, nfft2d, nmax, &kkm, &jjm,
              &xmin, &ymin, &first_z, aretecube, k0, E0, &eps0, w0, &ss_value,
              &pp_value, theta, phi, &beam_type, &conjugate_projection,
              &ux_host, &uy_host, &uz_host);
    if (initialize_status != 0) {
        set_error(nstop, infostr, infostr_length, "CUDA scalar first-plane initialization failed");
        return;
    }

    std::vector<int> plane_map(size_t(*nfft2d) * size_t(*nfft2d), 0);
    const int layer_status = ifdda_multilayerbornsca_layers_cuda_(
        imagex, FFloc, polarisa, plane_map.data(), imagey, nfft2d, nx, ny, nz, nmax,
        niso, &reflection, &deltakx, &deltaky, aretecube, k0, &ux_host, &uy_host, &uz_host);
    if (layer_status != 0) {
        set_error(nstop, infostr, infostr_length, "CUDA multilayer scalar propagation failed");
        return;
    }

    *nstop = 0;
    if (infostr && infostr_length)
        std::memset(infostr, ' ', infostr_length);
}

//-----------------------------------------------------------
static void run_multilayer_vector(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const CU_REAL *ss,
    const CU_REAL *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length, int reflection) {
    (void)epsilon;
    (void)nxm;
    (void)nym;
    (void)nzm;
    (void)FF;
    (void)plan2f;
    (void)plan2b;
    if (!nstop)
        return;
    *nstop = 0;
    if (infostr && infostr_length)
        std::memset(infostr, ' ', infostr_length);

    if (!xs || !ys || !zs || !aretecube || !k0 || !w0 || !E0 || !ss || !pp || !theta || !phi ||
        !xgaus || !ygaus || !zgaus || !beam || !polarisa || !niso || !ndipole || !nx || !ny ||
        !nz || !nmax || !nfft2d || !imagex || !imagey || !imagez || !FF0 || !FFloc || !infostr ||
        *ndipole <= 0 || *nmax < *ndipole || *nfft2d <= *nx || *nfft2d <= *ny || *nx <= 0 ||
        *ny <= 0 || *nz <= 0 || *aretecube <= CU_REAL(0) || *k0 <= CU_REAL(0)) {
        set_error(nstop, infostr, infostr_length, "Invalid multilayer vector input");
        return;
    }

    int beam_type = -1;
    if (beam_is(beam, "pwavelinear", 11))
        beam_type = 0;
    else if (beam_is(beam, "pwavecircular", 13))
        beam_type = 1;
    else if (beam_is(beam, "gparawavelinear", 15))
        beam_type = 2;
    else if (beam_is(beam, "gparawavecircular", 17))
        beam_type = 3;
    else if (beam_is(beam, "gwavelinear", 11) || beam_is(beam, "gfftwavelinear", 14))
        beam_type = 4;
    else if (beam_is(beam, "gwavecircular", 13) || beam_is(beam, "gfftwavecircular", 16))
        beam_type = 5;
    // The Fortran vector multilayer routine also applies polarisa(:,1,1) to
    // each field component; the first nmax values are contiguous in both
    // isotropic and 3x3 polarizability storage.
    if (beam_type < 0) {
        set_error(nstop, infostr, infostr_length,
                  "Beam not implemented for this approximation");
        return;
    }

    const CU_REAL two_pi = CU_REAL(2) * multilayer_pi;
    const CU_REAL deltakx = two_pi / (CU_REAL(*nfft2d) * *aretecube);
    const CU_REAL deltaky = deltakx;
    if (deltakx >= *k0 / CU_REAL(2) || deltaky >= *k0 / CU_REAL(2)) {
        set_error(nstop, infostr, infostr_length, "nfft2d too small");
        return;
    }

    CU_REAL xmin = xs[0];
    CU_REAL ymin = ys[0];
    for (int i = 1; i < *ndipole; ++i) {
        xmin = std::min(xmin, xs[i]);
        ymin = std::min(ymin, ys[i]);
    }
    const int kkm = (*nfft2d - *nx) / 2;
    const int jjm = (*nfft2d - *ny) / 2;
    const COMPLEX eps0(CU_REAL(1), CU_REAL(0));
    const COMPLEX *green_keys[6] = {imagex, imagey, imagez, FF0, FFloc, FF};
    if (ifdda_generate_multilayer_dyadic_green_cuda_(
            const_cast<COMPLEX *>(green_keys[0]), const_cast<COMPLEX *>(green_keys[1]),
            const_cast<COMPLEX *>(green_keys[2]), const_cast<COMPLEX *>(green_keys[3]),
            const_cast<COMPLEX *>(green_keys[4]), const_cast<COMPLEX *>(green_keys[5]),
            nfft2d, aretecube, k0) != 0) {
        set_error(nstop, infostr, infostr_length, "CUDA dyadic Green generation failed");
        return;
    }

    const CU_REAL rigorous_tolerance = CU_REAL(1e-3);
    const int circular = beam_type == 5 ? 1 : 0;
    const int initialize_status = beam_type >= 4
        ? ifdda_multilayer_vector_initialize_rigorous_plane_cuda_(
              imagex, imagey, imagez, FFloc, FF0, xs, ys, zs, ndipole, nfft2d,
              nmax, &kkm, &jjm, &xmin, &ymin, &zs[0], aretecube, k0, E0, ss, pp,
              theta, phi, xgaus, ygaus, zgaus, w0, &rigorous_tolerance, &circular)
        : ifdda_multilayer_vector_initialize_plane_cuda_(
              imagex, imagey, imagez, FFloc, FF0, xs, ys, zs, ndipole, nfft2d,
              nmax, &kkm, &jjm, &xmin, &ymin, &zs[0], aretecube, k0, E0, &eps0,
              w0, ss, pp, theta, phi, &beam_type);
    if (initialize_status != 0) {
        set_error(nstop, infostr, infostr_length,
                  "CUDA vector first-plane initialization failed");
        return;
    }

    std::vector<int> plane_map(size_t(*nfft2d) * size_t(*nfft2d), 0);
    const int layer_status = ifdda_multilayerbornvect_layers_cuda_(
        imagex, imagey, imagez, FFloc, polarisa, plane_map.data(),
        const_cast<COMPLEX *>(green_keys[0]), const_cast<COMPLEX *>(green_keys[1]),
        const_cast<COMPLEX *>(green_keys[2]), const_cast<COMPLEX *>(green_keys[3]),
        const_cast<COMPLEX *>(green_keys[4]), const_cast<COMPLEX *>(green_keys[5]), nfft2d,
        nx, ny, nz, nmax, niso, &reflection, &deltakx, &deltaky, aretecube, k0);
    if (layer_status != 0) {
        set_error(nstop, infostr, infostr_length,
                  "CUDA multilayer vector propagation failed");
        return;
    }
    *nstop = 0;
    if (infostr && infostr_length)
        std::memset(infostr, ' ', infostr_length);
}

//-----------------------------------------------------------
extern "C" void multilayerbornsca_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length) {
    const CU_REAL ss_value = static_cast<CU_REAL>(ss->real());
    const CU_REAL pp_value = static_cast<CU_REAL>(pp->real());
    run_multilayer_scalar(xs, ys, zs, aretecube, k0, w0, E0, &ss_value, &pp_value, theta, phi, xgaus, ygaus,
                          zgaus, beam, epsilon, polarisa, niso, ndipole, nx, ny, nz, nxm, nym,
                          nzm, nmax, nfft2d, imagex, imagey, imagez, FF0, FFloc, FF, plan2f,
                          plan2b, nstop, infostr, beam_length, infostr_length, 0, 0);
}

//-----------------------------------------------------------
extern "C" void multilayerbornscaref_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length) {
    const CU_REAL ss_value = static_cast<CU_REAL>(ss->real());
    const CU_REAL pp_value = static_cast<CU_REAL>(pp->real());
    run_multilayer_scalar(xs, ys, zs, aretecube, k0, w0, E0, &ss_value, &pp_value, theta, phi, xgaus, ygaus,
                          zgaus, beam, epsilon, polarisa, niso, ndipole, nx, ny, nz, nxm, nym,
                          nzm, nmax, nfft2d, imagex, imagey, imagez, FF0, FFloc, FF, plan2f,
                          plan2b, nstop, infostr, beam_length, infostr_length, 1, 0);
}

//-----------------------------------------------------------
extern "C" void multilayerbornscainc_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length) {
    const CU_REAL ss_value = static_cast<CU_REAL>(ss->real());
    const CU_REAL pp_value = static_cast<CU_REAL>(pp->real());
    run_multilayer_scalar(xs, ys, zs, aretecube, k0, w0, E0, &ss_value, &pp_value, theta, phi, xgaus, ygaus,
                          zgaus, beam, epsilon, polarisa, niso, ndipole, nx, ny, nz, nxm, nym,
                          nzm, nmax, nfft2d, imagex, imagey, imagez, FF0, FFloc, FF, plan2f,
                          plan2b, nstop, infostr, beam_length, infostr_length, 0, 1);
}

//-----------------------------------------------------------
extern "C" void multilayerbornscaincref_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length) {
    const CU_REAL ss_value = static_cast<CU_REAL>(ss->real());
    const CU_REAL pp_value = static_cast<CU_REAL>(pp->real());
    run_multilayer_scalar(xs, ys, zs, aretecube, k0, w0, E0, &ss_value, &pp_value, theta, phi, xgaus, ygaus,
                          zgaus, beam, epsilon, polarisa, niso, ndipole, nx, ny, nz, nxm, nym,
                          nzm, nmax, nfft2d, imagex, imagey, imagez, FF0, FFloc, FF, plan2f,
                          plan2b, nstop, infostr, beam_length, infostr_length, 1, 1);
}

//-----------------------------------------------------------
extern "C" void multilayerbornvect_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length) {
    const CU_REAL ss_value = static_cast<CU_REAL>(ss->real());
    const CU_REAL pp_value = static_cast<CU_REAL>(pp->real());
    run_multilayer_vector(xs, ys, zs, aretecube, k0, w0, E0, &ss_value, &pp_value, theta, phi, xgaus, ygaus,
                          zgaus, beam, epsilon, polarisa, niso, ndipole, nx, ny, nz, nxm, nym,
                          nzm, nmax, nfft2d, imagex, imagey, imagez, FF0, FFloc, FF, plan2f,
                          plan2b, nstop, infostr, beam_length, infostr_length, 0);
}

//-----------------------------------------------------------
extern "C" void multilayerbornvectref_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length) {
    const CU_REAL ss_value = static_cast<CU_REAL>(ss->real());
    const CU_REAL pp_value = static_cast<CU_REAL>(pp->real());
    run_multilayer_vector(xs, ys, zs, aretecube, k0, w0, E0, &ss_value, &pp_value, theta, phi, xgaus, ygaus,
                          zgaus, beam, epsilon, polarisa, niso, ndipole, nx, ny, nz, nxm, nym,
                          nzm, nmax, nfft2d, imagex, imagey, imagez, FF0, FFloc, FF, plan2f,
                          plan2b, nstop, infostr, beam_length, infostr_length, 1);
}
