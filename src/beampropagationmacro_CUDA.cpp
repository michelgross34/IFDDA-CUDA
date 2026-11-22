#include "cuda_data.h"
#include "cuda_interfaces.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>

static_assert(sizeof(CUCOMPLEX) == 2 * sizeof(CU_REAL),
              "Fortran COMPLEX ABI must match the selected CUDA precision");



//-----------------------------------------------------------
extern "C" void beampropagationmacro_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs,
    const CU_REAL *aretecube, const CU_REAL *k0, const CU_REAL *w0,
    const COMPLEX *E0, const CU_REAL *ss, const CU_REAL *pp,
    const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64],
    const COMPLEX *epsilon, const int *niso, const int *ndipole,
    const int *nx, const int *ny, const int *nz, const int *nxm,
    const int *nym, const int *nzm, const int *nmax, const int *nfft2d,
    COMPLEX *image_x, COMPLEX *image_y, COMPLEX *image_z,
    const COMPLEX *incident, COMPLEX *local_field, COMPLEX *field,
    const IFDDA_PLAN_INT64 *plan2f, const IFDDA_PLAN_INT64 *plan2b, int *nstop, char infostr[64],
    size_t beam_length, size_t infostr_length) {
    (void)nxm;
    (void)nym;
    (void)nzm;
    (void)image_x;
    (void)image_y;
    (void)image_z;
    (void)field;
    (void)plan2f;
    (void)plan2b;

    if (!nstop) return;
    *nstop = 1;
    if (infostr && infostr_length) std::memset(infostr, ' ', infostr_length);

    if (!xs || !ys || !zs || !aretecube || !k0 || !w0 || !E0 || !ss || !pp ||
        !theta || !phi || !xgaus || !ygaus || !zgaus || !beam || !epsilon ||
        !niso || !ndipole || !nx || !ny || !nz || !nmax || !nfft2d ||
        !incident || !local_field || !nstop || !infostr || *ndipole <= 0 ||
        *nmax <= 0 || *niso <= 0 || *nx <= 0 || *ny <= 0 || *nz <= 1 ||
        *nfft2d <= 0 || *nfft2d > std::numeric_limits<int>::max() / *nfft2d) {
        if (infostr && infostr_length) {
            const char message[] = "invalid beam propagation input";
            std::memcpy(infostr, message,
                        std::min(infostr_length, sizeof(message) - 1));
        }
        return;
    }

    int beam_type = -1;
    if (beam_length >= 11 && std::strncmp(beam, "pwavelinear", 11) == 0) {
        beam_type = 0;
    } else if (beam_length >= 13 && std::strncmp(beam, "pwavecircular", 13) == 0) {
        beam_type = 1;
    } else if (beam_length >= 15 && std::strncmp(beam, "gparawavelinear", 15) == 0) {
        beam_type = 2;
    } else if (beam_length >= 17 && std::strncmp(beam, "gparawavecircular", 17) == 0) {
        beam_type = 3;
    } else if ((beam_length >= 11 && std::strncmp(beam, "gwavelinear", 11) == 0) ||
               (beam_length >= 14 && std::strncmp(beam, "gfftwavelinear", 14) == 0)) {
        beam_type = 4;
    } else if ((beam_length >= 13 && std::strncmp(beam, "gwavecircular", 13) == 0) ||
               (beam_length >= 16 && std::strncmp(beam, "gfftwavecircular", 16) == 0)) {
        beam_type = 5;
    }
    if (beam_type < 0) {
        const char message[] = "unsupported beam for CUDA BPM";
        std::memcpy(infostr, message, std::min(infostr_length, sizeof(message) - 1));
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
    const int resident_mode = 0;
    ifdda_beampropagation_run_cuda_(
        xs, ys, zs, epsilon, incident, local_field, nx, ny, nz, nfft2d,
        nmax, niso, ndipole, &kkm, &jjm, &xmin, &ymin, zs, aretecube, k0,
        E0, ss, pp, theta, phi, xgaus, ygaus, zgaus, w0, &beam_type,
        resident_mode, 1, nstop);
    if (*nstop != 0) {
        const char message[] = "CUDA beam propagation failed";
        std::memcpy(infostr, message, std::min(infostr_length, sizeof(message) - 1));
    }
}
