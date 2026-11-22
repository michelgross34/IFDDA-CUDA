#include "cuda_data.h"
#define IFDDA_CUDA_CALLERS_DIFFRACTEFFT2DINCKZNEG_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_DIFFRACTEFFT2DINCKZNEG_CUDA
#include "cuda_utils.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <vector>

using Complex = std::complex<CU_REAL>;



namespace {

//-----------------------------------------------------------
void set_info(int *nstop, char *info, size_t length, const char *message) {
    if (nstop)
        *nstop = 1;
    if (!info || length == 0)
        return;
    std::memset(info, ' ', length);
    const size_t message_length = std::strlen(message);
    std::memcpy(info, message, message_length < length ? message_length : length);
}

//-----------------------------------------------------------
bool beam_is(const char beam[64], const char *name, size_t length) {
    return beam && std::memcmp(beam, name, length) == 0;
}

} // namespace

//-----------------------------------------------------------
extern "C" void diffractefft2dinckzneg_(
    const int *speckseed, const int *nfft2d, const CU_REAL *k0, const Complex *E0,
    const CU_REAL *ss, const CU_REAL *pp, const CU_REAL *theta, const CU_REAL *phi,
    const CU_REAL *thetam, const CU_REAL *phim, const CU_REAL *ppm, const CU_REAL *ssm,
    const Complex *E0m, const int *nbinc, const CU_REAL *xdip, const CU_REAL *ydip,
    const CU_REAL *zdip, const CU_REAL *xgaus, const CU_REAL *ygaus, const CU_REAL *zgaus,
    const CU_REAL *numaperil, const CU_REAL *w0, const CU_REAL *aretecube, const CU_REAL *tol,
    Complex *Eloinx, Complex *Eloiny, Complex *Eloinz, int *imax,
    CU_REAL *deltakx, CU_REAL *deltaky, Complex *Ediffkzpos, const CU_REAL *numaper,
    const char beam[64], const long long *plan2f, const long long *plan2b,
    int *nstop, char infostr[64], size_t infostr_length) {
    (void)ss;
    (void)theta;
    (void)phi;
    (void)thetam;
    (void)phim;
    (void)ppm;
    (void)ssm;
    (void)E0m;
    (void)nbinc;
    (void)xdip;
    (void)ydip;
    (void)zdip;
    (void)w0;
    (void)aretecube;
    (void)tol;
    (void)Eloinx;
    (void)Eloiny;
    (void)Eloinz;
    (void)numaper;
    (void)plan2f;
    (void)plan2b;

    if (nstop)
        *nstop = 0;
    if (infostr && infostr_length)
        std::memset(infostr, ' ', infostr_length);
    if (!speckseed || !nfft2d || !k0 || !E0 || !pp || !xgaus || !ygaus || !zgaus ||
        !numaperil || !imax || !deltakx || !deltaky || !Ediffkzpos || !beam || !nstop ||
        *nfft2d <= 0 || *nfft2d > 16384 || *imax < 0 || 2LL * *imax + 1 > *nfft2d ||
        *k0 <= CU_REAL(0) || *numaperil < CU_REAL(0) || *deltakx <= CU_REAL(0) || *deltaky <= CU_REAL(0)) {
        set_info(nstop, infostr, infostr_length, "Invalid negative-kz diffraction arguments");
        return;
    }

    const bool speckle = beam_is(beam, "speckle", 7);
    const bool confocal = beam_is(beam, "confocal", 8);
    const int illumination_kind = speckle ? 1 : confocal ? 2 : 0;
    const size_t nfft = size_t(*nfft2d);
    const size_t plane = nfft * nfft;
    if (plane > std::numeric_limits<size_t>::max() / (3 * sizeof(CUCOMPLEX))) {
        set_info(nstop, infostr, infostr_length, "Negative-kz diffraction dimensions overflow");
        return;
    }
    const CUCOMPLEX *phase_positive_device = nullptr;
    const CUCOMPLEX *phase_negative_device = nullptr;
    int phase_grid_size = 0;
    if (speckle) {
        phase_grid_size = 2 * *imax + 1;
        const CU_REAL phase_spacing =
            CU_REAL(6.283185307179586476925286766559L) /
            (CU_REAL(phase_grid_size) * *deltakx);
        const int bidirectional = 1;
        if (std::abs(*deltakx - *deltaky) >
            CU_REAL(32) * std::numeric_limits<CU_REAL>::epsilon() *
                std::max(std::abs(*deltakx), std::abs(*deltaky))) {
            set_info(nstop, infostr, infostr_length,
                     "Negative-kz speckle requires equal x and y spectral spacing");
            return;
        }
        const int phase_status = ifdda_prepare_speckle_phases_device_(
            &phase_grid_size, &phase_grid_size, k0, numaperil, &phase_spacing,
            speckseed, &bidirectional, &phase_positive_device, &phase_negative_device);
        if (phase_status != 0) {
            set_info(nstop, infostr, infostr_length,
                     "CUDA speckle phase generation failed");
            return;
        }
    }

    if (!initialize_cuda_device()) {
        set_info(nstop, infostr, infostr_length, "CUDA device initialization failed");
        return;
    }
    cuda_data::NegativeKzDiffractionDeviceBuffers buffers{};
    if (!cuda_data::get_device_negative_kz_diffraction_buffers(plane, &buffers) ||
        !buffers.random_phases || !buffers.spectrum) {
        set_info(nstop, infostr, infostr_length, "Could not acquire spectral GPU buffers");
        return;
    }
    const cudaError_t phase_grid_status = speckle
        ? speckle_embed_negative_kz_phase_wrapper(
            phase_negative_device, buffers.random_phases, phase_grid_size,
            *nfft2d, *imax)
        : lens_diffraction_clear_wrapper(buffers.random_phases, static_cast<int>(plane));
    if (!cuda_utils::check_cuda(phase_grid_status,
            "negative-kz diffraction", "prepare resident speckle phase grid") ||
        !cuda_utils::check_cuda(
            negative_kz_diffraction_spectrum_wrapper(
                buffers.random_phases, buffers.spectrum, *nfft2d, *imax, *deltakx, *deltaky,
                *k0, *numaperil, *pp, *xgaus, *ygaus, *zgaus,
                cuda_utils::device_value(E0[0]), illumination_kind),
            "negative-kz diffraction", "compute negative-kz spectrum") ||
        !cuda_utils::check_cuda(
            cuda_data::copy_device_to_host(Ediffkzpos, buffers.spectrum,
                3 * plane * sizeof(CUCOMPLEX), cuda_utils::TransferCategory::Result),
            "negative-kz diffraction", "copy spectrum to Fortran output")) {
        set_info(nstop, infostr, infostr_length, "CUDA negative-kz diffraction failed");
        return;
    }
}
