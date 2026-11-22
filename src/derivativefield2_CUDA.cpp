#include "cdmlib_cpp_precision.hpp"
#include "cuda_interfaces.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>



/* CUDA/cuFFT implementation of cdmlib/derivativefield2.f90. The field spectrum
   was prepared by calculforcefft_; this routine performs the tensor FFT,
   spectral product and inverse FFT on the GPU, then downloads the dipole field. */
extern "C" void derivativefield2_(
    const REAL *aretecube, const REAL *k0, const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nz2, const int *nxy2,
    const int *ntotal, const int *ntotalm, const int *nmax,
    const COMPLEX *vectx, const COMPLEX *vecty, const COMPLEX *vectz,
    const int *axis, COMPLEX *ff0, const int64_t *planb, const int64_t *planf) {
    (void)ntotalm;
    (void)nmax;
    (void)vectx;
    (void)vecty;
    (void)vectz;
    (void)planb;
    (void)planf;
    if (!ntotal || *ntotal <= 0 || !ff0) {
        std::fprintf(stderr, "derivativefield2_: invalid output buffer or count\n");
        std::abort();
    }
    int status = 1;
    ifdda_compute_force_green_derivative_cuda_(aretecube, k0, nx, ny, nz, nx2, ny2,
                                               nz2, nxy2, axis, ff0, &status);
    if (status != 0) {
        std::fprintf(stderr, "derivativefield2_: CUDA/cuFFT derivative calculation failed\n");
        std::abort();
    }
}
