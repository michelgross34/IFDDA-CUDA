#ifndef IFDDA_CUDA_CDMLIB_FFTW_ABI_H
#define IFDDA_CUDA_CDMLIB_FFTW_ABI_H

// Fortran FFTW wrappers; include only after the precision-specific
// CDMLIB_FFTW(name) macro and cdmlib_fftw_complex type are defined.
#ifdef USE_FFTW
extern "C" void CDMLIB_FFTW(init_threads)(int *);
extern "C" void CDMLIB_FFTW(plan_with_nthreads)(const int *);
extern "C" void CDMLIB_FFTW(plan_dft_3d)(
    int64_t *, const int *, const int *, const int *,
    cdmlib_fftw_complex *, cdmlib_fftw_complex *, const int *, const unsigned *);
extern "C" void CDMLIB_FFTW(plan_dft_2d)(
    int64_t *, const int *, const int *,
    cdmlib_fftw_complex *, cdmlib_fftw_complex *, const int *, const unsigned *);
extern "C" void CDMLIB_FFTW(plan_many_dft)(
    int64_t *, const int *, const int *, const int *,
    cdmlib_fftw_complex *, const int *, const int *, const int *,
    cdmlib_fftw_complex *, const int *, const int *, const int *,
    const int *, const unsigned *);
extern "C" void CDMLIB_FFTW(destroy_plan)(const int64_t *);
#ifdef _OPENMP
extern "C" int omp_get_max_threads(void);
#endif
#endif

#endif
