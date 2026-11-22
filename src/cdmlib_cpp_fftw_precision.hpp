#ifndef IFDDA_CDMLIB_CPP_FFTW_PRECISION_HPP
#define IFDDA_CDMLIB_CPP_FFTW_PRECISION_HPP

#include <fftw3.h>

#ifdef USE_SIMP
using cdmlib_fftw_plan = fftwf_plan;
using cdmlib_fftw_complex = fftwf_complex;
#define CDMLIB_FFTW_EXECUTE fftwf_execute
#define CDMLIB_FFTW_EXECUTE_DFT(plan, in, out)                                                     \
    fftwf_execute_dft(                                                                             \
        (plan), reinterpret_cast<fftwf_complex *>(in), reinterpret_cast<fftwf_complex *>(out))
#define CDMLIB_FFTW_DESTROY_PLAN fftwf_destroy_plan
#define CDMLIB_FFTW_PLAN_DFT_2D(ny, nx, in, out, sign, flags)                                      \
    fftwf_plan_dft_2d((ny),                                                                        \
                      (nx),                                                                        \
                      reinterpret_cast<fftwf_complex *>(in),                                       \
                      reinterpret_cast<fftwf_complex *>(out),                                      \
                      (sign),                                                                      \
                      (flags))
#define CDMLIB_FFTW_PLAN_DFT_3D(nz, ny, nx, in, out, sign, flags)                                  \
    fftwf_plan_dft_3d((nz),                                                                        \
                      (ny),                                                                        \
                      (nx),                                                                        \
                      reinterpret_cast<fftwf_complex *>(in),                                       \
                      reinterpret_cast<fftwf_complex *>(out),                                      \
                      (sign),                                                                      \
                      (flags))
#define CDMLIB_FFTW_PLAN_GURU_DFT(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags)    \
    fftwf_plan_guru_dft((rank),                                                                    \
                        (dims),                                                                    \
                        (howmany_rank),                                                            \
                        (howmany_dims),                                                            \
                        reinterpret_cast<fftwf_complex *>(in),                                     \
                        reinterpret_cast<fftwf_complex *>(out),                                    \
                        (sign),                                                                    \
                        (flags))
#define CDMLIB_LAPACK_GETRF cgetrf_
#define CDMLIB_LAPACK_GETRS cgetrs_
#else
using cdmlib_fftw_plan = fftw_plan;
using cdmlib_fftw_complex = fftw_complex;
#define CDMLIB_FFTW_EXECUTE fftw_execute
#define CDMLIB_FFTW_EXECUTE_DFT(plan, in, out)                                                     \
    fftw_execute_dft(                                                                              \
        (plan), reinterpret_cast<fftw_complex *>(in), reinterpret_cast<fftw_complex *>(out))
#define CDMLIB_FFTW_DESTROY_PLAN fftw_destroy_plan
#define CDMLIB_FFTW_PLAN_DFT_2D(ny, nx, in, out, sign, flags)                                      \
    fftw_plan_dft_2d((ny),                                                                         \
                     (nx),                                                                         \
                     reinterpret_cast<fftw_complex *>(in),                                         \
                     reinterpret_cast<fftw_complex *>(out),                                        \
                     (sign),                                                                       \
                     (flags))
#define CDMLIB_FFTW_PLAN_DFT_3D(nz, ny, nx, in, out, sign, flags)                                  \
    fftw_plan_dft_3d((nz),                                                                         \
                     (ny),                                                                         \
                     (nx),                                                                         \
                     reinterpret_cast<fftw_complex *>(in),                                         \
                     reinterpret_cast<fftw_complex *>(out),                                        \
                     (sign),                                                                       \
                     (flags))
#define CDMLIB_FFTW_PLAN_GURU_DFT(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags)    \
    fftw_plan_guru_dft((rank),                                                                     \
                       (dims),                                                                     \
                       (howmany_rank),                                                             \
                       (howmany_dims),                                                             \
                       reinterpret_cast<fftw_complex *>(in),                                       \
                       reinterpret_cast<fftw_complex *>(out),                                      \
                       (sign),                                                                     \
                       (flags))
#define CDMLIB_LAPACK_GETRF zgetrf_
#define CDMLIB_LAPACK_GETRS zgetrs_
#endif

#endif
