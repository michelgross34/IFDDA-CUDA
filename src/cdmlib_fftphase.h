#include "cdmlib_precision.h"
#ifndef IFDDA_CDMLIB_FFTPHASE_C99_H
#define IFDDA_CDMLIB_FFTPHASE_C99_H

#include <complex.h>

typedef struct {
    int nx2, ny2, nz2;
    int ntotal, nbsphere3, nxy2, nabs;
} cdmlib_fftgrid_c99;

/* C99 translation of cdmlib.f90's padded FFT-grid sizing block. */
int cdmlib_prepare_fft_grid_c99(int nx,
                                int ny,
                                int nz,
                                int nbsphere,
                                int nabc,
                                int nlentille,
                                int nforce,
                                int ntorque,
                                COMPLEX eps0,
                                REAL k0,
                                REAL aretecube,
                                cdmlib_fftgrid_c99 *grid,
                                int *nstop,
                                char infostr[64]);

#endif
