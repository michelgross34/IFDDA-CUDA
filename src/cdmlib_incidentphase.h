#include "cdmlib_precision.h"
#ifndef IFDDA_CDMLIB_INCIDENTPHASE_C99_H
#define IFDDA_CDMLIB_INCIDENTPHASE_C99_H

#include <complex.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int nbsphere;
    int nmax;
    int nenergie, ntypemic, nlentille, nfft2d, speckseed;
    int nbinc;
    REAL power, waist, k0, aretecube, numaperil;
    REAL antenna_x_nm, antenna_y_nm, antenna_z_nm;
    REAL xg, yg, zg, antenna_x, antenna_y, antenna_z;
    REAL xgaus_nm, ygaus_nm, zgaus_nm, xgaus, ygaus, zgaus;
    REAL theta, phi, ss, pp;
    REAL tolinit;
    COMPLEX eps0, E0;
    REAL irradiance, I0, norm, tmp;
    const REAL *xs, *ys, *zs;
    const int *tabmulti;
    const REAL *thetam, *phim, *ppm, *ssm;
    COMPLEX *E0m;
    COMPLEX *FF0;
    int nx, ny, nz, nxm, nym, nzm, ndipole;
    int64_t plan2f, plan2b;
    char beam[64];
    char namefileinc[64];
    size_t namefileinc_len;
    COMPLEX *imagex, *imagey, *imagez;
    int nstop;
    char infostr[64];
    /* Optional complete-grid index -> compact dipole mapping for CUDA waves. */
    const int *tabdip;
} cdmlib_incidentphase_c99;

/* C99 translation of cdmlib.f90's pwavelinear incident-field branch. */
int cdmlib_compute_plane_linear_incident_c99(cdmlib_incidentphase_c99 *s);
int cdmlib_compute_plane_circular_incident_c99(cdmlib_incidentphase_c99 *s);
int cdmlib_compute_plane_multi_incident_c99(cdmlib_incidentphase_c99 *s);
int cdmlib_compute_antenna_incident_c99(cdmlib_incidentphase_c99 *s);
int cdmlib_compute_gaussian_para_linear_incident_c99(cdmlib_incidentphase_c99 *s);
int cdmlib_compute_gaussian_para_circular_incident_c99(cdmlib_incidentphase_c99 *s);
int cdmlib_compute_gaussian_linear_incident_c99(cdmlib_incidentphase_c99 *s);
int cdmlib_compute_gaussian_circular_incident_c99(cdmlib_incidentphase_c99 *s);
int cdmlib_compute_gaussian_fft_incident_c99(cdmlib_incidentphase_c99 *s);
/* Dispatch the supported plane-wave and Gaussian phase branches by beam name.
   Returns 2 when the beam belongs to a phase that has not been ported yet. */
int cdmlib_compute_incident_field_phase_c99(cdmlib_incidentphase_c99 *s);

#endif
