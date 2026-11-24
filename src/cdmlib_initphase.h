#include "cdmlib_precision.h"
#ifndef IFDDA_CDMLIB_INITPHASE_C99_H
#define IFDDA_CDMLIB_INITPHASE_C99_H

#include "cdmlib_config.h"
#include <complex.h>

typedef struct {
    int nxm, nym, nzm;
    int nnnr;
    int numberobjet;
    char object[64];
    char trope[3];
    char polarizability[2];
    const char *materiaumulti;
    REAL tolinit;
    REAL lambda_nm;
    REAL w0_nm;
    REAL power;
    const REAL *xgmulti;
    const REAL *ygmulti;
    const REAL *zgmulti;
    const REAL *rayonmulti;
    COMPLEX *epsmulti;
    const COMPLEX *epsanimulti;

    int nmax, ntotalm, ntotalf, ldabi;
    REAL xg, yg, zg, rayon;
    COMPLEX eps, epsani[9], eps0;
    char materiau[64];
    REAL lambda, w0;
    int nstop;
    int nproche, nprochefft, nabc;
    REAL eps0i;
    char infostr[64];
} cdmlib_initphase_c99;

/* C99 translation of cdmlib.f90's size/scalar initialization and the
   validation immediately following it (source lines 489 onward). */
int cdmlib_initialize_phase_c99(cdmlib_initphase_c99 *state);

/* C99 translation of material interpolation and permittivity validation.
   Returns 2 for the original arbitrary-object jump to label 111. */
int cdmlib_validate_material_phase_c99(cdmlib_initphase_c99 *state);

/* Join the translated input, material, wavenumber and near-field setup
   phases that run before geometry creation in cdmlib.f90. */
int cdmlib_prepare_preobject_phase_c99(cdmlib_initphase_c99 *state,
                                       cdmlib_config_c99 *config,
                                       REAL *k0,
                                       REAL *k03);

/* Connected port of cdmlib's ordered setup through geometry construction
   and the post-construction dimension/mesh checks. Geometry-specific buffers
   and shape parameters remain supplied in the second argument. */
int cdmlib_construct_object_phase_c99(cdmlib_initphase_c99 *state,
                                      cdmlib_config_c99 *config,
                                      cdmlib_object_phase_c99 *geometry,
                                      REAL *k0,
                                      REAL *k03,
                                      REAL *lambda10n);

#endif
