#include "cdmlib_precision.h"
#ifndef IFDDA_CDMLIB_CONFIG_C99_H
#define IFDDA_CDMLIB_CONFIG_C99_H

#include <complex.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    CDMLIB_OBJECT_UNKNOWN = 0,
    CDMLIB_OBJECT_SPHERE,
    CDMLIB_OBJECT_INHOMO_SPHERE,
    CDMLIB_OBJECT_INHOMO_CUBOID1,
    CDMLIB_OBJECT_INHOMO_CUBOID2,
    CDMLIB_OBJECT_RANDOM_SPHERE1,
    CDMLIB_OBJECT_RANDOM_SPHERE2,
    CDMLIB_OBJECT_RANDOM_SPHERE_MULTI1,
    CDMLIB_OBJECT_CUBE,
    CDMLIB_OBJECT_CUBOID1,
    CDMLIB_OBJECT_CUBOID2,
    CDMLIB_OBJECT_ELLIPSOID,
    CDMLIB_OBJECT_NSPHERES,
    CDMLIB_OBJECT_CYLINDER,
    CDMLIB_OBJECT_CONCENTRIC_SPHERE,
    CDMLIB_OBJECT_ARBITRARY
} cdmlib_object_kind_c99;

cdmlib_object_kind_c99 cdmlib_classify_object_c99(const char object[64]);

typedef struct {
    char beam[64], object[64], methodeit[12], h5file[100];
    int nside, nrig, nproche, nprochefft, nlentille, nenergie;
    int nsection, nsectionsca, nquickdiffracte, nquicklens;
    int nlocal, nmacro, nobjet, nforce, nforced, ntorque, ntorqued;
    int nlecture, nmat, nprecon, ninitest, nfft2d, nnnr, ntypemic, testgreen;
    int nphi, ntheta, isotropic;
    int nxm, nym, nzm, nabc, nmax, ntotalm, ntotalf, ldabi;
    REAL lambda, w0, P0, theta, k0, numaper, numaperil, numaperinc;
    REAL numaperinc2, kcnax, kcnay, gross, zlens, tolinit;
    int has_hdf5;
} cdmlib_config_c99;

/* C99 translation of the ordered scalar checks from cdmlib.f90. Character
   buffers retain their Fortran fixed-width contents. */
int cdmlib_validate_config_c99(cdmlib_config_c99 *cfg, int *nstop, char infostr[64]);

/* C99 translation of the later near-field option normalization block. */
int cdmlib_normalize_nearfield_c99(cdmlib_config_c99 *cfg, int *nstop, char infostr[64]);

/* Beam-name check immediately following object creation in cdmlib.f90. */
int cdmlib_validate_object_beam_c99(cdmlib_config_c99 *cfg, int *nstop, char infostr[64]);

/* Text options file and HDF5 filename emitted after object validation. */
int cdmlib_write_options_text_c99(const cdmlib_config_c99 *cfg, int *nstop, char infostr[64]);

/* Object postcondition checks corresponding to cdmlib.f90 after geometry
   construction; returns nonzero when the original routine would return. */
int cdmlib_check_object_result_c99(int nx,
                                   int ny,
                                   int nz,
                                   int nxm,
                                   int nym,
                                   int nzm,
                                   int nstop,
                                   COMPLEX eps,
                                   REAL lambda,
                                   REAL aretecube,
                                   REAL *lambda10n,
                                   int *nstop_out,
                                   char infostr[64]);
int cdmlib_check_object_dimensions_c99(
    int nx, int ny, int nz, int nxm, int nym, int nzm, int nstop, int *nstop_out, char infostr[64]);
int cdmlib_check_meshsize_c99(
    COMPLEX eps, REAL lambda, REAL aretecube, REAL *lambda10n, int *nstop, char infostr[64]);
int cdmlib_write_epsilon_ascii_c99(int ndipole,
                                   const int *tabdip,
                                   int nmax,
                                   int niso,
                                   int isotropic,
                                   const COMPLEX *epsilon,
                                   int *nstop,
                                   char infostr[64]);

/* Geometry-construction branches translated from cdmlib.f90. Returns 1 when
   a supported branch was called, otherwise 0 for branches not ported yet. */
typedef struct {
    const char *object64;
    const char *trope3;
    const char *polarizability2;
    COMPLEX *eps, *epsani;
    const COMPLEX *epsmulti;
    const COMPLEX *epsanimulti;
    int *numberobjet;
    REAL *xgmulti, *ygmulti, *zgmulti, *rayonmulti;
    REAL *xswf, *yswf, *zswf, *density;
    REAL *xg, *yg, *zg;
    REAL *side, *sidex, *sidey, *sidez, *height;
    REAL *radius, *axis_a, *axis_b, *axis_c, *lnrbc, *lcos, *lc, *hc;
    REAL *phiobj, *thetaobj, *psiobj;
    int *ntypeabc, *pmlwrk, *ng;
    const COMPLEX *eps0;
    REAL *xs, *ys, *zs;
    const REAL *k0;
    const REAL *lambda;
    REAL *aretecube;
    int *tabdip, *tabmulti;
    const int *nnnr, *nmax;
    const int *nxm, *nym, *nzm, *nxmp, *nymp, *nzmp;
    int *nbsphere, *ndipole, *nx, *ny, *nz;
    int *nproche;
    const int *niso;
    COMPLEX *epsilon, *polarisa;
    COMPLEX *macroscopicfieldx, *epsb;
    REAL *sldr;
    const int *nmat;
    const int64_t *file_id, *group_iddip;
    char *infostr;
    int *nstop;
    const char *namefileobj;
} cdmlib_object_phase_c99;

int cdmlib_call_object_geometry_c99(const cdmlib_object_phase_c99 *args);

#endif
