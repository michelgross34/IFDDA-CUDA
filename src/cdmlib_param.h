#ifndef IFDDA_CUDA_CDMLIB_PARAM_H
#define IFDDA_CUDA_CDMLIB_PARAM_H

// Include after selecting the C++ precision and suppressing the C99
// precision macros, as in cdmlib.cpp. No GPU ownership is transferred.
#include "cdmlib_abi.h"
#include "cdmlib_initphase.h"
#include <vector>

typedef struct COMPLEX_ARRAY COMPLEX_ARRAY;

// Persistent local state for the orchestration; the geometry view below
// borrows addresses in this state and in the caller-owned Fortran arguments.
typedef struct {
    int nx, ny, nz;
    int nx2, ny2, nz2, ntotal, nbsphere3, nxy2, nabs;
    int pmlwrk[5];
    int64_t file_id, group_idopt, group_iddip, group_idff, group_idmic;
    int64_t group_idof, group_idnf, group_idver;
    REAL xg, yg, zg, radius, sldr;
    REAL antenna_x, antenna_y, antenna_z;
    REAL waist, irradiance, I0, norm, tmp, xgaus, ygaus, zgaus;
    COMPLEX E0;
    COMPLEX *vectx, *vecty, *vectz;
    COMPLEX_ARRAY *green_tensor_arrays[6];
    // The six vector Green tensors are built once per cdmlib_ invocation.
    // Reused by testgreen iterations and microscopy on the same FFT grid.
    int vector_green_ready;
    COMPLEX epsb[9];
    int64_t planf, planb, planXf, planXb, planYf, planYb, planZf, planZb;
    int64_t plan2f, plan2b;
} cdmlib_entry_runtime_c99;

// Geometry-only, non-owning view: all fields in object are addresses.
// It deliberately does not expose illumination, Green or solver buffers.
typedef struct {
    cdmlib_object_phase_c99 object;
} cdmlib_geometry_params_c99;

// Fill the geometry address view after phase initialization, immediately
// before cdmlib_construct_object_phase_c99. Does not allocate/copy arrays.
int cdmlib_prepare_geometry_params_c99(
    const cdmlib_abi_args_c99 *args,
    cdmlib_initphase_c99 *init,
    cdmlib_entry_runtime_c99 *runtime,
    REAL *k0,
    cdmlib_geometry_params_c99 *geometry);

// Non-owning illumination view. The legacy ABI/phase pointers are retained
// only for the existing beam dispatch and mapping helpers; the PML and GPU
// publication steps use the explicit addresses below.
typedef struct {
    const cdmlib_abi_args_c99 *abi;
    cdmlib_initphase_c99 *init;
    cdmlib_config_c99 *config;
    cdmlib_entry_runtime_c99 *runtime;
    REAL *xs, *ys, *zs;
    REAL *k0, *aretecube;
    COMPLEX *ff0;
    const int *tabmulti;
    cdmlib_ip *nbsphere, *ndipole, *nlecture, *nstop;
    char *infostr;
    size_t infostr_len;
    const char *beam;
    std::vector<REAL> *pml_mask;
} cdmlib_illumination_params_c99;

// Only pointer assignments: no host/device transfer or allocation.
int cdmlib_prepare_illumination_params_c99(
    const cdmlib_abi_args_c99 *args,
    cdmlib_initphase_c99 *init,
    cdmlib_config_c99 *config,
    cdmlib_entry_runtime_c99 *runtime,
    std::vector<REAL> *pml_mask,
    cdmlib_illumination_params_c99 *illumination);

// Green-only non-owning view. Six tensor output addresses and scalar
// parameters are borrowed; allocations and CUDA residency remain unchanged.
typedef struct {
    const cdmlib_abi_args_c99 *abi;
    const cdmlib_initphase_c99 *init;
    const cdmlib_config_c99 *config;
    cdmlib_entry_runtime_c99 *runtime;
    COMPLEX *xx, *xy, *xz, *yy, *yz, *zz;
    const REAL *k0, *aretecube, *theta, *phi;
    const int *nquad, *ntypeabc;
    const COMPLEX *eps0;
    const char *polarizability;
    cdmlib_ip *nstop;
    char *infostr;
    size_t infostr_len;
} cdmlib_green_params_c99;

int cdmlib_prepare_green_params_c99(
    const cdmlib_abi_args_c99 *args,
    const cdmlib_initphase_c99 *init,
    const cdmlib_config_c99 *config,
    cdmlib_entry_runtime_c99 *runtime,
    cdmlib_green_params_c99 *green);

// Scalar Green FFT parameters. All pointers borrow existing storage;
// the resulting scalar Green uses vectx (mode 8) or vecty (mode 9).
typedef struct {
    int mode;
    int polarization_prepared; // 0=normal; 1=provided; 2=vector warm-start
    const char *beam, *trope;
    const cdmlib_ip *nprecon, *ninitest;
    const REAL *ss, *pp, *theta, *phi, *w0;
    const REAL *xgaus, *ygaus, *zgaus;
    const COMPLEX *eps0, *e0;
    const int *nx, *ny, *nz, *nx2, *ny2, *nz2, *nxy2;
    const int *nxm, *nym, *nzm, *ntotalm;
    const REAL *aretecube, *k0;
    COMPLEX *ux, *uy, *uz;
    int *ready;
    COMPLEX *green;
    const int64_t *planb;
    cdmlib_ip *nstop;
    char *infostr;
    size_t infostr_len;
} cdmlib_green_scalar_params_c99;

// mode=8: projected scalar Green; mode=9: exp(ikr)/r scalar Green.
// Also used with mode=8 for the vector solver's scalar initial guess.
int cdmlib_prepare_green_scalar_params_c99(
    const cdmlib_abi_args_c99 *args,
    const cdmlib_initphase_c99 *init,
    cdmlib_entry_runtime_c99 *runtime,
    int mode,
    int polarization_prepared,
    COMPLEX *ux, COMPLEX *uy, COMPLEX *uz,
    int *ready,
    cdmlib_green_scalar_params_c99 *green);

// Parameters of the ninitest=4 scalar solve used to warm-start the vector
// solver. Pointers borrow existing CPU/CUDA-backed buffers; no array copy.
// The scalar Green operator and normalized polarization must already exist.
typedef struct {
    COMPLEX *green, *scratch, *macroscopicfieldx, *macroscopicfieldy;
    COMPLEX *ux, *uy, *uz;
    COMPLEX *xi, *xr, *wrk, *ff, *ff0, *ffloc, *polarisa;
    int *tabdip;
    const REAL *k0, *aretecube;
    int ntotalm, ntotal, ldabi, nlar, nmax, ndipole;
    int nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2;
    int nbsphere, niso, nlim, nproche;
    int64_t *planf, *planb;
    cdmlib_ip *nstop, *nloop, *ncompte;
    cdmlib_real *tol1;
    char *methodeit;
    char *infostr;
    size_t methodeit_len, infostr_len;
} cdmlib_scalar_initialization_params_c99;

int cdmlib_prepare_scalar_initialization_params_c99(
    const cdmlib_abi_args_c99 *args,
    const cdmlib_initphase_c99 *init,
    cdmlib_entry_runtime_c99 *runtime,
    COMPLEX *ux, COMPLEX *uy, COMPLEX *uz,
    cdmlib_scalar_initialization_params_c99 *solver);

#endif
