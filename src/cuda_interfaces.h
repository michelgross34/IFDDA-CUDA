// Shared C-linkage signatures, taken from the defining CUDA translation units.
// Regenerate with: python CUDA/tools/audit_interfaces.py --generate
// Include in the definitions and callers of all migrated symbols.
#ifndef IFDDA_CUDA_INTERFACES_H
#define IFDDA_CUDA_INTERFACES_H

#include "cuda_solver_interfaces.h"
#include "cuda_polarizability_interfaces.h"
#include "cdmlib_geometry_abi.hpp"
#include "plan_int64_abi.h"
#include <complex>

// Shared CUDA geometry selector for the C++ and hybrid Fortran orchestrators.
// The Fortran ABI passes scalar/array addresses; five CHARACTER hidden lengths
// are appended in the order object, trope, polarizability, namefileobj, infostr.
// The selector calls ifdda_object_all_*_cuda directly (no objet* adapters).
extern "C" void ifdda_geometry_dispatch_cuda_(
    const char* object64, const char* trope3, const char* polarizability2,
    const char* namefileobj, char* infostr,
    COMPLEX* eps, COMPLEX* epsani, const COMPLEX* epsmulti,
    const COMPLEX* epsanimulti, int* numberobjet,
    REAL* xgmulti, REAL* ygmulti, REAL* zgmulti, REAL* rayonmulti,
    REAL* xswf, REAL* yswf, REAL* zswf, REAL* density,
    REAL* xg, REAL* yg, REAL* zg,
    REAL* side, REAL* sidex, REAL* sidey, REAL* sidez, REAL* height,
    REAL* radius, REAL* axis_a, REAL* axis_b, REAL* axis_c,
    REAL* lnrbc, REAL* lcos, REAL* lc, REAL* hc,
    REAL* phiobj, REAL* thetaobj, REAL* psiobj,
    int* ntypeabc, int* pmlwrk, int* ng,
    const COMPLEX* eps0, REAL* xs, REAL* ys, REAL* zs,
    const REAL* k0, const REAL* lambda, REAL* aretecube,
    int* tabdip, int* tabmulti, const int* nnnr, const int* nmax,
    const int* nxm, const int* nym, const int* nzm,
    const int* nxmp, const int* nymp, const int* nzmp,
    int* nbsphere, int* ndipole, int* nx, int* ny, int* nz,
    int* nproche, const int* niso, COMPLEX* epsilon, COMPLEX* polarisa,
    COMPLEX* macroscopicfieldx, REAL* sldr, const int* nmat,
    const int64_t* file_id, const int64_t* group_iddip, int* nstop,
    size_t object_length, size_t trope_length, size_t polarizability_length,
    size_t namefileobj_length, size_t infostr_length);

// Fortran ABI: CPU tensor arguments are host identity keys (not GPU pointers).
// The GPU-only producer writes directly to cuda_data::get_green().
extern "C" void ifdda_generate_vector_green_cuda(
    COMPLEX* gxx, COMPLEX* gxy, COMPLEX* gxz,
    COMPLEX* gyy, COMPLEX* gyz, COMPLEX* gzz,
    const int* nx, const int* ny, const int* nz,
    const int* nxm, const int* nym, const int* nzm,
    const int* nx2, const int* ny2, const int* nz2,
    const REAL* aretecube, const REAL* k0, const COMPLEX* eps0,
    const int* medium, const int* nquad, const int* filtered, int* status);

// Fortran-ABI incident-field routines used by cdmlib.cpp.
// Keep their declarations here, not in individual translation units.
extern "C" void rytov_vector_field_(
    const COMPLEX *eref, const COMPLEX *delta, COMPLEX *result,
    const int *nb, int *nstop);
extern "C" void incidentarbitrary_(REAL *xs, REAL *ys, REAL *zs,
    REAL *aretecube, COMPLEX *ff0, int *nxm, int *nym, int *nzm,
    int *nbsphere, int *nstop, char *namefile, char *infostr,
    size_t namefile_len, size_t infostr_len);
extern "C" void speckle_(const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, COMPLEX *, const REAL *, const REAL *, const REAL *,
    COMPLEX *, const REAL *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, COMPLEX *,
    COMPLEX *, COMPLEX *, const int *, const REAL *, REAL *,
    const REAL *, int *, char *, const int64_t *, size_t);
extern "C" void demispeckle_(const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, COMPLEX *, const REAL *, const REAL *, const REAL *,
    COMPLEX *, const REAL *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, COMPLEX *,
    COMPLEX *, COMPLEX *, const int *, const REAL *, REAL *,
    const REAL *, int *, char *, const int64_t *, size_t);
extern "C" void confocal_(const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, COMPLEX *, const REAL *, const REAL *, const REAL *,
    COMPLEX *, const REAL *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, COMPLEX *,
    COMPLEX *, COMPLEX *, const REAL *, REAL *, const REAL *,
    int *, char *, const int64_t *, size_t);
extern "C" void demiconfocal_(const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, COMPLEX *, const REAL *, const REAL *, const REAL *,
    COMPLEX *, const REAL *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, COMPLEX *,
    COMPLEX *, COMPLEX *, const REAL *, REAL *, const REAL *,
    int *, char *, const int64_t *, size_t);

// Original Green tensor Fortran/C ABI. These functions operate on CPU
// arguments; CUDA uses the same signatures when dispatching from cdmlib.cpp.
extern "C" void greencalculfft_(const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, REAL *, REAL *, COMPLEX *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, const int64_t *);
extern "C" void greencalculfftc_(const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, REAL *, REAL *, const COMPLEX *, const int *,
    const REAL *, const REAL *, const REAL *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *,
    const int64_t *);
extern "C" void greenfilcalculfft_(const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, REAL *, REAL *, COMPLEX *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, const int64_t *);
extern "C" void greencalculquadfft_(const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, const int *, const REAL *, REAL *, REAL *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *,
    const int64_t *);
extern "C" void greencalculquadfftc_(const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, const int *, REAL *, REAL *, const COMPLEX *,
    const int *, const REAL *, const REAL *, const REAL *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *,
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *,
    const int64_t *);

// CPU-side polarizability calculations. GPU kernels and device allocations
// continue to use CUCOMPLEX; conversion occurs at their CUDA boundary.
void poladiff_cuda(
    const CU_REAL* radius, const COMPLEX* epsilon, const COMPLEX* epsilon0,
    const CU_REAL* k0, const int* discretized, const char* method,
    CU_REAL* sldr, COMPLEX* polarizability, size_t method_length);
void polaepstens_cuda(
    const CU_REAL* radius, const COMPLEX* epsilon, const COMPLEX* epsilon0,
    const CU_REAL* k0, const int* discretized, const char* method,
    const int* invert_result, COMPLEX* polarizability, size_t method_length);

// Historical C ABI retained for any external consumers.
extern "C" void ifdda_poladiff_cuda_abi_(
    const CU_REAL* radius, const IfddaCudaPolarizabilityComplex* epsilon,
    const IfddaCudaPolarizabilityComplex* epsilon0, const CU_REAL* k0,
    const int* discretized, const char* method, CU_REAL* sldr,
    IfddaCudaPolarizabilityComplex* polarizability, size_t method_length);
extern "C" void ifdda_polaepstens_cuda_abi_(
    const CU_REAL* radius, const IfddaCudaPolarizabilityComplex* epsilon,
    const IfddaCudaPolarizabilityComplex* epsilon0, const CU_REAL* k0,
    const int* discretized, const char* method, const int* invert_result,
    IfddaCudaPolarizabilityComplex* polarizability, size_t method_length);

// gfortran entry points used by the hybrid main0.f90 + cdmlib.f90 build.
// The complex arguments use the Fortran memory layout (two CPU REAL values);
// the trailing size_t is the hidden length of CHARACTER(len=2) methode.
// Both delegate to the existing CUDA polarizability implementations.
extern "C" void poladiff_(
    const REAL* radius, const COMPLEX* epsilon,
    const COMPLEX* epsilon0, const REAL* k0,
    const int* discretized, const char* method, REAL* sldr,
    COMPLEX* polarizability, size_t method_length);
extern "C" void polaepstens_(
    const REAL* radius, const COMPLEX* epsilon,
    const COMPLEX* epsilon0, const REAL* k0,
    const int* discretized, const char* method, const int* invert_result,
    COMPLEX* polarizability, size_t method_length);

// Calculation-owned persistent device arrays. Each getter allocates once,
// reuses its pointer for requests within capacity, and rejects larger requests.
// ifdda_reset_cuda_device_state() invalidates all five pointers and capacities.
namespace cuda_data {
CU_INT* get_dipole_map(size_t grid_count);
CU_INT* get_tapdip(size_t grid_count); // GPU-only access; does not copy from CPU
CUCOMPLEX* get_polarisa(size_t tensor_count); // compact nbsphere * niso * niso
CUCOMPLEX* get_epsilon(size_t tensor_count);  // compact nbsphere * niso * niso
CUCOMPLEX* get_ff0(size_t nbsphere);           // 3 * nbsphere, interleaved xyz
}
// CPU-produced tabdip: upload on call into the same resident GPU allocation.
extern "C" CU_INT* get_tapdip(const int* cpu_tabdip, size_t count);

namespace cuda_data { struct WideFieldDeviceBuffers; }

// CPU-only description of an optional regular grid. tabdip is indexed by the
// full-grid linear position and holds the one-based compact dipole slot.
// The CUDA execution entries below validate this mapping; irregular meshes
// automatically use explicit coordinates. Device arrays remain CU_* pointers.
struct IfddaIncidentRegularGridSpec {
    const int* tabdip;
    const int* nx;
    const int* ny;
    const int* nz;
    const int* grid_count;
    const REAL* cell_size;
};

// C99 illumination paths call the specialized CUDA execution directly.
// Other historical Fortran/widefield entry points retain their C ABI.
void ifdda_plane_wave_field_cuda_execute(
    const REAL*, const REAL*, const REAL*, const int*, const REAL*,
    const COMPLEX*, const COMPLEX*, const REAL*, const REAL*, const REAL*,
    const REAL*, const int*, COMPLEX*, CUCOMPLEX**, int*,
    const CU_REAL* = nullptr, const CU_REAL* = nullptr,
    const CU_REAL* = nullptr, CUCOMPLEX* = nullptr,
    const IfddaIncidentRegularGridSpec* = nullptr);
void ifdda_multi_plane_wave_field_cuda_execute(
    const REAL*, const REAL*, const REAL*, const int*, const REAL*,
    const COMPLEX*, const COMPLEX*, const REAL*, const REAL*, const REAL*,
    const REAL*, const int*, COMPLEX*, int*, bool,
    CUCOMPLEX*, const IfddaIncidentRegularGridSpec* = nullptr,
    const CU_REAL* = nullptr, const CU_REAL* = nullptr,
    const CU_REAL* = nullptr);
void ifdda_paraxial_gaussian_field_cuda_execute(
    const REAL*, const REAL*, const REAL*, const int*, const COMPLEX*,
    const COMPLEX*, const REAL*, const REAL*, const REAL*, const REAL*,
    const REAL*, const REAL*, const REAL*, const REAL*, const REAL*,
    const int*, COMPLEX*, int*, bool, CUCOMPLEX*,
    const IfddaIncidentRegularGridSpec* = nullptr,
    const CU_REAL* = nullptr, const CU_REAL* = nullptr,
    const CU_REAL* = nullptr);
void ifdda_rigorous_gaussian_field_cuda_execute(
    const REAL*, const REAL*, const REAL*, const int*, const REAL*,
    const REAL*, const REAL*, const REAL*, const REAL*, const REAL*,
    const REAL*, const REAL*, const REAL*, const COMPLEX*, const REAL*,
    const int*, const int*, COMPLEX*, int*, bool, CUCOMPLEX*,
    const IfddaIncidentRegularGridSpec* = nullptr,
    const CU_REAL* = nullptr, const CU_REAL* = nullptr,
    const CU_REAL* = nullptr);
void ifdda_antenna_field_cuda_execute(
    const REAL*, const REAL*, const REAL*, const int*, const REAL*,
    const REAL*, const REAL*, const REAL*, const REAL*, const COMPLEX*,
    const REAL*, const REAL*, const REAL*, const REAL*, const REAL*,
    COMPLEX*, COMPLEX*, int*, bool, CUCOMPLEX*,
    const IfddaIncidentRegularGridSpec* = nullptr,
    const CU_REAL* = nullptr, const CU_REAL* = nullptr,
    const CU_REAL* = nullptr);

// Shared ABI declarations for the host orchestration entry points. Keep these
// here so every CUDA translation unit sees the same signatures, including the
// MinGW inverse-driver build.
#include "ifdda_micros_abi_typed.hpp"


// GPU-side variants use CUDA complex representations and distinct C symbols.
extern "C" void fouriertoimage2_cuda_(
    const CU_REAL *deltakx, const CU_REAL *deltaky, const CU_REAL *gross,
    CUCOMPLEX *image_x, CUCOMPLEX *image_y, CUCOMPLEX *image_z,
    const int *nfft, const int *nfft_half, const long long *plan_backward,
    const long long *plan_forward);
extern "C" void rytov_vector_field_cuda_(
    const CUCOMPLEX *eref, const CUCOMPLEX *delta, CUCOMPLEX *result,
    const int *dipole_count, int *nstop);

extern "C" int donothing(int data);

extern "C" void inverserigopt_with_tabdip_cuda_(
    COMPLEX *txx, COMPLEX *txy, COMPLEX *txz, COMPLEX *tyy, COMPLEX *tyz,
    COMPLEX *tzz, COMPLEX *vx, COMPLEX *vy, COMPLEX *vz, int *ntotalm,
    int *ntotal, int *lda, int *nlar, int *nmax, int *nxm, int *nym,
    int *nzm, int *nx, int *ny, int *nz, int *nx2, int *ny2, int *nxy2,
    int *nz2, int *nbsphere, int *nbsphere3, int *ndipole, int *tabdip,
    COMPLEX *xi, COMPLEX *xr, COMPLEX *work, COMPLEX *field,
    COMPLEX *incident, COMPLEX *local, COMPLEX *polar, int *niso,
    char *method, REAL *tolerance, REAL *final_residual, int *products,
    int *product_calls, int *max_iterations, int64_t *plan_xf,
    int64_t *plan_xb, int64_t *plan_yf, int64_t *plan_yb,
    int64_t *plan_zf, int64_t *plan_zb, int64_t *planf, int64_t *planb,
    int *stop, char *info, size_t method_length, size_t info_length);

// Host-side Fortran ABI entry points used by the cdmlib orchestrator.
extern "C" void calculforcefft_(
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *,
    COMPLEX *, COMPLEX *, COMPLEX *, const COMPLEX *, const int64_t *);
extern "C" void calculforcefftopt_(
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *,
    COMPLEX *, COMPLEX *, COMPLEX *, const COMPLEX *, const int64_t *);
extern "C" void derivativefield2_(
    const REAL *, const REAL *, const int *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    const COMPLEX *, const COMPLEX *, const COMPLEX *, const int *, COMPLEX *,
    const int64_t *, const int64_t *);
extern "C" void diffractefft2d_(
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, int *, const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, COMPLEX *, COMPLEX *, COMPLEX *, const COMPLEX *, int *, REAL *,
    REAL *, COMPLEX *, COMPLEX *, const int64_t *, const int64_t *, int *, char *,
    size_t);
extern "C" void computegcfft2d_(
    const int *, const REAL *, const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, const int *, const COMPLEX *, const COMPLEX *, REAL *, REAL *);
extern "C" void inverserigoptscalaire_(
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *, const COMPLEX *, const COMPLEX *,
    const COMPLEX *, const int *, const int *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *,
    COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *, COMPLEX *,
    const int *, char *, const REAL *, REAL *, int *, int *, const int *,
    const int64_t *, const int64_t *, int *, char *, size_t, size_t);

extern "C" void objetsphererandom_(
    const char *trope, const COMPLEX *eps, const COMPLEX *epsani,
    const COMPLEX *eps0, REAL *xs, REAL *ys, REAL *zs, const REAL *k0,
    REAL *aretecube, int *tabdip, const int *nnnr, const int *nmax,
    int *nbsphere, int *ndipole, int *nx, int *ny, int *nz,
    const int *nxm, const int *nym, const int *nzm, const char *methode,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso, REAL *sidex,
    REAL *sidey, REAL *sidez, REAL *xg, REAL *yg, REAL *zg, REAL *xsphe,
    REAL *ysphe, REAL *zsphe, REAL *radius, REAL *density, const int *nseed,
    const int *na, REAL *sldr, const int *nmat, const int64_t *file_id,
    const int64_t *group_iddip, char *infostr, int *nstop,
    size_t trope_len, size_t methode_len, size_t infostr_len);
extern "C" void objetsphererandomnxnynz_(
    const char *trope, const COMPLEX *eps, const COMPLEX *epsani,
    const COMPLEX *eps0, REAL *xs, REAL *ys, REAL *zs, const REAL *k0,
    REAL *aretecube, int *tabdip, const int *nnnr, const int *nmax,
    int *nbsphere, int *ndipole, int *nx, int *ny, int *nz,
    const int *nxm, const int *nym, const int *nzm, const int *nxmp,
    const int *nymp, const int *nzmp, const char *methode,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    REAL *sidex, REAL *sidey, REAL *sidez, REAL *xg, REAL *yg, REAL *zg,
    REAL *xsphe, REAL *ysphe, REAL *zsphe, REAL *radius, REAL *density,
    const int *nseed, const int *na, REAL *sldr, const int *nmat,
    const int64_t *file_id, const int64_t *group_iddip, char *infostr,
    int *nstop, size_t trope_len, size_t methode_len, size_t infostr_len);

// Geometry wrappers. All arguments that point to caller-owned arrays are CPU
// ABI types; CUDA device buffers remain internal to the implementations.
extern "C" void objetsphereconcentric_(
    const char *, const COMPLEX *, const COMPLEX *, const int *, const int *,
    REAL *, REAL *, REAL *, REAL *, const COMPLEX *, REAL *, REAL *, REAL *,
    const REAL *, REAL *, int *, int *, const int *, const int *, int *, int *,
    int *, int *, int *, const char *, int *, COMPLEX *, COMPLEX *, const int *,
    REAL *, const int *, const int64_t *, const int64_t *, char *, int *,
    size_t, size_t, size_t);
extern "C" void objetpara_(
    const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *, REAL *,
    REAL *, REAL *, const REAL *, REAL *, int *, const int *, const int *, int *,
    int *, int *, int *, int *, const int *, const int *, const int *, const char *,
    COMPLEX *, COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *, REAL *,
    REAL *, REAL *, REAL *, REAL *, const int *, REAL *, const int *,
    const int64_t *, const int64_t *, char *, int *, size_t, size_t, size_t);
extern "C" void objetparanxnynz_(
    const char *, COMPLEX *, COMPLEX *, COMPLEX *, REAL *, REAL *, REAL *,
    const REAL *, REAL *, int *, const int *, const int *, int *, int *, int *,
    int *, int *, const int *, const int *, const int *, const int *, const int *,
    const int *, const char *, COMPLEX *, COMPLEX *, const int *, REAL *, REAL *,
    REAL *, REAL *, REAL *, REAL *, const int *, REAL *, const int *,
    const int64_t *, const int64_t *, char *, int *, size_t, size_t, size_t);
extern "C" void objetnspheres_(
    const char *, const COMPLEX *, const COMPLEX *, const int *, const int *,
    REAL *, REAL *, REAL *, REAL *, const COMPLEX *, REAL *, REAL *, REAL *,
    const REAL *, REAL *, int *, int *, const int *, const int *, int *, int *,
    int *, int *, int *, const char *, const int *, COMPLEX *, COMPLEX *,
    const int *, REAL *, const int *, const int64_t *, const int64_t *,
    char *, int *, size_t, size_t, size_t);
extern "C" void objetsphererandommulti_(
    const char *, COMPLEX *, COMPLEX *, const int *, const int *, REAL *, REAL *,
    REAL *, REAL *, const COMPLEX *, REAL *, REAL *, REAL *, const REAL *, REAL *,
    int *, const int *, const int *, int *, int *, int *, int *, int *, const int *,
    const int *, const int *, const char *, COMPLEX *, COMPLEX *, const int *,
    REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, const int *, int *,
    REAL *, const int *, const int64_t *, const int64_t *, char *, int *,
    size_t, size_t, size_t);
extern "C" void objetsphereinhomo_(
    COMPLEX *, const COMPLEX *, REAL *, REAL *, int *, REAL *, REAL *, REAL *,
    const REAL *, REAL *, int *, int *, int *, const int *, const int *, int *,
    int *, int *, int *, int *, const char *, const int *, COMPLEX *, COMPLEX *,
    const int *, REAL *, REAL *, REAL *, const int *, COMPLEX *, REAL *,
    const int *, const int64_t *, const int64_t *, char *, int *, size_t, size_t);
extern "C" void objetparainhomo_(
    COMPLEX *, COMPLEX *, REAL *, REAL *, int *, REAL *, REAL *, REAL *,
    const REAL *, REAL *, int *, int *, int *, const int *, const int *, int *,
    int *, int *, int *, int *, const int *, const int *, const int *, const char *,
    COMPLEX *, COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *, REAL *,
    REAL *, REAL *, const REAL *, const int *, COMPLEX *, const int *, REAL *,
    const int *, const int64_t *, const int64_t *, char *, int *, size_t, size_t);
extern "C" void objetcube_(
    const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *, REAL *,
    REAL *, REAL *, const REAL *, REAL *, int *, const int *, const int *, int *,
    int *, int *, int *, int *, const char *, COMPLEX *, COMPLEX *, const int *,
    REAL *, REAL *, REAL *, REAL *, const REAL *, const int *,
    const int64_t *, const int64_t *, char *, int *, size_t, size_t, size_t);
extern "C" void objetellipse_(
    const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *, REAL *,
    REAL *, REAL *, const REAL *, REAL *, int *, const int *, const int *, int *,
    int *, int *, int *, int *, const int *, const int *, const int *, const char *,
    const int *, COMPLEX *, COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *,
    REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, const int *, const int64_t *,
    const int64_t *, char *, int *, size_t, size_t, size_t);
extern "C" void objetparainhomonxnynz_(
    COMPLEX *, COMPLEX *, REAL *, REAL *, REAL *, const REAL *, REAL *, int *,
    const int *, const int *, int *, int *, int *, int *, int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const char *,
    COMPLEX *, COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *, REAL *,
    REAL *, REAL *, const REAL *, const int *, COMPLEX *, const int *, REAL *,
    const int *, const int64_t *, const int64_t *, char *, int *, size_t, size_t);
extern "C" void objetsphere_(
    const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *, REAL *,
    REAL *, int *, REAL *, REAL *, REAL *, const REAL *, REAL *, int *, int *,
    int *, const int *, const int *, int *, int *, int *, int *, int *, const char *,
    const int *, COMPLEX *, COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *,
    REAL *, const int *, const int64_t *, const int64_t *, char *, int *,
    size_t, size_t, size_t);
// The sphere and cube both enter object_all; other geometry entry points are
// migrated independently because their original Fortran signatures differ.
extern "C" void ifdda_object_all_sphere_cuda(
    int, const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *, REAL *,
    REAL *, int *, REAL *, REAL *, REAL *, const REAL *, REAL *, int *, int *,
    int *, const int *, const int *, int *, int *, int *, int *, int *, const char *,
    const int *, COMPLEX *, COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *,
    REAL *, const int *, const int64_t *, const int64_t *, char *, int *,
    size_t, size_t, size_t);
extern "C" void ifdda_object_all_cube_cuda(
    int, const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *,
    REAL *, REAL *, REAL *, const REAL *, REAL *, int *, const int *,
    const int *, int *, int *, int *, int *, int *, const char *,
    COMPLEX *, COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *,
    const REAL *, const int *, const int64_t *, const int64_t *,
    char *, int *, size_t, size_t, size_t);
extern "C" void ifdda_object_all_cuboid2_cuda(
    int, const char *, COMPLEX *, COMPLEX *, COMPLEX *, REAL *, REAL *,
    REAL *, const REAL *, REAL *, int *, const int *, const int *, int *,
    int *, int *, int *, int *, const int *, const int *, const int *,
    const int *, const int *, const int *, const char *, COMPLEX *,
    COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *, REAL *,
    REAL *, const int *, REAL *, const int *, const int64_t *,
    const int64_t *, char *, int *, size_t, size_t, size_t);
extern "C" void ifdda_object_all_cuboid1_cuda(
    int, const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *,
    REAL *, REAL *, REAL *, const REAL *, REAL *, int *, const int *,
    const int *, int *, int *, int *, int *, int *, const int *,
    const int *, const int *, const char *, COMPLEX *, COMPLEX *,
    const int *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *,
    REAL *, REAL *, REAL *, const int *, REAL *, const int *,
    const int64_t *, const int64_t *, char *, int *, size_t, size_t, size_t);
extern "C" void ifdda_object_all_ellipsoid_cuda(
    int, const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *,
    REAL *, REAL *, REAL *, const REAL *, REAL *, int *, const int *,
    const int *, int *, int *, int *, int *, int *, const int *,
    const int *, const int *, const char *, const int *, COMPLEX *,
    COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *, REAL *,
    REAL *, REAL *, REAL *, REAL *, REAL *, const int *,
    const int64_t *, const int64_t *, char *, int *, size_t, size_t, size_t);
extern "C" void ifdda_object_all_cylinder_cuda(
    int, const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *,
    REAL *, REAL *, REAL *, const REAL *, REAL *, int *, const int *,
    const int *, int *, int *, int *, int *, int *, const int *,
    const int *, const int *, const char *, const int *, COMPLEX *,
    COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *, REAL *,
    REAL *, REAL *, REAL *, REAL *, const int *,
    const int64_t *, const int64_t *, char *, int *, size_t, size_t, size_t);
extern "C" void ifdda_object_all_nspheres_cuda(
    int geometry, 
    const char *trope, const COMPLEX *epss, const COMPLEX *epsanis,
    const int *numbersphere, const int *numberspheremax, CU_REAL *xg, CU_REAL *yg,
    CU_REAL *zg, CU_REAL *rayons, const COMPLEX *eps0, CU_REAL *xs,
    CU_REAL *ys, CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube, int *tabdip,
    int *tabnbs, const int *nnnr, const int *nmax, int *nbsphere, int *ndipole,
    int *nx, int *ny, int *nz, const char *methode, const int *na,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip, char *infostr,
    int *nstop, std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length);
extern "C" void ifdda_object_all_concentricsphere_cuda(
    int geometry, 
    const char *trope, const COMPLEX *epss, const COMPLEX *epsanis,
    const int *numbersphere, const int *numberspheremax, CU_REAL *xg, CU_REAL *yg,
    CU_REAL *zg, CU_REAL *rayons, const COMPLEX *eps0, CU_REAL *xs,
    CU_REAL *ys, CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube, int *tabdip,
    int *tabnbs, const int *nnnr, const int *nmax, int *nbsphere, int *ndipole,
    int *nx, int *ny, int *nz, const char *methode, int *na,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip, char *infostr,
    int *nstop, std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length);
extern "C" void ifdda_object_all_randomsphere_cuda(
    int geometry,
    const char *trope, const COMPLEX *eps,
    const COMPLEX *epsani, const COMPLEX *eps0, CU_REAL *xs,
    CU_REAL *ys, CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube,
    int *tabdip, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const int *nxm,
    const int *nym, const int *nzm, const char *methode,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez, CU_REAL *xg,
    CU_REAL *yg, CU_REAL *zg, CU_REAL *xsphe, CU_REAL *ysphe,
    CU_REAL *zsphe, CU_REAL *radius, CU_REAL *density, const int *nseed,
    const int *na, CU_REAL *sldr, const int *nmat, const int64_t *file_id,
    const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length);
extern "C" void ifdda_object_all_randommulti_cuda(
    int geometry, 
    const char *trope, COMPLEX *epss, COMPLEX *epsanis,
    const int *numbersphere, const int *numberspheremax, CU_REAL *xg,
    CU_REAL *yg, CU_REAL *zg, CU_REAL *rayons,
    const COMPLEX *eps0, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, const int *nnnr,
    const int *nmax, int *nbsphere, int *ndipole, int *nx, int *ny, int *nz,
    const int *nxm, const int *nym, const int *nzm, const char *methode,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez, CU_REAL *xsphe,
    CU_REAL *ysphe, CU_REAL *zsphe, CU_REAL *density, const int *nseed,
    int *na, CU_REAL *sldr, const int *nmat, const int64_t *file_id,
    const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t trope_length, std::size_t methode_length,
    std::size_t infostr_length);
extern "C" void ifdda_object_all_inhomocuboid2_cuda(
    int geometry, 
    COMPLEX *eps, COMPLEX *eps0, CU_REAL *xs, CU_REAL *ys,
    CU_REAL *zs, const CU_REAL *k0, CU_REAL *aretecube, int *tabdip,
    const int *nnnr, const int *nmax, int *nbsphere, int *ndipole,
    int *nx, int *ny, int *nz, const int *nxm, const int *nym,
    const int *nzm, const int *nxmp, const int *nymp, const int *nzmp,
    const char *methode, COMPLEX *epsilon, COMPLEX *polarisa,
    const int *niso, CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez,
    CU_REAL *xg, CU_REAL *yg, CU_REAL *zg, CU_REAL *lc, const CU_REAL *hc,
    const int *ng, COMPLEX *epsb, const int *na, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip,
    char *infostr, int *nstop,
    std::size_t methode_length, std::size_t infostr_length);
extern "C" void ifdda_object_all_inhomosphere_cuda(
    int geometry, 
    COMPLEX *eps, const COMPLEX *eps0, CU_REAL *lnrbc,
    CU_REAL *lcos, int *ntypeabc, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, int *tabmulti,
    int *pmlwrk, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const char *methode,
    const int *na, COMPLEX *epsilon, COMPLEX *polarisa,
    const int *niso, CU_REAL *rayon, CU_REAL *lc, CU_REAL *hc,
    const int *ng, COMPLEX *epsb, CU_REAL *sldr, const int *nmat,
    const int64_t *file_id, const int64_t *group_iddip, char *infostr, int *nstop,
    std::size_t methode_length, std::size_t infostr_length);
extern "C" void ifdda_object_all_inhomocuboid1_cuda(
    int geometry, 
    COMPLEX *eps, COMPLEX *eps0, CU_REAL *lnrbc,
    CU_REAL *lcos, int *ntypeabc, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
    const CU_REAL *k0, CU_REAL *aretecube, int *tabdip, int *tabmulti,
    int *pmlwrk, const int *nnnr, const int *nmax, int *nbsphere,
    int *ndipole, int *nx, int *ny, int *nz, const int *nxm,
    const int *nym, const int *nzm, const char *methode,
    COMPLEX *epsilon, COMPLEX *polarisa, const int *niso,
    CU_REAL *sidex, CU_REAL *sidey, CU_REAL *sidez, CU_REAL *xg,
    CU_REAL *yg, CU_REAL *zg, CU_REAL *lc, const CU_REAL *hc,
    const int *ng, COMPLEX *epsb, const int *na, CU_REAL *sldr,
    const int *nmat, const int64_t *file_id, const int64_t *group_iddip,
    char *infostr, int *nstop, std::size_t methode_length,
    std::size_t infostr_length);
extern "C" void ifdda_object_all_arbitrary_cuda(
    int geometry, 
    const char* object_type, COMPLEX* epsilon_value, COMPLEX* epsilon_anisotropic,
    const COMPLEX* epsilon_background, CU_REAL* xs, CU_REAL* ys, CU_REAL* zs,
    const CU_REAL* k0, CU_REAL* cell_size, int* dipole_map, const int* random_count,
    const int* nmax, int* dipole_count, int* ndipole, int* nx, int* ny, int* nz,
    const char* polarizability_method, const char* filename, int* particle_count,
    COMPLEX* epsilon, COMPLEX* polarizability, const int* isotropy,
    CU_REAL* sldr, const int* output_mode, const int64_t* file_id, const int64_t* group_id,
    char* infostr, int* nstop, std::size_t object_type_length,
    std::size_t method_length, std::size_t filename_length,
    std::size_t infostr_length);
extern "C" void objetsphere_legacy_cuda_(
    const char *, const COMPLEX *, const COMPLEX *, const COMPLEX *, REAL *,
    REAL *, int *, REAL *, REAL *, REAL *, const REAL *, REAL *, int *, int *,
    int *, const int *, const int *, int *, int *, int *, int *, int *, const char *,
    const int *, COMPLEX *, COMPLEX *, const int *, REAL *, REAL *, REAL *, REAL *,
    REAL *, const int *, const int64_t *, const int64_t *, char *, int *,
    size_t, size_t, size_t);

// Host-side illumination entry points shared by cdmlib and microscopy callers.
// COMPLEX arrays here are CPU-owned; device buffers stay CUCOMPLEX* inside the
// CUDA implementation and its explicitly device-oriented entry points.
extern "C" void beampropagation_(
    const REAL *, const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, const COMPLEX *, const REAL *, const REAL *, const REAL *,
    const REAL *, const REAL *, const REAL *, const REAL *, const char *,
    const COMPLEX *, const int *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    COMPLEX *, COMPLEX *, COMPLEX *, const COMPLEX *, COMPLEX *, COMPLEX *,
    const IFDDA_PLAN_INT64 *, const IFDDA_PLAN_INT64 *, int *, char *, const int *, size_t, size_t);
extern "C" void beampropagationmacro_(
    const REAL *, const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, const COMPLEX *, const REAL *, const REAL *, const REAL *,
    const REAL *, const REAL *, const REAL *, const REAL *, const char *,
    const COMPLEX *, const int *, const int *, const int *, const int *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    COMPLEX *, COMPLEX *, COMPLEX *, const COMPLEX *, COMPLEX *, COMPLEX *,
    const IFDDA_PLAN_INT64 *, const IFDDA_PLAN_INT64 *, int *, char *, size_t, size_t);
extern "C" void derivechamp_(const int *, const int *, const int *, const int *,
                              const REAL *, const COMPLEX *, const int *, COMPLEX *);
extern "C" void incidentarbitraryder1_(const int *, const REAL *, const REAL *,
                                        const REAL *, const char *, COMPLEX *, size_t);
extern "C" void incidentarbitrarydercreate_(const char *, size_t);
extern "C" void ifdda_prepare_speckle_phases_(
    const int *, const int *, const REAL *, const REAL *, const REAL *,
    const int *, const int *, COMPLEX *, COMPLEX *);
extern "C" int ifdda_prepare_speckle_phases_device_(
    const int *, const int *, const CU_REAL *, const CU_REAL *, const CU_REAL *,
    const int *, const int *, const CUCOMPLEX **, const CUCOMPLEX **);
extern "C" void ifdda_rigorous_gaussian_field_cuda_(
    const REAL *, const REAL *, const REAL *, const int *,
    const REAL *, const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, const REAL *, const REAL *, const REAL *, const COMPLEX *,
    const REAL *, const int *, const int *, COMPLEX *, int *);
extern "C" void ifdda_paraxial_gaussian_field_cuda_(
    const REAL *, const REAL *, const REAL *, const int *, const COMPLEX *,
    const COMPLEX *, const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, const REAL *, const REAL *, const REAL *, const REAL *,
    const int *, COMPLEX *, int *);
extern "C" void ifdda_plane_wave_field_cuda_(
    const REAL *, const REAL *, const REAL *, const int *, const REAL *,
    const COMPLEX *, const COMPLEX *, const REAL *, const REAL *, const REAL *,
    const REAL *, const int *, COMPLEX *, int *);
extern "C" void ifdda_multi_plane_wave_field_cuda_(
    const REAL *, const REAL *, const REAL *, const int *, const REAL *,
    const COMPLEX *, const COMPLEX *, const REAL *, const REAL *, const REAL *,
    const REAL *, const int *, COMPLEX *, int *);
// Batched derivative wrappers exchange host COMPLEX arrays with Fortran;
// device field buffers remain CUCOMPLEX inside their implementations.
extern "C" void ifdda_plane_wave_derivatives_cuda_(
    const REAL *, const REAL *, const REAL *, const int *, const int *,
    const int *, const REAL *, const COMPLEX *, const REAL *, const REAL *,
    const REAL *, const REAL *, const int *, COMPLEX *, int *);
extern "C" void ifdda_multi_plane_wave_derivatives_cuda_(
    const REAL *, const REAL *, const REAL *, const int *, const int *,
    const int *, const REAL *, const COMPLEX *, const REAL *, const REAL *,
    const REAL *, const int *, COMPLEX *, int *);
extern "C" void ifdda_paraxial_gaussian_derivatives_cuda_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *tabdip,
    const int *coordinate_count, const int *dipole_count, const REAL *x0,
    const REAL *y0, const REAL *z0, const REAL *theta_degrees,
    const REAL *phi_degrees, const REAL *waist, const REAL *k0,
    const REAL *ss, const REAL *pp, const COMPLEX *amplitude,
    const int *circular, COMPLEX *output, int *status_out);
extern "C" void ifdda_rigorous_gaussian_derivatives_cuda_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *tabdip,
    const int *coordinate_count, const int *dipole_count, const REAL *x0,
    const REAL *y0, const REAL *z0, const REAL *theta, const REAL *phi,
    const REAL *waist, const REAL *k0, const REAL *ss, const REAL *pp,
    const COMPLEX *amplitude, const REAL *tolerance, const int *nloin,
    const int *circular, COMPLEX *output, int *status_out);
extern "C" void ifdda_antenna_derivatives_cuda_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *tabdip,
    const int *coordinate_count, const int *dipole_count, const REAL *xdip,
    const REAL *ydip, const REAL *zdip, const REAL *theta, const REAL *phi,
    const REAL *aretecube, const REAL *k0, const COMPLEX *amplitude,
    COMPLEX *output, int *status_out);
extern "C" void dipoleincp_batch_(
    const REAL *, const REAL *, const REAL *, const COMPLEX *, const REAL *,
    const REAL *, const REAL *, const int *, const REAL *, const REAL *,
    COMPLEX *, int *, char *, size_t);
extern "C" void ifdda_incident_arbitrary_cuda_(
    const REAL *, const REAL *, const REAL *, const REAL *, COMPLEX *,
    const int *, const int *, const int *, const int *, const int *, int *,
    const char *, char *);
extern "C" void ifdda_speckle_field_cuda_(
    const int *, const int *, const int *, const int *, const int *,
    const REAL *, const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, const REAL *, const REAL *, COMPLEX *, const COMPLEX *,
    const COMPLEX *, const REAL *, const REAL *, const REAL *, const int *,
    COMPLEX *, REAL *, int *);
extern "C" void ifdda_confocal_field_cuda_(
    const int *, const int *, const int *, const int *, const int *,
    const REAL *, const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, const REAL *, const REAL *, COMPLEX *, const REAL *,
    const REAL *, const REAL *, const int *, COMPLEX *, REAL *, int *);
extern "C" void ifdda_gaussianfft_field_cuda_(
    const REAL *, const REAL *, const REAL *, const REAL *, const REAL *,
    const REAL *, const COMPLEX *, const REAL *, const REAL *, const REAL *,
    const REAL *, const REAL *, const REAL *, const REAL *, const char *,
    const int *, const int *, const int *, const int *, const int *, const int *,
    const int *, const int *, const int *, COMPLEX *, COMPLEX *, COMPLEX *,
    COMPLEX *, const REAL *, const int64_t *, const int64_t *, int *, char *,
    size_t, size_t);
extern "C" void gaussianchampfft_(const REAL *, const REAL *, const REAL *,
                                   const REAL *, const REAL *, const REAL *,
                                   const COMPLEX *, const REAL *, const REAL *,
                                   const REAL *, const REAL *, const REAL *,
                                   const REAL *, const REAL *, const char *,
                                   const int *, const int *, const int *, const int *,
                                   const int *, const int *, const int *, const int *,
                                   const int *, COMPLEX *, COMPLEX *, COMPLEX *,
                                   COMPLEX *, const REAL *, const int64_t *,
                                   const int64_t *, int *, char *, size_t, size_t);
extern "C" void ondeplanemulti_(REAL *, REAL *, REAL *, REAL *,
                                const COMPLEX *, const REAL *, const REAL *,
                                const REAL *, const REAL *, const int *,
                                COMPLEX *, COMPLEX *, COMPLEX *, int *, char *, size_t);
extern "C" void ondeplanekxky_(
    const REAL *, const REAL *, const REAL *, const REAL *, const COMPLEX *,
    const REAL *, const REAL *, const REAL *, const REAL *, COMPLEX *, COMPLEX *,
    COMPLEX *, int *, char *, size_t);
extern "C" int ifdda_widefield_generate_green_c99(
    const void *const host_tensor_keys[6], int nx, int ny, int nz,
    int nxm, int nym, int nzm, int nx2, int ny2, int nz2,
    CU_REAL cell_size, CU_REAL k0, int nquad, GreenTensorDevicePointers *device_green);
extern "C" int ifdda_widefield_prepare_grid_c99(
    const REAL *dipole_x, const REAL *dipole_y, const REAL *dipole_z,
    int dipole_count, int nx, int ny, int nz, int nxmp, int nymp, int nzmp,
    CU_REAL xmin, CU_REAL ymin, CU_REAL zmin, CU_REAL cell_size,
    CU_REAL comparison_tolerance, cuda_data::WideFieldDeviceBuffers *buffers);
extern "C" int ifdda_widefield_scatter_dipoles_c99(
    const COMPLEX *dipole_scattered, const COMPLEX *dipole_incident,
    const COMPLEX *dipole_epsilon, int dipole_count, int grid_count,
    int isotropy, int preserve_incident_field,
    cuda_data::WideFieldDeviceBuffers *buffers);
extern "C" int ifdda_widefield_scatter_dipoles_host_c99(
    const COMPLEX *dipole_scattered, const COMPLEX *dipole_incident,
    const COMPLEX *fortran_epsilon, int nmax, int dipole_count,
    int grid_count, int isotropy, int preserve_incident_field,
    cuda_data::WideFieldDeviceBuffers *buffers);
extern "C" int ifdda_widefield_scatter_dipoles_device_c99(
    const CUCOMPLEX *dipole_scattered_device, const COMPLEX *fortran_epsilon,
    int nmax, int dipole_count, int grid_count, int isotropy,
    cuda_data::WideFieldDeviceBuffers *buffers);
extern "C" int ifdda_widefield_plane_wave_c99(
    int grid_count, CU_REAL kx, CU_REAL ky, CU_REAL kz,
    COMPLEX ex0, COMPLEX ey0, COMPLEX ez0, int dipole_count,
    cuda_data::WideFieldDeviceBuffers *buffers);
extern "C" const CUCOMPLEX *diffractefft2dlens_get_resident_output();
extern "C" void diffractefft2dlens_set_resident_input(const CUCOMPLEX *device_input);
extern "C" void ifdda_local_macro_batch_cuda_(
    const CUCOMPLEX *, const CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *, CUCOMPLEX *,
    CU_REAL *, const int *, const int *, const int *, const CU_REAL *,
    const CU_REAL *, const int *, int *);

// Optional host orchestration declarations are enabled only by cdmlib.cpp.
// CPU illumination ABI shared by CUDA definitions and orchestration callers.
extern "C" void gaussianchamp_(const REAL *x,const REAL *y,const REAL *z,
    const REAL *x0,const REAL *y0,const REAL *z0,const REAL *theta,
    const REAL *phi,const REAL *w0,const REAL *k0,const REAL *ss,
    const REAL *pp,const COMPLEX *e0,COMPLEX *ex,
    COMPLEX *ey,COMPLEX *ez,const REAL *tol,const int *nloin,
    int *nstop,char *infostr,size_t infostr_len);
extern "C" void gaussianchampcirc_(const REAL *x,const REAL *y,const REAL *z,
    const REAL *x0,const REAL *y0,const REAL *z0,const REAL *theta,
    const REAL *phi,const REAL *w0,const REAL *k0,const REAL *ss,
    const COMPLEX *e0,COMPLEX *ex,COMPLEX *ey,
    COMPLEX *ez,const REAL *tol,const int *nloin,int *nstop,
    char *infostr,size_t infostr_len);
extern "C" void gaussianparacirc_(const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const COMPLEX *,
        const COMPLEX *,COMPLEX *,COMPLEX *,COMPLEX *,
        int *,char *,size_t);
extern "C" void gaussianparacircd_(const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const COMPLEX *,
        COMPLEX *,int *,char *,size_t);
extern "C" void gaussianparalinear_(const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,
        const COMPLEX *,const COMPLEX *,COMPLEX *,
        COMPLEX *,COMPLEX *,int *,char *,size_t);
extern "C" void gaussianparalineard_(const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,
        const COMPLEX *,COMPLEX *,int *,char *,size_t);
extern "C" void ondecirce_(const REAL *,const REAL *,const REAL *,
        const REAL *,const COMPLEX *,const REAL *,const REAL *,
        const REAL *,COMPLEX *,COMPLEX *,COMPLEX *);
extern "C" void ondecirced_(const REAL *,const REAL *,const REAL *,
        const REAL *,const COMPLEX *,const REAL *,const REAL *,
        const REAL *,const int *,COMPLEX [3][3]);
extern "C" void ondeplane_(REAL *,REAL *,REAL *,REAL *,
        const COMPLEX *,const COMPLEX *,REAL *,REAL *,
        REAL *,REAL *,COMPLEX *,COMPLEX *,COMPLEX *,
        int *,char *,size_t);
extern "C" void ondeplaned_(REAL *,REAL *,REAL *,REAL *,
        const COMPLEX *,REAL *,REAL *,REAL *,REAL *,const int *,
        COMPLEX [3][3]);
extern "C" void ondeplanedmulti_(REAL *,REAL *,REAL *,REAL *,
        const COMPLEX [10],const REAL [10],const REAL [10],
        const REAL [10],const REAL [10],const int *,const int *,
        COMPLEX [3][3],int *,char *,size_t);

// Cross-unit device interfaces for multilayer illumination Green functions.
extern "C" int ifdda_generate_multilayer_scalar_green_cuda_(COMPLEX *cpu_green, const int *nfft2d,
                                                            const CU_REAL *aretecube,
                                                            const CU_REAL *k0);
extern "C" int ifdda_generate_multilayer_projected_green_cuda_(
    COMPLEX *cpu_green, const int *nfft2d, const CU_REAL *aretecube, const CU_REAL *k0,
    const COMPLEX *ux, const COMPLEX *uy, const COMPLEX *uz);
extern "C" int ifdda_multilayer_scalar_initialize_plane_cuda_(
    COMPLEX *cpu_image, COMPLEX *cpu_local_field, const COMPLEX *cpu_incident_dipoles,
    const CU_REAL *cpu_xs, const CU_REAL *cpu_ys, const CU_REAL *cpu_zs, const int *ndipole,
    const int *nfft2d, const int *nmax, const int *kkm, const int *jjm, const CU_REAL *xmin,
    const CU_REAL *ymin, const CU_REAL *first_z, const CU_REAL *aretecube, const CU_REAL *k0,
    const COMPLEX *E0, const COMPLEX *eps0, const CU_REAL *w0, const CU_REAL *ss,
    const CU_REAL *pp, const CU_REAL *theta, const CU_REAL *phi, const int *beam_type,
    const int *conjugate_projection, const COMPLEX *ux, const COMPLEX *uy,
    const COMPLEX *uz);
extern "C" int ifdda_multilayer_scalar_initialize_rigorous_plane_cuda_(
    COMPLEX *cpu_image, COMPLEX *cpu_local_field, const COMPLEX *cpu_incident_dipoles,
    const CU_REAL *cpu_xs, const CU_REAL *cpu_ys, const CU_REAL *cpu_zs, const int *ndipole,
    const int *nfft2d, const int *nmax, const int *kkm, const int *jjm, const CU_REAL *xmin,
    const CU_REAL *ymin, const CU_REAL *first_z, const CU_REAL *aretecube, const CU_REAL *k0,
    const COMPLEX *E0, const CU_REAL *ss, const CU_REAL *pp, const CU_REAL *theta,
    const CU_REAL *phi, const CU_REAL *x0, const CU_REAL *y0, const CU_REAL *z0,
    const CU_REAL *w0, const CU_REAL *tolerance, const int *circular,
    const int *conjugate_projection, const COMPLEX *ux, const COMPLEX *uy,
    const COMPLEX *uz);
extern "C" int ifdda_multilayerbornsca_layers_cuda_(
    COMPLEX *cpu_image, COMPLEX *cpu_local_field, const COMPLEX *cpu_polarizability,
    const int *cpu_plane_map, const COMPLEX *cpu_green_key, const int *nfft2d, const int *nx,
    const int *ny, const int *nz, const int *nmax, const int *niso, const int *reflection,
    const CU_REAL *deltakx, const CU_REAL *deltaky, const CU_REAL *aretecube, const CU_REAL *k0,
    const COMPLEX *ux, const COMPLEX *uy, const COMPLEX *uz);
extern "C" int ifdda_generate_multilayer_dyadic_green_cuda_(
    COMPLEX *cpu_xx, COMPLEX *cpu_xy, COMPLEX *cpu_xz, COMPLEX *cpu_yy, COMPLEX *cpu_yz,
    COMPLEX *cpu_zz, const int *nfft2d, const CU_REAL *aretecube, const CU_REAL *k0);
extern "C" int ifdda_multilayer_vector_initialize_plane_cuda_(
    COMPLEX *cpu_image_x, COMPLEX *cpu_image_y, COMPLEX *cpu_image_z,
    COMPLEX *cpu_local_field, const COMPLEX *cpu_incident_dipoles, const CU_REAL *cpu_xs,
    const CU_REAL *cpu_ys, const CU_REAL *cpu_zs, const int *ndipole, const int *nfft2d,
    const int *nmax, const int *kkm, const int *jjm, const CU_REAL *xmin, const CU_REAL *ymin,
    const CU_REAL *first_z, const CU_REAL *aretecube, const CU_REAL *k0, const COMPLEX *E0,
    const COMPLEX *eps0, const CU_REAL *w0, const CU_REAL *ss, const CU_REAL *pp,
    const CU_REAL *theta, const CU_REAL *phi, const int *beam_type);
extern "C" int ifdda_multilayer_vector_initialize_rigorous_plane_cuda_(
    COMPLEX *cpu_image_x, COMPLEX *cpu_image_y, COMPLEX *cpu_image_z,
    COMPLEX *cpu_local_field, const COMPLEX *cpu_incident_dipoles,
    const CU_REAL *cpu_xs, const CU_REAL *cpu_ys, const CU_REAL *cpu_zs,
    const int *ndipole, const int *nfft2d, const int *nmax, const int *kkm,
    const int *jjm, const CU_REAL *xmin, const CU_REAL *ymin, const CU_REAL *first_z,
    const CU_REAL *aretecube, const CU_REAL *k0, const COMPLEX *E0, const CU_REAL *ss,
    const CU_REAL *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *x0,
    const CU_REAL *y0, const CU_REAL *z0, const CU_REAL *w0, const CU_REAL *tolerance,
    const int *circular);
extern "C" int ifdda_multilayerbornvect_layers_cuda_(
    COMPLEX *cpu_ix, COMPLEX *cpu_iy, COMPLEX *cpu_iz, COMPLEX *cpu_local,
    const COMPLEX *cpu_alpha, const int *cpu_map, COMPLEX *cpu_gxx, COMPLEX *cpu_gxy,
    COMPLEX *cpu_gxz, COMPLEX *cpu_gyy, COMPLEX *cpu_gyz, COMPLEX *cpu_gzz, const int *nfft,
    const int *nx, const int *ny, const int *nz, const int *nmax, const int *niso,
    const int *reflection,
    const CU_REAL *dkx, const CU_REAL *dky, const CU_REAL *spacing, const CU_REAL *k0);

extern "C" void diffractefft2dlens_(const int *,const int *,const int *,
 const int *,const int *,const int *,const int *,int *,const REAL *,
 const REAL *,const REAL *,const REAL *,const REAL *,COMPLEX *,
 COMPLEX *,COMPLEX *,const COMPLEX *,int *,REAL *,REAL *,
 COMPLEX *,const REAL *,const int *,const IFDDA_PLAN_INT64 *,const IFDDA_PLAN_INT64 *,
 int *,char *,size_t);
extern "C" void inverserigscalaire_(const COMPLEX *vectx,COMPLEX *vecty,
        const int *tabdip,COMPLEX *ffscal,COMPLEX *fflocscal,
        const COMPLEX *uincx,const COMPLEX *uincy,
        const COMPLEX *uincz,const int *ntotalm,const int *ntotal,
        const int *ldabi,const int *nlar,const int *nmax,const int *ndipole,
        const int *nxm,const int *nym,const int *nzm,const int *nx,const int *ny,
        const int *nz,const int *nx2,const int *ny2,const int *nxy2,const int *nz2,
        const int *nbsphere,COMPLEX *xi,COMPLEX *xr,
        COMPLEX *wrk,COMPLEX *ff,COMPLEX *ff0,
        COMPLEX *ffloc,COMPLEX *polarisa,const int *niso,
        char *methodeit,const REAL *tol,REAL *tol1,int *nloop,int *ncompte,
        const int *nlim,const int64_t *planf,const int64_t *planb,int *nstop,
        char *infostr,size_t methodeit_len,size_t infostr_len);

extern "C" void ifdda_beampropagation_initialize_plane_cuda_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CUCOMPLEX *epsilon,
    const CUCOMPLEX *incident, CUCOMPLEX *image_x, CUCOMPLEX *image_y, CUCOMPLEX *image_z,
    int *first_plane_map, CUCOMPLEX *local_field, const int *nx, const int *ny, const int *nfft2d,
    const int *nmax, const int *niso, const int *ndipole, const int *kkm, const int *jjm,
    const CU_REAL *xmin, const CU_REAL *ymin, const CU_REAL *first_z, const CU_REAL *aretecube,
    const CU_REAL *k0, const CUCOMPLEX *E0, const CU_REAL *ss, const CU_REAL *pp,
    const CU_REAL *theta_degrees, const CU_REAL *phi_degrees, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const CU_REAL *w0, const int *beam_type,
    int *nstop);

extern "C" void ifdda_beampropagation_propagate_cuda_(
    CUCOMPLEX *image_x, CUCOMPLEX *image_y, CUCOMPLEX *image_z, const int *first_plane_map,
    const CUCOMPLEX *epsilon, CUCOMPLEX *local_field, const int *nx, const int *ny, const int *nz,
    const int *nfft2d, const int *nmax, const int *niso, const int *ndipole,
    const CU_REAL *aretecube, const CU_REAL *k0, const int *convert_local_to_macro, int *nstop);

extern "C" int ifdda_generate_scalar_green_cuda_(const int *nx, const int *ny, const int *nz,
                                                 const int *nx2, const int *ny2, const int *nz2,
                                                 const CU_REAL *aretecube, const CU_REAL *k0,
                                                 const CUCOMPLEX *uincx, const CUCOMPLEX *uincy,
                                                 const CUCOMPLEX *uincz, const int *ntotalm,
                                                 const int *nrig, const COMPLEX *cpu_green_key);

extern "C" void ifdda_derive_incident_field_cuda_(
    const CUCOMPLEX* field, const int* nx, const int* ny, const int* nz, const int* nmax,
    const CU_REAL* spacing, const int* axis, CUCOMPLEX* derivative, int* status_out);

extern "C" int ifdda_quicklens_direct_cuda_(
    const COMPLEX* field, const REAL* xs, const REAL* ys, const REAL* zs,
    const COMPLEX* incident, int dipole_count, int nfft, int half, int imax,
    REAL dkx, REAL k0, REAL na, int side, REAL zlens,
    COMPLEX* fx, COMPLEX* fy, COMPLEX* fz,
    COMPLEX* fix, COMPLEX* fiy, COMPLEX* fiz);

extern "C" void ifdda_gaussianfft_cuda_(
    const int *nfft2d, const int *nx, const int *ny, const int *nz, const int *nmax,
    const int *dipole_count, const int *kkm, const int *jjm, const CU_REAL *dipole_xs,
    const CU_REAL *dipole_ys, const CU_REAL *xmin, const CU_REAL *ymin, const CU_REAL *z,
    const CU_REAL *aretecube, const CU_REAL *x0, const CU_REAL *y0, const CU_REAL *z0,
    const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *waist, const CU_REAL *k0,
    const CU_REAL *ss, const CU_REAL *pp, const COMPLEX *e0, const CU_REAL *tolerance,
    const int *circular, COMPLEX *dipole_field, int *nstop,
    const CU_REAL *dipole_xs_device,
    const CU_REAL *dipole_ys_device,
    CUCOMPLEX **resident_field);

extern "C" void ifdda_arbitrary_derivatives_cuda_(
    const CU_REAL* xs, const CU_REAL* ys, const CU_REAL* zs, const int* tabdip,
    const int* coordinate_count, const int* dipole_count,
    const char* filename, const int* filename_length,
    CUCOMPLEX* output, int* status_out);

extern "C" void incidentarbitrarypos_(
    const CU_REAL* x, const CU_REAL* y, const CU_REAL* z, const CU_REAL* cube_size,
    CUCOMPLEX* ex, CUCOMPLEX* ey, CUCOMPLEX* ez, int* stop, const char* namefile,
    char* info, size_t namefile_length, size_t info_length);

#ifdef IFDDA_CUDA_CDMLIB_HOST_ABI
extern "C" void calculangle_(REAL *theta,REAL *phi,const REAL *k0,
                         const REAL *aretecube,const int *nfft2d);
extern "C" void calculwidefield_(int *, ...);
extern "C" void copiani_(const COMPLEX *,COMPLEX *,const int *,
                     const int *,const int *);
extern "C" void diffractefft2denergie_(const int *nx,const int *ny,const int *nz,const int *nxm,const int *nym,
 const int *nzm,const int *nfft2d,int *tabfft2,const REAL *k0,
 const REAL *xs,const REAL *ys,const REAL *zs,const COMPLEX *E0,
 const REAL *ss,const REAL *pp,const REAL *theta,const REAL *phi,
 const REAL *thetam,const REAL *phim,const REAL *ppm,const REAL *ssm,
 const COMPLEX *E0m,const int *nbinc,const REAL *xdip,
 const REAL *ydip,const REAL *zdip,const REAL *xgaus,const REAL *ygaus,
 const REAL *zgaus,const REAL *w0,const REAL *aretecube,const REAL *tol,
 COMPLEX *Eloinx,COMPLEX *Eloiny,COMPLEX *Eloinz,
 const COMPLEX *FF,int *imax,REAL *deltakx,REAL *deltaky,
 COMPLEX *Ediffkzpos,COMPLEX *Ediffkzneg,const char beam[64],
 REAL *efficacite,REAL *efficaciteref,REAL *efficacitetrans,
 const int *nsectionsca,const int *nquickdiffracte,const int64_t *plan2f,
 const int64_t *plan2b,int *nstop,char infostr[64],size_t infostr_len);
extern "C" void diffractefft2denergieconfocal_(const int *speckseed,const REAL *numaperil,const int *nx,const int *ny,
 const int *nz,const int *nxm,const int *nym,const int *nzm,const int *nfft2d,
 int *tabfft2,const REAL *k0,const REAL *xs,const REAL *ys,const REAL *zs,
 const COMPLEX *E0,const REAL *ss,const REAL *pp,const REAL *theta,
 const REAL *phi,const REAL *xgaus,const REAL *ygaus,const REAL *zgaus,
 const REAL *w0,const REAL *aretecube,COMPLEX *Eloinx,
 COMPLEX *Eloiny,COMPLEX *Eloinz,const COMPLEX *FF,
 int *imax,REAL *deltakx,REAL *deltaky,COMPLEX *Ediffkzpos,
 COMPLEX *Ediffkzneg,const char beam[64],REAL *efficacite,
 REAL *efficaciteref,REAL *efficacitetrans,const int *nsectionsca,
 const int *nquickdiffracte,const int64_t *plan2f,const int64_t *plan2b,
 const int *nside,int *nstop,char infostr[64],size_t infostr_len);
extern "C" void diffractefft2dinc_(const int *,const int *,const REAL *,
 COMPLEX *,REAL *,REAL *,REAL *,REAL *,const REAL *,
 const REAL *,const REAL *,const REAL *,const COMPLEX *,const int *,
 const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
 const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
 COMPLEX *,COMPLEX *,COMPLEX *,int *,REAL *,REAL *,
 COMPLEX *,const REAL *,const char *,const int64_t *,const int64_t *,
 int *,char *,size_t);
extern "C" void diffractefft2dinckzneg_(const int *,const int *,const REAL *,
 COMPLEX *,REAL *,REAL *,REAL *,REAL *,const REAL *,
 const REAL *,const REAL *,const REAL *,const COMPLEX *,const int *,
 const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
 const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
 COMPLEX *,COMPLEX *,COMPLEX *,int *,REAL *,REAL *,
 COMPLEX *,const REAL *,const char *,const int64_t *,const int64_t *,
 int *,char *,size_t);

extern "C" void dipoleinc_(const REAL *xdip,const REAL *ydip,const REAL *zdip,
                       const REAL *theta,const REAL *phi,const REAL *x,
                       const REAL *y,const REAL *z,const REAL *aretecube,
                       const REAL *k0,const COMPLEX *e0,
                       COMPLEX *ex,COMPLEX *ey,COMPLEX *ez,
                       int *nstop,char *infostr,size_t infostr_len);
extern "C" void dipoleincder_(const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,
        const COMPLEX *,COMPLEX *,int *,int *,char *,size_t);
extern "C" void dipoleincp_(const REAL *,const REAL *,const REAL *,
    const COMPLEX *,const REAL *,const REAL *,const REAL *,
    const REAL *,const REAL *,COMPLEX *,
    COMPLEX *,COMPLEX *,int *,char *,size_t);
extern "C" void dipoleinsidep_(REAL *,REAL *,REAL *,const REAL *,
    const REAL *,const REAL *,const REAL *,const int *,const int *,int *);
extern "C" void factorto235_(int *n);
extern "C" void fouriertoimage_(const REAL *,const REAL *,const REAL *,
 COMPLEX *,COMPLEX *,COMPLEX *,COMPLEX *,
 COMPLEX *,COMPLEX *,const int *,const int *,const int64_t *,
 const int64_t *);


extern "C" void gaussianchampd_(const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,
        const COMPLEX *,COMPLEX *,const REAL *,const int *);
extern "C" void gaussianchampdcirc_(const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const REAL *,const REAL *,
        const REAL *,const REAL *,const REAL *,const COMPLEX *,
        COMPLEX *,const REAL *,const int *);




extern "C" void gaussianpuissance_(REAL *power,REAL *irradiance,REAL *w0,
                               REAL *k0,COMPLEX *e0);
extern "C" void gaussianpuissancepara_(const REAL *w0,const REAL *p0,
                                   REAL *irradiance,COMPLEX *e0);
extern "C" void greencalculscalairefft_(const int *,const int *,const int *,
        const int *,const int *,const int *,const int *,const int *,const int *,
        const int *,const REAL *,const REAL *,const COMPLEX *,
        const COMPLEX *,const COMPLEX *,COMPLEX *,
        const int64_t *,const int *,int *,char *,size_t);
extern "C" void greencalculscalairefft2_(const int *,const int *,const int *,
        const int *,const int *,const int *,const int *,const int *,const int *,
        const int *,const REAL *,const REAL *,const COMPLEX *,
        const COMPLEX *,const COMPLEX *,COMPLEX *,
        const int64_t *,const int *,int *,char *,size_t);
extern "C" void interpdielec_(const REAL *lambda, const char *materiau,
                          REAL *epr, REAL *epi, char *infostr, int *nstop,
                          size_t materiau_len, size_t infostr_len);

extern "C" void irradiance_(const REAL *power,const REAL *waist,
                        COMPLEX *field,REAL *irradiance);
extern "C" void local_macro_(COMPLEX [3],COMPLEX [3],
                         COMPLEX [9],const REAL *,const REAL *,
                         const int *);
extern "C" void micromasque_(COMPLEX *,const int *,int *,int *,int *,
    const int *,const int *,REAL *,REAL *,REAL *,REAL *,REAL *,
    REAL *,REAL *,const REAL *,const REAL *,const REAL *,
    const REAL *,const int *,const int *,const int *,const int *,
    const int *,int *,char *,size_t);
extern "C" void multilayerbornsca_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length);
extern "C" void multilayerbornscainc_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length);
extern "C" void multilayerbornscaincref_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length);
extern "C" void multilayerbornscaref_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length);
extern "C" void multilayerbornvect_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length);
extern "C" void multilayerbornvectref_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const CU_REAL *aretecube,
    const CU_REAL *k0, const CU_REAL *w0, const COMPLEX *E0, const COMPLEX *ss,
    const COMPLEX *pp, const CU_REAL *theta, const CU_REAL *phi, const CU_REAL *xgaus,
    const CU_REAL *ygaus, const CU_REAL *zgaus, const char beam[64], const COMPLEX *epsilon,
    const COMPLEX *polarisa, const int *niso, const int *ndipole, const int *nx, const int *ny,
    const int *nz, const int *nxm, const int *nym, const int *nzm, const int *nmax,
    const int *nfft2d, COMPLEX *imagex, COMPLEX *imagey, COMPLEX *imagez,
    const COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *FF, const long long *plan2f,
    const long long *plan2b, int *nstop, char infostr[64], size_t beam_length,
    size_t infostr_length);





extern "C" void produitfftmatvect3_(COMPLEX *,COMPLEX *,
        COMPLEX *,COMPLEX *,COMPLEX *,COMPLEX *,
        COMPLEX *,COMPLEX *,COMPLEX *,const int *,
        const int *,const int *,const int *,const int *,const int *,const int *,
        const int *,const int *,const int *,const int *,const int *,const int *,
        const int *,const int *,const COMPLEX *,COMPLEX *,
        const int64_t *,const int64_t *);
extern "C" void relecture_(REAL *aretecube,REAL *lambda,int *nrig,int *nfft2,
        char *beam,char *object,char *trope,int *nnnr,REAL *tolinit,
        REAL *density,REAL *side,REAL *sidex,REAL *sidey,REAL *sidez,
        REAL *hauteur,int *numberobjet,REAL *rayonmulti,REAL *xgmulti,
        REAL *ygmulti,REAL *zgmulti,COMPLEX *epsmulti,
        COMPLEX *epsanimulti,REAL *lc,REAL *hc,int *ng,
        REAL *demiaxea,REAL *demiaxeb,REAL *demiaxec,REAL *thetaobj,
        REAL *phiobj,REAL *psiobj,char *namefileobj,REAL *theta,
        REAL *phi,REAL *pp,REAL *ss,REAL *thetam,REAL *phim,
        REAL *ppm,REAL *ssm,COMPLEX *E0m,int *nbinc,REAL *P0,
        REAL *w0,REAL *xgaus,REAL *ygaus,REAL *zgaus,int *speckseed,
        REAL *numaperil,char *namefileinc,int *numberobjetmax,
        char *polarizability,int *nquad,char *filereread,int *nlecture,
        int *nx,int *ny,int *nz,int *nstop,char *infostr,size_t beam_len,
        size_t object_len,size_t trope_len,size_t namefileobj_len,
        size_t namefileinc_len,size_t polarizability_len,size_t filereread_len,
        size_t infostr_len);
extern "C" void savemacroscopicfield_(const int *,const int *,const int *,
    const int *,const int *,const int *,const REAL *,const REAL *,
    const COMPLEX *,const COMPLEX *,const int *,REAL *,
    COMPLEX *,COMPLEX *,COMPLEX *,COMPLEX *,
    const int64_t *);
#endif

// PML host API: complex arguments are CPU-resident. Only the kernel/device
// buffer interfaces use CUCOMPLEX pointers.
extern "C" void ifdda_apply_pml_mask_cuda_(
    const REAL* xs, const REAL* ys, const REAL* zs,
    const COMPLEX* background_epsilon, const REAL* k0,
    const int* pml_work, const int* mask_type, const int* dipole_count,
    const REAL* spacing, COMPLEX* incident_field, REAL* mask, int* status_out);

extern "C" void rerangechamppola_(
    COMPLEX* field_local, COMPLEX* epsilon, COMPLEX* polarizability,
    const int* isotropy, const REAL* cell_size, const REAL* k0,
    COMPLEX* field, REAL* xs, REAL* ys, REAL* zs,
    REAL* xs_wide, REAL* ys_wide, REAL* zs_wide,
    const int* dipole_map, const int* multi_map, REAL* pml_mask,
    int* nx, int* ny, int* nz, const int* nxm, const int* nym,
    const int* nzm, const int* dipole_count, int* active_count,
    const int* nmax, const char* geometry_type, const char* method,
    REAL* sldr, int* near_field_domain,
    size_t geometry_type_length, size_t method_length);

// Import of CPU/Fortran Chan blocks is packed directly into the unique GPU LU
// matrix; no padded second device-wide matrix is allocated.
extern "C" int chan_preconditioner_upload_cuda_(
    const CUCOMPLEX *cpu_matrices, const int *cpu_pivots,
    const int *nx, const int *ny, const int *nz,
    const int *nxm, const int *nym, const int *nzm);
extern "C" int chan_preconditioner_factorize_cuda_(
    const CUCOMPLEX *cpu_matrices, const int *nx, const int *ny, const int *nz,
    const int *nxm, const int *nym, const int *nzm);

extern "C" int chan_precon_pol_build_cuda_(const CUCOMPLEX *polarizabilities,
                                           const int *dipole_count, const CU_REAL *aretecube,
                                           const CU_REAL *k0, const int *nx, const int *ny,
                                           const int *nz, const int *nxm, const int *nym,
                                           const int *nzm);

extern "C" int chan_precon_pol_build_lu_cuda_(const CUCOMPLEX *polarizabilities,
                                              const int *dipole_count, const CU_REAL *aretecube,
                                              const CU_REAL *k0, const int *nx, const int *ny,
                                              const int *nz, const int *nxm, const int *nym,
                                              const int *nzm);

extern "C" int chan_preconditioner_apply_cuda_(CUCOMPLEX *cpu_field, const int *cpu_dipole_map,
                                               const int *nx, const int *ny, const int *nz,
                                               const int *nxm, const int *nym, const int *nzm);

extern "C" int chan_preconditioner_apply_device_cuda_(CUCOMPLEX *device_field,
                                                      const int *cpu_dipole_map, const int *nx,
                                                      const int *ny, const int *nz, const int *nxm,
                                                      const int *nym, const int *nzm);

extern "C" int chan_preconditioner_apply_initial_dense_device_cuda_(
    CUCOMPLEX *cpu_field, COMPLEX *cpu_solution, const int *dipole_count, const int *nx,
    const int *ny, const int *nz, const int *nxm, const int *nym, const int *nzm);

extern "C" int chan_preconditioner_apply_initial_dense_resident_cuda_(
    CUCOMPLEX *cpu_field, COMPLEX *cpu_solution, const int *dipole_count, const int *nx,
    const int *ny, const int *nz, const int *nxm, const int *nym, const int *nzm);

extern "C" int chan_preconditioner_apply_initial_device_cuda_(
    CUCOMPLEX *cpu_field, COMPLEX *cpu_solution, const int *dipole_count,
    const int *cpu_dipole_map, const int *nx, const int *ny, const int *nz, const int *nxm,
    const int *nym, const int *nzm);

extern "C" int chan_preconditioner_apply_initial_resident_cuda_(
    CUCOMPLEX *cpu_field, COMPLEX *cpu_solution, const int *dipole_count,
    const int *cpu_dipole_map, const int *nx, const int *ny, const int *nz, const int *nxm,
    const int *nym, const int *nzm);

extern "C" int chan_preconditioner_factorize_resident_cuda_(const int *nx, const int *ny,
                                                            const int *nz, const int *nxm,
                                                            const int *nym, const int *nzm);

extern "C" int chan_preconditioner_initialize_born_resident_cuda_(
    CUCOMPLEX *incident, COMPLEX *cpu_solution, const int *dipole_count,
    const int *cpu_dipole_map, const int *nx, const int *ny, const int *nz, const int *nxm,
    const int *nym, const int *nzm, const int *dense_grid);

extern "C" void cors_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda, const int *ndim,
                      const int *nlar, int *nou, COMPLEX *work, int *itno, const int *maxit,
                      const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *rho,
                      CUCOMPLEX *rhoa, CUCOMPLEX *beta, int *nstat, int *steperr);

extern "C" void gcors2_(COMPLEX *xi, COMPLEX *xr, COMPLEX *xsol_host, COMPLEX *b,
                        const int *ldap, const int *ndimp, const int *nlarp, int *nou,
                        COMPLEX *work_host, int *itno, const int *maxitp,
                        const CU_REAL *tolerancep, CU_REAL *norm, CUCOMPLEX *alpha,
                        CUCOMPLEX *alphat, CUCOMPLEX *rho, CUCOMPLEX *rhot, int *nstat,
                        int *steperr);

extern "C" void gcors2_random_values(int n, REAL *random_imag,
                                     REAL *random_real);

extern "C" void gpbicg2_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                         const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                         const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                         CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn,
                         int *nstat, int *steperr);

extern "C" void gpbicg_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda, const int *ndim,
                        const int *nlar, int *nou, COMPLEX *work, int *itno, const int *maxit,
                        const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *beta,
                        CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn, int *nstat,
                        int *steperr);

extern "C" void gpbicgar2_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                           const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                           const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                           CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn,
                           int *nstat, int *steperr);

extern "C" void gpbicgar_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                          const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                          const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                          CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn,
                          int *nstat, int *steperr);

extern "C" void gpbicgplus_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                            const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                            const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                            CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn,
                            int *nstat, int *steperr);

extern "C" void gpbicgsafe_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                            const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                            const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                            CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *r0rn,
                            int *nstat, int *steperr);

extern "C" void gpbicgstabl_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, COMPLEX *xsol,
                             CU_REAL *norm, COMPLEX *work, CUCOMPLEX *alpha, CUCOMPLEX *beta,
                             CUCOMPLEX *sigma, CUCOMPLEX *rho, const int *l, const int *lda,
                             const int *ndim, const int *nlar, int *nou, int *nr, int *np, int *ns,
                             int *nq, int *j, int *nstat, int *steperr, int *itno, const int *maxit,
                             CU_REAL *tole, const CU_REAL *tol);

extern "C" void gpbicgstarplus_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                                const int *ndim, const int *nlar, int *nou, COMPLEX *work,
                                int *itno, const int *maxit, const CU_REAL *tol, CU_REAL *norm,
                                CUCOMPLEX *alpha, CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta,
                                CUCOMPLEX *r0rn, int *nstat, int *steperr);

extern "C" void gpbicor_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, COMPLEX *ffloc,
                         const int *lda, const int *ndim, const int *nlar, int *nou,
                         COMPLEX *work, int *itno, const int *maxit, const CU_REAL *tol,
                         CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *beta, CUCOMPLEX *eta,
                         CUCOMPLEX *dzeta, CUCOMPLEX *rho, CUCOMPLEX *rhoa, int *nstat,
                         int *steperr);

extern "C" void hdf5write1d_(const int *file_id, const char *dataset_name,
                               const CU_REAL *data, const int *dimensions,
                               size_t dataset_length);

extern "C" void hdf5write1d_i8_(const std::int64_t *file_id, const char *dataset_name,
                                  const CU_REAL *data, const int *dimensions,
                                  size_t dataset_length);

extern "C" void idrs_(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                      const int *ndim, const int *nlar, int *nou, int *ns, COMPLEX *work,
                      int *itno, const int *maxit, const CU_REAL *tol, CU_REAL *residu,
                      CU_REAL *norm, CUCOMPLEX *omega, int *is, int *nstat, int *steperr);

extern "C" void idrs_fill_random_(const int *ndim, const int *ns,
                                   CUCOMPLEX *values);

extern "C" void ifdda_compute_optical_force_cuda_(const COMPLEX *polarization,
                                                  const COMPLEX *field_derivatives,
                                                  const int *dipole_count, const int *mesh_size,
                                                  const REAL *scale, REAL *force_x,
                                                  REAL *force_y, REAL *force_z,
                                                  REAL *force_total, int *status_out);

extern "C" void ifdda_compute_optical_torque_cuda_(const COMPLEX *local_field, const REAL *xs,
                                                   const REAL *ys, const REAL *zs,
                                                   const int *dipole_count, const CU_REAL *k03,
                                                   const REAL *scale, REAL *center,
                                                   REAL *torque_x, REAL *torque_y,
                                                   REAL *torque_z, REAL *torque_total,
                                                   int *status_out);

extern "C" int ifdda_concentric_sphere_geometry_cuda(int subdivisions, CU_REAL cell_size,
                                                     CU_REAL center, int sphere_count,
                                                     const CU_REAL *radii, CU_REAL *xs, CU_REAL *ys,
                                                     CU_REAL *zs, int *layer_map);

extern "C" int ifdda_correlated_material_field_cuda(
    int nx, int ny, int nz, CU_REAL spacing, CU_REAL coherence_length,
    CU_REAL standard_deviation, CU_REAL epsilon_real, CU_REAL epsilon_imag,
    const CU_REAL *phases, CU_REAL *real_field, CU_REAL *mean,
    CU_REAL *standard_deviation_result);

extern "C" int ifdda_cube_geometry_cuda(int subdivisions, CU_REAL side, CU_REAL cell_size,
                                        CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
                                        CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, int *dipole_map);

// Homogeneous geometry: CPU tabdip/counts, GPU polarizability and permittivity.
// Reuses the occupancy map left on the device by the geometry kernel.
extern "C" int ifdda_fill_homogeneous_geometry_materials_cuda(
    const int* tabdip, size_t grid, int nmax, int niso,
    int compact, int anisotropic, const CUCOMPLEX* eps,
    const CUCOMPLEX* epsani, CUCOMPLEX scalar_alpha,
    const CUCOMPLEX* tensor_alpha, CUCOMPLEX* epsilon,
    CUCOMPLEX* polarisa);

// Typed host counterpart: std::complex arrays on CPU, CUDA buffers internal.
// This interface never exposes a GPU pointer through the C/Fortran ABI.
extern "C" int ifdda_fill_homogeneous_geometry_materials_host_cuda(
    const int* tabdip, size_t grid, int nmax, int niso,
    int compact, int anisotropic, const COMPLEX* eps,
    const COMPLEX* epsani, COMPLEX scalar_alpha,
    const COMPLEX* tensor_alpha, COMPLEX* epsilon,
    COMPLEX* polarisa);

extern "C" void objetcylindre_(
    const char* trope, const COMPLEX* eps, const COMPLEX* epsani,
    const COMPLEX* eps0, REAL* xs, REAL* ys, REAL* zs,
    const REAL* k0, REAL* aretecube, int* tabdip, const int* nnnr,
    const int* nmax, int* nbsphere, int* ndipole, int* nx, int* ny, int* nz,
    const int* nxm, const int* nym, const int* nzm, const char* methode,
    const int* na, COMPLEX* epsilon, COMPLEX* polarisa, const int* niso,
    REAL* radius, REAL* height, REAL* xg, REAL* yg, REAL* zg,
    REAL* phi, REAL* theta, REAL* psi, REAL* sldr, const int* nmat,
    const int64_t* file_id, const int64_t* group_iddip, char* infostr, int* nstop,
    size_t trope_length, size_t methode_length, size_t infostr_length);

// Multiple materials: CPU tabdip and dipole labels (1..label_count, 0=vacuum),
// GPU material assignment from small scalar/tensor lookup tables.
extern "C" int ifdda_fill_labeled_geometry_materials_cuda(
    const int* tabdip, size_t grid, const int* material_labels,
    int nmax, int niso, int label_count, int vacuum_background,
    const COMPLEX* label_epsilon, const COMPLEX* label_alpha,
    COMPLEX* epsilon, COMPLEX* polarisa);

// Scatter already-computed inhomogeneous device material values to the
// compact/full host dipole ordering supplied by CPU tabdip.
extern "C" int ifdda_scatter_gpu_cell_materials_cuda(
    const int* tabdip, size_t grid, int nmax,
    const CU_INT* device_inside_map,
    const CUCOMPLEX* device_cell_epsilon,
    const CUCOMPLEX* device_cell_alpha,
    COMPLEX* epsilon, COMPLEX* polarisa);

// All complex pointers in these Fortran-facing interfaces are CPU-resident.

extern "C" void objetarbitrary_(
    const char* object_type, COMPLEX* epsilon_value, COMPLEX* epsilon_anisotropic,
    const COMPLEX* epsilon_background, REAL* xs, REAL* ys, REAL* zs,
    const REAL* k0, REAL* cell_size, int* dipole_map, const int* random_count,
    const int* nmax, int* dipole_count, int* ndipole, int* nx, int* ny, int* nz,
    const char* polarizability_method, const char* filename, int* particle_count,
    COMPLEX* epsilon, COMPLEX* polarizability, const int* isotropy,
    REAL* sldr, const int* output_mode, const int64_t* file_id,
    const int64_t* group_id, char* infostr, int* nstop,
    size_t object_type_length, size_t method_length, size_t filename_length,
    size_t infostr_length);

extern "C" int ifdda_cuboid_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size, CU_REAL side_x,
                                          CU_REAL side_y, CU_REAL side_z, CU_REAL center_x,
                                          CU_REAL center_y, CU_REAL center_z, CU_REAL *xs,
                                          CU_REAL *ys, CU_REAL *zs, int *dipole_map);

extern "C" int ifdda_cylinder_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size,
                                            CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
                                            CU_REAL radius, CU_REAL height, CU_REAL m0, CU_REAL m1,
                                            CU_REAL m2, CU_REAL m3, CU_REAL m4, CU_REAL m5,
                                            CU_REAL m6, CU_REAL m7, CU_REAL m8, CU_REAL *xs,
                                            CU_REAL *ys, CU_REAL *zs, int *inside_map);

extern "C" int ifdda_ellipsoid_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size,
                                             CU_REAL extent_x, CU_REAL extent_y, CU_REAL extent_z,
                                             CU_REAL radius_a, CU_REAL radius_b, CU_REAL radius_c,
                                             CU_REAL m0, CU_REAL m1, CU_REAL m2, CU_REAL m3,
                                             CU_REAL m4, CU_REAL m5, CU_REAL m6, CU_REAL m7,
                                             CU_REAL m8, CU_REAL *xs, CU_REAL *ys, CU_REAL *zs,
                                             int *inside_map);

extern "C" int ifdda_get_initial_guess_mode_all_cuda_();

extern "C" int ifdda_get_last_initial_guess_mode_all_cuda_();

extern "C" int ifdda_nspheres_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size,
                                            CU_REAL x_min, CU_REAL y_min, CU_REAL z_min,
                                            int sphere_count, int fixed_sphere_count,
                                            const CU_REAL *sphere_parameters, CU_REAL *xs,
                                            CU_REAL *ys, CU_REAL *zs, int *sphere_map);

extern "C" void ifdda_reset_preconditionneur_apply_cuda_state();
extern "C" void ifdda_reset_incident_gaussian_gkp_state();
extern "C" void ifdda_reset_gaussianfft_gkp_state();
extern "C" void ifdda_reset_beampropagation_gkp_state();

extern "C" void ifdda_reset_preconditionneurscal_cuda_state();

extern "C" int ifdda_rotated_cuboid_geometry_cuda(int nx, int ny, int nz, CU_REAL cell_size,
                                                  CU_REAL extent_x, CU_REAL extent_y,
                                                  CU_REAL extent_z, CU_REAL side_x, CU_REAL side_y,
                                                  CU_REAL side_z, CU_REAL center_x,
                                                  CU_REAL center_y, CU_REAL center_z,
                                                  const CU_REAL *inverse_rotation, CU_REAL *xs,
                                                  CU_REAL *ys, CU_REAL *zs, int *inside_map);

extern "C" int ifdda_scalar_chan_apply_resident(CUCOMPLEX *field, int field_count,
                                                const int *tabdip, int solve_blocks);

extern "C" int ifdda_scalar_chan_prepare_resident(const CUCOMPLEX *polarizabilities, int nbsphere,
                                                  const CUCOMPLEX *uincx, const CUCOMPLEX *uincy,
                                                  const CUCOMPLEX *uincz, CU_REAL aretecube,
                                                  CU_REAL k0, int nrig, int nx, int ny, int nz,
                                                  int nxm, int nym, int nzm, int factorize);

extern "C" void ifdda_set_initial_guess_mode_all_cuda_(int mode);

extern "C" int ifdda_set_resident_chan_preconditioner_all_(const int *enabled, const int *nx,
                                                       const int *ny, const int *nz, const int *nxm,
                                                       const int *nym, const int *nzm);

extern "C" int ifdda_sphere_geometry_compact_cuda(
    int subdivisions, CU_REAL radius, CU_REAL cell_size,
    CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
    CU_REAL* mesh_x, CU_REAL* mesh_y, CU_REAL* mesh_z, int* inside_map,
    CU_REAL* packed_x, CU_REAL* packed_y, CU_REAL* packed_z,
    int* packed_count);

extern "C" int ifdda_sphere_geometry_cuda(int subdivisions, CU_REAL radius, CU_REAL cell_size,
                                          CU_REAL center_x, CU_REAL center_y, CU_REAL center_z,
                                          CU_REAL *xs, CU_REAL *ys, CU_REAL *zs, int *inside_map);

extern "C" void pimzbicgstab_(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, const COMPLEX *b,
                              const int *lda, const int *nlar, const int *ndim, int *nou,
                              COMPLEX *work, COMPLEX *alpha, COMPLEX *omega, COMPLEX *rho,
                              CU_REAL *norm, const CU_REAL *tole, int *itno, const int *maxit,
                              int *nstat, int *steperr);

extern "C" void pimzqmr_(COMPLEX *xs, COMPLEX *xi, COMPLEX *xr, const COMPLEX *b, COMPLEX *work,
                         CU_REAL *norm, const int *lda, const int *ndim, const int *nlar,
                         CUCOMPLEX *lambda, CUCOMPLEX *kappa, CUCOMPLEX *theta, CUCOMPLEX *gamma,
                         CUCOMPLEX *ksi, CUCOMPLEX *rho, CUCOMPLEX *epsilon, CUCOMPLEX *mu,
                         CUCOMPLEX *tau, CUCOMPLEX *dots1, CUCOMPLEX *dots2, CUCOMPLEX *dots3,
                         CUCOMPLEX *dots4, int *nou, int *nt, int *itno, const int *maxit,
                         CU_REAL *tole, const CU_REAL *tol, int *nstat, int *steperr);

extern "C" void preconditionneurinit2_(
    CUCOMPLEX *ff, COMPLEX *xi, CUCOMPLEX *xr, const int *tabdip,
    const CUCOMPLEX *polarisa, const int *niso,
    CUCOMPLEX *chanxxvect, CUCOMPLEX *chanxyvect, CUCOMPLEX *chanxzvect,
    CUCOMPLEX *chanyyvect, CUCOMPLEX *chanyzvect, CUCOMPLEX *chanzzvect,
    const int *nx, const int *ny, const int *nz, const int *nbsphere,
    const int *ndipole, const int *nxm, const int *nym, const int *nzm,
    const CU_REAL *aretecube, const CU_REAL *k0, const int *ntest,
    CUCOMPLEX *sdetnn, CUCOMPLEX *vectxx, CUCOMPLEX *vectyy,
    CUCOMPLEX *vectzz, const long long *plan2b, const long long *plan2f,
    char *infostr, int *nstop, size_t infostr_len);

extern "C" void preconditionneurinit_(
    CUCOMPLEX *ff, COMPLEX *xi, CUCOMPLEX *xr,
    const CUCOMPLEX *polarisa, const int *niso,
    CUCOMPLEX *chanxxvect, CUCOMPLEX *chanxyvect, CUCOMPLEX *chanxzvect,
    CUCOMPLEX *chanyyvect, CUCOMPLEX *chanyzvect, CUCOMPLEX *chanzzvect,
    const int *nx, const int *ny, const int *nz, const int *nbsphere,
    const int *nxm, const int *nym, const int *nzm,
    const CU_REAL *aretecube, const CU_REAL *k0, const int *ntest,
    CUCOMPLEX *sdetnn, CUCOMPLEX *vectxx, CUCOMPLEX *vectyy,
    CUCOMPLEX *vectzz, const long long *plan2b, const long long *plan2f,
    char *infostr, int *nstop, size_t infostr_len);

extern "C" void produitfftmatvect3device_(
    const CUCOMPLEX *tensor_xx, const CUCOMPLEX *tensor_xy, const CUCOMPLEX *tensor_xz,
    const CUCOMPLEX *tensor_yy, const CUCOMPLEX *tensor_yz, const CUCOMPLEX *tensor_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz, const CU_INT *tabdip, const int *ntotalm,
    const int *ntotal, const int *nmax, const int *ndipole, const int *nxm, const int *nym,
    const int *nzm, const int *nx, const int *ny, const int *nz, const int *nx2, const int *ny2,
    const int *nxy2, const int *nz2, const CUCOMPLEX *device_ffx, CUCOMPLEX *device_ffb,
    const long long *planb, const long long *planf);

// Explicit CPU-tabdip entry; performs a typed CPU -> GPU map transfer.
extern "C" void produitfftmatvect3host_(
    const CUCOMPLEX *tensor_xx, const CUCOMPLEX *tensor_xy, const CUCOMPLEX *tensor_xz,
    const CUCOMPLEX *tensor_yy, const CUCOMPLEX *tensor_yz, const CUCOMPLEX *tensor_zz,
    CUCOMPLEX *vectx, CUCOMPLEX *vecty, CUCOMPLEX *vectz, const int *cpu_tabdip,
    const int *ntotalm, const int *ntotal, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm, const int *nx, const int *ny,
    const int *nz, const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const CUCOMPLEX *device_ffx, CUCOMPLEX *device_ffb,
    const long long *planb, const long long *planf);

extern "C" void produitfftmatvectoptdevice_(const CUCOMPLEX* tensor_xx,
                                            const CUCOMPLEX* tensor_xy,
                                            const CUCOMPLEX* tensor_xz,
                                            const CUCOMPLEX* tensor_yy,
                                            const CUCOMPLEX* tensor_yz,
                                            const CUCOMPLEX* tensor_zz,
                                            CUCOMPLEX* device_vectx,
                                            CUCOMPLEX* device_vecty,
                                            CUCOMPLEX* device_vectz,
                                            const int* ntotalm,
                                            const int* ntotal,
                                            const int* nmax,
                                            const int* nxm,
                                            const int* nym,
                                            const int* nzm,
                                            const int* nx,
                                            const int* ny,
                                            const int* nz,
                                            const int* nx2,
                                            const int* ny2,
                                            const int* nxy2,
                                            const int* nz2,
                                            const CUCOMPLEX* device_ffx,
                                            CUCOMPLEX* device_ffb,
                                            const long long* planf,
                                            const long long* planb);

extern "C" void produitfftmatvectoptscalaire_(
    const COMPLEX *green, COMPLEX *work, const int *ntotalm, const int *ntotal,
    const int *nmax, const int *nxm, const int *nym, const int *nzm, const int *nx,
    const int *ny, const int *nz, const int *nx2, const int *ny2, const int *nxy2,
    const int *nz2, const COMPLEX *ffx, COMPLEX *ffb, const long long *planf,
    const long long *planb);

extern "C" void produitpolavect_(const COMPLEX *polarizability, const int *niso, const int *nmax,
                                 const int *ndipoles, const COMPLEX *input, COMPLEX *output);

extern "C" void qmrbicgstab_(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, COMPLEX *b,
                             const int *lda, const int *ndim, const int *nlar, int *nou,
                             COMPLEX *work, int *itno, const int *maxit, const CU_REAL *tol,
                             CU_REAL *residu, CU_REAL *norm, CUCOMPLEX *alpha, CUCOMPLEX *rho,
                             CUCOMPLEX *omega, CUCOMPLEX *tau, CUCOMPLEX *taut, CUCOMPLEX *theta,
                             CUCOMPLEX *thetat, CUCOMPLEX *eta, CUCOMPLEX *etat, const int *methode,
                             int *nstat, int *steperr);

extern "C" void random_bridge_init(int seed);

extern "C" REAL random_bridge_next(void);

extern "C" void random_bridge_reset(void);

extern "C" void random_bridge_stream_init(int channel, int seed);

extern "C" REAL random_bridge_stream_next(int channel);

extern "C" bool register_gpbicg_device_b_alias(const void *cpu_b_key, CUCOMPLEX *gpu_b,
                                               size_t count);
// Immutable resident illumination source: the solver keeps its own mutable RHS
// and initializes it with a device-to-device copy instead of uploading CPU FF0.
extern "C" bool register_gpbicg_device_b_source(const void *cpu_b_key,
                                                 const CUCOMPLEX *gpu_b,
                                                 size_t count);
extern "C" bool has_gpbicg_device_b_source(const void *cpu_b_key);

extern "C" void ssbicgsafe2_(COMPLEX *xi, COMPLEX *xr, const COMPLEX *b, const int *lda,
                             const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                             const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                             CUCOMPLEX *beta, CUCOMPLEX *eta, CUCOMPLEX *dzeta, CUCOMPLEX *ffa,
                             int *nstat, int *steperr);

extern "C" void tfqmr_(COMPLEX *x, COMPLEX *xi, COMPLEX *xr, COMPLEX *b, const int *lda,
                       const int *ndim, const int *nlar, int *nou, COMPLEX *work, int *itno,
                       const int *maxit, const CU_REAL *tol, CU_REAL *norm, CUCOMPLEX *alpha,
                       CUCOMPLEX *beta, CUCOMPLEX *rho, CUCOMPLEX *tau, CUCOMPLEX *theta,
                       CUCOMPLEX *eta, int *nstat, int *steperr);

extern "C" void writehdf5mic_(const COMPLEX *ex, const COMPLEX *ey,
                                const COMPLEX *ez, const int *nfft2d,
                                const int *imaxk0, COMPLEX *ediff,
                                const int *ntest, const char *name,
                                const std::int64_t *group_idmic, size_t name_length);

extern "C" void zbicgstabl_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, COMPLEX *xsol,
                            CU_REAL *norm, COMPLEX *work, CUCOMPLEX *alpha, CUCOMPLEX *rho,
                            CUCOMPLEX *omega, const int *l, int *nr, int *nu, int *j,
                            const int *lda, const int *ndim, const int *nlar, int *nou, int *nstat,
                            int *steperr, int *itno, const int *maxit, const CU_REAL *tole,
                            const CU_REAL *tol);

extern "C" void zcg_(COMPLEX *xi, COMPLEX *xr, COMPLEX *b, CU_REAL *norm, COMPLEX *work,
                     CUCOMPLEX *alpha, CUCOMPLEX *rho0, CUCOMPLEX *xxi, const int *lda,
                     const int *ndim, const int *nlar, int *nou, int *nstat, int *nloop,
                     const int *maxit, CU_REAL *tole, const CU_REAL *tol);

// Canonical declarations migrated from call-site-local prototypes.
extern "C" void diffractefft2dlens_set_resident_output(int enabled);

extern "C" void ifdda_beampropagation_run_cuda_(
    const CU_REAL *xs, const CU_REAL *ys, const CU_REAL *zs, const COMPLEX *epsilon,
    const COMPLEX *incident, COMPLEX *local_field, const int *nx, const int *ny, const int *nz,
    const int *nfft2d, const int *nmax, const int *niso, const int *ndipole, const int *kkm,
    const int *jjm, const CU_REAL *xmin, const CU_REAL *ymin, const CU_REAL *first_z,
    const CU_REAL *aretecube, const CU_REAL *k0, const COMPLEX *E0, const CU_REAL *ss,
    const CU_REAL *pp, const CU_REAL *theta_degrees, const CU_REAL *phi_degrees,
    const CU_REAL *xgaus, const CU_REAL *ygaus, const CU_REAL *zgaus, const CU_REAL *w0,
    const int *beam_type, const int resident_mode, const int macro_mode,
    int *nstop);

extern "C" void ifdda_compute_force_green_derivative_cuda_(
    const REAL *spacing, const REAL *k0, const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nz2, const int *nxy2, const int *axis,
    COMPLEX *output, int *status_out);

extern "C" int microsbf_nrig0_fields_c99(
    const int *nproche, const int *ntotalm, const int *ntotal,
    const int *ldabi, const int *nlar, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *nbsphere, const int *nbsphere3, const int *nlim, const int *niso,
    COMPLEX *tensor_xx, COMPLEX *tensor_xy,
    COMPLEX *tensor_xz, COMPLEX *tensor_yy,
    COMPLEX *tensor_yz, COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    int *tabdip, COMPLEX *XI, COMPLEX *XR, COMPLEX *wrk,
    COMPLEX *FF, COMPLEX *FF0, COMPLEX *FFloc,
    COMPLEX *polarisa, char *methodeit, const REAL *tolinit,
    REAL *tol1, int *nloop, int *ncompte,
    const int64_t *planxf, const int64_t *planxb,
    const int64_t *planyf, const int64_t *planyb,
    const int64_t *planzf, const int64_t *planzb,
    const int64_t *planf, const int64_t *planb,
    int *nstop, char *infostr, size_t methodeit_len, size_t infostr_len);

extern "C" int microsbf_nrig2_fields_c99(
    const int *nbsphere, const int *nmax, const int *niso,
    const COMPLEX *epsilon, const COMPLEX *polarisa,
    const COMPLEX *FF0, const REAL *aretecube, const REAL *k0,
    COMPLEX *FFloc, COMPLEX *FF);

extern "C" int microsbf_nrig3_fields_c99(
    const int *nbsphere, const int *nmax, const int *niso,
    const int *ntotalm, const int *ntotal, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *tabdip,
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy,
    const COMPLEX *tensor_xz, const COMPLEX *tensor_yy,
    const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    const int64_t *planf, const int64_t *planb,
    const COMPLEX *polarisa, const COMPLEX *FF0,
    COMPLEX *XI, COMPLEX *XR, COMPLEX *FF,
    COMPLEX *FFloc);

extern "C" int microsbf_nrig4_fields_c99(
    const int *nbsphere, const int *nmax, const int *niso,
    const int *ntotalm, const int *ntotal, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *tabdip,
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy,
    const COMPLEX *tensor_xz, const COMPLEX *tensor_yy,
    const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    const int64_t *planf, const int64_t *planb,
    const COMPLEX *polarisa, const COMPLEX *FF0,
    COMPLEX *XI, COMPLEX *XR, COMPLEX *FF,
    COMPLEX *FFloc);

extern "C" int microsbf_nrig5_fields_host_c99(
    const int *nbsphere, const int *nmax, const int *niso,
    const int *ntotalm, const int *ntotal, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *tabdip,
    const COMPLEX *tensor_xx, const COMPLEX *tensor_xy,
    const COMPLEX *tensor_xz, const COMPLEX *tensor_yy,
    const COMPLEX *tensor_yz, const COMPLEX *tensor_zz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    const int64_t *planf, const int64_t *planb,
    const COMPLEX *polarisa, const COMPLEX *epsilon,
    const COMPLEX *eps0, const REAL *aretecube, const REAL *k0,
    const COMPLEX *FF0, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *FF, COMPLEX *FFloc);

extern "C" int microsbf_nrig67_fields_c99(
    const int *nrig, const int *nbsphere, const int *nmax, const int *niso,
    const int *ndipole, const int *nx, const int *ny, const int *nz,
    const int *nxm, const int *nym, const int *nzm, const int *nfft2d,
    const REAL *xs, const REAL *ys, const REAL *zs,
    const REAL *aretecube, const REAL *k0, const REAL *w0,
    const COMPLEX *E0, const REAL *ss, const REAL *pp,
    const REAL *kxinc, const REAL *kyinc,
    const COMPLEX *epsilon, const COMPLEX *polarisa,
    COMPLEX *Efourierx, COMPLEX *Efouriery,
    COMPLEX *Efourierz, COMPLEX *FF0, COMPLEX *FFloc,
    COMPLEX *FF, const int64_t *plan2f, const int64_t *plan2b,
    int *nstop, char *infostr, size_t infostr_len);

extern "C" int microsbf_polarization_fields_cuda(
    const int *nbsphere, const int *nmax,
    const int *niso, const COMPLEX *polarisa, const COMPLEX *local_field,
    COMPLEX *scattered_field);

extern "C" int microsconfocal_advanced_fields_c99(
    const int *nrig, const int *nproche,
    const int *ntotalm, const int *ntotal, const int *ldabi,
    const int *nlar, const int *nmax, const int *ndipole,
    const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz,
    const int *nx2, const int *ny2, const int *nxy2, const int *nz2,
    const int *nbsphere, const int *nbsphere3, const int *nlim, const int *niso,
    int *tabdip, COMPLEX *const tensor[6], COMPLEX *const vect[3],
    COMPLEX *XI, COMPLEX *XR, COMPLEX *wrk,
    COMPLEX *FF, COMPLEX *FF0, COMPLEX *FFloc,
    COMPLEX *polarisa, const COMPLEX *epsilon,
    const COMPLEX *E0, const REAL *psi,
    const REAL *ss, const REAL *pp,
    const REAL *kxinc, const REAL *kyinc,
    const REAL *xs, const REAL *ys, const REAL *zs,
    const REAL *aretecube, const REAL *k0, const REAL *w0,
    const int *nfft2d, COMPLEX *Efourierx,
    COMPLEX *Efouriery, COMPLEX *Efourierz,
    const REAL *tolinit, REAL *tol1, int *nloop, int *ncompte,
    char *methodeit, char *infostr, int *nstop,
    const int64_t *planXf, const int64_t *planXb,
    const int64_t *planYf, const int64_t *planYb,
    const int64_t *planZf, const int64_t *planZb,
    const int64_t *planf, const int64_t *planb,
    const int64_t *plan2f, const int64_t *plan2b,
    size_t methodeit_len, size_t infostr_len);

extern "C" int microsconfocal_approx_fields_c99(
    const int *nrig, const int *nbsphere,
    const int *nmax, const int *niso, const int *ntotalm, const int *ntotal,
    const int *ndipole, const int *nxm, const int *nym, const int *nzm,
    const int *nx, const int *ny, const int *nz, const int *nx2,
    const int *ny2, const int *nxy2, const int *nz2, const int *tabdip,
    COMPLEX *const tensor[6], COMPLEX *const vect[3],
    const int64_t *planf, const int64_t *planb,
    const COMPLEX *polarisa, const COMPLEX *epsilon,
    const COMPLEX *eps0, const REAL *aretecube, const REAL *k0,
    const COMPLEX *FF0, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *FF, COMPLEX *FFloc);

extern "C" int microsconfocal_diffraction_c99(
    const int *nquicklens,const int *nx,const int *ny,const int *nz,
    const int *nxm,const int *nym,const int *nzm,const int *nfft2d,
    int *tabfft2,const int *nside,int *imaxk0,const int *nbsphere,
    const REAL *k0,const REAL *numaperk,const REAL *aretecube,
    const REAL *xs,const REAL *ys,const REAL *zs,COMPLEX *Efourierx,
    COMPLEX *Efouriery,COMPLEX *Efourierz,const COMPLEX *FF,
    REAL *deltakx,REAL *deltaky,COMPLEX *Ediffkzneg,
    const int64_t *plan2f,const int64_t *plan2b,int *nstop,char *infostr,
    size_t infostr_len);

// Host input and output are REAL/COMPLEX; only device buffers use CU_REAL/CUCOMPLEX.
extern "C" void ifdda_antenna_field_cuda_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count,
    const REAL *xdip, const REAL *ydip, const REAL *zdip,
    const REAL *aretecube, const REAL *k0, const COMPLEX *e0,
    const REAL *theta_deg, const REAL *phi_deg,
    const REAL *reference_x, const REAL *reference_y,
    const REAL *reference_z, COMPLEX *output,
    COMPLEX *reference_field, int *status_out);

// Prototypes moved from cross-unit local declarations; scoped to callers.
#ifdef IFDDA_CUDA_CALLERS_CALCULWIDEFIELD_CUDA
extern "C" void ifdda_antenna_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const REAL *xdip,
    const REAL *ydip, const REAL *zdip, const REAL *aretecube,
    const REAL *k0, const COMPLEX *e0, const REAL *theta_deg,
    const REAL *phi_deg, const REAL *reference_x,
    const REAL *reference_y, const REAL *reference_z,
    COMPLEX *reference_field, int *status_out, CUCOMPLEX *output_device);
extern "C" void ifdda_confocal_field_cuda_resident_grid_origin_(
    const int *nx, const int *ny, const int *nz, const int *dipole_count,
    const int *nfft, const REAL *k0, const REAL *xx0, const REAL *yy0,
    const REAL *zz0, const REAL *psi_degrees, const REAL *spacing,
    const REAL *power, const REAL *aperture, COMPLEX *e0,
    const REAL *grid_origin_x, const REAL *grid_origin_y,
    const REAL *grid_origin_z, const int *bidirectional,
    CUCOMPLEX **output_device, REAL *irradiance, int *status_out);
extern "C" void ifdda_gaussianfft_cuda_resident_grid_origin_(
    const int *nfft2d, const int *nx, const int *ny, const int *nz,
    const int *nmax, const int *grid_count, const REAL *grid_x,
    const REAL *grid_y, const REAL *grid_origin_x,
    const REAL *grid_origin_y, const REAL *grid_origin_z,
    const REAL *aretecube, const REAL *x0, const REAL *y0,
    const REAL *z0, const REAL *theta, const REAL *phi,
    const REAL *waist, const REAL *k0, const REAL *ss,
    const REAL *pp, const COMPLEX *e0, const REAL *tolerance,
    const int *circular, CUCOMPLEX **resident_field, int *status_out);
extern "C" void ifdda_incident_arbitrary_cuda_device_grid_bounds_(
    const REAL *grid_xmin, const REAL *grid_xmax,
    const REAL *grid_ymin, const REAL *grid_ymax,
    const REAL *grid_zmin, const REAL *grid_zmax,
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const REAL *cube_size,
    CUCOMPLEX *output_device, const int *nxm, const int *nym,
    const int *nzm, const int *dipole_count, int *stop,
    const char *namefile, char *info);
extern "C" void ifdda_multi_plane_wave_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const REAL *k0,
    const COMPLEX *e0, const COMPLEX *e0m, const REAL *ssm,
    const REAL *ppm, const REAL *theta_deg, const REAL *phi_deg,
    const int *plane_count, CUCOMPLEX *output_device, int *status_out);
extern "C" void ifdda_paraxial_gaussian_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const COMPLEX *e0,
    const COMPLEX *epsilon, const REAL *ss, const REAL *pp,
    const REAL *theta_deg, const REAL *phi_deg, const REAL *x0,
    const REAL *y0, const REAL *z0, const REAL *w0, const REAL *k0,
    const int *circular, CUCOMPLEX *output_device, int *status_out);
extern "C" void ifdda_plane_wave_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const REAL *k0,
    const COMPLEX *eps0, const COMPLEX *e0, const REAL *ss,
    const REAL *pp_input, const REAL *theta_deg,
    const REAL *phi_deg, const int *circular,
    CUCOMPLEX *output_device, int *status_out);
extern "C" void ifdda_plane_wave_field_cuda_resident_(
    const REAL *xs, const REAL *ys, const REAL *zs, const int *count,
    const REAL *k0, const COMPLEX *eps0, const COMPLEX *e0,
    const REAL *ss, const REAL *pp_input, const REAL *theta_deg,
    const REAL *phi_deg, const int *circular, CUCOMPLEX **output_device,
    int *status_out);
extern "C" void ifdda_rigorous_gaussian_field_cuda_device_coordinates_(
    const CU_REAL *xs_device, const CU_REAL *ys_device,
    const CU_REAL *zs_device, const int *count, const REAL *x0,
    const REAL *y0, const REAL *z0, const REAL *theta_degrees,
    const REAL *phi_degrees, const REAL *waist, const REAL *k0,
    const REAL *ss, const REAL *pp, const COMPLEX *e0,
    const REAL *tolerance, const int *circular, const int *nloin,
    CUCOMPLEX *output_device, int *status_out);
extern "C" void ifdda_speckle_field_cuda_resident_grid_origin_(
    const int *nx, const int *ny, const int *nz, const int *dipole_count,
    const int *nfft, const CU_REAL *k0, const CU_REAL *xx0, const CU_REAL *yy0,
    const CU_REAL *zz0, const CU_REAL *psi_degrees, const CU_REAL *spacing,
    const CU_REAL *power, const CU_REAL *aperture, COMPLEX *e0,
    const CUCOMPLEX *phase_positive, const CUCOMPLEX *phase_negative,
    const CU_REAL *grid_origin_x, const CU_REAL *grid_origin_y,
    const CU_REAL *grid_origin_z, const int *bidirectional,
    CUCOMPLEX **output_device, CU_REAL *irradiance, int *status_out);
extern "C" void produitfftmatvectoptresident_(GreenTensorDeviceView device_green,
                                              CUCOMPLEX* device_ffx,
                                              CUCOMPLEX* device_ffb,
                                              const int* ntotalm,
                                              const int* ntotal,
                                              const int* nmax,
                                              const int* nxm,
                                              const int* nym,
                                              const int* nzm,
                                              const int* nx,
                                              const int* ny,
                                              const int* nz,
                                              const int* nx2,
                                              const int* ny2,
                                              const int* nxy2,
                                              const int* nz2,
                                              const long long* planf,
                                              const long long* planb);
#endif
extern "C" void deltakroutine_(const REAL *kxinc, const REAL *kyinc,
    const REAL *deltakx, const REAL *deltaky, const REAL *numaper,
    int *i, int *j);

extern "C" void inverserig_(
    COMPLEX *txx, COMPLEX *txy, COMPLEX *txz, COMPLEX *tyy, COMPLEX *tyz,
    COMPLEX *tzz, COMPLEX *vx, COMPLEX *vy, COMPLEX *vz, int *tabdip,
    int *ntotalm, int *ntotal, int *lda, int *nlar, int *nmax, int *ndipole,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz, int *nx2,
    int *ny2, int *nxy2, int *nz2, int *nbsphere, int *nbsphere3,
    COMPLEX *xi, COMPLEX *xr, COMPLEX *work, COMPLEX *field,
    COMPLEX *incident, COMPLEX *local, COMPLEX *polar, int *niso,
    char *method, REAL *tolerance, REAL *final_residual, int *products,
    int *product_calls, int *max_iterations, int64_t *plan_xf,
    int64_t *plan_xb, int64_t *plan_yf, int64_t *plan_yb,
    int64_t *plan_zf, int64_t *plan_zb, int64_t *planf, int64_t *planb,
    int *stop, char *info, size_t method_length, size_t info_length);

// Original Fortran inverserigopt ABI: unlike inverserig_, there is no
// tabdip/ndipole argument. This compatibility entry point forwards to the
// same resident CUDA inverserig driver as inverserig_.
extern "C" void inverserigopt_(
    COMPLEX *txx, COMPLEX *txy, COMPLEX *txz, COMPLEX *tyy, COMPLEX *tyz,
    COMPLEX *tzz, COMPLEX *vx, COMPLEX *vy, COMPLEX *vz,
    int *ntotalm, int *ntotal, int *lda, int *nlar, int *nmax,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz,
    int *nx2, int *ny2, int *nxy2, int *nz2, int *nbsphere, int *nbsphere3,
    COMPLEX *xi, COMPLEX *xr, COMPLEX *work, COMPLEX *field,
    COMPLEX *incident, COMPLEX *local, COMPLEX *polar, int *niso,
    char *method, REAL *tolerance, REAL *final_residual, int *products,
    int *product_calls, int *max_iterations, int64_t *plan_xf,
    int64_t *plan_xb, int64_t *plan_yf, int64_t *plan_yb,
    int64_t *plan_zf, int64_t *plan_zb, int64_t *planf, int64_t *planb,
    int *stop, char *info, size_t method_length, size_t info_length);

#ifdef IFDDA_CUDA_CALLERS_MICROSBF_CUDA

#endif
#ifdef IFDDA_CUDA_CALLERS_MICROSBFNEG_CUDA
extern "C" void microsbf_(
    COMPLEX *FFTTENSORxx, COMPLEX *FFTTENSORxy, COMPLEX *FFTTENSORxz,
    COMPLEX *FFTTENSORyy, COMPLEX *FFTTENSORyz, COMPLEX *FFTTENSORzz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    int *ntotalm, int *ntotal, int *ldabi, int *nlar, int *nmax,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz,
    int *nx2, int *ny2, int *nxy2, int *nz2, int *ndipole,
    int *nbsphere, int *nbsphere3, int *nproche, int *nrig,
    int *nfft2d, int *tabfft2, int *Tabdip, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *wrk, COMPLEX *FF, COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *polarisa,
    int *niso, COMPLEX *epsilon, char *methodeit, char *trope,
    REAL *tolinit, REAL *tol1, int *nloop, int *ncompte, int *nlim,
    REAL *xs, REAL *ys, REAL *zs, REAL *aretecube, REAL *numaper,
    REAL *numaperinc, REAL *numaperinc2, int *npolainc_in,
    int *nquicklens, COMPLEX *eps0, REAL *k0, REAL *P0, REAL *irra,
    REAL *w0, REAL *gross, REAL *zlens, COMPLEX *Eimagex, COMPLEX *Eimagey,
    COMPLEX *Eimagez, COMPLEX *Efourierx, COMPLEX *Efouriery, COMPLEX *Efourierz,
    COMPLEX *Eimageincx, COMPLEX *Eimageincy, COMPLEX *Eimageincz,
    COMPLEX *Efourierincx, COMPLEX *Efourierincy, COMPLEX *Efourierincz,
    COMPLEX *Ediffkzpos, COMPLEX *Ediffkzneg, REAL *kxy, REAL *xy,
    int *nside, COMPLEX *masque, int *nmasque, int *ideltam,
    int *imaxk0, int *nbitermic, REAL *deltak, REAL *deltakx,
    int *ntypemic, int64_t *planXf, int64_t *planXb, int64_t *planYf,
    int64_t *planYb, int64_t *planZf, int64_t *planZb, int64_t *planf, int64_t *planb,
    int64_t *plan2f, int64_t *plan2b, int *nmat, int *file_id_in,
    int *group_idmic_in, int *nstop, char *infostr,
    size_t methodeit_len, size_t trope_len, size_t infostr_len);
#endif
#ifdef IFDDA_CUDA_CALLERS_MICROSCONFOCALFLUO_CUDA
extern "C" int microsconfocal_incident_field_c99(
    const int* nbsphere,
    const REAL* k0,
    const REAL* kxinc,
    const REAL* kyinc,
    const REAL* psi,
    const COMPLEX* E0,
    const REAL* xs,
    const REAL* ys,
    const REAL* zs,
    COMPLEX* FF0,
    COMPLEX* XI,
    COMPLEX* Ax,
    COMPLEX* Ay,
    COMPLEX* Az,
    REAL* kzinc);
extern "C" int microsconfocal_validate_c99(
    const int* nfft2d,
    const REAL* nmtom,
    const REAL* numaperk,
    const REAL* deltak,
    const int* nside,
    const REAL* stepzin,
    const REAL* zin,
    const REAL* zsupin,
    REAL* stepz,
    REAL* zinf,
    REAL* zsup,
    int* nplanz,
    int* nplanze,
    int* nstop,
    char* infostr,
    std::size_t infostr_len);
#endif
#ifdef IFDDA_CUDA_CALLERS_MICROSDFNEG_CUDA
extern "C" void microsdf_(
    COMPLEX *FFTTENSORxx, COMPLEX *FFTTENSORxy, COMPLEX *FFTTENSORxz,
    COMPLEX *FFTTENSORyy, COMPLEX *FFTTENSORyz, COMPLEX *FFTTENSORzz,
    COMPLEX *vectx, COMPLEX *vecty, COMPLEX *vectz,
    int *ntotalm, int *ntotal, int *ldabi, int *nlar, int *nmax,
    int *nxm, int *nym, int *nzm, int *nx, int *ny, int *nz,
    int *nx2, int *ny2, int *nxy2, int *nz2, int *ndipole,
    int *nbsphere, int *nbsphere3, int *nproche, int *nrig,
    int *nfft2d, int *tabfft2, int *tabdip, COMPLEX *XI, COMPLEX *XR,
    COMPLEX *wrk, COMPLEX *FF, COMPLEX *FF0, COMPLEX *FFloc, COMPLEX *polarisa,
    int *niso, COMPLEX *epsilon, char *methodeit, char *trope,
    REAL *tolinit, REAL *tol1,
    int *nloop, int *ncompte, int *nlim,
    REAL *xs, REAL *ys, REAL *zs, REAL *aretecube, REAL *numaper,
    REAL *numaperinc, int *npolainc, int *nquicklens, COMPLEX *eps0,
    REAL *k0, REAL *P0, REAL *irra, REAL *w0, REAL *gross, REAL *zlens,
    COMPLEX *Eimagex, COMPLEX *Eimagey, COMPLEX *Eimagez,
    COMPLEX *Efourierx, COMPLEX *Efouriery, COMPLEX *Efourierz,
    COMPLEX *Eimageincx, COMPLEX *Eimageincy, COMPLEX *Eimageincz,
    COMPLEX *Efourierincx, COMPLEX *Efourierincy, COMPLEX *Efourierincz,
    COMPLEX *Ediffkzpos, COMPLEX *Ediffkzneg, REAL *kxy, REAL *xy, int *nside,
    int64_t *planXf, int64_t *planXb, int64_t *planYf, int64_t *planYb,
    int64_t *planZf, int64_t *planZb, int64_t *planf, int64_t *planb,
    int64_t *plan2f, int64_t *plan2b, int *nmat, int64_t *file_id,
    int64_t *group_idmic, int *nstop, char *infostr,
    size_t methodeit_len, size_t trope_len, size_t infostr_len);
#endif
#ifdef IFDDA_CUDA_CALLERS_PRODUITPOLAVECT_CUDA
extern "C" int ifdda_get_multilayer_resident_polarization_inputs_cuda(
    const COMPLEX *cpu_local_key, const COMPLEX *cpu_alpha_key, const int *niso,
    const int *nmax, const int *ndipoles, CUCOMPLEX **gpu_local, CUCOMPLEX **gpu_alpha);
#endif

#ifdef IFDDA_CUDA_CALLERS_MICROSDF_CUDA

extern "C" int ifdda_produitpolavect_device_input_c99(
    const CUCOMPLEX* device_input, const COMPLEX* host_polarizability,
    const int* niso, const int* nmax, const int* ndipoles, COMPLEX* host_output);
#endif
#endif // IFDDA_CUDA_INTERFACES_H
