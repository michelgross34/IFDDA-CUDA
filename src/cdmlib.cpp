/* Active source mapping: CUDA/cdmlib.f90 lines 1-7253; historical translated blocks are checked against cdmlib/cdmlib.f90 before integration. */
#include "cdmlib_cpp_precision.hpp"
#include <vector>
#define IFDDA_CUDA_CDMLIB_HOST_ABI
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CDMLIB_HOST_ABI
#define IFDDA_CUDA_DECLARE_HOST_ABI
#include "ifdda.h"
#include "ifdda_solver_abi_typed.hpp"
#undef IFDDA_CUDA_DECLARE_HOST_ABI
#include "ifdda_micros_abi_typed.hpp"
#include "cuda_data.h"
#include "cuda_utils.h"
#include "cuda_solver_interfaces.h"




#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>
#undef USE_FFTW
/* Fortran precision and by-reference ABI assumptions, checked in both builds. */
static_assert(sizeof(int) == 4, "Fortran integer(ip) ABI requires 32-bit int");
static_assert(sizeof(int64_t) == 8, "Fortran integer(kind=8) ABI requires 64 bits");
static_assert(sizeof(COMPLEX) == 2 * sizeof(REAL), "Fortran complex(wp) ABI size mismatch");
static_assert(std::is_same<REAL, float>::value || std::is_same<REAL, double>::value,
              "Unsupported Fortran real(wp) ABI");




/* Reuse the phase structs with C++ scalar and FFTW precision aliases. */
#define IFDDA_CDMLIB_PRECISION_C99_H
#ifdef USE_FFTW
#include "cdmlib_cpp_fftw_precision.hpp"
#endif
extern "C" {
/* Entry-phase declarations integrated in this translation unit. */
/* Solver-phase declarations integrated in this translation unit. */
#include "cdmlib_abi.h"
#include "cdmlib_initphase.h"
#include "cdmlib_param.h"

/* The absorbing PML mask changes ff0 after the incident-field construction.
   Refresh the CPU-visible per-dipole incident arrays from the masked ff0. */
static int cdmlib_entryphase__map_incident_to_dipoles_c99(
    const cdmlib_abi_args_c99 *a, const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c, int ndipole);

/* Build the typed phase views over the caller-owned Fortran arguments.
   No arrays are copied: the C99 phases operate on the same column-major
   buffers allocated by the caller. */
int cdmlib_build_phase_views_c99(const cdmlib_abi_args_c99 *args,
                                 cdmlib_initphase_c99 *init,
                                 cdmlib_config_c99 *config);

/* Execute the C99 entry initialization/config/material pre-object phases and
   write their status and normalized input values back through the ABI views. */
int cdmlib_prepare_preobject_from_abi_c99(const cdmlib_abi_args_c99 *args,
                                          cdmlib_initphase_c99 *init,
                                          cdmlib_config_c99 *config,
                                          REAL *k0,REAL *k03);

/* Initialize geometry inputs before preparing its parameter view in cdmlib_. */
int cdmlib_initialize_geometry_c99(const cdmlib_abi_args_c99 *args,
                                   cdmlib_initphase_c99 *init,
                                   cdmlib_config_c99 *config,
                                   cdmlib_entry_runtime_c99 *runtime);
/* Execute geometry using the already prepared, caller-visible view. */
int cdmlib_execute_geometry_c99(const cdmlib_abi_args_c99 *args,
                               cdmlib_initphase_c99 *init,
                               cdmlib_config_c99 *config,
                               cdmlib_entry_runtime_c99 *runtime,
                               cdmlib_geometry_params_c99 *geometry,
                               REAL *k0,REAL *k03,REAL *lambda10n);

/* Dispatch a ported incident-field beam using the same input/output buffers. */
int cdmlib_compute_incident_from_abi_c99(const cdmlib_abi_args_c99 *args,
                                         cdmlib_initphase_c99 *init,
                                         cdmlib_config_c99 *config,
                                         cdmlib_entry_runtime_c99 *runtime,
                                         int nbsphere,int ndipole);

/* Run incident-field dispatch, optional PML remapping and FF0 GPU publishing. */
int cdmlib_execute_illumination_c99(
    const cdmlib_illumination_params_c99 *illumination);

/* Fill FF0 for one Cartesian point-dipole polarization used by greentensor. */
int cdmlib_set_greentensor_source_c99(const cdmlib_abi_args_c99 *args,
                                      const cdmlib_initphase_c99 *init,
                                      int polarization);
int cdmlib_map_incident_fields_c99(const cdmlib_abi_args_c99 *args,
                                   const cdmlib_initphase_c99 *init,
                                   const cdmlib_config_c99 *config,
                                   int ndipole);

/* Prepare the padded Green/solver grid from the geometry and option state. */
int cdmlib_prepare_fft_grid_from_abi_c99(const cdmlib_abi_args_c99 *args,
                                         const cdmlib_initphase_c99 *init,
                                         const cdmlib_config_c99 *config,
                                         cdmlib_entry_runtime_c99 *runtime,
                                         int nbsphere);

/* Own the temporary FFT work arrays allocated inside cdmlib.f90. */
int cdmlib_allocate_fft_workspace_c99(cdmlib_entry_runtime_c99 *runtime);
void cdmlib_free_fft_workspace_c99(cdmlib_entry_runtime_c99 *runtime);

/* Dispatch one translated Green tensor branch into the caller's six buffers. */
int cdmlib_compute_green_from_abi_c99(const cdmlib_abi_args_c99 *args,
                                      const cdmlib_initphase_c99 *init,
                                      const cdmlib_config_c99 *config,
                                      cdmlib_entry_runtime_c99 *runtime);

/* Execute the Green phase from its prepared, non-owning parameter view. */
int cdmlib_execute_green_c99(const cdmlib_green_params_c99 *green);

/* Dispatch the projected or exp(ikr)/r scalar Green FFT from borrowed data. */
int cdmlib_execute_scalar_initialization_c99(
    const cdmlib_scalar_initialization_params_c99 *s)
{
    if(!s || !s->green || !s->scratch || !s->macroscopicfieldx ||
       !s->macroscopicfieldy || !s->ux || !s->uy || !s->uz ||
       !s->xi || !s->xr || !s->wrk || !s->ff || !s->ff0 ||
       !s->ffloc || !s->polarisa || !s->tabdip || !s->k0 ||
       !s->aretecube || !s->planf || !s->planb || !s->nstop ||
       !s->nloop || !s->ncompte || !s->tol1 || !s->methodeit ||
       !s->infostr || s->nbsphere<=0)
        return 1;

    const int nb=s->nbsphere,nd=s->ndipole;
    const int ntotalm=s->ntotalm,ntotal=s->ntotal;
    const int ldabi=s->ldabi,nlar=s->nlar,nmax=s->nmax;
    const int nxm=s->nxm,nym=s->nym,nzm=s->nzm;
    const int nx=s->nx,ny=s->ny,nz=s->nz;
    const int nx2=s->nx2,ny2=s->ny2,nxy2=s->nxy2,nz2=s->nz2;
    const int ni=s->niso,nlim=s->nlim;
    REAL scalar_tol=0.01,scalar_tol1=0.0;
    int scalar_loop=0,scalar_compte=0;
    // Legacy CUDA wrappers take a pointer to an INTEGER*8 FFTW-plan handle.
    // Keep its 64-bit value unchanged while matching their long long ABI.
    const long long fft_planf = static_cast<long long>(*s->planf);
    const long long fft_planb = static_cast<long long>(*s->planb);

    for(int i=0;i<nb;++i) s->xi[i]=0.0;
    *s->nstop=0;
    *s->nloop=0; *s->ncompte=0; *s->tol1=0.0;

    if(s->nproche==-1)
        inverserigscalaire_(s->green,s->scratch,s->tabdip,
            s->macroscopicfieldx,s->macroscopicfieldy,
            s->ux,s->uy,s->uz,
            &ntotalm,&ntotal,&ldabi,&nlar,&nmax,&nd,&nxm,&nym,&nzm,
            &nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,&nb,s->xi,
            s->xr,s->wrk,s->ff,s->ff0,s->ffloc,s->polarisa,
            &ni,s->methodeit,&scalar_tol,&scalar_tol1,&scalar_loop,
            &scalar_compte,&nlim,s->planf,s->planb,
            (int *)s->nstop,s->infostr,s->methodeit_len,s->infostr_len);
    else {
        // Scalar warm-start solver: resident CUDA matrix products, no
        // preconditioner, zero initial guess; distinct from Green generation.
        // inverserigoptscalaire.f90 initializes xi from the incident field
        // regardless of the outer ninitest=4 warm-start selector.
        const int no_preconditioner=0,born_guess=1;
        const int optimized_product=1,scalar_nrig=8;
        inverserigscalaire_cuda_driver_(s->green,s->scratch,s->tabdip,
            s->macroscopicfieldx,s->macroscopicfieldy,
            s->ux,s->uy,s->uz,
            &ntotalm,&ntotal,&ldabi,&nlar,&nmax,&nd,&nxm,&nym,&nzm,
            &nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,&nb,s->xi,
            s->xr,s->wrk,s->ff,s->ff0,s->ffloc,s->polarisa,
            &ni,s->methodeit,&scalar_tol,&scalar_tol1,&scalar_loop,
            &scalar_compte,&nlim,&no_preconditioner,&scalar_nrig,
            &born_guess,&optimized_product,s->k0,s->aretecube,
            &fft_planf,&fft_planb,(int *)s->nstop,s->infostr,
            s->methodeit_len,s->infostr_len);
    }
    if(*s->nstop==1) return 1;
    if(scalar_tol1>=scalar_tol) {
        const char msg[]="Converge do not reach!";
        memset(s->infostr,' ',s->infostr_len);
        memcpy(s->infostr,msg,std::min(sizeof msg-1,s->infostr_len));
        *s->nstop=1;
        return 1;
    }

    // Scalar solution expands to the 3-component vector initial guess.
    for(int i=0;i<3*nb;++i) s->xi[i]=s->ffloc[i];
    // The scalar CUDA driver cached xi as a scalar (nb elements).  The
    // caller now owns a different, 3*nb-element vector initial iterate at
    // the same host address. Force the vector driver to upload this iterate.
    mark_gpbicg_device_xi_current(s->xi, false);
    *s->nstop=0;
    return 0;
}
//-------------------------------------------

int cdmlib_execute_green_scalar_c99(
    const cdmlib_green_scalar_params_c99 *green);

/* Solve ninitest=4 scalar warm start and publish its vector initial guess. */
int cdmlib_execute_scalar_initialization_c99(
    const cdmlib_scalar_initialization_params_c99 *solver);


/* Route the nrig=0, nprecon=0 solver branch through the C99 inverserig_. */
int cdmlib_solve_from_abi_c99(const cdmlib_abi_args_c99 *args,
                              const cdmlib_initphase_c99 *init,
                              const cdmlib_config_c99 *config,
                              const cdmlib_entry_runtime_c99 *runtime,
                              int nbsphere,int ndipole);

/* C99 translation of the nsection=1 extinction/absorption/scattering block. */
int cdmlib_compute_cross_sections_from_abi_c99(
    const cdmlib_abi_args_c99 *args,
    const cdmlib_entry_runtime_c99 *runtime,int nbsphere);

/* Map the solved local field and run the translated macroscopic-field helper. */
int cdmlib_map_solved_fields_from_abi_c99(
    const cdmlib_abi_args_c99 *args,
    const cdmlib_initphase_c99 *init,
    const cdmlib_config_c99 *config,
    int ndipole);


#include "hdf5_helpers.h"
}
#ifdef I
#undef I
#endif
#define I COMPLEX(0, 1)
#define cabs(z) std::abs(z)
#define cimag(z) std::imag(z)
#define creal(z) std::real(z)
#define conj(z) std::conj(z)
#define cexp(z) std::exp(z)
#define csqrt(z) std::sqrt(z)
template<class T> static inline void cdmlib_zero_n(T *values, size_t count)
{ std::fill_n(values,count,T{}); }

extern "C" {/* Translation of the active CUDA/cdmlib.f90 orchestration path. The historical C++ blocks below were retained only where their ABI and arithmetic match the active Fortran interfaces; this unit is compiled independently before CMake integration. Includes reread path; numerical parity NOT certified. */
/* UNVALIDATED C99 conversion candidate for cdmlib/cdmlib.f90.
 * This translation unit uses the companion cdmlib_C99 modules listed by CMake.
 * Do not treat it as numerically certified; see CONVERSION_PROGRESS.md.
 */
#include <stdint.h>
/* Without USE_HDF5, the HDF5 symbols must not be referenced by the
 * linker.  The configuration validator already rejects nmat == 2.
 * Keep the complete HDF5 output code available for USE_HDF5 builds. */
#ifndef USE_HDF5
#define cdmlib_hdf5_create_groups_c99 cdmlib_disabled_hdf5_create_groups_c99
#define cdmlib_hdf5_close_groups_c99 cdmlib_disabled_hdf5_close_groups_c99
#define cdmlib_hdf5_write1d_c99 cdmlib_disabled_hdf5_write1d_c99
static int cdmlib_disabled_hdf5_create_groups_c99(const char *name,
        size_t len,int64_t ids[8])
{ (void)name; (void)len; (void)ids; return -1; }
static void cdmlib_disabled_hdf5_close_groups_c99(int64_t ids[8])
{ (void)ids; }
static int cdmlib_disabled_hdf5_write1d_c99(int64_t group,const char *name,
        const REAL *data,int32_t count,int32_t memory_size)
{ (void)group; (void)name; (void)data; (void)count; (void)memory_size; return -1; }
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Polarization-aware Rytov reconstruction; shared algebra with the
 * Fortran implementation in cdmlib/rytov_vector_field.f90. */
static int cdmlib_rytov_vector_field_c99(const COMPLEX *eref,
    const COMPLEX *delta, COMPLEX *result, int nb)
{
    int nstop = 1;
    rytov_vector_field_(eref, delta, result, &nb, &nstop);
    return nstop;
}
static void cdmlib_rytov_error_c99(char *infostr, size_t n)
{
    const char msg[]="Rytov vector field is outside numerical range";
    const size_t len=sizeof msg-1;
    memset(infostr,' ',n);
    memcpy(infostr,msg,len<n?len:n);
}

/* FFTW planning translated from cdmlib.f90. Entirely inactive unless USE_FFTW.
 * Use the FFTW Fortran-77 wrapper ABI already used by the Fortran routines:
 * integer(8) plan handles and integer(4) dimensions passed by address. */
#ifdef USE_FFTW
#ifdef USE_SIMP
#define CDMLIB_FFTW(name) sfftw_##name##_
#else
#define CDMLIB_FFTW(name) dfftw_##name##_
#endif
#include "cuda_cdmlib_fftw_abi.h"
//---------------------------------------------
static void cdmlib_fftw_destroy_c99(cdmlib_entry_runtime_c99 *r)
{
    int64_t *plans[]={&r->planf,&r->planb,&r->planXf,&r->planXb,
        &r->planYf,&r->planYb,&r->planZf,&r->planZb,&r->plan2f,&r->plan2b};
    for(size_t i=0;i<sizeof plans/sizeof plans[0];++i)
        if(*plans[i]) { CDMLIB_FFTW(destroy_plan)(plans[i]); *plans[i]=0; }
}

//------------------------------------------

static int cdmlib_fftw_create_c99(cdmlib_entry_runtime_c99 *r,
                                   const cdmlib_abi_args_c99 *a)
{
    if(!r||!a||!r->vectx||!a->eimagex||!a->nfft2d) return 1;
    int threads_ok=0;
    CDMLIB_FFTW(init_threads)(&threads_ok);
    if(!threads_ok) return 1;
    int threads=1;
#ifdef _OPENMP
    threads=omp_get_max_threads();
#endif
    CDMLIB_FFTW(plan_with_nthreads)(&threads);
    const unsigned flags=(r->nx>0 && r->ny>0 && r->nz>0 &&
        (REAL)r->nx*r->ny*r->nz<=1000000.0) ||
#if defined(USE_CUDA) || defined(IFDDA_CUDA_INVERSE_DRIVER)
        1
#else
        0
#endif
        ?64u:0u; /* FFTW_ESTIMATE or FFTW_MEASURE */
    const int backward=1,forward=-1,rank=1,one=1;
    const int nx2=r->nx2,ny2=r->ny2,nz2=r->nz2;
    const int nfft=(int)*a->nfft2d;
    if(nfft<=0) return 1;
    cdmlib_fftw_complex *work=(cdmlib_fftw_complex *)r->vectx;
    cdmlib_fftw_complex *image=(cdmlib_fftw_complex *)a->eimagex;
    CDMLIB_FFTW(plan_dft_3d)(&r->planb,&nx2,&ny2,&nz2,work,work,&backward,&flags);
    CDMLIB_FFTW(plan_dft_3d)(&r->planf,&nx2,&ny2,&nz2,work,work,&forward,&flags);
    int n=nx2, embed=nx2, howmany=ny2*r->nz, dist=nx2;
    CDMLIB_FFTW(plan_many_dft)(&r->planXb,&rank,&n,&howmany,work,&embed,&one,&dist,work,&embed,&one,&dist,&backward,&flags);
    CDMLIB_FFTW(plan_many_dft)(&r->planXf,&rank,&n,&howmany,work,&embed,&one,&dist,work,&embed,&one,&dist,&forward,&flags);
    n=ny2; embed=ny2; howmany=r->nx; dist=1;
    CDMLIB_FFTW(plan_many_dft)(&r->planYb,&rank,&n,&howmany,work,&embed,&nx2,&dist,work,&embed,&nx2,&dist,&backward,&flags);
    CDMLIB_FFTW(plan_many_dft)(&r->planYf,&rank,&n,&howmany,work,&embed,&nx2,&dist,work,&embed,&nx2,&dist,&forward,&flags);
    n=nz2; embed=nz2; howmany=r->nxy2; dist=1;
    CDMLIB_FFTW(plan_many_dft)(&r->planZb,&rank,&n,&howmany,work,&embed,&r->nxy2,&dist,work,&embed,&r->nxy2,&dist,&backward,&flags);
    CDMLIB_FFTW(plan_many_dft)(&r->planZf,&rank,&n,&howmany,work,&embed,&r->nxy2,&dist,work,&embed,&r->nxy2,&dist,&forward,&flags);
    CDMLIB_FFTW(plan_dft_2d)(&r->plan2b,&nfft,&nfft,image,image,&backward,&flags);
    CDMLIB_FFTW(plan_dft_2d)(&r->plan2f,&nfft,&nfft,image,image,&forward,&flags);
    if(!r->planf||!r->planb||!r->planXf||!r->planXb||!r->planYf||
       !r->planYb||!r->planZf||!r->planZb||!r->plan2f||!r->plan2b) {
        cdmlib_fftw_destroy_c99(r);
        return 1;
    }
    return 0;
}
#endif /* USE_FFTW */

/* Port of cdmlib.f90's nlecture/reload branch. The companion relecture.c99
 * routine validates/creates .data/.bin metadata; the .lf record round trip
 * was compared against Fortran for the sphere/plane-wave fixture. */
//-------------------------------------------
static int cdmlib_relecture_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_entry_runtime_c99 *r,
    int *loaded)
{
    
    if(loaded) *loaded=0;
    if(!a||!r||!loaded||!a->nlecture||*a->nlecture!=1) return 0;
    if(!a->aretecube||!a->lambda||!a->beam||!a->object||!a->trope||
       !a->filereread||!a->nrig||!a->nfft2d||!a->nnnr||!a->tolinit||
       !a->density||!a->side||!a->sidex||!a->sidey||!a->sidez||!a->hauteur||
       !a->numberobjet||!a->rayonmulti||!a->xgmulti||!a->ygmulti||!a->zgmulti||
       !a->epsmulti||!a->epsanimulti||!a->lc||!a->hc||!a->ng||!a->demiaxea||
       !a->demiaxeb||!a->demiaxec||!a->thetaobj||!a->phiobj||!a->psiobj||
       !a->namefileobj||!a->theta||!a->phi||!a->pp||!a->ss||!a->thetam||
       !a->phim||!a->ppm||!a->ssm||!a->e0m||!a->nbinc||!a->p0||!a->w0||
       !a->xgaus||!a->ygaus||!a->zgaus||!a->speckseed||!a->numaperil||
       !a->namefileinc||!a->polarizability||!a->nquad||!a->nstop||
       !a->infostr||!a->nbsphere||!a->ff||!a->ffloc) {
        if(a&&a->nstop&&a->infostr) {
            static const char msg[]="Invalid nlecture/reload arguments";
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
            *a->nstop=1;
        }
        return 1;
    }
    int nrig=(int)*a->nrig,nfft=(int)*a->nfft2d,nnnr=(int)*a->nnnr;
    int numberobjet=(int)*a->numberobjet,numberobjetmax=numberobjet;
    int ng=(int)*a->ng,nbinc=(int)*a->nbinc,speckseed=(int)*a->speckseed;
    int nquad=(int)*a->nquad,nlecture1=0,nx=r->nx,ny=r->ny,nz=r->nz;
    relecture_((REAL *)a->aretecube,(REAL *)a->lambda,&nrig,&nfft,
        a->beam,a->object,a->trope,&nnnr,(REAL *)a->tolinit,
        (REAL *)a->density,(REAL *)a->side,(REAL *)a->sidex,
        (REAL *)a->sidey,(REAL *)a->sidez,(REAL *)a->hauteur,
        &numberobjet,(REAL *)a->rayonmulti,(REAL *)a->xgmulti,
        (REAL *)a->ygmulti,(REAL *)a->zgmulti,
        (COMPLEX *)a->epsmulti,(COMPLEX *)a->epsanimulti,
        (REAL *)a->lc,(REAL *)a->hc,&ng,(REAL *)a->demiaxea,
        (REAL *)a->demiaxeb,(REAL *)a->demiaxec,(REAL *)a->thetaobj,
        (REAL *)a->phiobj,(REAL *)a->psiobj,a->namefileobj,
        (REAL *)a->theta,(REAL *)a->phi,(REAL *)a->pp,(REAL *)a->ss,
        (REAL *)a->thetam,(REAL *)a->phim,(REAL *)a->ppm,
        (REAL *)a->ssm,(COMPLEX *)a->e0m,&nbinc,(REAL *)a->p0,
        (REAL *)a->w0,(REAL *)a->xgaus,(REAL *)a->ygaus,
        (REAL *)a->zgaus,&speckseed,(REAL *)a->numaperil,a->namefileinc,
        &numberobjetmax,(char *)a->polarizability,&nquad,a->filereread,
        &nlecture1,&nx,&ny,&nz,(int *)a->nstop,a->infostr,a->beam_len,
        a->object_len,a->trope_len,a->namefileobj_len,a->namefileinc_len,
        a->polarizability_len,a->filereread_len,a->infostr_len);
    if(*a->nstop!=0) return 1;
    if(nlecture1==0) return 0;

    char filename[65];
    size_t base=a->filereread_len;
    while(base>0&&a->filereread[base-1]==' ') --base;
    if(base>61) base=61; /* CHARACTER(64) assignment of name//'.lf'. */
    memcpy(filename,a->filereread,base);
    memcpy(filename+base,".lf",3);
    filename[base+3]='\0';
    FILE *f=fopen(filename,"rb");
    if(!f) {
        static const char msg[]="file for read local field do not exist!";
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
               sizeof msg-1:a->infostr_len);
        *a->nstop=1;
        return 1;
    }
    COMPLEX *ff=(COMPLEX *)a->ff;
    COMPLEX *ffloc=(COMPLEX *)a->ffloc;
    const int n3=3*(int)*a->nbsphere;
    const uint32_t expected=(uint32_t)(2u*sizeof(COMPLEX));
    for(int i=0;i<n3;++i) {
        uint32_t leading=0,trailing=0;
        if(fread(&leading,sizeof leading,1,f)!=1||leading!=expected||
           fread(&ff[i],sizeof ff[i],1,f)!=1||
           fread(&ffloc[i],sizeof ffloc[i],1,f)!=1||
           fread(&trailing,sizeof trailing,1,f)!=1||trailing!=leading) {
            fclose(f);
            static const char msg[]="Invalid sequential-unformatted .lf record";
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
            *a->nstop=1;
            return 1;
        }
    }
    fclose(f);
    *a->nstop=0;
    *loaded=1;
    return 0;
}
//-------------------------------------------

static int cdmlib_write_density_triplet_c99(
    const cdmlib_abi_args_c99 *a, const int nd,
    const REAL *fx, const REAL *fy, const REAL *fz,
    const char *filex, const char *filey, const char *filez)
{
    FILE *xfile=fopen(filex,"w"),*yfile=fopen(filey,"w"),*zfile=fopen(filez,"w");
    if(!xfile || !yfile || !zfile) {
        static const char msg[]="Could not write force/torque density files";
        if(xfile) fclose(xfile);
        if(yfile) fclose(yfile);
        if(zfile) fclose(zfile);
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
               sizeof msg-1:a->infostr_len);
        *a->nstop=(cdmlib_ip)1;
        return 1;
    }
    const int *tab=(const int *)a->tabdip;
    for(int i=0;i<nd;++i) {
        const int dip=tab[i]-1;
        const REAL x=dip>=0?fx[dip]:0.0;
        const REAL y=dip>=0?fy[dip]:0.0;
        const REAL z=dip>=0?fz[dip]:0.0;
        if(fprintf(xfile," % .16E\n",x)<0 ||
           fprintf(yfile," % .16E\n",y)<0 ||
           fprintf(zfile," % .16E\n",z)<0) {
            static const char msg[]="Error writing force/torque density files";
            fclose(xfile); fclose(yfile); fclose(zfile);
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
            *a->nstop=(cdmlib_ip)1;
            return 1;
        }
    }
    fclose(xfile); fclose(yfile); fclose(zfile);
    return 0;
}
//-------------------------------------------

static int cdmlib_write_hdf5_density_triplet_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *init,
    const int64_t group_id,const int nd,const REAL *x,const REAL *y,
    const REAL *z,const char *const names[3])
{
    const int32_t *tab=(const int32_t *)a->tabdip;
    const REAL *values[3]={x,y,z};
    const int32_t memory_size=(int32_t)(3*init->nmax);
    REAL *buffer=(REAL *)calloc((size_t)memory_size,sizeof *buffer);
    if(!buffer) {
        static const char msg[]="Could not allocate HDF5 density buffer";
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?sizeof msg-1:a->infostr_len);
        *a->nstop=(cdmlib_ip)1;
        return 1;
    }
    for(int component=0;component<3;++component) {
        cdmlib_zero_n(buffer,(size_t)memory_size);
        for(int i=0;i<nd;++i) {
            const int dip=tab[i]-1;
            buffer[i]=dip>=0?values[component][dip]:0.0;
        }
        if(cdmlib_hdf5_write1d_c99(group_id,names[component],buffer,nd,
                                    memory_size)!=0) {
            free(buffer);
            static const char msg[]="Error writing HDF5 force/torque density";
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?sizeof msg-1:a->infostr_len);
            *a->nstop=(cdmlib_ip)1;
            return 1;
        }
    }
    free(buffer);
    return 0;
}
//-------------------------------------------

static int cdmlib_write_hdf5_poynting_dataset_c99(
    const cdmlib_abi_args_c99 *a,const int64_t group_id,const char *name,
    const REAL *data,const int count,const int memory_size)
{
    if(cdmlib_hdf5_write1d_c99(group_id,name,data,(int32_t)count,
                                (int32_t)memory_size)==0) return 0;
    static const char msg[]="Error writing HDF5 Poynting output";
    memset(a->infostr,' ',a->infostr_len);
    memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
           sizeof msg-1:a->infostr_len);
    *a->nstop=(cdmlib_ip)1;
    return 1;
}
//-------------------------------------------

static int cdmlib_compute_force_plane_c99(
    const cdmlib_abi_args_c99 *a, const cdmlib_initphase_c99 *init,
    const cdmlib_config_c99 *cfg,
    const cdmlib_entry_runtime_c99 *runtime)
{
    const int nx=runtime->nx>0?runtime->nx:init->nxm;
    const int ny=runtime->ny>0?runtime->ny:init->nym;
    const int nz=runtime->nz>0?runtime->nz:init->nzm;
    const int nx2=runtime->nx2>0?runtime->nx2:2*nx;
    const int ny2=runtime->ny2>0?runtime->ny2:2*ny;
    const int nz2=runtime->nz2>0?runtime->nz2:2*nz;
    const int nxy2=runtime->nxy2>0?runtime->nxy2:nx2*ny2;
    const int ntotal=runtime->ntotal>0?runtime->ntotal:nx2*ny2*nz2;
    const int ntotalm=init->ntotalm,nmax=init->nmax;
    const int nb=(int)*a->nbsphere,nd=(int)*a->ndipole;
    const int nxm=init->nxm,nym=init->nym,nzm=init->nzm;
    const int nmaxp=init->nmax;
    const int linear=memcmp(a->beam,"pwavelinear",11)==0;
    const int circular=memcmp(a->beam,"pwavecircular",13)==0;
    const int multimode=memcmp(a->beam,"wavelinearmulti",15)==0;
    const int antenna=memcmp(a->beam,"antenna",7)==0;
    const int gausslinear=memcmp(a->beam,"gwavelinear",11)==0;
    const int gausscircular=memcmp(a->beam,"gwavecircular",13)==0;
    const int paralinear=memcmp(a->beam,"gparawavelinear",15)==0;
    const int paracircular=memcmp(a->beam,"gparawavecircular",17)==0;
    const int arbitrary=memcmp(a->beam,"arbitrary",9)==0;
    const int fftbeam=memcmp(a->beam,"gfftwave",8)==0;
    const int speckle=memcmp(a->beam,"speckle",7)==0;
    const int demispeckle=memcmp(a->beam,"demispeckle",11)==0;
    const int confocal=memcmp(a->beam,"confocal",8)==0;
    const int demiconfocal=memcmp(a->beam,"demiconfocal",12)==0;
    const int fieldderivative=fftbeam||speckle||demispeckle||confocal||demiconfocal;
    if(!linear && !circular && !multimode && !antenna &&
       !gausslinear && !gausscircular &&
       !paralinear && !paracircular && !arbitrary && !fieldderivative) {
        static const char msg[]="C99 optical force beam branch not connected";
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
               sizeof msg-1:a->infostr_len);
        *a->nstop=(cdmlib_ip)1;
        return 1;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    if(arbitrary && (nx<5 || ny<5 || nz<5)) {
        static const char msg[]="Discretization too small for the derivative";
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
               sizeof msg-1:a->infostr_len);
        *a->nstop=(cdmlib_ip)1;
        return 1;
    }

    COMPLEX *vx=runtime->vectx,*vy=runtime->vecty,*vz=runtime->vectz;
    int own_workspace=0;
    if(!vx || !vy || !vz) {
        vx=(COMPLEX *)calloc((size_t)ntotalm,sizeof *vx);
        vy=(COMPLEX *)calloc((size_t)ntotalm,sizeof *vy);
        vz=(COMPLEX *)calloc((size_t)ntotalm,sizeof *vz);
        if(!vx || !vy || !vz) {
            free(vx); free(vy); free(vz);
            static const char msg[]="Could not allocate optical force FFT workspace";
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
            *a->nstop=(cdmlib_ip)1;
            return 1;
        }
        own_workspace=1;
    }
    const COMPLEX *ff=(const COMPLEX *)a->ff;
    const int64_t *planb=&runtime->planb,*planf=&runtime->planf;
    if(cfg->nproche==-1)
        calculforcefft_(&nx,&ny,&nz,&nx2,&ny2,&nz2,&nxy2,&nxm,&nym,&nzm,
                        (const int *)a->tabdip,vx,vy,vz,ff,planb);
    else
        calculforcefftopt_(&nx,&ny,&nz,&nx2,&ny2,&nz2,&nxy2,&nxm,&nym,&nzm,
                           vx,vy,vz,ff,planb);

    if(arbitrary)
        incidentarbitrarydercreate_(a->namefileinc,a->namefileinc_len);

    COMPLEX *ff0=(COMPLEX *)a->ff0;
    COMPLEX *wrk=(COMPLEX *)a->wrk;
    const REAL arete=(REAL)*a->aretecube,k0=(REAL)*a->k0;
    const REAL theta=(REAL)*a->theta,phi=(REAL)*a->phi;
    const REAL ss=(REAL)*a->ss,pp=(REAL)*a->pp;
    const COMPLEX e0=(COMPLEX)*a->e0;
    const REAL x0=(REAL)*a->xgaus,y0=(REAL)*a->ygaus,z0=(REAL)*a->zgaus;
    const REAL w0=(REAL)*a->w0;
    const REAL tol=(REAL)*a->tolinit;
    const int nloin=0;
    if(fieldderivative) {
        if(!a->xi) {
            if(own_workspace) { free(vx); free(vy); free(vz); }
            static const char msg[]="Missing incident derivative workspace";
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
            *a->nstop=(cdmlib_ip)1;
            return 1;
        }
        for(int derivative=1;derivative<=3;++derivative) {
            derivechamp_(&nx,&ny,&nz,&nmaxp,&arete,ff0,&derivative,
                (COMPLEX *)a->xi);
            for(int i=0;i<nd;++i) {
                const size_t src=3u*(size_t)i;
                for(int component=0;component<3;++component)
                    wrk[(size_t)i+(size_t)(3*(derivative-1)+component)*
                        3u*(size_t)nmaxp]=((COMPLEX *)a->xi)[src+
                        (size_t)component];
            }
        }
    }
    for(int axis=1;axis<=3;++axis) {
        derivativefield2_(&arete,&k0,&nx,&ny,&nz,&nx2,&ny2,&nz2,&nxy2,
            &ntotal,&ntotalm,&nmax,vx,vy,vz,&axis,ff0,planb,planf);
        for(int i=0;i<nd;++i) {
            const int j=((const int *)a->tabdip)[i]-1;
            if(j<0) continue;
            COMPLEX eder[3][3]={{0.0}};
            REAL x=((const REAL *)a->xs)[j];
            REAL y=((const REAL *)a->ys)[j];
            REAL z=((const REAL *)a->zs)[j];
            REAL wave=k0,ssv=ss,ppv=pp,thetav=theta,phiv=phi;
            if(linear)
                ondeplaned_(&x,&y,&z,&wave,&e0,&ssv,&ppv,&thetav,&phiv,
                            &axis,eder);
            else if(circular)
                ondecirced_(&x,&y,&z,&wave,&e0,&ssv,&thetav,&phiv,&axis,eder);
            else if(multimode) {
                int stop=0;
                ondeplanedmulti_(&x,&y,&z,&wave,
                    (const COMPLEX *)a->e0m,
                    (const REAL *)a->ssm,(const REAL *)a->ppm,
                    (const REAL *)a->thetam,(const REAL *)a->phim,
                    (const int *)a->nbinc,&axis,eder,&stop,a->infostr,
                    a->infostr_len);
                if(stop!=0) {
                    if(own_workspace) { free(vx); free(vy); free(vz); }
                    *a->nstop=(cdmlib_ip)stop;
                    return 1;
                }
            }
            else if(antenna) {
                int stop=0;
                dipoleincder_(&runtime->antenna_x,&runtime->antenna_y,
                    &runtime->antenna_z,&thetav,&phiv,&x,&y,&z,&arete,&wave,
                    &e0,&eder[0][0],&axis,&stop,a->infostr,a->infostr_len);
                if(stop!=0) {
                    if(own_workspace) { free(vx); free(vy); free(vz); }
                    *a->nstop=(cdmlib_ip)stop;
                    return 1;
                }
            }
            else if(arbitrary)
                incidentarbitraryder1_(&axis,&x,&y,&z,a->namefileinc,
                    &eder[0][0],a->namefileinc_len);
            else if(fieldderivative) {
                for(int component=0;component<3;++component)
                    eder[axis-1][component]=
                        wrk[(size_t)j+(size_t)(3*(axis-1)+component)*
                            3u*(size_t)nmaxp];
            }
            else if(gausslinear)
                gaussianchampd_(&x,&y,&z,&x0,&y0,&z0,&thetav,&phiv,&w0,
                    &wave,&ssv,&ppv,&e0,&eder[0][0],&tol,&nloin);
            else if(gausscircular)
                gaussianchampdcirc_(&x,&y,&z,&x0,&y0,&z0,&thetav,&phiv,
                    &w0,&wave,&ssv,&e0,&eder[0][0],&tol,&nloin);
            else {
                int stop=0;
                if(paralinear)
                    gaussianparalineard_(&x,&y,&z,&x0,&y0,&z0,&thetav,&phiv,
                        &w0,&wave,&ssv,&ppv,&e0,&eder[0][0],&stop,a->infostr,
                        a->infostr_len);
                else
                    gaussianparacircd_(&x,&y,&z,&x0,&y0,&z0,&thetav,&phiv,
                        &w0,&wave,&ssv,&e0,&eder[0][0],&stop,a->infostr,
                        a->infostr_len);
                if(stop!=0) {
                    if(own_workspace) { free(vx); free(vy); free(vz); }
                    *a->nstop=(cdmlib_ip)stop;
                    return 1;
                }
            }
            const size_t field=(size_t)3*(size_t)i;
            const size_t row=(size_t)(axis-1)*(size_t)nmaxp+(size_t)j;
            for(int component=0;component<3;++component)
                wrk[row+(size_t)component*3u*(size_t)nmaxp]=
                    eder[axis-1][component]+ff0[field+(size_t)component];
        }
    }
    for(int i=0;i<3*nb;++i) ff0[i]=conj(ff[i]);
    const REAL quatpieps0=1.0/(299792458.0*299792458.0*1.0e-7);
    REAL force[3]={0.0,0.0,0.0};
    int gpu_force_status=1;
    if(nb>0 && nmaxp>=nb) {
        ifdda_compute_optical_force_cuda_(ff,wrk,&nb,&nmaxp,&quatpieps0,
            (REAL *)a->forcex,(REAL *)a->forcey,(REAL *)a->forcez,force,
            &gpu_force_status);
    }
    if(gpu_force_status!=0) {
        for(int i=0;i<nb;++i) {
            const size_t k=3u*(size_t)i;
            for(int axis=0;axis<3;++axis) {
                const size_t row=(size_t)axis*(size_t)nmaxp+(size_t)i;
                const REAL f=0.5*creal(ff0[k]*wrk[row]+
                    ff0[k+1]*wrk[row+3u*(size_t)nmaxp]+
                    ff0[k+2]*wrk[row+6u*(size_t)nmaxp])*quatpieps0;
                ((REAL *)a->forcex)[i]=axis==0?f:((REAL *)a->forcex)[i];
                ((REAL *)a->forcey)[i]=axis==1?f:((REAL *)a->forcey)[i];
                ((REAL *)a->forcez)[i]=axis==2?f:((REAL *)a->forcez)[i];
                force[axis]+=f;
            }
        }
    }
    if(cfg->nforced==1) {
        if(cfg->nmat==0) {
            if(cdmlib_write_density_triplet_c99(a,nd,a->forcex,a->forcey,
                a->forcez,"forcex.mat","forcey.mat","forcez.mat")!=0) {
                if(own_workspace) { free(vx); free(vy); free(vz); }
                return 1;
            }
        } else if(cfg->nmat==2) {
            static const char *const names[3]={"Optical force x component",
                "Optical force y component","Optical force z component"};
            if(cdmlib_write_hdf5_density_triplet_c99(a,init,runtime->group_idof,
                nd,a->forcex,a->forcey,a->forcez,names)!=0) {
                if(own_workspace) { free(vx); free(vy); free(vz); }
                return 1;
            }
        }
    }
    for(int axis=0;axis<3;++axis) ((REAL *)a->forcet)[axis]=force[axis];
    *a->forcem=sqrt(force[0]*force[0]+force[1]*force[1]+force[2]*force[2]);
    if(cfg->ntorque==1) {
        const COMPLEX *fl=(const COMPLEX *)a->ffloc;
        const REAL *xs=(const REAL *)a->xs;
        const REAL *ys=(const REAL *)a->ys;
        const REAL *zs=(const REAL *)a->zs;
        const REAL k03=k0*k0*k0;
        REAL center[3]={0.0,0.0,0.0},torque[3]={0.0,0.0,0.0};
        int gpu_torque_status=1;
        if(nb>0 && gpu_force_status==0) {
            ifdda_compute_optical_torque_cuda_(fl,xs,ys,zs,&nb,&k03,&quatpieps0,
                center,(REAL *)a->torquex,(REAL *)a->torquey,(REAL *)a->torquez,
                torque,&gpu_torque_status);
        }
        if(gpu_torque_status!=0) {
          for(int i=0;i<nb;++i) {
            center[0]+=xs[i]; center[1]+=ys[i]; center[2]+=zs[i];
          }
          for(int axis=0;axis<3;++axis) center[axis]/=(REAL)nb;
          for(int i=0;i<nb;++i) {
            const size_t k=3u*(size_t)i;
            COMPLEX em[3];
            for(int j=0;j<3;++j)
                em[j]=(2.0/3.0)*I*k03*ff[k+(size_t)j]+fl[k+(size_t)j];
            const REAL tx=0.5*creal(ff0[k+1]*em[2]-ff0[k+2]*em[1])*
                quatpieps0+(ys[i]-center[1])*((REAL *)a->forcez)[i]-
                (zs[i]-center[2])*((REAL *)a->forcey)[i];
            const REAL ty=0.5*creal(-ff0[k]*em[2]+ff0[k+2]*em[0])*
                quatpieps0+(zs[i]-center[2])*((REAL *)a->forcex)[i]-
                (xs[i]-center[0])*((REAL *)a->forcez)[i];
            const REAL tz=0.5*creal(ff0[k]*em[1]-ff0[k+1]*em[0])*
                quatpieps0+(xs[i]-center[0])*((REAL *)a->forcey)[i]-
                (ys[i]-center[1])*((REAL *)a->forcex)[i];
            ((REAL *)a->torquex)[i]=tx;
            ((REAL *)a->torquey)[i]=ty;
            ((REAL *)a->torquez)[i]=tz;
            torque[0]+=tx; torque[1]+=ty; torque[2]+=tz;
          }
        }
        if(cfg->ntorqued==1) {
            if(cfg->nmat==0) {
                if(cdmlib_write_density_triplet_c99(a,nd,a->torquex,a->torquey,
                    a->torquez,"torquex.mat","torquey.mat","torquez.mat")!=0) {
                    if(own_workspace) { free(vx); free(vy); free(vz); }
                    return 1;
                }
            } else if(cfg->nmat==2) {
                static const char *const names[3]={"Optical torque x component",
                    "Optical torque y component","Optical torque z component"};
                if(cdmlib_write_hdf5_density_triplet_c99(a,init,runtime->group_idof,
                    nd,a->torquex,a->torquey,a->torquez,names)!=0) {
                    if(own_workspace) { free(vx); free(vy); free(vz); }
                    return 1;
                }
            }
        }
        for(int axis=0;axis<3;++axis) ((REAL *)a->couplet)[axis]=torque[axis];
        *a->couplem=sqrt(torque[0]*torque[0]+torque[1]*torque[1]+
                         torque[2]*torque[2]);
    }
    if(own_workspace) { free(vx); free(vy); free(vz); }
    return 0;
}














//-------------------------------------------


static int cdmlib_compute_quick_poynting_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,const cdmlib_entry_runtime_c99 *r)
{
    if(!a||!s||!c||!r||c->nsectionsca!=1||c->nquickdiffracte!=1) return 0;
    if(!a->nfft2d||!a->tabfft2||!a->k0||!a->aretecube||!a->xs||!a->ys||
       !a->zs||!a->ff||!a->efourierx||!a->efouriery||!a->efourierz||
       !a->ediffkzpos||!a->ediffkzneg||!a->npoynting||
       !a->deltapoyntingx||!a->deltapoyntingy||!a->poyntingfieldpos||
       !a->poyntingfieldneg||!a->kxypoynting||!a->poyntingfield||
       !a->thetafield||!a->phifield||!a->ntheta||!a->nphi||!a->nstop||
       !a->infostr||!a->gasym||!a->cscai) {
        static const char msg[]="Missing buffer for C99 FFT Poynting calculation";
        if(a&&a->infostr) {
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
        }
        if(a&&a->nstop) *a->nstop=(cdmlib_ip)1;
        return 1;
    }
    const int nx=r->nx,ny=r->ny,nz=r->nz,nxm=s->nxm,nym=s->nym,nzm=s->nzm;
    const int nfft=(int)*a->nfft2d,ntheta=(int)*a->ntheta,nphi=(int)*a->nphi;
    int imax=0,nstop=0;
    REAL k0=(REAL)*a->k0,arete=(REAL)*a->aretecube;
    REAL dkx=0.0,dky=0.0;
    int64_t plan2f=r->plan2f,plan2b=r->plan2b;
    const size_t plane=(size_t)nfft*(size_t)nfft;
    const REAL constpoynting=pow(k0,4.0)*299792458.0/
        (8.0*3.14159265358979323846)*
        (1.0/(299792458.0*299792458.0*1.0e-7));
    int output_np=0,cnt=(ntheta+1)*nphi,gpu_status=1;
    ifdda_quick_diffraction_cuda_(&nx,&ny,&nz,&nfft,(int *)a->tabfft2,&k0,
        (const REAL *)a->xs,(const REAL *)a->ys,(const REAL *)a->zs,&arete,
        (const COMPLEX *)a->ff,&ntheta,&nphi,(const REAL *)&r->I0,
        (const REAL *)a->theta,(const REAL *)a->phi,&constpoynting,&imax,&dkx,&dky,
        (REAL *)a->poyntingfieldpos,(REAL *)a->poyntingfieldneg,
        (REAL *)a->kxypoynting,(REAL *)a->poyntingfield,(REAL *)a->thetafield,
        (REAL *)a->phifield,(REAL *)a->gasym,(REAL *)a->cscai,&gpu_status);
    if(gpu_status==0) {
        output_np=2*imax+1;
        *a->nstop=(cdmlib_ip)0;
        *a->npoynting=(cdmlib_ip)output_np;
        *a->deltapoyntingx=(cdmlib_real)dkx;
        *a->deltapoyntingy=(cdmlib_real)dky;
        if(memcmp(a->beam,"pwavelinear",11)!=0&&
           memcmp(a->beam,"pwavecircular",13)!=0) {
            *a->gasym=(cdmlib_real)0;
            *a->cscai=(cdmlib_real)0;
        }
    } else {
    diffractefft2d_(&nx,&ny,&nz,&nxm,&nym,&nzm,&nfft,(int *)a->tabfft2,
        &k0,(const REAL *)a->xs,(const REAL *)a->ys,(const REAL *)a->zs,
        &arete,(COMPLEX *)a->efourierx,(COMPLEX *)a->efouriery,
        (COMPLEX *)a->efourierz,(const COMPLEX *)a->ff,&imax,
        &dkx,&dky,(COMPLEX *)a->ediffkzpos,
        (COMPLEX *)a->ediffkzneg,&plan2f,&plan2b,&nstop,a->infostr,
        a->infostr_len);
    *a->nstop=(cdmlib_ip)nstop;
    if(nstop!=0) return 1;
    output_np=2*imax+1;
    const int np=output_np;
    *a->npoynting=(cdmlib_ip)output_np;
    *a->deltapoyntingx=(cdmlib_real)dkx;
    *a->deltapoyntingy=(cdmlib_real)dky;
    for(int i=-imax;i<=imax;++i) {
        ((REAL *)a->kxypoynting)[i+imax]=(REAL)i*dkx;
        for(int j=-imax;j<=imax;++j) {
            const size_t out=(size_t)(i+imax)+(size_t)np*(size_t)(j+imax);
            const int ii=imax+i,jj=imax+j;
            const REAL kx=(REAL)i*dkx,ky=(REAL)j*dky;
            if(k0*k0-kx*kx-ky*ky>=0.0) {
                const size_t at=(size_t)ii+(size_t)nfft*(size_t)jj;
                REAL ep=0.0,en=0.0;
                for(int q=0;q<3;++q) {
                    const size_t z=at+plane*(size_t)q;
                    const COMPLEX zp=((COMPLEX *)a->ediffkzpos)[z];
                    const COMPLEX zn=((COMPLEX *)a->ediffkzneg)[z];
                    ep+=creal(zp*conj(zp)); en+=creal(zn*conj(zn));
                }
                ((REAL *)a->poyntingfieldpos)[out]=ep*constpoynting;
                ((REAL *)a->poyntingfieldneg)[out]=en*constpoynting;
            } else {
                ((REAL *)a->poyntingfieldpos)[out]=0.0;
                ((REAL *)a->poyntingfieldneg)[out]=0.0;
            }
        }
    }
    REAL theta=(REAL)*a->theta,phi=(REAL)*a->phi;
    REAL g=0.0,csca=0.0;
    computegcfft2d_(&imax,&dkx,&dky,&k0,(REAL *)&r->I0,&theta,&phi,&nfft,
        (const COMPLEX *)a->ediffkzpos,
        (const COMPLEX *)a->ediffkzneg,&g,&csca);
    if(memcmp(a->beam,"pwavelinear",11)!=0&&
       memcmp(a->beam,"pwavecircular",13)!=0) { g=0.0; csca=0.0; }
    *a->gasym=(cdmlib_real)g;
    *a->cscai=(cdmlib_real)csca;
    const REAL pi=3.141592653589793238462643383279502884;
    const REAL dtheta=pi/(REAL)ntheta,dphi=2.0*pi/(REAL)nphi;
    int cnt=0;
    for(int itheta=0;itheta<=ntheta;++itheta) {
        const REAL th=dtheta*(REAL)itheta;
        for(int iphi=0;iphi<nphi;++iphi) {
            const REAL ph=dphi*(REAL)iphi;
            const size_t out=(size_t)cnt++;
            ((REAL *)a->thetafield)[out]=th*180.0/pi;
            ((REAL *)a->phifield)[out]=ph*180.0/pi;
            const REAL kx=k0*sin(th)*cos(ph),ky=k0*sin(th)*sin(ph);
            int i2=(int)floor(kx/dkx),j2=(int)floor(ky/dkx);
            if(i2 < -imax) i2=-imax;
            if(i2 > imax) i2=imax;
            if(j2 < -imax) j2=-imax;
            if(j2 > imax) j2=imax;
            const int ii=imax+i2,jj=imax+j2;
            const COMPLEX *field=th<=pi/2.0?
                (const COMPLEX *)a->ediffkzpos:
                (const COMPLEX *)a->ediffkzneg;
            REAL e[4]={0.0,0.0,0.0,0.0};
            const int oi[4]={ii,ii+1,ii+1,ii},oj[4]={jj,jj+1,jj,jj+1};
            int zeros=0;
            for(int q=0;q<4;++q) {
                const size_t base=(size_t)oi[q]+(size_t)nfft*(size_t)oj[q];
                for(int comp=0;comp<3;++comp) {
                    const COMPLEX z=field[base+plane*(size_t)comp];
                    e[q]+=creal(z*conj(z));
                }
                if(e[q]==0.0) ++zeros;
            }
            const REAL wx=(kx-(REAL)i2*dkx)/dkx;
            const REAL wy=(ky-(REAL)j2*dkx)/dkx;
            REAL emod;
            if(zeros==0)
                emod=(e[2]-e[0])*wx+(e[3]-e[0])*wy+
                     (e[0]+e[1]-e[3]-e[2])*wx*wy+e[0];
            else {
                REAL sum=0.0;
                for(int q=0;q<4;++q) sum+=e[q];
                emod=sum/(4.0-(REAL)zeros);
            }
            ((REAL *)a->poyntingfield)[out]=emod*constpoynting;
        }
    }
    }
    const int np=output_np;
    if(c->nmat==2) {
        const int nfmem=nfft*nfft;
        const int afmem=(ntheta+1)*nphi>nfmem?(ntheta+1)*nphi:nfmem;
        if(cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "kx Poynting",(const REAL *)a->kxypoynting,np,nfft) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "ky Poynting",(const REAL *)a->kxypoynting,np,nfft) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "Poynting positive",(const REAL *)a->poyntingfieldpos,
              np*np,nfmem) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "Poynting negative",(const REAL *)a->poyntingfieldneg,
              np*np,nfmem) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "Poynting",(const REAL *)a->poyntingfield,cnt,afmem) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "theta for Poynting",(const REAL *)a->thetafield,cnt,afmem) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "phi for Poynting",(const REAL *)a->phifield,cnt,afmem))
            return 1;
    }
    if(c->nmat==0) {
        FILE *files[4]={fopen("poyntingpos.mat","w"),
            fopen("poyntingneg.mat","w"),fopen("kx.mat","w"),
            fopen("ky.mat","w")};
        for(int q=0;q<4;++q) if(!files[q]) {
            for(int t=0;t<4;++t) if(files[t]) fclose(files[t]);
            static const char msg[]="Could not write FFT Poynting files";
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
            *a->nstop=(cdmlib_ip)1;
            return 1;
        }
        for(int i=0;i<np;++i) for(int j=0;j<np;++j) {
            const int k=i+np*j;
            fprintf(files[0]," %.16E\n",((REAL *)a->poyntingfieldpos)[k]);
            fprintf(files[1]," %.16E\n",((REAL *)a->poyntingfieldneg)[k]);
        }
        for(int i=0;i<np;++i) {
            fprintf(files[2]," %.16E\n",((REAL *)a->kxypoynting)[i]);
            fprintf(files[3]," %.16E\n",((REAL *)a->kxypoynting)[i]);
        }
        for(int q=0;q<4;++q) fclose(files[q]);
        FILE *angular[3]={fopen("poynting.mat","w"),fopen("theta.mat","w"),
                          fopen("phi.mat","w")};
        for(int q=0;q<3;++q) if(!angular[q]) {
            for(int t=0;t<3;++t) if(angular[t]) fclose(angular[t]);
            static const char msg[]="Could not write angular Poynting files";
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
            *a->nstop=(cdmlib_ip)1;
            return 1;
        }
        for(int i=0;i<cnt;++i) {
            fprintf(angular[0]," %.16E\n",((REAL *)a->poyntingfield)[i]);
            fprintf(angular[1]," %.16E\n",((REAL *)a->thetafield)[i]);
            fprintf(angular[2]," %.16E\n",((REAL *)a->phifield)[i]);
        }
        for(int q=0;q<3;++q) fclose(angular[q]);
    }
    return 0;
}
//-------------------------------------------

static int cdmlib_compute_slow_poynting_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,const cdmlib_entry_runtime_c99 *r)
{
    if(!a||!s||!c||!r||c->nsectionsca!=1||c->nquickdiffracte!=0) return 0;
    if(!a->ntheta||!a->nphi||!a->nfft2d||!a->nbsphere||!a->k0||!a->xs||!a->ys||!a->zs||
       !a->ff||!a->theta||!a->phi||!a->npoynting||!a->deltapoyntingx||
       !a->deltapoyntingy||!a->poyntingfield||!a->thetafield||!a->phifield||
       !a->kxypoynting||!a->poyntingfieldpos||!a->poyntingfieldneg||
       !a->gasym||!a->cscai||!a->nstop||!a->infostr) {
        static const char msg[]="Missing buffer for C99 direct Poynting calculation";
        if(a&&a->infostr) {
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
        }
        if(a&&a->nstop) *a->nstop=(cdmlib_ip)1;
        return 1;
    }
    const int nd=(int)*a->nbsphere,ntheta=(int)*a->ntheta,nphi=(int)*a->nphi;
    if(ntheta<=10||nphi<=20) {
        static const char msg[]="ntheta and nphi too small!";
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
               sizeof msg-1:a->infostr_len);
        *a->nstop=(cdmlib_ip)1;
        return 1;
    }
    const REAL pi=3.141592653589793238462643383279502884;
    const REAL twopi=2.0*pi,k0=(REAL)*a->k0;
    const REAL theta0=(REAL)*a->theta*pi/180.0;
    const REAL phi0=(REAL)*a->phi*pi/180.0;
    const REAL dtheta=pi/(REAL)ntheta,dphi=twopi/(REAL)nphi;
    const REAL c0=299792458.0,quatpieps0=1.0/(c0*c0*1.0e-7);
    const REAL constpoynting=pow(k0,4.0)*c0/(8.0*pi)*quatpieps0;
    const COMPLEX *ff=(const COMPLEX *)a->ff;
    const REAL *xs=(const REAL *)a->xs,*ys=(const REAL *)a->ys;
    const REAL *zs=(const REAL *)a->zs;
    const int halfden=(ntheta-1<nphi-1?ntheta-1:nphi-1)/2;
    const REAL dp=k0/(REAL)halfden;
    const int ll=(int)(k0/dp),np=2*ll+1;
    *a->deltapoyntingx=(cdmlib_real)dp;
    *a->deltapoyntingy=(cdmlib_real)dp;
    *a->npoynting=(cdmlib_ip)np;
    const int cnt=(ntheta+1)*nphi;
    const REAL theta_degrees=(REAL)*a->theta,phi_degrees=(REAL)*a->phi;
    const REAL intensity=(REAL)r->I0;
    REAL cscai=0.0,gasym=0.0;
    int gpu_status=1;
    ifdda_direct_scattering_cuda_(ff,xs,ys,zs,&nd,a->ntheta,a->nphi,
        a->nfft2d,&k0,&theta_degrees,&phi_degrees,&intensity,&constpoynting,
        (REAL *)a->poyntingfield,(REAL *)a->thetafield,(REAL *)a->phifield,
        (REAL *)a->poyntingfieldpos,(REAL *)a->poyntingfieldneg,
        (REAL *)a->kxypoynting,&gasym,&cscai,&gpu_status);
    if(gpu_status==0) {
        if(memcmp(a->beam,"pwavelinear",11)!=0&&
           memcmp(a->beam,"pwavecircular",13)!=0) { cscai=0.0; gasym=0.0; }
    } else {
        std::fprintf(stderr,
            "CUDA direct scattering failed (status %d); using CPU fallback.\n",
            gpu_status);
        cscai=0.0; gasym=0.0;
        int cpu_count=0;
        for(int itheta=0;itheta<=ntheta;++itheta) {
            const REAL th=dtheta*(REAL)itheta,st=sin(th),ct=cos(th);
            for(int iphi=0;iphi<nphi;++iphi) {
                const REAL ph=dphi*(REAL)iphi,cp=cos(ph),sp=sin(ph);
                const REAL nx=st*cp,ny=st*sp,nz=ct;
                COMPLEX ex=0.0,ey=0.0,ez=0.0;
                for(int dip=0;dip<nd;++dip) {
                    const size_t k=(size_t)3*(size_t)dip;
                    const COMPLEX phase=cexp(-I*k0*(nx*xs[dip]+ny*ys[dip]+nz*zs[dip]));
                    const COMPLEX longitudinal=nx*ff[k]+ny*ff[k+1]+nz*ff[k+2];
                    ex+=phase*(ff[k]-longitudinal*nx);
                    ey+=phase*(ff[k+1]-longitudinal*ny);
                    ez+=phase*(ff[k+2]-longitudinal*nz);
                }
                const REAL emod=creal(ex*conj(ex)+ey*conj(ey)+ez*conj(ez));
                const size_t at=(size_t)cpu_count++;
                ((REAL *)a->poyntingfield)[at]=emod*constpoynting;
                ((REAL *)a->thetafield)[at]=th*180.0/pi;
                ((REAL *)a->phifield)[at]=ph*180.0/pi;
                const REAL weight=dphi*dtheta*st*emod;
                cscai+=weight;
                gasym+=weight*(nx*sin(theta0)*cos(phi0)+
                               ny*sin(theta0)*sin(phi0)+nz*cos(theta0));
            }
        }
        if(cscai!=0.0) gasym/=cscai;
        cscai*=pow(k0,4.0)/r->I0;
        for(int i=-ll;i<=ll;++i) {
            ((REAL *)a->kxypoynting)[i+ll]=(REAL)i*dp;
            for(int j=-ll;j<=ll;++j) {
                const size_t out=(size_t)(i+ll)+(size_t)np*(size_t)(j+ll);
                const REAL kx=(REAL)i*dp,ky=(REAL)j*dp;
                if(k0*k0-kx*kx-ky*ky>0.0) {
                    const REAL nz=sqrt(1.0-(kx*kx+ky*ky)/(k0*k0));
                    COMPLEX px=0.0,py=0.0,pz=0.0;
                    COMPLEX mx=0.0,my=0.0,mz=0.0;
                    for(int dip=0;dip<nd;++dip) {
                        const size_t k=(size_t)3*(size_t)dip;
                        COMPLEX phase=cexp(-I*k0*((kx/k0)*xs[dip]+
                            (ky/k0)*ys[dip]+nz*zs[dip]));
                        COMPLEX longitudinal=(kx/k0)*ff[k]+
                            (ky/k0)*ff[k+1]+nz*ff[k+2];
                        px+=phase*(ff[k]-longitudinal*(kx/k0));
                        py+=phase*(ff[k+1]-longitudinal*(ky/k0));
                        pz+=phase*(ff[k+2]-longitudinal*nz);
                        phase=cexp(-I*k0*((kx/k0)*xs[dip]+
                            (ky/k0)*ys[dip]-nz*zs[dip]));
                        longitudinal=(kx/k0)*ff[k]+(ky/k0)*ff[k+1]-nz*ff[k+2];
                        mx+=phase*(ff[k]-longitudinal*(kx/k0));
                        my+=phase*(ff[k+1]-longitudinal*(ky/k0));
                        mz+=phase*(ff[k+2]+longitudinal*nz);
                    }
                    ((REAL *)a->poyntingfieldpos)[out]=
                        creal(px*conj(px)+py*conj(py)+pz*conj(pz))*constpoynting;
                    ((REAL *)a->poyntingfieldneg)[out]=
                        creal(mx*conj(mx)+my*conj(my)+mz*conj(mz))*constpoynting;
                } else {
                    ((REAL *)a->poyntingfieldpos)[out]=0.0;
                    ((REAL *)a->poyntingfieldneg)[out]=0.0;
                }
            }
        }
        if(memcmp(a->beam,"pwavelinear",11)!=0&&
           memcmp(a->beam,"pwavecircular",13)!=0) { cscai=0.0; gasym=0.0; }
    }
    *a->cscai=(cdmlib_real)cscai;
    *a->gasym=(cdmlib_real)gasym;
    if(std::getenv("IFDDA_TRACE_SLOW_SCATTERING")) {
        REAL angular_sum=REAL(0),angular_max=REAL(0),positive_sum=REAL(0),positive_max=REAL(0);
        REAL negative_sum=REAL(0),negative_max=REAL(0);
        const REAL *angular_values=(const REAL *)a->poyntingfield;
        const REAL *positive_values=(const REAL *)a->poyntingfieldpos;
        const REAL *negative_values=(const REAL *)a->poyntingfieldneg;
        for(int i=0;i<cnt;++i) {
            angular_sum+=angular_values[i];
            angular_max=std::max(angular_max,angular_values[i]);
        }
        const int spectrum_count=np*np;
        for(int i=0;i<spectrum_count;++i) {
            positive_sum+=positive_values[i];
            positive_max=std::max(positive_max,positive_values[i]);
            negative_sum+=negative_values[i];
            negative_max=std::max(negative_max,negative_values[i]);
        }
        std::fprintf(stdout,
            "CDMLIB_SLOW_SCATTERING %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
            double(cscai),double(gasym),double(angular_sum),double(angular_max),
            double(positive_sum),double(positive_max),double(negative_sum),double(negative_max));
    }
    if(c->nmat==2) {
        const int nfmem=(int)*a->nfft2d*(int)*a->nfft2d;
        const int afmem=(ntheta+1)*nphi>nfmem?(ntheta+1)*nphi:nfmem;
        if(cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "Poynting",(const REAL *)a->poyntingfield,cnt,afmem) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "theta for Poynting",(const REAL *)a->thetafield,cnt,afmem) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "phi for Poynting",(const REAL *)a->phifield,cnt,afmem) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "kx Poynting",(const REAL *)a->kxypoynting,np,
              (int)*a->nfft2d) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "ky Poynting",(const REAL *)a->kxypoynting,np,
              (int)*a->nfft2d) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "Poynting positive",(const REAL *)a->poyntingfieldpos,
              np*np,nfmem) ||
           cdmlib_write_hdf5_poynting_dataset_c99(a,r->group_idff,
              "Poynting negative",(const REAL *)a->poyntingfieldneg,
              np*np,nfmem)) return 1;
    }
    if(c->nmat==0) {
        FILE *files[7]={fopen("poynting.mat","w"),fopen("theta.mat","w"),
            fopen("phi.mat","w"),fopen("poyntingpos.mat","w"),
            fopen("poyntingneg.mat","w"),fopen("kx.mat","w"),
            fopen("ky.mat","w")};
        for(int q=0;q<7;++q) if(!files[q]) {
            for(int t=0;t<7;++t) if(files[t]) fclose(files[t]);
            static const char msg[]="Could not write direct Poynting files";
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
            *a->nstop=(cdmlib_ip)1;
            return 1;
        }
        for(int i=0;i<cnt;++i) {
            fprintf(files[0]," %.16E\n",((REAL *)a->poyntingfield)[i]);
            fprintf(files[1]," %.16E\n",((REAL *)a->thetafield)[i]);
            fprintf(files[2]," %.16E\n",((REAL *)a->phifield)[i]);
        }
        for(int i=0;i<np;++i) for(int j=0;j<np;++j) {
            const int k=i+np*j;
            fprintf(files[3]," %.16E\n",((REAL *)a->poyntingfieldpos)[k]);
            fprintf(files[4]," %.16E\n",((REAL *)a->poyntingfieldneg)[k]);
        }
        for(int i=0;i<np;++i) {
            fprintf(files[5]," %.16E\n",((REAL *)a->kxypoynting)[i]);
            fprintf(files[6]," %.16E\n",((REAL *)a->kxypoynting)[i]);
        }
        for(int q=0;q<7;++q) fclose(files[q]);
    }
    return 0;
}
//-------------------------------------------

static int cdmlib_write_holographic_field_hdf5_c99(
    const cdmlib_abi_args_c99 *a,const int64_t group_id,const char *prefix,
    const COMPLEX *x,const COMPLEX *y,const COMPLEX *z,
    int nfft,int half,int imax,int compact)
{
    const int np=2*imax+1;
    const int count=compact?np*np:nfft*nfft;
    const int memory_size=nfft*nfft;
    const COMPLEX *fields[3]={x,y,z};
    const char *components[3]={"x","y","z"};
    char name[100];
    REAL *buffer=(REAL *)calloc((size_t)memory_size,sizeof *buffer);
    if(!buffer) goto error;
    for(int part=0;part<7;++part) {
        if(part==0) {
            for(int j=0;j<(compact?np:nfft);++j) for(int i=0;i<(compact?np:nfft);++i) {
                const int src=compact?(i-imax+half)+nfft*(j-imax+half):i+nfft*j;
                const int dst=compact?i+np*j:i+nfft*j;
                const COMPLEX vx=x[src],vy=y[src],vz=z[src];
                buffer[dst]=sqrt(creal(vx*conj(vx)+vy*conj(vy)+vz*conj(vz)));
            }
            snprintf(name,sizeof name,"%s field modulus",prefix);
        } else {
            const int component=(part-1)/2;
            const int imaginary=(part-1)%2;
            for(int j=0;j<(compact?np:nfft);++j) for(int i=0;i<(compact?np:nfft);++i) {
                const int src=compact?(i-imax+half)+nfft*(j-imax+half):i+nfft*j;
                const int dst=compact?i+np*j:i+nfft*j;
                const COMPLEX value=fields[component][src];
                buffer[dst]=imaginary?cimag(value):creal(value);
            }
            snprintf(name,sizeof name,"%s field %s component %s part",prefix,
                     components[component],imaginary?"imaginary":"real");
        }
        if(cdmlib_hdf5_write1d_c99(group_id,name,buffer,count,memory_size)!=0)
            goto error;
        cdmlib_zero_n(buffer,(size_t)memory_size);
    }
    free(buffer);
    return 0;
error:
    free(buffer);
    memset(a->infostr,' ',a->infostr_len);
    memcpy(a->infostr,"Error writing HDF5 Quick Lens fields",36<a->infostr_len?36:a->infostr_len);
    *a->nstop=(cdmlib_ip)1;
    return 1;
}
//-------------------------------------------

static int cdmlib_write_holographic_quicklens_hdf5_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_entry_runtime_c99 *r,
    int nfft,int half,int imax)
{
    if(*a->nquicklens==0) {
        if(cdmlib_hdf5_write1d_c99(r->group_idmic,"x Image",
              (const REAL *)a->xy,nfft,nfft)!=0 ||
           cdmlib_hdf5_write1d_c99(r->group_idmic,"kx Fourier",
              (const REAL *)a->kxy,2*imax+1,nfft)!=0) goto error;
    }
    if(cdmlib_write_holographic_field_hdf5_c99(a,r->group_idmic,"Fourier",
          (const COMPLEX *)a->efourierx,
          (const COMPLEX *)a->efouriery,
          (const COMPLEX *)a->efourierz,nfft,half,imax,1) ||
       cdmlib_write_holographic_field_hdf5_c99(a,r->group_idmic,"Fourier+incident",
          (const COMPLEX *)a->efourierincx,
          (const COMPLEX *)a->efourierincy,
          (const COMPLEX *)a->efourierincz,nfft,half,imax,1) ||
       cdmlib_write_holographic_field_hdf5_c99(a,r->group_idmic,"Image",
          (const COMPLEX *)a->eimagex,
          (const COMPLEX *)a->eimagey,
          (const COMPLEX *)a->eimagez,nfft,half,imax,0) ||
       cdmlib_write_holographic_field_hdf5_c99(a,r->group_idmic,"Image+incident",
          (const COMPLEX *)a->eimageincx,
          (const COMPLEX *)a->eimageincy,
          (const COMPLEX *)a->eimageincz,nfft,half,imax,0)) return 1;
    return 0;
error:
    memset(a->infostr,' ',a->infostr_len);
    memcpy(a->infostr,"Error writing HDF5 Quick Lens coordinates",42<a->infostr_len?42:a->infostr_len);
    *a->nstop=(cdmlib_ip)1;
    return 1;
}
//-------------------------------------------

static int cdmlib_write_holographic_quicklens_files_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_entry_runtime_c99 *r,
    int nfft, int half, int imax)
{
    FILE *files[18]={0};
    static const char *names[18]={"kxfourier.mat","ximage.mat",
        "fourier.mat","fourierx.mat","fouriery.mat","fourierz.mat",
        "fourierinc.mat","fourierincx.mat","fourierincy.mat",
        "fourierincz.mat","image.mat","imagex.mat","imagey.mat",
        "imagez.mat","imageinc.mat","imageincx.mat","imageincy.mat",
        "imageincz.mat"};
    const COMPLEX *fields[6]={(const COMPLEX *)a->efourierx,
        (const COMPLEX *)a->efouriery,
        (const COMPLEX *)a->efourierz,
        (const COMPLEX *)a->efourierincx,
        (const COMPLEX *)a->efourierincy,
        (const COMPLEX *)a->efourierincz};
    const COMPLEX *images[6]={(const COMPLEX *)a->eimagex,
        (const COMPLEX *)a->eimagey,
        (const COMPLEX *)a->eimagez,
        (const COMPLEX *)a->eimageincx,
        (const COMPLEX *)a->eimageincy,
        (const COMPLEX *)a->eimageincz};
    int i,j,k;
    if(*a->nmat==2)
        return cdmlib_write_holographic_quicklens_hdf5_c99(a,r,nfft,half,imax);
    if(*a->nmat!=0) return 0;
    for(k=0;k<18;++k) {
        files[k]=fopen(names[k],"w");
        if(!files[k]) goto error;
    }
    for(i=-half;i<half;++i)
        if(fprintf(files[1]," % .16E\n",((const REAL *)a->xy)[i+half])<0)
            goto error;
    for(i=-imax;i<=imax;++i)
        if(fprintf(files[0]," % .16E\n",((const REAL *)a->kxy)[i+imax])<0)
            goto error;
    for(j=-imax;j<=imax;++j) for(i=-imax;i<=imax;++i) {
        const size_t at=(size_t)(i+half)+(size_t)nfft*(size_t)(j+half);
        const COMPLEX x=fields[0][at],y=fields[1][at],z=fields[2][at];
        const COMPLEX ix=fields[3][at],iy=fields[4][at],iz=fields[5][at];
        if(fprintf(files[2]," % .16E\n",sqrt(creal(x*conj(x)+y*conj(y)+z*conj(z))))<0 ||
           fprintf(files[3]," % .16E % .16E\n",creal(x),cimag(x))<0 ||
           fprintf(files[4]," % .16E % .16E\n",creal(y),cimag(y))<0 ||
           fprintf(files[5]," % .16E % .16E\n",creal(z),cimag(z))<0 ||
           fprintf(files[6]," % .16E\n",sqrt(creal(ix*conj(ix)+iy*conj(iy)+iz*conj(iz))))<0 ||
           fprintf(files[7]," % .16E % .16E\n",creal(ix),cimag(ix))<0 ||
           fprintf(files[8]," % .16E % .16E\n",creal(iy),cimag(iy))<0 ||
           fprintf(files[9]," % .16E % .16E\n",creal(iz),cimag(iz))<0)
            goto error;
    }
    for(j=0;j<nfft;++j) for(i=0;i<nfft;++i) {
        const size_t at=(size_t)i+(size_t)nfft*(size_t)j;
        const COMPLEX x=images[0][at],y=images[1][at],z=images[2][at];
        const COMPLEX ix=images[3][at],iy=images[4][at],iz=images[5][at];
        if(fprintf(files[10]," % .16E\n",sqrt(creal(x*conj(x)+y*conj(y)+z*conj(z))))<0 ||
           fprintf(files[11]," % .16E % .16E\n",creal(x),cimag(x))<0 ||
           fprintf(files[12]," % .16E % .16E\n",creal(y),cimag(y))<0 ||
           fprintf(files[13]," % .16E % .16E\n",creal(z),cimag(z))<0 ||
           fprintf(files[14]," % .16E\n",sqrt(creal(ix*conj(ix)+iy*conj(iy)+iz*conj(iz))))<0 ||
           fprintf(files[15]," % .16E % .16E\n",creal(ix),cimag(ix))<0 ||
           fprintf(files[16]," % .16E % .16E\n",creal(iy),cimag(iy))<0 ||
           fprintf(files[17]," % .16E % .16E\n",creal(iz),cimag(iz))<0)
            goto error;
    }
    for(k=0;k<18;++k) if(fclose(files[k])!=0) {
        files[k]=NULL;
        goto error;
    } else files[k]=NULL;
    return 0;
error:
    for(k=0;k<18;++k) if(files[k]) fclose(files[k]);
    memset(a->infostr,' ',a->infostr_len);
    memcpy(a->infostr,"Could not write Quick Lens output",33<a->infostr_len?33:a->infostr_len);
    *a->nstop=(cdmlib_ip)1;
    return 1;
}
//-------------------------------------------

static int cdmlib_compute_holographic_quicklens_slow_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,const cdmlib_entry_runtime_c99 *r,
    int nfft,int half,int nx,int ny,int nz,int nxm,int nym,int nzm,
    int side,REAL k0,REAL na)
{
    const REAL pi=3.141592653589793238462643383279502884;
    const REAL arete=(REAL)*a->aretecube;
    const size_t plane=(size_t)nfft*(size_t)nfft;
    REAL dkx=2.0*pi/((REAL)nfft*arete),dky,deltax;
    COMPLEX *fx=(COMPLEX *)a->efourierx;
    COMPLEX *fy=(COMPLEX *)a->efouriery;
    COMPLEX *fz=(COMPLEX *)a->efourierz;
    COMPLEX *fix=(COMPLEX *)a->efourierincx;
    COMPLEX *fiy=(COMPLEX *)a->efourierincy;
    COMPLEX *fiz=(COMPLEX *)a->efourierincz;
    COMPLEX *pos=(COMPLEX *)a->ediffkzpos;
    COMPLEX *neg=(COMPLEX *)a->ediffkzneg;
    COMPLEX *ix=(COMPLEX *)a->eimagex;
    COMPLEX *iy=(COMPLEX *)a->eimagey;
    COMPLEX *iz=(COMPLEX *)a->eimagez;
    COMPLEX *iix=(COMPLEX *)a->eimageincx;
    COMPLEX *iiy=(COMPLEX *)a->eimageincy;
    COMPLEX *iiz=(COMPLEX *)a->eimageincz;
    const REAL gross=(REAL)*a->gross;
    const int nd=(int)*a->nbsphere;
    int imax=(int)nearbyint(k0/dkx)+1,nstop=0;
    int64_t plan2f=r->plan2f,plan2b=r->plan2b;
    COMPLEX e0=*a->e0;
    REAL ss=*a->ss,pp=*a->pp,theta=*a->theta,phi=*a->phi;
    REAL xdip=0.0,ydip=0.0,zdip=0.0;
    while(imax<=20) {
        dkx*=0.5;
        imax=(int)nearbyint(k0/dkx)+1;
    }
    dky=dkx;
    deltax=2.0*pi/((REAL)nfft*dkx);
    cdmlib_zero_n(fx,plane); cdmlib_zero_n(fy,plane);
    cdmlib_zero_n(fz,plane); cdmlib_zero_n(fix,plane);
    cdmlib_zero_n(fiy,plane); cdmlib_zero_n(fiz,plane);
    cdmlib_zero_n(pos,3u*plane); cdmlib_zero_n(neg,3u*plane);
    cdmlib_zero_n(ix,plane); cdmlib_zero_n(iy,plane);
    cdmlib_zero_n(iz,plane); cdmlib_zero_n(iix,plane);
    cdmlib_zero_n(iiy,plane); cdmlib_zero_n(iiz,plane);
    /* Both Fortran incident-field variants populate Ediffkzneg here. */
    COMPLEX *incident=neg;
    if(side>0) {
        diffractefft2dinc_((const int *)a->speckseed,&nfft,&k0,
            &e0,&ss,&pp,&theta,&phi,(const REAL *)a->thetam,
            (const REAL *)a->phim,(const REAL *)a->ppm,
            (const REAL *)a->ssm,(const COMPLEX *)a->e0m,
            (const int *)a->nbinc,&xdip,&ydip,&zdip,
            (const REAL *)a->xgaus,(const REAL *)a->ygaus,
            (const REAL *)a->zgaus,(const REAL *)a->numaperil,
            (const REAL *)a->w0,&arete,(const REAL *)a->tolinit,
            fx,fy,fz,&imax,&dkx,&dky,incident,&na,c->beam,&plan2f,&plan2b,
            &nstop,a->infostr,a->infostr_len);
    } else {
        diffractefft2dinckzneg_((const int *)a->speckseed,&nfft,&k0,
            &e0,&ss,&pp,&theta,&phi,(const REAL *)a->thetam,
            (const REAL *)a->phim,(const REAL *)a->ppm,
            (const REAL *)a->ssm,(const COMPLEX *)a->e0m,
            (const int *)a->nbinc,&xdip,&ydip,&zdip,
            (const REAL *)a->xgaus,(const REAL *)a->ygaus,
            (const REAL *)a->zgaus,(const REAL *)a->numaperil,
            (const REAL *)a->w0,&arete,(const REAL *)a->tolinit,
            fx,fy,fz,&imax,&dkx,&dky,incident,&na,c->beam,&plan2f,&plan2b,
            &nstop,a->infostr,a->infostr_len);
    }
    if(nstop) { *a->nstop=(cdmlib_ip)nstop; return 1; }
    /* The incident-field routine uses fx/fy/fz as scratch/output buffers.
       The Fortran slow-lens path clears these arrays after saving the incident
       spectrum in Ediffkzneg, then fills only the scattered spectrum inside
       the numerical aperture. */
    cdmlib_zero_n(fx,plane);
    cdmlib_zero_n(fy,plane);
    cdmlib_zero_n(fz,plane);
    for(int i=-half;i<half;++i)
        ((REAL *)a->xy)[i+half]=deltax*(REAL)i*gross;
    for(int i=-imax;i<=imax;++i)
        ((REAL *)a->kxy)[i+imax]=dkx*(REAL)i/k0;
    const COMPLEX *ff=(const COMPLEX *)a->ff;
    const REAL *xs=(const REAL *)a->xs,*ys=(const REAL *)a->ys,
                 *zs=(const REAL *)a->zs;
    const bool direct_sum_gpu = ifdda_quicklens_direct_cuda_(
        ff,xs,ys,zs,incident,nd,nfft,half,imax,dkx,k0,na,side,
        (REAL)*a->zlens,fx,fy,fz,fix,fiy,fiz)==0;
    if(direct_sum_gpu) puts("CDMLIB_QUICKLENS_DIRECT_SUM_GPU=1");
    else puts("CDMLIB_QUICKLENS_DIRECT_SUM_GPU=0 (CPU fallback)");
    for(int j=-imax;j<=imax;++j) for(int i=-imax;i<=imax;++i) {
        const REAL kx=dkx*(REAL)i,ky=dky*(REAL)j;
        if(kx*kx+ky*ky>na*na) continue;
        const REAL kz=sqrt(k0*k0-kx*kx-ky*ky);
        const REAL ux=kx/k0,uy=ky/k0,uz=side*sqrt(1.0-ux*ux-uy*uy);
        const COMPLEX phase=cexp(I*kz*1.0e-9*(REAL)*a->zlens*(REAL)side);
        const size_t at=(size_t)(i+half)+(size_t)nfft*(size_t)(j+half);
        const size_t incat=(size_t)(imax+i)+(size_t)nfft*(size_t)(imax+j);
        COMPLEX ex,ey,ez,exi,eyi,ezi;
        if(direct_sum_gpu) {
            ex=fx[at]; ey=fy[at]; ez=fz[at];
            exi=fix[at]; eyi=fiy[at]; ezi=fiz[at];
        } else {
            COMPLEX emx=0.0,emy=0.0,emz=0.0;
            for(int d=0;d<nd;++d) {
                const size_t q=3u*(size_t)d;
                const COMPLEX ph=cexp(-I*k0*(ux*xs[d]+uy*ys[d]+uz*zs[d]));
                const COMPLEX dot=ux*ff[q]+uy*ff[q+1]+uz*ff[q+2];
                emx+=ph*(ff[q]-dot*ux);
                emy+=ph*(ff[q+1]-dot*uy);
                emz+=ph*(ff[q+2]-dot*uz);
            }
            const COMPLEX scale=k0*k0/(-2.0*pi*I*kz)*phase;
            ex=emx*scale; ey=emy*scale; ez=emz*scale;
            exi=ex+incident[incat]*phase;
            eyi=ey+incident[incat+plane]*phase;
            ezi=ez+incident[incat+2u*plane]*phase;
            fx[at]=ex; fy[at]=ey; fz[at]=ez;
            fix[at]=exi; fiy[at]=eyi; fiz[at]=ezi;
        }
        const REAL vx=-ux/gross,vy=-uy/gross;
        const REAL vz=side*sqrt(1.0-vx*vx-vy*vy);
        const REAL ax=uy*vz-uz*vy,ay=-ux*vz+uz*vx;
        const REAL cost=ux*vx+uy*vy+uz*vz,sint=sqrt(ax*ax+ay*ay);
        const size_t img=(size_t)(i<0?nfft+i:i)+(size_t)nfft*(size_t)(j<0?nfft+j:j);
        if(sint==0.0) {
            ix[img]=ex; iy[img]=ey; iz[img]=ez;
            iix[img]=exi; iiy[img]=eyi; iiz[img]=ezi;
        } else {
            const REAL u1=ax/sint,u2=ay/sint,tmp=sqrt(uz/vz),one=1.0-cost;
            ix[img]=((u1*u1+(1.0-u1*u1)*cost)*ex+u1*u2*one*ey+u2*sint*ez)*tmp;
            iy[img]=(u1*u2*one*ex+(u2*u2+(1.0-u2*u2)*cost)*ey-u1*sint*ez)*tmp;
            iz[img]=(-u2*sint*ex+u1*sint*ey+cost*ez)*tmp;
            iix[img]=((u1*u1+(1.0-u1*u1)*cost)*exi+u1*u2*one*eyi+u2*sint*ezi)*tmp;
            iiy[img]=(u1*u2*one*exi+(u2*u2+(1.0-u2*u2)*cost)*eyi-u1*sint*ezi)*tmp;
            iiz[img]=(-u2*sint*exi+u1*sint*eyi+cost*ezi)*tmp;
        }
    }
    fouriertoimage_(&dkx,&dky,(const REAL *)a->gross,ix,iy,iz,iix,iiy,iiz,
        &nfft,&half,&r->plan2b,&r->plan2f);
    if(cdmlib_write_holographic_quicklens_files_c99(a,r,nfft,half,imax)) return 1;
    *a->nstop=(cdmlib_ip)0;
    puts("C99 direct holographic Quick Lens finished");
    (void)s; (void)nx; (void)ny; (void)nz; (void)nxm; (void)nym; (void)nzm;
    return 0;
}
//-------------------------------------------

static int cdmlib_compute_holographic_quicklens_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,const cdmlib_entry_runtime_c99 *r)
{
    const int nfft=(int)*a->nfft2d,half=nfft/2;
    const int nx=(int)*a->nnnr,ny=nx,nz=nx;
    const int nxm=s->nxm,nym=s->nym,nzm=s->nzm;
    const int side=c->nside<0?-1:1;
    const REAL k0=(REAL)*a->k0;
    const REAL na=(REAL)*a->numaper*k0;
    REAL dkx=0.0,dky=0.0,deltax;
    COMPLEX e0=(COMPLEX)*a->e0;
    REAL ss=(REAL)*a->ss,pp=(REAL)*a->pp;
    REAL theta=(REAL)*a->theta,phi=(REAL)*a->phi;
    int imax=0,nstop=0;
    REAL xdip=0.0,ydip=0.0,zdip=0.0;
    const size_t plane=(size_t)nfft*(size_t)nfft;
    COMPLEX *pos=(COMPLEX *)a->ediffkzpos;
    COMPLEX *neg=(COMPLEX *)a->ediffkzneg;
    COMPLEX *fx=(COMPLEX *)a->efourierx;
    COMPLEX *fy=(COMPLEX *)a->efouriery;
    COMPLEX *fz=(COMPLEX *)a->efourierz;
    COMPLEX *fix=(COMPLEX *)a->efourierincx;
    COMPLEX *fiy=(COMPLEX *)a->efourierincy;
    COMPLEX *fiz=(COMPLEX *)a->efourierincz;
    COMPLEX *ix=(COMPLEX *)a->eimagex;
    COMPLEX *iy=(COMPLEX *)a->eimagey;
    COMPLEX *iz=(COMPLEX *)a->eimagez;
    COMPLEX *iix=(COMPLEX *)a->eimageincx;
    COMPLEX *iiy=(COMPLEX *)a->eimageincy;
    COMPLEX *iiz=(COMPLEX *)a->eimageincz;
    if(c->nlentille!=1) return 0;
    if(c->ntypemic!=0 || (c->nquicklens!=0&&c->nquicklens!=1) || nfft<=0 ||
       !a->tabfft2 || !a->xy || !a->kxy ||
       !a->efourierx || !a->efouriery || !a->efourierz ||
       !a->efourierincx || !a->efourierincy || !a->efourierincz ||
       !a->eimagex || !a->eimagey || !a->eimagez || !a->eimageincx ||
       !a->eimageincy || !a->eimageincz || !a->ediffkzpos || !a->ediffkzneg) {
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,"Unsupported C99 holographic lens configuration",
               46<a->infostr_len?46:a->infostr_len);
        *a->nstop=(cdmlib_ip)1;
        return 1;
    }
    if(*a->nmat<0 || *a->nmat>2) {
        *a->nstop=(cdmlib_ip)1;
        return 1;
    }
    if(c->nquicklens==0)
        return cdmlib_compute_holographic_quicklens_slow_c99(a,s,c,r,
            nfft,half,nx,ny,nz,nxm,nym,nzm,side,k0,na);
    diffractefft2dlens_(&nx,&ny,&nz,&nxm,&nym,&nzm,&nfft,(int *)a->tabfft2,
        &k0,(const REAL *)a->xs,(const REAL *)a->ys,(const REAL *)a->zs,
        (const REAL *)a->aretecube,(COMPLEX *)a->efourierx,
        (COMPLEX *)a->efouriery,(COMPLEX *)a->efourierz,
        (const COMPLEX *)a->ff,&imax,&dkx,&dky,pos,&na,&side,
        &r->plan2f,&r->plan2b,&nstop,a->infostr,a->infostr_len);
    if(nstop) { *a->nstop=(cdmlib_ip)nstop; return 1; }
    if(side<0)
        memcpy(neg,pos,3u*plane*sizeof *pos);
    if(side>0) {
        diffractefft2dinc_((const int *)a->speckseed,&nfft,&k0,&e0,&ss,&pp,
            &theta,&phi,(const REAL *)a->thetam,(const REAL *)a->phim,
            (const REAL *)a->ppm,(const REAL *)a->ssm,
            (const COMPLEX *)a->e0m,(const int *)a->nbinc,
            &xdip,&ydip,&zdip,
            (const REAL *)a->xgaus,(const REAL *)a->ygaus,(const REAL *)a->zgaus,
            (const REAL *)a->numaperil,(const REAL *)a->w0,
            (const REAL *)a->aretecube,(const REAL *)a->tolinit,
            fx,fy,fz,&imax,&dkx,&dky,neg,&na,c->beam,&r->plan2f,&r->plan2b,
            &nstop,a->infostr,a->infostr_len);
    } else {
        diffractefft2dinckzneg_((const int *)a->speckseed,&nfft,&k0,&e0,&ss,&pp,
            &theta,&phi,(const REAL *)a->thetam,(const REAL *)a->phim,
            (const REAL *)a->ppm,(const REAL *)a->ssm,
            (const COMPLEX *)a->e0m,(const int *)a->nbinc,
            &xdip,&ydip,&zdip,
            (const REAL *)a->xgaus,(const REAL *)a->ygaus,(const REAL *)a->zgaus,
            (const REAL *)a->numaperil,(const REAL *)a->w0,
            (const REAL *)a->aretecube,(const REAL *)a->tolinit,
            fx,fy,fz,&imax,&dkx,&dky,pos,&na,c->beam,&r->plan2f,&r->plan2b,
            &nstop,a->infostr,a->infostr_len);
    }
    if(nstop) { *a->nstop=(cdmlib_ip)nstop; return 1; }
    deltax=2.0*3.14159265358979323846/((REAL)nfft*dkx);
    for(int i=-half;i<half;++i) ((REAL *)a->xy)[i+half]=deltax*(REAL)i*(REAL)*a->gross;
    for(int i=-imax;i<=imax;++i) ((REAL *)a->kxy)[i+imax]=dkx*(REAL)i/k0;
    cdmlib_zero_n(fx,plane); cdmlib_zero_n(fy,plane); cdmlib_zero_n(fz,plane);
    cdmlib_zero_n(fix,plane); cdmlib_zero_n(fiy,plane); cdmlib_zero_n(fiz,plane);
    cdmlib_zero_n(ix,plane); cdmlib_zero_n(iy,plane); cdmlib_zero_n(iz,plane);
    cdmlib_zero_n(iix,plane); cdmlib_zero_n(iiy,plane); cdmlib_zero_n(iiz,plane);
    for(int i=-imax;i<=imax;++i) for(int j=-imax;j<=imax;++j) {
        const REAL kx=dkx*(REAL)i,ky=dky*(REAL)j,kp2=kx*kx+ky*ky;
        const int ii=imax+i,jj=imax+j;
        if(kp2>=na*na) continue;
        const REAL kz=sqrt(k0*k0-kp2),u1x=kx/k0,u1y=ky/k0;
        const REAL u1z=(REAL)side*sqrt(1.0-u1x*u1x-u1y*u1y);
        const REAL v1x=-u1x/(REAL)*a->gross,v1y=-u1y/(REAL)*a->gross;
        const REAL v1z=(REAL)side*sqrt(1.0-v1x*v1x-v1y*v1y);
        const REAL cx=u1y*v1z-u1z*v1y,cy=-u1x*v1z+u1z*v1x;
        const REAL cost=u1x*v1x+u1y*v1y+u1z*v1z,sint=sqrt(cx*cx+cy*cy);
        const COMPLEX phase=cexp(I*kz*1.0e-9*(REAL)*a->zlens*(REAL)side);
        const size_t src=(size_t)ii+(size_t)nfft*(size_t)jj;
        const size_t dst=(size_t)(i+half)+(size_t)nfft*(size_t)(j+half);
        const size_t grid=(size_t)(i<0?nfft+i:i)+(size_t)nfft*(size_t)(j<0?nfft+j:j);
        COMPLEX sx=pos[src],sy=pos[src+plane],sz=pos[src+2*plane];
        COMPLEX tx=neg[src],ty=neg[src+plane],tz=neg[src+2*plane];
        if(side<0) { sx=neg[src];sy=neg[src+plane];sz=neg[src+2*plane];tx=pos[src];ty=pos[src+plane];tz=pos[src+2*plane]; }
        fx[dst]=sx*phase; fy[dst]=sy*phase; fz[dst]=sz*phase;
        fix[dst]=(sx+tx)*phase; fiy[dst]=(sy+ty)*phase; fiz[dst]=(sz+tz)*phase;
        if(sint==0.0) {
            ix[grid]=fx[dst]; iy[grid]=fy[dst]; iz[grid]=fz[dst];
            iix[grid]=fix[dst]; iiy[grid]=fiy[dst]; iiz[grid]=fiz[dst];
        } else {
            const REAL ax=cx/sint,ay=cy/sint,scale=sqrt(u1z/v1z),one=1.0-cost;
            ix[grid]=((ax*ax+(1.0-ax*ax)*cost)*fx[dst]+ax*ay*one*fy[dst]+ay*sint*fz[dst])*scale;
            iy[grid]=(ax*ay*one*fx[dst]+(ay*ay+(1.0-ay*ay)*cost)*fy[dst]-ax*sint*fz[dst])*scale;
            iz[grid]=(-ay*sint*fx[dst]+ax*sint*fy[dst]+cost*fz[dst])*scale;
            iix[grid]=((ax*ax+(1.0-ax*ax)*cost)*fix[dst]+ax*ay*one*fiy[dst]+ay*sint*fiz[dst])*scale;
            iiy[grid]=(ax*ay*one*fix[dst]+(ay*ay+(1.0-ay*ay)*cost)*fiy[dst]-ax*sint*fiz[dst])*scale;
            iiz[grid]=(-ay*sint*fix[dst]+ax*sint*fiy[dst]+cost*fiz[dst])*scale;
        }
    }
    fouriertoimage_(&dkx,&dky,(const REAL *)a->gross,ix,iy,iz,iix,iiy,iiz,
        &nfft,&half,&r->plan2b,&r->plan2f);
    if(cdmlib_write_holographic_quicklens_files_c99(a,r,nfft,half,imax)) return 1;
    *a->nstop=(cdmlib_ip)0;
    puts("C99 holographic Quick Lens finished");
    return 0;
}
//-------------------------------------------

static int cdmlib_compute_darkfield_microscopy_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,cdmlib_entry_runtime_c99 *r)
{
    int ntotalm=s->ntotalm,ldabi=s->ldabi,nmax=s->nmax;
    int nxm=s->nxm,nym=s->nym,nzm=s->nzm;
    int nx=r->nx,ny=r->ny,nz=r->nz,nx2=r->nx2,ny2=r->ny2;
    int nxy2=r->nxy2,nz2=r->nz2,ntotal=r->ntotal;
    int nbsphere3=r->nbsphere3,npolainc=0;
    int64_t file_id=0,group_idmic=0;
    int nside=c->nside;
    REAL numaper_k=c->numaper*(REAL)*a->k0;
    REAL zlens=c->zlens;
    if(c->ntypemic!=4) return 0;
    if(!a->nstop||!a->infostr||!a->nfft2d||!a->ndipole||!a->nbsphere||
       !a->nrig||!a->nproche||!a->tabfft2||!a->tabdip||!a->xi||!a->xr||
       !a->wrk||!a->ff||!a->ff0||!a->ffloc||!a->polarisa||!a->niso||
       !a->epsilon||!a->methodeit||!a->tol1||!a->nloop||!a->ncompte||
       !a->nlim||!a->xs||!a->ys||!a->zs||!a->aretecube||!a->numaperinc||
       !a->nquicklens||!a->k0||!a->p0||!a->irra||!a->w0||!a->gross||
       !a->eimagex||!a->eimagey||!a->eimagez||!a->efourierx||!a->efouriery||
       !a->efourierz||!a->eimageincx||!a->eimageincy||!a->eimageincz||
       !a->efourierincx||!a->efourierincy||!a->efourierincz||!a->ediffkzpos||
       !a->ediffkzneg||!a->kxy||!a->xy||!a->nside||!a->nmat||
       !a->ffttensorxx||!a->ffttensorxy||
       !a->ffttensorxz||!a->ffttensoryy||!a->ffttensoryz||!a->ffttensorzz) {
        static const char msg[]="Invalid C99 dark-field microscopy arguments";
        *a->nstop=(cdmlib_ip)1;
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?sizeof msg-1:a->infostr_len);
        return 1;
    }
    if(cdmlib_compute_green_from_abi_c99(a,s,c,r)!=0) return 1;
    if(c->nside<0) {
        microsdfneg_(a->ffttensorxx,a->ffttensorxy,a->ffttensorxz,
            a->ffttensoryy,a->ffttensoryz,a->ffttensorzz,
            r->vectx,r->vecty,r->vectz,&ntotalm,&ntotal,&ldabi,a->nlar,&nmax,
            &nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,a->ndipole,
            a->nbsphere,&nbsphere3,a->nproche,a->nrig,a->nfft2d,a->tabfft2,
            a->tabdip,a->xi,a->xr,a->wrk,a->ff,a->ff0,a->ffloc,a->polarisa,
            a->niso,a->epsilon,a->methodeit,a->trope,a->tolinit,a->tol1,
            a->nloop,a->ncompte,a->nlim,a->xs,a->ys,a->zs,a->aretecube,
            &numaper_k,a->numaperinc,&npolainc,a->nquicklens,const_cast<COMPLEX*>(&s->eps0),
            a->k0,a->p0,a->irra,a->w0,a->gross,&zlens,a->eimagex,a->eimagey,
            a->eimagez,a->efourierx,a->efouriery,a->efourierz,a->ediffkzpos,
            a->ediffkzneg,a->kxy,a->xy,&nside,&r->planXf,&r->planXb,
            &r->planYf,&r->planYb,&r->planZf,&r->planZb,&r->planf,&r->planb,
            &r->plan2f,&r->plan2b,a->nmat,&file_id,&group_idmic,a->nstop,
            a->infostr,a->methodeit_len,a->trope_len,a->infostr_len);
    } else {
        microsdf_(a->ffttensorxx,a->ffttensorxy,a->ffttensorxz,
            a->ffttensoryy,a->ffttensoryz,a->ffttensorzz,
            r->vectx,r->vecty,r->vectz,&ntotalm,&ntotal,&ldabi,a->nlar,&nmax,
            &nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,a->ndipole,
            a->nbsphere,&nbsphere3,a->nproche,a->nrig,a->nfft2d,a->tabfft2,
            a->tabdip,a->xi,a->xr,a->wrk,a->ff,a->ff0,a->ffloc,a->polarisa,
            a->niso,a->epsilon,a->methodeit,a->trope,a->tolinit,a->tol1,
            a->nloop,a->ncompte,a->nlim,a->xs,a->ys,a->zs,a->aretecube,
            &numaper_k,a->numaperinc,&npolainc,a->nquicklens,const_cast<COMPLEX*>(&s->eps0),
            a->k0,a->p0,a->irra,a->w0,a->gross,&zlens,a->eimagex,a->eimagey,
            a->eimagez,a->efourierx,a->efouriery,a->efourierz,a->eimageincx,
            a->eimageincy,a->eimageincz,a->efourierincx,a->efourierincy,
            a->efourierincz,a->ediffkzpos,a->ediffkzneg,a->kxy,a->xy,&nside,
            &r->planXf,&r->planXb,&r->planYf,&r->planYb,&r->planZf,&r->planZb,
            &r->planf,&r->planb,&r->plan2f,&r->plan2b,a->nmat,&file_id,
            &group_idmic,a->nstop,a->infostr,a->methodeit_len,a->trope_len,
            a->infostr_len);
    }
    return *a->nstop!=0;
}
//-------------------------------------------

static int cdmlib_compute_brightfield_microscopy_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,cdmlib_entry_runtime_c99 *r)
{
    enum { nmasque=100 };
    int ntotalm=s->ntotalm,ldabi=s->ldabi,nmax=s->nmax;
    int nxm=s->nxm,nym=s->nym,nzm=s->nzm;
    int nx=r->nx,ny=r->ny,nz=r->nz,nx2=r->nx2,ny2=r->ny2;
    int nxy2=r->nxy2,nz2=r->nz2,ntotal=r->ntotal,nbsphere3=r->nbsphere3;
    int npolainc=0,imaxinc=0,imaxk0=0,nbitermic=0;
    // microsbf_/microsbfneg_ expect Fortran integer(ip) IDs (32-bit).
    int file_id=0,group_idmic=0;
    int masklen=nmasque;
    REAL deltakinc=0.0,deltakx=0.0;
    REAL numaper=c->numaper*(REAL)*a->k0;
    REAL zlens=c->zlens;
    COMPLEX *masque;

    if(c->ntypemic!=1 && c->ntypemic!=2 && c->ntypemic!=3 &&
       c->ntypemic!=5) return 0;
    if(!a->nstop||!a->infostr||!a->nfft2d||!a->ndipole||!a->nbsphere||
       !a->nrig||!a->nproche||!a->tabfft2||!a->tabdip||!a->xi||!a->xr||
       !a->wrk||!a->ff||!a->ff0||!a->ffloc||!a->polarisa||!a->niso||
       !a->epsilon||!a->methodeit||!a->tolinit||!a->tol1||!a->nloop||
       !a->ncompte||!a->nlim||!a->xs||!a->ys||!a->zs||!a->aretecube||
       !a->numaperinc||!a->numaperinc2||!a->nquicklens||!a->k0||!a->p0||
       !a->irra||!a->w0||!a->gross||!a->eimagex||!a->eimagey||!a->eimagez||
       !a->efourierx||!a->efouriery||!a->efourierz||!a->eimageincx||
       !a->eimageincy||!a->eimageincz||!a->efourierincx||!a->efourierincy||
       !a->efourierincz||!a->ediffkzpos||!a->ediffkzneg||!a->kxy||!a->xy||
       !a->nside||!a->nmat||!a->nfft2d||!a->numaper||!a->kcnax||!a->kcnay||
       !a->ffttensorxx||!a->ffttensorxy||
       !a->ffttensorxz||!a->ffttensoryy||!a->ffttensoryz||!a->ffttensorzz) {
        static const char msg[]="Invalid C99 bright-field microscopy arguments";
        *a->nstop=(cdmlib_ip)1;
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?sizeof msg-1:a->infostr_len);
        return 1;
    }
    masque=(COMPLEX *)calloc((size_t)nmasque*(size_t)nmasque,sizeof *masque);
    if(!masque) {
        static const char msg[]="Could not allocate C99 bright-field mask";
        *a->nstop=(cdmlib_ip)1;
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?sizeof msg-1:a->infostr_len);
        return 1;
    }
    *a->nstop=(cdmlib_ip)0;
    micromasque_(masque,&masklen,&imaxinc,&imaxk0,&nbitermic,&c->ntypemic,
        a->nfft2d,&deltakinc,&deltakx,&numaper,
        (REAL *)a->numaperinc,(REAL *)a->numaperinc2,(REAL *)a->kcnax,
        (REAL *)a->kcnay,(REAL *)a->aretecube,a->k0,a->xs,a->ys,
        &nxm,&nym,&nzm,a->nbsphere,a->nquicklens,a->nstop,a->infostr,
        a->infostr_len);
    if(*a->nstop!=0) { free(masque); return 1; }
    if(cdmlib_compute_green_from_abi_c99(a,s,c,r)!=0) {
        free(masque);
        return 1;
    }
    if(c->nside<0) {
        microsbfneg_(a->ffttensorxx,a->ffttensorxy,a->ffttensorxz,
            a->ffttensoryy,a->ffttensoryz,a->ffttensorzz,r->vectx,r->vecty,
            r->vectz,&ntotalm,&ntotal,&ldabi,a->nlar,&nmax,&nxm,&nym,&nzm,
            &nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,a->ndipole,a->nbsphere,
            &nbsphere3,a->nproche,a->nrig,a->nfft2d,a->tabfft2,a->tabdip,
            a->xi,a->xr,a->wrk,a->ff,a->ff0,a->ffloc,a->polarisa,a->niso,
            a->epsilon,a->methodeit,a->trope,a->tolinit,a->tol1,a->nloop,
            a->ncompte,a->nlim,a->xs,a->ys,a->zs,a->aretecube,&numaper,
            a->numaperinc,a->numaperinc2,&npolainc,a->nquicklens,
            const_cast<COMPLEX*>(&s->eps0),a->k0,a->p0,a->irra,a->w0,a->gross,&zlens,
            a->eimagex,a->eimagey,a->eimagez,a->efourierx,a->efouriery,
            a->efourierz,a->ediffkzpos,a->ediffkzneg,a->kxy,a->xy,a->nside,
            masque,&masklen,&imaxinc,&imaxk0,&nbitermic,&deltakinc,&deltakx,
            a->ntypemic,&r->planXf,&r->planXb,&r->planYf,&r->planYb,
            &r->planZf,&r->planZb,&r->planf,&r->planb,&r->plan2f,&r->plan2b,
            a->nmat,&file_id,&group_idmic,a->nstop,a->infostr,
            a->methodeit_len,a->trope_len,a->infostr_len);
    } else {
        microsbf_(a->ffttensorxx,a->ffttensorxy,a->ffttensorxz,
            a->ffttensoryy,a->ffttensoryz,a->ffttensorzz,r->vectx,r->vecty,
            r->vectz,&ntotalm,&ntotal,&ldabi,a->nlar,&nmax,&nxm,&nym,&nzm,
            &nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,a->ndipole,a->nbsphere,
            &nbsphere3,a->nproche,a->nrig,a->nfft2d,a->tabfft2,a->tabdip,
            a->xi,a->xr,a->wrk,a->ff,a->ff0,a->ffloc,a->polarisa,a->niso,
            a->epsilon,a->methodeit,a->trope,a->tolinit,a->tol1,a->nloop,
            a->ncompte,a->nlim,a->xs,a->ys,a->zs,a->aretecube,&numaper,
            a->numaperinc,a->numaperinc2,&npolainc,a->nquicklens,
            const_cast<COMPLEX*>(&s->eps0),a->k0,a->p0,a->irra,a->w0,a->gross,&zlens,
            a->eimagex,a->eimagey,a->eimagez,a->efourierx,a->efouriery,
            a->efourierz,a->eimageincx,a->eimageincy,a->eimageincz,
            a->efourierincx,a->efourierincy,a->efourierincz,a->ediffkzpos,
            a->ediffkzneg,a->kxy,a->xy,a->nside,masque,&masklen,&imaxinc,
            &imaxk0,&nbitermic,&deltakinc,&deltakx,a->ntypemic,
            &r->planXf,&r->planXb,&r->planYf,&r->planYb,&r->planZf,&r->planZb,
            &r->planf,&r->planb,&r->plan2f,&r->plan2b,a->nmat,&file_id,
            &group_idmic,a->nstop,a->infostr,a->methodeit_len,a->trope_len,
            a->infostr_len);
    }
    free(masque);
    return *a->nstop!=0;
}
//-------------------------------------------

static int cdmlib_compute_confocal_microscopy_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,cdmlib_entry_runtime_c99 *r)
{
    enum { nmasque=100 };
    int ntotalm=s->ntotalm,ldabi=s->ldabi,nmax=s->nmax;
    int nxm=s->nxm,nym=s->nym,nzm=s->nzm;
    int nx=r->nx,ny=r->ny,nz=r->nz,nx2=r->nx2,ny2=r->ny2;
    int nxy2=r->nxy2,nz2=r->nz2,ntotal=r->ntotal,nbsphere3=r->nbsphere3;
    int imaxinc=0,imaxk0=0,nbitermic=0,masklen=nmasque;
    int64_t file_id=0,group_idmic=0;
    int nside=c->nside;
    REAL deltakinc=0.0,deltakx=0.0;
    REAL numaper=c->numaper*(REAL)*a->k0;
    REAL numaperinc=(c->ntypemic==6||c->ntypemic==7)?
        (REAL)*a->numaper:(REAL)*a->numaperinc;
    REAL zlens=c->zlens;
    COMPLEX *masque;

    if(c->ntypemic!=6 && c->ntypemic!=7) return 0;
    if(!a->nstop||!a->infostr||!a->nfft2d||!a->ndipole||!a->nbsphere||
       !a->nrig||!a->nproche||!a->tabfft2||!a->tabdip||!a->xi||!a->xr||
       !a->wrk||!a->ff||!a->ff0||!a->ffloc||!a->polarisa||!a->niso||
       !a->epsilon||!a->methodeit||!a->tolinit||!a->tol1||!a->nloop||
       !a->ncompte||!a->nlim||!a->xs||!a->ys||!a->zs||!a->aretecube||
       !a->numaper||!a->numaperinc||!a->numaperinc2||!a->psiinc||
       !a->nquicklens||!a->k0||!a->p0||!a->irra||!a->w0||!a->gross||
       !a->zlens||!a->kcnax||!a->kcnay||!a->eimagex||!a->eimagey||
       !a->eimagez||!a->efourierx||!a->efouriery||!a->efourierz||
       !a->eimageincx||!a->eimageincy||!a->eimageincz||!a->efourierincx||
       !a->efourierincy||!a->efourierincz||!a->ediffkzpos||!a->ediffkzneg||
       !a->kxy||!a->xy||!a->nside||!a->nmat||
       !a->ffttensorxx||!a->ffttensorxy||!a->ffttensorxz||
       !a->ffttensoryy||!a->ffttensoryz||!a->ffttensorzz) {
        static const char msg[]="Invalid C99 confocal microscopy arguments";
        *a->nstop=(cdmlib_ip)1;
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?sizeof msg-1:a->infostr_len);
        return 1;
    }
    masque=(COMPLEX *)calloc((size_t)nmasque*(size_t)nmasque,sizeof *masque);
    if(!masque) {
        static const char msg[]="Could not allocate C99 confocal mask";
        *a->nstop=(cdmlib_ip)1;
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?sizeof msg-1:a->infostr_len);
        return 1;
    }
    *a->nstop=(cdmlib_ip)0;
    micromasque_(masque,&masklen,&imaxinc,&imaxk0,&nbitermic,&c->ntypemic,
        a->nfft2d,&deltakinc,&deltakx,&numaper,
        &numaperinc,(REAL *)a->numaperinc2,(REAL *)a->kcnax,
        (REAL *)a->kcnay,(REAL *)a->aretecube,a->k0,a->xs,a->ys,
        &nxm,&nym,&nzm,a->nbsphere,a->nquicklens,a->nstop,a->infostr,
        a->infostr_len);
    if(*a->nstop!=0) { free(masque); return 1; }
    if(cdmlib_compute_green_from_abi_c99(a,s,c,r)!=0) {
        free(masque);
        return 1;
    }
#define CDMLIB_MICROSCONFOCAL_ARGS \
    a->ffttensorxx,a->ffttensorxy,a->ffttensorxz,a->ffttensoryy, \
    a->ffttensoryz,a->ffttensorzz,r->vectx,r->vecty,r->vectz, \
    &ntotalm,&ntotal,&ldabi,a->nlar,&nmax,&nxm,&nym,&nzm,&nx,&ny,&nz, \
    &nx2,&ny2,&nxy2,&nz2,a->ndipole,a->nbsphere,&nbsphere3,a->nproche, \
    a->nrig,a->nfft2d,a->tabfft2,a->tabdip,a->xi,a->xr,a->wrk,a->ff, \
    a->ff0,a->ffloc,a->polarisa,a->niso,a->epsilon,a->methodeit,a->trope, \
    a->tolinit,a->tol1,a->nloop,a->ncompte,a->nlim,a->xs,a->ys,a->zs, \
    a->aretecube,&numaper,&numaperinc,a->numaperinc2,a->psiinc, \
    a->nquicklens,(const COMPLEX *)&s->eps0,a->k0,a->p0,a->irra,a->w0,a->gross, \
    &zlens,a->kcnax,a->kcnay,a->eimagex,a->eimagey,a->eimagez, \
    a->efourierx,a->efouriery,a->efourierz,a->eimageincx,a->eimageincy, \
    a->eimageincz,a->efourierincx,a->efourierincy,a->efourierincz, \
    a->ediffkzpos,a->ediffkzneg,a->kxy,a->xy,&nside,masque,&masklen, \
    &imaxinc,&imaxk0,&nbitermic,&deltakinc,&deltakx,a->ntypemic, \
    &r->planXf,&r->planXb,&r->planYf,&r->planYb,&r->planZf,&r->planZb, \
    &r->planf,&r->planb,&r->plan2f,&r->plan2b,a->nmat,&file_id, \
    &group_idmic,a->nstop,a->infostr,a->methodeit_len,a->trope_len, \
    a->infostr_len
    if(c->ntypemic==6)
        microsconfocal_(CDMLIB_MICROSCONFOCAL_ARGS);
    else
        microsconfocalfluo_(CDMLIB_MICROSCONFOCAL_ARGS);
#undef CDMLIB_MICROSCONFOCAL_ARGS
    free(masque);
    return *a->nstop!=0;
}
//-------------------------------------------

static int cdmlib_compute_energy_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,const cdmlib_entry_runtime_c99 *r)
{
    if(!a||!s||!c||!r||!a->nenergie||*a->nenergie!=1) return 0;
    if(!a->nfft2d||!a->tabfft2||!a->k0||!a->xs||!a->ys||!a->zs||!a->e0||
       !a->ss||!a->pp||!a->theta||!a->phi||!a->thetam||!a->phim||!a->ppm||
       !a->ssm||!a->e0m||!a->nbinc||!a->xgaus||!a->ygaus||!a->zgaus||
       !a->w0||!a->aretecube||!a->tolinit||!a->efourierx||!a->efouriery||
       !a->efourierz||!a->ff||!a->ediffkzpos||!a->ediffkzneg||!a->beam||
       !a->nsectionsca||!a->nquickdiffracte||!a->nstop||!a->infostr||
       !a->efficacite||!a->efficaciteref||!a->efficacitetrans) {
        static const char msg[]="Missing buffer for C99 energy calculation";
        if(a&&a->infostr) {
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
        }
        if(a&&a->nstop) *a->nstop=(cdmlib_ip)1;
        return 1;
    }
    int nx=r->nx,ny=r->ny,nz=r->nz,nxm=s->nxm,nym=s->nym,nzm=s->nzm;
    int nfft=(int)*a->nfft2d,nbinc=(int)*a->nbinc;
    int nsectionsca=(int)*a->nsectionsca,nquick=(int)*a->nquickdiffracte;
    int imax=0,nstop=0,nside=a->nside?(int)*a->nside:1;
    REAL k0=(REAL)*a->k0,arete=(REAL)*a->aretecube;
    REAL tol=(REAL)*a->tolinit,deltakx=0.0,deltaky=0.0;
    REAL xdip_dummy=r->antenna_x,ydip_dummy=r->antenna_y,zdip_dummy=r->antenna_z;
    REAL numaperil=a->numaperil?(REAL)*a->numaperil:1.0;
    int64_t plan2f=r->plan2f,plan2b=r->plan2b;
    if(memcmp(a->beam,"speckle",7)==0||
       memcmp(a->beam,"demispeckle",11)==0||
       memcmp(a->beam,"confocal",8)==0||
       memcmp(a->beam,"demiconfocal",12)==0) {
        if(!a->speckseed||!a->numaperil) {
            static const char msg[]="Missing confocal energy parameters";
            memset(a->infostr,' ',a->infostr_len);
            memcpy(a->infostr,msg,sizeof msg-1<a->infostr_len?
                   sizeof msg-1:a->infostr_len);
            *a->nstop=(cdmlib_ip)1;
            return 1;
        }
        int seed=(int)*a->speckseed;
        diffractefft2denergieconfocal_(&seed,&numaperil,&nx,&ny,&nz,&nxm,
            &nym,&nzm,&nfft,(int *)a->tabfft2,&k0,(const REAL *)a->xs,
            (const REAL *)a->ys,(const REAL *)a->zs,
            (const COMPLEX *)a->e0,(const REAL *)a->ss,
            (const REAL *)a->pp,(const REAL *)a->theta,
            (const REAL *)a->phi,(const REAL *)a->xgaus,
            (const REAL *)a->ygaus,(const REAL *)a->zgaus,
            (const REAL *)a->w0,&arete,(COMPLEX *)a->efourierx,
            (COMPLEX *)a->efouriery,(COMPLEX *)a->efourierz,
            (const COMPLEX *)a->ff,&imax,&deltakx,&deltaky,
            (COMPLEX *)a->ediffkzpos,(COMPLEX *)a->ediffkzneg,
            a->beam,(REAL *)a->efficacite,(REAL *)a->efficaciteref,
            (REAL *)a->efficacitetrans,&nsectionsca,&nquick,&plan2f,
            &plan2b,&nside,&nstop,a->infostr,a->infostr_len);
    } else {
        diffractefft2denergie_(&nx,&ny,&nz,&nxm,&nym,&nzm,&nfft,
            (int *)a->tabfft2,&k0,(const REAL *)a->xs,(const REAL *)a->ys,
            (const REAL *)a->zs,(const COMPLEX *)a->e0,
            (const REAL *)a->ss,(const REAL *)a->pp,
            (const REAL *)a->theta,(const REAL *)a->phi,
            (const REAL *)a->thetam,(const REAL *)a->phim,
            (const REAL *)a->ppm,(const REAL *)a->ssm,
            (const COMPLEX *)a->e0m,&nbinc,&xdip_dummy,&ydip_dummy,
            &zdip_dummy,
            (const REAL *)a->xgaus,(const REAL *)a->ygaus,
            (const REAL *)a->zgaus,(const REAL *)a->w0,&arete,&tol,
            (COMPLEX *)a->efourierx,(COMPLEX *)a->efouriery,
            (COMPLEX *)a->efourierz,(const COMPLEX *)a->ff,
            &imax,&deltakx,&deltaky,(COMPLEX *)a->ediffkzpos,
            (COMPLEX *)a->ediffkzneg,a->beam,
            (REAL *)a->efficacite,(REAL *)a->efficaciteref,
            (REAL *)a->efficacitetrans,&nsectionsca,&nquick,&plan2f,
            &plan2b,&nstop,a->infostr,a->infostr_len);
    }
    *a->nstop=(cdmlib_ip)nstop;
    if(nstop!=0) return 1;
    printf("Conservation of energy : %.16g\n",(REAL)*a->efficacite);
    printf("Absorptivity           : %.16g\n",1.0-(REAL)*a->efficacite);
    printf("Reflextivity           : %.16g\n",(REAL)*a->efficaciteref);
    printf("Transmittivity         : %.16g\n",(REAL)*a->efficacitetrans);
    (void)c;(void)numaperil;
    return 0;
}
//-------------------------------------------

static int cdmlib_compute_widefield_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,cdmlib_entry_runtime_c99 *r,int *subunits)
{
    if(!a||!s||!c||!r||!subunits||!a->nstop||!a->infostr||!a->nmat||
       !a->nrig||!a->nbsphere||!a->nquad||!a->nxmp||!a->nymp||!a->nzmp||
       !a->nmaxpp||!a->nfft2d||!a->speckseed||!a->irra||!a->numaperil||
       !a->aretecube||!a->k0||!a->theta||!a->phi||!a->pp||!a->ss||!a->p0||
       !a->w0||!a->xgaus||!a->ygaus||!a->zgaus||!a->e0||!a->e0m||!a->ssm||
       !a->ppm||!a->thetam||!a->phim||!a->nbinc||!a->xs||!a->ys||!a->zs||
       !a->xswf||!a->yswf||!a->zswf||!a->xi||!a->xr||!a->ffloc||!a->ff||
       !a->ff0||!a->incidentfieldx||!a->incidentfieldy||!a->incidentfieldz||
       !a->incidentfield||!a->eimagex||!a->eimagey||!a->eimagez||
       !a->polarisa||!a->niso||!a->epsilon||!a->beam||!a->namefileinc||
       !a->ffttensorxx||!a->ffttensorxy||!a->ffttensorxz||!a->ffttensoryy||
       !a->ffttensoryz||!a->ffttensorzz) return 1;

    int nmat=(int)*a->nmat,nrig=c->nrig;
    int nx=r->nx,ny=r->ny,nz=r->nz,nx2=r->nx2,ny2=r->ny2,nz2=r->nz2;
    int nb=(int)*a->nbsphere,subunit=0,nmaxpp=(int)*a->nmaxpp;
    int nxmp=(int)*a->nxmp,nymp=(int)*a->nymp,nzmp=(int)*a->nzmp;
    int nquad=(int)*a->nquad,nxm=s->nxm,nym=s->nym,nzm=s->nzm;
    int nfft2d=(int)*a->nfft2d,ntotalm=s->ntotalm,nmax=s->nmax;
    int seed=(int)*a->speckseed,nbinc=(int)*a->nbinc,niso=(int)*a->niso;
    int nstop=0;
    int64_t group_idnf=0;
    REAL irra=(REAL)*a->irra,numaperil=(REAL)*a->numaperil;
    REAL arete=(REAL)*a->aretecube,k0=(REAL)*a->k0,lambda=s->lambda;
    REAL theta=(REAL)*a->theta,phi=(REAL)*a->phi,pp=(REAL)*a->pp;
    REAL ss=(REAL)*a->ss,power=s->power,w0=s->w0;
    REAL xdip=r->antenna_x,ydip=r->antenna_y,zdip=r->antenna_z;
    REAL xgaus=(REAL)*a->xgaus,ygaus=(REAL)*a->ygaus,zgaus=(REAL)*a->zgaus;
    REAL tol=s->tolinit;
    COMPLEX e0=(COMPLEX)*a->e0;
    int64_t plan2f=r->plan2f,plan2b=r->plan2b;
    calculwidefield_(&nmat,&nrig,&nx,&ny,&nz,&nx2,&ny2,&nz2,&nb,&subunit,
        &nmaxpp,&nxmp,&nymp,&nzmp,&nquad,&nxm,&nym,&nzm,&nfft2d,&ntotalm,
        &nmax,&seed,&irra,&numaperil,&arete,&k0,&lambda,&theta,&phi,&pp,&ss,
        &power,&w0,&xdip,&ydip,&zdip,&xgaus,&ygaus,&zgaus,&tol,&e0,
        (COMPLEX *)a->e0m,(REAL *)a->ssm,(REAL *)a->ppm,
        (REAL *)a->thetam,(REAL *)a->phim,&nbinc,(REAL *)a->xs,
        (REAL *)a->ys,(REAL *)a->zs,(REAL *)a->xswf,(REAL *)a->yswf,
        (REAL *)a->zswf,(COMPLEX *)a->xi,(COMPLEX *)a->xr,
        (COMPLEX *)a->ffloc,(COMPLEX *)a->ff,
        (COMPLEX *)a->ff0,(COMPLEX *)a->incidentfieldx,
        (COMPLEX *)a->incidentfieldy,(COMPLEX *)a->incidentfieldz,
        (REAL *)a->incidentfield,(COMPLEX *)a->eimagex,
        (COMPLEX *)a->eimagey,(COMPLEX *)a->eimagez,
        (COMPLEX *)a->polarisa,&niso,(COMPLEX *)a->epsilon,
        a->beam,a->infostr,a->namefileinc,(COMPLEX *)a->ffttensorxx,
        (COMPLEX *)a->ffttensorxy,(COMPLEX *)a->ffttensorxz,
        (COMPLEX *)a->ffttensoryy,(COMPLEX *)a->ffttensoryz,
        (COMPLEX *)a->ffttensorzz,&plan2f,&plan2b,&group_idnf,&nstop,
        a->beam_len,a->infostr_len,a->namefileinc_len);
    *a->nmaxpp=(cdmlib_ip)nmaxpp;
    *a->nstop=(cdmlib_ip)nstop;
    *subunits=subunit;
    return nstop==0?0:1;
}
//-------------------------------------------

static int cdmlib_map_widefield_outputs_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,int subunits)
{
    const int nmax=s->nmax,niso=(int)*a->niso,nsens=1;
    const REAL arete=(REAL)*a->aretecube,k0=(REAL)*a->k0;
    const COMPLEX *xr=(const COMPLEX *)a->xr;
    COMPLEX *lx=(COMPLEX *)a->localfieldx;
    COMPLEX *ly=(COMPLEX *)a->localfieldy;
    COMPLEX *lz=(COMPLEX *)a->localfieldz;
    REAL *lm=(REAL *)a->localfield;
    if(c->nlocal==1) {
        if(!lx||!ly||!lz||!lm) return 1;
        FILE *files[4]={NULL,NULL,NULL,NULL};
        if(c->nmat==0) {
            const char *names[4]={"localfieldxwf.mat","localfieldywf.mat",
                                  "localfieldzwf.mat","localfieldwf.mat"};
            for(int j=0;j<4;++j) {
                files[j]=fopen(names[j],"w");
                if(!files[j]) {
                    for(int k=0;k<4;++k) if(files[k]) fclose(files[k]);
                    return 1;
                }
            }
        }
        for(int i=0;i<subunits;++i) {
            const size_t k=(size_t)3*(size_t)i;
            lx[i]=xr[k]; ly[i]=xr[k+1]; lz[i]=xr[k+2];
            lm[i]=sqrt(creal(xr[k]*conj(xr[k])+xr[k+1]*conj(xr[k+1])+
                             xr[k+2]*conj(xr[k+2])));
            if(files[0]) {
                fprintf(files[0]," %.17g %.17g\n",creal(lx[i]),cimag(lx[i]));
                fprintf(files[1]," %.17g %.17g\n",creal(ly[i]),cimag(ly[i]));
                fprintf(files[2]," %.17g %.17g\n",creal(lz[i]),cimag(lz[i]));
                fprintf(files[3]," %.17g\n",lm[i]);
            }
        }
        for(int j=0;j<4;++j) if(files[j]&&fclose(files[j])!=0) return 1;
    }
    if(c->nmacro==1) {
        if(!a->polarisa||!a->macroscopicfieldx||!a->macroscopicfieldy||
           !a->macroscopicfieldz||!a->macroscopicfield) return 1;
        COMPLEX *mx=(COMPLEX *)a->macroscopicfieldx;
        COMPLEX *my=(COMPLEX *)a->macroscopicfieldy;
        COMPLEX *mz=(COMPLEX *)a->macroscopicfieldz;
        REAL *mm=(REAL *)a->macroscopicfield;
        FILE *files[4]={NULL,NULL,NULL,NULL};
        if(c->nmat==0) {
            const char *names[4]={"macroscopicfieldxwf.mat",
                "macroscopicfieldywf.mat","macroscopicfieldzwf.mat",
                "macroscopicfieldwf.mat"};
            for(int j=0;j<4;++j) {
                files[j]=fopen(names[j],"w");
                if(!files[j]) {
                    for(int k=0;k<4;++k) if(files[k]) fclose(files[k]);
                    return 1;
                }
            }
        }
        for(int i=0;i<subunits;++i) {
            const int k=i+1;
            COMPLEX epsani[9],eloc[3],em[3];
            const size_t f=(size_t)3*(size_t)i;
            eloc[0]=xr[f]; eloc[1]=xr[f+1]; eloc[2]=xr[f+2];
            copiani_((const COMPLEX *)a->polarisa,epsani,&nmax,&niso,&k);
            local_macro_(eloc,em,epsani,&arete,&k0,&nsens);
            mx[i]=em[0]; my[i]=em[1]; mz[i]=em[2];
            mm[i]=sqrt(creal(em[0]*conj(em[0])+em[1]*conj(em[1])+
                             em[2]*conj(em[2])));
            if(files[0]) {
                fprintf(files[0]," %.17g %.17g\n",creal(em[0]),cimag(em[0]));
                fprintf(files[1]," %.17g %.17g\n",creal(em[1]),cimag(em[1]));
                fprintf(files[2]," %.17g %.17g\n",creal(em[2]),cimag(em[2]));
                fprintf(files[3]," %.17g\n",mm[i]);
            }
        }
        for(int j=0;j<4;++j) if(files[j]&&fclose(files[j])!=0) return 1;
    }
    return 0;
}
//-------------------------------------------

void cdmlib_(
    REAL * lambda,
    REAL * eps0i,
    int * ntypeabc,
    REAL * lnrbc,
    REAL * lcos,
    char * beam,
    char * object,
    char * trope,
    int * niso,
    char * materiaumulti,
    int * nnnr,
    REAL * tolinit,
    int * nlim,
    char * methodeit,
    char * polarizability,
    int * nprecon,
    int * ninitest,
    int * nquad,
    int * nlecture,
    char * filereread,
    int * nmat,
    char * h5file,
    int * nlocal,
    int * nmacro,
    int * nsection,
    int * nsectionsca,
    int * nquickdiffracte,
    int * nrig,
    int * nforce,
    int * nforced,
    int * ntorque,
    int * ntorqued,
    int * nproche,
    int * nlentille,
    int * nquicklens,
    int * nenergie,
    int * nobjet,
    REAL * density,
    REAL * side,
    REAL * sidex,
    REAL * sidey,
    REAL * sidez,
    REAL * hauteur,
    int * numberobjet,
    REAL * rayonmulti,
    REAL * xgmulti,
    REAL * ygmulti,
    REAL * zgmulti,
    COMPLEX * epsmulti,
    COMPLEX * epsanimulti,
    REAL * lc,
    REAL * hc,
    int * ng,
    REAL * demiaxea,
    REAL * demiaxeb,
    REAL * demiaxec,
    REAL * thetaobj,
    REAL * phiobj,
    REAL * psiobj,
    char * namefileobj,
    REAL * theta,
    REAL * phi,
    REAL * pp,
    REAL * ss,
    REAL * p0,
    REAL * w0,
    REAL * xgaus,
    REAL * ygaus,
    REAL * zgaus,
    int * speckseed,
    char * namefileinc,
    REAL * thetam,
    REAL * phim,
    REAL * ppm,
    REAL * ssm,
    COMPLEX * e0m,
    int * nbinc,
    char * infostr,
    int * nstop,
    int * nbsphere,
    int * ndipole,
    REAL * aretecube,
    REAL * lambda10n,
    REAL * k0,
    REAL * tol1,
    REAL * tempsreelmvp,
    REAL * tempsreeltotal,
    int * ncompte,
    int * nloop,
    REAL * efficacite,
    REAL * efficaciteref,
    REAL * efficacitetrans,
    REAL * cext,
    REAL * cabs_out,
    REAL * csca,
    REAL * cscai,
    REAL * gasym,
    REAL * irra,
    COMPLEX * e0,
    REAL * forcet,
    REAL * forcem,
    REAL * couplet,
    REAL * couplem,
    int * nxm,
    int * nym,
    int * nzm,
    int * nxmp,
    int * nymp,
    int * nzmp,
    int * nmaxpp,
    int * nxym1,
    int * nzm1,
    int * nzms1,
    REAL * incidentfield,
    REAL * localfield,
    REAL * macroscopicfield,
    REAL * xs,
    REAL * ys,
    REAL * zs,
    REAL * xswf,
    REAL * yswf,
    REAL * zswf,
    int * ntheta,
    int * nphi,
    REAL * thetafield,
    REAL * phifield,
    REAL * poyntingfield,
    REAL * kxypoynting,
    REAL * poyntingfieldpos,
    REAL * poyntingfieldneg,
    REAL * forcex,
    REAL * forcey,
    REAL * forcez,
    REAL * forcexmulti,
    REAL * forceymulti,
    REAL * forcezmulti,
    REAL * torquex,
    REAL * torquey,
    REAL * torquez,
    REAL * torquexmulti,
    REAL * torqueymulti,
    REAL * torquezmulti,
    COMPLEX * incidentfieldx,
    COMPLEX * incidentfieldy,
    COMPLEX * incidentfieldz,
    COMPLEX * localfieldx,
    COMPLEX * localfieldy,
    COMPLEX * localfieldz,
    COMPLEX * macroscopicfieldx,
    COMPLEX * macroscopicfieldy,
    COMPLEX * macroscopicfieldz,
    COMPLEX * polarisa,
    COMPLEX * epsilon,
    int * nfft2d,
    int * npoynting,
    REAL * deltapoyntingx,
    REAL * deltapoyntingy,
    COMPLEX * eimagex,
    COMPLEX * eimagey,
    COMPLEX * eimagez,
    COMPLEX * eimageincx,
    COMPLEX * eimageincy,
    COMPLEX * eimageincz,
    COMPLEX * efourierx,
    COMPLEX * efouriery,
    COMPLEX * efourierz,
    COMPLEX * efourierincx,
    COMPLEX * efourierincy,
    COMPLEX * efourierincz,
    REAL * kxy,
    REAL * xy,
    REAL * numaper,
    REAL * numaperinc,
    REAL * numaperinc2,
    REAL * kcnax,
    REAL * kcnay,
    REAL * numaperil,
    REAL * gross,
    REAL * zlens,
    REAL * psiinc,
    int * ntypemic,
    int * nside,
    COMPLEX * ff,
    COMPLEX * ff0,
    COMPLEX * ffloc,
    COMPLEX * xr,
    COMPLEX * xi,
    COMPLEX * ffprecon,
    COMPLEX * wrk,
    int * nlar,
    COMPLEX * ffttensorxx,
    COMPLEX * ffttensorxy,
    COMPLEX * ffttensorxz,
    COMPLEX * ffttensoryy,
    COMPLEX * ffttensoryz,
    COMPLEX * ffttensorzz,
    COMPLEX * ediffkzpos,
    COMPLEX * ediffkzneg,
    COMPLEX * sdetnn,
    int * tabdip,
    int * tabmulti,
    int * tabfft2,
    size_t beam_len,
    size_t object_len,
    size_t trope_len,
    size_t materiaumulti_len,
    size_t methodeit_len,
    size_t polarizability_len,
    size_t filereread_len,
    size_t h5file_len,
    size_t namefileobj_len,
    size_t namefileinc_len,
    size_t infostr_len
)
{
    if (!ifdda_reset_cuda_device_state()) {
        if (nstop)
            *nstop = 1;
        return;
    }
    ifdda_reset_preconditionneur_apply_cuda_state();
    ifdda_reset_preconditionneurscal_cuda_state();

    // copy cdmlib_(...) arguments int a;

    cdmlib_abi_args_c99 a{};
    a.lambda = lambda;
    a.eps0i = eps0i;
    a.ntypeabc = ntypeabc;
    a.lnrbc = lnrbc;
    a.lcos = lcos;
    a.beam = beam;
    a.object = object;
    a.trope = trope;
    a.niso = niso;
    a.materiaumulti = materiaumulti;
    a.nnnr = nnnr;
    a.tolinit = tolinit;
    a.nlim = nlim;
    a.methodeit = methodeit;
    a.polarizability = polarizability;
    a.nprecon = nprecon;
    a.ninitest = ninitest;
    a.nquad = nquad;
    a.nlecture = nlecture;
    a.filereread = filereread;
    a.nmat = nmat;
    a.h5file = h5file;
    a.nlocal = nlocal;
    a.nmacro = nmacro;
    a.nsection = nsection;
    a.nsectionsca = nsectionsca;
    a.nquickdiffracte = nquickdiffracte;
    a.nrig = nrig;
    a.nforce = nforce;
    a.nforced = nforced;
    a.ntorque = ntorque;
    a.ntorqued = ntorqued;
    a.nproche = nproche;
    a.nlentille = nlentille;
    a.nquicklens = nquicklens;
    a.nenergie = nenergie;
    a.nobjet = nobjet;
    a.density = density;
    a.side = side;
    a.sidex = sidex;
    a.sidey = sidey;
    a.sidez = sidez;
    a.hauteur = hauteur;
    a.numberobjet = numberobjet;
    a.rayonmulti = rayonmulti;
    a.xgmulti = xgmulti;
    a.ygmulti = ygmulti;
    a.zgmulti = zgmulti;
    a.epsmulti = epsmulti;
    a.epsanimulti = epsanimulti;
    a.lc = lc;
    a.hc = hc;
    a.ng = ng;
    a.demiaxea = demiaxea;
    a.demiaxeb = demiaxeb;
    a.demiaxec = demiaxec;
    a.thetaobj = thetaobj;
    a.phiobj = phiobj;
    a.psiobj = psiobj;
    a.namefileobj = namefileobj;
    a.theta = theta;
    a.phi = phi;
    a.pp = pp;
    a.ss = ss;
    a.p0 = p0;
    a.w0 = w0;
    a.xgaus = xgaus;
    a.ygaus = ygaus;
    a.zgaus = zgaus;
    a.speckseed = speckseed;
    a.namefileinc = namefileinc;
    a.thetam = thetam;
    a.phim = phim;
    a.ppm = ppm;
    a.ssm = ssm;
    a.e0m = e0m;
    a.nbinc = nbinc;
    a.infostr = infostr;
    a.nstop = nstop;
    a.nbsphere = nbsphere;
    a.ndipole = ndipole;
    a.aretecube = aretecube;
    a.lambda10n = lambda10n;
    a.k0 = k0;
    a.tol1 = tol1;
    a.tempsreelmvp = tempsreelmvp;
    a.tempsreeltotal = tempsreeltotal;
    a.ncompte = ncompte;
    a.nloop = nloop;
    a.efficacite = efficacite;
    a.efficaciteref = efficaciteref;
    a.efficacitetrans = efficacitetrans;
    a.cext = cext;
    a.cabs = cabs_out;
    a.csca = csca;
    a.cscai = cscai;
    a.gasym = gasym;
    a.irra = irra;
    a.e0 = e0;
    a.forcet = forcet;
    a.forcem = forcem;
    a.couplet = couplet;
    a.couplem = couplem;
    a.nxm = nxm;
    a.nym = nym;
    a.nzm = nzm;
    a.nxmp = nxmp;
    a.nymp = nymp;
    a.nzmp = nzmp;
    a.nmaxpp = nmaxpp;
    a.nxym1 = nxym1;
    a.nzm1 = nzm1;
    a.nzms1 = nzms1;
    a.incidentfield = incidentfield;
    a.localfield = localfield;
    a.macroscopicfield = macroscopicfield;
    a.xs = xs;
    a.ys = ys;
    a.zs = zs;
    a.xswf = xswf;
    a.yswf = yswf;
    a.zswf = zswf;
    a.ntheta = ntheta;
    a.nphi = nphi;
    a.thetafield = thetafield;
    a.phifield = phifield;
    a.poyntingfield = poyntingfield;
    a.kxypoynting = kxypoynting;
    a.poyntingfieldpos = poyntingfieldpos;
    a.poyntingfieldneg = poyntingfieldneg;
    a.forcex = forcex;
    a.forcey = forcey;
    a.forcez = forcez;
    a.forcexmulti = forcexmulti;
    a.forceymulti = forceymulti;
    a.forcezmulti = forcezmulti;
    a.torquex = torquex;
    a.torquey = torquey;
    a.torquez = torquez;
    a.torquexmulti = torquexmulti;
    a.torqueymulti = torqueymulti;
    a.torquezmulti = torquezmulti;
    a.incidentfieldx = incidentfieldx;
    a.incidentfieldy = incidentfieldy;
    a.incidentfieldz = incidentfieldz;
    a.localfieldx = localfieldx;
    a.localfieldy = localfieldy;
    a.localfieldz = localfieldz;
    a.macroscopicfieldx = macroscopicfieldx;
    a.macroscopicfieldy = macroscopicfieldy;
    a.macroscopicfieldz = macroscopicfieldz;
    a.polarisa = polarisa;
    a.epsilon = epsilon;
    a.nfft2d = nfft2d;
    a.npoynting = npoynting;
    a.deltapoyntingx = deltapoyntingx;
    a.deltapoyntingy = deltapoyntingy;
    a.eimagex = eimagex;
    a.eimagey = eimagey;
    a.eimagez = eimagez;
    a.eimageincx = eimageincx;
    a.eimageincy = eimageincy;
    a.eimageincz = eimageincz;
    a.efourierx = efourierx;
    a.efouriery = efouriery;
    a.efourierz = efourierz;
    a.efourierincx = efourierincx;
    a.efourierincy = efourierincy;
    a.efourierincz = efourierincz;
    a.kxy = kxy;
    a.xy = xy;
    a.numaper = numaper;
    a.numaperinc = numaperinc;
    a.numaperinc2 = numaperinc2;
    a.kcnax = kcnax;
    a.kcnay = kcnay;
    a.numaperil = numaperil;
    a.gross = gross;
    a.zlens = zlens;
    a.psiinc = psiinc;
    a.ntypemic = ntypemic;
    a.nside = nside;
    a.ff = ff;
    a.ff0 = ff0;
    a.ffloc = ffloc;
    a.xr = xr;
    a.xi = xi;
    a.ffprecon = ffprecon;
    a.wrk = wrk;
    a.nlar = nlar;
    a.ffttensorxx = ffttensorxx;
    a.ffttensorxy = ffttensorxy;
    a.ffttensorxz = ffttensorxz;
    a.ffttensoryy = ffttensoryy;
    a.ffttensoryz = ffttensoryz;
    a.ffttensorzz = ffttensorzz;
    a.ediffkzpos = ediffkzpos;
    a.ediffkzneg = ediffkzneg;
    a.sdetnn = sdetnn;
    a.tabdip = tabdip;
    a.tabmulti = tabmulti;
    a.tabfft2 = tabfft2;
    a.beam_len = beam_len;
    a.object_len = object_len;
    a.trope_len = trope_len;
    a.materiaumulti_len = materiaumulti_len;
    a.methodeit_len = methodeit_len;
    a.polarizability_len = polarizability_len;
    a.filereread_len = filereread_len;
    a.h5file_len = h5file_len;
    a.namefileobj_len = namefileobj_len;
    a.namefileinc_len = namefileinc_len;
    a.infostr_len = infostr_len;
    
    
    // INITIALISATIONS
    
    cdmlib_initphase_c99 init{};
    cdmlib_config_c99 cfg{};
    cdmlib_entry_runtime_c99 runtime{};
    std::vector<REAL> pml_mask;
    int init5_two_stage=0;
    int init5_saved_precon=0;
    REAL init5_saved_tol=0.0;
    int reread_local_fields=0;
    // Only scalar nrig=8/9 needs these results beyond the Green phase:
    // the normalized polarization is consumed later by the scalar solver.
    // No borrowed parameter view is retained across phases.
    struct {
        COMPLEX ux=0.0,uy=0.0,uz=0.0;
        int ready=0;
    } scalar_green_result;
    runtime.file_id=runtime.group_idopt=runtime.group_iddip=runtime.group_idff=-1;
    runtime.group_idmic=runtime.group_idof=runtime.group_idnf=runtime.group_idver=-1;

    // --GEOMETRY START--

    int rc=cdmlib_initialize_geometry_c99(&a,&init,&cfg,&runtime);
    if(rc==0) {
        // Keep this view visible in the cdmlib_ debugger frame.
        // Initialize all borrowed pointers immediately after declaration.
        REAL geometry_k0=0.0,geometry_k03=0.0,geometry_lambda10n=0.0;
        cdmlib_geometry_params_c99 geometry{};
        rc=cdmlib_prepare_geometry_params_c99(
            &a,&init,&runtime,&geometry_k0,&geometry);
        if(rc==0)
            rc=cdmlib_execute_geometry_c99(
                &a,&init,&cfg,&runtime,&geometry,
                &geometry_k0,&geometry_k03,&geometry_lambda10n);
    }

    // --GEOMETRY END--

    // --ILLUMINATION START--
    {
        cdmlib_illumination_params_c99 illumination{};
        if(rc==0)
            rc=cdmlib_prepare_illumination_params_c99(
                &a,&init,&cfg,&runtime,&pml_mask,&illumination);
        if(rc==0)
            rc=cdmlib_execute_illumination_c99(&illumination);
    }

    // --ILLUMINATION END--


    // --RELECTURE START--

    // Chemin distinct : relecture des champs au lieu d'une illumination normale.
    if(rc==0 && (int)*nlecture==1)
        rc=cdmlib_relecture_from_abi_c99(&a,&runtime,
                                                &reread_local_fields);
    // --RELECTURE END--

    // --CALCULATION MODE CONFIGURATION START--
    // Special "greentensor" excitation: solve the rigorous vector problem
    // for successive Cartesian incident-field components (testgreen).
    // Disable unrelated observables and keep only the macroscopic field.
    // Update both the local configuration and the caller's ABI parameters.
    if(rc==0 && cfg.testgreen>0) {
        cfg.nrig=0; cfg.nforce=0; cfg.ntorque=0; cfg.nforced=0; cfg.ntorqued=0;
        cfg.nsection=0; cfg.nsectionsca=0; cfg.nquickdiffracte=0;
        cfg.nlentille=0; cfg.nquicklens=0; cfg.nenergie=0; cfg.nlocal=0;
        cfg.nmacro=1;
        *nrig=(cdmlib_ip)0; *nforce=(cdmlib_ip)0; *ntorque=(cdmlib_ip)0;
        *nforced=(cdmlib_ip)0; *ntorqued=(cdmlib_ip)0;
        *nsection=(cdmlib_ip)0; *nsectionsca=(cdmlib_ip)0;
        *nquickdiffracte=(cdmlib_ip)0; *nlentille=(cdmlib_ip)0;
        *nquicklens=(cdmlib_ip)0; *nenergie=(cdmlib_ip)0;
        *nlocal=(cdmlib_ip)0; *nmacro=(cdmlib_ip)1;
    }
    // --SOLVER INITIAL GUESS CONFIGURATION START--
    /* Fortran ninitest=5 with nrig=0 enters the nrig=8 scalar-only path.
       The scalar-to-vector restart is exclusive to the nrig=9 branch. */
    if(rc==0 && cfg.ninitest==5 && cfg.nrig==9) {
        init5_two_stage=1;
        init5_saved_precon=(int)*nprecon;
        init5_saved_tol=(REAL)*tolinit;
    }
    if(rc==0 && cfg.ninitest==5 && cfg.nrig==0) {
        // Scalar-only nrig=8 solve at the originally requested tolerance.
        cfg.nrig=8;
        cfg.nprecon=0;
        *nrig=(cdmlib_ip)8;
        *nprecon=(cdmlib_ip)0;
    }
    if(rc==0 && cfg.ninitest==4 && (cfg.nrig==8 || cfg.nrig==9)) {
        // Scalar modes do not use the vector solver's optional scalar
        // warm start (ninitest=4); use initial guess 1 instead.
        cfg.ninitest=1;
        *ninitest=(cdmlib_ip)1;
    }
    if(rc==0 && init5_two_stage) {
        // First stage of the two-stage path: solve the scalar problem at 1%.
        // The requested preconditioner/tolerance were saved above and will
        // be restored before the subsequent vector solve.
        if(cfg.nrig!=9) {
            cfg.nrig=8;
            *nrig=(cdmlib_ip)8;
        }
        cfg.nprecon=0;
        *nprecon=(cdmlib_ip)0;
        *tolinit=(cdmlib_real)0.01;
    }
    // --SOLVER INITIAL GUESS CONFIGURATION STOP--
    // --CALCULATION MODE CONFIGURATION STOP--

    // --FFT WORKSPACE PREPARATION START--
    // Re-read path: prepare the padded grid, temporary FFT work arrays and
    // FFTW plans needed to process an existing field. This is not Green
    // generation or a solver step.
    if(rc==0 && reread_local_fields) {
        rc=cdmlib_prepare_fft_grid_from_abi_c99(&a,&init,&cfg,&runtime,
                                                 (int)*nbsphere);
        // Re-reading needs the host FFT workspace for field mapping.
        if(rc==0) rc=cdmlib_allocate_fft_workspace_c99(&runtime);
#ifdef USE_FFTW
        if(rc==0) rc=cdmlib_fftw_create_c99(&runtime,&a);
#endif
    }
    // Normal computation: the listed modes require FFT workspace before
    // Green construction and/or field propagation. Re-reading is excluded
    // to avoid initializing the same workspace twice.
    if(rc==0 && !reread_local_fields &&
       (cfg.nrig==0 || cfg.nrig==3 || cfg.nrig==4 || cfg.nrig==5 ||
                 cfg.nrig==8 || cfg.nrig==9 ||
                 (cfg.nrig>=10 && cfg.nrig<=15))) {
        rc=cdmlib_prepare_fft_grid_from_abi_c99(&a,&init,&cfg,&runtime,
                                                 (int)*nbsphere);
        // The GPU-native nprecon=0 vector solver uses resident cuFFT buffers.
        // GPU Chan solvers accept null legacy CPU vectx/vecty/vectz and
        // perform FFTs using persistent GPU buffers and cuFFT plans.
        // Allocate host FFT scratch only for other legacy paths.
        const bool needs_legacy_fft_workspace =
            cfg.nrig==3 || cfg.nrig==4 || cfg.nrig==5 ||
            cfg.nrig==8 || cfg.nrig==9 ||
            // The vectorial C++ microscopy paths call the resident GPU
            // inverse directly. Retain host scratch only for legacy
            // scalar/approximation modes, not for nrig=0 microscopy.
            (cfg.nlentille==1 && cfg.nrig!=0) ||
            (cfg.nrig==0 && (cfg.ninitest==4 || cfg.ninitest==5));
        if(rc==0 && needs_legacy_fft_workspace)
            rc=cdmlib_allocate_fft_workspace_c99(&runtime);
#ifdef USE_FFTW
        if(rc==0 && needs_legacy_fft_workspace)
            rc=cdmlib_fftw_create_c99(&runtime,&a);
#endif
        // --GREEN SCALAR START--

        // Centralized beam validation, polarization and scalar Green FFT.
        // Preserve normalized polarization for the scalar solver.
        if(rc==0 && (cfg.nrig==8 || cfg.nrig==9)) {
            cdmlib_green_scalar_params_c99 scalar_green{};
            rc=cdmlib_prepare_green_scalar_params_c99(
                &a,&init,&runtime,cfg.nrig,0,
                &scalar_green_result.ux,&scalar_green_result.uy,
                &scalar_green_result.uz,&scalar_green_result.ready,
                &scalar_green);
            if(rc==0)
                rc=cdmlib_execute_green_scalar_c99(&scalar_green);
        }
        // --GREEN SCALAR STOP--

    }
    // --FFT WORKSPACE PREPARATION STOP--

solver_repeated:
    if(reread_local_fields)
        goto cdmlib_after_solver_dispatch;


    // --VECTOR SOLVER OPTIONAL SCALAR INITIALIZATION START--

    if(rc==0 && cfg.nrig==0) {

        // Option ninitest=4 only: build the scalar Green operator (mode 8)
        // and solve the cheaper scalar problem to 1% tolerance. The scalar
        // field in ffloc initializes xi for the rigorous vector solver.
        // This is optional and may not reduce total runtime. It must run
        // before vector Green generation, since both reuse FFT work buffers.
        if(cfg.ninitest==4) {
            COMPLEX ux=0.0,uy=0.0,uz=0.0;
            int warm_start_green_ready=0;

            // GREEN
            cdmlib_green_scalar_params_c99 scalar_green{};
            rc=cdmlib_prepare_green_scalar_params_c99(
                &a,&init,&runtime,8,2,&ux,&uy,&uz,
                &warm_start_green_ready,&scalar_green);
            if(rc==0)
                rc=cdmlib_execute_green_scalar_c99(&scalar_green);

            // SCALAR SOLVER
            if(rc==0 && warm_start_green_ready) {
                cdmlib_scalar_initialization_params_c99 scalar_solver{};
                rc=cdmlib_prepare_scalar_initialization_params_c99(
                    &a,&init,&runtime,&ux,&uy,&uz,&scalar_solver);
                if(rc==0)
                    rc=cdmlib_execute_scalar_initialization_c99(&scalar_solver);
            }
        }
    }
    // --VECTOR SOLVER OPTIONAL SCALAR INITIALIZATION STOP--

    // --GREEN VECTOR START--
    // Single vector Green phase for nrig=0,3,4,5. For nrig=0, the optional
    // ninitest=4 scalar Green and scalar solve have already completed.
    // Also used after the scalar-to-vector transition of ninitest=5.
    if(rc==0 && !runtime.vector_green_ready &&
       (cfg.nrig==0 || cfg.nrig==3 ||
        cfg.nrig==4 || cfg.nrig==5)) {
        cdmlib_green_params_c99 green{};
        rc=cdmlib_prepare_green_params_c99(&a,&init,&cfg,&runtime,&green);
        if(rc==0)
            rc=cdmlib_execute_green_c99(&green);
    }
    // --GREEN VECTOR STOP--

    // Keep specialized lambdas inside the dispatch scope: the nlecture
    // fast path jumps directly to cdmlib_after_solver_dispatch.
    {
    // --BORN AND RYTOV SPECIALIZED CALCULATIONS START--
    // --BORN RENORMALISE START--
    auto cdmlib_born_renormalise_c99 = [&]() {
        puts("Renormalized Born approximation");
        const int n3=3*(int)*nbsphere;
        COMPLEX *fl=(COMPLEX *)ffloc;
        const COMPLEX *inc=(const COMPLEX *)ff0;
        for(int i=0;i<n3;++i) fl[i]=inc[i];
        const int ni=(int)*niso,nm=init.nmax,nb=(int)*nbsphere;
        produitpolavect_(polarisa,&ni,&nm,&nb,fl,(COMPLEX *)ff);
    };
    // --BORN RENORMALISE STOP--

    // --BORN START--
    auto cdmlib_born_c99 = [&]() {
        puts("Born approximation");
        const int ni=(int)*niso,nm=init.nmax,nb=(int)*nbsphere,kdir=-1;
        const REAL a=(REAL)*aretecube, wave=(REAL)*k0;
        const COMPLEX *eps=(const COMPLEX *)epsilon;
        const COMPLEX *inc=(const COMPLEX *)ff0;
        COMPLEX *fl=(COMPLEX *)ffloc;
        COMPLEX *out=(COMPLEX *)ff;
        for(int i=0;i<nb;++i) {
            const size_t k=(size_t)3*(size_t)i;
            const int fi=i+1;
            COMPLEX epsani[9],em[3],eloc[3];
            em[0]=inc[k]; em[1]=inc[k+1]; em[2]=inc[k+2];
            copiani_(eps,epsani,&nm,&ni,&fi);
            local_macro_(eloc,em,epsani,&a,&wave,&kdir);
            fl[k]=eloc[0]; fl[k+1]=eloc[1]; fl[k+2]=eloc[2];
        }
        produitpolavect_(polarisa,&ni,&nm,&nb,fl,out);
    };
    // --BORN STOP--

    // --BORN SERIES ORDER 1 START--
    auto cdmlib_born_series1_c99 = [&]() {
        puts("Born series order 1");
        const int n3=3*(int)*nbsphere;
        COMPLEX *xr=(COMPLEX *)a.xr;
        COMPLEX *xi=(COMPLEX *)a.xi;
        const COMPLEX *ff0c=(const COMPLEX *)ff0;
        memcpy(xr,ff0c,(size_t)n3*sizeof *xr);
        const int ni=(int)*niso,nm=init.nmax,nb=(int)*nbsphere;
        produitpolavect_(polarisa,&ni,&nm,&nb,xr,xi);
        for(int i=0;i<n3;++i) xi[i]=-xi[i];
        const int ntotalm=init.ntotalm,ntotal=runtime.ntotal,nd=(int)*ndipole;
        const int nxm=init.nxm,nym=init.nym,nzm=init.nzm;
        const int nx=runtime.nx,ny=runtime.ny,nz=runtime.nz;
        const int nx2=runtime.nx2,ny2=runtime.ny2,nxy2=runtime.nxy2,nz2=runtime.nz2;
        produitfftmatvect3_((COMPLEX *)a.ffttensorxx,
            (COMPLEX *)a.ffttensorxy,(COMPLEX *)a.ffttensorxz,
            (COMPLEX *)a.ffttensoryy,(COMPLEX *)a.ffttensoryz,
            (COMPLEX *)a.ffttensorzz,runtime.vectx,runtime.vecty,
            runtime.vectz,(const int *)a.tabdip,&ntotalm,&ntotal,&nm,&nd,
            &nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,xi,xr,
            &runtime.planb,&runtime.planf);
        produitpolavect_(polarisa,&ni,&nm,&nb,xr,(COMPLEX *)ff);
        memcpy((COMPLEX *)ffloc,xr,(size_t)n3*sizeof *xr);
    };
    // --BORN SERIES ORDER 1 STOP--

    // --RYTOV LOCAL START--
    auto cdmlib_rytov_local_c99 = [&]() {
        puts("Rytov approximation on local field");
        const int n3=3*(int)*nbsphere;
        COMPLEX *xr=(COMPLEX *)a.xr;
        COMPLEX *xi=(COMPLEX *)a.xi;
        const COMPLEX *ff0c=(const COMPLEX *)ff0;
        COMPLEX *fl=(COMPLEX *)ffloc;
        for(int i=0;i<n3;++i) xr[i]=0.0;
        const int ni=(int)*niso,nm=init.nmax,nb=(int)*nbsphere;
        produitpolavect_(polarisa,&ni,&nm,&nb,ff0c,xi);
        for(int i=0;i<n3;++i) xi[i]=-xi[i];
        const int ntotalm=init.ntotalm,ntotal=runtime.ntotal,nd=(int)*ndipole;
        const int nxm=init.nxm,nym=init.nym,nzm=init.nzm;
        const int nx=runtime.nx,ny=runtime.ny,nz=runtime.nz;
        const int nx2=runtime.nx2,ny2=runtime.ny2,nxy2=runtime.nxy2,nz2=runtime.nz2;
        produitfftmatvect3_((COMPLEX *)a.ffttensorxx,
            (COMPLEX *)a.ffttensorxy,(COMPLEX *)a.ffttensorxz,
            (COMPLEX *)a.ffttensoryy,(COMPLEX *)a.ffttensoryz,
            (COMPLEX *)a.ffttensorzz,runtime.vectx,runtime.vecty,
            runtime.vectz,(const int *)a.tabdip,&ntotalm,&ntotal,&nm,&nd,
            &nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,xi,xr,
            &runtime.planb,&runtime.planf);
        if(cdmlib_rytov_vector_field_c99(ff0c,xr,fl,nb)) {
            *nstop=1;
            cdmlib_rytov_error_c99(infostr,infostr_len);
            rc=1;
        } else {
            produitpolavect_(polarisa,&ni,&nm,&nb,fl,(COMPLEX *)ff);
        }
    };
    // --RYTOV LOCAL STOP--

    // --RYTOV MACRO START--
    auto cdmlib_rytov_macro_c99 = [&]() {
        puts("Rytov approximation on macroscopic field");
        const int n3=3*(int)*nbsphere;
        const int ni=(int)*niso,nm=init.nmax,nb=(int)*nbsphere,ndir=-1;
        const int ntotalm=init.ntotalm,ntotal=runtime.ntotal,nd=(int)*ndipole;
        const int nxm=init.nxm,nym=init.nym,nzm=init.nzm;
        const int nx=runtime.nx,ny=runtime.ny,nz=runtime.nz;
        const int nx2=runtime.nx2,ny2=runtime.ny2,nxy2=runtime.nxy2,nz2=runtime.nz2;
        const REAL aretecube_value=(REAL)*aretecube,wave=(REAL)*k0;
        const COMPLEX *eps=(const COMPLEX *)epsilon;
        const COMPLEX *alpha=(const COMPLEX *)polarisa;
        const COMPLEX *inc=(const COMPLEX *)ff0;
        COMPLEX *macro=(COMPLEX *)ff;
        COMPLEX *fl=(COMPLEX *)ffloc;
        COMPLEX *xi=(COMPLEX *)a.xi;
        COMPLEX *xr=(COMPLEX *)a.xr;
        for(int i=0;i<nb;++i) {
            const size_t k=(size_t)3*(size_t)i;
            const int fi=i+1;
            COMPLEX epsani[9],em[3],eloc[3];
            em[0]=inc[k]; em[1]=inc[k+1]; em[2]=inc[k+2];
            copiani_(eps,epsani,&nm,&ni,&fi);
            local_macro_(eloc,em,epsani,&aretecube_value,&wave,&ndir);
            macro[k]=eloc[0]; macro[k+1]=eloc[1]; macro[k+2]=eloc[2];
            if(ni==1) {
                const COMPLEX p=eps[i]-init.eps0;
                xr[k]=-inc[k]*p/3.0; xr[k+1]=-inc[k+1]*p/3.0;
                xr[k+2]=-inc[k+2]*p/3.0;
                xi[k]=-alpha[i]*macro[k];
                xi[k+1]=-alpha[i]*macro[k+1];
                xi[k+2]=-alpha[i]*macro[k+2];
            } else {
                const size_t e0=(size_t)i+(size_t)nm*0;
                const size_t e1=(size_t)i+(size_t)nm*4;
                const size_t e2=(size_t)i+(size_t)nm*8;
                xr[k]=-inc[k]*(eps[e0]-init.eps0)/3.0;
                xr[k+1]=-inc[k+1]*(eps[e1]-init.eps0)/3.0;
                xr[k+2]=-inc[k+2]*(eps[e2]-init.eps0)/3.0;
                const size_t p0=(size_t)i;
                const size_t p1=(size_t)i+(size_t)nm*3;
                const size_t p2=(size_t)i+(size_t)nm*6;
                const size_t p3=(size_t)i+(size_t)nm*1;
                const size_t p4=(size_t)i+(size_t)nm*4;
                const size_t p5=(size_t)i+(size_t)nm*7;
                const size_t p6=(size_t)i+(size_t)nm*2;
                const size_t p7=(size_t)i+(size_t)nm*5;
                const size_t p8=(size_t)i+(size_t)nm*8;
                xi[k]=-alpha[p0]*macro[k]-alpha[p3]*macro[k+1]-alpha[p6]*macro[k+2];
                xi[k+1]=-alpha[p1]*macro[k]-alpha[p4]*macro[k+1]-alpha[p7]*macro[k+2];
                xi[k+2]=-alpha[p2]*macro[k]-alpha[p5]*macro[k+1]-alpha[p8]*macro[k+2];
            }
        }
        produitfftmatvect3_((COMPLEX *)a.ffttensorxx,
            (COMPLEX *)a.ffttensorxy,(COMPLEX *)a.ffttensorxz,
            (COMPLEX *)a.ffttensoryy,(COMPLEX *)a.ffttensoryz,
            (COMPLEX *)a.ffttensorzz,runtime.vectx,runtime.vecty,
            runtime.vectz,(const int *)a.tabdip,&ntotalm,&ntotal,&nm,&nd,
            &nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,xi,xr,
            &runtime.planb,&runtime.planf);
        if(cdmlib_rytov_vector_field_c99(macro,xr,fl,nb)) {
            *nstop=1;
            cdmlib_rytov_error_c99(infostr,infostr_len);
            rc=1;
        } else {
            produitpolavect_(polarisa,&ni,&nm,&nb,fl,(COMPLEX *)ff);
        }
        (void)n3;
    };
    // --RYTOV MACRO STOP--

    // --BORN AND RYTOV SPECIALIZED CALCULATIONS STOP--

    // --CALCULATION METHOD SELECTION START--
    // nrig selects the electromagnetic calculation method:
    //   0     : Rigorous vector DDA (iterative solver)
    //   1     : Renormalized Born approximation
    //   2     : Born approximation
    //   3     : Born series, first order
    //   4     : Rytov approximation on the local field
    //   5     : Rytov approximation on the macroscopic field
    //   6     : Beam Propagation Method (BPM)
    //   7     : Renormalized BPM
    //   8     : Scalar approximation Uinc.GUinv
    //   9     : Scalar approximation exp(ikr)/r
    //   10-13 : Multilayer scalar Born methods
    //   14-15 : Multilayer vector Born methods
    // nrig is distinct from:
    //   methodeit : iterative algorithm (GPBICG1, etc.)
    //   nabc      : 0 = sans PML (DDA standard)
    //               1 = avec PML (mode MDDA)
    // --CALCULATION METHOD SELECTION END--

    if(rc==0 && cfg.nrig==0) {
        // --VECTOR SOLVER START--
        rc=cdmlib_solve_from_abi_c99(&a,&init,&cfg,&runtime,
                                    (int)*nbsphere,(int)*ndipole);
        // --VECTOR SOLVER STOP--
    }
    // --BORN RENORMALISE START--
    else if(rc==0 && cfg.nrig==1) {
        cdmlib_born_renormalise_c99();
    // --BORN RENORMALISE STOP--
    // --BORN START--
    } else if(rc==0 && cfg.nrig==2) {
        cdmlib_born_c99();
    // --BORN STOP--
    // --BORN SERIES ORDER 1 START--
    } else if(rc==0 && cfg.nrig==3) {
        cdmlib_born_series1_c99();
    // --BORN SERIES ORDER 1 STOP--
    // --RYTOV LOCAL START--
    } else if(rc==0 && cfg.nrig==4) {
        cdmlib_rytov_local_c99();
    // --RYTOV LOCAL STOP--
    // --RYTOV MACRO START--
    } else if(rc==0 && cfg.nrig==5) {
        cdmlib_rytov_macro_c99();
    // --RYTOV MACRO STOP--
    } else if(rc==0 && (cfg.nrig==6 || cfg.nrig==7)) {
        const int forbidden=memcmp(beam,"antenna",7)==0 ||
                            memcmp(beam,"wavelinearmulti",15)==0;
        if(forbidden) {
            static const char msg[]="BPM not possible with this beam";
            memset(infostr,' ',infostr_len);
            memcpy(infostr,msg,sizeof msg-1<infostr_len?sizeof msg-1:infostr_len);
            *nstop=1;
            rc=1;
        } else {
            const int resident_mode=0;
            if(cfg.nrig==6) {
                puts("Beam propagation method");
                beampropagationmacro_(xs,ys,zs,aretecube,k0,w0,e0,ss,pp,
                    theta,phi,xgaus,ygaus,zgaus,beam,epsilon,(const int *)niso,
                    (const int *)ndipole,(const int *)&runtime.nx,
                    (const int *)&runtime.ny,(const int *)&runtime.nz,
                    (const int *)&init.nxm,(const int *)&init.nym,
                    (const int *)&init.nzm,(const int *)&init.nmax,
                    (const int *)nfft2d,(COMPLEX *)eimagex,
                    (COMPLEX *)eimagey,(COMPLEX *)eimagez,
                    (COMPLEX *)ff0,(COMPLEX *)ffloc,
                    (COMPLEX *)ff,&runtime.plan2f,&runtime.plan2b,
                    (int *)nstop,infostr,beam_len,infostr_len);
            } else {
                puts("renormalized Beam propagation method");
                beampropagation_(xs,ys,zs,aretecube,k0,w0,e0,ss,pp,
                    theta,phi,xgaus,ygaus,zgaus,beam,epsilon,(const int *)niso,
                    (const int *)ndipole,(const int *)&runtime.nx,
                    (const int *)&runtime.ny,(const int *)&runtime.nz,
                    (const int *)&init.nxm,(const int *)&init.nym,
                    (const int *)&init.nzm,(const int *)&init.nmax,
                    (const int *)nfft2d,(COMPLEX *)eimagex,
                    (COMPLEX *)eimagey,(COMPLEX *)eimagez,
                    (COMPLEX *)ff0,(COMPLEX *)ffloc,
                    (COMPLEX *)ff,&runtime.plan2f,&runtime.plan2b,
                    (int *)nstop,infostr,&resident_mode,beam_len,infostr_len);
            }
            if(*nstop!=0) rc=1;
            else {
                const int ni=(int)*niso,nm=init.nmax,nb=(int)*nbsphere;
                produitpolavect_(polarisa,&ni,&nm,&nb,
                                  (COMPLEX *)ffloc,(COMPLEX *)ff);
            }
        }
    } else if(rc==0 && (cfg.nrig==8 || cfg.nrig==9)) {
        const int scalar2=cfg.nrig==9;
        if(scalar2) puts("Scalar approximation G=e(ikr)/r");
        else puts("Scalar approximation Uinc.GUinv");
        if(scalar_green_result.ready) {
                const COMPLEX ux=scalar_green_result.ux;
                const COMPLEX uy=scalar_green_result.uy;
                const COMPLEX uz=scalar_green_result.uz;
                const int nb=(int)*nbsphere;
                COMPLEX *xi=(COMPLEX *)a.xi;
                COMPLEX *xr=(COMPLEX *)a.xr;
                if((int)*ninitest==0 || (int)*ninitest==5) {
                    for(int i=0;i<nb;++i) xi[i]=0.0;
                } else if((int)*ninitest==1) {
                    const COMPLEX *f0=(const COMPLEX *)ff0;
                    for(int i=0;i<nb;++i) {
                        const size_t k=(size_t)3*(size_t)i;
                        xi[i]=f0[k]*conj(ux)+f0[k+1]*conj(uy)+f0[k+2]*conj(uz);
                    }
                } else if((int)*ninitest==3) {
                    const int forbidden=memcmp(beam,"antenna",7)==0 ||
                        memcmp(beam,"wavelinearmulti",15)==0 ||
                        memcmp(beam,"gparawavelinear",15)==0 ||
                        memcmp(beam,"gparawavecircular",17)==0;
                    if(forbidden) {
                        static const char msg[]="BPM not possible with this beam";
                        memset(infostr,' ',infostr_len);
                        memcpy(infostr,msg,sizeof msg-1<infostr_len?
                               sizeof msg-1:infostr_len);
                        *nstop=1;
                        rc=1;
                    } else {
                        const int nx=runtime.nx,ny=runtime.ny,nz=runtime.nz;
                        const int nxm=init.nxm,nym=init.nym,nzm=init.nzm;
                        const int nmax=init.nmax,nfft=(int)*a.nfft2d;
                        beampropagation_(xs,ys,zs,aretecube,k0,w0,e0,ss,pp,
                            theta,phi,xgaus,ygaus,zgaus,beam,epsilon,
                            (const int *)niso,(const int *)ndipole,&nx,&ny,&nz,
                            &nxm,&nym,&nzm,&nmax,&nfft,
                            (COMPLEX *)eimagex,(COMPLEX *)eimagey,
                            (COMPLEX *)eimagez,(COMPLEX *)ff0,
                            (COMPLEX *)a.xr,(COMPLEX *)ff,
                            &runtime.plan2f,&runtime.plan2b,(int *)nstop,infostr,
                            &cfg.nrig,beam_len,infostr_len);
                        if(*nstop!=0) rc=1;
                        else {
                            const COMPLEX *bpm=(const COMPLEX *)a.xr;
                            for(int i=0;i<nb;++i) {
                                const size_t k=(size_t)3*(size_t)i;
                                xi[i]=bpm[k]*conj(ux)+bpm[k+1]*conj(uy)+
                                      bpm[k+2]*conj(uz);
                            }
                        }
                    }
                } else {
                    const COMPLEX *f0=(const COMPLEX *)ff0;
                    const COMPLEX *alpha=(const COMPLEX *)polarisa;
                    COMPLEX *fftmp=(COMPLEX *)ff;
                    for(int i=0;i<nb;++i) {
                        const size_t k=(size_t)3*(size_t)i;
                        xr[i]=0.0;
                        fftmp[i]=f0[k]*conj(ux)+f0[k+1]*conj(uy)+f0[k+2]*conj(uz);
                        xi[i]=-alpha[i]*fftmp[i];
                    }
                    const int ntotalm=init.ntotalm,ntotal=runtime.ntotal;
                    const int nmax=init.nmax,nxm=init.nxm,nym=init.nym,nzm=init.nzm;
                    const int nx=runtime.nx,ny=runtime.ny,nz=runtime.nz;
                    const int nx2=runtime.nx2,ny2=runtime.ny2,nxy2=runtime.nxy2,nz2=runtime.nz2;
                    COMPLEX *green=scalar2?runtime.vecty:runtime.vectx;
                    COMPLEX *scratch=scalar2?runtime.vectx:runtime.vecty;
                    const long long fft_planf = static_cast<long long>(runtime.planf);
                    const long long fft_planb = static_cast<long long>(runtime.planb);
                    produitfftmatvectoptscalaire_(green,scratch,&ntotalm,&ntotal,
                        &nmax,&nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,
                        xi,xr,&fft_planf,&fft_planb);
                    for(int i=0;i<nb;++i)
                        xi[i]=cabs(fftmp[i])==0.0?0.0:
                            fftmp[i]*cexp(xr[i]/fftmp[i]);
                }
                {
                    const int ntotalm=init.ntotalm,ntotal=runtime.ntotal;
                    const int ldabi=init.ldabi,nlar=(int)*a.nlar,nmax=init.nmax;
                    const int nd=(int)*ndipole,nxm=init.nxm,nym=init.nym,nzm=init.nzm;
                    const int nx=runtime.nx,ny=runtime.ny,nz=runtime.nz;
                    const int nx2=runtime.nx2,ny2=runtime.ny2,nxy2=runtime.nxy2,nz2=runtime.nz2;
                    const int ni=(int)*niso,nlim=(int)*a.nlim;
                    const REAL tol=(REAL)*tolinit;
                    COMPLEX *green=scalar2?runtime.vecty:runtime.vectx;
                    COMPLEX *scratch=scalar2?runtime.vectx:runtime.vecty;
                    const long long fft_planf = static_cast<long long>(runtime.planf);
                    const long long fft_planb = static_cast<long long>(runtime.planb);
                    if(rc==0 && ((int)*ninitest==2 || (int)*nproche!=-1)) {
                        // Dense optimized FFT, but all Krylov vectors, Chan
                        // applications and reductions stay resident on GPU.
                        const int optimized_product=1;
                        // The legacy optimized scalar solver constructs a
                        // projected incident-field initial iterate for mode 5.
                        const int initial_guess=(int)*ninitest==5 ? 1 : (int)*ninitest;
                        const int preconditioner=(int)*nprecon;
                        inverserigscalaire_cuda_driver_(green,scratch,(int *)a.tabdip,
                            (COMPLEX *)a.macroscopicfieldx,
                            (COMPLEX *)a.macroscopicfieldy,&ux,&uy,&uz,
                            &ntotalm,&ntotal,&ldabi,&nlar,&nmax,&nd,&nxm,&nym,&nzm,
                            &nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,&nb,xi,
                            (COMPLEX *)a.xr,(COMPLEX *)a.wrk,
                            (COMPLEX *)a.ff,(COMPLEX *)a.ff0,
                            (COMPLEX *)a.ffloc,(COMPLEX *)a.polarisa,
                            &ni,methodeit,&tol,(REAL *)tol1,(int *)nloop,
                            (int *)ncompte,&nlim,&preconditioner,(int *)nrig,
                            &initial_guess,&optimized_product,(REAL *)k0,
                            (REAL *)aretecube,&fft_planf,&fft_planb,
                            (int *)nstop,infostr,methodeit_len,infostr_len);
                    } else if(rc==0 && (int)*nprecon==0)
                        inverserigscalaire_(green,scratch,(int *)a.tabdip,
                            (COMPLEX *)a.macroscopicfieldx,
                            (COMPLEX *)a.macroscopicfieldy,&ux,&uy,&uz,
                            &ntotalm,&ntotal,&ldabi,&nlar,&nmax,&nd,&nxm,&nym,&nzm,
                            &nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,&nb,xi,(COMPLEX *)a.xr,
                            (COMPLEX *)a.wrk,(COMPLEX *)a.ff,(COMPLEX *)a.ff0,
                            (COMPLEX *)a.ffloc,(COMPLEX *)a.polarisa,&ni,
                            methodeit,&tol,(REAL *)tol1,(int *)nloop,(int *)ncompte,&nlim,
                            &runtime.planf,&runtime.planb,(int *)nstop,infostr,
                            methodeit_len,infostr_len);
                    else if(rc==0) {
                        auto fn=(int)*nprecon==1?inverserigscalairepg_:
                            inverserigscalairepd_;
                        fn(green,scratch,(int *)a.tabdip,
                            (COMPLEX *)a.macroscopicfieldx,
                            (COMPLEX *)a.macroscopicfieldy,&ux,&uy,&uz,
                            &ntotalm,&ntotal,&ldabi,&nlar,&nmax,&nd,&nxm,&nym,&nzm,
                            &nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,&nb,xi,
                            (COMPLEX *)a.xr,(COMPLEX *)a.wrk,
                            (COMPLEX *)a.ff,(COMPLEX *)a.ff0,
                            (COMPLEX *)a.ffloc,(COMPLEX *)a.ffprecon,
                            (COMPLEX *)a.polarisa,&ni,methodeit,&tol,
                            (REAL *)tol1,(int *)nloop,(int *)ncompte,&nlim,
                            (REAL *)k0,(REAL *)aretecube,
                            (COMPLEX *)a.localfieldx,
                            (COMPLEX *)a.sdetnn,&runtime.planf,
                            &runtime.planb,(int *)nrig,(int *)nstop,infostr,
                            methodeit_len,infostr_len);
                    }
                    if(rc==0 && !isfinite((REAL)*tol1)) {
                        static const char msg[]="Non-finite scalar solver residual";
                        memset(infostr,' ',infostr_len);
                        memcpy(infostr,msg,sizeof msg-1<infostr_len?
                               sizeof msg-1:infostr_len);
                        *nstop=1;
                        rc=1;
                    } else if(*nstop!=0 || (REAL)*tol1>=(REAL)*tolinit)
                        rc=1;
                }
                if(rc==0 && init5_two_stage) {
                    /* Fortran init=5 first solves the scalar problem at 1%,
                       then uses its vector field as the initial guess for
                       the requested rigorous vector solve. */
                    const COMPLEX *scalarfield=
                        (const COMPLEX *)a.ffloc;
                    COMPLEX *vector_xi=(COMPLEX *)a.xi;
                    const int n3=3*(int)*nbsphere;
                    for(int i=0;i<n3;++i) vector_xi[i]=scalarfield[i];
                    cfg.nrig=0;
                    cfg.nprecon=init5_saved_precon;
                    init5_two_stage=0;
                    *nrig=(cdmlib_ip)0;
                    *nprecon=(cdmlib_ip)init5_saved_precon;
                    *tolinit=(cdmlib_real)init5_saved_tol;
                    // Re-enter the vector branch: it constructs the vector
                    // Green once, then starts the rigorous vector solver.
                    goto solver_repeated;
                }
        }
    } else if(rc==0 && cfg.nrig>=10 && cfg.nrig<=15) {
        const int forbidden=memcmp(beam,"antenna",7)==0 ||
                            memcmp(beam,"wavelinearmulti",15)==0;
        if(forbidden) {
            static const char msg[]="MLB not possible with this beam";
            memset(infostr,' ',infostr_len);
            memcpy(infostr,msg,sizeof msg-1<infostr_len?sizeof msg-1:infostr_len);
            *nstop=1; rc=1;
        } else {
            const int *ni=(const int *)niso,*nd=(const int *)ndipole;
            const int nx=runtime.nx,ny=runtime.ny,nz=runtime.nz;
            const int nxm=init.nxm,nym=init.nym,nzm=init.nzm;
            const int nmax=init.nmax,nfft2d=cfg.nfft2d;
            COMPLEX *ix=(COMPLEX *)eimagex;
            COMPLEX *iy=(COMPLEX *)eimagey;
            COMPLEX *iz=(COMPLEX *)eimagez;
            COMPLEX *f0=(COMPLEX *)ff0;
            COMPLEX *fl=(COMPLEX *)ffloc;
            COMPLEX *out=(COMPLEX *)ff;
            const long long born_plan2f = static_cast<long long>(runtime.plan2f);
            const long long born_plan2b = static_cast<long long>(runtime.plan2b);
            int *stop=(int *)nstop;
            if(cfg.nrig==10 || cfg.nrig==11 || cfg.nrig==12 || cfg.nrig==13) {
                if(cfg.nrig==10)
                    puts("Multi layer Born scalar");
                else if(cfg.nrig==11)
                    puts("Multi layer Born scalar with reflexion");
                else if(cfg.nrig==12)
                    puts("Multi layer Born scalar inc");
                else
                    puts("Multi layer Born scalar inc with reflexion");
                void (*fn)(const REAL *,const REAL *,const REAL *,
                    const REAL *,const REAL *,const REAL *,
                    const COMPLEX *,const COMPLEX *,
                    const COMPLEX *,const REAL *,const REAL *,
                    const REAL *,const REAL *,const REAL *,const char *,
                    const COMPLEX *,const COMPLEX *,const int *,
                    const int *,const int *,const int *,const int *,const int *,
                    const int *,const int *,const int *,const int *,COMPLEX *,
                    COMPLEX *,COMPLEX *,const COMPLEX *,
                    COMPLEX *,COMPLEX *,const long long *,
                    const long long *,int *,char *,size_t,size_t);
                if(cfg.nrig==10) fn=multilayerbornsca_;
                else if(cfg.nrig==11) fn=multilayerbornscaref_;
                else if(cfg.nrig==12) fn=multilayerbornscainc_;
                else fn=multilayerbornscaincref_;
                fn((const REAL *)a.xs,(const REAL *)a.ys,(const REAL *)a.zs,
                    (const REAL *)aretecube,(const REAL *)k0,(const REAL *)w0,
                    (const COMPLEX *)e0,(const COMPLEX *)ss,
                    (const COMPLEX *)pp,(const REAL *)theta,(const REAL *)phi,
                    (const REAL *)xgaus,(const REAL *)ygaus,(const REAL *)zgaus,
                    beam,(const COMPLEX *)epsilon,(const COMPLEX *)polarisa,
                    ni,nd,&nx,&ny,&nz,&nxm,&nym,&nzm,&nmax,&nfft2d,ix,iy,iz,f0,fl,out,
                    &born_plan2f,&born_plan2b,stop,infostr,beam_len,infostr_len);
            } else {
                if(cfg.nrig==14) puts("Multi layer Born vectorial");
                else puts("Multi layer Born vectorial with reflexion");
                void (*fn)(const REAL *,const REAL *,const REAL *,
                    const REAL *,const REAL *,const REAL *,
                    const COMPLEX *,const COMPLEX *,
                    const COMPLEX *,const REAL *,const REAL *,
                    const REAL *,const REAL *,const REAL *,const char *,
                    const COMPLEX *,const COMPLEX *,const int *,
                    const int *,const int *,const int *,const int *,const int *,
                    const int *,const int *,const int *,const int *,COMPLEX *,
                    COMPLEX *,COMPLEX *,const COMPLEX *,
                    COMPLEX *,COMPLEX *,const long long *,
                    const long long *,int *,char *,size_t,size_t);
                fn=cfg.nrig==14?multilayerbornvect_:multilayerbornvectref_;
                fn((const REAL *)a.xs,(const REAL *)a.ys,(const REAL *)a.zs,
                    (const REAL *)aretecube,(const REAL *)k0,(const REAL *)w0,
                    (const COMPLEX *)e0,(const COMPLEX *)ss,
                    (const COMPLEX *)pp,(const REAL *)theta,(const REAL *)phi,
                    (const REAL *)xgaus,(const REAL *)ygaus,(const REAL *)zgaus,
                    beam,(const COMPLEX *)epsilon,(const COMPLEX *)polarisa,
                    ni,nd,&nx,&ny,&nz,&nxm,&nym,&nzm,&nmax,&nfft2d,ix,iy,iz,f0,fl,out,
                    &born_plan2f,&born_plan2b,stop,infostr,beam_len,infostr_len);
            }
            if(*nstop!=0) rc=1;
            else {
                const int nm=init.nmax,nb=(int)*nbsphere;
                produitpolavect_(polarisa,ni,&nm,&nb,fl,out);
            }
        }
    } else if(rc==0) {
        static const char msg[]="Invalid or unsupported nrig mode";
        memset(infostr,' ',infostr_len);
        memcpy(infostr,msg,sizeof msg-1<infostr_len?sizeof msg-1:infostr_len);
        *nstop=1;
        rc=1;
    }
    } // end solver dispatch scope (Born/Rytov local functions)
cdmlib_after_solver_dispatch:
    // --FIELD SAVE START--
    // Save FF and FFloc in Fortran sequential record format when nlecture=1.
    auto cdmlib_post_field_save_c99 = [&]() {
        /* Match cdmlib.f90's first-run nlecture path: after solving, persist the
           three complex components of FF and FFloc as GNU Fortran sequential
           unformatted records so a later invocation can reload them. */
        if(rc==0 && cfg.nlecture==1 && !reread_local_fields) {
            char filename[65];
            size_t base=a.filereread_len;
            while(base>0 && a.filereread[base-1]==' ') --base;
            if(base==0 || base>61) {
                static const char msg[]="Invalid nlecture output filename";
                memset(infostr,' ',infostr_len);
                memcpy(infostr,msg,sizeof msg-1<infostr_len?sizeof msg-1:infostr_len);
                *nstop=1; rc=1;
            } else {
                memcpy(filename,a.filereread,base);
                memcpy(filename+base,".lf",3);
                filename[base+3]='\0';
                FILE *existing=fopen(filename,"rb");
                if(existing) {
                    fclose(existing);
                    static const char msg[]="nlecture .lf output file already exists";
                    memset(infostr,' ',infostr_len);
                    memcpy(infostr,msg,sizeof msg-1<infostr_len?sizeof msg-1:infostr_len);
                    *nstop=1; rc=1;
                } else {
                    FILE *field=fopen(filename,"wb");
                    if(!field) {
                        static const char msg[]="Could not create nlecture .lf output file";
                        memset(infostr,' ',infostr_len);
                        memcpy(infostr,msg,sizeof msg-1<infostr_len?sizeof msg-1:infostr_len);
                        *nstop=1; rc=1;
                    } else {
                        const uint32_t record_size=(uint32_t)(2u*sizeof(COMPLEX));
                        const int n3=3*(int)*nbsphere;
                        const COMPLEX *ff=(const COMPLEX *)a.ff;
                        const COMPLEX *ffloc=(const COMPLEX *)a.ffloc;
                        for(int i=0;i<n3;++i) {
                            if(fwrite(&record_size,sizeof record_size,1,field)!=1 ||
                               fwrite(&ff[i],sizeof ff[i],1,field)!=1 ||
                               fwrite(&ffloc[i],sizeof ffloc[i],1,field)!=1 ||
                               fwrite(&record_size,sizeof record_size,1,field)!=1) {
                                static const char msg[]="Could not write nlecture .lf output file";
                                memset(infostr,' ',infostr_len);
                                memcpy(infostr,msg,sizeof msg-1<infostr_len?sizeof msg-1:infostr_len);
                                *nstop=1; rc=1;
                                break;
                            }
                        }
                        if(fclose(field)!=0 && rc==0) {
                            static const char msg[]="Could not close nlecture .lf output file";
                            memset(infostr,' ',infostr_len);
                            memcpy(infostr,msg,sizeof msg-1<infostr_len?sizeof msg-1:infostr_len);
                            *nstop=1; rc=1;
                        }
                    }
                }
            }
        }
    };
    cdmlib_post_field_save_c99();
    // --FIELD SAVE STOP--

    // --CONVERGENCE OUTPUT START--
    auto cdmlib_post_convergence_c99 = [&]() {
        if(rc==0 && cfg.nrig==0 && !reread_local_fields) {
            printf(" Tolerance obtained                : %26.16E\n",(REAL)*tol1);
            printf(" Tolerance asked                   : %26.16E\n",(REAL)*tolinit);
            printf(" Number of product Ax needs        : %12d\n",(int)*ncompte);
            printf(" Number of iterations              : %12d\n",(int)*nloop);
        }
    };
    cdmlib_post_convergence_c99();
    // --CONVERGENCE OUTPUT STOP--

    // --PML REMOVAL AND VACUUM REFERENCE START--
    // nabc=1: remove the absorbing layer only AFTER the solver has used it.
    // rerangechamppola_ compacts the physical dipoles and updates nbsphere.
    // Replace the absorbing-medium reference by the vacuum incident wave.
    // The Fresnel transmission phase compensates the PML/reference change.
    // Keep the original FF and FFloc results; do not solve again here.
    auto cdmlib_post_pml_c99 = [&]() {
        if(rc==0 && init.nabc==1) {
            // Follow cdmlib.f90: remove the absorbing border only after solving.
            // The solver must see the complete PML box, whereas the observables
            // use only the physical region and the vacuum reference incident wave.
            rerangechamppola_(ffloc,epsilon,polarisa,niso,aretecube,
                k0,ff,xs,ys,zs,xswf,yswf,zswf,tabdip,
                tabmulti,pml_mask.data(),&runtime.nx,&runtime.ny,
                &runtime.nz,&init.nxm,&init.nym,&init.nzm,ndipole,
                nbsphere,&init.nmax,init.trope,init.polarizability,
                &runtime.sldr,&init.nproche,3,2);
            if(*nbsphere<=0 || *nbsphere>*ndipole) {
                const char msg[]="CUDA MDDA reduction of PML region failed";
                memset(infostr,' ',infostr_len);
                memcpy(infostr,msg,std::min(sizeof msg-1,infostr_len));
                *nstop=1;
                rc=1;
            } else {
                const COMPLEX absorbing_eps0=init.eps0;
                const COMPLEX vacuum_eps0(REAL(1),REAL(0));
                const COMPLEX beam_amplitude=(COMPLEX)*a.e0;
                const REAL k0_value=(REAL)*k0;
                const REAL s_value=(REAL)*ss,p_value=(REAL)*pp;
                const REAL theta_value=(REAL)*theta,phi_value=(REAL)*phi;
                REAL origin=REAL(0),zref=((REAL *)zs)[0];
                COMPLEX ref_ex,ref_ey,ref_ez;
                int stop=0;
                ondeplane_(&origin,&origin,&zref,(REAL *)&k0_value,
                    &absorbing_eps0,&beam_amplitude,(REAL *)&s_value,
                    (REAL *)&p_value,(REAL *)&theta_value,(REAL *)&phi_value,
                    &ref_ex,&ref_ey,&ref_ez,&stop,infostr,infostr_len);
                const COMPLEX phase_field=p_value==REAL(0)?ref_ey:ref_ex;
                const REAL phase=std::arg(phase_field);
                const COMPLEX trans=REAL(2)*std::sqrt(absorbing_eps0)/
                    (vacuum_eps0+std::sqrt(absorbing_eps0));
                const REAL phase_trans=std::arg(trans);
                runtime.I0=std::norm(beam_amplitude);
                init.eps0=vacuum_eps0;
                ondeplane_(&origin,&origin,&zref,(REAL *)&k0_value,
                    &vacuum_eps0,&beam_amplitude,(REAL *)&s_value,
                    (REAL *)&p_value,(REAL *)&theta_value,(REAL *)&phi_value,
                    &ref_ex,&ref_ey,&ref_ez,&stop,infostr,infostr_len);
                const REAL phase_ref=std::arg(p_value==REAL(0)?ref_ey:ref_ex);
                const COMPLEX phase_factor=std::exp(COMPLEX(REAL(0),
                    -(phase_ref-phase-phase_trans)));
                COMPLEX *incident=(COMPLEX *)ff0;
                for(int i=0;i<(int)*nbsphere && stop==0;++i) {
                    REAL x=((REAL *)xs)[i],y=((REAL *)ys)[i],z=((REAL *)zs)[i];
                    ondeplane_(&x,&y,&z,(REAL *)&k0_value,
                        &vacuum_eps0,&beam_amplitude,(REAL *)&s_value,
                        (REAL *)&p_value,(REAL *)&theta_value,(REAL *)&phi_value,
                        &incident[3*i],&incident[3*i+1],&incident[3*i+2],
                        &stop,infostr,infostr_len);
                    for(int component=0;component<3;++component)
                        incident[3*i+component]*=phase_factor;
                }
                if(stop!=0) { *nstop=1; rc=1; }
            }
            // PML compaction overwrites FFloc/FF, and the vacuum reference
            // overwrites FF0 on the host. Invalidate the solver's GPU aliases
            // so the subsequent CUDA cross sections consume the new values.
            clear_device_result_vector_alias(ff0);
            clear_device_result_vector_alias(ff);
            clear_device_result_vector_alias(ffloc);
        }
    };
    cdmlib_post_pml_c99();
    // --PML REMOVAL AND VACUUM REFERENCE STOP--

    // --CROSS SECTIONS START--
    auto cdmlib_post_cross_sections_c99 = [&]() {
        if(rc==0) rc=cdmlib_compute_cross_sections_from_abi_c99(&a,&runtime,
                                                                 (int)*nbsphere);
        if(rc==0 && *nsection==1) {
            printf(" Cext (DDA)  = %26.16E m2\n",(REAL)*cext);
            printf(" Csca (DDA)  = %26.16E m2\n",(REAL)*csca);
            printf(" Cabs = %26.16E m2\n",(REAL)*cabs_out);
        }
    };
    cdmlib_post_cross_sections_c99();
    // --CROSS SECTIONS STOP--

    // --POYNTING START--
    auto cdmlib_post_poynting_c99 = [&]() {
        if(rc==0 && cfg.nsectionsca==1 && cfg.nquickdiffracte==1)
            rc=cdmlib_compute_quick_poynting_from_abi_c99(&a,&init,&cfg,&runtime);
        if(rc==0 && cfg.nsectionsca==1 && cfg.nquickdiffracte==0)
            rc=cdmlib_compute_slow_poynting_from_abi_c99(&a,&init,&cfg,&runtime);
    };
    cdmlib_post_poynting_c99();
    // --POYNTING STOP--

    // --WIDEFIELD AND FIELD MAPPING START--
    auto cdmlib_post_field_mapping_c99 = [&]() {
        if(rc==0 && init.nprochefft>=1) {
            int subunits=0;
            rc=cdmlib_compute_widefield_from_abi_c99(&a,&init,&cfg,&runtime,
                                                      &subunits);
            if(rc==0)
                rc=cdmlib_map_widefield_outputs_c99(&a,&init,&cfg,subunits);
        } else if(rc==0) {
            rc=cdmlib_map_solved_fields_from_abi_c99(&a,&init,&cfg,
                                                      (int)*ndipole);
        }
    };
    cdmlib_post_field_mapping_c99();
    // --WIDEFIELD AND FIELD MAPPING STOP--

    // --GREEN REPEAT START--
    // The final goto remains OUTSIDE the local function to preserve control flow.
    auto cdmlib_post_green_repeat_c99 = [&]() -> bool {
        if(rc==0 && cfg.testgreen>=2 && cfg.testgreen<4) {
            rc=cdmlib_set_greentensor_source_c99(&a,&init,cfg.testgreen-1);
            if(rc==0) {
                ++cfg.testgreen;
                rc=cdmlib_map_incident_fields_c99(&a,&init,&cfg,(int)*ndipole);
            }
            if(rc==0) {
                *a.nloop=0; *a.ncompte=0; *a.tol1=0;
                *a.nstop=0;
                return true;
            }
        }
        return false;
    };
    if(cdmlib_post_green_repeat_c99()) goto solver_repeated;
    // --GREEN REPEAT STOP--

    // --OPTICAL FORCE START--
    auto cdmlib_post_force_c99 = [&]() {
        /* Fortran computes optical forces only when nforce=1; optical
           torque is conditional on both nforce=1 and ntorque=1. */
        if(rc==0 && cfg.nforce==1)
            rc=cdmlib_compute_force_plane_c99(&a,&init,&cfg,&runtime);
    };
    cdmlib_post_force_c99();
    // --OPTICAL FORCE STOP--

    // --ENERGY START--
    auto cdmlib_post_energy_c99 = [&]() {
        if(rc==0 && cfg.nenergie==1)
            rc=cdmlib_compute_energy_from_abi_c99(&a,&init,&cfg,&runtime);
    };
    cdmlib_post_energy_c99();
    // --ENERGY STOP--

    // --MICROSCOPY START--
    auto cdmlib_post_microscopy_c99 = [&]() {
        if(rc==0 && cfg.nlentille==1) {
            if(cfg.ntypemic==1 || cfg.ntypemic==2 || cfg.ntypemic==3 ||
               cfg.ntypemic==5)
                rc=cdmlib_compute_brightfield_microscopy_from_abi_c99(&a,&init,&cfg,
                                                                       &runtime);
            else if(cfg.ntypemic==4)
                rc=cdmlib_compute_darkfield_microscopy_from_abi_c99(&a,&init,&cfg,
                                                                     &runtime);
            else if(cfg.ntypemic==6 || cfg.ntypemic==7)
                rc=cdmlib_compute_confocal_microscopy_from_abi_c99(&a,&init,&cfg,
                                                                    &runtime);
            else if(cfg.ntypemic==0)
                rc=cdmlib_compute_holographic_quicklens_from_abi_c99(&a,&init,&cfg,
                                                                      &runtime);
            else {
                static const char msg[]="C99 microscopy type not connected yet";
                memset(infostr,' ',infostr_len);
                memcpy(infostr,msg,sizeof msg-1<infostr_len?sizeof msg-1:infostr_len);
                *nstop=1;
                rc=1;
            }
        }
    };
    cdmlib_post_microscopy_c99();
    // --MICROSCOPY STOP--

cdmlib_cleanup:
    // --CLEANUP START--
    auto cdmlib_post_cleanup_c99 = [&]() {
        cdmlib_free_fft_workspace_c99(&runtime);
        if(runtime.file_id>=0) {
            int64_t ids[8]={runtime.file_id,runtime.group_idopt,runtime.group_iddip,
                runtime.group_idff,runtime.group_idmic,runtime.group_idof,
                runtime.group_idnf,runtime.group_idver};
            cdmlib_hdf5_close_groups_c99(ids);
        }
        if(rc!=0 && *nstop==0) *nstop=1;
    };
    cdmlib_post_cleanup_c99();
    // --CLEANUP STOP--

}

/* Incorporated cdmlib_config */
#include "cdmlib_config.h"

#include <stdint.h>
#include <math.h>
#include <string.h>
//-------------------------------------------

static int32_t cdmlib_config__wrap_i32(uint32_t bits)
{
    int32_t value;
    memcpy(&value, &bits, sizeof value);
    return value;
}
//-------------------------------------------

static int32_t cdmlib_config__add_i32(int32_t a, int32_t b)
{
    return cdmlib_config__wrap_i32((uint32_t)a + (uint32_t)b);
}
//-------------------------------------------

static int32_t cdmlib_config__mul_i32(int32_t a, int32_t b)
{
    return cdmlib_config__wrap_i32((uint32_t)a * (uint32_t)b);
}
//-------------------------------------------

static int cdmlib_config__write_list_directed_real(FILE *f,REAL value)
{
    char text[64];
    if(snprintf(text,sizeof text,"%.17g",value)<0) return -1;
    if(strchr(text,'.')==NULL && strchr(text,'e')==NULL && strchr(text,'E')==NULL) {
        const size_t n=strlen(text);
        if(n+18>=sizeof text) return -1;
        memcpy(text+n,".0000000000000000",18);
    }
    for(char *p=text;*p;++p) if(*p=='e') *p='E';
    return fprintf(f,"%21s     \r\n",text);
}
//-------------------------------------------

static int cdmlib_config__starts(const char *s,const char *p,size_t n)
{ return memcmp(s,p,n)==0; }
//-------------------------------------------

static void cdmlib_config__message(char dst[64],const char *src)
{
    size_t n=strlen(src);
    if(n>64) n=64;
    memset(dst,' ',64);
    memcpy(dst,src,n);
}
//-------------------------------------------

static int cdmlib_config__fail(int code,const char *text,int *nstop,char infostr[64])
{
    *nstop=code;
    cdmlib_config__message(infostr,text);
    return 1;
}
//-------------------------------------------

// The shared object classifier is defined in geometry_dispatch_CUDA.cpp.
//-------------------------------------------

static int cdmlib_config__is_grid_object(const char *object)
{
    cdmlib_object_kind_c99 kind=cdmlib_classify_object_c99(object);
    return kind==CDMLIB_OBJECT_CUBOID2||kind==CDMLIB_OBJECT_INHOMO_CUBOID2||
           kind==CDMLIB_OBJECT_RANDOM_SPHERE2||kind==CDMLIB_OBJECT_ARBITRARY;
}
//-------------------------------------------

static int cdmlib_config__is_random_object(const char *object)
{
    cdmlib_object_kind_c99 kind=cdmlib_classify_object_c99(object);
    return kind==CDMLIB_OBJECT_RANDOM_SPHERE1||kind==CDMLIB_OBJECT_RANDOM_SPHERE2||
           kind==CDMLIB_OBJECT_RANDOM_SPHERE_MULTI1;
}
//-------------------------------------------

int cdmlib_validate_config_c99(cdmlib_config_c99 *c,int *nstop,char infostr[64])
{
    REAL product;

    *nstop=0;
    c->testgreen=0;
    if(c->nside==1) c->nside=-1;
    if(c->nside==0) c->nside=1;

    if(c->nmat==2 && !c->has_hdf5)
        return cdmlib_config__fail(1,"To use HDF5 you should compile with the option",nstop,infostr);
    if(c->nxm==0||c->nym==0||c->nzm==0)
        return cdmlib_config__fail(-1,"One of the number of subunit is equal to zero",nstop,infostr);
    if(!cdmlib_config__is_grid_object(c->object) &&
       c->nnnr > (c->nxm < c->nym ?
                  (c->nxm < c->nzm ? c->nxm : c->nzm) :
                  (c->nym < c->nzm ? c->nym : c->nzm)))
        return cdmlib_config__fail(1,"nxm nym or nzm smaller than discretization",nstop,infostr);

    if(cdmlib_config__starts(c->beam,"greentensor",11)) {
        c->nmacro=1;
        c->testgreen=1;
        c->nlecture=0;
    }
    if(c->nobjet==0&&c->nlocal==0&&c->nmacro==0&&c->nsection==0&&
       c->nsectionsca==0&&c->nforce==0&&c->nforced==0&&c->ntorque==0&&
       c->ntorqued==0&&c->nlentille==0&&c->nenergie==0)
        return cdmlib_config__fail(1,"No calculation requested!",nstop,infostr);

    if(c->nenergie==1) {
        if(cdmlib_config__starts(c->beam,"antenna",7)||cdmlib_config__starts(c->beam,"arbitrary",9))
            return cdmlib_config__fail(1,"Can not compute emissivity with this beam",nstop,infostr);
        if(fabs(c->theta)>=90.0)
            return cdmlib_config__fail(1,"Illumination on the wrong side with the options",nstop,infostr);
        if(fabs(c->theta)>90.0&&fabs(c->theta)>=45.0)
            return cdmlib_config__fail(1,"Angle too strong: Imprecise calculation",nstop,infostr);
    }

    if(c->nlentille==1) {
        if(cdmlib_config__starts(c->beam,"antenna",7)||cdmlib_config__starts(c->beam,"arbitrary",9))
            return cdmlib_config__fail(1,"Can not compute total field with this beam",nstop,infostr);
        if(c->numaper<=0.0||c->numaper>1.0)
            return cdmlib_config__fail(1,"Problem with numerical aperture!",nstop,infostr);
        if(fabs(c->gross)<1.0)
            return cdmlib_config__fail(1,"Problem with magnification!",nstop,infostr);
        if(fabs(c->theta)>=90.0)
            return cdmlib_config__fail(1,"Illumination on the wrong side with the options",nstop,infostr);
        c->zlens*=1.0e-9*(REAL)c->nside;
    }
    if(c->nsectionsca==1&&fabs(c->theta)>=45.0)
        return cdmlib_config__fail(1,"Angle too strong: Imprecise calculation with FFT",nstop,infostr);
    if(c->nsection==1&&!cdmlib_config__starts(c->beam,"pwavelinear",11)&&
       !cdmlib_config__starts(c->beam,"pwavecircular",13))
        return cdmlib_config__fail(1,"Cross section only with plane wave",nstop,infostr);

    product=(REAL)c->nxm*(REAL)c->nym*(REAL)c->nzm;
    if(3.0*product>2147483647.0)
        return cdmlib_config__fail(1,"Number of subunits too large for integer in 32 bits",nstop,infostr);
    c->nmax=cdmlib_config__mul_i32(cdmlib_config__mul_i32(c->nxm,c->nym),c->nzm);
    c->ntotalm=cdmlib_config__mul_i32(8,c->nmax);
    c->ntotalf=cdmlib_config__mul_i32(cdmlib_config__mul_i32(cdmlib_config__add_i32(c->nxm,1),cdmlib_config__add_i32(c->nym,1)),
                         cdmlib_config__add_i32(c->nzm,1));
    c->ldabi=cdmlib_config__mul_i32(3,c->nmax);
    if(c->nmax<8)
        return cdmlib_config__fail(1,"Check Nxm Nym and Nzm!",nstop,infostr);
    if(!cdmlib_config__is_grid_object(c->object)) {
        if(c->nnnr<2&&!cdmlib_config__starts(c->object,"arbitrary",9))
            return cdmlib_config__fail(1,"There is no discretization!",nstop,infostr);
    } else {
        if(c->nxm<=1) return cdmlib_config__fail(1,"Discretization follow x too small",nstop,infostr);
        if(c->nym<=1) return cdmlib_config__fail(1,"Discretization follow y too small",nstop,infostr);
        if(c->nzm<=1) return cdmlib_config__fail(1,"Discretization follow z too small",nstop,infostr);
    }
    if(c->tolinit<1.0e-12)
        return cdmlib_config__fail(1,"Tolerance for the iterative method too small!",nstop,infostr);
    if(c->tolinit>0.1)
        return cdmlib_config__fail(1,"Tolerance  for the iterative method too large!",nstop,infostr);

    c->lambda*=1.0e-9;
    c->w0*=1.0e-9;
    if(c->w0<=0.0) return cdmlib_config__fail(1,"Waist=0!",nstop,infostr);
    if(c->P0<=0.0) return cdmlib_config__fail(1,"Power=0!",nstop,infostr);

    return 0;
}
//-------------------------------------------

int cdmlib_normalize_nearfield_c99(cdmlib_config_c99 *c,int *nstop,char infostr[64])
{
    *nstop=0;
    c->nproche-=1;
    c->nabc=0;
    if(c->nproche==2) {
        c->nabc=1;
        c->nproche=0;
        if(!(cdmlib_config__starts(c->object,"sphere",6)||cdmlib_config__starts(c->object,"inhomosphere",12)||
             cdmlib_config__starts(c->object,"inhomocuboid1",13)))
            return cdmlib_config__fail(1,"object not done for MDDA",nstop,infostr);
        if(!(cdmlib_config__starts(c->beam,"pwavelinear",11)||cdmlib_config__starts(c->beam,"gparawavelinear",15)||
             cdmlib_config__starts(c->beam,"gparawavecircular",17)))
            return cdmlib_config__fail(1,"Beam not done for MDDA",nstop,infostr);
        if(c->nprecon!=0) return cdmlib_config__fail(1,"No preconditionner with MDDA",nstop,infostr);
        if(c->nforce==1||c->ntorque==1)
            return cdmlib_config__fail(1,"Not recommended to calculate optical forces with MDDA",nstop,infostr);
    }
    if(c->nlecture==1&&c->nproche==-1) c->nproche=0;
    if(c->nenergie==1&&c->nproche==-1) c->nproche=0;
    if(c->nquickdiffracte==1&&c->nproche==-1) c->nproche=0;
    if(c->nquicklens==1&&c->nproche==-1) c->nproche=0;
    if((c->nrig==6||c->nrig==7||c->nrig==10||c->nrig==11||c->nrig==12||c->nrig==13)&&c->nproche==-1)
        c->nproche=0;
    if(cdmlib_config__starts(c->beam,"gfftwave",8)&&c->nproche==-1) c->nproche=0;
    if(cdmlib_config__starts(c->beam,"speckle",7)&&c->nproche==-1) c->nproche=0;
    if(cdmlib_config__starts(c->beam,"demispeckle",11)&&c->nproche==-1) c->nproche=0;
    if(cdmlib_config__starts(c->beam,"confocal",8)&&c->nproche==-1) c->nproche=0;
    if(cdmlib_config__starts(c->beam,"demiconfocal",12)&&c->nproche==-1) c->nproche=0;
    if(c->ninitest==2||c->ninitest==3) c->nproche=0;
    if(c->nquicklens+c->nquickdiffracte>=1&&c->nfft2d<(c->nxm>c->nym?c->nxm:c->nym))
        return cdmlib_config__fail(1,"Increase size of FFT: larger than discretization",nstop,infostr);
    if(cdmlib_config__is_random_object(c->object)&&c->nproche==-1) c->nproche=0;
    if(c->nprecon!=0&&cdmlib_config__starts(c->methodeit,"QMRCLA",6))
        return cdmlib_config__fail(1,"iterative method not implemented with preconditionner",nstop,infostr);
    c->nprochefft=0;
    if(c->nproche>=1) { c->nprochefft=c->nproche; c->nproche=0; }
    return 0;
}
//-------------------------------------------

int cdmlib_validate_object_beam_c99(cdmlib_config_c99 *c,int *nstop,
                                   char infostr[64])
{
    const int allowed = cdmlib_config__starts(c->beam,"pwavelinear",11) ||
        cdmlib_config__starts(c->beam,"pwavecircular",13) || cdmlib_config__starts(c->beam,"antenna",7) ||
        cdmlib_config__starts(c->beam,"gwavelinear",11) || cdmlib_config__starts(c->beam,"gwavecircular",13) ||
        cdmlib_config__starts(c->beam,"gfftwavelinear",14) || cdmlib_config__starts(c->beam,"gfftwavecircular",16) ||
        cdmlib_config__starts(c->beam,"gparawavelinear",15) || cdmlib_config__starts(c->beam,"gparawavecircular",17) ||
        cdmlib_config__starts(c->beam,"wavelinearmulti",15) || cdmlib_config__starts(c->beam,"greentensor",11) ||
        cdmlib_config__starts(c->beam,"speckle",7) || cdmlib_config__starts(c->beam,"demispeckle",11) ||
        cdmlib_config__starts(c->beam,"confocal",8) || cdmlib_config__starts(c->beam,"demiconfocal",12) ||
        cdmlib_config__starts(c->beam,"arbitrary",9);
    if (!allowed) return cdmlib_config__fail(1,"Beam unknown!",nstop,infostr);
    if (!(cdmlib_config__starts(c->beam,"speckle",7) || cdmlib_config__starts(c->beam,"demispeckle",11) ||
          cdmlib_config__starts(c->beam,"confocal",8) || cdmlib_config__starts(c->beam,"demiconfocal",12)))
        c->numaperil=0.0;
    return 0;
}
//-------------------------------------------

int cdmlib_write_options_text_c99(const cdmlib_config_c99 *c,int *nstop,
                                  char infostr[64])
{
    const int values[]={c->nproche,c->nlocal,c->nmacro,c->nsection,
        c->nsectionsca,c->nquickdiffracte,c->nforce,c->nforced,c->ntorque,
        c->ntorqued,c->nlentille,c->nquicklens,c->nphi,c->ntheta+1,
        c->isotropic?0:1,c->nfft2d};
    const REAL reals[]={c->k0,c->numaper};
    const int tail[]={c->nprochefft,c->nobjet,c->ntypemic,c->nside,c->nmat};
    const REAL apertures[]={c->numaperinc,c->numaperinc2,c->kcnax,c->kcnay};
    FILE *f=fopen("inputmatlab.mat","wb");
    if(!f) return cdmlib_config__fail(1,"Cannot write inputmatlab.mat",nstop,infostr);
    for(size_t i=0;i<sizeof values/sizeof values[0];++i)
        if(fprintf(f,"%12d\r\n",values[i])<0) goto write_error;
    for(size_t i=0;i<sizeof reals/sizeof reals[0];++i)
        if(cdmlib_config__write_list_directed_real(f,reals[i])<0) goto write_error;
    for(size_t i=0;i<sizeof tail/sizeof tail[0];++i)
        if(fprintf(f,"%12d\r\n",tail[i])<0) goto write_error;
    for(size_t i=0;i<sizeof apertures/sizeof apertures[0];++i)
        if(cdmlib_config__write_list_directed_real(f,apertures[i])<0) goto write_error;
    if(fprintf(f,"%12d\r\n",c->testgreen)<0 ||
       cdmlib_config__write_list_directed_real(f,c->numaperil)<0) goto write_error;
    if(fclose(f)!=0) return cdmlib_config__fail(1,"Cannot write inputmatlab.mat",nstop,infostr);
    if(c->nmat==2) {
        f=fopen("filenameh5","wb");
        if(!f) return cdmlib_config__fail(1,"Cannot write filenameh5",nstop,infostr);
        size_t n=sizeof c->h5file;
        while(n>0&&c->h5file[n-1]==' ') --n;
        if(n&&fwrite(c->h5file,1,n,f)!=n) { fclose(f); return cdmlib_config__fail(1,"Cannot write filenameh5",nstop,infostr); }
        if(fputc('\r',f)==EOF||fputc('\n',f)==EOF||fclose(f)!=0)
            return cdmlib_config__fail(1,"Cannot write filenameh5",nstop,infostr);
    }
    return 0;
write_error:
    fclose(f);
    return cdmlib_config__fail(1,"Cannot write inputmatlab.mat",nstop,infostr);
}
//-------------------------------------------

int cdmlib_check_object_result_c99(int nx,int ny,int nz,
                                   int nxm,int nym,int nzm,
                                   int nstop,COMPLEX eps,
                                   REAL lambda,REAL aretecube,
                                   REAL *lambda10n,int *nstop_out,
                                   char infostr[64])
{
    if(cdmlib_check_object_dimensions_c99(nx,ny,nz,nxm,nym,nzm,nstop,
                                          nstop_out,infostr)!=0) return 1;
    return cdmlib_check_meshsize_c99(eps,lambda,aretecube,lambda10n,
                                     nstop_out,infostr);
}
//-------------------------------------------

int cdmlib_check_object_dimensions_c99(int nx,int ny,int nz,
                                       int nxm,int nym,int nzm,
                                       int nstop,int *nstop_out,
                                       char infostr[64])
{
    *nstop_out=nstop;
    if(nstop==1) return 1;
    if(nx>nxm||ny>nym||nz>nzm)
        return cdmlib_config__fail(1,"Dimension Problem of the Box : Box too small!",nstop_out,infostr);
    if(nx<=1||ny<=1||nz<=1)
        return cdmlib_config__fail(1,"Dimension Problem nx ny nz too small!",nstop_out,infostr);
    if(nstop==-1) {
        cdmlib_config__message(infostr,"Calculation cancelled after object created!");
        return 1;
    }
    return 0;
}
//-------------------------------------------

int cdmlib_check_meshsize_c99(COMPLEX eps,REAL lambda,
                              REAL aretecube,REAL *lambda10n,
                              int *nstop,char infostr[64])
{
    *lambda10n=lambda/10.0/cabs(csqrt(eps));
    if(aretecube>2.5* *lambda10n) {
        fprintf(stderr,
                "[IF-DDA] mesh check failed: mesh=%g lambda=%g lambda10n=%g eps=(%g,%g)\n",
                (double)aretecube, (double)lambda, (double)*lambda10n,
                (double)std::real(eps), (double)std::imag(eps));
        return cdmlib_config__fail(1,"Meshsize larger than lambda/4!",nstop,infostr);
    }
    return 0;
}
//-------------------------------------------

static int cdmlib_config__write_roundtrip_r8(FILE *f,REAL value)
{
    char text[128];
    const int n=snprintf(text,sizeof text,"%.17g",value);
    if(n<0||(size_t)n>=sizeof text) return -1;
    for(char *p=text;*p;++p) if(*p=='e') *p='E';
    return fputs(text,f);
}
//-------------------------------------------

int cdmlib_write_epsilon_ascii_c99(int ndipole,const int *tabdip,
                                   int nmax,int niso,int isotropic,
                                   const COMPLEX *epsilon,
                                   int *nstop,char infostr[64])
{
    FILE *f=fopen("epsilon.mat","wb");
    if(!f) return cdmlib_config__fail(1,"Cannot write epsilon.mat",nstop,infostr);
    const int ncomponent=isotropic?1:9;
    for(int i=0;i<ndipole;++i) {
        const int k=tabdip[i];
        for(int col=0;col<ncomponent;++col) {
            COMPLEX value;
            if(k!=0) {
                const int row=isotropic?0:col/3;
                const int column=isotropic?0:col%3;
                value=epsilon[(size_t)(k-1)+(size_t)nmax*((size_t)row+(size_t)niso*(size_t)column)];
            } else {
                const int diagonal=isotropic||(col%3==col/3);
                value=diagonal?1.0+0.0*I:0.0+0.0*I;
            }
            if(cdmlib_config__write_roundtrip_r8(f,creal(value))<0||
               fputs("        ",f)==EOF||
               cdmlib_config__write_roundtrip_r8(f,cimag(value))<0||
               fputs("     \r\n",f)==EOF) {
                fclose(f);
                return cdmlib_config__fail(1,"Cannot write epsilon.mat",nstop,infostr);
            }
        }
    }
    if(fclose(f)!=0) return cdmlib_config__fail(1,"Cannot write epsilon.mat",nstop,infostr);
    return 0;
}


//-------------------------------------------

// Shared geometry selector and all CUDA geometry routing live in geometry_dispatch_CUDA.cpp.



/* Incorporated cdmlib_initphase */
#include "cdmlib_initphase.h"

#include <stdint.h>
#include <string.h>
//-------------------------------------------

static int32_t cdmlib_initphase__wrap_i32(uint32_t bits)
{
    int32_t value;
    memcpy(&value, &bits, sizeof value);
    return value;
}
//-------------------------------------------

static int32_t cdmlib_initphase__mul_i32(int32_t a, int32_t b)
{
    return cdmlib_initphase__wrap_i32((uint32_t)a * (uint32_t)b);
}
//-------------------------------------------

static int cdmlib_initphase__starts(const char value[64], const char *prefix, size_t n)
{
    return memcmp(value, prefix, n) == 0;
}
//-------------------------------------------

static int cdmlib_initphase__fail(cdmlib_initphase_c99 *s, const char *message)
{
    size_t n = strlen(message);
    if (n > sizeof s->infostr) n = sizeof s->infostr;
    memset(s->infostr, ' ', sizeof s->infostr);
    memcpy(s->infostr, message, n);
    s->nstop = 1;
    return 1;
}
//-------------------------------------------

int cdmlib_initialize_phase_c99(cdmlib_initphase_c99 *s)
{
    const int grid = cdmlib_initphase__starts(s->object, "cuboid2", 7) ||
                     cdmlib_initphase__starts(s->object, "inhomocuboid2", 13) ||
                     cdmlib_initphase__starts(s->object, "randomsphere2", 13) ||
                     cdmlib_initphase__starts(s->object, "arbitrary", 9);
    REAL product;

    s->nstop = 0;
    s->nmax = cdmlib_initphase__mul_i32(cdmlib_initphase__mul_i32(s->nxm, s->nym), s->nzm);
    s->ntotalm = cdmlib_initphase__mul_i32(cdmlib_initphase__mul_i32(cdmlib_initphase__mul_i32(8, s->nxm), s->nym), s->nzm);
    s->ntotalf = cdmlib_initphase__mul_i32(cdmlib_initphase__mul_i32(s->nxm + 1, s->nym + 1), s->nzm + 1);

    product = 3.0 * (REAL)s->nxm * (REAL)s->nym * (REAL)s->nzm;
    if (product > 2147483647.0)
        return cdmlib_initphase__fail(s, "Number of subunits too large for integer in 32 bits");

    s->ldabi = cdmlib_initphase__mul_i32(3, s->nmax);
    s->xg = s->xgmulti[0];
    s->yg = s->ygmulti[0];
    s->zg = s->zgmulti[0];
    s->rayon = s->rayonmulti[0];
    s->eps = s->epsmulti[0];
    memcpy(s->epsani, s->epsanimulti, sizeof s->epsani);

    if (s->nmax < 8) return cdmlib_initphase__fail(s, "Check Nxm Nym and Nzm!");
    if (!grid) {
        if (s->nnnr < 2)
            return cdmlib_initphase__fail(s, "There is no discretization!");
    } else {
        if (s->nxm <= 1) return cdmlib_initphase__fail(s, "Discretization follow x too small");
        if (s->nym <= 1) return cdmlib_initphase__fail(s, "Discretization follow y too small");
        if (s->nzm <= 1) return cdmlib_initphase__fail(s, "Discretization follow z too small");
    }
    if (s->tolinit < 1.0e-12)
        return cdmlib_initphase__fail(s, "Tolerance for the iterative method too small!");
    if (s->tolinit > 0.1)
        return cdmlib_initphase__fail(s, "Tolerance  for the iterative method too large!");

    s->eps0 = 1.0 + 0.0 * I;
    memset(s->infostr, ' ', sizeof s->infostr);
    memcpy(s->infostr, "STARTED!", 8);
    s->lambda = s->lambda_nm * 1.0e-9;
    s->w0 = s->w0_nm * 1.0e-9;
    if (s->w0 <= 0.0) return cdmlib_initphase__fail(s, "Waist=0!");
    if (s->power <= 0.0) return cdmlib_initphase__fail(s, "Power=0!");
    return 0;
}


/* Incorporated cdmlib_materialphase */
#include "cdmlib_initphase.h"

#include <math.h>
#include <string.h>


//-------------------------------------------

static int cdmlib_materialphase__starts(const char *value, const char *prefix, size_t n)
{
    return memcmp(value, prefix, n) == 0;
}
//-------------------------------------------

static void cdmlib_materialphase__fail(cdmlib_initphase_c99 *s, const char *message)
{
    size_t n = strlen(message);
    if (n > sizeof s->infostr) n = sizeof s->infostr;
    memset(s->infostr, ' ', sizeof s->infostr);
    memcpy(s->infostr, message, n);
    s->nstop = 1;
}
//-------------------------------------------

int cdmlib_validate_material_phase_c99(cdmlib_initphase_c99 *s)
{


    if (cdmlib_materialphase__starts(s->object, "arbitrary", 9)) return 2;

    for (int k = 0; k < s->numberobjet; ++k) {
        char materiau[64];
        memcpy(materiau, s->materiaumulti + 64 * (size_t)k, sizeof materiau);
        if (materiau[0] != 'x' || materiau[1] != 'x') {
            REAL epr = 0.0, epi = 0.0;
            interpdielec_(&s->lambda, materiau, &epr, &epi,
                          s->infostr, &s->nstop, 64, 64);
            if (s->nstop == 1) return 1;
            s->epsmulti[k] = epr + I * epi;
        }

        if (memcmp(s->trope, "iso", 3) == 0) {
            if (creal(s->epsmulti[k]) == 1.0 && cimag(s->epsmulti[k]) == 0.0)
                cdmlib_materialphase__fail(s, "Object is in vacuum");
            else if (cabs(s->epsmulti[k]) == 0.0)
                cdmlib_materialphase__fail(s, "No relative permittivity given");
        } else {
            REAL sum = 0.0;
            for (int i = 0; i < 9; ++i) sum += cabs(s->epsani[i]);
            if (sum == 0.0) cdmlib_materialphase__fail(s, "No relative permittivity given");
            if (memcmp(s->polarizability, "RR", 2) != 0 &&
                memcmp(s->polarizability, "CM", 2) != 0)
                cdmlib_materialphase__fail(s, "Polarizability should be CM or RR with anisotropy");
        }
    }

    if (s->nstop == 1) return 1;
    s->eps = s->epsmulti[0];
    memcpy(s->materiau, s->materiaumulti, sizeof s->materiau);
    return 0;
}


/* Incorporated cdmlib_preobjectphase */
#include "cdmlib_initphase.h"

#include <math.h>
#include <string.h>
//-------------------------------------------

int cdmlib_prepare_preobject_phase_c99(cdmlib_initphase_c99 *s,
                                       cdmlib_config_c99 *c,
                                       REAL *k0, REAL *k03)
{
    int nstop = 0;
    int material_status = cdmlib_initialize_phase_c99(s);
    if (material_status != 0) return material_status;

    material_status = cdmlib_validate_material_phase_c99(s);
    if (material_status == 1) return 1;
    /* Status 2 is the original arbitrary-object jump past material checks. */

    *k0 = 2.0 * 3.14159265358979323846264338327950288 / s->lambda;
    *k03 = *k0 * *k0 * *k0;

    memcpy(c->object, s->object, sizeof c->object);
    c->lambda = s->lambda;
    c->w0 = s->w0;
    c->P0 = s->power;
    c->nproche = s->nproche;
    if (cdmlib_normalize_nearfield_c99(c, &nstop, s->infostr)) {
        s->nproche = c->nproche;
        s->nprochefft = c->nprochefft;
        s->nabc = c->nabc;
        s->nstop = nstop;
        return 1;
    }

    s->nproche = c->nproche;
    s->nprochefft = c->nprochefft;
    s->nabc = c->nabc;
    if (s->nabc == 1) {
        // Fortran cdmlib.f90 sets na=1 for PML/MDDA object construction.
        s->nproche = 1;
        c->nproche = 1;
        s->eps0 = 1.0 + I * s->eps0i;
    }
    return 0;
}


/* Incorporated cdmlib_objectphase */
#include "cdmlib_initphase.h"

#include <complex.h>
#include <stdio.h>
#include <string.h>
//-------------------------------------------

int cdmlib_construct_object_phase_c99(cdmlib_initphase_c99 *s,
                                      cdmlib_config_c99 *c,
                                      cdmlib_object_phase_c99 *g,
                                      REAL *k0, REAL *k03,
                                      REAL *lambda10n)
{
    memcpy(c->object, s->object, sizeof c->object);
    c->lambda = s->lambda_nm;
    c->w0 = s->w0_nm;
    c->P0 = s->power;
    c->tolinit = s->tolinit;
    c->isotropic = memcmp(s->trope,"iso",3)==0;
    c->nxm = s->nxm;
    c->nym = s->nym;
    c->nzm = s->nzm;
    c->nnnr = s->nnnr;
    c->nproche = s->nproche;

    if (cdmlib_validate_config_c99(c, &s->nstop, s->infostr) != 0)
        return 1;

    if (cdmlib_prepare_preobject_phase_c99(s, c, k0, k03) != 0)
        return 1;
    c->k0 = *k0;

    g->object64 = s->object;
    g->trope3 = s->trope;
    g->polarizability2 = s->polarizability;
    g->eps = &s->eps;
    g->epsani = s->epsani;
    g->epsmulti = s->epsmulti;
    g->epsanimulti = s->epsanimulti;
    g->numberobjet = &s->numberobjet;
    g->xgmulti = (REAL *)s->xgmulti;
    g->ygmulti = (REAL *)s->ygmulti;
    g->zgmulti = (REAL *)s->zgmulti;
    g->rayonmulti = (REAL *)s->rayonmulti;
    g->eps0 = &s->eps0;
    g->k0 = k0;
    g->lambda = &s->lambda;
    g->nnnr = &s->nnnr;
    g->nmax = &s->nmax;
    g->nxm = &s->nxm;
    g->nym = &s->nym;
    g->nzm = &s->nzm;
    g->nproche = &s->nproche;
    g->nstop = &s->nstop;
    g->infostr = s->infostr;

    if (!cdmlib_call_object_geometry_c99(g)) {
        memset(s->infostr, ' ', sizeof s->infostr);
        memcpy(s->infostr, "Unsupported object geometry", 27);
        s->nstop = 1;
        return 1;
    }
    if (cdmlib_check_object_dimensions_c99(*g->nx, *g->ny, *g->nz,
                                           s->nxm, s->nym, s->nzm,
                                           s->nstop, &s->nstop,
                                           s->infostr) != 0)
        return 1;
    if (c->nmat == 0 &&
        cdmlib_write_epsilon_ascii_c99(*g->ndipole,g->tabdip,*g->nmax,
            *g->niso,c->isotropic,g->epsilon,&s->nstop,s->infostr)!=0)
        return 1;
    if (cdmlib_check_meshsize_c99(s->eps,s->lambda,*g->aretecube,
                                  lambda10n,&s->nstop,s->infostr)!=0)
        return 1;
    if(cdmlib_validate_object_beam_c99(c, &s->nstop, s->infostr)!=0)
        return 1;
    return cdmlib_write_options_text_c99(c,&s->nstop,s->infostr);
}


/* Incorporated cdmlib_fftphase */
#include "cdmlib_fftphase.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


//-------------------------------------------

static int cdmlib_fftphase__fail(int *nstop,char infostr[64],const char *message)
{
    size_t n=strlen(message);
    if(n>64) n=64;
    memset(infostr,' ',64);
    memcpy(infostr,message,n);
    *nstop=1;
    return 1;
}
//-------------------------------------------

static int cdmlib_fftphase__padded_size(int n,int nabs)
{
    int result;
    if(nabs>=n) return 2*n;
    result=n+nabs;
    factorto235_(&result);
    if(result>2*n) result=2*n;
    return result;
}
//-------------------------------------------

int cdmlib_prepare_fft_grid_c99(int nx,int ny,int nz,int nbsphere,
                                int nabc,int nlentille,int nforce,
                                int ntorque,COMPLEX eps0,
                                REAL k0,REAL aretecube,
                                cdmlib_fftgrid_c99 *grid,
                                int *nstop,char infostr[64])
{
    REAL product;
    *nstop=0;
    grid->nabs=0;
    if(nabc==1&&nlentille==0&&nforce==0&&ntorque==0) {
        const REAL indiceimag=cimag(csqrt(eps0));
        const REAL raw=(-log(0.001)/k0/indiceimag)/aretecube;
        if(!isfinite(raw)||raw>(REAL)INT32_MAX||raw<(REAL)INT32_MIN)
            return cdmlib_fftphase__fail(nstop,infostr,"FFT padding size is outside 32-bit range");
        grid->nabs=(int)raw;
        grid->nx2=cdmlib_fftphase__padded_size(nx,grid->nabs);
        grid->ny2=cdmlib_fftphase__padded_size(ny,grid->nabs);
        grid->nz2=cdmlib_fftphase__padded_size(nz,grid->nabs);
    } else {
        grid->nx2=2*nx;
        grid->ny2=2*ny;
        grid->nz2=2*nz;
    }
    product=(REAL)grid->nx2*(REAL)grid->ny2*(REAL)grid->nz2;
    if(product>2147483647.0)
        return cdmlib_fftphase__fail(nstop,infostr,"Number of subunits too large for integer in 32 bits");
    grid->ntotal=grid->nx2*grid->ny2*grid->nz2;
    grid->nbsphere3=3*nbsphere;
    grid->nxy2=grid->nx2*grid->ny2;
    memset(infostr,' ',64);
    return 0;
}


/* Incorporated cdmlib_incidentphase */
#include "cdmlib_incidentphase.h"

#include <math.h>
#include <stddef.h>
#include <string.h>














//-------------------------------------------

static REAL cdmlib_incidentphase__abs_square(COMPLEX value)
{
    return pow(cabs(value),2.0);
}
//-------------------------------------------

// CPU-only geometry descriptor: the five regular-grid CUDA illuminations
// validate and consume it inside their specialized execution entry points.
// It preserves the arbitrary/irregular-coordinate fallback.
static IfddaIncidentRegularGridSpec cdmlib_incident_grid_spec(
    const cdmlib_incidentphase_c99* s) {
    return {s->tabdip, &s->nx, &s->ny, &s->nz,
            &s->ndipole, &s->aretecube};
}

int cdmlib_compute_plane_linear_incident_c99(cdmlib_incidentphase_c99 *s)
{
    s->nstop=0;
    if(s->nenergie==1)
        s->waist=(REAL)s->nfft2d*s->aretecube/sqrt(3.141592653589793238462643383279502884);
    if(s->ntypemic==0&&s->nlentille==1)
        calculangle_(&s->theta,&s->phi,&s->k0,&s->aretecube,&s->nfft2d);
    irradiance_(&s->power,&s->waist,&s->E0,&s->irradiance);
    s->I0=cabs(s->E0)*cabs(s->E0);
    s->norm=0.0;
    s->tmp=0.0;
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    {
        const int circular=0;
        int status=1;
        const auto grid=cdmlib_incident_grid_spec(s);
        ifdda_plane_wave_field_cuda_execute(s->xs,s->ys,s->zs,
            &s->nbsphere,&s->k0,&s->eps0,&s->E0,&s->ss,&s->pp,
            &s->theta,&s->phi,&circular,s->FF0,nullptr,&status,
            nullptr,nullptr,nullptr,nullptr,&grid);
        if(status!=0) {
            memcpy(s->infostr,"CUDA plane-wave illumination failed",
                   sizeof "CUDA plane-wave illumination failed"-1);
            s->nstop=1;
            return 1;
        }
    }
#endif
    for(int i=0;i<s->nbsphere;++i) {
        COMPLEX *field=s->FF0+3*(size_t)i;
#ifndef IFDDA_CUDA_INVERSE_DRIVER
        ondeplane_((REAL *)&s->xs[i],(REAL *)&s->ys[i],(REAL *)&s->zs[i],
                   &s->k0,&s->eps0,&s->E0,&s->ss,&s->pp,&s->theta,&s->phi,
                   &field[0],&field[1],&field[2],&s->nstop,s->infostr,64);
#endif
        s->norm=s->norm+cdmlib_incidentphase__abs_square(field[0]);
        s->norm=s->norm+cdmlib_incidentphase__abs_square(field[1]);
        s->norm=s->norm+cdmlib_incidentphase__abs_square(field[2]);
        if(s->tabmulti[i]==-1)
            field[0]=field[1]=field[2]=0.0+0.0*I;
        s->tmp=s->tmp+cdmlib_incidentphase__abs_square(field[0]);
        s->tmp=s->tmp+cdmlib_incidentphase__abs_square(field[1]);
        s->tmp=s->tmp+cdmlib_incidentphase__abs_square(field[2]);
    }
    return 0;
}
//-------------------------------------------

int cdmlib_compute_plane_circular_incident_c99(cdmlib_incidentphase_c99 *s)
{
    s->nstop=0;
    if(s->nenergie==1)
        s->waist=(REAL)s->nfft2d*s->aretecube/sqrt(3.141592653589793238462643383279502884);
    irradiance_(&s->power,&s->waist,&s->E0,&s->irradiance);
    s->I0=cabs(s->E0)*cabs(s->E0);
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    {
        const int circular=1;
        int status=1;
        const auto grid=cdmlib_incident_grid_spec(s);
        ifdda_plane_wave_field_cuda_execute(s->xs,s->ys,s->zs,
            &s->nbsphere,&s->k0,&s->eps0,&s->E0,&s->ss,&s->pp,
            &s->theta,&s->phi,&circular,s->FF0,nullptr,&status,
            nullptr,nullptr,nullptr,nullptr,&grid);
        if(status!=0) {
            memcpy(s->infostr,"CUDA circular plane-wave illumination failed",
                   sizeof "CUDA circular plane-wave illumination failed"-1);
            s->nstop=1;
            return 1;
        }
    }
#else
    for(int i=0;i<s->nbsphere;++i) {
        COMPLEX *field=s->FF0+3*(size_t)i;
        ondecirce_(&s->xs[i],&s->ys[i],&s->zs[i],&s->k0,&s->E0,
                   &s->ss,&s->theta,&s->phi,&field[0],&field[1],&field[2]);
    }
#endif
    return 0;
}
//-------------------------------------------

int cdmlib_compute_plane_multi_incident_c99(cdmlib_incidentphase_c99 *s)
{
    s->nstop=0;
    if(s->nenergie==1)
        s->waist=(REAL)s->nfft2d*s->aretecube/sqrt(3.141592653589793238462643383279502884);
    irradiance_(&s->power,&s->waist,&s->E0,&s->irradiance);
    s->I0=cabs(s->E0)*cabs(s->E0);
    s->tmp=0.0;
    for(int i=0;i<s->nbinc;++i) {
        s->tmp=s->tmp+pow(cabs(s->E0m[i]),2.0);
        if(cabs(s->E0m[i])==0.0) {
            memset(s->infostr,' ',sizeof s->infostr);
            memcpy(s->infostr,"Magnitude equal to zero in incident beam",
                   sizeof "Magnitude equal to zero in incident beam"-1);
            s->nstop=1;
            return 1;
        }
    }
    s->tmp=sqrt(pow(cabs(s->E0),2.0)/s->tmp);
    for(int i=0;i<s->nbinc;++i) s->E0m[i]=s->E0m[i]*s->tmp;
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    {
        int status=1;
        const auto grid=cdmlib_incident_grid_spec(s);
        ifdda_multi_plane_wave_field_cuda_execute(s->xs,s->ys,s->zs,
            &s->nbsphere,&s->k0,&s->E0,s->E0m,s->ssm,s->ppm,
            s->thetam,s->phim,&s->nbinc,s->FF0,&status,false,nullptr,&grid);
        if(status!=0) {
            memcpy(s->infostr,"CUDA multi-plane-wave illumination failed",
                   sizeof "CUDA multi-plane-wave illumination failed"-1);
            s->nstop=1;
            return 1;
        }
    }
#endif
    for(int i=0;i<s->nbsphere;++i) {
        COMPLEX *field=s->FF0+3*(size_t)i;
#ifndef IFDDA_CUDA_INVERSE_DRIVER
        ondeplanemulti_((REAL *)&s->xs[i],(REAL *)&s->ys[i],(REAL *)&s->zs[i],
            &s->k0,s->E0m,s->ssm,s->ppm,s->thetam,s->phim,&s->nbinc,
            &field[0],&field[1],&field[2],&s->nstop,s->infostr,64);
#endif
    }
    return 0;
}
//-------------------------------------------

int cdmlib_compute_antenna_incident_c99(cdmlib_incidentphase_c99 *s)
{
    donothing(3);

    const REAL c=299792458.0;
    const REAL quatpieps0=1.0/(c*c*1.0e-7);
    const REAL k02=s->k0*s->k0;
    const REAL k04=k02*k02;
    s->E0=sqrt(3.0*quatpieps0/k04*s->power)/quatpieps0+0.0*I;
    s->antenna_x=s->antenna_x_nm*1.0e-9;
    s->antenna_y=s->antenna_y_nm*1.0e-9;
    s->antenna_z=s->antenna_z_nm*1.0e-9;
    dipoleinside_(&s->antenna_x,&s->antenna_y,&s->antenna_z,
                  s->xs,s->ys,s->zs,&s->aretecube,&s->nmax,&s->nbsphere);
    s->nstop=0;
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    COMPLEX at_object[3]{};
    int status=1;
    const auto grid=cdmlib_incident_grid_spec(s);
    ifdda_antenna_field_cuda_execute(s->xs,s->ys,s->zs,&s->nbsphere,
        &s->antenna_x,&s->antenna_y,&s->antenna_z,&s->aretecube,
        &s->k0,&s->E0,&s->theta,&s->phi,&s->xg,&s->yg,&s->zg,
        s->FF0,at_object,&status,false,nullptr,&grid);
    if(status!=0) {
        memcpy(s->infostr,"CUDA antenna illumination failed",
               sizeof "CUDA antenna illumination failed"-1);
        s->nstop=1;
        return 1;
    }
#else
    for(int i=0;i<s->nbsphere;++i) {
        COMPLEX *field=s->FF0+3*(size_t)i;
        dipoleinc_(&s->antenna_x,&s->antenna_y,&s->antenna_z,
            &s->theta,&s->phi,&s->xs[i],&s->ys[i],&s->zs[i],
            &s->aretecube,&s->k0,&s->E0,&field[0],&field[1],&field[2],
            &s->nstop,s->infostr,64);
    }
    COMPLEX at_object[3];
    dipoleinc_(&s->antenna_x,&s->antenna_y,&s->antenna_z,&s->theta,&s->phi,
        &s->xg,&s->yg,&s->zg,&s->aretecube,&s->k0,&s->E0,
        &at_object[0],&at_object[1],&at_object[2],&s->nstop,s->infostr,64);
#endif
    s->I0=pow(cabs(at_object[0]),2.0);
    s->I0=s->I0+pow(cabs(at_object[1]),2.0);
    s->I0=s->I0+pow(cabs(at_object[2]),2.0);
    return 0;
}
//-------------------------------------------

int cdmlib_compute_gaussian_para_linear_incident_c99(cdmlib_incidentphase_c99 *s)
{
    gaussianpuissancepara_(&s->waist,&s->power,&s->irradiance,&s->E0);
    s->xgaus=s->xgaus_nm*1.0e-9;
    s->ygaus=s->ygaus_nm*1.0e-9;
    s->zgaus=s->zgaus_nm*1.0e-9;
    s->I0=pow(cabs(s->E0),2.0);
    s->nstop=0;
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    {
        const int circular=0;
        int status=1;
        const auto grid=cdmlib_incident_grid_spec(s);
        ifdda_paraxial_gaussian_field_cuda_execute(
            s->xs,s->ys,s->zs,&s->nbsphere,&s->E0,&s->eps0,&s->ss,&s->pp,
            &s->theta,&s->phi,&s->xgaus,&s->ygaus,&s->zgaus,&s->waist,
            &s->k0,&circular,s->FF0,&status,false,nullptr,&grid);
        if(status!=0) {
            memcpy(s->infostr,"CUDA paraxial Gaussian illumination failed",
                   sizeof "CUDA paraxial Gaussian illumination failed"-1);
            s->nstop=1;
            return 1;
        }
    }
#else
    for(int i=0;i<s->nbsphere;++i) {
        COMPLEX *field=s->FF0+3*(size_t)i;
        gaussianparalinear_(&s->xs[i],&s->ys[i],&s->zs[i],
            &s->xgaus,&s->ygaus,&s->zgaus,&s->theta,&s->phi,
            &s->waist,&s->k0,&s->ss,&s->pp,&s->eps0,&s->E0,
            &field[0],&field[1],&field[2],&s->nstop,s->infostr,64);
    }
#endif
    return s->nstop==0?0:1;
}
//-------------------------------------------

int cdmlib_compute_gaussian_para_circular_incident_c99(cdmlib_incidentphase_c99 *s)
{
    s->xgaus=s->xgaus_nm*1.0e-9;
    s->ygaus=s->ygaus_nm*1.0e-9;
    s->zgaus=s->zgaus_nm*1.0e-9;
    gaussianpuissancepara_(&s->waist,&s->power,&s->irradiance,&s->E0);
    s->I0=pow(cabs(s->E0),2.0);
    s->nstop=0;
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    {
        const int circular=1;
        int status=1;
        const auto grid=cdmlib_incident_grid_spec(s);
        ifdda_paraxial_gaussian_field_cuda_execute(
            s->xs,s->ys,s->zs,&s->nbsphere,&s->E0,&s->eps0,&s->ss,&s->pp,
            &s->theta,&s->phi,&s->xgaus,&s->ygaus,&s->zgaus,&s->waist,
            &s->k0,&circular,s->FF0,&status,false,nullptr,&grid);
        if(status!=0) {
            memcpy(s->infostr,"CUDA circular paraxial Gaussian illumination failed",
                   sizeof "CUDA circular paraxial Gaussian illumination failed"-1);
            s->nstop=1;
            return 1;
        }
    }
#else
    for(int i=0;i<s->nbsphere;++i) {
        COMPLEX *field=s->FF0+3*(size_t)i;
        gaussianparacirc_(&s->xs[i],&s->ys[i],&s->zs[i],
            &s->xgaus,&s->ygaus,&s->zgaus,&s->theta,&s->phi,
            &s->waist,&s->k0,&s->ss,&s->eps0,&s->E0,
            &field[0],&field[1],&field[2],&s->nstop,s->infostr,64);
    }
#endif
    return s->nstop==0?0:1;
}
//-------------------------------------------

int cdmlib_compute_gaussian_linear_incident_c99(cdmlib_incidentphase_c99 *s)
{
    const int nloin=0;
    s->xgaus=s->xgaus_nm*1.0e-9;
    s->ygaus=s->ygaus_nm*1.0e-9;
    s->zgaus=s->zgaus_nm*1.0e-9;
    gaussianpuissance_(&s->power,&s->irradiance,&s->waist,&s->k0,&s->E0);
    s->I0=pow(cabs(s->E0),2.0);
    s->nstop=0;
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    {
        const int circular=0;
        const int nloin=0;
        int status=1;
        const auto grid=cdmlib_incident_grid_spec(s);
        ifdda_rigorous_gaussian_field_cuda_execute(
            s->xs,s->ys,s->zs,&s->nbsphere,&s->xgaus,&s->ygaus,&s->zgaus,
            &s->theta,&s->phi,&s->waist,&s->k0,&s->ss,&s->pp,&s->E0,
            &s->tolinit,&circular,&nloin,s->FF0,&status,false,nullptr,&grid);
        if(status!=0) {
            memcpy(s->infostr,"CUDA rigorous Gaussian illumination failed",
                   sizeof "CUDA rigorous Gaussian illumination failed"-1);
            s->nstop=1;
            return 1;
        }
    }
#else
    for(int i=0;i<s->nbsphere;++i) {
        COMPLEX *field=s->FF0+3*(size_t)i;
        gaussianchamp_(&s->xs[i],&s->ys[i],&s->zs[i],
            &s->xgaus,&s->ygaus,&s->zgaus,&s->theta,&s->phi,
            &s->waist,&s->k0,&s->ss,&s->pp,&s->E0,
            &field[0],&field[1],&field[2],&s->tolinit,&nloin,
            &s->nstop,s->infostr,64);
    }
#endif
    return s->nstop==0?0:1;
}
//-------------------------------------------

int cdmlib_compute_gaussian_circular_incident_c99(cdmlib_incidentphase_c99 *s)
{
    const int nloin=0;
    s->xgaus=s->xgaus_nm*1.0e-9;
    s->ygaus=s->ygaus_nm*1.0e-9;
    s->zgaus=s->zgaus_nm*1.0e-9;
    gaussianpuissance_(&s->power,&s->irradiance,&s->waist,&s->k0,&s->E0);
    s->I0=pow(cabs(s->E0),2.0);
    s->nstop=0;
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    {
        const int circular=1;
        const int nloin=0;
        int status=1;
        const auto grid=cdmlib_incident_grid_spec(s);
        ifdda_rigorous_gaussian_field_cuda_execute(
            s->xs,s->ys,s->zs,&s->nbsphere,&s->xgaus,&s->ygaus,&s->zgaus,
            &s->theta,&s->phi,&s->waist,&s->k0,&s->ss,&s->pp,&s->E0,
            &s->tolinit,&circular,&nloin,s->FF0,&status,false,nullptr,&grid);
        if(status!=0) {
            memcpy(s->infostr,"CUDA circular rigorous Gaussian illumination failed",
                   sizeof "CUDA circular rigorous Gaussian illumination failed"-1);
            s->nstop=1;
            return 1;
        }
    }
#else
    for(int i=0;i<s->nbsphere;++i) {
        COMPLEX *field=s->FF0+3*(size_t)i;
        gaussianchampcirc_(&s->xs[i],&s->ys[i],&s->zs[i],
            &s->xgaus,&s->ygaus,&s->zgaus,&s->theta,&s->phi,
            &s->waist,&s->k0,&s->ss,&s->E0,
            &field[0],&field[1],&field[2],&s->tolinit,&nloin,
            &s->nstop,s->infostr,64);
    }
#endif
    return s->nstop==0?0:1;
}


//-------------------------------------------

int cdmlib_compute_gaussian_fft_incident_c99(cdmlib_incidentphase_c99 *s)
{
    s->xgaus=s->xgaus_nm*1.0e-9;
    s->ygaus=s->ygaus_nm*1.0e-9;
    s->zgaus=s->zgaus_nm*1.0e-9;
    gaussianpuissance_(&s->power,&s->irradiance,&s->waist,&s->k0,&s->E0);
    s->I0=pow(cabs(s->E0),2.0);
    s->nstop=0;
    gaussianchampfft_(s->xs,s->ys,s->zs,&s->aretecube,&s->k0,&s->waist,
        &s->E0,&s->ss,&s->pp,&s->theta,&s->phi,&s->xgaus,&s->ygaus,
        &s->zgaus,s->beam,&s->ndipole,&s->nx,&s->ny,&s->nz,&s->nxm,
        &s->nym,&s->nzm,&s->nmax,&s->nfft2d,s->imagex,s->imagey,
        s->imagez,s->FF0,&s->tolinit,&s->plan2f,&s->plan2b,&s->nstop,
        s->infostr,64,64);
    return s->nstop==0?0:1;
}
//-------------------------------------------

int cdmlib_compute_arbitrary_incident_c99(cdmlib_incidentphase_c99 *s)
{
    s->nstop=0;
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    {
        const int direct_device_consumer=0;
        int status=1;
        ifdda_incident_arbitrary_cuda_(s->xs,s->ys,s->zs,&s->aretecube,s->FF0,
            &s->nxm,&s->nym,&s->nzm,&s->nbsphere,&direct_device_consumer,
            &status,s->namefileinc,s->infostr);
        if(status!=0) {
            s->nstop=1;
            return 1;
        }
    }
#else
    incidentarbitrary_((REAL *)s->xs,(REAL *)s->ys,(REAL *)s->zs,
        &s->aretecube,s->FF0,&s->nxm,&s->nym,&s->nzm,&s->nbsphere,
        &s->nstop,s->namefileinc,s->infostr,s->namefileinc_len,
        sizeof s->infostr);
#endif
    return s->nstop==0?0:1;
}
//-------------------------------------------

static int cdmlib_incidentphase__cdmlib_compute_angular_spectrum_incident_c99(
    cdmlib_incidentphase_c99 *s,int kind)
{
    const REAL one=1.0;
    const int nf=s->nfft2d;
    s->xgaus=s->xgaus_nm*1.0e-9;
    s->ygaus=s->ygaus_nm*1.0e-9;
    s->zgaus=s->zgaus_nm*1.0e-9;
    s->E0=one+0.0*I;
    s->nstop=0;
    if(kind==0||kind==1) {
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        /* Match cdmlib.f90: speckle has both propagation directions,
           demispeckle only the positive direction.  The second random
           phase grid uses the Fortran-compatible seed speckseed+1. */
        const int bidirectional=kind==0?1:0;
        int status=1;
        ifdda_prepare_speckle_phases_(&nf,&nf,&s->k0,&s->numaperil,
            &s->aretecube,&s->speckseed,&bidirectional,s->imagex,s->imagey);
        ifdda_speckle_field_cuda_(&s->nx,&s->ny,&s->nz,&s->ndipole,&nf,
            &s->k0,&s->xgaus,&s->ygaus,&s->zgaus,&s->pp,&s->aretecube,
            &s->power,&s->numaperil,&s->E0,s->imagex,s->imagey,s->xs,
            s->ys,s->zs,&bidirectional,s->FF0,&s->irradiance,&status);
        if(status!=0) {
            memcpy(s->infostr,"CUDA speckle illumination failed",
                   sizeof "CUDA speckle illumination failed"-1);
            s->nstop=1;
            return 1;
        }
#else
        void (*routine)(const REAL *,const REAL *,const REAL *,const REAL *,
            const REAL *,COMPLEX *,const REAL *,const REAL *,const REAL *,
            COMPLEX *,const REAL *,const int *,const int *,const int *,
            const int *,const int *,const int *,const int *,COMPLEX *,
            COMPLEX *,COMPLEX *,const int *,const REAL *,REAL *,
            const REAL *,int *,char *,const int64_t *,size_t)=
                kind==0?speckle_:demispeckle_;
        routine(&s->k0,&s->xgaus,&s->ygaus,&s->zgaus,&s->pp,&s->E0,
            s->xs,s->ys,s->zs,s->FF0,&s->aretecube,&s->nx,&s->ny,&s->nz,
            &s->ndipole,&s->nmax,&nf,&nf,s->imagex,s->imagey,s->imagez,
            &s->speckseed,&s->power,&s->irradiance,&s->numaperil,&s->nstop,
            s->infostr,&s->plan2b,sizeof s->infostr);
#endif
    } else {
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        const int bidirectional=kind==2?1:0;
        int status=1;
        ifdda_confocal_field_cuda_(&s->nx,&s->ny,&s->nz,&s->ndipole,&nf,
            &s->k0,&s->xgaus,&s->ygaus,&s->zgaus,&s->pp,&s->aretecube,
            &s->power,&s->numaperil,&s->E0,s->xs,s->ys,s->zs,
            &bidirectional,s->FF0,&s->irradiance,&status);
        if(status!=0) {
            memcpy(s->infostr,kind==2?
                "CUDA confocal illumination failed":
                "CUDA demi-confocal illumination failed",
                kind==2?
                sizeof "CUDA confocal illumination failed"-1:
                sizeof "CUDA demi-confocal illumination failed"-1);
            s->nstop=1;
            return 1;
        }
#else
        void (*routine)(const REAL *,const REAL *,const REAL *,const REAL *,
            const REAL *,COMPLEX *,const REAL *,const REAL *,const REAL *,
            COMPLEX *,const REAL *,const int *,const int *,const int *,
            const int *,const int *,const int *,const int *,COMPLEX *,
            COMPLEX *,COMPLEX *,const REAL *,REAL *,const REAL *,
            int *,char *,const int64_t *,size_t)=kind==2?confocal_:demiconfocal_;
        routine(&s->k0,&s->xgaus,&s->ygaus,&s->zgaus,&s->pp,&s->E0,
            s->xs,s->ys,s->zs,s->FF0,&s->aretecube,&s->nx,&s->ny,&s->nz,
            &s->ndipole,&s->nmax,&nf,&nf,s->imagex,s->imagey,s->imagez,
            &s->power,&s->irradiance,&s->numaperil,&s->nstop,s->infostr,
            &s->plan2b,sizeof s->infostr);
#endif
    }
    return s->nstop==0?0:1;
}
//-------------------------------------------

int cdmlib_compute_incident_field_phase_c99(cdmlib_incidentphase_c99 *s)
{


    donothing(2);

    if (memcmp(s->beam,"pwavelinear",11)==0)
        return cdmlib_compute_plane_linear_incident_c99(s);
    if (memcmp(s->beam,"pwavecircular",13)==0)
        return cdmlib_compute_plane_circular_incident_c99(s);
    if (memcmp(s->beam,"wavelinearmulti",15)==0)
        return cdmlib_compute_plane_multi_incident_c99(s);
    if (memcmp(s->beam,"antenna",7)==0)
        return cdmlib_compute_antenna_incident_c99(s);
    if (memcmp(s->beam,"gparawavelinear",15)==0)
        return cdmlib_compute_gaussian_para_linear_incident_c99(s);
    if (memcmp(s->beam,"gparawavecircular",17)==0)
        return cdmlib_compute_gaussian_para_circular_incident_c99(s);
    if (memcmp(s->beam,"gfftwave",8)==0)
        return cdmlib_compute_gaussian_fft_incident_c99(s);
    if (memcmp(s->beam,"speckle",7)==0)
        return cdmlib_incidentphase__cdmlib_compute_angular_spectrum_incident_c99(s,0);
    if (memcmp(s->beam,"demispeckle",11)==0)
        return cdmlib_incidentphase__cdmlib_compute_angular_spectrum_incident_c99(s,1);
    if (memcmp(s->beam,"confocal",8)==0)
        return cdmlib_incidentphase__cdmlib_compute_angular_spectrum_incident_c99(s,2);
    if (memcmp(s->beam,"demiconfocal",12)==0)
        return cdmlib_incidentphase__cdmlib_compute_angular_spectrum_incident_c99(s,3);
    if (memcmp(s->beam,"gwavelinear",11)==0)
        return cdmlib_compute_gaussian_linear_incident_c99(s);
    if (memcmp(s->beam,"gwavecircular",13)==0)
        return cdmlib_compute_gaussian_circular_incident_c99(s);
    if (memcmp(s->beam,"arbitrary",9)==0)
        return cdmlib_compute_arbitrary_incident_c99(s);
    memset(s->infostr,' ',sizeof s->infostr);
    memcpy(s->infostr,"Beam unknown!",13);
    s->nstop=1;
    return 1;
}


/* Incorporated cdmlib_entryphase */
/* Entry-phase declarations integrated in this translation unit. */
#include "hdf5_helpers.h"
#include "cdmlib_fftphase.h"
#include "cdmlib_incidentphase.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>




//-------------------------------------------

static int cdmlib_entryphase__copy_fixed(char *dst,size_t dst_len,const char *src,size_t src_len)
{
    if(!dst||!src||src_len<dst_len) return 1;
    memcpy(dst,src,dst_len);
    return 0;
}
//-------------------------------------------

int cdmlib_build_phase_views_c99(const cdmlib_abi_args_c99 *a,
                                 cdmlib_initphase_c99 *s,
                                 cdmlib_config_c99 *c)
{
    if(!a||!s||!c||!a->nxm||!a->nym||!a->nzm||!a->numberobjet||
       !a->nnnr||!a->tolinit||!a->lambda||!a->w0||!a->p0||
       !a->xgmulti||!a->ygmulti||!a->zgmulti||!a->rayonmulti||
       !a->epsmulti||!a->epsanimulti||!a->nside||!a->beam||!a->object||
       !a->trope||!a->polarizability||!a->materiaumulti) return 1;

    *s=cdmlib_initphase_c99{};
    *c=cdmlib_config_c99{};

    s->nxm=(int)*a->nxm; s->nym=(int)*a->nym; s->nzm=(int)*a->nzm;
    s->nnnr=(int)*a->nnnr; s->numberobjet=(int)*a->numberobjet;
    if(cdmlib_entryphase__copy_fixed(s->object,sizeof s->object,a->object,a->object_len)||
       cdmlib_entryphase__copy_fixed(s->trope,sizeof s->trope,a->trope,a->trope_len)||
       cdmlib_entryphase__copy_fixed(s->polarizability,sizeof s->polarizability,
                  a->polarizability,a->polarizability_len)) return 1;
    s->materiaumulti=a->materiaumulti;
    s->tolinit=*a->tolinit;
    s->lambda_nm=*a->lambda;
    s->w0_nm=*a->w0;
    s->power=*a->p0;
    s->xgmulti=a->xgmulti; s->ygmulti=a->ygmulti; s->zgmulti=a->zgmulti;
    s->rayonmulti=a->rayonmulti;
    s->epsmulti=(COMPLEX *)a->epsmulti;
    s->epsanimulti=a->epsanimulti;
    s->nproche=(int)*a->nproche;
    s->eps0i=*a->eps0i;

    if(cdmlib_entryphase__copy_fixed(c->beam,sizeof c->beam,a->beam,a->beam_len)||
       cdmlib_entryphase__copy_fixed(c->object,sizeof c->object,a->object,a->object_len)||
       cdmlib_entryphase__copy_fixed(c->methodeit,sizeof c->methodeit,a->methodeit,
                  a->methodeit_len)||
       cdmlib_entryphase__copy_fixed(c->h5file,sizeof c->h5file,a->h5file,a->h5file_len))
        return 1;

    /* The validator performs the original 0 -> +1 / +1 -> -1 conversion.
       Keep this view builder side-effect free: it runs again after validation
       and must not invert an already normalized side a second time. */
    c->nside=(int)*a->nside;

    if(memcmp(c->beam,"greentensor",11)==0) {
        *a->nmacro=1;
        *a->nlecture=0;
    }
    c->nrig=(int)*a->nrig; c->nproche=(int)*a->nproche;
    c->nlentille=(int)*a->nlentille; c->nenergie=(int)*a->nenergie;
    c->nsection=(int)*a->nsection; c->nsectionsca=(int)*a->nsectionsca;
    c->nquickdiffracte=(int)*a->nquickdiffracte;
    c->nquicklens=(int)*a->nquicklens;
    c->nlocal=(int)*a->nlocal; c->nmacro=(int)*a->nmacro;
    c->nobjet=(int)*a->nobjet; c->nforce=(int)*a->nforce;
    c->nforced=(int)*a->nforced; c->ntorque=(int)*a->ntorque;
    c->ntorqued=(int)*a->ntorqued; c->nlecture=(int)*a->nlecture;
    c->nmat=(int)*a->nmat; c->nprecon=(int)*a->nprecon;
    c->ninitest=(int)*a->ninitest; c->nfft2d=(int)*a->nfft2d;
    c->nnnr=(int)*a->nnnr; c->ntypemic=(int)*a->ntypemic;
    c->testgreen=memcmp(c->beam,"greentensor",11)==0 ? 1 : 0;
    c->nphi=(int)*a->nphi; c->ntheta=(int)*a->ntheta;
    c->isotropic=memcmp(a->trope,"iso",3)==0;
    c->nxm=s->nxm; c->nym=s->nym; c->nzm=s->nzm;
    c->lambda=*a->lambda; c->w0=*a->w0; c->P0=*a->p0;
    c->theta=*a->theta; c->tolinit=*a->tolinit;
    c->numaper=*a->numaper; c->numaperinc=*a->numaperinc;
    c->numaperinc2=*a->numaperinc2; c->numaperil=*a->numaperil;
    c->kcnax=*a->kcnax; c->kcnay=*a->kcnay;
    c->gross=*a->gross; c->zlens=*a->zlens;
#ifdef USE_HDF5
    c->has_hdf5=1;
#else
    c->has_hdf5=0;
#endif
    return 0;
}
//-------------------------------------------

static void cdmlib_entryphase__set_failure(const cdmlib_abi_args_c99 *a,const char *message)
{
    size_t n=strlen(message);
    if(n>a->infostr_len) n=a->infostr_len;
    memset(a->infostr,' ',a->infostr_len);
    memcpy(a->infostr,message,n);
    *a->nstop=1;
}
//-------------------------------------------

int cdmlib_prepare_preobject_from_abi_c99(const cdmlib_abi_args_c99 *a,
                                          cdmlib_initphase_c99 *s,
                                          cdmlib_config_c99 *c,
                                          REAL *k0,REAL *k03)
{
    int nstop=0;
    if(!a||!s||!c||!k0||!k03||!a->nstop||!a->infostr) return 1;
    *a->nstop=0;
    memset(a->infostr,' ',a->infostr_len);
    if(cdmlib_build_phase_views_c99(a,s,c)!=0) {
        cdmlib_entryphase__set_failure(a,"Invalid cdmlib ABI arguments");
        return 1;
    }
    if(cdmlib_validate_config_c99(c,&nstop,s->infostr)!=0) {
        *a->nstop=(cdmlib_ip)nstop;
        memcpy(a->infostr,s->infostr,a->infostr_len<sizeof s->infostr?
               a->infostr_len:sizeof s->infostr);
        return 1;
    }
    *a->nside=(cdmlib_ip)c->nside;
    *a->nmacro=(cdmlib_ip)c->nmacro;
    *a->nlecture=(cdmlib_ip)c->nlecture;
    *a->nproche=(cdmlib_ip)c->nproche;
    s->nproche=c->nproche;
    if(cdmlib_prepare_preobject_phase_c99(s,c,k0,k03)!=0) {
        *a->nstop=(cdmlib_ip)s->nstop;
        memcpy(a->infostr,s->infostr,a->infostr_len<sizeof s->infostr?
               a->infostr_len:sizeof s->infostr);
        return 1;
    }
    *a->nproche=(cdmlib_ip)s->nproche;
    return 0;
}
//-------------------------------------------

int cdmlib_initialize_geometry_c99(const cdmlib_abi_args_c99 *a,
                                  cdmlib_initphase_c99 *s,
                                  cdmlib_config_c99 *c,
                                  cdmlib_entry_runtime_c99 *r)
{
    if(!a||!s||!c||!r||!a->nstop||!a->infostr||!a->nbsphere||
       !a->ndipole||!a->aretecube||!a->lambda10n||!a->k0||!a->numberobjet||
       !a->xswf||!a->yswf||!a->zswf||!a->density||!a->side||!a->sidex||
       !a->sidey||!a->sidez||!a->hauteur||!a->lnrbc||!a->lcos||!a->lc||
       !a->hc||!a->ng||!a->ntypeabc||!a->nmat||!a->niso||!a->tabdip||
       !a->tabmulti||!a->xs||!a->ys||!a->zs||!a->nxmp||!a->nymp||!a->nzmp||
       !a->demiaxea||!a->demiaxeb||!a->demiaxec||!a->phiobj||!a->thetaobj||
       !a->psiobj||!a->macroscopicfieldx||!a->epsilon||!a->polarisa||
       !a->namefileobj) {
        if(a&&a->nstop&&a->infostr) cdmlib_entryphase__set_failure(a,"Invalid cdmlib geometry arguments");
        return 1;
    }

    *a->nstop=0;
    memset(a->infostr,' ',a->infostr_len);
    if(cdmlib_build_phase_views_c99(a,s,c)!=0) {
        cdmlib_entryphase__set_failure(a,"Invalid cdmlib ABI arguments");
        return 1;
    }

    *r=cdmlib_entry_runtime_c99{};
    r->file_id=r->group_idopt=r->group_iddip=r->group_idff=-1;
    r->group_idmic=r->group_idof=r->group_idnf=r->group_idver=-1;
    if(c->nmat==2) {
        int64_t ids[8];
        if(!c->has_hdf5 || cdmlib_hdf5_create_groups_c99(
             a->h5file,a->h5file_len,ids)!=0) {
            cdmlib_entryphase__set_failure(a,"Could not initialize C99 HDF5 output");
            return 1;
        }
        r->file_id=ids[0]; r->group_idopt=ids[1]; r->group_iddip=ids[2];
        r->group_idff=ids[3]; r->group_idmic=ids[4]; r->group_idof=ids[5];
        r->group_idnf=ids[6]; r->group_idver=ids[7];
    }
    return 0;
}
//-------------------------------------------

int cdmlib_execute_geometry_c99(const cdmlib_abi_args_c99 *a,
                               cdmlib_initphase_c99 *s,
                               cdmlib_config_c99 *c,
                               cdmlib_entry_runtime_c99 *r,
                               cdmlib_geometry_params_c99 *geometry,
                               REAL *k0,REAL *k03,REAL *lambda10n)
{
    if(!a||!s||!c||!r||!geometry||!k0||!k03||!lambda10n||
       !geometry->object.tabmulti||!geometry->object.nmax) {
        if(a && a->nstop && a->infostr)
            cdmlib_entryphase__set_failure(a,"Invalid geometry parameter view");
        return 1;
    }
    cdmlib_object_phase_c99 *g=&geometry->object;

    /* cdmlib.f90 clears the complete tabmulti workspace immediately before
       dispatching to the selected geometry constructor. */
    for(int i=0;i<s->nmax;++i) g->tabmulti[i]=0;
    const int rc=cdmlib_construct_object_phase_c99(s,c,g,k0,k03,lambda10n);
    r->xg=s->xg; r->yg=s->yg; r->zg=s->zg; r->radius=s->rayon;
    *a->nstop=(cdmlib_ip)s->nstop;
    *a->k0=(cdmlib_real)*k0;
    *a->lambda10n=(cdmlib_real)*lambda10n;
    *a->lambda=(cdmlib_real)s->lambda;
    *a->nproche=(cdmlib_ip)s->nproche;
    *a->nmacro=(cdmlib_ip)c->nmacro;
    *a->nlecture=(cdmlib_ip)c->nlecture;
    memcpy(a->infostr,s->infostr,a->infostr_len<sizeof s->infostr?
           a->infostr_len:sizeof s->infostr);
    return rc;
}
//-------------------------------------------

static int cdmlib_entryphase__map_incident_to_dipoles_c99(const cdmlib_abi_args_c99 *a,
                                      const cdmlib_initphase_c99 *s,
                                      const cdmlib_config_c99 *c,
                                      int ndipole)
{
    const COMPLEX *ff0=(const COMPLEX *)a->ff0;
    const int *tabdip=(const int *)a->tabdip;
    COMPLEX *fx=(COMPLEX *)a->incidentfieldx;
    COMPLEX *fy=(COMPLEX *)a->incidentfieldy;
    COMPLEX *fz=(COMPLEX *)a->incidentfieldz;
    REAL *magnitude=(REAL *)a->incidentfield;
    COMPLEX *wrk=(COMPLEX *)a->wrk;
    int subunit=0;
    FILE *files[4]={NULL,NULL,NULL,NULL};
    const int write_files=c->nmat==0&&c->testgreen<=2;

    if(!a->incidentfield||!a->incidentfieldx||!a->incidentfieldy||
       !a->incidentfieldz||!a->wrk) return 1;
    if(write_files) {
        const char *names[4]={"incidentfieldx.mat","incidentfieldy.mat",
                              "incidentfieldz.mat","incidentfield.mat"};
        for(int j=0;j<4;++j) {
            files[j]=fopen(names[j],"w");
            if(!files[j]) {
                for(int k=0;k<4;++k) if(files[k]) fclose(files[k]);
                return 1;
            }
        }
    }
    for(int i=0;i<ndipole;++i) {
        const int k=tabdip[i];
        if(k==0) {
            if(write_files) {
                for(int j=0;j<3;++j) fprintf(files[j]," 0 0\n");
                fputs(" 0\n",files[3]);
            }
            if(c->nmat==0) continue;
            if(c->nmat==2) {
                wrk[(size_t)i+(size_t)s->nmax*0]=0.0+0.0*I;
                wrk[(size_t)i+(size_t)s->nmax*1]=0.0+0.0*I;
                wrk[(size_t)i+(size_t)s->nmax*2]=0.0+0.0*I;
                wrk[(size_t)i+(size_t)s->nmax*3]=0.0+0.0*I;
            }
            continue;
        }
        const size_t f=(size_t)3*(size_t)(k-1);
        const COMPLEX ex=ff0[f],ey=ff0[f+1],ez=ff0[f+2];
        const REAL value=sqrt(creal(ex*conj(ex)+ey*conj(ey)+ez*conj(ez)));
        const size_t out=(size_t)(c->nmat==0?++subunit:k)-1;
        fx[out]=ex; fy[out]=ey; fz[out]=ez; magnitude[out]=value;
        if(write_files) {
            fprintf(files[0]," %.17g %.17g\n",creal(ex),cimag(ex));
            fprintf(files[1]," %.17g %.17g\n",creal(ey),cimag(ey));
            fprintf(files[2]," %.17g %.17g\n",creal(ez),cimag(ez));
            fprintf(files[3]," %.17g\n",value);
        }
        if(c->nmat==2) {
            wrk[(size_t)i]=ex; wrk[(size_t)i+(size_t)s->nmax]=ey;
            wrk[(size_t)i+2*(size_t)s->nmax]=ez;
            wrk[(size_t)i+3*(size_t)s->nmax]=value+0.0*I;
        }
    }
    for(int j=0;j<4;++j) if(files[j]&&fclose(files[j])!=0) return 1;
    if(memcmp(c->beam,"gfftwave",8)==0) {
        REAL mx=0.0, mi=0.0;
        for(int i=0;i<ndipole;++i) {
            const REAL v=sqrt(creal(fx[i]*conj(fx[i])+fy[i]*conj(fy[i])+fz[i]*conj(fz[i])));
            if(v>mx) mx=v;
            if(magnitude[i]>mi) mi=magnitude[i];
        }
    }
    return 0;
}
//-------------------------------------------

int cdmlib_compute_incident_from_abi_c99(const cdmlib_abi_args_c99 *a,
                                         cdmlib_initphase_c99 *s,
                                         cdmlib_config_c99 *c,
                                         cdmlib_entry_runtime_c99 *r,
                                         int nbsphere,int ndipole)
{
    cdmlib_incidentphase_c99 p;
    int rc;
    if(!a||!s||!c||!r||!a->nstop||!a->infostr||!a->e0||!a->irra||
       !a->ff0||!a->xs||!a->ys||!a->zs||!a->tabmulti||!a->e0m||
       !a->xgaus||!a->ygaus||!a->zgaus||!a->theta||!a->phi||!a->ss||!a->pp||
       !a->thetam||!a->phim||!a->ppm||!a->ssm||!a->nfft2d||!a->ntypemic||
       !a->nlentille||!a->nenergie||!a->numaperil||!a->speckseed||
       !a->eimagex||!a->eimagey||!a->eimagez) {
        if(a&&a->nstop&&a->infostr) cdmlib_entryphase__set_failure(a,"Invalid cdmlib incident arguments");
        return 1;
    }
    p=cdmlib_incidentphase_c99{};
    p.nbsphere=nbsphere; p.ndipole=ndipole; p.nmax=s->nmax;
    p.nenergie=(int)*a->nenergie; p.ntypemic=(int)*a->ntypemic;
    p.nlentille=(int)*a->nlentille; p.nfft2d=(int)*a->nfft2d;
    p.speckseed=(int)*a->speckseed;
    p.nbinc=(int)*a->nbinc; p.power=s->power; p.waist=s->w0;
    p.k0=(REAL)*a->k0; p.aretecube=(REAL)*a->aretecube;
    p.numaperil=(REAL)*a->numaperil;
    p.antenna_x_nm=(REAL)*a->xgaus; p.antenna_y_nm=(REAL)*a->ygaus;
    p.antenna_z_nm=(REAL)*a->zgaus;
    p.xg=r->xg; p.yg=r->yg; p.zg=r->zg;
    p.xgaus_nm=(REAL)*a->xgaus; p.ygaus_nm=(REAL)*a->ygaus;
    p.zgaus_nm=(REAL)*a->zgaus;
    p.theta=(REAL)*a->theta; p.phi=(REAL)*a->phi;
    p.ss=(REAL)*a->ss; p.pp=(REAL)*a->pp; p.tolinit=s->tolinit;
    p.eps0=s->eps0; p.E0=(COMPLEX)*a->e0;
    p.irradiance=(REAL)*a->irra;
    p.xs=(const REAL *)a->xs; p.ys=(const REAL *)a->ys;
    p.zs=(const REAL *)a->zs; p.tabmulti=(const int *)a->tabmulti;
    p.tabdip=(const int *)a->tabdip;
    p.thetam=(const REAL *)a->thetam; p.phim=(const REAL *)a->phim;
    p.ppm=(const REAL *)a->ppm; p.ssm=(const REAL *)a->ssm;
    p.E0m=(COMPLEX *)a->e0m; p.FF0=(COMPLEX *)a->ff0;
    p.nx=r->nx; p.ny=r->ny; p.nz=r->nz;
    p.nxm=s->nxm; p.nym=s->nym; p.nzm=s->nzm;
    p.plan2f=r->plan2f; p.plan2b=r->plan2b;
    memcpy(p.beam,c->beam,sizeof p.beam);
    if(a->namefileinc && a->namefileinc_len<=sizeof p.namefileinc) {
        memcpy(p.namefileinc,a->namefileinc,a->namefileinc_len);
        p.namefileinc_len=a->namefileinc_len;
    }
    p.imagex=(COMPLEX *)a->eimagex;
    p.imagey=(COMPLEX *)a->eimagey;
    p.imagez=(COMPLEX *)a->eimagez;
    p.nstop=(int)*a->nstop;
    memcpy(p.infostr,a->infostr,sizeof p.infostr);

    // The fixed device FF0 buffer must be allocated for the maximum grid,
    // before an illumination wrapper or solver can retain its address.
    if (!cuda_data::get_ff0(size_t(s->nmax))) {
        cdmlib_entryphase__set_failure(a,"Cannot reserve persistent CUDA FF0");
        return 1;
    }

    if(c->testgreen==1) {
        rc=cdmlib_set_greentensor_source_c99(a,s,0);
        if(rc==0) c->testgreen=2;
        else {
            p.nstop=(int)*a->nstop;
            memcpy(p.infostr,a->infostr,sizeof p.infostr);
        }
    } else {
        rc=cdmlib_compute_incident_field_phase_c99(&p);
    }
    if(rc==0&&cdmlib_entryphase__map_incident_to_dipoles_c99(a,s,c,ndipole)!=0) {
        memset(p.infostr,' ',sizeof p.infostr);
        memcpy(p.infostr,"Invalid incident field output buffers",37);
        p.nstop=1;
        rc=1;
    }
    *a->e0=(cdmlib_complex)p.E0;
    *a->irra=(cdmlib_real)p.irradiance;
    *a->w0=(cdmlib_real)p.waist;
    if(memcmp(c->beam,"gwave",5)==0||memcmp(c->beam,"gfftwave",8)==0||
       memcmp(c->beam,"gparawave",9)==0) {
        *a->xgaus=(cdmlib_real)p.xgaus;
        *a->ygaus=(cdmlib_real)p.ygaus;
        *a->zgaus=(cdmlib_real)p.zgaus;
    }
    *a->theta=(cdmlib_real)p.theta; *a->phi=(cdmlib_real)p.phi;
    r->E0=p.E0; r->irradiance=p.irradiance; r->I0=p.I0;
    r->norm=p.norm; r->tmp=p.tmp; r->waist=p.waist;
    r->xgaus=p.xgaus; r->ygaus=p.ygaus; r->zgaus=p.zgaus;
    r->antenna_x=p.antenna_x; r->antenna_y=p.antenna_y;
    r->antenna_z=p.antenna_z;
    *a->nstop=(cdmlib_ip)p.nstop;
    memcpy(a->infostr,p.infostr,a->infostr_len<sizeof p.infostr?
           a->infostr_len:sizeof p.infostr);
    return rc;
}

/* Single orchestration point for all illumination types. The specialized
   beam dispatcher is unchanged and uses the same persistent CUDA FF0. */
int cdmlib_execute_illumination_c99(
    const cdmlib_illumination_params_c99 *v)
{
    if(!v || !v->abi || !v->init || !v->config || !v->runtime ||
       !v->xs || !v->ys || !v->zs || !v->ff0 || !v->k0 ||
       !v->aretecube || !v->nbsphere || !v->ndipole ||
       !v->nlecture || !v->nstop || !v->infostr || !v->tabmulti ||
       !v->beam || !v->pml_mask)
        return 1;

    const cdmlib_abi_args_c99 *a=v->abi;
    cdmlib_initphase_c99 *init=v->init;
    cdmlib_config_c99 *cfg=v->config;
    cdmlib_entry_runtime_c99 *runtime=v->runtime;

    // --ILLUMINATION COMPUTATION--
    int rc=cdmlib_compute_incident_from_abi_c99(
        a,init,cfg,runtime,(int)*v->nbsphere,(int)*v->ndipole);

    // --PML START--
    // Correct the incident field in absorbing cells, then refresh its
    // per-dipole CPU views because the PML modifies FF0.
    if(rc==0 && init->nabc==1) {
        v->pml_mask->resize(size_t(init->nmax));
        const int mask_type=1;
        const int count=(int)*v->nbsphere;
        int mask_status=1;
        ifdda_apply_pml_mask_cuda_(v->xs,v->ys,v->zs,&init->eps0,
            v->k0,runtime->pmlwrk,&mask_type,&count,v->aretecube,
            v->ff0,v->pml_mask->data(),&mask_status);
        if(mask_status!=0) {
            const char msg[]="CUDA MDDA incident PML mask failed";
            memset(v->infostr,' ',v->infostr_len);
            memcpy(v->infostr,msg,std::min(sizeof msg-1,v->infostr_len));
            *v->nstop=1;
            rc=1;
        }
    }
    if(rc==0 && init->nabc==1) {
        rc=cdmlib_entryphase__map_incident_to_dipoles_c99(
            a,init,cfg,(int)*v->ndipole);
        if(rc!=0) {
            const char msg[]="CUDA MDDA masked incident field mapping failed";
            memset(v->infostr,' ',v->infostr_len);
            memcpy(v->infostr,msg,std::min(sizeof msg-1,v->infostr_len));
            *v->nstop=1;
            rc=1;
        }
    }
    // --PML END--

    // --ILLUMINATION GPU FF0--
    // The CPU FF0 remains authoritative after optional PML/label masking.
    // The solver receives the immutable resident GPU FF0 and makes its own
    // device-to-device RHS copy; upload is needed only after CPU correction.
    if(rc==0 && cfg->testgreen==0 && (int)*v->nlecture!=1) {
        const size_t count=size_t(*v->nbsphere);
        CUCOMPLEX *resident_ff0=cuda_data::get_ff0(size_t(init->nmax));
        bool cpu_modified=init->nabc==1;
        if(memcmp(v->beam,"pwavelinear",11)==0) {
            for(size_t dipole=0; dipole<count; ++dipole) {
                if(v->tabmulti[dipole]==-1) {
                    cpu_modified=true;
                    break;
                }
            }
        }
        if(!resident_ff0 ||
           (cpu_modified &&
            cuda_data::copy_host_complex_to_device(
                resident_ff0,(const COMPLEX*)v->ff0,3*count,
                cuda_utils::TransferCategory::IncidentField)!=cudaSuccess) ||
           !register_gpbicg_device_b_source(v->ff0,resident_ff0,3*count)) {
            const char msg[]="Cannot publish persistent CUDA illumination FF0";
            memset(v->infostr,' ',v->infostr_len);
            memcpy(v->infostr,msg,std::min(sizeof msg-1,v->infostr_len));
            *v->nstop=1;
            rc=1;
        }
    }
    return rc;
}
//-------------------------------------------

int cdmlib_set_greentensor_source_c99(const cdmlib_abi_args_c99 *a,
                                      const cdmlib_initphase_c99 *s,
                                      int polarization)
{
    if(!a||!s||polarization<0||polarization>2||!a->xgaus||!a->ygaus||
       !a->zgaus||!a->xs||!a->ys||!a->zs||!a->aretecube||!a->nbsphere||
       !a->ff0||!a->k0||!a->nstop||!a->infostr) return 1;
    REAL x=(REAL)*a->xgaus*1.0e-9;
    REAL y=(REAL)*a->ygaus*1.0e-9;
    REAL z=(REAL)*a->zgaus*1.0e-9;
    int outside=0;
    const int nb=(int)*a->nbsphere;
    dipoleinsidep_(&x,&y,&z,(const REAL *)a->xs,(const REAL *)a->ys,
        (const REAL *)a->zs,(const REAL *)a->aretecube,&s->nmax,&nb,&outside);
    if(outside) {
        cdmlib_entryphase__set_failure(a,"Point outside the object under study");
        return 1;
    }
    COMPLEX p[3]={0.0+0.0*I,0.0+0.0*I,0.0+0.0*I};
    p[polarization]=1.0+0.0*I;
    int nstop=0;
    for(int i=0;i<nb;++i) {
        COMPLEX *field=(COMPLEX *)a->ff0+3*(size_t)i;
        dipoleincp_(&x,&y,&z,p,(const REAL *)a->xs+i,
            (const REAL *)a->ys+i,(const REAL *)a->zs+i,
            (const REAL *)a->aretecube,(const REAL *)a->k0,
            &field[0],&field[1],&field[2],&nstop,a->infostr,a->infostr_len);
        if(nstop!=0) {
            *a->nstop=(cdmlib_ip)nstop;
            return 1;
        }
    }
    *a->nstop=0;
    return 0;
}
//-------------------------------------------

int cdmlib_map_incident_fields_c99(const cdmlib_abi_args_c99 *a,
                                   const cdmlib_initphase_c99 *s,
                                   const cdmlib_config_c99 *c,
                                   int ndipole)
{
    if(!a||!s||!c) return 1;
    return cdmlib_entryphase__map_incident_to_dipoles_c99(a,s,c,ndipole);
}
//-------------------------------------------

int cdmlib_prepare_fft_grid_from_abi_c99(const cdmlib_abi_args_c99 *a,
                                         const cdmlib_initphase_c99 *s,
                                         const cdmlib_config_c99 *c,
                                         cdmlib_entry_runtime_c99 *r,
                                         int nbsphere)
{
    cdmlib_fftgrid_c99 grid;
    int nstop=0;
    char infostr[64];
    if(!a||!s||!c||!r||!a->nstop||!a->infostr||!a->aretecube||!a->k0)
        return 1;
    memcpy(infostr,s->infostr,sizeof infostr);
    if(cdmlib_prepare_fft_grid_c99(r->nx,r->ny,r->nz,nbsphere,s->nabc,
          c->nlentille,c->nforce,c->ntorque,s->eps0,(REAL)*a->k0,
          (REAL)*a->aretecube,&grid,&nstop,infostr)!=0) {
        *a->nstop=(cdmlib_ip)nstop;
        memcpy(a->infostr,infostr,a->infostr_len<sizeof infostr?
               a->infostr_len:sizeof infostr);
        return 1;
    }
    r->nx2=grid.nx2; r->ny2=grid.ny2; r->nz2=grid.nz2;
    r->ntotal=grid.ntotal; r->nbsphere3=grid.nbsphere3;
    r->nxy2=grid.nxy2; r->nabs=grid.nabs;
    *a->nstop=(cdmlib_ip)nstop;
    return 0;
}
//-------------------------------------------

int cdmlib_allocate_fft_workspace_c99(cdmlib_entry_runtime_c99 *r)
{
    if(!r||r->ntotal<=0||(size_t)r->ntotal>SIZE_MAX/sizeof(COMPLEX))
        return 1;
    r->vectx=(COMPLEX *)calloc((size_t)r->ntotal,sizeof(COMPLEX));
    r->vecty=(COMPLEX *)calloc((size_t)r->ntotal,sizeof(COMPLEX));
    r->vectz=(COMPLEX *)calloc((size_t)r->ntotal,sizeof(COMPLEX));
    if(!r->vectx||!r->vecty||!r->vectz) {
        cdmlib_free_fft_workspace_c99(r);
        return 1;
    }
    return 0;
}
//-------------------------------------------

void cdmlib_free_fft_workspace_c99(cdmlib_entry_runtime_c99 *r)
{
    if(!r) return;
#ifdef USE_FFTW
    cdmlib_fftw_destroy_c99(r);
#endif
    free(r->vectx); free(r->vecty); free(r->vectz);
    r->vectx=NULL; r->vecty=NULL; r->vectz=NULL;
}
//-------------------------------------------

int cdmlib_execute_green_scalar_c99(
    const cdmlib_green_scalar_params_c99 *g)
{
    if(!g || (g->mode!=8 && g->mode!=9) ||
       !g->beam || !g->trope || !g->nprecon || !g->ninitest ||
       !g->ss || !g->pp || !g->theta || !g->phi || !g->w0 ||
       !g->xgaus || !g->ygaus || !g->zgaus || !g->eps0 || !g->e0 ||
       !g->nx || !g->ny || !g->nz || !g->nx2 || !g->ny2 ||
       !g->nz2 || !g->nxy2 || !g->nxm || !g->nym || !g->nzm ||
       !g->ntotalm || !g->aretecube || !g->k0 ||
       !g->ux || !g->uy || !g->uz || !g->green || !g->planb ||
       !g->nstop || !g->infostr || !g->ready)
        return 1;

    *g->ready=0;
    if(g->polarization_prepared!=1) {
        const bool warm_start=g->polarization_prepared==2;
        const int allowed=memcmp(g->beam,"pwavelinear",11)==0 ||
            memcmp(g->beam,"pwavecircular",13)==0 ||
            memcmp(g->beam,"gparawavelinear",15)==0 ||
            memcmp(g->beam,"gparawavecircular",17)==0 ||
            memcmp(g->beam,"gfftwavelinear",14)==0 ||
            memcmp(g->beam,"gfftwavecircular",16)==0;
        if(!allowed && warm_start) {
            const char msg[]="Scalar initialization beam not supported";
            memset(g->infostr,' ',g->infostr_len);
            memcpy(g->infostr,msg,std::min(sizeof msg-1,g->infostr_len));
            *g->nstop=1;
            return 1;
        }
        if(!warm_start && (!allowed || memcmp(g->trope,"ani",3)==0 ||
           !((int)*g->nprecon==0 || (int)*g->nprecon==1 ||
             (int)*g->nprecon==2) ||
           !((int)*g->ninitest==0 || (int)*g->ninitest==1 ||
             (int)*g->ninitest==5 || (int)*g->ninitest==2 ||
             (int)*g->ninitest==3))) {
            const char msg[]="C99 scalar branch options not connected";
            memset(g->infostr,' ',g->infostr_len);
            memcpy(g->infostr,msg,std::min(sizeof msg-1,g->infostr_len));
            *g->nstop=1;
            return 1;
        }

        REAL x=0.0,y=0.0,z=0.0;
        REAL k0_value=*g->k0,ss_value=*g->ss,pp_value=*g->pp;
        REAL theta_value=*g->theta,phi_value=*g->phi;
        REAL w0_value=*g->w0,xg=*g->xgaus,yg=*g->ygaus,zg=*g->zgaus;
        const COMPLEX eps0=*g->eps0,e0_value=*g->e0;
        int unit_stop=0;
        if(memcmp(g->beam,"pwavelinear",11)==0)
            ondeplane_(&x,&y,&z,&k0_value,&eps0,&e0_value,&ss_value,
                &pp_value,&theta_value,&phi_value,g->ux,g->uy,g->uz,
                &unit_stop,g->infostr,g->infostr_len);
        else if(memcmp(g->beam,"pwavecircular",13)==0)
            ondecirce_(&x,&y,&z,&k0_value,&e0_value,&ss_value,
                &theta_value,&phi_value,g->ux,g->uy,g->uz);
        else if(memcmp(g->beam,"gparawavelinear",15)==0 ||
                memcmp(g->beam,"gfftwavelinear",14)==0)
            gaussianparalinear_(&x,&y,&z,&xg,&yg,&zg,&theta_value,
                &phi_value,&w0_value,&k0_value,&ss_value,&pp_value,
                &eps0,&e0_value,g->ux,g->uy,g->uz,
                &unit_stop,g->infostr,g->infostr_len);
        else
            gaussianparacirc_(&x,&y,&z,&xg,&yg,&zg,&theta_value,
                &phi_value,&w0_value,&k0_value,&ss_value,&eps0,
                &e0_value,g->ux,g->uy,g->uz,
                &unit_stop,g->infostr,g->infostr_len);
        if(unit_stop!=0) {
            *g->nstop=(cdmlib_ip)unit_stop;
            return 1;
        }
        const REAL mag=sqrt(creal((*g->ux)*conj(*g->ux)+
            (*g->uy)*conj(*g->uy)+(*g->uz)*conj(*g->uz)));
        if(!(mag>0.0) || !isfinite(mag)) {
            const char msg[]="Invalid scalar incident polarization";
            memset(g->infostr,' ',g->infostr_len);
            memcpy(g->infostr,msg,std::min(sizeof msg-1,g->infostr_len));
            *g->nstop=1;
            return 1;
        }
        *g->ux/=mag; *g->uy/=mag; *g->uz/=mag;
    }

    // Uses the existing FFT work arrays. No GPU allocation or new transfer.
    if(g->mode==9)
        greencalculscalairefft2_(g->nx,g->ny,g->nz,
            g->nx2,g->ny2,g->nz2,g->nxy2,g->nxm,g->nym,g->nzm,
            g->aretecube,g->k0,g->ux,g->uy,g->uz,g->green,g->planb,
            g->ntotalm,(int *)g->nstop,g->infostr,g->infostr_len);
    else
        greencalculscalairefft_(g->nx,g->ny,g->nz,
            g->nx2,g->ny2,g->nz2,g->nxy2,g->nxm,g->nym,g->nzm,
            g->aretecube,g->k0,g->ux,g->uy,g->uz,g->green,g->planb,
            g->ntotalm,(int *)g->nstop,g->infostr,g->infostr_len);
    if(*g->nstop!=0) return 1;
    *g->ready=1;
    return 0;
}
//-------------------------------------------

int cdmlib_execute_green_c99(const cdmlib_green_params_c99 *green)
{
    if(!green || !green->abi || !green->init || !green->config ||
       !green->runtime || !green->xx || !green->xy || !green->xz ||
       !green->yy || !green->yz || !green->zz || !green->k0 ||
       !green->aretecube || !green->nquad || !green->ntypeabc ||
       !green->theta || !green->phi || !green->eps0 ||
       !green->polarizability || !green->nstop || !green->infostr)
        return 1;
    return cdmlib_compute_green_from_abi_c99(
        green->abi,green->init,green->config,green->runtime);
}
//-------------------------------------------

int cdmlib_compute_green_from_abi_c99(const cdmlib_abi_args_c99 *a,
                                      const cdmlib_initphase_c99 *s,
                                      const cdmlib_config_c99 *c,
                                      cdmlib_entry_runtime_c99 *r)
{
    // Reuse the six tensors on the unchanged grid. Microscopy and testgreen
    // may re-enter this function after the main vector solve; neither needs
    // to regenerate the same GPU Green operator.
    if(r && r->vector_green_ready) return 0;
    donothing(3);
    if(!a||!s||!c||!r||!a->nstop||
       !a->ffttensorxx||!a->ffttensorxy||!a->ffttensorxz||!a->ffttensoryy||
       !a->ffttensoryz||!a->ffttensorzz||!a->nquad||!a->aretecube||!a->k0||
       !a->ntypeabc||!a->theta||!a->phi) {
        if(a&&a->nstop&&a->infostr) cdmlib_entryphase__set_failure(a,"Invalid cdmlib Green arguments");
        return 1;
    }
#ifndef IFDDA_CUDA_INVERSE_DRIVER
    // The CPU fallback still requires the three FFTW work arrays.
    if(!r->vectx||!r->vecty||!r->vectz) {
        cdmlib_entryphase__set_failure(a,"Invalid CPU FFTW workspace");
        return 1;
    }
#endif
    const int nx=r->nx,ny=r->ny,nz=r->nz;
    const int nx2=r->nx2,ny2=r->ny2,nz2=r->nz2;
    const int nxm=s->nxm,nym=s->nym,nzm=s->nzm,nquad=(int)*a->nquad;
    const COMPLEX *eps0=&s->eps0;
    COMPLEX *xx=(COMPLEX *)a->ffttensorxx;
    COMPLEX *xy=(COMPLEX *)a->ffttensorxy;
    COMPLEX *xz=(COMPLEX *)a->ffttensorxz;
    COMPLEX *yy=(COMPLEX *)a->ffttensoryy;
    COMPLEX *yz=(COMPLEX *)a->ffttensoryz;
    COMPLEX *zz=(COMPLEX *)a->ffttensorzz;
    /*
     * CUDA/cdmlib.f90, lines 333-385: the active inverse-driver branch
     * generates every vector Green tensor through the CUDA ABI.  Keep the
     * medium, quadrature and filtered-polarizability selectors explicit;
     * the CPU Green routines remain the non-CUDA fallback.
     */
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    const int medium=s->nabc==1 ? 1 : 0;
    const int filtered=(s->nabc==0 && nquad==0 &&
        memcmp(s->polarizability,"FG",2)==0) ? 1 : 0;
    int status=1;
    ifdda_generate_vector_green_cuda(xx,xy,xz,yy,yz,zz,
        &nx,&ny,&nz,&nxm,&nym,&nzm,&nx2,&ny2,&nz2,
        (const REAL *)a->aretecube,(const REAL *)a->k0,eps0,
        &medium,&nquad,&filtered,&status);
    if(status!=0) {
        cdmlib_entryphase__set_failure(a,
            "CUDA vector Green tensor generation failed");
        return 1;
    }
#else
    const int64_t planb=r->planb;
    if(s->nabc==1) {
        if(nquad==0)
            greencalculfftc_(&nx,&ny,&nz,&nx2,&ny2,&nz2,&r->nxy2,&nxm,&nym,&nzm,
                (REAL *)a->aretecube,(REAL *)a->k0,eps0,(const int *)a->ntypeabc,
                (const REAL *)a->theta,(const REAL *)a->phi,&s->tolinit,
                xx,xy,xz,yy,yz,zz,r->vectx,r->vecty,r->vectz,&planb);
        else
            greencalculquadfftc_(&nx,&ny,&nz,&nx2,&ny2,&nz2,&r->nxy2,&nxm,&nym,&nzm,
                &nquad,(REAL *)a->aretecube,(REAL *)a->k0,eps0,
                (const int *)a->ntypeabc,(const REAL *)a->theta,
                (const REAL *)a->phi,&s->tolinit,xx,xy,xz,yy,yz,zz,
                r->vectx,r->vecty,r->vectz,&planb);
    } else if(nquad==0) {
        if(memcmp(s->polarizability,"FG",2)==0)
            greenfilcalculfft_(&nx,&ny,&nz,&nx2,&ny2,&nz2,&r->nxy2,&nxm,&nym,&nzm,
                (REAL *)a->aretecube,(REAL *)a->k0,xx,xy,xz,yy,yz,zz,
                r->vectx,r->vecty,r->vectz,&planb);
        else
            greencalculfft_(&nx,&ny,&nz,&nx2,&ny2,&nz2,&r->nxy2,&nxm,&nym,&nzm,
                (REAL *)a->aretecube,(REAL *)a->k0,xx,xy,xz,yy,yz,zz,
                r->vectx,r->vecty,r->vectz,&planb);
    } else {
        greencalculquadfft_(&nx,&ny,&nz,&nx2,&ny2,&nz2,&r->nxy2,&nxm,&nym,&nzm,
            &nquad,&s->tolinit,(REAL *)a->aretecube,(REAL *)a->k0,xx,xy,xz,yy,yz,zz,
            r->vectx,r->vecty,r->vectz,&planb);
    }
#endif
    if(*a->nstop==-1) {
        size_t n=sizeof "Calculation cancelled after FFT Green function!"-1;
        if(n>a->infostr_len) n=a->infostr_len;
        memset(a->infostr,' ',a->infostr_len);
        memcpy(a->infostr,"Calculation cancelled after FFT Green function!",n);
        return 1;
    }
    r->vector_green_ready=1;
    return 0;
}


/* Incorporated cdmlib_solverphase */
#include <stdint.h>
/* Solver-phase declarations integrated in this translation unit. */

#include <string.h>
#include <math.h>
#include <stdio.h>





//-------------------------------------------

static void cdmlib_solverphase__fail(const cdmlib_abi_args_c99 *a,const char *message)
{
    const size_t n0=strlen(message);
    const size_t n=n0<a->infostr_len?n0:a->infostr_len;
    memset(a->infostr,' ',a->infostr_len);
    memcpy(a->infostr,message,n);
    *a->nstop=1;
}
//-------------------------------------------

int cdmlib_solve_from_abi_c99(const cdmlib_abi_args_c99 *a,
                              const cdmlib_initphase_c99 *s,
                              const cdmlib_config_c99 *c,
                              const cdmlib_entry_runtime_c99 *r,
                              int nbsphere,int ndipole)
{
    if(!a||!s||!c||!r||!a->nstop||!a->infostr) return 1;
    if(!(c->nrig>=0&&c->nrig<=5)||
       !(c->nprecon==0||c->nprecon==1||c->nprecon==2)||
       !(c->ninitest==0||c->ninitest==1||c->ninitest==2||
         c->ninitest==3||c->ninitest==4||c->ninitest==5)) {
        cdmlib_solverphase__fail(a,"C99 nrig branch not connected for these options");
        return 2;
    }
    if(c->nrig==1||c->nrig==2) {
        if(!a->ff0||!a->polarisa||!a->ffloc||!a->ff||!a->niso) {
            cdmlib_solverphase__fail(a,"Invalid Born approximation buffers");
            return 1;
        }
        const int nmax=s->nmax;
        const int niso=(int)*a->niso;
        const int ndipoles=nbsphere;
        int nstop=0;
        if(c->nrig==1) {
            ifdda_renormalized_born_all_cuda_(
                (const COMPLEX *)a->ff0,(const COMPLEX *)a->polarisa,
                (COMPLEX *)a->ffloc,(COMPLEX *)a->ff,
                &nmax,&niso,&ndipoles,&nstop);
        } else {
            if(!a->epsilon||!a->aretecube||!a->k0) {
                cdmlib_solverphase__fail(a,"Invalid Born material buffers");
                return 1;
            }
            ifdda_born_all_cuda_(
                (const COMPLEX *)a->ff0,(const COMPLEX *)a->epsilon,
                (const COMPLEX *)a->polarisa,(COMPLEX *)a->ffloc,
                (COMPLEX *)a->ff,&nmax,&niso,&ndipoles,
                (const REAL *)a->aretecube,(const REAL *)a->k0,&nstop);
        }
        *a->nstop=(cdmlib_ip)nstop;
        if(nstop!=0) {
            cdmlib_solverphase__fail(a,
                c->nrig==1?"CUDA renormalized Born field calculation failed":
                "CUDA Born field calculation failed");
            return 1;
        }
        if(a->nloop) *a->nloop=0;
        if(a->ncompte) *a->ncompte=0;
        if(a->tol1) *a->tol1=0;
        return 0;
    }
    if(c->nrig==3||c->nrig==4) {
        if(!a->xi||!a->xr||!a->ff0||!a->ffloc||!a->ff||
           !a->polarisa||!a->niso||!a->ffttensorxx||!a->ffttensorxy||
           !a->ffttensorxz||!a->ffttensoryy||!a->ffttensoryz||
           !a->ffttensorzz||!a->tabdip) {
            cdmlib_solverphase__fail(a,"Invalid Born-series buffers");
            return 1;
        }
        const int n3=3*nbsphere;
        const int niso=(int)*a->niso;
        const int nmax=s->nmax;
        COMPLEX *xr=(COMPLEX *)a->xr;
        COMPLEX *xi=(COMPLEX *)a->xi;
        const COMPLEX *ff0=(const COMPLEX *)a->ff0;
        int nstop=0;
        if(c->nrig==3) {
            const int negate=0;
            ifdda_vector_copy_all_cuda_(ff0,xr,&n3,&negate,&nstop);
            if(nstop==0) produitpolavect_((const COMPLEX *)a->polarisa,
                &niso,&nmax,&nbsphere,xr,xi);
            const int negate_values=1;
            if(nstop==0) ifdda_vector_negate_all_cuda_(
                xi,&n3,&nstop);
        } else {
            ifdda_vector_zero_all_cuda_(xr,&n3,&nstop);
            if(nstop==0) produitpolavect_((const COMPLEX *)a->polarisa,
                &niso,&nmax,&nbsphere,ff0,xi);
            const int negate_values=1;
            if(nstop==0) ifdda_vector_negate_all_cuda_(
                xi,&n3,&nstop);
        }
        if(nstop==0) {
            int ntotalm=s->ntotalm,ntotal=r->ntotal,ndipole=ndipole;
            int nxm=s->nxm,nym=s->nym,nzm=s->nzm;
            int nx=r->nx,ny=r->ny,nz=r->nz,nx2=r->nx2,ny2=r->ny2;
            int nxy2=r->nxy2,nz2=r->nz2;
            long long planf=(long long)r->planf,planb=(long long)r->planb;
            produitfftmatvect3_((COMPLEX *)a->ffttensorxx,
                (COMPLEX *)a->ffttensorxy,(COMPLEX *)a->ffttensorxz,
                (COMPLEX *)a->ffttensoryy,(COMPLEX *)a->ffttensoryz,
                (COMPLEX *)a->ffttensorzz,r->vectx,r->vecty,r->vectz,
                (int *)a->tabdip,&ntotalm,&ntotal,&nmax,&ndipole,
                &nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,
                xi,xr,(const int64_t *)&planf,(const int64_t *)&planb);
        }
        if(nstop==0 && c->nrig==4)
            nstop=cdmlib_rytov_vector_field_c99(
                ff0,xr,(COMPLEX *)a->ffloc,nbsphere);
        else if(nstop==0)
            memcpy(a->ffloc,xr,(size_t)n3*sizeof(COMPLEX));
        if(nstop==0)
            produitpolavect_((const COMPLEX *)a->polarisa,&niso,&nmax,
                &nbsphere,(const COMPLEX *)a->ffloc,(COMPLEX *)a->ff);
        *a->nstop=(cdmlib_ip)nstop;
        if(nstop!=0) {
            cdmlib_solverphase__fail(a,
                c->nrig==4?"Rytov vector field is outside numerical range":
                "CUDA Born-series field calculation failed");
            return 1;
        }
        if(a->nloop) *a->nloop=0;
        if(a->ncompte) *a->ncompte=0;
        if(a->tol1) *a->tol1=0;
        return 0;
    }
    if(c->nrig==5) {
        if(!a->xi||!a->xr||!a->ff0||!a->ffloc||!a->ff||
           !a->epsilon||!a->polarisa||!a->niso||!a->aretecube||!a->k0||
           !a->ffttensorxx||!a->ffttensorxy||!a->ffttensorxz||
           !a->ffttensoryy||!a->ffttensoryz||!a->ffttensorzz||
           !a->tabdip) {
            cdmlib_solverphase__fail(a,"Invalid macroscopic Rytov buffers");
            return 1;
        }
        const int nmax=s->nmax,niso=(int)*a->niso,ndipoles=nbsphere;
        const int n3=3*nbsphere;
        int nstop=0;
        ifdda_rytov_macro_prepare_all_cuda_(
            (const COMPLEX *)a->ff0,(const COMPLEX *)a->epsilon,
            (const COMPLEX *)a->polarisa,&s->eps0,(COMPLEX *)a->ff,
            (COMPLEX *)a->xr,(COMPLEX *)a->xi,&nmax,&niso,&ndipoles,
            (const REAL *)a->aretecube,(const REAL *)a->k0,&nstop);
        if(nstop==0) {
            int ntotalm=s->ntotalm,ntotal=r->ntotal,ndipole=ndipole;
            int nxm=s->nxm,nym=s->nym,nzm=s->nzm;
            int nx=r->nx,ny=r->ny,nz=r->nz,nx2=r->nx2,ny2=r->ny2;
            int nxy2=r->nxy2,nz2=r->nz2;
            long long planf=(long long)r->planf,planb=(long long)r->planb;
            produitfftmatvect3_((COMPLEX *)a->ffttensorxx,
                (COMPLEX *)a->ffttensorxy,(COMPLEX *)a->ffttensorxz,
                (COMPLEX *)a->ffttensoryy,(COMPLEX *)a->ffttensoryz,
                (COMPLEX *)a->ffttensorzz,r->vectx,r->vecty,r->vectz,
                (int *)a->tabdip,&ntotalm,&ntotal,&nmax,&ndipole,
                &nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,
                (COMPLEX *)a->xi,(COMPLEX *)a->xr,
                (const int64_t *)&planf,(const int64_t *)&planb);
        }
        if(nstop==0)
            nstop=cdmlib_rytov_vector_field_c99(
                (const COMPLEX *)a->ff,(const COMPLEX *)a->xr,
                (COMPLEX *)a->ffloc,nbsphere);
        if(nstop==0)
            produitpolavect_((const COMPLEX *)a->polarisa,&niso,&nmax,
                &nbsphere,(const COMPLEX *)a->ffloc,(COMPLEX *)a->ff);
        (void)n3;
        *a->nstop=(cdmlib_ip)nstop;
        if(nstop!=0) {
            cdmlib_solverphase__fail(a,
                "Rytov macroscopic field calculation failed");
            return 1;
        }
        if(a->nloop) *a->nloop=0;
        if(a->ncompte) *a->ncompte=0;
        if(a->tol1) *a->tol1=0;
        return 0;
    }
    if(!a->xi||!a->xr||!a->wrk||!a->ff||!a->ff0||!a->ffloc||!a->polarisa||
       !a->tabdip||!a->nlar||!a->tol1||!a->nloop||!a->ncompte||!a->nlim||
       !a->methodeit) {
        cdmlib_solverphase__fail(a,"Invalid cdmlib solver buffers");
        return 1;
    }

    const int nbsphere3=r->nbsphere3;
    COMPLEX *xi=(COMPLEX *)a->xi;
    const COMPLEX *ff0=(const COMPLEX *)a->ff0;
    int ntotalm=s->ntotalm,ntotal=r->ntotal,ldabi=s->ldabi;
    int nlar=(int)*a->nlar,nmax=s->nmax,nbs=nbsphere,ndip=ndipole;
    int nxm=s->nxm,nym=s->nym,nzm=s->nzm,nx=r->nx,ny=r->ny,nz=r->nz;
    int nx2=r->nx2,ny2=r->ny2,nxy2=r->nxy2,nz2=r->nz2;
    int niso=(int)*a->niso,nlim=(int)*a->nlim,nloop=0,ncompte=0,nstop=0;
    int nbs3=nbsphere3;
    REAL tol=s->tolinit,tol1=0.0;
    long long planXf=(long long)r->planXf,planXb=(long long)r->planXb;
    long long planYf=(long long)r->planYf,planYb=(long long)r->planYb;
    long long planZf=(long long)r->planZf,planZb=(long long)r->planZb;
    long long planf=(long long)r->planf,planb=(long long)r->planb;
    long long plan2f=(long long)r->plan2f,plan2b=(long long)r->plan2b;

    if(c->ninitest==0 || c->ninitest==1) {
        for(int i=0;i<nbsphere3;++i)
            xi[i]=c->ninitest==0?0.0+0.0*I:ff0[i];
    } else if(c->ninitest==2) {
        if(!a->ffttensorxx||!a->ffttensorxy||!a->ffttensorxz||
           !a->ffttensoryy||!a->ffttensoryz||!a->ffttensorzz||
           !a->polarisa||!a->niso) {
            cdmlib_solverphase__fail(a,"Invalid Rytov initialization buffers");
            return 1;
        }
        COMPLEX *xr=(COMPLEX *)a->xr;
        const COMPLEX *alpha=(const COMPLEX *)a->polarisa;
        for(int i=0;i<nbsphere3;++i) xr[i]=0.0;
        for(int i=0;i<nbs;++i) {
            const size_t k=(size_t)3*(size_t)i;
            if(niso==1) {
                const COMPLEX p=alpha[i];
                xi[k]=-p*ff0[k]; xi[k+1]=-p*ff0[k+1]; xi[k+2]=-p*ff0[k+2];
            } else {
                xi[k]=-(alpha[i]*ff0[k]+alpha[i+(size_t)nmax*3]*ff0[k+1]+
                        alpha[i+(size_t)nmax*6]*ff0[k+2]);
                xi[k+1]=-(alpha[i+(size_t)nmax]*ff0[k]+
                          alpha[i+(size_t)nmax*4]*ff0[k+1]+
                          alpha[i+(size_t)nmax*7]*ff0[k+2]);
                xi[k+2]=-(alpha[i+(size_t)nmax*2]*ff0[k]+
                          alpha[i+(size_t)nmax*5]*ff0[k+1]+
                          alpha[i+(size_t)nmax*8]*ff0[k+2]);
            }
        }
        produitfftmatvect3_((COMPLEX *)a->ffttensorxx,
            (COMPLEX *)a->ffttensorxy,(COMPLEX *)a->ffttensorxz,
            (COMPLEX *)a->ffttensoryy,(COMPLEX *)a->ffttensoryz,
            (COMPLEX *)a->ffttensorzz,r->vectx,r->vecty,r->vectz,
            (int *)a->tabdip,&ntotalm,&ntotal,&nmax,&ndip,&nxm,&nym,&nzm,
            &nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,xi,xr,(const int64_t *)&planf,(const int64_t *)&planb);
        if(cdmlib_rytov_vector_field_c99(ff0,xr,xi,nbs)) {
            cdmlib_solverphase__fail(a,"Rytov vector field is outside numerical range");
            return 1;
        }
    } else if(c->ninitest==3) {
        if(!a->xs||!a->ys||!a->zs||!a->aretecube||!a->k0||!a->w0||
           !a->e0||!a->ss||!a->pp||!a->theta||!a->phi||!a->xgaus||
           !a->ygaus||!a->zgaus||!a->beam||!a->epsilon||!a->nfft2d||
           !a->eimagex||!a->eimagey||!a->eimagez) {
            cdmlib_solverphase__fail(a,"Invalid BPM initialization buffers");
            return 1;
        }
        const int resident_mode=0;
        beampropagation_((REAL *)a->xs,(REAL *)a->ys,(REAL *)a->zs,
            (REAL *)a->aretecube,(REAL *)a->k0,(REAL *)a->w0,
            (COMPLEX *)a->e0,(REAL *)a->ss,(REAL *)a->pp,
            (REAL *)a->theta,(REAL *)a->phi,(REAL *)a->xgaus,
            (REAL *)a->ygaus,(REAL *)a->zgaus,a->beam,
            (COMPLEX *)a->epsilon,(int *)a->niso,&ndip,&nx,&ny,&nz,
            &nxm,&nym,&nzm,&nmax,(int *)a->nfft2d,
            (COMPLEX *)a->eimagex,(COMPLEX *)a->eimagey,
            (COMPLEX *)a->eimagez,(COMPLEX *)a->ff0,xi,
            (COMPLEX *)a->ff,(const int64_t *)&plan2f,(const int64_t *)&plan2b,&nstop,a->infostr,&resident_mode,a->beam_len,
            a->infostr_len);
        if(nstop!=0) {
            *a->nstop=(cdmlib_ip)nstop;
            return 1;
        }
    }

    *a->nstop=0;
    if(c->nprecon==0) {
        // CUDA/main0.f90 and CUDA/main0.cpp now reach the SAME C++
        // inverserig_all_cuda_driver_ entry point. The six arguments are
        // CPU identity keys only; inverserig retrieves the six resident GPU
        // Green pointers and passes them directly to the FFT products.
        // Allocate genuine GPU FFT scratch via cuda_data, not CPU arrays
        // mislabeled as CUCOMPLEX pointers in the old legacy bridge.
        if(ntotal<=0 || size_t(ntotal)>std::numeric_limits<size_t>::max()/3) {
            cdmlib_solverphase__fail(a,"Invalid CUDA inverse FFT workspace size");
            return 1;
        }
        CUCOMPLEX *fft_work=get_device_vectors(size_t(3)*size_t(ntotal));
        if(!fft_work) {
            cdmlib_solverphase__fail(a,"Unable to acquire CUDA inverse FFT workspace");
            return 1;
        }
        const int optimized_product=s->nproche!=-1 ? 1 : 0;
        const int no_preconditioner=0;
        int cuda_guess_mode=-1;
        if(c->ninitest==0 || c->ninitest==1)
            cuda_guess_mode=c->ninitest;
        else if(c->ninitest==2)
            cuda_guess_mode=2;
        else if(c->ninitest==3)
            cuda_guess_mode=4;
        inverserig_all_cuda_driver_(
            (const COMPLEX *)a->ffttensorxx,(const COMPLEX *)a->ffttensorxy,
            (const COMPLEX *)a->ffttensorxz,(const COMPLEX *)a->ffttensoryy,
            (const COMPLEX *)a->ffttensoryz,(const COMPLEX *)a->ffttensorzz,
            fft_work,fft_work+ntotal,fft_work+size_t(2)*ntotal,
            optimized_product ? nullptr : (const int *)a->tabdip,
            &ntotalm,&ntotal,&ldabi,&nlar,&nmax,&ndip,&nxm,&nym,&nzm,
            &nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,&nbs,&nbs3,
            xi,(COMPLEX *)a->xr,(COMPLEX *)a->wrk,
            (COMPLEX *)a->ff,(const COMPLEX *)a->ff0,
            (COMPLEX *)a->ffloc,(const COMPLEX *)a->polarisa,&niso,
            a->methodeit,&tol,&tol1,&nloop,&ncompte,&nlim,
            &planXf,&planXb,&planYf,&planYb,&planZf,&planZb,&planf,&planb,
            &optimized_product,&cuda_guess_mode,&no_preconditioner,
            &nstop,a->infostr,a->methodeit_len,a->infostr_len);
    } else if(c->nprecon==1||c->nprecon==2) {
        // Mirror CUDA/cdmlib.f90: the right-Chan resident solver must know
        // whether to construct the Born initial iterate on the device.
        // Without this hand-off it defaults to mode -1 and discards the
        // requested ninitest=1, producing the ninitest=0 solution instead.
        int cuda_guess_mode=-1;
        if(c->nrig==0 && c->nprecon==2) {
            if(c->ninitest==0 || c->ninitest==1)
                cuda_guess_mode=c->ninitest;
            else if(c->ninitest==2)
                cuda_guess_mode=3;
            else if(c->ninitest==3)
                cuda_guess_mode=4;
        }
        ifdda_set_initial_guess_mode_all_cuda_(cuda_guess_mode);
        if(!a->ffprecon||!a->localfieldx||!a->localfieldy||!a->localfieldz||
           !a->macroscopicfieldx||!a->macroscopicfieldy||
           !a->macroscopicfieldz||!a->sdetnn) {
            cdmlib_solverphase__fail(a,"Invalid preconditioner buffers");
            return 1;
        }
        const int64_t abi_planf = static_cast<int64_t>(planf);
        const int64_t abi_planb = static_cast<int64_t>(planb);
        if(s->nproche==-1) {
            auto fn=c->nprecon==1?inverserigpg_:inverserigpd_;
            fn((COMPLEX *)a->ffttensorxx,(COMPLEX *)a->ffttensorxy,
            (COMPLEX *)a->ffttensorxz,(COMPLEX *)a->ffttensoryy,
            (COMPLEX *)a->ffttensoryz,(COMPLEX *)a->ffttensorzz,
            r->vectx,r->vecty,r->vectz,(int *)a->tabdip,&ntotalm,&ntotal,&ldabi,
            &nlar,&nmax,&ndip,&nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,
            &nbs,&nbs3,xi,(COMPLEX *)a->xr,(COMPLEX *)a->wrk,
            (COMPLEX *)a->ff,(COMPLEX *)a->ff0,
            (COMPLEX *)a->ffloc,(COMPLEX *)a->ffprecon,
            (COMPLEX *)a->polarisa,&niso,a->methodeit,&tol,&tol1,
            &nloop,&ncompte,&nlim,(REAL *)a->k0,(REAL *)a->aretecube,
            (COMPLEX *)a->localfieldx,(COMPLEX *)a->localfieldy,
            (COMPLEX *)a->localfieldz,(COMPLEX *)a->macroscopicfieldx,
            (COMPLEX *)a->macroscopicfieldy,(COMPLEX *)a->macroscopicfieldz,
            (COMPLEX *)a->sdetnn,
            &abi_planf,&abi_planb,&nstop,a->infostr,
            a->methodeit_len,a->infostr_len);
        } else {
            auto fn=c->nprecon==1?inverserigoptpg_:inverserigoptpd_;
            fn((COMPLEX *)a->ffttensorxx,(COMPLEX *)a->ffttensorxy,
            (COMPLEX *)a->ffttensorxz,(COMPLEX *)a->ffttensoryy,
            (COMPLEX *)a->ffttensoryz,(COMPLEX *)a->ffttensorzz,
            r->vectx,r->vecty,r->vectz,(int *)a->tabdip,&ndip,
            &ntotalm,&ntotal,&ldabi,&nlar,&nmax,
            &nxm,&nym,&nzm,&nx,&ny,&nz,&nx2,&ny2,&nxy2,&nz2,&nbs,&nbs3,
            xi,(COMPLEX *)a->xr,(COMPLEX *)a->wrk,
            (COMPLEX *)a->ff,(COMPLEX *)a->ff0,
            (COMPLEX *)a->ffloc,(COMPLEX *)a->ffprecon,
            (COMPLEX *)a->polarisa,&niso,a->methodeit,&tol,&tol1,
            &nloop,&ncompte,&nlim,(REAL *)a->k0,(REAL *)a->aretecube,
            (COMPLEX *)a->localfieldx,(COMPLEX *)a->localfieldy,
            (COMPLEX *)a->localfieldz,(COMPLEX *)a->macroscopicfieldx,
            (COMPLEX *)a->macroscopicfieldy,(COMPLEX *)a->macroscopicfieldz,
            (COMPLEX *)a->sdetnn,
            &abi_planf,&abi_planb,&nstop,a->infostr,
            a->methodeit_len,a->infostr_len);
        }
    } else {
        cdmlib_solverphase__fail(a,"C99 solver branch not connected for these options");
        return 2;
    }
    *a->nstop=(cdmlib_ip)nstop;
    *a->nloop=(cdmlib_ip)nloop;
    *a->ncompte=(cdmlib_ip)ncompte;
    *a->tol1=(cdmlib_real)tol1;
    if(nstop==1) return 1;
    if(tol1>=s->tolinit) {
        cdmlib_solverphase__fail(a,"Converge do not reach!");
        return 1;
    }
    if(nstop==-1) {
        cdmlib_solverphase__fail(a,"Calculation cancelled after iterative method!");
        return 1;
    }
    return 0;
}
//-------------------------------------------

int cdmlib_compute_cross_sections_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_entry_runtime_c99 *r,
    int nbsphere)
{
    if(!a||!r||!a->nsection||!a->cext||!a->cabs||!a->csca||!a->ff0||
       !a->ff||!a->ffloc||!a->k0) return 1;
    if(*a->nsection!=1) return 0;
    if(nbsphere>0 && nbsphere<=std::numeric_limits<int>::max()/3) {
        const int dipole_count=nbsphere;
        const REAL k0=(REAL)*a->k0;
        const REAL intensity=(REAL)r->I0;
        REAL cext=0.0,cabs=0.0,csca=0.0;
        const int gpu_status=ifdda_compute_cross_sections_cuda_(
            (const COMPLEX *)a->ff0,(const COMPLEX *)a->ff,
            (const COMPLEX *)a->ffloc,&dipole_count,&k0,&intensity,
            &cext,&cabs,&csca);
        if(gpu_status==0) {
            *a->cext=(cdmlib_real)cext;
            *a->cabs=(cdmlib_real)cabs;
            *a->csca=(cdmlib_real)csca;
            return 0;
        }
        std::fprintf(stderr,
            "CUDA cross-section reduction failed (status %d); using CPU fallback.\n",
            gpu_status);
    }
    const REAL k0=(REAL)*a->k0;
    const REAL k03=k0*k0*k0;
    REAL cext=0.0,cabs=0.0;
    const COMPLEX *ff0=(const COMPLEX *)a->ff0;
    const COMPLEX *ff=(const COMPLEX *)a->ff;
    const COMPLEX *ffloc=(const COMPLEX *)a->ffloc;
    for(int i=0;i<nbsphere;++i) {
        const size_t k=(size_t)3*(size_t)i;
        cext+=cimag(conj(ff0[k])*ff[k]+conj(ff0[k+1])*ff[k+1]+
                    conj(ff0[k+2])*ff[k+2]);
        cabs+=cimag(ff[k]*conj(ffloc[k])+ff[k+1]*conj(ffloc[k+1])+
                    ff[k+2]*conj(ffloc[k+2]));
        cabs-=(2.0/3.0)*k03*creal(ff[k]*conj(ff[k])+ff[k+1]*conj(ff[k+1])+
                                  ff[k+2]*conj(ff[k+2]));
    }
    cext=4.0*3.141592653589793238462643383279502884*k0*cext/r->I0;
    cabs=4.0*3.141592653589793238462643383279502884*k0*cabs/r->I0;
    const REAL csca=cext-cabs;
    if(cabs<=cext*1.0e-12) cabs=0.0;
    *a->cext=(cdmlib_real)cext;
    *a->cabs=(cdmlib_real)cabs;
    *a->csca=(cdmlib_real)csca;
    return 0;
}
//-------------------------------------------

int cdmlib_map_solved_fields_from_abi_c99(
    const cdmlib_abi_args_c99 *a,const cdmlib_initphase_c99 *s,
    const cdmlib_config_c99 *c,int ndipole)
{
    if(!a||!s||!c||!a->tabdip) return 1;
    if(c->nlocal==1) {
        if(!a->localfield||!a->localfieldx||!a->localfieldy||
           !a->localfieldz||!a->ffloc) return 1;
        const int *tabdip=(const int *)a->tabdip;
        const COMPLEX *ffloc=(const COMPLEX *)a->ffloc;
        REAL *modulus=(REAL *)a->localfield;
        COMPLEX *fx=(COMPLEX *)a->localfieldx;
        COMPLEX *fy=(COMPLEX *)a->localfieldy;
        COMPLEX *fz=(COMPLEX *)a->localfieldz;
        COMPLEX *wrk=(COMPLEX *)a->wrk;
        for(int i=0;i<ndipole;++i) {
            const int k=tabdip[i];
            if(k==0) {
                if(c->nmat==2) {
                    wrk[(size_t)i]=0.0+0.0*I;
                    wrk[(size_t)i+(size_t)s->nmax]=0.0+0.0*I;
                    wrk[(size_t)i+2*(size_t)s->nmax]=0.0+0.0*I;
                    wrk[(size_t)i+3*(size_t)s->nmax]=0.0+0.0*I;
                }
                continue;
            }
            const size_t src=(size_t)3*(size_t)(k-1);
            const COMPLEX ex=ffloc[src],ey=ffloc[src+1],ez=ffloc[src+2];
            const size_t dst=(size_t)(k-1);
            fx[dst]=ex; fy[dst]=ey; fz[dst]=ez;
            modulus[dst]=sqrt(creal(ex*conj(ex)+ey*conj(ey)+ez*conj(ez)));
            if(c->nmat==2) {
                wrk[(size_t)i]=ex; wrk[(size_t)i+(size_t)s->nmax]=ey;
                wrk[(size_t)i+2*(size_t)s->nmax]=ez;
                wrk[(size_t)i+3*(size_t)s->nmax]=modulus[dst]+0.0*I;
            }
        }
        if(c->nmat==0) {
            static const char *const names[4]={"localfieldx.mat",
                "localfieldy.mat","localfieldz.mat","localfield.mat"};
            FILE *files[4]={NULL,NULL,NULL,NULL};
            for(int j=0;j<4;++j) {
                files[j]=fopen(names[j],"w");
                if(!files[j]) {
                    for(int k=0;k<4;++k) if(files[k]) fclose(files[k]);
                    cdmlib_solverphase__fail(a,"Could not write local field output files");
                    return 1;
                }
            }
            for(int i=0;i<ndipole;++i) {
                const int k=tabdip[i];
                if(k==0) {
                    if(fprintf(files[0]," 0.0000000000000000 0.0000000000000000\n")<0 ||
                       fprintf(files[1]," 0.0000000000000000 0.0000000000000000\n")<0 ||
                       fprintf(files[2]," 0.0000000000000000 0.0000000000000000\n")<0 ||
                       fprintf(files[3]," 0.0000000000000000\n")<0) {
                        for(int j=0;j<4;++j) fclose(files[j]);
                        cdmlib_solverphase__fail(a,"Error writing local field output files");
                        return 1;
                    }
                } else {
                    const size_t k0=(size_t)(k-1);
                    if(fprintf(files[0]," %.16E %.16E\n",creal(fx[k0]),cimag(fx[k0]))<0 ||
                       fprintf(files[1]," %.16E %.16E\n",creal(fy[k0]),cimag(fy[k0]))<0 ||
                       fprintf(files[2]," %.16E %.16E\n",creal(fz[k0]),cimag(fz[k0]))<0 ||
                       fprintf(files[3]," %.16E\n",modulus[k0])<0) {
                        for(int j=0;j<4;++j) fclose(files[j]);
                        cdmlib_solverphase__fail(a,"Error writing local field output files");
                        return 1;
                    }
                }
            }
            for(int j=0;j<4;++j) {
                if(fclose(files[j])!=0) {
                    cdmlib_solverphase__fail(a,"Error closing local field output files");
                    return 1;
                }
            }
        }
    }
    if(c->nmacro==1) {
        int64_t group_idnf=0;
        if(!a->tabdip||!a->aretecube||!a->k0||!a->ffloc||!a->epsilon||
           !a->niso||!a->macroscopicfield||!a->macroscopicfieldx||
           !a->macroscopicfieldy||!a->macroscopicfieldz||!a->wrk||!a->nlar)
            return 1;
        savemacroscopicfield_((const int *)a->nmat,(const int *)a->ndipole,
            &s->nmax,(const int *)a->nlar,&c->testgreen,
            (const int *)a->tabdip,(const REAL *)a->aretecube,
            (const REAL *)a->k0,(const COMPLEX *)a->ffloc,
            (const COMPLEX *)a->epsilon,(const int *)a->niso,
            (REAL *)a->macroscopicfield,
            (COMPLEX *)a->macroscopicfieldx,
            (COMPLEX *)a->macroscopicfieldy,
            (COMPLEX *)a->macroscopicfieldz,
            (COMPLEX *)a->wrk,&group_idnf);
    }
    return 0;
}
//---------------------------
int donothing(int data){
    if(data==1)
        data=data;
    if(data==2)
        data=data;
    if(data==3)
        data=data;


    return 0;
}
//--------------------------------------

}
