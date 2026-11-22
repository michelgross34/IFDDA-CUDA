#include "cdmlib_cpp_abi.hpp"
#define IFDDA_CUDA_CALLERS_MICROSCONFOCAL_LEGACY_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_MICROSCONFOCAL_LEGACY_CUDA
#include "microsconfocal_CUDA.h"
#include "cuda_data.h"
#include "cuda_utils.h"
/*
 * C99 implementation of cdmlib/microsconfocal.f90. The public gfortran ABI
 * entry is microsconfocal_; numerical branches and the detector pipeline are
 * factored into helpers so they can be compared independently with Fortran.
 */

#include <cmath>
#include <algorithm>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define I (COMPLEX(0.0, 1.0))

static void microsconfocal_set_info(char *dst,size_t length,const char *message)
{
    size_t n=strlen(message);
    if(n>length)n=length;
    memset(dst,' ',length);
    memcpy(dst,message,n);
}

/* Translate the confocal input checks and Z-stack normalization. */
/* Translated nrig=1..5 field dispatch shared with microsconfocal's scheduler. */
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
    COMPLEX *FF, COMPLEX *FFloc)
{
    const int count = 3 * *nbsphere;
    if (*nrig == 1) {
        memcpy(FFloc, FF0, (size_t)count * sizeof *FFloc);
        return microsbf_polarization_fields_cuda(
            nbsphere,nmax,niso,polarisa,FFloc,FF);
    }
    if (*nrig == 2)
        return microsbf_nrig2_fields_c99(nbsphere,nmax,niso,epsilon,polarisa,
            FF0,aretecube,k0,FFloc,FF);
    if (*nrig == 3)
        return microsbf_nrig3_fields_c99(nbsphere,nmax,niso,ntotalm,ntotal,
            ndipole,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,tabdip,
            tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],
            vect[0],vect[1],vect[2],planf,planb,polarisa,FF0,XI,XR,FF,FFloc);
    if (*nrig == 4)
        return microsbf_nrig4_fields_c99(nbsphere,nmax,niso,ntotalm,ntotal,
            ndipole,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,tabdip,
            tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],
            vect[0],vect[1],vect[2],planf,planb,polarisa,FF0,XI,XR,FF,FFloc);
    if (*nrig == 5)
        return microsbf_nrig5_fields_host_c99(nbsphere,nmax,niso,ntotalm,ntotal,
            ndipole,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,tabdip,
            tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],
            vect[0],vect[1],vect[2],planf,planb,polarisa,epsilon,eps0,
            aretecube,k0,FF0,XI,XR,FF,FFloc);
    return 1;
}

/* Construct the incident plane wave and seed XI before the nrig dispatch. */
/* Classical far-field diffraction used by the confocal microscope when
 * Quick Lens is disabled. Ediffkzneg keeps the Fortran (nfft,nfft,3) layout. */
/* Select the existing C99 Quick Lens kernel or the translated direct sum. */
extern "C" int microsconfocal_diffraction_c99(
    const int *nquicklens,const int *nx,const int *ny,const int *nz,
    const int *nxm,const int *nym,const int *nzm,const int *nfft2d,
    int *tabfft2,const int *nside,int *imaxk0,const int *nbsphere,
    const REAL *k0,const REAL *numaperk,const REAL *aretecube,
    const REAL *xs,const REAL *ys,const REAL *zs,COMPLEX *Efourierx,
    COMPLEX *Efouriery,COMPLEX *Efourierz,const COMPLEX *FF,
    REAL *deltakx,REAL *deltaky,COMPLEX *Ediffkzneg,
    const int64_t *plan2f,const int64_t *plan2b,int *nstop,char *infostr,
    size_t infostr_len)
{
    if (*nquicklens == 1) {
        diffractefft2dlens_(nx,ny,nz,nxm,nym,nzm,nfft2d,tabfft2,k0,xs,ys,zs,
            aretecube,Efourierx,Efouriery,Efourierz,FF,imaxk0,
            deltakx,deltaky,Ediffkzneg,numaperk,nside,plan2f,plan2b,nstop,
            infostr,infostr_len);
        return *nstop == 1 ? 1 : 0;
    }
    *deltaky = *deltakx;
    *nstop = microsconfocal_classical_diffraction_c99(nfft2d,imaxk0,nside,
        nbsphere,k0,numaperk,deltakx,deltaky,xs,ys,zs,FF,Ediffkzneg);
    return *nstop;
}

/* Dispatch the exact vector solver and beam propagation modes.  The caller
 * supplies the already translated incident wave-vector components; the
 * per-incidence angle loop remains in the ABI-level scheduler. */
static int microsconfocal_scalar_fields_c99(
    const int *,const int *,const int *,const int *,const int *,const int *,
    const int *,const int *,const int *,const int *,const int *,const int *,
    const int *,const int *,const int *,const int *,const int *,const int *,
    const int *,const int *,const int *,COMPLEX *,COMPLEX *,
    int *,COMPLEX *,COMPLEX *,const REAL *,
    const COMPLEX *,const REAL *,const REAL *,const REAL *,
    const REAL *,REAL *,int *,int *,const int64_t *,const int64_t *,
    COMPLEX *,COMPLEX *,COMPLEX *,COMPLEX *,
    COMPLEX *,COMPLEX *,COMPLEX *,char *,int *,char *,
    size_t,size_t);
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
    size_t methodeit_len, size_t infostr_len)
{
    if (*nrig == 0)
        return microsbf_nrig0_fields_c99(nproche,ntotalm,ntotal,ldabi,nlar,
            nmax,ndipole,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,nbsphere,
            nbsphere3,nlim,niso,tensor[0],tensor[1],tensor[2],tensor[3],
            tensor[4],tensor[5],vect[0],vect[1],vect[2],tabdip,XI,XR,wrk,FF,
            FF0,FFloc,polarisa,methodeit,tolinit,tol1,nloop,ncompte,
            planXf,planXb,planYf,planYb,planZf,planZb,planf,planb,nstop,
            infostr,methodeit_len,infostr_len);
    if (*nrig == 6 || *nrig == 7)
        return microsbf_nrig67_fields_c99(nrig,nbsphere,nmax,niso,ndipole,
            nx,ny,nz,nxm,nym,nzm,nfft2d,xs,ys,zs,aretecube,k0,w0,E0,ss,pp,
            kxinc,kyinc,epsilon,polarisa,Efourierx,Efouriery,Efourierz,FF0,
            FFloc,FF,plan2f,plan2b,nstop,infostr,infostr_len);
    if (*nrig == 8 || *nrig == 9)
        return microsconfocal_scalar_fields_c99(nrig,nproche,ntotalm,ntotal,
            ldabi,nlar,nmax,ndipole,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,
            nbsphere,nlim,niso,tensor[0],vect[0],tabdip,tensor[3],tensor[5],
            k0,E0,psi,kxinc,kyinc,tolinit,tol1,nloop,ncompte,planf,planb,
            XI,XR,wrk,FF,FF0,FFloc,polarisa,methodeit,nstop,infostr,
            methodeit_len,infostr_len);
    return 1;
}

/* Scalar solve modes use the confocal polarization angle psiinc.  Keep the
 * component construction and normalization in the same order as Fortran. */
int microsconfocal_scalar_fields_c99(
    const int *nrig,const int *nproche,
    const int *ntotalm,const int *ntotal,const int *ldabi,const int *nlar,
    const int *nmax,const int *ndipole,const int *nxm,const int *nym,
    const int *nzm,const int *nx,const int *ny,const int *nz,
    const int *nx2,const int *ny2,const int *nxy2,const int *nz2,
    const int *nbsphere,const int *nlim,const int *niso,
    COMPLEX *tensor_xx,COMPLEX *vectx,int *tabdip,
    COMPLEX *tensor_yy_scalar,COMPLEX *tensor_zz_scalar,
    const REAL *k0,const COMPLEX *E0,const REAL *psi,
    const REAL *kxinc,const REAL *kyinc,const REAL *tolinit,
    REAL *tol1,int *nloop,int *ncompte,const int64_t *planf,
    const int64_t *planb,COMPLEX *XI,COMPLEX *XR,
    COMPLEX *wrk,COMPLEX *FF,COMPLEX *FF0,
    COMPLEX *FFloc,COMPLEX *polarisa,char *methodeit,
    int *nstop,char *infostr,size_t methodeit_len,size_t infostr_len)
{
    const REAL kzinc = sqrt((*k0)*(*k0)-(*kxinc)*(*kxinc)-(*kyinc)*(*kyinc));
    const REAL cpsi = cos(*psi), spsi = sin(*psi);
    const REAL axp = kzinc/sqrt(kzinc*kzinc+(*kxinc)*(*kxinc));
    const REAL azp = -*kxinc/sqrt(kzinc*kzinc+(*kxinc)*(*kxinc));
    const REAL ays = kzinc/sqrt(kzinc*kzinc+(*kyinc)*(*kyinc));
    const REAL azs = -*kyinc/sqrt(kzinc*kzinc+(*kyinc)*(*kyinc));
    COMPLEX uincx = *E0*cpsi*axp;
    COMPLEX uincy = *E0*spsi*ays;
    COMPLEX uincz = *E0*(cpsi*azp+spsi*azs);
    const REAL ex = std::abs(uincx), ey = std::abs(uincy), ez = std::abs(uincz);
    const REAL emod = sqrt(ex*ex+ey*ey+ez*ez);
    REAL tol = *tolinit;

    if ((*nrig != 8 && *nrig != 9) || !(emod > 0.0) || !std::isfinite(emod)) {
        *nstop = 1;
        return 1;
    }
    uincx /= emod;
    uincy /= emod;
    uincz /= emod;
    *ncompte = 0;
    *nloop = 0;
    *nstop = 0;
    if (*nproche == -1) {
        inverserigscalaire_(tensor_xx,vectx,tabdip,tensor_yy_scalar,
            tensor_zz_scalar,&uincx,&uincy,&uincz,ntotalm,ntotal,ldabi,nlar,
            nmax,ndipole,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,nbsphere,
            XI,XR,wrk,FF,FF0,FFloc,polarisa,niso,methodeit,&tol,tol1,nloop,
            ncompte,nlim,planf,planb,nstop,infostr,methodeit_len,infostr_len);
    } else {
        inverserigscalaire_(tensor_xx,vectx,tabdip,tensor_yy_scalar,tensor_zz_scalar,
            &uincx,&uincy,&uincz,ntotalm,ntotal,ldabi,nlar,nmax,ndipole,nxm,nym,nzm,
            nx,ny,nz,nx2,ny2,nxy2,nz2,nbsphere,XI,XR,wrk,FF,FF0,FFloc,
            polarisa,niso,methodeit,&tol,tol1,nloop,ncompte,nlim,planf,planb,
            nstop,infostr,methodeit_len,infostr_len);
    }
    return *nstop == 1 ? 1 : 0;
}

/* Translate the confocal detector transform and accumulate one Z plane.
 * `Ediffkzneg` is Fortran column-major (nfft,nfft,3); each image stack is
 * (nfft,nfft,nplanz). `nside==10` is the original total-field mode. */
static int microsconfocal_detector_plane_c99(
    const int *nfft2d, const int *imaxk0, const int *nside,
    const int *iz, const int *nplanz, const REAL *k0,
    const REAL *numaperk, const REAL *deltakx, const REAL *deltaky,
    const REAL *gross, const REAL *zlens, const REAL *aretecube,
    const REAL *kxinc, const REAL *kyinc, const REAL *kzinc,
    const COMPLEX *Ax, const COMPLEX *Ay,
    const COMPLEX *Az, const CUCOMPLEX *device_diffraction,
    const int64_t *plan2b, const int64_t *plan2f,
    const COMPLEX *Ediffkzneg,
    COMPLEX *Efourierx, COMPLEX *Efouriery,
    COMPLEX *Efourierz, COMPLEX *Efourierincx,
    COMPLEX *Efourierincy, COMPLEX *Efourierincz,
    COMPLEX *imageconfx, COMPLEX *imageconfy,
    COMPLEX *imageconfz)
{
    const int nf = *nfft2d, half = nf / 2, imax = *imaxk0;
    const size_t plane = (size_t)nf * (size_t)nf;
    const REAL dxy = *deltakx * *deltaky;
    int ikxinc, jkyinc;

    if (nf <= 0 || imax < 0 || imax >= half || *iz < 0 || *iz >= *nplanz ||
        *gross == 0.0 || *k0 <= 0.0 || *nside == 0) return 1;
    if (!initialize_cuda_device())
        return 1;
    deltakroutine_(kxinc, kyinc, deltakx, deltaky, k0, &ikxinc, &jkyinc);
    cuda_data::ConfocalDetectorDeviceBuffers buffers{};
    if (!cuda_data::get_device_confocal_detector_buffers(
            plane, plane * size_t(*nplanz), &buffers))
        return 1;
    const CUCOMPLEX *diffraction_input = device_diffraction;
    if (!diffraction_input) {
        if (!cuda_utils::check_cuda(
                cuda_data::copy_host_complex_to_device(
                    buffers.diffraction, Ediffkzneg, 3u * plane,
                    cuda_utils::TransferCategory::Result),
                "confocal detector", "upload diffraction spectrum"))
            return 1;
        diffraction_input = buffers.diffraction;
    }
#if PRODUITMATVECTFFT_SINGLE_PRECISION
    const CUCOMPLEX ax = make_cuFloatComplex((CU_REAL)std::real(*Ax), (CU_REAL)std::imag(*Ax));
    const CUCOMPLEX ay = make_cuFloatComplex((CU_REAL)std::real(*Ay), (CU_REAL)std::imag(*Ay));
    const CUCOMPLEX az = make_cuFloatComplex((CU_REAL)std::real(*Az), (CU_REAL)std::imag(*Az));
#else
    const CUCOMPLEX ax = make_cuDoubleComplex((CU_REAL)std::real(*Ax), (CU_REAL)std::imag(*Ax));
    const CUCOMPLEX ay = make_cuDoubleComplex((CU_REAL)std::real(*Ay), (CU_REAL)std::imag(*Ay));
    const CUCOMPLEX az = make_cuDoubleComplex((CU_REAL)std::real(*Az), (CU_REAL)std::imag(*Az));
#endif
    if (!cuda_utils::check_cuda(microsconfocal_detector_prepare_wrapper(
            diffraction_input, buffers.fourier_fields, nf,
            imax, *k0, *numaperk, *deltakx, *deltaky, *gross, *zlens,
            *nside, ikxinc, jkyinc, dxy == 0.0 ? 0.0 : 1.0/dxy,
            ax, ay, az, buffers.status),
            "confocal detector", "prepare Fourier fields") ||
        !cuda_utils::check_cufft(
            CUFFT_EXECUTE(get_speckle_fft_plan(nf), buffers.fourier_fields,
                          buffers.fourier_fields, CUFFT_FORWARD),
            "confocal detector", "transform incident Fourier fields") ||
        !cuda_utils::check_cufft(
            CUFFT_EXECUTE(get_speckle_fft_plan(nf), buffers.fourier_fields + 3u*plane,
                          buffers.fourier_fields + 3u*plane, CUFFT_FORWARD),
            "confocal detector", "transform scattered Fourier fields") ||
        !cuda_utils::check_cuda(fourier_to_image_rearrange_wrapper(
            buffers.fourier_fields, nf, (*deltakx * *deltaky) / *gross),
            "confocal detector", "rearrange Fourier fields") ||
        !cuda_utils::check_cuda(microsconfocal_detector_accumulate_wrapper(
            buffers.fourier_fields, buffers.image_stack, nf, *iz, *nplanz,
            *aretecube, *kxinc, *kyinc, *kzinc, *zlens),
            "confocal detector", "accumulate detector image"))
        return 1;
    (void)Efourierx;(void)Efouriery;(void)Efourierz;
    (void)Efourierincx;(void)Efourierincy;(void)Efourierincz;
    (void)imageconfx;(void)imageconfy;(void)imageconfz;
    (void)plan2b;(void)plan2f;(void)half;
    return 0;
}

/* Write the nmat=0 confocal files in the same Fortran traversal order. */
static int microsconfocal_write_ascii_c99(
    const int *nfft2d,const int *nplanz,const int *imaxk0,
    const int *ideltam,const REAL *k0,const REAL *deltax,
    const REAL *deltakx,const REAL *deltak,
    const REAL *zinf,const REAL *stepz,const COMPLEX *masque,
    const COMPLEX *imageconfx,const COMPLEX *imageconfy,
    const COMPLEX *imageconfz)
{
    static const char *const names[10] = {
        "kxfourier.mat","ximage.mat","zimage.mat","kxmic.mat",
        "kymic.mat","masquemic.mat","fieldconfm.mat",
        "fieldconfx.mat","fieldconfy.mat","fieldconfz.mat"
    };
    FILE *files[10] = {NULL};
    const int nf = *nfft2d, half = nf/2, maskn = 2*(*ideltam)+1;
    const size_t plane = (size_t)nf*(size_t)nf;
    int status = 1;

    if (nf <= 0 || *nplanz <= 0 || *imaxk0 < 0 || *ideltam < 0 ||
        *k0 == 0.0) return 1;
    for (size_t q=0;q<10u;++q) {
        files[q]=fopen(names[q],"w");
        if (!files[q]) goto done;
    }
    for (int i=-half;i<half;++i) fprintf(files[1]," %.17g\n",*deltax*(REAL)i);
    for (int iz=0;iz<*nplanz;++iz)
        fprintf(files[2]," %.17g\n",*zinf+*stepz*(REAL)iz);
    for (int i=-*imaxk0;i<=*imaxk0;++i)
        fprintf(files[0]," %.17g\n",*deltakx*(REAL)i/ *k0);
    for (int jd=-*ideltam;jd<=*ideltam;++jd) {
        for (int id=-*ideltam;id<=*ideltam;++id) {
            const REAL kx=*deltak*(REAL)id,ky=*deltak*(REAL)jd;
            const size_t m=(size_t)(id+*ideltam)+(size_t)maskn*(size_t)(jd+*ideltam);
            if (jd==0) fprintf(files[3]," %.17g\n",kx/ *k0);
            if (id==0) fprintf(files[4]," %.17g\n",ky/ *k0);
            fprintf(files[5]," %.17g\n",std::abs(masque[m]));
        }
    }
    for (int iz=0;iz<*nplanz;++iz) {
        for (int j=0;j<nf;++j) {
            for (int i=0;i<nf;++i) {
                const size_t q=(size_t)i+(size_t)nf*(size_t)j+plane*(size_t)iz;
                const COMPLEX ex=imageconfx[q],ey=imageconfy[q],ez=imageconfz[q];
                const REAL ex2=std::real(ex*std::conj(ex));
                const REAL ey2=std::real(ey*std::conj(ey));
                const REAL ez2=std::real(ez*std::conj(ez));
                fprintf(files[6]," %.17g\n",sqrt(ex2+ey2+ez2));
                fprintf(files[7]," %.17g %.17g\n",std::real(ex),std::imag(ex));
                fprintf(files[8]," %.17g %.17g\n",std::real(ey),std::imag(ey));
                fprintf(files[9]," %.17g %.17g\n",std::real(ez),std::imag(ez));
            }
        }
    }
    /* The Fortran routine writes fieldconfy between x and z. */
    status=0;
done:
    for (size_t q=0;q<10u;++q) if (files[q]) fclose(files[q]);
    return status;
}

typedef struct MicrosconfocalSchedulerArgs {
    const int *nrig,*nproche,*nfft2d,*nside,*nquicklens,*ideltam;
    const int *nbsphere,*nbsphere3,*nbitermic,*niso,*nmax;
    const int *ntotalm,*ntotal,*ndipole,*nxm,*nym,*nzm,*nx,*ny,*nz;
    const int *nx2,*ny2,*nxy2,*nz2,*ldabi,*nlar,*nlim;
    int *imaxk0,*tabfft2,*nloop,*ncompte,*nstop;
    int *tabdip;
    REAL *P0,*deltakx,*deltaky,*tol1;
    const REAL *tolinit,*k0,*aretecube,*numaperinc,*w0,*gross;
    REAL *irra;
    const REAL *deltak,*stepz,*zinf;
    const REAL *xs,*ys,*zs,*psi,*ss,*pp;
    COMPLEX *tensor[6],*vect[3];
    COMPLEX *XI,*XR,*wrk,*FF,*FF0,*FFloc,*polarisa;
    const COMPLEX *epsilon,*eps0,*masque;
    COMPLEX *Efourierx,*Efouriery,*Efourierz;
    COMPLEX *Efourierincx,*Efourierincy,*Efourierincz;
    COMPLEX *Ediffkzneg,*imageconfx,*imageconfy,*imageconfz;
    const int64_t *planXf,*planXb,*planYf,*planYb,*planZf,*planZb;
    const int64_t *planf,*planb,*plan2f,*plan2b;
    char *methodeit,*infostr;
    size_t methodeit_len,infostr_len;
} MicrosconfocalSchedulerArgs;

/* Run all masked illuminations, solve their fields, diffract and accumulate
 * each detector plane. HDF5 metadata and the public Fortran ABI are handled
 * by the outer entry point. */
static int microsconfocal_run_scheduler_c99(const MicrosconfocalSchedulerArgs *a,
    const int *nplanz,const REAL *numaperk,int *niterii)
{
    COMPLEX E0,Ax,Ay,Az;
    const int maskwidth=2*(*a->ideltam)+1;
    int status=0;
    const size_t detector_plane = size_t(*a->nfft2d) * size_t(*a->nfft2d);
    const size_t detector_image_values = detector_plane * size_t(*nplanz);
    cuda_data::ConfocalDetectorDeviceBuffers detector_buffers{};
    if (!cuda_data::get_device_confocal_detector_buffers(
            detector_plane, detector_image_values, &detector_buffers) ||
        !cuda_utils::check_cuda(microsconfocal_detector_zero_wrapper(
            detector_buffers.image_stack, size_t(3)*detector_image_values),
            "confocal scheduler", "clear detector image stack") ||
        !cuda_utils::check_cuda(cudaMemset(detector_buffers.status, 0, sizeof(int)),
            "confocal scheduler", "clear detector status")) {
        microsconfocal_set_info(a->infostr, a->infostr_len,
            "confocal detector GPU buffer initialization failed");
        *a->nstop=1;
        return 1;
    }
    const int last_guess_mode = ifdda_get_last_initial_guess_mode_all_cuda_();
    const bool incident_fields_device_only =
        *a->nrig == 0 && last_guess_mode >= 0 && last_guess_mode <= 2;

    *a->P0 /= (REAL)*a->nbitermic;
    irradiance_(a->P0,a->w0,&E0,a->irra);
    *niterii=0;
    for(int idelta=-*a->ideltam;idelta<=*a->ideltam;++idelta){
        const REAL kxinc=(REAL)idelta*(*a->deltak);
        const int i=*a->ideltam+idelta;
        for(int jdelta=-*a->ideltam;jdelta<=*a->ideltam;++jdelta){
            const REAL kyinc=(REAL)jdelta*(*a->deltak);
            const int j=*a->ideltam+jdelta;
            const size_t midx=(size_t)i+(size_t)maskwidth*(size_t)j;
            REAL kzinc,zlens;
            const CUCOMPLEX *device_diffraction = nullptr;

            if(std::abs(a->masque[midx])>2.0)continue;
            ++*niterii;
            if (incident_fields_device_only)
                request_device_only_result_vector(a->FF0);
            request_device_only_result_vector(a->FF);
            if(microsconfocal_incident_field_c99(a->nbsphere,a->k0,&kxinc,
                    &kyinc,a->psi,&E0,a->xs,a->ys,a->zs,a->FF0,a->XI,
                    &Ax,&Ay,&Az,&kzinc)!=0){
                clear_device_only_result_vector_request(a->FF);
                microsconfocal_set_info(a->infostr,a->infostr_len,
                    "invalid incident field");
                *a->nstop=1;return 1;
            }
            if(*a->nrig>=1&&*a->nrig<=5){
                status=microsconfocal_approx_fields_c99(a->nrig,a->nbsphere,
                    a->nmax,a->niso,a->ntotalm,a->ntotal,a->ndipole,a->nxm,
                    a->nym,a->nzm,a->nx,a->ny,a->nz,a->nx2,a->ny2,a->nxy2,
                    a->nz2,(const int *)a->tabdip,a->tensor,a->vect,a->planf,
                    a->planb,a->polarisa,a->epsilon,a->eps0,a->aretecube,a->k0,
                    a->FF0,a->XI,a->XR,a->FF,a->FFloc);
            }else{
                status=microsconfocal_advanced_fields_c99(a->nrig,a->nproche,
                    a->ntotalm,a->ntotal,a->ldabi,a->nlar,a->nmax,a->ndipole,
                    a->nxm,a->nym,a->nzm,a->nx,a->ny,a->nz,a->nx2,a->ny2,
                    a->nxy2,a->nz2,a->nbsphere,a->nbsphere3,a->nlim,a->niso,
                    a->tabdip,a->tensor,a->vect,a->XI,a->XR,a->wrk,a->FF,
                    a->FF0,a->FFloc,a->polarisa,a->epsilon,&E0,a->psi,a->ss,
                    a->pp,&kxinc,&kyinc,a->xs,a->ys,a->zs,a->aretecube,a->k0,
                    a->w0,a->nfft2d,a->Efourierx,a->Efouriery,a->Efourierz,
                    a->tolinit,a->tol1,a->nloop,a->ncompte,a->methodeit,
                    a->infostr,a->nstop,a->planXf,a->planXb,a->planYf,a->planYb,
                    a->planZf,a->planZb,a->planf,a->planb,a->plan2f,a->plan2b,
                    a->methodeit_len,a->infostr_len);
            }
            clear_device_only_result_vector_request(a->FF);
            clear_device_only_result_vector_request(a->FF0);
            release_gpbicg_device_b_alias(a->FF0);
            if(status!=0||*a->nstop==1)return 1;
            if (*a->nquicklens == 1) {
                const size_t mesh_count = size_t(*a->nx)*size_t(*a->ny)*size_t(*a->nz);
                const size_t dipole_count = size_t(*a->nbsphere);
                cuda_data::ConfocalMicroscopeDeviceBuffers lens_buffers{};
                CU_INT *device_dipole_map = get_device_tabdip(a->tabdip, mesh_count);
                const CUCOMPLEX *device_scattered =
                    get_device_result_vector_alias(a->FF, size_t(3)*dipole_count);
                if (!device_dipole_map ||
                    !cuda_data::get_device_confocal_microscope_buffers(
                        dipole_count, 0, size_t(3)*mesh_count, &lens_buffers) ||
                    !lens_buffers.dense_field) {
                    status=1;
                } else {
                    if (!device_scattered) {
                        if (!cuda_utils::check_cuda(
                                cuda_data::copy_host_complex_to_device(
                                    lens_buffers.dipole_field, a->FF,
                                    size_t(3)*dipole_count,
                                    cuda_utils::TransferCategory::Result),
                                "confocal lens diffraction", "upload scattered field"))
                            status=1;
                        else
                            device_scattered=lens_buffers.dipole_field;
                    }
                    if (status==0 && !cuda_utils::check_cuda(
                            microscope_expand_packed_field_wrapper(
                                device_scattered, device_dipole_map,
                                lens_buffers.dense_field, int(mesh_count),
                                int(dipole_count)),
                            "confocal lens diffraction", "expand scattered field"))
                        status=1;
                    if (status==0) {
                        diffractefft2dlens_set_resident_input(lens_buffers.dense_field);
                        diffractefft2dlens_set_resident_output(1);
                        diffractefft2dlens_(a->nx,a->ny,a->nz,a->nxm,a->nym,
                            a->nzm,a->nfft2d,a->tabfft2,a->k0,a->xs,a->ys,a->zs,
                            a->aretecube,a->Efourierx,a->Efouriery,a->Efourierz,
                            a->FF,a->imaxk0,a->deltakx,a->deltaky,a->Ediffkzneg,
                            numaperk,a->nside,a->plan2f,
                            a->plan2b,a->nstop,a->infostr,
                            a->infostr_len);
                        diffractefft2dlens_set_resident_input(nullptr);
                        diffractefft2dlens_set_resident_output(0);
                        status=*a->nstop==1 ? 1 : 0;
                        if (status==0) {
                            device_diffraction=diffractefft2dlens_get_resident_output();
                            if (!device_diffraction)
                                status=1;
                        }
                    }
                }
                if (status!=0 && *a->nstop!=1) {
                    microsconfocal_set_info(a->infostr,a->infostr_len,
                        "CUDA confocal lens diffraction failed");
                    *a->nstop=1;
                }
            } else {
                *a->deltaky = *a->deltakx;
                status=microsconfocal_classical_diffraction_device_c99(
                    a->nfft2d,a->imaxk0,a->nside,a->nbsphere,a->k0,numaperk,
                    a->deltakx,a->deltaky,a->xs,a->ys,a->zs,a->FF,
                    &device_diffraction);
                *a->nstop = status == 0 ? 0 : 1;
            }
            clear_device_result_vector_alias(a->FF);
            if(status!=0||*a->nstop==1)return 1;
            for(int iz=0;iz<*nplanz;++iz){
                zlens=*a->zinf+*a->stepz*(REAL)iz;
                status=microsconfocal_detector_plane_c99(a->nfft2d,
                    a->imaxk0,a->nside,&iz,nplanz,a->k0,numaperk,a->deltakx,
                    a->deltaky,a->gross,&zlens,a->aretecube,&kxinc,&kyinc,
                    &kzinc,&Ax,&Ay,&Az,device_diffraction,a->plan2b,a->plan2f,
                    a->Ediffkzneg,
                    a->Efourierx,a->Efouriery,a->Efourierz,a->Efourierincx,
                    a->Efourierincy,a->Efourierincz,a->imageconfx,a->imageconfy,
                    a->imageconfz);
                if(status!=0){
                    microsconfocal_set_info(a->infostr,a->infostr_len,
                        "confocal detector transform failed");
                    *a->nstop=1;return 1;
                }
            }
            if(*a->nstop==-1){
                microsconfocal_set_info(a->infostr,a->infostr_len,
                    "Calculation cancelled during iterative method");
                return 1;
            }
        }
    }
    int detector_status = 0;
    if (!cuda_utils::check_cuda(cuda_data::copy_device_to_host(
            &detector_status, detector_buffers.status, sizeof(detector_status),
            cuda_utils::TransferCategory::Diagnostic),
            "confocal scheduler", "read detector status") || detector_status != 0) {
        microsconfocal_set_info(a->infostr, a->infostr_len,
            "confocal detector polarization rotation failed");
        *a->nstop=1;
        return 1;
    }
    if (!cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
            a->imageconfx, detector_buffers.image_stack, detector_image_values,
            cuda_utils::TransferCategory::Result),
            "confocal scheduler", "download detector image x") ||
        !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
            a->imageconfy, detector_buffers.image_stack + detector_image_values,
            detector_image_values, cuda_utils::TransferCategory::Result),
            "confocal scheduler", "download detector image y") ||
        !cuda_utils::check_cuda(cuda_data::copy_device_complex_to_host(
            a->imageconfz, detector_buffers.image_stack + 2u*detector_image_values,
            detector_image_values, cuda_utils::TransferCategory::Result),
            "confocal scheduler", "download detector image z")) {
        microsconfocal_set_info(a->infostr, a->infostr_len,
            "confocal detector image download failed");
        *a->nstop=1;
        return 1;
    }
    return 0;
}

/* Public gfortran ABI entry point. Hidden CHARACTER lengths are appended by
 * gfortran after all explicit arguments. */
extern "C" void microsconfocal_(
    COMPLEX *FFTTENSORxx,COMPLEX *FFTTENSORxy,
    COMPLEX *FFTTENSORxz,COMPLEX *FFTTENSORyy,
    COMPLEX *FFTTENSORyz,COMPLEX *FFTTENSORzz,
    COMPLEX *vectx,COMPLEX *vecty,COMPLEX *vectz,
    const int *ntotalm,const int *ntotal,const int *ldabi,const int *nlar,
    const int *nmax,const int *nxm,const int *nym,const int *nzm,
    const int *nx,const int *ny,const int *nz,const int *nx2,const int *ny2,
    const int *nxy2,const int *nz2,const int *ndipole,const int *nbsphere,
    const int *nbsphere3,const int *nproche,const int *nrig,const int *nfft2d,
    int *tabfft2,int *Tabdip,COMPLEX *XI,COMPLEX *XR,
    COMPLEX *wrk,COMPLEX *FF,COMPLEX *FF0,
    COMPLEX *FFloc,COMPLEX *polarisa,const int *niso,
    COMPLEX *epsilon,char *methodeit,char *trope,
    const REAL *tolinit,REAL *tol1,int *nloop,int *ncompte,
    const int *nlim,const REAL *xs,const REAL *ys,const REAL *zs,
    const REAL *aretecube,const REAL *numaper,const REAL *numaperinc,
    const REAL *numaperinc2,const REAL *psiinc,const int *nquicklens,
    const COMPLEX *eps0,const REAL *k0,REAL *P0,REAL *irra,
    const REAL *w0,const REAL *gross,REAL *stepz,REAL *zinf,
    REAL *zsup,COMPLEX *Eimagex,COMPLEX *Eimagey,
    COMPLEX *Eimagez,COMPLEX *Efourierx,
    COMPLEX *Efouriery,COMPLEX *Efourierz,
    COMPLEX *Eimageincx,COMPLEX *Eimageincy,
    COMPLEX *Eimageincz,COMPLEX *Efourierincx,
    COMPLEX *Efourierincy,COMPLEX *Efourierincz,
    COMPLEX *Ediffkzpos,COMPLEX *Ediffkzneg,REAL *kxy,
    REAL *xy,const int *nside,COMPLEX *masque,const int *nmasque,
    const int *ideltam,int *imaxk0,const int *nbitermic,REAL *deltak,
    REAL *deltakx,const int *ntypemic,const int64_t *planXf,
    const int64_t *planXb,const int64_t *planYf,const int64_t *planYb,
    const int64_t *planZf,const int64_t *planZb,const int64_t *planf,
    const int64_t *planb,const int64_t *plan2f,const int64_t *plan2b,
    const int *nmat,int64_t *file_id,int64_t *group_idmic,int *nstop,char *infostr,
    size_t methodeit_len,size_t trope_len,size_t infostr_len)
{
    const REAL pi=3.141592653589793238462643383279502884;
    const REAL nmtom=1.0e-9;
    REAL numaperk=(*k0)*(*numaperinc),psi=(*psiinc)*pi/180.0;
    REAL ss=cos(psi),pp=sin(psi);
    REAL deltax;
    int nplanz=0,nplanze=0,niterii=0,status=0;
    size_t cells=0,imagecells=0,plane=0,center=0;
    COMPLEX *imageconfx=NULL,*imageconfy=NULL,*imageconfz=NULL;
    COMPLEX *tensor[6]={FFTTENSORxx,FFTTENSORxy,FFTTENSORxz,
        FFTTENSORyy,FFTTENSORyz,FFTTENSORzz};
    COMPLEX *vect[3]={vectx,vecty,vectz};
    MicrosconfocalSchedulerArgs a;
    (void)numaper;(void)numaperinc2;(void)ntypemic;(void)nmasque;(void)trope_len;(void)file_id;
    (void)group_idmic;(void)trope;(void)Eimageincx;(void)Eimageincy;
    (void)Eimageincz;(void)Ediffkzpos;

    if(microsconfocal_validate_c99(nfft2d,&nmtom,&numaperk,deltak,nside,
            stepz,zinf,zsup,stepz,zinf,zsup,&nplanz,&nplanze,nstop,infostr,
            infostr_len)!=0)return;
    if(*nfft2d<=0||nplanz<=0){*nstop=1;return;}
    if((size_t)*nfft2d>SIZE_MAX/(size_t)*nfft2d ||
       (size_t)*nfft2d*(size_t)*nfft2d>SIZE_MAX/(size_t)nplanz ||
       (size_t)*nfft2d*(size_t)*nfft2d*(size_t)nplanz>
            SIZE_MAX/sizeof(COMPLEX)){
        microsconfocal_set_info(infostr,infostr_len,
            "confocal image allocation failed");
        *nstop=1;return;
    }
    cells=(size_t)*nfft2d*(size_t)*nfft2d*(size_t)nplanz;
    imageconfx=(COMPLEX *)calloc(cells,sizeof(COMPLEX));
    imageconfy=(COMPLEX *)calloc(cells,sizeof(COMPLEX));
    imageconfz=(COMPLEX *)calloc(cells,sizeof(COMPLEX));
    if(!imageconfx||!imageconfy||!imageconfz){
        microsconfocal_set_info(infostr,infostr_len,
            "confocal image allocation failed");
        *nstop=1;goto cleanup;
    }
    imagecells=(size_t)*nfft2d*(size_t)*nfft2d;
    std::fill_n(Eimagex, imagecells, COMPLEX{});
    std::fill_n(Eimagey, imagecells, COMPLEX{});
    std::fill_n(Eimagez, imagecells, COMPLEX{});
    std::fill_n(Eimageincx, imagecells, COMPLEX{});
    std::fill_n(Eimageincy, imagecells, COMPLEX{});
    std::fill_n(Eimageincz, imagecells, COMPLEX{});

    memset(&a,0,sizeof a);
    a.nrig=nrig;a.nproche=nproche;a.nfft2d=nfft2d;a.nside=nside;
    a.nquicklens=nquicklens;a.ideltam=ideltam;a.nbsphere=nbsphere;
    a.nbsphere3=nbsphere3;a.nbitermic=nbitermic;a.niso=niso;a.nmax=nmax;
    a.ntotalm=ntotalm;a.ntotal=ntotal;a.ndipole=ndipole;a.nxm=nxm;a.nym=nym;
    a.nzm=nzm;a.nx=nx;a.ny=ny;a.nz=nz;a.nx2=nx2;a.ny2=ny2;a.nxy2=nxy2;
    a.nz2=nz2;a.ldabi=ldabi;a.nlar=nlar;a.nlim=nlim;a.imaxk0=imaxk0;
    a.tabfft2=tabfft2;a.tabdip=Tabdip;a.nloop=nloop;a.ncompte=ncompte;
    a.nstop=nstop;a.P0=P0;a.deltakx=deltakx;a.deltaky=deltakx;a.tol1=tol1;
    a.tolinit=tolinit;a.k0=k0;a.aretecube=aretecube;a.numaperinc=numaperinc;
    a.w0=w0;a.irra=irra;a.gross=gross;a.deltak=deltak;a.stepz=stepz;
    a.zinf=zinf;a.xs=xs;a.ys=ys;a.zs=zs;a.psi=&psi;a.ss=&ss;a.pp=&pp;
    a.tensor[0]=tensor[0];a.tensor[1]=tensor[1];a.tensor[2]=tensor[2];
    a.tensor[3]=tensor[3];a.tensor[4]=tensor[4];a.tensor[5]=tensor[5];
    a.vect[0]=vect[0];a.vect[1]=vect[1];a.vect[2]=vect[2];
    a.XI=XI;a.XR=XR;a.wrk=wrk;a.FF=FF;a.FF0=FF0;a.FFloc=FFloc;
    a.polarisa=polarisa;a.epsilon=epsilon;a.eps0=eps0;a.masque=masque;
    a.Efourierx=Efourierx;a.Efouriery=Efouriery;a.Efourierz=Efourierz;
    a.Efourierincx=Efourierincx;a.Efourierincy=Efourierincy;
    a.Efourierincz=Efourierincz;a.Ediffkzneg=Ediffkzneg;
    a.imageconfx=imageconfx;a.imageconfy=imageconfy;a.imageconfz=imageconfz;
    a.planXf=planXf;a.planXb=planXb;a.planYf=planYf;a.planYb=planYb;
    a.planZf=planZf;a.planZb=planZb;a.planf=planf;a.planb=planb;
    a.plan2f=plan2f;a.plan2b=plan2b;a.methodeit=methodeit;a.infostr=infostr;
    a.methodeit_len=methodeit_len;a.infostr_len=infostr_len;
    status=microsconfocal_run_scheduler_c99(&a,&nplanz,&numaperk,&niterii);
    if(status!=0||*nstop==1)goto cleanup;

    plane=(size_t)*nfft2d*(size_t)*nfft2d;
    center=(size_t)(nplanze-1)*plane;
    memcpy(Eimagex,imageconfx+center,plane*sizeof(*Eimagex));
    memcpy(Eimagey,imageconfy+center,plane*sizeof(*Eimagey));
    memcpy(Eimagez,imageconfz+center,plane*sizeof(*Eimagez));
    deltax=(2.0*pi/(REAL)*nfft2d)/ *deltakx;
    if(*nmat==0){
        status=microsconfocal_write_ascii_c99(nfft2d,&nplanz,imaxk0,ideltam,
            k0,&deltax,deltakx,deltak,zinf,stepz,masque,imageconfx,imageconfy,
            imageconfz);
        if(status!=0){
            microsconfocal_set_info(infostr,infostr_len,
                "confocal ASCII output failed");
            *nstop=1;goto cleanup;
        }
    }else if(*nmat==2){
        const int maskwidth=2*(*ideltam)+1;
        for(int jd=0;jd<maskwidth;++jd)
            for(int id=0;id<maskwidth;++id){
                const size_t q=(size_t)id+(size_t)maskwidth*(size_t)jd;
                masque[q]=std::abs(masque[q]);
            }
    }
    for(int i=0;i<*nfft2d;++i){
        const int offset=i-*nfft2d/2;
        xy[i]=deltax*(REAL)offset*(*gross);
        kxy[i]=(*deltakx)*(REAL)offset/(*k0);
    }
    printf("NSTOP %d\n",*nstop);
cleanup:
    free(imageconfx);free(imageconfy);free(imageconfz);
    (void)niterii;
}
