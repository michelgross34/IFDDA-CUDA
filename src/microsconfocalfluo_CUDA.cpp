#include "cdmlib_cpp_abi.hpp"
#define IFDDA_CUDA_CALLERS_MICROSCONFOCALFLUO_CUDA
#include "cuda_interfaces.h"
#undef IFDDA_CUDA_CALLERS_MICROSCONFOCALFLUO_CUDA
#include "cuda_data.h"
#include "cuda_utils.h"
#include "kernel.h"
#include "microsconfocal_CUDA.h"
/* C99 implementation of cdmlib/microsconfocalfluo.f90.
 * The fluorescence image is the square of the incoherent illumination sum;
 * per-illumination field solving and diffraction reuse the C99 confocal core.
 */

#include <cmath>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>











static void fluo_set_info(char *dst, size_t n, const char *s)
{
    size_t m = strlen(s);
    if (m > n) m = n;
    memset(dst, ' ', n);
    memcpy(dst, s, m);
}

static int fluo_write_ascii(const int *nf, const int *nplanes,
    const int *imax, const int *ideltam, const REAL *k0,
    const REAL *deltax, const REAL *deltakx, const REAL *deltak,
    const REAL *zinf, const REAL *stepz, const COMPLEX *mask,
    const REAL *image)
{
    FILE *fx = NULL, *fz = NULL, *fk = NULL, *fkmx = NULL, *fkmy = NULL;
    FILE *fm = NULL, *fi = NULL;
    const REAL pi = 3.141592653589793238462643383279502884;
    int status = 1;
    int half=0,width=0;
    fx=fopen("ximage.mat","w"); fz=fopen("zimage.mat","w");
    fk=fopen("kxfourier.mat","w"); fkmx=fopen("kxmic.mat","w");
    fkmy=fopen("kymic.mat","w"); fm=fopen("masquemic.mat","w");
    fi=fopen("fluoconfocal.mat","w");
    if(!fx||!fz||!fk||!fkmx||!fkmy||!fm||!fi) goto done;
    half=*nf/2; width=2*(*ideltam)+1;
    for(int i=-half;i<half;++i) fprintf(fx," %.17g\n",*deltax*(REAL)i);
    for(int z=0;z<*nplanes;++z) fprintf(fz," %.17g\n",*zinf+*stepz*(REAL)z);
    for(int i=-*imax;i<=*imax;++i)
        fprintf(fk," %.17g\n",(*deltakx*(REAL)i)/(*k0));
    for(int j=-*ideltam;j<=*ideltam;++j)
        for(int i=-*ideltam;i<=*ideltam;++i){
            const size_t q=(size_t)(i+*ideltam)+(size_t)width*(size_t)(j+*ideltam);
            if(j==0)fprintf(fkmx," %.17g\n",(*deltak*(REAL)i)/(*k0));
            if(i==0)fprintf(fkmy," %.17g\n",(*deltak*(REAL)j)/(*k0));
            fprintf(fm," %.17g\n",std::abs(mask[q]));
        }
    for(int z=0;z<*nplanes;++z)
        for(int j=0;j<*nf;++j)
            for(int i=0;i<*nf;++i)
                fprintf(fi," %.17g\n",image[(size_t)i+(size_t)*nf*(size_t)j+
                    (size_t)*nf*(size_t)*nf*(size_t)z]);
    (void)pi;
    status=0;
done:
    if(fx)fclose(fx);
    if(fz)fclose(fz);
    if(fk)fclose(fk);
    if(fkmx)fclose(fkmx);
    if(fkmy)fclose(fkmy);
    if(fm)fclose(fm);
    if(fi)fclose(fi);
    return status;
}

/* The gfortran ABI entry point. CHARACTER lengths are trailing hidden args. */
extern "C" void microsconfocalfluo_(
    COMPLEX *Txx,COMPLEX *Txy,COMPLEX *Txz,
    COMPLEX *Tyy,COMPLEX *Tyz,COMPLEX *Tzz,
    COMPLEX *vx,COMPLEX *vy,COMPLEX *vz,
    const int *ntotalm,const int *ntotal,const int *ldabi,const int *nlar,
    const int *nmax,const int *nxm,const int *nym,const int *nzm,
    const int *nx,const int *ny,const int *nz,const int *nx2,const int *ny2,
    const int *nxy2,const int *nz2,const int *ndipole,const int *nbsphere,
    const int *nbsphere3,const int *nproche,const int *nrig,const int *nfft2d,
    int *tabfft2,int *Tabdip,COMPLEX *XI,COMPLEX *XR,
    COMPLEX *wrk,COMPLEX *FF,COMPLEX *FF0,
    COMPLEX *FFloc,COMPLEX *polarisa,const int *niso,
    COMPLEX *epsilon,char *methodeit,char *trope,const REAL *tolinit,
    REAL *tol1,int *nloop,int *ncompte,int *nlim,const REAL *xs,
    const REAL *ys,const REAL *zs,const REAL *aretecube,
    const REAL *numaper,const REAL *numaperinc,const REAL *numaperinc2,
    const REAL *psiinc,const int *nquicklens,const COMPLEX *eps0,
    const REAL *k0,REAL *P0,REAL *irra,const REAL *w0,
    const REAL *gross,REAL *stepz,REAL *zinf,REAL *zsup,
    COMPLEX *Eimagex,COMPLEX *Eimagey,COMPLEX *Eimagez,
    COMPLEX *Efourierx,COMPLEX *Efouriery,COMPLEX *Efourierz,
    COMPLEX *Eimageincx,COMPLEX *Eimageincy,COMPLEX *Eimageincz,
    COMPLEX *Efourierincx,COMPLEX *Efourierincy,COMPLEX *Efourierincz,
    COMPLEX *Ediffkzpos,COMPLEX *Ediffkzneg,REAL *kxy,REAL *xy,
    const int *nside,COMPLEX *masque,const int *nmasque,const int *ideltam,
    int *imaxk0,int *nbitermic,REAL *deltak,REAL *deltakx,const int *ntypemic,
    const int64_t *planXf,const int64_t *planXb,const int64_t *planYf,const int64_t *planYb,
    const int64_t *planZf,const int64_t *planZb,const int64_t *planf,const int64_t *planb,
    const int64_t *plan2f,const int64_t *plan2b,const int *nmat,int64_t *file_id,
    int64_t *group_idmic,int *nstop,char *infostr,size_t methodeit_len,
    size_t trope_len,size_t infostr_len)
{
    const REAL pi=3.141592653589793238462643383279502884, nmtom=1.0e-9;
    REAL numaperk=(*k0)*(*numaperinc), deltax, E0abs2;
    int nplanz=0,nplanze=0,niter=0,status=0;
    size_t middle=0;
    const int nf=*nfft2d, half=nf/2, npol=(*nrig==9)?1:2;
    const int maskwidth=2*(*ideltam)+1;
    COMPLEX E0;
    REAL *image=NULL;
    COMPLEX *tensor[6]={Txx,Txy,Txz,Tyy,Tyz,Tzz};
    COMPLEX *vect[3]={vx,vy,vz};
    const int64_t ignored_plan=0;
    (void)numaper;(void)numaperinc2;(void)ntypemic;(void)nmasque;(void)trope_len;
    (void)file_id;(void)group_idmic;(void)Ediffkzpos;(void)Eimagey;(void)Eimagez;
    (void)Eimageincx;(void)Eimageincy;(void)Eimageincz;(void)ignored_plan;

    if(microsconfocal_validate_c99(nfft2d,&nmtom,&numaperk,deltak,nside,
            stepz,zinf,zsup,stepz,zinf,zsup,&nplanz,&nplanze,nstop,infostr,
            infostr_len)!=0)return;
    if(*nside!=1&&*nside!=-1){
        fluo_set_info(infostr,infostr_len,"problem with side");*nstop=-1;return;
    }
    if(*nbitermic<=0||*nbsphere<0||nf<=0||nplanz<=0){
        fluo_set_info(infostr,infostr_len,"invalid confocal fluorescence dimensions");*nstop=1;return;
    }
    *nbitermic *= npol;
    if(*nbitermic<=0){*nstop=1;return;}
    *P0 /= (REAL)*nbitermic;
    irradiance_(P0,w0,&E0,irra);
    E0abs2=std::real(E0*std::conj(E0));
    (void)E0abs2;
    const size_t plane=(size_t)nf*(size_t)nf;
    if(plane>SIZE_MAX/(size_t)nplanz || plane*(size_t)nplanz>SIZE_MAX/sizeof(REAL)){
        fluo_set_info(infostr,infostr_len,"confocal fluorescence allocation failed");*nstop=1;return;
    }
    const size_t cells=plane*(size_t)nplanz;
    if (!initialize_cuda_device()) {
        fluo_set_info(infostr,infostr_len,"CUDA device initialization failed");*nstop=1;return;
    }
    cuda_data::ConfocalFluorescenceDeviceBuffers device{};
    if (!cuda_data::get_device_confocal_fluorescence_buffers(
            0u,cells,0u,6u*plane,&device) ||
        !device.intensity_stack || !device.spectrum_planes ||
        (*nquicklens != 0 && *nquicklens != 1)) {
        fluo_set_info(infostr,infostr_len,"confocal fluorescence CUDA buffers unavailable");*nstop=1;return;
    }
    cufftHandle fft_plan=get_speckle_fft_plan(nf);
    if (!fft_plan) {
        fluo_set_info(infostr,infostr_len,"confocal fluorescence cuFFT plan failed");*nstop=1;return;
    }
    std::vector<REAL> host_image;
    try {
        host_image.resize(cells);
    } catch (...) {
        fluo_set_info(infostr,infostr_len,"confocal fluorescence host staging allocation failed");*nstop=1;return;
    }
    if (confocal_fluorescence_zero_wrapper(device.intensity_stack,cells)!=cudaSuccess) {
        fluo_set_info(infostr,infostr_len,"confocal fluorescence CUDA initialization failed");*nstop=1;return;
    }
    std::fill_n(Eimagex, plane, COMPLEX{});
    std::fill_n(Efourierx, plane, COMPLEX{});
    std::fill_n(Efouriery, plane, COMPLEX{});
    std::fill_n(Efourierz, plane, COMPLEX{});
    std::fill_n(Efourierincx, plane, COMPLEX{});
    std::fill_n(Efourierincy, plane, COMPLEX{});
    std::fill_n(Efourierincz, plane, COMPLEX{});

    for(int id=-*ideltam;id<=*ideltam;++id){
      const REAL kxinc=(REAL)id*(*deltak);
      for(int jd=-*ideltam;jd<=*ideltam;++jd){
        const REAL kyinc=(REAL)jd*(*deltak);
        const size_t midx=(size_t)(id+*ideltam)+(size_t)maskwidth*(size_t)(jd+*ideltam);
        if(std::abs(masque[midx])>2.0)continue;
        for(int ipol=0;ipol<npol;++ipol){
          const REAL psi=(ipol==0)?0.0:pi/2.0;
          const REAL ss=cos(psi),pp=sin(psi);
          COMPLEX Ax,Ay,Az;
          REAL kzinc;
          ++niter;
          // Override a persistent *initial* illumination source only when one
          // is registered (native cdmlib.cpp).  The Fortran/hybrid path has no
          // such source and retains its existing host-backed solver semantics.
          const bool override_initial_source=has_gpbicg_device_b_source(FF0);
          if(override_initial_source) request_device_only_result_vector(FF0);
          request_device_only_result_vector(FF);
          if(microsconfocal_incident_field_c99(nbsphere,k0,&kxinc,&kyinc,&psi,&E0,
                  xs,ys,zs,FF0,XI,&Ax,&Ay,&Az,&kzinc)!=0){
              clear_device_only_result_vector_request(FF);
              clear_device_only_result_vector_request(FF0);
              if(override_initial_source) release_gpbicg_device_b_alias(FF0);
              fluo_set_info(infostr,infostr_len,"invalid incident field");*nstop=1;goto cleanup;
          }
          if(*nrig>=1&&*nrig<=5){
              status=microsconfocal_approx_fields_c99(nrig,nbsphere,nmax,niso,
                  ntotalm,ntotal,ndipole,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,
                  Tabdip,tensor,vect,planf,planb,polarisa,epsilon,eps0,aretecube,
                  k0,FF0,XI,XR,FF,FFloc);
          }else{
              status=microsconfocal_advanced_fields_c99(nrig,nproche,ntotalm,
                  ntotal,ldabi,nlar,nmax,ndipole,nxm,nym,nzm,nx,ny,nz,nx2,ny2,
                  nxy2,nz2,nbsphere,nbsphere3,nlim,niso,Tabdip,tensor,vect,XI,XR,
                  wrk,FF,FF0,FFloc,polarisa,epsilon,&E0,&psi,&ss,&pp,&kxinc,
                  &kyinc,xs,ys,zs,aretecube,k0,w0,nfft2d,Efourierx,Efouriery,
                  Efourierz,tolinit,tol1,nloop,ncompte,methodeit,infostr,nstop,
                  planXf,planXb,planYf,planYb,planZf,planZb,planf,planb,plan2f,
                  plan2b,methodeit_len,infostr_len);
          }
          clear_device_only_result_vector_request(FF);
          clear_device_only_result_vector_request(FF0);
          if(override_initial_source) release_gpbicg_device_b_alias(FF0);
          if(status!=0||*nstop!=0){
             if(status!=0&&*nstop==0){
                 fluo_set_info(infostr,infostr_len,
                     "confocal fluorescence field solve failed");
                 *nstop=1;
             }
             clear_device_result_vector_alias(FF);
             goto cleanup;
          }
          REAL deltakx_local=*deltakx,deltaky_local=*deltakx;
          const CUCOMPLEX *resident_diffraction=nullptr;
          if (*nquicklens == 0) {
              status=microsconfocal_classical_diffraction_device_c99(
                  nfft2d,imaxk0,nside,nbsphere,k0,&numaperk,&deltakx_local,
                  &deltaky_local,xs,ys,zs,FF,&resident_diffraction);
              if(status!=0||resident_diffraction==nullptr) {
                  fluo_set_info(infostr,infostr_len,
                      "confocal fluorescence diffraction failed");
                  *nstop=1;
                  clear_device_result_vector_alias(FF);
                  goto cleanup;
              }
          } else {
              const size_t mesh_count=size_t(*nx)*size_t(*ny)*size_t(*nz);
              const size_t dipole_count=size_t(*nbsphere);
              cuda_data::ConfocalMicroscopeDeviceBuffers lens_buffers{};
              CU_INT *device_dipole_map=get_device_tabdip(Tabdip,mesh_count);
              const CUCOMPLEX *device_scattered=
                  get_device_result_vector_alias(FF,3u*dipole_count);
              if(!device_dipole_map ||
                 !cuda_data::get_device_confocal_microscope_buffers(
                     dipole_count,0u,3u*mesh_count,&lens_buffers) ||
                 !lens_buffers.dense_field) {
                  fluo_set_info(infostr,infostr_len,
                      "confocal fluorescence lens buffers unavailable");
                  *nstop=1;
                  clear_device_result_vector_alias(FF);
                  goto cleanup;
              }
              if(!device_scattered) {
                  if(cuda_data::copy_host_complex_to_device(
                         lens_buffers.dipole_field,FF,3u*dipole_count,
                         cuda_utils::TransferCategory::Result)!=cudaSuccess) {
                      fluo_set_info(infostr,infostr_len,
                          "confocal fluorescence scattered-field upload failed");
                      *nstop=1;
                      clear_device_result_vector_alias(FF);
                      goto cleanup;
                  }
                  device_scattered=lens_buffers.dipole_field;
              }
              if(!cuda_utils::check_cuda(microscope_expand_packed_field_wrapper(
                     device_scattered,device_dipole_map,lens_buffers.dense_field,
                     int(mesh_count),int(dipole_count)),
                     "confocal fluorescence lens","expand scattered field")) {
                  fluo_set_info(infostr,infostr_len,
                      "confocal fluorescence lens field expansion failed");
                  *nstop=1;
                  clear_device_result_vector_alias(FF);
                  goto cleanup;
              }
              int local_imax=*imaxk0;
              diffractefft2dlens_set_resident_input(lens_buffers.dense_field);
              diffractefft2dlens_set_resident_output(1);
              diffractefft2dlens_(nx,ny,nz,nxm,nym,nzm,nfft2d,tabfft2,k0,
                  xs,ys,zs,aretecube,Efourierx,Efouriery,Efourierz,FF,
                  &local_imax,&deltakx_local,&deltaky_local,Ediffkzneg,
                  &numaperk,nside,plan2f,
                  plan2b,nstop,infostr,infostr_len);
              diffractefft2dlens_set_resident_input(nullptr);
              diffractefft2dlens_set_resident_output(0);
              status=*nstop==1 ? 1 : 0;
              resident_diffraction=diffractefft2dlens_get_resident_output();
              if(status!=0 || !resident_diffraction) {
                  fluo_set_info(infostr,infostr_len,
                      "confocal fluorescence lens diffraction failed");
                  *nstop=1;
                  clear_device_result_vector_alias(FF);
                  goto cleanup;
              }
              *imaxk0=local_imax;
          }
          clear_device_result_vector_alias(FF);
          *deltakx=deltakx_local;
          const REAL deltaky=deltaky_local;
          for(int iz=0;iz<nplanz;++iz){
            const REAL zlens=*zinf+(*stepz)*(REAL)iz;
            cudaError_t cuda_status=confocal_fluorescence_build_wrapper(
                resident_diffraction,device.spectrum_planes,nf,*imaxk0,
                *deltakx,deltaky,*k0,numaperk,zlens,*nside);
            if(cuda_status==cudaSuccess)
                cuda_status=confocal_fluorescence_rotate_wrapper(
                    device.spectrum_planes,nf,*imaxk0,*deltakx,deltaky,
                    *k0,*gross,*nside,numaperk);
            if(cuda_status==cudaSuccess)
                cuda_status=confocal_fluorescence_rotate_wrapper(
                    device.spectrum_planes+3u*plane,nf,*imaxk0,*deltakx,deltaky,
                    *k0,*gross,*nside,numaperk);
            if(cuda_status!=cudaSuccess) {
                fluo_set_info(infostr,infostr_len,"confocal fluorescence CUDA spectrum failed");*nstop=1;goto cleanup;
            }
#if PRODUITMATVECTFFT_SINGLE_PRECISION
            cufftResult fft_status=cufftExecC2C(fft_plan,device.spectrum_planes,
                device.spectrum_planes,CUFFT_FORWARD);
            if(fft_status==CUFFT_SUCCESS)
                fft_status=cufftExecC2C(fft_plan,device.spectrum_planes+3u*plane,
                    device.spectrum_planes+3u*plane,CUFFT_FORWARD);
#else
            cufftResult fft_status=cufftExecZ2Z(fft_plan,device.spectrum_planes,
                device.spectrum_planes,CUFFT_FORWARD);
            if(fft_status==CUFFT_SUCCESS)
                fft_status=cufftExecZ2Z(fft_plan,device.spectrum_planes+3u*plane,
                    device.spectrum_planes+3u*plane,CUFFT_FORWARD);
#endif
            if(!cuda_utils::check_cufft(fft_status,"confocal fluorescence","forward 2-D cuFFT")) {
                fluo_set_info(infostr,infostr_len,"confocal fluorescence forward cuFFT failed");*nstop=1;goto cleanup;
            }
            cuda_status=fourier_to_image_rearrange_wrapper(
                device.spectrum_planes,nf,deltakx_local*deltaky/(*gross));
            if(cuda_status==cudaSuccess)
                cuda_status=confocal_fluorescence_phase_wrapper(
                    device.spectrum_planes+3u*plane,nf,*aretecube,
                    kxinc,kyinc,kzinc,zlens);
            if(cuda_status==cudaSuccess)
                cuda_status=confocal_fluorescence_accumulate_wrapper(
                    device.spectrum_planes+3u*plane,device.intensity_stack,
                    plane,(size_t)iz);
            if(cuda_status!=cudaSuccess) {
                fluo_set_info(infostr,infostr_len,"confocal fluorescence CUDA image accumulation failed");*nstop=1;goto cleanup;
            }
          }
        }
      }
    }
    if(confocal_fluorescence_square_wrapper(device.intensity_stack,cells)!=cudaSuccess ||
       cuda_data::copy_device_to_host(host_image.data(),device.intensity_stack,
          cells*sizeof(REAL),cuda_utils::TransferCategory::Result)!=cudaSuccess) {
        fluo_set_info(infostr,infostr_len,"confocal fluorescence result download failed");*nstop=1;goto cleanup;
    }
    image=host_image.data();
    deltax=2.0*pi/(REAL)nf/(*deltakx);
    if(*nmat==0&&fluo_write_ascii(nfft2d,&nplanz,imaxk0,ideltam,k0,&deltax,
            deltakx,deltak,zinf,stepz,masque,image)!=0){
        fluo_set_info(infostr,infostr_len,"confocal fluorescence ASCII output failed");
        *nstop=1;goto cleanup;
    }else if(*nmat==2){
        for(int iz=0;iz<nplanz;++iz)xy[iz]=*zinf+(*stepz)*(REAL)iz;
        for(int i=-half;i<half;++i)xy[i+half]=deltax*(REAL)i;
        for(int id=-*ideltam;id<=*ideltam;++id)
            kxy[id+*ideltam]=(*deltak)*(REAL)id/(*k0);
        for(int j=-*ideltam;j<=*ideltam;++j)
            for(int i=-*ideltam;i<=*ideltam;++i){
                const size_t q=(size_t)(i+*ideltam)+
                    (size_t)maskwidth*(size_t)(j+*ideltam);
                masque[q]=std::abs(masque[q]);
            }
    }
    middle=(size_t)(nplanze-1)*plane;
    for(size_t q=0;q<plane;++q)Eimagex[q]=image[middle+q];
    for(int i=0;i<nf;++i){const int k=i-half;xy[i]=deltax*(REAL)k*(*gross);kxy[i]=(*deltakx)*(REAL)k/(*k0);}
    *nstop=0;
cleanup:
    (void)numaperinc2;(void)psiinc;(void)trope;(void)Ediffkzpos;(void)Eimageincx;
}
