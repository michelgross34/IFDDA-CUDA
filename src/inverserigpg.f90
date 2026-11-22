      subroutine inverserigpg(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
           FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz,tabdip, &
           ntotalm,ntotal,ldabi,nlar,nmax,ndipole,nxm,nym,nzm,nx,ny,nz, &
           nx2,ny2,nxy2,nz2,nbsphere,nbsphere3,XI,XR,wrk,FF,FF0,FFloc, &
           FFprecon,polarisa,niso,methodeit,tol,tol1,nloop,ncompte, &
           nlim,k0,aretecube,chanxxvect,chanxyvect,chanxzvect, &
           chanyyvect,chanyzvect,chanzzvect,sdetnn,planf,planb, &
           nstop,infostr)
!
      use precision
      implicit none
!
      integer(ip) :: nmax,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2
      integer(ip) :: ntotal, ntotalm
      integer(ip) :: ncompte,nt,ldabi,nlar,i,ii,jj,k,nbsphere,nbsphere3,nstop,ndipole
      integer(ip) :: NLIM,ndim,nou,nstat,nloop,STEPERR,is,nr,nu,np,ns,nq,LL,JL,niso
      real(wp) :: NORM,TOL,tol1,tole,tmp,k0,aretecube,RESIDU
      complex(wp) :: ALPHA,BETA,SIGMA,GPETA,DZETA,R0RN,QMR1,QMR2,QMR3, &
                     QMR4,QMR5,QMR6,QMR7,QMR8,QMR9,DOTS1,DOTS2,DOTS3, &
                     DOTS4,OMEGA,RHO,ctmp1,ctmp2
!
      integer(ip) :: nmaxcompo, ntest
      integer(ip), dimension(nxm*nym*nzm) :: tabdip
      complex(wp), dimension(:), allocatable :: vectxx,vectyy,vectzz
      complex(wp), dimension(3*nxm*nym*nzm) :: xr,xi
      complex(wp), dimension(3*nxm*nym*nzm,nlar) :: wrk
      complex(wp), dimension(3*nxm*nym*nzm) :: FF,FF0,FFloc,FFprecon
      complex(wp), dimension((nxm+1)*(nym+1)*(nzm+1)) :: FFTTENSORxx, &
           FFTTENSORxy,FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,FFTTENSORzz
      complex(wp), dimension(nx2*ny2*nz2) :: vectx,vecty,vectz
      complex(wp) :: chanxxvect(nxm*nym*nzm), chanxyvect(nxm*nym*nzm), &
                     chanxzvect(nxm*nym*nzm), chanyyvect(nxm*nym*nzm), &
                     chanyzvect(nxm*nym*nzm), chanzzvect(nxm*nym*nzm)
      complex(wp), dimension(nxm*nym*nzm,niso,niso) :: polarisa
      complex(wp), dimension(3*nzm,3*nzm,nxm*nym) :: sdetnn
      integer(ip), dimension(:,:), allocatable :: ipvtnn
      character(12) :: methodeit
      character(64) :: infostr
      integer*8 :: planf,planb,plan2b,plan2f
      integer(ip) :: FFTW_FORWARD, FFTW_ESTIMATE, FFTW_BACKWARD
#ifdef IFDDA_CUDA_INVERSE_DRIVER
      integer(ip) :: cuda_status, cuda_disabled
      integer(ip), external :: ifdda_set_resident_chan_preconditioner_all
      integer(ip), external :: chan_precon_pol_build_lu_cuda
#endif
!
      write(*,*) 'Left preconditionner used non optimized'
!
!     Initialise les tableaux pour preconditionner
#ifndef IFDDA_CUDA_INVERSE_DRIVER
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
      do i=1,nbsphere3
         FFprecon(i)=FF0(i)
         wrk(i,1)=xi(i)
      enddo
!$OMP ENDDO
!$OMP END PARALLEL
#endif
!
      nmaxcompo=nx*ny
      allocate(vectxx(1:nmaxcompo))
      allocate(vectyy(1:nmaxcompo))
      allocate(vectzz(1:nmaxcompo))
      allocate(ipvtnn(nz*3,nmaxcompo))
!
      FFTW_FORWARD=-1
      FFTW_BACKWARD=+1
      FFTW_ESTIMATE=64
!
#ifdef IFDDA_CUDA_INVERSE_DRIVER
!     CUDA Chan setup and applications use resident GPU FFTs, not host FFTW.
      plan2b=0
      plan2f=0
#else
#ifdef USE_FFTW
#ifdef USE_SIMP
      call sfftw_plan_dft_2d(plan2b,nx,ny,vectxx,vectxx,FFTW_BACKWARD,FFTW_ESTIMATE)
      call sfftw_plan_dft_2d(plan2f,nx,ny,vectxx,vectxx,FFTW_FORWARD,FFTW_ESTIMATE)
#else
      call dfftw_plan_dft_2d(plan2b,nx,ny,vectxx,vectxx,FFTW_BACKWARD,FFTW_ESTIMATE)
      call dfftw_plan_dft_2d(plan2f,nx,ny,vectxx,vectxx,FFTW_FORWARD,FFTW_ESTIMATE)
#endif    
#endif
#endif
!
      ntest=0
      write(*,*) 'Create Chan s matrix'
#ifdef IFDDA_CUDA_INVERSE_DRIVER
!     Build the Chan Green tensor, blocks and LU factors on the GPU. The
!     resident inverse driver applies P^-1 to the RHS and each Ax product.
      cuda_status=chan_precon_pol_build_lu_cuda(polarisa,ndipole,aretecube, &
           k0,nx,ny,nz,nxm,nym,nzm)
      if (cuda_status /= 0) then
         nstop=1
         infostr='CUDA Chan matrix generation/factorization failed'
         return
      endif
      nstop=0
#else
      call preconditionneur(FFprecon,xi,xr,tabdip,polarisa,niso, &
           chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
           nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
           sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
#endif
!
!     calcul P-1 FF0
      ntest=1
#ifdef IFDDA_CUDA_INVERSE_DRIVER
!     Apply the initial left-Chan transform after the CUDA driver uploads FF0.
!     Keeping it resident avoids downloading the transformed RHS here and
!     uploading it again when the iterative GPU solver starts.
      nstop=0
#else
      call preconditionneur(FFprecon,xi,xr,tabdip,polarisa,niso, &
           chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
           nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
           sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
#endif
      write(*,*) 'Compute the new incident field done'
!
#ifndef IFDDA_CUDA_INVERSE_DRIVER
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
      do i=1,nbsphere3
         xi(i)=wrk(i,1)
      enddo
!$OMP ENDDO
!$OMP END PARALLEL
#endif
!
      if (nstop == 1) return
#ifdef IFDDA_CUDA_INVERSE_DRIVER
!     Match the solver set accepted by this Fortran left-Chan driver. The
!     unpreconditioned CUDA dispatcher also implements GCORS2 and Born series,
!     but the Fortran preconditioned branch does not.
      if (methodeit(1:7) == 'GPBICG1' .or. &
          methodeit(1:7) == 'GPBICG2' .or. &
          methodeit(1:10) == 'GPBICGsafe' .or. &
          methodeit(1:10) == 'GPBICGplus' .or. &
          methodeit(1:8) == 'GPBICGAR' .or. &
          methodeit(1:12) == 'BICGSTARPLUS' .or. &
          methodeit(1:5) == 'TFQMR' .or. &
          methodeit(1:2) == 'CG' .or. &
          methodeit(1:8) == 'BICGSTAB' .or. &
          methodeit(1:12) == 'QMRBICGSTAB1' .or. &
          methodeit(1:12) == 'QMRBICGSTAB2' .or. &
          methodeit(1:7) == 'GPBICOR' .or. &
          methodeit(1:4) == 'CORS' .or. &
          methodeit(1:4) == 'IDRS' .or. &
          methodeit(1:9) == 'BICGLSTAB' .or. &
          methodeit(1:11) == 'GPBICGSTABL') then
         cuda_status=ifdda_set_resident_chan_preconditioner_all(1,nx,ny,nz,nxm,nym,nzm)
         if (cuda_status /= 0) then
            nstop=1
            infostr='Unable to activate resident Chan preconditioner'
            return
         endif
         call inverserig_all_cuda_driver(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz,tabdip, &
              ntotalm,ntotal,ldabi,nlar,nmax,ndipole,nxm,nym,nzm,nx,ny,nz, &
              nx2,ny2,nxy2,nz2,nbsphere,nbsphere3,xi,xr,wrk,FF,FF0, &
              FFloc,polarisa,niso,methodeit,tol,tol1,nloop,ncompte,nlim, &
              planf,planb,planf,planb,planf,planb,planf,planb,0,-1,1,nstop,infostr)
         cuda_disabled=ifdda_set_resident_chan_preconditioner_all(0,nx,ny,nz,nxm,nym,nzm)
         if (cuda_disabled /= 0 .and. nstop == 0) then
            nstop=1
            infostr='Unable to deactivate resident Chan preconditioner'
         endif
         return
      endif
#endif
!
      if (nstop == 1) return
!
      nou=0
      ndim=nbsphere3
!
      if (methodeit(1:7) == 'GPBICG1') then
 2002    call GPBICG(XI,XR,FFprecon,ldabi,ndim,nlar,nou,WRK,NLOOP,Nlim, &
              TOL,NORM,ALPHA,BETA,GPETA,DZETA,R0RN,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat /= 1) goto 2002
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
!     as we begin with ITNO=-1
         nloop=nloop+1
!
      elseif (methodeit(1:7) == 'GPBICG2') then
 2009    call GPBICG2(XI,XR,FFprecon,ldabi,ndim,nlar,nou,WRK,NLOOP,Nlim, &
              TOL,NORM,ALPHA,BETA,GPETA,DZETA,R0RN,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat /= 1) goto 2009
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            FFloc(i)=FF(i)/polarisa(1,1,1)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
         nloop=nloop+1
!
      elseif (methodeit(1:10) == 'GPBICGsafe') then
 2019    call GPBICGsafe(XI,XR,FFprecon,ldabi,ndim,nlar,nou,WRK,NLOOP, &
              Nlim,TOL,NORM,ALPHA,BETA,GPETA,DZETA,R0RN,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat /= 1) goto 2019
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
         nloop=nloop+1
!
      elseif (methodeit(1:10) == 'GPBICGplus') then
 2016    call GPBICGplus(XI,XR,FFprecon,ldabi,ndim,nlar,nou,WRK,NLOOP, &
              Nlim,TOL,NORM,ALPHA,BETA,GPETA,DZETA,R0RN,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat /= 1) goto 2016
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
         nloop=nloop+1
!
      elseif (methodeit(1:8) == 'GPBICGAR') then
 2010    call GPBICGAR(XI,XR,FFprecon,ldabi,ndim,nlar,nou,WRK,NLOOP,Nlim, &
              TOL,NORM,ALPHA,BETA,GPETA,DZETA,R0RN,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat /= 1) goto 2010
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
         nloop=nloop+1
!
      elseif (methodeit(1:9) == 'GPBICGAR2') then
 2011    call GPBICGAR2(XI,XR,FFprecon,ldabi,ndim,nlar,nou,WRK,NLOOP, &
              Nlim,TOL,NORM,ALPHA,BETA,GPETA,DZETA,R0RN,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat /= 1) goto 2011
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
         nloop=nloop+1
!
      elseif (methodeit(1:12) == 'BICGSTARPLUS') then
 2015    call GPBICGSTARPLUS(XI,XR,FFprecon,ldabi,ndim,nlar,nou,WRK, &
              NLOOP,Nlim,TOL,NORM,ALPHA,BETA,GPETA,DZETA,R0RN, &
              NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat /= 1) goto 2015
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
         nloop=nloop+1
!
      elseif (methodeit(1:5) == 'TFQMR') then
 2004    call TFQMR(FFloc,Xi,XR,FFprecon,ldabi,ndim,nlar,nou,WRK,nloop, &
              NLIM,TOL,NORM,QMR1,QMR2,QMR3,QMR4,QMR5,QMR6,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               xi(i)=FFloc(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstat /= 1) goto 2004
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
         nloop=nloop+1
!
      elseif (methodeit(1:5) == 'CG') then
 2005    call ZCG(XI,XR,FFprecon,NORM,WRK,QMR1,QMR2,QMR3,LDABI,NDIM,NLAR, &
              NOU,NSTAT,NLOOP,NLIM,TOLE,TOL)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstat /= 1) goto 2005
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
!
      elseif (methodeit(1:8) == 'BICGSTAB') then
 2006    call PIMZBICGSTAB(FFLOC,Xi,XR,FFprecon,ldabi,nlar,ndim,nou,WRK, &
              QMR1,QMR2,QMR3,NORM,TOL,nloop,nlim,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               xi(i)=FFLOC(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstat /= 1) goto 2006
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
!
      elseif (methodeit(1:12) == 'QMRBICGSTAB1') then
         nt=1
 2007    call QMRBICGSTAB(FFloc,Xi,XR,FFprecon,ldabi,ndim,nlar,nou,WRK, &
              nloop,nlim,TOL,TOLE,NORM,QMR1,QMR2,QMR3,QMR4,QMR5, &
              QMR6,QMR7,QMR8,QMR9,NT,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               xi(i)=FFLOC(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstat /= 1) goto 2007
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
!
      elseif (methodeit(1:12) == 'QMRBICGSTAB2') then
         nt=2
 2008    call QMRBICGSTAB(FFloc,Xi,XR,FFprecon,ldabi,ndim,nlar,nou,WRK, &
              nloop,nlim,TOL,TOLE,NORM,QMR1,QMR2,QMR3,QMR4,QMR5, &
              QMR6,QMR7,QMR8,QMR9,NT,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               xi(i)=FFLOC(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstat /= 1) goto 2008
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
!
      elseif (methodeit(1:7) == 'GPBICOR') then
 2013    call GPBICOR(XI,XR,FFprecon,FFloc,ldabi,ndim,nlar,nou,WRK, &
              NLOOP,Nlim,TOL,NORM,ALPHA,BETA,GPETA,DZETA,QMR1, &
              QMR2,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat /= 1) goto 2013
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
!
      elseif (methodeit(1:7) == 'CORS') then
 2014    call CORS(XI,XR,FFprecon,ldabi,ndim,nlar,nou,WRK,NLOOP,Nlim,TOL, &
              NORM,ALPHA,GPETA,DZETA,BETA,NSTAT,STEPERR)
!
         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               FFloc(i)=xi(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!
         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)
!
         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)
!
         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)
!
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif
!
         if (nstat /= 1) goto 2014
!     compute the Residue
         tol1=zero
!
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL
!
         tol1=sqrt(tol1)/NORM
         nloop=nloop+1
!
      elseif (methodeit(1:4) == 'IDRS') then
         if (methodeit(1:5) == 'IDRS2') then
            ns=2
         elseif (methodeit(1:5) == 'IDRS4') then
            ns=4
         elseif (methodeit(1:5) == 'IDRS8') then
            ns=8
         endif

 2020    call IDRS(FFloc,Xi,XR,FFprecon,ldabi,ndim,nlar,nou,ns,WRK,NLOOP, &
              NLIM,TOL,RESIDU,NORM,ALPHA,is,NSTAT,STEPERR)

         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif

         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               xi(i)=FFloc(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)

         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)

         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)

         if (nstat /= 1) goto 2020
!     compute the Residue
         tol1=zero

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

         tol1=sqrt(tol1)/NORM
!     as we begin with ITNO=-1
         nloop=nloop+1

      elseif (methodeit(1:9) == 'BICGLSTAB') then
         if (methodeit(1:10) == 'BICGLSTAB2') then
            ns=2
         elseif (methodeit(1:10) == 'BICGLSTAB4') then
            ns=4
         elseif (methodeit(1:10) == 'BICGLSTAB8') then
            ns=8
         endif

 2021    call ZBICGSTABL(XI,XR,FFprecon,FFloc,NORM,WRK,ALPHA,RHO,OMEGA, &
              ns,nr,nu,JJ,LDABI,NDIM,NLAR,NOU,NSTAT,STEPERR,NLOOP,NLIM, &
              TOLE,TOL)

         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif

         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               xi(i)=FFloc(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)

         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)

         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)

         if (nstat /= 1) goto 2021
!     compute the Residue
         tol1=zero

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

         tol1=sqrt(tol1)/NORM
!     as we begin with ITNO=-1
         nloop=nloop+1

      elseif (methodeit(1:11) == 'GPBICGSTABL') then
         if (methodeit(1:12) == 'GPBICGSTABL2') then
            LL=2
         elseif (methodeit(1:12) == 'GPBICGSTABL4') then
            LL=4
         elseif (methodeit(1:12) == 'GPBICGSTABL8') then
            LL=8
         endif

 2022    call GPBICGSTABL(XI,XR,FFprecon,FFloc,NORM,WRK,ALPHA,BETA,SIGMA, &
              RHO,LL,LDABI,NDIM,NLAR,NOU,nr,np,ns,nq,JL,NSTAT,STEPERR, &
              NLOOP,NLIM,TOLE,TOL)

         if (nstat < 0) then
            nstop=1
            infostr='Problem to solve Ax=b!'
            write(*,*) 'Problem to solve Ax=b',nstat,STEPERR
            return
         endif
         ncompte=ncompte+1
         if (nstop == -1) then
            infostr = 'Calculation cancelled during iterative method'
            return
         endif

         if (nstat == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
            do i=1,nbsphere3
               xi(i)=FFloc(i)
            enddo
!$OMP ENDDO
!$OMP END PARALLEL
         endif
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xi(i)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

         call produitpolavect(polarisa,niso,nmax,nbsphere,xr,xi)

         call produitfftmatvect3(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
              tabdip,ntotalm,ntotal,nmax,ndipole,nxm,nym,nzm,nx,ny, &
              nz,nx2,ny2,nxy2,nz2,XI,XR,planf,planb)

         call preconditionneur(xr,FF,FFloc,tabdip,polarisa,niso, &
              chanxxvect,chanxyvect,chanxzvect,chanyyvect,chanyzvect,chanzzvect, &
              nx,ny,nz,nbsphere,ndipole,nxm,nym,nzm,aretecube,k0,ntest, &
              sdetnn,ipvtnn,vectxx,vectyy,vectzz,plan2b,plan2f,infostr,nstop)

         if (nstat /= 1) goto 2022
!     compute the Residue
         tol1=zero

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:tol1)
!$OMP DO SCHEDULE(STATIC)
         do i=1,nbsphere3
            xr(i)=xr(i)-FFprecon(i)
            FF(i)=xi(i)
            tol1=tol1+real(xr(i)*conjg(xr(i)),wp)
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

         tol1=sqrt(tol1)/NORM
!     as we begin with ITNO=-1
         nloop=nloop+1

      else
         write(*,*) 'Iterative method not correct',methodeit
         nstop=1
         infostr='Iterative method not correct!'
         return
      endif

      deallocate(vectxx)
      deallocate(vectyy)
      deallocate(vectzz)
      deallocate(ipvtnn)

      end subroutine inverserigpg
