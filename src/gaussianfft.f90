!*****************************************************************
subroutine gaussianchampfft(xs,ys,zs,aretecube,k0,w0,E0,ss,pp,theta,phi,xgaus,ygaus,zgaus,beam,ndipole,nx,ny,nz, &
     nxm,nym,nzm,nmax,nfft2d,imagex,imagey,imagez,FF0,tolinit,plan2f,plan2b,nstop,infostr)
      use precision
      use iso_c_binding, only: c_int
      implicit none
      integer(ip), intent(in) :: ndipole,nx,ny,nz,nxm,nym,nzm,nmax,nfft2d
      integer(ip), intent(out) :: nstop
      integer(ip) :: i,j,k,l,kk,nxy,ii,jj,kkm,jjm,tabfft(nfft2d*nfft2d),comparaison,imaxk0,indicex,indicey,nloin,indice
      real(wp), intent(in) :: xs(nmax),ys(nmax),zs(nmax),k0,w0,ss,pp,theta,phi,aretecube,xgaus,ygaus,zgaus,tolinit
      real(wp) :: xmax,xmin,ymax,ymin,x,y,z,lambda,tol,tolc,Emod,Emodmax,deltakx,deltaky,kz,kx,ky
      complex(wp), intent(in) :: E0
      complex(wp), intent(out) :: FF0(3*nmax)
      complex(wp) :: imagex(nfft2d*nfft2d),imagey(nfft2d*nfft2d),imagez(nfft2d*nfft2d),ctmp,Ex,Ey,Ez,fac,eps0
      character(64), intent(in) :: beam
      character(64), intent(out) :: infostr
      integer(8), intent(in) :: plan2f, plan2b
      integer(ip), parameter :: FFTW_BACKWARD=+1, FFTW_FORWARD=-1
#ifdef IFDDA_CUDA_INVERSE_DRIVER
      integer(c_int) :: cuda_status,c_nfft,c_nx,c_ny,c_nz,c_nmax,c_circular
#endif
      interface
#ifdef IFDDA_CUDA_INVERSE_DRIVER
         subroutine ifdda_gaussianfft_cuda(nfft2d,nx,ny,nz,nmax,dipole_count,kkm,jjm, &
              dipole_xs,dipole_ys, &
              xmin,ymin,z,aretecube,x0,y0,z0,theta,phi,waist,k0,ss,pp, &
              e0,tolerance,circular,dipole_field,status) &
              bind(C,name="ifdda_gaussianfft_cuda_")
            use iso_c_binding, only: c_int
            use precision, only: wp
            integer(c_int), intent(in) :: nfft2d,nx,ny,nz,nmax,dipole_count
            integer(c_int), intent(in) :: kkm,jjm,circular
            real(wp), intent(in) :: xmin,ymin,z,aretecube
            real(wp), intent(in) :: dipole_xs(*),dipole_ys(*)
            real(wp), intent(in) :: x0,y0,z0
            real(wp), intent(in) :: theta,phi,waist,k0,ss,pp,tolerance
            complex(wp), intent(in) :: e0
            complex(wp), intent(out) :: dipole_field(*)
            integer(c_int), intent(out) :: status
         end subroutine ifdda_gaussianfft_cuda
#endif
      end interface

      lambda=twopi/k0
      nxy=nx*ny
      fac=zone/real(nfft2d*nfft2d,wp)
      eps0 = zone
      
!     arret si theta trop fort car ne prend que les kz positifs
      if (abs(theta) >= 45.0_wp) then
         infostr='Angle too large for fft gaussian'
         nstop=1
      endif

!     verifie taille fft
      if (nfft2d <= nx) then
         nstop=1
         infostr='nfft2d too small'
         return
      endif
      if (nfft2d <= ny) then
         nstop=1
         infostr='nfft2d too small'
         return
      endif

      deltakx=twopi/(real(nfft2d,wp)*aretecube)
      deltaky=twopi/(real(nfft2d,wp)*aretecube)

      if (deltakx >= k0/2.0_wp) then
         nstop=1
         infostr='nfft2d too small'
         return
      endif
      if (deltaky >= k0/2.0_wp) then
         nstop=1
         infostr='nfft2d too small'
         return
      endif
      if (nstop == 1) return

      xmax=-huge(1.0_wp)/100.0_wp
      xmin=huge(1.0_wp)/100.0_wp
      ymax=-huge(1.0_wp)/100.0_wp
      ymin=huge(1.0_wp)/100.0_wp

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC) REDUCTION(max:xmax,ymax) REDUCTION(min:xmin,ymin)
      do i=1,ndipole
         xmax=max(xmax,xs(i))
         xmin=min(xmin,xs(i))
         ymax=max(ymax,ys(i))
         ymin=min(ymin,ys(i))
      enddo
!$OMP ENDDO
!$OMP END PARALLEL

      kkm=(nfft2d-nx)/2
      jjm=(nfft2d-ny)/2

      Emodmax=zero
      z=zgaus
#ifndef IFDDA_CUDA_INVERSE_DRIVER
      if (beam(1:11) == 'gwavelinear' .or. beam(1:14) == 'gfftwavelinear') then
         do j=1,nfft2d
            do k=1,nfft2d
               x=xmin+real(k-kkm-1,wp)*aretecube
               y=ymin+real(j-jjm-1,wp)*aretecube
               call gaussianparalinear(x,y,z,xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,pp,eps0, E0,Ex,Ey,Ez,nstop,infostr)
               Emodmax=max(real(Ex*conjg(Ex)+Ey*conjg(Ey)+Ez*conjg(Ez),wp),Emodmax)
            enddo
         enddo
      endif
      if (beam(1:13) == 'gwavecircular' .or. beam(1:16) == 'gfftwavecircular') then
         do j=1,nfft2d
            do k=1,nfft2d
               x=xmin+real(k-kkm-1,wp)*aretecube
               y=ymin+real(j-jjm-1,wp)*aretecube
               call gaussianparacirc(x,y,z,xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,eps0, E0,Ex,Ey,Ez,nstop,infostr)
               Emodmax=max(real(Ex*conjg(Ex)+Ey*conjg(Ey)+Ez*conjg(Ez),wp),Emodmax)
            enddo
         enddo
      endif
#endif

      l=1
      z=zs(1)
      nloin=0
      tol=tolinit

!     intialise le premier z au champ incident avec la phase
#ifdef IFDDA_CUDA_INVERSE_DRIVER
!     La carte pixel-dipole est construite par un kernel CUDA plus bas.
#else
      do j=1,nfft2d
         do k=1,nfft2d
            ii=k+(j-1)*nfft2d
            x=xmin+real(k-kkm-1,wp)*aretecube
            y=ymin+real(j-jjm-1,wp)*aretecube

            if (beam(1:14) == 'gfftwavelinear') then
               call gaussianparalinear(x,y,z,xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,pp,eps0, E0,Ex,Ey,Ez,nstop,infostr)
               Emod=real(Ex*conjg(Ex)+Ey*conjg(Ey)+Ez*conjg(Ez),wp)
               tolc=sqrt(Emod/Emodmax)
               if (tolc >= tol) then
                  call gaussianchamp(x,y,z,xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,pp,E0,Imagex(ii),Imagey(ii),Imagez(ii),tol,nloin&
                       &,nstop,infostr)
               else
                  Imagex(ii)=Ex
                  Imagey(ii)=Ey
                  Imagez(ii)=Ez
               endif

            elseif (beam(1:16) == 'gfftwavecircular') then
               call gaussianparacirc(x,y,z,xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,eps0, E0,Ex,Ey,Ez,nstop,infostr)
               Emod=real(Ex*conjg(Ex)+Ey*conjg(Ey)+Ez*conjg(Ez),wp)
               tolc=sqrt(Emod/Emodmax)*10.0_wp
               if (tolc >= tol) then
                  call gaussianchampcirc(x,y,z,xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,E0,Imagex(ii),Imagey(ii),Imagez(ii),tol,nloin&
                       &,nstop,infostr)
               else
                  Imagex(ii)=Ex
                  Imagey(ii)=Ey
                  Imagez(ii)=Ez
               endif

            endif

            if (l <= ndipole) then
               if (comparaison(x,y,z,xs(l),ys(l),z,aretecube) == 1) then
                  tabfft(ii)=l
                  l=l+1
               else
                  tabfft(ii)=0
               endif
            else
               tabfft(ii)=0
            endif

         enddo
      enddo
#endif

#ifdef IFDDA_CUDA_INVERSE_DRIVER
!     Construire aussi la premiere image (paraxiale + correction rigoureuse)
!     sur GPU, puis conserver la propagation CUDA deja en place.
      c_nfft=int(nfft2d,c_int)
      c_nx=int(nx,c_int)
      c_ny=int(ny,c_int)
      c_nz=int(nz,c_int)
      c_nmax=int(nmax,c_int)
      c_circular=0_c_int
      if (beam(1:16) == 'gfftwavecircular') c_circular=1_c_int
      call ifdda_gaussianfft_cuda(c_nfft,c_nx,c_ny,c_nz,c_nmax, &
           int(ndipole,c_int),int(kkm,c_int),int(jjm,c_int),xs,ys, &
           xmin,ymin,z,aretecube,xgaus,ygaus,zgaus,theta,phi,w0,k0, &
           ss,pp,E0,tol,c_circular,FF0,cuda_status)
      nstop=int(cuda_status,ip)
      if (nstop /= 0) infostr='CUDA Gaussian FFT pipeline failed'
      return
#else

!     calcul premiere couche en filtrant les frequences hors propagatives
#ifdef USE_FFTW
#ifdef USE_SIMP
      call sfftw_execute_dft(plan2b,Imagex,Imagex)
      call sfftw_execute_dft(plan2b,Imagey,Imagey)
      call sfftw_execute_dft(plan2b,Imagez,Imagez) 
#else
      call dfftw_execute_dft(plan2b,Imagex,Imagex)
      call dfftw_execute_dft(plan2b,Imagey,Imagey)
      call dfftw_execute_dft(plan2b,Imagez,Imagez)
#endif     
  
#else
!$OMP PARALLEL DEFAULT(SHARED)
!$OMP SECTIONS
!$OMP SECTION
      call fftsingletonz2d(Imagex,nfft2d,nfft2d,FFTW_BACKWARD)
!$OMP SECTION
      call fftsingletonz2d(Imagey,nfft2d,nfft2d,FFTW_BACKWARD)
!$OMP SECTION
      call fftsingletonz2d(Imagez,nfft2d,nfft2d,FFTW_BACKWARD)
!$OMP END SECTIONS
!$OMP END PARALLEL
#endif

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,indice,ii,jj,kx,ky,kz)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
         do j=1,nfft2d
            do i=1,nfft2d
               indice=i+nfft2d*(j-1)
               if (i > nfft2d/2) then
                  ii=i-nfft2d-1
               else
                  ii=i-1
               endif
               if (j > nfft2d/2) then
                  jj=j-nfft2d-1
               else
                  jj=j-1
               endif
               kx=deltakx*real(ii,wp)
               ky=deltaky*real(jj,wp)
               kz=k0*k0-kx*kx-ky*ky
               if (kz <= zero) then
                  Imagex(indice)=zzero
                  Imagey(indice)=zzero
                  Imagez(indice)=zzero
               else

                  Imagex(indice)=Imagex(indice)*fac
                  Imagey(indice)=Imagey(indice)*fac
                  Imagez(indice)=Imagez(indice)*fac

               endif
            enddo
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

#ifdef USE_FFTW
#ifdef USE_SIMP
         call sfftw_execute_dft(plan2f,Imagex,Imagex)
         call sfftw_execute_dft(plan2f,Imagey,Imagey)
         call sfftw_execute_dft(plan2f,Imagez,Imagez)
#else
         call dfftw_execute_dft(plan2f,Imagex,Imagex)
         call dfftw_execute_dft(plan2f,Imagey,Imagey)
         call dfftw_execute_dft(plan2f,Imagez,Imagez)
#endif        
      
#else
!$OMP PARALLEL DEFAULT(SHARED)
!$OMP SECTIONS
!$OMP SECTION
         call fftsingletonz2d(Imagex,nfft2d,nfft2d,FFTW_FORWARD)
!$OMP SECTION
         call fftsingletonz2d(Imagey,nfft2d,nfft2d,FFTW_FORWARD)
!$OMP SECTION
         call fftsingletonz2d(Imagez,nfft2d,nfft2d,FFTW_FORWARD)
!$OMP END SECTIONS
!$OMP END PARALLEL
#endif

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,indice,x,y,ii,kk)
!$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
         do j=1,nfft2d
            do i=1,nfft2d
               indice=i+(j-1)*nfft2d
               x=xmin+real(i-kkm-1,wp)*aretecube
               y=ymin+real(j-jjm-1,wp)*aretecube

               if (tabfft(indice) /= 0) then

                  ii=tabfft(indice)
                  kk=3*(ii-1)

                  FF0(kk+1)=Imagex(indice)
                  FF0(kk+2)=Imagey(indice)
                  FF0(kk+3)=Imagez(indice)

               endif
            enddo
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

!     commence boucle sur les z
      do k=2,nz

!     FFT à deux dimensions
#ifdef USE_FFTW
#ifdef USE_SIMP
         call sfftw_execute_dft(plan2b,Imagex,Imagex)
         call sfftw_execute_dft(plan2b,Imagey,Imagey)
         call sfftw_execute_dft(plan2b,Imagez,Imagez)   
#else
         call dfftw_execute_dft(plan2b,Imagex,Imagex)
         call dfftw_execute_dft(plan2b,Imagey,Imagey)
         call dfftw_execute_dft(plan2b,Imagez,Imagez)
#endif        
     
#else
!$OMP PARALLEL DEFAULT(SHARED)
!$OMP SECTIONS
!$OMP SECTION
         call fftsingletonz2d(Imagex,nfft2d,nfft2d,FFTW_BACKWARD)
!$OMP SECTION
         call fftsingletonz2d(Imagey,nfft2d,nfft2d,FFTW_BACKWARD)
!$OMP SECTION
         call fftsingletonz2d(Imagez,nfft2d,nfft2d,FFTW_BACKWARD)
!$OMP END SECTIONS
!$OMP END PARALLEL
#endif

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,indice,ii,jj,kx,ky,kz,ctmp)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
         do j=1,nfft2d
            do i=1,nfft2d
               indice=i+nfft2d*(j-1)
               if (i > nfft2d/2) then
                  ii=i-nfft2d-1
               else
                  ii=i-1
               endif
               if (j > nfft2d/2) then
                  jj=j-nfft2d-1
               else
                  jj=j-1
               endif
               kx=deltakx*real(ii,wp)
               ky=deltaky*real(jj,wp)
               kz=k0*k0-kx*kx-ky*ky
               if (kz >= zero) then
                  kz=sqrt(kz)
                  ctmp=exp(icomp*kz*aretecube)
                  Imagex(indice)=Imagex(indice)*ctmp*fac
                  Imagey(indice)=Imagey(indice)*ctmp*fac
                  Imagez(indice)=Imagez(indice)*ctmp*fac
               else
                  Imagex(indice)=zzero
                  Imagey(indice)=zzero
                  Imagez(indice)=zzero

               endif

            enddo
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

#ifdef USE_FFTW
#ifdef USE_SIMP
         call sfftw_execute_dft(plan2f,Imagex,Imagex)
         call sfftw_execute_dft(plan2f,Imagey,Imagey)
         call sfftw_execute_dft(plan2f,Imagez,Imagez)
#else
         call dfftw_execute_dft(plan2f,Imagex,Imagex)
         call dfftw_execute_dft(plan2f,Imagey,Imagey)
         call dfftw_execute_dft(plan2f,Imagez,Imagez)
#endif
        
#else
!$OMP PARALLEL DEFAULT(SHARED)
!$OMP SECTIONS
!$OMP SECTION
         call fftsingletonz2d(Imagex,nfft2d,nfft2d,FFTW_FORWARD)
!$OMP SECTION
         call fftsingletonz2d(Imagey,nfft2d,nfft2d,FFTW_FORWARD)
!$OMP SECTION
         call fftsingletonz2d(Imagez,nfft2d,nfft2d,FFTW_FORWARD)
!$OMP END SECTIONS
!$OMP END PARALLEL
#endif

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,indice,x,y,ii,kk)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
         do j=1,nfft2d
            do i=1,nfft2d
               indice=i+(j-1)*nfft2d
               x=xmin+real(i-kkm-1,wp)*aretecube
               y=ymin+real(j-jjm-1,wp)*aretecube

               if (tabfft(indice) /= 0) then

                  ii=nxy*(k-1)+tabfft(indice)
                  kk=3*(ii-1)

                  FF0(kk+1)=Imagex(indice)
                  FF0(kk+2)=Imagey(indice)
                  FF0(kk+3)=Imagez(indice)

               endif
            enddo
         enddo
!$OMP ENDDO
!$OMP END PARALLEL

      enddo

      write(*,*) 'fin fft gaussian'
#endif
      end subroutine gaussianchampfft
