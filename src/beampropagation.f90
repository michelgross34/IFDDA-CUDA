subroutine beampropagation(xs,ys,zs,aretecube,k0,w0,E0,ss,pp,theta, &
     phi,xgaus,ygaus,zgaus,beam,epsilon,niso,ndipole,nx,ny,nz,nxm, &
     nym,nzm,nmax,nfft2d,imagex,imagey,imagez,FF0,FFloc,FF, &
     plan2f,plan2b,nstop,infostr,resident_mode)

  use precision
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  use, intrinsic :: iso_c_binding, only: c_int, c_float, c_double, &
       c_float_complex, c_double_complex
#endif
  implicit none

  integer(ip) :: i,j,k,l,kk,nfft2d,ii,jj,nstop,kkm,jjm
  integer(ip) :: tabfft(nfft2d*nfft2d),comparaison,imaxk0
  integer(ip) :: indicex,indicey,nloin,niso
  complex(wp) :: imagex(nfft2d*nfft2d),imagey(nfft2d*nfft2d)
  complex(wp) :: imagez(nfft2d*nfft2d),ctmp,E0,expik0a
  real(wp) :: xmax,xmin,ymax,ymin,x,y,z,xgaus,ygaus,zgaus
  real(wp) :: lambda,tol
  character(64) :: infostr
  character(64) :: beam
  integer(ip), intent(in) :: resident_mode
  !     input data
  integer(ip) :: nx,ny,nz,nxm,nym,nzm,nmax,ndipole,indice
  real(wp) :: xs(nmax),ys(nmax),zs(nmax),k0,w0,ss,pp,theta,phi
  real(wp) :: aretecube,kz,kx,ky,deltakx,deltaky,kzc
  real(wp) :: var1,var2

  complex(wp) :: epsilon(nmax,niso,niso),FFloc(3*nmax),FF(3*nmax)
  complex(wp) :: FF0(3*nmax),Ex,Ey,Ez,fac,eps0
  integer(kind=8) :: plan2f,plan2b
  integer(ip) :: FFTW_BACKWARD, FFTW_FORWARD
  real(wp) :: Emod,Emodmax,tolc
  integer(ip) :: trace_status
  integer(ip) :: trace_occupied,trace_rigorous
  real(wp) :: trace_threshold_min,trace_threshold_max
  logical :: trace_bpm
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  integer(c_int) :: cuda_macro_mode
  integer(c_int) :: cuda_beam_type
#endif

#ifdef IFDDA_CUDA_INVERSE_DRIVER
  interface
     subroutine ifdda_beampropagation_propagate_cuda(imagex,imagey,imagez,tabfft, &
          epsilon,ffloc,nx,ny,nz,nfft2d,nmax,niso,ndipole,aretecube,k0, &
          macro_mode,nstop) &
          bind(C,name="ifdda_beampropagation_propagate_cuda_")
       import :: c_int, c_float, c_double, c_float_complex, c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(inout) :: imagex(*),imagey(*),imagez(*)
       complex(c_float_complex), intent(in) :: epsilon(*)
       complex(c_float_complex), intent(inout) :: ffloc(*)
       real(c_float), intent(in) :: aretecube,k0
#else
       complex(c_double_complex), intent(inout) :: imagex(*),imagey(*),imagez(*)
       complex(c_double_complex), intent(in) :: epsilon(*)
       complex(c_double_complex), intent(inout) :: ffloc(*)
       real(c_double), intent(in) :: aretecube,k0
#endif
       integer(c_int), intent(in) :: tabfft(*)
       integer(c_int), intent(in) :: nx,ny,nz,nfft2d,nmax,niso,ndipole,macro_mode
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_beampropagation_propagate_cuda

     subroutine ifdda_beampropagation_initialize_plane_cuda(xs,ys,zs,epsilon, &
          FF0,imagex,imagey,imagez,tabfft,FFloc,nx,ny,nfft2d,nmax,niso, &
          ndipole,kkm,jjm,xmin,ymin,first_z,aretecube,k0,E0,ss,pp,theta, &
          phi,xgaus,ygaus,zgaus,w0,beam_type,nstop) &
          bind(C,name="ifdda_beampropagation_initialize_plane_cuda_")
       import :: c_int, c_float, c_double, c_float_complex, c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),xmin,ymin,first_z
       real(c_float), intent(in) :: aretecube,k0,ss,pp,theta,phi
       real(c_float), intent(in) :: xgaus,ygaus,zgaus,w0
       complex(c_float_complex), intent(in) :: epsilon(*),FF0(*),E0
       complex(c_float_complex), intent(out) :: imagex(*),imagey(*),imagez(*)
       complex(c_float_complex), intent(inout) :: FFloc(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),xmin,ymin,first_z
       real(c_double), intent(in) :: aretecube,k0,ss,pp,theta,phi
       real(c_double), intent(in) :: xgaus,ygaus,zgaus,w0
       complex(c_double_complex), intent(in) :: epsilon(*),FF0(*),E0
       complex(c_double_complex), intent(out) :: imagex(*),imagey(*),imagez(*)
       complex(c_double_complex), intent(inout) :: FFloc(*)
#endif
       integer(c_int), intent(out) :: tabfft(*)
       integer(c_int), intent(in) :: nx,ny,nfft2d,nmax,niso,ndipole,kkm,jjm,beam_type
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_beampropagation_initialize_plane_cuda

     subroutine ifdda_beampropagation_run_cuda(xs,ys,zs,epsilon,FF0,FFloc, &
          nx,ny,nz,nfft2d,nmax,niso,ndipole,kkm,jjm,xmin,ymin,first_z, &
          aretecube,k0,E0,ss,pp,theta,phi,xgaus,ygaus,zgaus,w0,beam_type, &
          resident_mode,nstop) bind(C,name="ifdda_beampropagation_run_cuda_")
       import :: c_int, c_float, c_double, c_float_complex, c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),xmin,ymin,first_z
       real(c_float), intent(in) :: aretecube,k0,ss,pp,theta,phi
       real(c_float), intent(in) :: xgaus,ygaus,zgaus,w0
       complex(c_float_complex), intent(in) :: epsilon(*),FF0(*),E0
       complex(c_float_complex), intent(out) :: FFloc(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),xmin,ymin,first_z
       real(c_double), intent(in) :: aretecube,k0,ss,pp,theta,phi
       real(c_double), intent(in) :: xgaus,ygaus,zgaus,w0
       complex(c_double_complex), intent(in) :: epsilon(*),FF0(*),E0
       complex(c_double_complex), intent(out) :: FFloc(*)
#endif
       integer(c_int), intent(in) :: nx,ny,nz,nfft2d,nmax,niso,ndipole,kkm,jjm,beam_type
       integer(c_int), value :: resident_mode
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_beampropagation_run_cuda
  end interface
#endif

  call get_environment_variable('IFDDA_BPM_TRACE', status=trace_status)
  trace_bpm = trace_status == 0
  FFTW_BACKWARD = +1
  FFTW_FORWARD  = -1

  eps0=zone
  lambda = twopi/k0
  expik0a = exp(icomp*k0*aretecube)
  fac     = one/real(nfft2d*nfft2d,wp)

  !     verifie taille fft
  if (nfft2d <= nx) then
     nstop  = 1
     infostr = 'nfft2d too small'
  end if
  if (nfft2d <= ny) then
     nstop  = 1
     infostr = 'nfft2d too small'
  end if

  deltakx = 2._wp*pi/(real(nfft2d,wp)*aretecube)
  deltaky = 2._wp*pi/(real(nfft2d,wp)*aretecube)
  var1    = (real(nfft2d/2,wp)*aretecube)*deltakx - aretecube*deltakx/2._wp
  var2    = (real(nfft2d/2,wp)*aretecube)*deltaky - aretecube*deltakx/2._wp

  if (deltakx >= k0/2._wp) then
     nstop  = 1
     infostr = 'nfft2d too small'
  end if
  if (deltaky >= k0/2._wp) then
     nstop  = 1
     infostr = 'nfft2d too small'
  end if
  if (nstop == 1) return

  xmax = -huge(1.0_wp)/100.0_wp
  xmin =  huge(1.0_wp)/100.0_wp
  ymax = -huge(1.0_wp)/100.0_wp
  ymin =  huge(1.0_wp)/100.0_wp

  !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
  !$OMP DO SCHEDULE(STATIC) REDUCTION(max:xmax,ymax) REDUCTION(min:xmin,ymin)
  do i = 1,ndipole
     xmax = max(xmax,xs(i))
     xmin = min(xmin,xs(i))
     ymax = max(ymax,ys(i))
     ymin = min(ymin,ys(i))
  end do
  !$OMP ENDDO
  !$OMP END PARALLEL

  kkm = (nfft2d-nx)/2
  jjm = (nfft2d-ny)/2

#ifdef IFDDA_CUDA_INVERSE_DRIVER
  ! Plane, paraxial, and rigorous Gaussian beams use the GPU BPM path.
  if (beam(1:11) == 'pwavelinear' .or. beam(1:13) == 'pwavecircular' .or. &
      beam(1:15) == 'gparawavelinear' .or. beam(1:17) == 'gparawavecircular' .or. &
      beam(1:11) == 'gwavelinear' .or. beam(1:13) == 'gwavecircular' .or. &
      beam(1:14) == 'gfftwavelinear' .or. beam(1:16) == 'gfftwavecircular') then
     if (beam(1:11) == 'pwavelinear') then
        cuda_beam_type = 0
     else if (beam(1:13) == 'pwavecircular') then
        cuda_beam_type = 1
     else if (beam(1:17) == 'gparawavecircular') then
        cuda_beam_type = 3
     else if (beam(1:13) == 'gwavecircular' .or. &
              beam(1:16) == 'gfftwavecircular') then
        cuda_beam_type = 5
     else if (beam(1:11) == 'gwavelinear' .or. &
              beam(1:14) == 'gfftwavelinear') then
        cuda_beam_type = 4
     else
        cuda_beam_type = 2
     end if
     if (trace_bpm) then
        call ifdda_beampropagation_initialize_plane_cuda(xs,ys,zs,epsilon, &
             FF0,Imagex,Imagey,Imagez,tabfft,FFloc,nx,ny,nfft2d,nmax,niso, &
             ndipole,kkm,jjm,xmin,ymin,zs(1),aretecube,k0,E0,ss,pp, &
             theta,phi,xgaus,ygaus,zgaus,w0,cuda_beam_type,nstop)
        if (nstop /= 0) then
           infostr = 'CUDA beam plane initialization failed'
           return
        end if
        cuda_macro_mode = 0
        call ifdda_beampropagation_propagate_cuda(Imagex,Imagey,Imagez,tabfft, &
             epsilon,FFloc,nx,ny,nz,nfft2d,nmax,niso,ndipole,aretecube,k0, &
             cuda_macro_mode,nstop)
     else
        cuda_macro_mode = int(resident_mode,c_int)
        call ifdda_beampropagation_run_cuda(xs,ys,zs,epsilon,FF0,FFloc, &
             nx,ny,nz,nfft2d,nmax,niso,ndipole,kkm,jjm,xmin,ymin,zs(1), &
             aretecube,k0,E0,ss,pp,theta,phi,xgaus,ygaus,zgaus,w0, &
             cuda_beam_type,cuda_macro_mode,nstop)
     end if
     if (nstop /= 0) infostr = 'CUDA beam propagation failed'
     return
  else
     nstop = 1
     infostr = 'Incident beam is not available in the CUDA propagation path'
     return
  end if
#else

  Emodmax = zero
  trace_occupied = 0
  trace_rigorous = 0
  trace_threshold_min = huge(1.0_wp)
  trace_threshold_max = zero
  if (beam(1:11) == 'gwavelinear' .or. beam(1:14) == &
       &   'gfftwavelinear') then
     do j = 1,nfft2d
        do k = 1,nfft2d
           x = xmin + real(k-kkm-1,wp)*aretecube
           y = ymin + real(j-jjm-1,wp)*aretecube
           call gaussianparalinear(x,y,z,xgaus,ygaus,zgaus,theta, phi,w0,k0,ss,pp,eps0, E0,Ex,Ey,Ez,nstop,infostr)
           Emodmax = max(real(Ex*conjg(Ex)+Ey*conjg(Ey)+Ez*conjg(Ez),wp),Emodmax)
        end do
     end do
  end if
  if (beam(1:13) == 'gwavecircular' .or. beam(1:16) == &
       &   'gfftwavecircular') then
     do j = 1,nfft2d
        do k = 1,nfft2d
           x = xmin + real(k-kkm-1,wp)*aretecube
           y = ymin + real(j-jjm-1,wp)*aretecube
           call gaussianparalinear(x,y,z,xgaus,ygaus,zgaus,theta, phi,w0,k0,ss,pp,eps0, E0,Ex,Ey,Ez,nstop,infostr)
           Emodmax = max(real(Ex*conjg(Ex)+Ey*conjg(Ey)+Ez*conjg(Ez),wp),Emodmax)
        end do
     end do
  end if

  !     write(*,*) 'int',k0/deltakx,theta*180.d0/pi
  write(*,*) 'beam',beam
  l = 1
  z = zs(1)
  !     intialise le premier z au champ incident avec la phase
  do j = 1,nfft2d
     do k = 1,nfft2d
        ii = k + (j-1)*nfft2d
        x  = xmin + real(k-kkm-1,wp)*aretecube
        y  = ymin + real(j-jjm-1,wp)*aretecube
        if (comparaison(x,y,z,xs(l),ys(l),z,aretecube) == 1 .and. &
             &          l <= ndipole) then
           trace_occupied = trace_occupied + 1
           tabfft(ii) = l
           kk = 3*(l-1)
           if (niso == 1) then
              FFloc(kk+1) = FF0(kk+1)*exp(icomp*k0*(sqrt(epsilon(l,1,1)) - one)*aretecube)
              FFloc(kk+2) = FF0(kk+2)*exp(icomp*k0*(sqrt(epsilon(l,1,1)) - one)*aretecube)
              FFloc(kk+3) = FF0(kk+3)*exp(icomp*k0*(sqrt(epsilon(l,1,1)) - one)*aretecube)
           else
              FFloc(kk+1) = FF0(kk+1)*exp(icomp*k0*(sqrt(epsilon(l,1,1)) - one)*aretecube)
              FFloc(kk+2) = FF0(kk+2)*exp(icomp*k0*(sqrt(epsilon(l,2,2)) - one)*aretecube)
              FFloc(kk+3) = FF0(kk+3)*exp(icomp*k0*(sqrt(epsilon(l,3,3)) - one)*aretecube)
           end if
           Imagex(ii) = FFloc(kk+1)
           Imagey(ii) = FFloc(kk+2)
           Imagez(ii) = FFloc(kk+3)
           l = l + 1
        else
           tabfft(ii) = 0
           if (beam(1:11) == 'pwavelinear') then
              call ondeplane(x,y,z,k0,eps0,E0,ss,pp,theta,phi,Imagex(ii), Imagey(ii),Imagez(ii),nstop,infostr)
           elseif (beam(1:13) == 'pwavecircular') then
              call ondecirce(x,y,z,k0,E0,ss,theta,phi,Imagex(ii), Imagey(ii),Imagez(ii))
           elseif (beam(1:15) == 'gparawavelinear') then
              call gaussianparalinear(x,y,z,xgaus,ygaus,zgaus,theta, phi,w0,k0,ss,pp,eps0, E0,Imagex(ii),Imagey(ii), Imagez(ii)&
                   &,nstop ,infostr)
           elseif (beam(1:17) == 'gparawavecircular') then
              call gaussianparacirc(x,y,z,xgaus,ygaus,zgaus,theta, phi,w0,k0,ss,eps0, E0,Imagex(ii),Imagey(ii),Imagez(ii), nstop&
                   &,infostr)
           elseif (beam(1:11) == 'gwavelinear' .or. beam(1:14) == 'gfftwavelinear') then
              nloin = 0
              tol   = 1.e-3_wp

              call gaussianparalinear(x,y,z,xgaus,ygaus,zgaus,theta, phi,w0,k0,ss,pp,eps0, E0,Ex,Ey,Ez,nstop,infostr)
              Emod = real(Ex*conjg(Ex)+Ey*conjg(Ey)+Ez*conjg(Ez),wp)
              tolc = sqrt(Emod/Emodmax)*10._wp
              trace_threshold_min = min(trace_threshold_min,tolc)
              trace_threshold_max = max(trace_threshold_max,tolc)
              if (tolc >= tol) then
                 trace_rigorous = trace_rigorous + 1
                 call gaussianchamp(x,y,z,xgaus,ygaus,zgaus,theta, phi,w0,k0,ss,pp,E0,Imagex(ii),Imagey(ii), Imagez(ii),tol,nloin&
                      &,nstop,infostr)
              else
                 Imagex(ii) = Ex
                 Imagey(ii) = Ey
                 Imagez(ii) = Ez
              end if
           elseif (beam(1:13) == 'gwavecircular' .or. beam(1:16) == &
                &                 'gfftwavecircular') then
              nloin = 0
              tol   = 1.e-3_wp
              call gaussianparacirc(x,y,z,xgaus,ygaus,zgaus,theta, phi,w0,k0,ss,eps0, E0,Ex,Ey,Ez,nstop,infostr)
              Emod = real(Ex*conjg(Ex)+Ey*conjg(Ey)+Ez*conjg(Ez),wp)
              tolc = sqrt(Emod/Emodmax)*10._wp
              trace_threshold_min = min(trace_threshold_min,tolc)
              trace_threshold_max = max(trace_threshold_max,tolc)
              if (tolc >= tol) then
                 trace_rigorous = trace_rigorous + 1
                 call gaussianchampcirc(x,y,z,xgaus,ygaus,zgaus, theta,phi,w0,k0,ss,E0,Imagex(ii),Imagey(ii), Imagez(ii),tol,nloin&
                      &,nstop,infostr)
              else
                 Imagex(ii) = Ex
                 Imagey(ii) = Ey
                 Imagez(ii) = Ez
              end if
           else
              write(*,*) 'Beam not implemented for this approximation'
              nstop  = 1
              infostr = 'Beam not implemented for this approximation'
              return
           end if
        end if
     end do
  end do

  if (trace_bpm) write(*,*) 'BPM_TRACE',1,'initial', &
       sum(abs(Imagex)),sum(abs(Imagey)),sum(abs(Imagez))
  if (trace_bpm) write(*,*) 'BPM_TRACE',1,'fill-stats',Emodmax, &
       trace_occupied,trace_rigorous,trace_threshold_min,trace_threshold_max
  if (trace_bpm) write(*,*) 'BPM_TRACE',1,'complex-sums', &
       sum(real(Imagex)),sum(aimag(Imagex)), &
       sum(real(Imagey)),sum(aimag(Imagey)), &
       sum(real(Imagez)),sum(aimag(Imagez))
  if (trace_bpm) write(*,*) 'BPM_TRACE',1,'samples', &
       real(Imagey(1)),aimag(Imagey(1)),real(Imagez(1)),aimag(Imagez(1)), &
       real(Imagey(nfft2d*nfft2d/2)),aimag(Imagey(nfft2d*nfft2d/2)), &
       real(Imagez(nfft2d*nfft2d/2)),aimag(Imagez(nfft2d*nfft2d/2))

  !     commence boucle sur les z
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  cuda_macro_mode = 0
  call ifdda_beampropagation_propagate_cuda(Imagex,Imagey,Imagez,tabfft, &
       epsilon,FFloc,nx,ny,nz,nfft2d,nmax,niso,ndipole,aretecube,k0, &
       cuda_macro_mode,nstop)
  if (nstop /= 0) then
     infostr = 'CUDA beam propagation failed'
     return
  endif
#else
  do k = 2,nz

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

     if (trace_bpm) write(*,*) 'BPM_TRACE',k,'inverse', &
          sum(abs(Imagex)),sum(abs(Imagey)),sum(abs(Imagez))

     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,indice,ii,jj,kx,ky,kz,ctmp)
     !$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
     do j = 1,nfft2d
        do i = 1,nfft2d
           indice = i + nfft2d*(j-1)
           if (i > nfft2d/2) then
              ii = i - nfft2d - 1
           else
              ii = i - 1
           end if
           if (j > nfft2d/2) then
              jj = j - nfft2d - 1
           else
              jj = j - 1
           end if
           kx = deltakx*real(ii,wp)
           ky = deltaky*real(jj,wp)
           kz = k0*k0 - kx*kx - ky*ky
           if (kz >= zero) then
              kz   = sqrt(kz)
              ctmp = exp(-icomp*(k0-kz)*aretecube)
              Imagex(indice) = Imagex(indice)*ctmp
              Imagey(indice) = Imagey(indice)*ctmp
              Imagez(indice) = Imagez(indice)*ctmp
           else
              Imagex(indice) = zzero
              Imagey(indice) = zzero
              Imagez(indice) = zzero
           end if
        end do
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL

     if (trace_bpm) write(*,*) 'BPM_TRACE',k,'spectral', &
          sum(abs(Imagex)),sum(abs(Imagey)),sum(abs(Imagez))

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

     if (trace_bpm) write(*,*) 'BPM_TRACE',k,'forward', &
          sum(abs(Imagex)),sum(abs(Imagey)),sum(abs(Imagez))

     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,indice,x,y,ii,kk)
     !$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
     do j = 1,nfft2d
        do i = 1,nfft2d
           indice = i + (j-1)*nfft2d
           x      = xmin + real(i-kkm-1,wp)*aretecube
           y      = ymin + real(j-jjm-1,wp)*aretecube
           if (tabfft(indice) /= 0) then

              ii = nx*ny*(k-1) + tabfft(indice)
              kk = 3*(ii-1)
              if (niso == 1) then
                 FFloc(kk+1) = Imagex(indice)*exp(icomp*k0*sqrt(epsilon(ii,1,1))*aretecube)*fac
                 FFloc(kk+2) = Imagey(indice)*exp(icomp*k0*sqrt(epsilon(ii,1,1))*aretecube)*fac
                 FFloc(kk+3) = Imagez(indice)*exp(icomp*k0*sqrt(epsilon(ii,1,1))*aretecube)*fac
              else
                 FFloc(kk+1) = Imagex(indice)*exp(icomp*k0*sqrt(epsilon(ii,1,1))*aretecube)*fac
                 FFloc(kk+2) = Imagey(indice)*exp(icomp*k0*sqrt(epsilon(ii,2,2))*aretecube)*fac
                 FFloc(kk+3) = Imagez(indice)*exp(icomp*k0*sqrt(epsilon(ii,3,3))*aretecube)*fac
              end if
              Imagex(indice) = FFloc(kk+1)
              Imagey(indice) = FFloc(kk+2)
              Imagez(indice) = FFloc(kk+3)
           else
              Imagex(indice) = Imagex(indice)*expik0a*fac
              Imagey(indice) = Imagey(indice)*expik0a*fac
              Imagez(indice) = Imagez(indice)*expik0a*fac
           end if
        end do
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
     if (trace_bpm) write(*,*) 'BPM_TRACE',k,'update', &
          sum(abs(Imagex)),sum(abs(Imagey)),sum(abs(Imagez))
     if (trace_bpm) write(*,*) 'BPM_LAYER',k, &
          sum(abs(Imagex)),sum(abs(Imagey)),sum(abs(Imagez))
     !     stop
  end do
#endif
#endif

end subroutine beampropagation
