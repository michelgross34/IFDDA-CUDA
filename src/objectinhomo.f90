subroutine objetsphereinhomo(eps, eps0, lnrbc, lcos, ntypeabc, xs, ys, zs, k0, aretecube, tabdip, tabmulti, pmlwrk, nnnr, nmax,&
     & nbsphere, ndipole, nx, ny, nz, methode, na, epsilon, polarisa, niso, rayon, lc, hc, ng, epsb, sldr, nmat, file_id,&
     & group_iddip, infostr, nstop)
#ifdef USE_HDF5
  use HDF5
#endif
  use precision
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  use, intrinsic :: iso_c_binding, only: c_int, c_float, c_double
#endif
  implicit none

#ifdef IFDDA_CUDA_INVERSE_DRIVER
  interface
     integer(c_int) function ifdda_sphere_geometry_cuda(subdivisions, radius, &
          cell_size, center_x, center_y, center_z, xs, ys, zs, inside_map) &
          bind(C, name="ifdda_sphere_geometry_cuda")
       import :: c_int, c_float, c_double
       integer(c_int), value :: subdivisions
#ifdef USE_SIMP
       real(c_float), value :: radius, cell_size, center_x, center_y, center_z
       real(c_float) :: xs(*), ys(*), zs(*)
#else
       real(c_double), value :: radius, cell_size, center_x, center_y, center_z
       real(c_double) :: xs(*), ys(*), zs(*)
#endif
       integer(c_int) :: inside_map(*)
     end function ifdda_sphere_geometry_cuda
     integer(c_int) function ifdda_correlated_material_field_cuda(nx, ny, nz, &
          spacing, coherence_length, standard_deviation, epsilon_real, epsilon_imag, &
          phases, real_field, mean, standard_deviation_result) &
          bind(C, name="ifdda_correlated_material_field_cuda")
       import :: c_int, c_float, c_double
       integer(c_int), value :: nx, ny, nz
#ifdef USE_SIMP
       real(c_float), value :: spacing, coherence_length, standard_deviation
       real(c_float), value :: epsilon_real, epsilon_imag
       real(c_float) :: phases(*), real_field(*), mean, standard_deviation_result
#else
       real(c_double), value :: spacing, coherence_length, standard_deviation
       real(c_double), value :: epsilon_real, epsilon_imag
       real(c_double) :: phases(*), real_field(*), mean, standard_deviation_result
#endif
     end function ifdda_correlated_material_field_cuda
  end interface
#endif

  !--- Variables en entrée/sortie ---
  integer(ip), intent(in) :: nmax, nnnr, nmat, niso, na, ng
  integer(ip), intent(out) :: nbsphere, ndipole, nstop
  integer(ip), intent(inout) :: tabdip(nmax), tabmulti(nmax), nx, ny, nz, ntypeabc, pmlwrk(5)
  real(wp), intent(in) :: k0, sldr
  real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, rayon, lc, hc, lnrbc, lcos
  complex(wp), intent(in) ::  eps0
  complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso), epsb(nmax), eps
  character(2), intent(in) :: methode
  character(64), intent(out) :: infostr

  !--- Variables locales ---
  integer(ip) :: i, j, k, nk, ngraine, iret, test, dddis, inv
  real(wp) :: x, y, z, kx, ky, kz, dkx, dky, dkz, coeff1, coeff2, coeff3, phase, spec, moyenne, ecartype, lx, ly, lz, ran, constc,&
       & xmin
  integer, allocatable :: seed(:) 
  integer :: seed_size
  integer(ip) :: npml, nrbc, ii,jj, kk, nnx, nny, nnz,nnn
  complex(wp) :: ctmp, chi, epsani(3,3), epsv
  real(wp) :: lbord, lpml, objetmax, ray
  character(LEN=100) :: datasetname
#ifndef USE_HDF5
  integer(ip), parameter :: hid_t=4
#endif
  integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  integer(c_int) :: cuda_geometry_status
  integer(ip) :: sphere_count, linear_index
  integer(c_int) :: cuda_material_status
  real(wp), allocatable :: random_phases(:), material_real(:)
  real(wp) :: cuda_mean, cuda_standard_deviation
  logical :: cuda_inside
#endif
  integer(hid_t), intent(in) :: file_id, group_iddip    
  integer*8 :: planb
  integer(ip), parameter :: FFTW_FORWARD = -1, FFTW_BACKWARD = +1, FFTW_ESTIMATE = 64

  ! 1. Récupère la taille requise pour la graine
  call random_seed(size=seed_size)
  ! 2. Alloue un tableau de cette taille
  allocate(seed(seed_size))
  seed = ng
  ! Initialise ran
  call random_seed(put=seed)

  !--- Vérification des paramètres ---
  if (lc <= zero) then
     infostr = 'coherence length equal to zero'
     nstop = 1
     return
  endif
  if (hc <= zero) then
     infostr = 'standard deviation equal to zero'
     nstop = 1
     return
  endif

  !--- Initialisation ---
  nbsphere = 0
  ndipole = 0
  call polaepsto0(polarisa, epsilon, tabdip, niso, nmax)
  dddis = 1
  inv = 1

  ngraine = 4 * ng + 1

  lc = lc * nmtom
  if (nmat == 0) then
     !--- mesh ---
     open(20, file='x.mat')
     open(21, file='y.mat')
     open(22, file='z.mat')
     !--- discretization of the object under study ---
     open(10, file='xc.mat')
     open(11, file='yc.mat')
     open(12, file='zc.mat')
  endif

  rayon = rayon * nmtom
  if (rayon <= 1.0e-12_wp) then
     nstop = 1
     infostr = 'object sphere: radius=0!'
     return
  endif

  !--- Vérification de la discrétisation ---
  nx = nnnr
  ny = nnnr
  nz = nnnr

  if (na.eq.1) then
     objetmax=2.0_wp*rayon
     call initnrbc(k0,lcos,lnrbc,lpml,objetmax,nnnr,nrbc,npml,pmlwrk,aretecube)
     ! Include every interior coordinate; INT can under-size the FFT field.
     nnx=ceiling(2.0_wp * rayon/aretecube)
     nny=nnx
     nnz=nnx
  else
     aretecube = 2.0_wp * rayon / real(nnnr, wp)
     nnx = nx
     nny = ny
     nnz = nz
  endif

  if (nnx <= 0 .or. nnx*nny*nnz > nmax) then
     infostr = 'inhomogeneous sphere: random field exceeds nmax'
     nstop = 1
     return
  endif

  !--- Plan FFT 3D ---
#if defined(USE_FFTW) && !defined(IFDDA_CUDA_INVERSE_DRIVER)
#ifdef USE_SIMP
  call sfftw_plan_dft_3d(planb, nnx, nny, nnz, epsb, epsb, FFTW_BACKWARD, FFTW_ESTIMATE)    
#else
  call dfftw_plan_dft_3d(planb, nnx, nny, nnz, epsb, epsb, FFTW_BACKWARD, FFTW_ESTIMATE)
#endif
#endif

  lx = real(nnx, wp) * aretecube
  ly = real(nny, wp) * aretecube
  lz = real(nnz, wp) * aretecube
  dkx = twopi / lx
  dky = twopi / ly
  dkz = twopi / lz
  nk = 0

  coeff1 = 0.125_wp * hc * hc * lc * lc * lc / (pi * sqrt(pi))
  coeff2 = lc * lc
  coeff3 = 4.0_wp * dkx * dky * dkz

#ifdef IFDDA_CUDA_INVERSE_DRIVER
  ! The random material spectrum above remains on CPU. Generate the sphere
  ! lattice and occupancy mask on GPU for the two full-grid modes.
  if (na == -1 .or. na == 0) then
     if (nnnr <= 0 .or. nnnr > huge(sphere_count) / nnnr) then
        nstop = 1
        infostr = 'object sphere: invalid CUDA discretization'
        return
     endif
     sphere_count = nnnr * nnnr
     if (sphere_count > huge(sphere_count) / nnnr) then
        nstop = 1
        infostr = 'object sphere: dimensions exceed CUDA index range'
        return
     endif
     sphere_count = sphere_count * nnnr
     if (sphere_count > nmax) then
        nstop = 1
        infostr = 'nmax parameter too small: increase nxm nym nzm'
        return
     endif
     cuda_geometry_status = ifdda_sphere_geometry_cuda(int(nnnr,c_int), &
          real(rayon,kind=wp), real(aretecube,kind=wp), real(zero,kind=wp), &
          real(zero,kind=wp), real(zero,kind=wp), xs, ys, zs, tabdip)
     if (cuda_geometry_status /= 0) then
        nstop = 1
        infostr = 'CUDA inhomogeneous sphere geometry generation failed'
        return
     endif
  endif
#endif

  !--- Calcul du spectre ---
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  nk = nnx * nny * nnz
  allocate(random_phases(nk), material_real(nk))
  do nk = 1, nnx * nny * nnz
     call random_number(ran)
     random_phases(nk) = ran
  enddo
  cuda_material_status = ifdda_correlated_material_field_cuda(int(nnx,c_int), &
       int(nny,c_int), int(nnz,c_int), real(aretecube,kind=wp), real(lc,kind=wp), &
       real(hc,kind=wp), real(eps,kind=wp), aimag(eps), random_phases, &
       material_real, cuda_mean, cuda_standard_deviation)
  if (cuda_material_status /= 0) then
     nstop = 1
     infostr = 'CUDA correlated sphere material generation failed'
     return
  endif
  do nk = 1, nnx * nny * nnz
     epsb(nk) = cmplx(material_real(nk), aimag(eps), kind=wp)
  enddo
  moyenne = cuda_mean
  ecartype = cuda_standard_deviation * cuda_standard_deviation + moyenne * moyenne
#else
  do i = 1, nnz
     do j = 1, nny
        do k = 1, nnx
           nk = nk + 1
           if (k >= 1 .and. k <= (nnx/2 + 1)) then
              kx = real(k - 1, wp) * dkx
           else
              kx = real(k - 1 - nnx, wp) * dkx
           endif
           if (j >= 1 .and. j <= (nny/2 + 1)) then
              ky = real(j - 1, wp) * dky
           else
              ky = real(j - 1 - nny, wp) * dky
           endif
           if (i >= 1 .and. i <= (nnz/2 + 1)) then
              kz = real(i - 1, wp) * dkz
           else
              kz = real(i - 1 - nnz, wp) * dkz
           endif

           !--- Calcul du spectre ---
           spec = coeff1 * exp(-0.25_wp * coeff2 * (kx*kx + ky*ky + kz*kz))

           !--- Phase aléatoire ---
           call random_number(ran)
           phase = 2.0_wp * pi * ran ! SUNIF(ngraine)

           !--- Amplitude complexe ---
           epsb(nk) = sqrt(spec * coeff3) * exp(icomp * phase)

           !--- Spectre symétrique pour diffraction harmonique ---
           if (nk == 1) epsb(nk) = zzero
           if (nk >= nnx/2 + 2 .and. nk <= nnx) epsb(nk) = zzero
           if (nk >= (nny/2 + 1) * nnx + 1 .and. nk <= nnx * nny) epsb(nk) = zzero
           if (nk >= (nnz/2 + 1) * nnx * nny + 1 .and. nk <= nnx * nny * nnz) epsb(nk) = zzero

        enddo
     enddo
  enddo

  !--- FFT inverse pour obtenir le profil des hauteurs ---
#ifdef USE_FFTW
#ifdef USE_SIMP
  call sfftw_execute_dft(planb, epsb, epsb)    
#else
  call dfftw_execute_dft(planb, epsb, epsb)
#endif
#else
  call fftsingletonz3d(epsb, nnx, nny, nnz, FFTW_BACKWARD)
#endif

  !--- Calcul de la moyenne et de l'écart-type ---
  moyenne = zero
  ecartype = zero
  do i = 1, nk
     epsb(i) = real(epsb(i),wp) + eps
     moyenne = moyenne + real(epsb(i),wp)
     ecartype = ecartype + abs(epsb(i))**2
  enddo

  moyenne = moyenne / real(nk, wp)
  ecartype = ecartype / real(nk, wp)
#endif

  !--- Construction de la sphère ---
  if (na == -1) then
     do i = 1, nnnr
        do j = 1, nnnr
           do k = 1, nnnr
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              linear_index = (i - 1) * nnnr * nnnr + (j - 1) * nnnr + k
              x = xs(linear_index)
              y = ys(linear_index)
              z = zs(linear_index)
              cuda_inside = tabdip(linear_index) /= 0
#else
              x = -rayon + aretecube * (real(k, wp) - 0.5_wp)
              y = -rayon + aretecube * (real(j, wp) - 0.5_wp)
              z = -rayon + aretecube * (real(i, wp) - 0.5_wp)
#endif

              if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z
              if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y
              if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x

              ndipole = ndipole + 1
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              if (cuda_inside) then
#else
              if (sqrt(x*x + y*y + z*z) < rayon) then
#endif
                 nbsphere = nbsphere + 1
                 Tabdip(ndipole) = nbsphere
                 xs(nbsphere) = x
                 ys(nbsphere) = y
                 zs(nbsphere) = z
                 eps = epsb(ndipole)
                 call poladiff(aretecube, eps, eps0, k0, dddis, methode, sldr, ctmp)
                 polarisa(nbsphere,1,1) = ctmp
                 epsilon(nbsphere,1,1) = eps
              endif
           enddo
        enddo
     enddo
  elseif (na == 0) then
     do i = 1, nnnr
        do j = 1, nnnr
           do k = 1, nnnr
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              linear_index = (i - 1) * nnnr * nnnr + (j - 1) * nnnr + k
              x = xs(linear_index)
              y = ys(linear_index)
              z = zs(linear_index)
              cuda_inside = tabdip(linear_index) /= 0
#else
              x = -rayon + aretecube * (real(k, wp) - 0.5_wp)
              y = -rayon + aretecube * (real(j, wp) - 0.5_wp)
              z = -rayon + aretecube * (real(i, wp) - 0.5_wp)
#endif

              if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z
              if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y
              if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x

              ndipole = ndipole + 1
              nbsphere = nbsphere + 1
              Tabdip(ndipole) = nbsphere
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              if (cuda_inside) then
#else
              if (sqrt(x*x + y*y + z*z) < rayon) then
#endif
                 xs(nbsphere) = x
                 ys(nbsphere) = y
                 zs(nbsphere) = z
                 eps = epsb(ndipole)
                 call poladiff(aretecube, eps, eps0, k0, dddis, methode, sldr, ctmp)
                 polarisa(nbsphere,1,1) = ctmp
                 epsilon(nbsphere,1,1) = eps
              else
                 xs(nbsphere) = x
                 ys(nbsphere) = y
                 zs(nbsphere) = z
              endif
           enddo
        enddo
     enddo
  elseif (na.eq.1) then
     
     epsv = zone
     constc = (aretecube**3.0_wp) / 4.0_wp / pi

     ! Initialisation des compteurs
     ndipole = 0
     nbsphere = 0
     kk = 0
     tabmulti = 0
     !calcul du xmin pour passer le epsilon inhomogene
     xmin=huge(1.0_wp)/100.0_wp
     do k = 1, nnnr
        x = -objetmax/2.0_wp + aretecube*(real(k,wp)-0.5_wp)
        if (abs(x) < rayon) xmin=min(x,xmin)
     enddo
     do k = 1, nnnr
        x = -objetmax/2.0_wp + aretecube*(real(k,wp)-0.5_wp)
        if (abs(x) < rayon) then
           ii=nint((x-xmin)/aretecube)+1
           if (ii < 1 .or. ii > nnx) then
              infostr = 'inhomogeneous sphere: random field index outside FFT domain'
              nstop = 1
              return
           endif
        endif
     enddo
  
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,k,x,y,z,ii,jj,kk,nnn,eps,ndipole,nbsphere,ray,ctmp,chi)
     !$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(3)      
      do i = 1, nnnr
        do j = 1, nnnr
           do k = 1, nnnr
              x = -objetmax/2.0_wp + aretecube*(real(k,wp)-0.5_wp)
              y = -objetmax/2.0_wp + aretecube*(real(j,wp)-0.5_wp)
              z = -objetmax/2.0_wp + aretecube*(real(i,wp)-0.5_wp)

              ndipole=k+nnnr*(j-1)+nnnr*nnnr*(i-1)
              nbsphere=ndipole
              Tabdip(ndipole) = nbsphere

              ray = sqrt(x*x + y*y + z*z)
              if (ray < rayon) then
                 xs(nbsphere) = x
                 ys(nbsphere) = y
                 zs(nbsphere) = z
                 ii=nint((x-xmin)/aretecube)+1
                 jj=nint((y-xmin)/aretecube)+1
                 kk=nint((z-xmin)/aretecube)+1                 
                 nnn=ii+(jj-1)*nnx+(kk-1)*nnx*nny                
                 eps = epsb(nnn)

                 ctmp = (eps - eps0) * constc
                 polarisa(nbsphere, 1, 1) = ctmp
                 epsilon(nbsphere, 1, 1) = eps


              else
                 xs(nbsphere) = x
                 ys(nbsphere) = y
                 zs(nbsphere) = z
                  call abcchi(ntypeabc, i, j, k, nx,ny,nz, npml, epsilon, niso, nmax, nbsphere, epsv, eps0, epsani, k0, methode,&
                       & constc, aretecube, lbord, lpml, nrbc, tabmulti, chi)
                  polarisa(nbsphere, 1, 1) = chi

              endif
           enddo
        enddo
     enddo

     !$OMP ENDDO
     !$OMP END PARALLEL

     
     nbsphere=nnnr*nnnr*nnnr
     ndipole=nbsphere
 
     if (nmat == 0) then
        do k = 1, nnnr
           ! Calcul des coordonnées
           x = -objetmax/2.0_wp + aretecube*(real(k,wp)-0.5_wp)
           y = -objetmax/2.0_wp + aretecube*(real(k,wp)-0.5_wp)
           z = -objetmax/2.0_wp + aretecube*(real(k,wp)-0.5_wp) 
           ! Écriture des coordonnées pour nmat=0
           write(22, *) z 
           write(21, *) y 
           write(20, *) x 
        enddo
     endif

     
  else
     infostr = 'na should be equal to -1 or 0 or 1'
     nstop = 1
     return
  endif

  if (ndipole > nmax) then
     infostr = 'nmax parameter too small: increase nxm nym nzm'
     nstop = 1
     return
  endif

  if (nmat == 0) then
     do i = 1, nbsphere
        write(10,*) xs(i)
        write(11,*) ys(i)
        write(12,*) zs(i)
     enddo
  elseif (nmat == 2) then
     dim(1) = nbsphere
     dim(2) = nmax
     datasetname = "Dipole position x"
     call hdf5write1d(group_iddip, datasetname, xs, dim)
     datasetname = "Dipole position y"
     call hdf5write1d(group_iddip, datasetname, ys, dim)
     datasetname = "Dipole position z"
     call hdf5write1d(group_iddip, datasetname, zs, dim)
  endif

  close(10)
  close(11)
  close(12)
  close(20)
  close(21)
  close(22)
  deallocate(seed)

  write(*,*) 'OBJECT SPHERE INHOMOGENEOUS FINISHED'
  write(*,*) 'SEED', ng, 'Average', moyenne, 'Standard deviation', &
       sqrt(ecartype - moyenne * moyenne)

end subroutine objetsphereinhomo
