subroutine objetsphere(trope, eps, epsani, eps0, lnrbc, lcos, ntypeabc, xs, ys, zs, k0, aretecube, tabdip, tabmulti, pmlwrk, nnnr,&
     & nmax, nbsphere, ndipole, nx, ny, nz, methode, na, epsilon, polarisa, niso, rayon, xg, yg, zg, sldr, nmat, file_id,&
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
  end interface
#endif

  !--- Variables en entrée/sortie ---
  integer(ip), intent(in) :: nmax, nnnr, nmat, niso, na
  integer(ip), intent(out) :: nbsphere, ndipole, nstop
  integer(ip), intent(inout) :: tabdip(nmax), tabmulti(nmax), pmlwrk(5), nx, ny, nz, ntypeabc
  real(wp), intent(in) :: k0, sldr
  real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, rayon, xg, yg, zg, lcos,lnrbc
  complex(wp), intent(in) :: eps, epsani(3,3), eps0
  complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso)
  character(2), intent(in) :: methode
  character(3), intent(in) :: trope
  character(64), intent(out) :: infostr

  !--- Variables locales ---
  integer(ip) :: i, j, k, ii, jj, test, dddis, inv, kk, npml, nrbc
  real(wp) :: x, y, z, lambda, lbord, lpml, objetmax, ray
  complex(wp) :: ctmp, polaeps(3,3),chi, constc, epsv
  character(LEN=100) :: datasetname
#ifndef USE_HDF5
  integer(ip), parameter :: hid_t=4
#endif
  integer(hid_t), intent(in) :: file_id, group_iddip    
  integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  integer(c_int) :: cuda_geometry_status
  integer(ip) :: sphere_count, linear_index
  logical :: cuda_inside
#endif

  !--- Initialisation ---
  nbsphere = 0
  ndipole = 0
  call polaepsto0(polarisa, epsilon, tabdip, niso, nmax)
  dddis = 1
  inv = 1

  if (trope == 'ani' .and. methode == 'PS') then
     nstop = 1
     infostr = 'anisotrope with PS'
     return
  endif

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

  !--- Conversion des unités ---
  rayon = rayon * nmtom
  xg = xg * nmtom
  yg = yg * nmtom
  zg = zg * nmtom

  if (rayon == zero) then
     nstop = 1
     infostr = 'object sphere: radius=0!'
     return
  endif

  !--- Calcul des dimensions ---
  nx = nnnr
  ny = nnnr
  nz = nnnr
  aretecube = 2.0_wp * rayon / real(nnnr, wp)

#ifdef IFDDA_CUDA_INVERSE_DRIVER
  if ((na == -1) .or. (na == 0 .and. methode /= 'PS')) then
     if (nnnr <= 0) then
        nstop = 1
        infostr = 'object sphere: invalid CUDA discretization'
        return
     endif
     if (nnnr > huge(sphere_count) / nnnr) then
        nstop = 1
        infostr = 'object sphere: dimensions exceed CUDA index range'
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
          real(rayon,kind=wp), real(aretecube,kind=wp), &
          real(xg,kind=wp), real(yg,kind=wp), real(zg,kind=wp), &
          xs, ys, zs, tabdip)
     if (cuda_geometry_status /= 0) then
        nstop = 1
        infostr = 'CUDA sphere geometry generation failed'
        return
     endif
  endif
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

#ifdef IFDDA_CUDA_INVERSE_DRIVER
              if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z
              if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y
              if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x
#else
              if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z + zg
              if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y + yg
              if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x + xg
#endif

              ndipole = ndipole + 1
#ifndef IFDDA_CUDA_INVERSE_DRIVER
              if (sqrt(x*x + y*y + z*z) < rayon) then
#else
              if (cuda_inside) then
#endif
                 nbsphere = nbsphere + 1
                 Tabdip(ndipole) = nbsphere
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                 xs(nbsphere) = x
                 ys(nbsphere) = y
                 zs(nbsphere) = z
#else
                 xs(nbsphere) = x + xg
                 ys(nbsphere) = y + yg
                 zs(nbsphere) = z + zg
#endif
                 if (trope == 'iso') then
                    call poladiff(aretecube, eps, eps0, k0, dddis, methode, sldr, ctmp)
                    polarisa(nbsphere,1,1) = ctmp
                    epsilon(nbsphere,1,1) = eps
                 else
                    call polaepstens(aretecube, epsani, eps0, k0, dddis, methode, inv, polaeps)
                    do ii = 1, 3
                       do jj = 1, 3
                          epsilon(nbsphere,ii,jj) = epsani(ii,jj)
                          polarisa(nbsphere,ii,jj) = polaeps(ii,jj)
                       enddo
                    enddo
                 endif
              endif
           enddo
        enddo
     enddo

     if (methode == 'PS') then
        call polamodifsphere(nbsphere, nmax, xs, ys, zs, k0, polarisa, niso, aretecube, eps, eps0)
     endif
  elseif (na == 0 .and. methode /= 'PS') then
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

#ifdef IFDDA_CUDA_INVERSE_DRIVER
              if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z
              if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y
              if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x
#else
              if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z + zg
              if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y + yg
              if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x + xg
#endif

              ndipole = ndipole + 1
              nbsphere = nbsphere + 1
              Tabdip(ndipole) = nbsphere
#ifndef IFDDA_CUDA_INVERSE_DRIVER
              if (sqrt(x*x + y*y + z*z) < rayon) then
#else
              if (cuda_inside) then
#endif
#ifndef IFDDA_CUDA_INVERSE_DRIVER
                 xs(nbsphere) = x + xg
                 ys(nbsphere) = y + yg
                 zs(nbsphere) = z + zg
#endif
                 if (trope == 'iso') then
                    call poladiff(aretecube, eps, eps0, k0, dddis, methode, sldr, ctmp)
                    polarisa(nbsphere,1,1) = ctmp
                    epsilon(nbsphere,1,1) = eps
                 else
                    call polaepstens(aretecube, epsani, eps0, k0, dddis, methode, inv, polaeps)
                    do ii = 1, 3
                       do jj = 1, 3
                          epsilon(nbsphere,ii,jj) = epsani(ii,jj)
                          polarisa(nbsphere,ii,jj) = polaeps(ii,jj)
                       enddo
                    enddo
                 endif
              else
#ifndef IFDDA_CUDA_INVERSE_DRIVER
                 xs(nbsphere) = x + xg
                 ys(nbsphere) = y + yg
                 zs(nbsphere) = z + zg
#endif
              endif
           enddo
        enddo
     enddo
  elseif (na == 0 .and. methode == 'PS') then
     nbsphere = 0
     do i = 1, nnnr
        do j = 1, nnnr
           do k = 1, nnnr
              x = -rayon + aretecube * (real(k, wp) - 0.5_wp)
              y = -rayon + aretecube * (real(j, wp) - 0.5_wp)
              z = -rayon + aretecube * (real(i, wp) - 0.5_wp)
              if (sqrt(x*x + y*y + z*z) < rayon) then
                 nbsphere = nbsphere + 1
                 xs(nbsphere) = x + xg
                 ys(nbsphere) = y + yg
                 zs(nbsphere) = z + zg
              endif
           enddo
        enddo
     enddo

     call polamodifsphere(nbsphere, nmax, xs, ys, zs, k0, epsilon, niso, aretecube, eps, eps0)

     kk = 0
     ndipole = 0
     nbsphere = 0
     do i = 1, nnnr
        do j = 1, nnnr
           do k = 1, nnnr
              x = -rayon + aretecube * (real(k, wp) - 0.5_wp)
              y = -rayon + aretecube * (real(j, wp) - 0.5_wp)
              z = -rayon + aretecube * (real(i, wp) - 0.5_wp)

              if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z + zg
              if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y + yg
              if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x + xg

              ndipole = ndipole + 1
              nbsphere = nbsphere + 1
              Tabdip(ndipole) = nbsphere
              xs(nbsphere) = x + xg
              ys(nbsphere) = y + yg
              zs(nbsphere) = z + zg
              if (sqrt(x*x + y*y + z*z) < rayon) then
                 kk = kk + 1
                 polarisa(nbsphere,1,1) = epsilon(kk,1,1)
                 polarisa(nbsphere,1,2) = epsilon(kk,1,2)
                 polarisa(nbsphere,1,3) = epsilon(kk,1,3)
                 polarisa(nbsphere,2,1) = epsilon(kk,2,1)
                 polarisa(nbsphere,2,2) = epsilon(kk,2,2)
                 polarisa(nbsphere,2,3) = epsilon(kk,2,3)
                 polarisa(nbsphere,3,1) = epsilon(kk,3,1)
                 polarisa(nbsphere,3,2) = epsilon(kk,3,2)
                 polarisa(nbsphere,3,3) = epsilon(kk,3,3)
              endif
           enddo
        enddo
     enddo

     nbsphere = 0
     do i = 1, nnnr
        do j = 1, nnnr
           do k = 1, nnnr
              x = -rayon + aretecube * (real(k, wp) - 0.5_wp)
              y = -rayon + aretecube * (real(j, wp) - 0.5_wp)
              z = -rayon + aretecube * (real(i, wp) - 0.5_wp)
              nbsphere = nbsphere + 1
              if (sqrt(x*x + y*y + z*z) < rayon) then
                 epsilon(nbsphere,1,1) = eps
                 epsilon(nbsphere,1,2) = zzero
                 epsilon(nbsphere,1,3) = zzero
                 epsilon(nbsphere,2,1) = zzero
                 epsilon(nbsphere,2,2) = eps
                 epsilon(nbsphere,2,3) = zzero
                 epsilon(nbsphere,3,1) = zzero
                 epsilon(nbsphere,3,2) = zzero
                 epsilon(nbsphere,3,3) = eps
              else
                 epsilon(nbsphere,1,1) = zone
                 epsilon(nbsphere,1,2) = zzero
                 epsilon(nbsphere,1,3) = zzero
                 epsilon(nbsphere,2,1) = zzero
                 epsilon(nbsphere,2,2) = zone
                 epsilon(nbsphere,2,3) = zzero
                 epsilon(nbsphere,3,1) = zzero
                 epsilon(nbsphere,3,2) = zzero
                 epsilon(nbsphere,3,3) = zone
              endif
           enddo
        enddo
     enddo
  elseif (na.eq.1) then

     objetmax=2.0_wp*rayon
     call initnrbc(k0,lcos,lnrbc,lpml,objetmax,nnnr,nrbc,npml,pmlwrk,aretecube)    
     epsv = zone
     constc = (aretecube**3.0_wp) / 4.0_wp / pi

     ! Initialisation des compteurs
     ndipole = 0
     nbsphere = 0
     kk = 0
     tabmulti = 0
     open(1000,file='toto.mat')
     ! Boucle sur la grille 3D
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,k,ii,jj,x,y,z,ndipole,nbsphere,ray,ctmp,chi)
     !$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(3)     
     do i = 1, nnnr
        do j = 1, nnnr
           do k = 1, nnnr
              ! Calcul des coordonnées
              x = -objetmax/2.0_wp + aretecube*(real(k,wp)-0.5_wp)
              y = -objetmax/2.0_wp + aretecube*(real(j,wp)-0.5_wp)
              z = -objetmax/2.0_wp + aretecube*(real(i,wp)-0.5_wp) 

              ! Incrémentation des compteurs
!              ndipole = ndipole + 1
!              nbsphere = nbsphere + 1
              ndipole=k+nnnr*(j-1)+nnnr*nnnr*(i-1)
              nbsphere=ndipole
              tabdip(ndipole) = nbsphere

!              ! Vérification si le point est dans l'objet
!              if (x < rayon .and. x > -rayon .and. y < rayon .and. y > -rayon .and. z < rayon .and. z > -rayon) then
!                 kk = kk + 1
!                 tabmulti(ndipole) = kk
!              end if

              ! Calcul de la distance au centre
              ray = sqrt(x*x + y*y + z*z)

              ! Si le point est dans l'objet
              if (ray < rayon) then
                 xs(nbsphere) = x + xg
                 ys(nbsphere) = y + yg
                 zs(nbsphere) = z + zg
                 if (trope == 'iso') then
                    ctmp = (eps - eps0) * constc
                    polarisa(nbsphere, 1, 1) = ctmp
                    epsilon(nbsphere, 1, 1) = eps
                 else
                    do ii = 1, 3
                       do jj = 1, 3
                          polarisa(nbsphere, ii, jj) = (epsilon(nbsphere, ii, jj) -eps0)*constc
                       end do
                    end do
                 end if
                 ! Si le point est hors de l'objet (dans les PML)
              else
                 xs(nbsphere) = x + xg
                 ys(nbsphere) = y + yg
                 zs(nbsphere) = z + zg
                 call abcchi(ntypeabc, i, j, k, nx,ny,nz, npml, epsilon, niso, nmax, nbsphere, epsv, eps0, epsani, k0, methode,&
                      & constc, aretecube, lbord, lpml, nrbc, tabmulti, chi)
                 polarisa(nbsphere, 1, 1) = chi

              end if
           end do
        end do
     end do
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
           write(22, *) z + zg
           write(21, *) y + yg
           write(20, *) x + xg
        enddo
     endif
     ! Fermeture des fichiers
     close(1000)
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

  write(*,*) 'OBJECT SPHERE FINISHED'

end subroutine objetsphere
