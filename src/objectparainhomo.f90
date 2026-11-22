subroutine objetparainhomo(eps, eps0,  lnrbc, lcos, ntypeabc, xs, ys, zs, k0, aretecube, tabdip, tabmulti, pmlwrk, nnnr, nmax,&
     & nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, methode, epsilon, polarisa, niso, sidex, sidey, sidez, xg, yg, zg, lc, hc, ng,&
     & epsb, na, sldr, nmat, file_id, group_iddip, infostr, nstop)
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
        integer(c_int) function ifdda_cuboid_geometry_cuda(nx, ny, nz, &
             cell_size, side_x, side_y, side_z, center_x, center_y, center_z, &
             xs, ys, zs, dipole_map) bind(C, name="ifdda_cuboid_geometry_cuda")
            import :: c_int, c_float, c_double
            integer(c_int), value :: nx, ny, nz
#ifdef USE_SIMP
            real(c_float), value :: cell_size, side_x, side_y, side_z
            real(c_float), value :: center_x, center_y, center_z
            real(c_float) :: xs(*), ys(*), zs(*)
#else
            real(c_double), value :: cell_size, side_x, side_y, side_z
            real(c_double), value :: center_x, center_y, center_z
            real(c_double) :: xs(*), ys(*), zs(*)
#endif
            integer(c_int) :: dipole_map(*)
        end function ifdda_cuboid_geometry_cuda
    end interface
#endif

    !--- Variables en entrée/sortie ---
    integer(ip), intent(in) :: nmax, nnnr, nxm, nym, nzm, nmat, niso, na, ng,ntypeabc
    integer(ip), intent(out) :: nbsphere, ndipole, nstop
    integer(ip), intent(inout) :: tabdip(nmax), nx, ny, nz, pmlwrk(5), tabmulti(nmax)
    real(wp), intent(in) :: k0, sldr
    real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, sidex, sidey, sidez, xg, yg, zg, lc, hc, lnrbc, lcos
    complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso), epsb(nmax), eps, eps0
    character(2), intent(in) :: methode
    character(64), intent(out) :: infostr

    !--- Variables locales ---
    integer(ip) :: i, j, k, test, nk, ngraine, dddis, inv, ii, jj,  nnx, nny, nnz
    real(wp) :: x, y, z, kx, ky, kz, dkx, dky, dkz, coeff1, coeff2, coeff3, phase, spec, moyenne, ecartype, lx, ly, lz, ran, side,&
         & constc, lpml,lbord, objetmax,xmin,ymin,zmin,sidexm, sideym,sidezm, coteplus
    integer, allocatable :: seed(:) 
    integer :: seed_size
    integer(ip) :: kk, nbord, nrbc, npml, nnn   
    complex(wp) :: ctmp, epsv, chi, epsani(3,3)
    character(LEN=100) :: datasetname
#ifndef USE_HDF5
    integer(ip), parameter :: hid_t=4
#endif
    integer(hid_t), intent(in) :: file_id, group_iddip    
    integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    integer(c_int) :: cuda_geometry_status
    integer(ip) :: cuboid_count, linear_index
    logical :: cuda_inside
#endif
    integer*8 :: planbpara
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
    sidex = sidex * nmtom
    sidey = sidey * nmtom
    sidez = sidez * nmtom
    xg = xg * nmtom
    yg = yg * nmtom
    zg = zg * nmtom
    lc = lc * nmtom

    write(*,*) 'object', sidex, sidey, sidez
    if (sidex == zero) then
        nstop = 1
        infostr = 'object cuboid: sidex=0!'
        return
    elseif (sidey == zero) then
        nstop = 1
        infostr = 'object cuboid: sidey=0!'
        return
    elseif (sidez == zero) then
        nstop = 1
        infostr = 'object cuboid: sidez=0!'
        return
    endif

    !--- Calcul de la taille de la boîte ---
    side = max(sidex, sidey, sidez)

    if (na.eq.1) then
       objetmax=side
       call initnrbc(k0,lcos,lnrbc,lpml,objetmax,nnnr,nrbc,npml,pmlwrk,aretecube)
       coteplus=objetmax-side
       sidexm=sidex+coteplus
       sideym=sidey+coteplus
       sidezm=sidez+coteplus       

       nx = nint((sidexm) / aretecube)
       ny = nint((sideym) / aretecube)
       nz = nint((sidezm) / aretecube) 
       nnx= nint(sidex/aretecube)
       nny= nint(sidey/aretecube)
       nnz= nint(sidez/aretecube)     
    else
       !--- Boîte au plus près de l'objet ---
       aretecube = side / real(nnnr, wp)
       nx = nint(sidex / aretecube)
       ny = nint(sidey / aretecube)
       nz = nint(sidez / aretecube)    
       nnx = nx
       nny = ny
       nnz = nz
    endif

    write(*,*) 'discretization', nx, ny, nz

    if (nx > nxm .or. ny > nym .or. nz > nzm) then
        nstop = 1
        infostr = 'Dimension Problem of the Box : Box too small!'
        write(*,*) 'dimension Problem', nx, nxm, ny, nym, nz, nzm
        return
    endif

#ifdef IFDDA_CUDA_INVERSE_DRIVER
    ! Generate the axis-aligned object lattice on GPU for the full-grid modes.
    if (na == -1 .or. na == 0) then
        if (nx <= 0 .or. ny <= 0 .or. nz <= 0 .or. &
            nx > huge(cuboid_count) / ny) then
            nstop = 1
            infostr = 'object cuboid: invalid CUDA dimensions'
            return
        endif
        cuboid_count = nx * ny
        if (cuboid_count > huge(cuboid_count) / nz) then
            nstop = 1
            infostr = 'object cuboid: dimensions exceed CUDA index range'
            return
        endif
        cuboid_count = cuboid_count * nz
        if (cuboid_count > nmax) then
            nstop = 1
            infostr = 'nmax parameter too small: increase nxm nym nzm'
            return
        endif
        cuda_geometry_status = ifdda_cuboid_geometry_cuda(int(nx,c_int), &
             int(ny,c_int), int(nz,c_int), real(aretecube,kind=wp), &
             real(sidex,kind=wp), real(sidey,kind=wp), real(sidez,kind=wp), &
             real(zero,kind=wp), real(zero,kind=wp), real(zero,kind=wp), &
             xs, ys, zs, tabdip)
        if (cuda_geometry_status /= 0) then
            nstop = 1
            infostr = 'CUDA inhomogeneous cuboid geometry generation failed'
            return
        endif
    endif
#endif

    !--- Plan FFT 3D ---
#ifdef USE_FFTW
#ifdef USE_SIMP
    call sfftw_plan_dft_3d(planbpara, nnx, nny, nnz, epsb, epsb, FFTW_BACKWARD, FFTW_ESTIMATE)    
#else    
    call dfftw_plan_dft_3d(planbpara, nnx, nny, nnz, epsb, epsb, FFTW_BACKWARD, FFTW_ESTIMATE)
#endif
#endif

    !--- Calcul des paramètres pour la FFT ---
    lx = real(nnx, wp) * aretecube
    ly = real(nny, wp) * aretecube
    lz = real(nnz, wp) * aretecube
    dkx = twopi / lx
    dky = twopi / ly
    dkz = twopi / lz
    nk = 0

    !--- Calcul du spectre ---
    coeff1 = 0.125_wp * hc * hc * lc * lc * lc / (pi * sqrt(pi))
    coeff2 = lc * lc
    coeff3 = 4.0_wp * dkx * dky * dkz
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
                phase = twopi * ran !SUNIF(ngraine)

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
    call sfftw_execute_dft(planbpara, epsb, epsb)    
#else
    call dfftw_execute_dft(planbpara, epsb, epsb)
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
    write(*,*) 'average espilon   :', moyenne
    write(*,*) 'standard deviation:', sqrt(ecartype - moyenne * moyenne)

    !--- Construction du parallélépipède ---
    if (na == -1) then
        do i = 1, nz
            do j = 1, ny
                do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    linear_index = k + nx * (j - 1) + nx * ny * (i - 1)
                    x = xs(linear_index)
                    y = ys(linear_index)
                    z = zs(linear_index)
                    cuda_inside = tabdip(linear_index) /= 0
#else
                    x = -sidex / 2.0_wp + aretecube * (real(k, wp) - 0.5_wp)
                    y = -sidey / 2.0_wp + aretecube * (real(j, wp) - 0.5_wp)
                    z = -sidez / 2.0_wp + aretecube * (real(i, wp) - 0.5_wp)
#endif

                    if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z + zg
                    if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y + yg
                    if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x + xg

                    ndipole = ndipole + 1

#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    if (cuda_inside) then
#else
                    if (abs(x) < sidex / 2.0_wp .and. abs(y) < sidey / 2.0_wp .and. abs(z) < sidez / 2.0_wp) then
#endif
                        nbsphere = nbsphere + 1
                        Tabdip(ndipole) = nbsphere
                        xs(nbsphere) = x + xg
                        ys(nbsphere) = y + yg
                        zs(nbsphere) = z + zg
                        eps = epsb(ndipole)
                        call poladiff(aretecube, eps, eps0, k0, dddis, &
                                     methode, sldr, ctmp)
                        polarisa(nbsphere,1,1) = ctmp
                        epsilon(nbsphere,1,1) = eps
                    endif
                enddo
            enddo
        enddo
    elseif (na == 0) then
        do i = 1, nz
            do j = 1, ny
                do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    linear_index = k + nx * (j - 1) + nx * ny * (i - 1)
                    x = xs(linear_index)
                    y = ys(linear_index)
                    z = zs(linear_index)
                    cuda_inside = tabdip(linear_index) /= 0
#else
                    x = -sidex / 2.0_wp + aretecube * (real(k, wp) - 0.5_wp)
                    y = -sidey / 2.0_wp + aretecube * (real(j, wp) - 0.5_wp)
                    z = -sidez / 2.0_wp + aretecube * (real(i, wp) - 0.5_wp)
#endif

                    if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z + zg
                    if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y + yg
                    if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x + xg

                    ndipole = ndipole + 1
                    nbsphere = nbsphere + 1
                    Tabdip(ndipole) = nbsphere

#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    if (cuda_inside) then
#else
                    if (abs(x) < sidex / 2.0_wp .and. abs(y) < sidey / 2.0_wp .and. abs(z) < sidez / 2.0_wp) then
#endif
                        xs(nbsphere) = x + xg
                        ys(nbsphere) = y + yg
                        zs(nbsphere) = z + zg
                        eps = epsb(ndipole)
                        call poladiff(aretecube, eps, eps0, k0, dddis, &
                                     methode, sldr, ctmp)
                        polarisa(nbsphere,1,1) = ctmp
                        epsilon(nbsphere,1,1) = eps
                    else
                        xs(nbsphere) = x + xg
                        ys(nbsphere) = y + yg
                        zs(nbsphere) = z + zg
                    endif
                enddo
            enddo
        enddo
     elseif (na.eq.1) then
     
       epsv = zone
       constc = (aretecube**3.0_wp) / 4.0_wp / pi
       !calcul les mins de x y z        
       xmin=huge(1.0_wp)/100.0_wp
       do k = 1, nx
          x = -sidexm/2.0_wp + aretecube*(real(k,wp)-0.5_wp)
          if (abs(x) < sidex/2.0_wp) xmin=min(x,xmin)
       enddo

       ymin=huge(1.0_wp)/100.0_wp
       do j = 1, ny
          y = -sideym/2.0_wp + aretecube*(real(j,wp)-0.5_wp)
          if (abs(y) < sidey/2.0_wp) ymin=min(y,ymin)
       enddo

       zmin=huge(1.0_wp)/100.0_wp
       do i = 1, nz
          z = -sidezm/2.0_wp + aretecube*(real(i,wp)-0.5_wp)
          if (abs(z) < sidez/2.0_wp) zmin=min(z,zmin)
       enddo

       write(*,*) 'size box',xmin,ymin,zmin,sidex,sidey,sidez
       
     ! Initialisation des compteurs
       ndipole = 0
       nbsphere = 0
       kk = 0
       nbord= pmlwrk(3)
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,k,x,y,z,ii,jj,kk,nnn,eps,ndipole,nbsphere,ctmp,chi)
     !$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(3)      
       do i = 1, nz
          do j = 1, ny
             do k = 1, nx
                x = -sidexm/2.0_wp + aretecube*(real(k,wp)-0.5_wp)
                y = -sideym/2.0_wp + aretecube*(real(j,wp)-0.5_wp)
                z = -sidezm/2.0_wp + aretecube*(real(i,wp)-0.5_wp)
                
                ndipole=k+nx*(j-1)+nx*ny*(i-1)
                nbsphere=ndipole
                Tabdip(ndipole) = nbsphere
                
                if (abs(x) < sidex / 2.0_wp .and. abs(y) < sidey / 2.0_wp .and. abs(z) < sidez / 2.0_wp) then
                   ii=nint((x-xmin)/aretecube)+1
                   jj=nint((y-ymin)/aretecube)+1
                   kk=nint((z-zmin)/aretecube)+1
                   nnn=ii+(jj-1)*nnx+(kk-1)*nnx*nny       
                   xs(nbsphere) = x+ xg
                   ys(nbsphere) = y+ yg
                   zs(nbsphere) = z+ zg
                   eps = epsb(nnn)
                   ctmp = (eps - eps0) * constc
                   polarisa(nbsphere, 1, 1) = ctmp
                   epsilon(nbsphere, 1, 1) = eps
                   
                else
                   xs(nbsphere) = x+ xg
                   ys(nbsphere) = y+ yg
                   zs(nbsphere) = z+ zg
                  call abcchi(ntypeabc, i, j, k, nx,ny,nz, npml, epsilon, niso, nmax, nbsphere, epsv, eps0, epsani, k0, methode,&
                       & constc , aretecube, lbord, lpml, nrbc, tabmulti, chi)
                  polarisa(nbsphere, 1, 1) = chi
               endif
            enddo
         enddo
      enddo

     !$OMP ENDDO
     !$OMP END PARALLEL

     
     nbsphere=nx*ny*nz
     ndipole=nbsphere
 
     if (nmat == 0) then

        do i = 1, nz
           z = -sidezm/2.0_wp + aretecube*(real(i,wp)-0.5_wp)
           write(22, *) z 
        enddo

        do j = 1, ny
           y = -sideym/2.0_wp + aretecube*(real(j,wp)-0.5_wp)
           write(21, *) y 
        enddo

        do k = 1, nx
           x = -sidexm/2.0_wp + aretecube*(real(k,wp)-0.5_wp)
           write(20, *) x 
        enddo                          
     endif
  else
        infostr = 'na should be equal to -1 or 0'
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
    write(*,*) 'Cuboid inhomogeneous::nbsphere=', nbsphere

end subroutine objetparainhomo
