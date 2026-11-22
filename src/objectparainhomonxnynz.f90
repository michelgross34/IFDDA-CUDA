subroutine objetparainhomonxnynz(eps, eps0, xs, ys, zs, k0, aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym,&
     & nzm, nxmp, nymp, nzmp, methode, epsilon, polarisa, niso, sidex, sidey, sidez, xg, yg, zg, lc, hc, ng, epsb, na, sldr, nmat,&
     & file_id, group_iddip, infostr, nstop)
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
    integer(ip), intent(in) :: nmax, nnnr, nxm, nym, nzm, nxmp, nymp, nzmp, nmat, niso, na, ng
    integer(ip), intent(out) :: nbsphere, ndipole, nstop
    integer(ip), intent(inout) :: tabdip(nmax), nx, ny, nz
    real(wp), intent(in) :: k0, sldr
    real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, sidex, sidey, sidez, xg, yg, zg, lc, hc
    complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso), epsb(nmax), eps, eps0
    character(2), intent(in) :: methode
    character(64), intent(out) :: infostr

    !--- Variables locales ---
    integer(ip) :: i, j, k, nk, ngraine, dddis, inv
    real(wp) :: x, y, z, kx, ky, kz, dkx, dky, dkz, coeff1, coeff2, coeff3, phase, spec, moyenne, ecartype, lx, ly, lz, ran, side
    integer, allocatable :: seed(:) 
    integer :: seed_size   
    complex(wp) :: ctmp
    character(LEN=100) :: datasetname
#ifndef USE_HDF5
    integer(ip), parameter :: hid_t=4
#endif
    integer(hid_t), intent(in) :: file_id, group_iddip    
    integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    integer(c_int) :: cuda_geometry_status
    integer(ip) :: cuboid_count, linear_index
    integer(c_int) :: cuda_material_status
    real(wp), allocatable :: random_phases(:), material_real(:)
    real(wp) :: cuda_mean, cuda_standard_deviation
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

    !--- Calcul des dimensions ---
    nx = nxm - 2 * nxmp
    ny = nym - 2 * nymp
    nz = nzm - 2 * nzmp
    aretecube = aretecube * nmtom
    sidex = aretecube * real(nx, wp)
    sidey = aretecube * real(ny, wp)
    sidez = aretecube * real(nz, wp)
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

    if (nx > nxm .or. ny > nym .or. nz > nzm) then
        nstop = 1
        infostr = 'Dimension Problem of the Box : Box too small!'
        write(*,*) 'dimension Problem', nx, nxm, ny, nym, nz, nzm
        return
    endif

#ifdef IFDDA_CUDA_INVERSE_DRIVER
    if (nx <= 0 .or. ny <= 0 .or. nz <= 0 .or. nx > huge(cuboid_count) / ny) then
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
#endif

    !--- Plan FFT 3D ---
#ifdef USE_FFTW
#ifdef USE_SIMP
    call sfftw_plan_dft_3d(planbpara, nx, ny, nz, epsb, epsb, FFTW_BACKWARD, FFTW_ESTIMATE)    
#else    
    call dfftw_plan_dft_3d(planbpara, nx, ny, nz, epsb, epsb, FFTW_BACKWARD, FFTW_ESTIMATE)
#endif
#endif

    !--- Calcul des paramètres pour la FFT ---
    lx = real(nx, wp) * aretecube
    ly = real(ny, wp) * aretecube
    lz = real(nz, wp) * aretecube
    dkx = twopi / lx
    dky = twopi / ly
    dkz = twopi / lz
    nk = 0

    !--- Calcul du spectre ---
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    nk = nx * ny * nz
    allocate(random_phases(nk), material_real(nk))
    do nk = 1, nx * ny * nz
        call random_number(ran)
        random_phases(nk) = ran
    enddo
    cuda_material_status = ifdda_correlated_material_field_cuda(int(nx,c_int), &
         int(ny,c_int), int(nz,c_int), real(aretecube,kind=wp), real(lc,kind=wp), &
         real(hc,kind=wp), real(eps,kind=wp), aimag(eps), random_phases, &
         material_real, cuda_mean, cuda_standard_deviation)
    if (cuda_material_status /= 0) then
        nstop = 1
        infostr = 'CUDA correlated cuboid material generation failed'
        return
    endif
    do nk = 1, nx * ny * nz
        epsb(nk) = cmplx(material_real(nk), aimag(eps), kind=wp)
    enddo
    moyenne = cuda_mean
    ecartype = cuda_standard_deviation * cuda_standard_deviation + moyenne * moyenne
#else
    coeff1 = 0.125_wp * hc * hc * lc * lc * lc / (pi * sqrt(pi))
    coeff2 = lc * lc
    coeff3 = 4.0_wp * dkx * dky * dkz
    do i = 1, nz
        do j = 1, ny
            do k = 1, nx
                nk = nk + 1
                if (k >= 1 .and. k <= (nx/2 + 1)) then
                    kx = real(k - 1, wp) * dkx
                else
                    kx = real(k - 1 - nx, wp) * dkx
                endif
                if (j >= 1 .and. j <= (ny/2 + 1)) then
                    ky = real(j - 1, wp) * dky
                else
                    ky = real(j - 1 - ny, wp) * dky
                endif
                if (i >= 1 .and. i <= (nz/2 + 1)) then
                    kz = real(i - 1, wp) * dkz
                else
                    kz = real(i - 1 - nz, wp) * dkz
                endif

                !--- Calcul du spectre ---
                spec = coeff1 * exp(-0.25_wp * coeff2 * (kx*kx + ky*ky + kz*kz))

                !--- Phase aléatoire --- 
                call random_number(ran)
                phase = twopi * ran ! SUNIF(ngraine)

                !--- Amplitude complexe ---
                epsb(nk) = sqrt(spec * coeff3) * exp(icomp * phase)

                !--- Spectre symétrique pour diffraction harmonique ---
                if (nk == 1) epsb(nk) = zzero
                if (nk >= nx/2 + 2 .and. nk <= nx) epsb(nk) = zzero
                if (nk >= (ny/2 + 1) * nx + 1 .and. nk <= nx * ny) epsb(nk) = zzero
                if (nk >= (nz/2 + 1) * nx * ny + 1 .and. nk <= nx * ny * nz) epsb(nk) = zzero
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
    call fftsingletonz3d(epsb, nx, ny, nz, FFTW_BACKWARD)
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
    write(*,*) 'average espilon   :', moyenne
    write(*,*) 'standard deviation:', sqrt(ecartype - moyenne * moyenne)

    !--- Construction du parallélépipède ---
    do i = 1, nz
        do j = 1, ny
            do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                linear_index = k + nx * (j - 1) + nx * ny * (i - 1)
                x = xs(linear_index)
                y = ys(linear_index)
                z = zs(linear_index)
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

                xs(nbsphere) = x + xg
                ys(nbsphere) = y + yg
                zs(nbsphere) = z + zg
                eps = epsb(ndipole)
                call poladiff(aretecube, eps, eps0, k0, dddis, &
                             methode, sldr, ctmp)
                polarisa(nbsphere,1,1) = ctmp
                epsilon(nbsphere,1,1) = eps
            enddo
        enddo
    enddo

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

end subroutine objetparainhomonxnynz
