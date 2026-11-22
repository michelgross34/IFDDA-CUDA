subroutine objetsphereconcentric(trope, epss, epsanis, numbersphere, &
     numberspheremax, xg, yg, zg, rayons, eps0, xs, ys, zs, k0, &
     aretecube, tabdip, tabnbs, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, &
     methode, na, epsilon, polarisa, niso, sldr, nmat, file_id, &
     group_iddip, infostr, nstop)
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
        integer(c_int) function ifdda_concentric_sphere_geometry_cuda(n, &
             cell_size, center, sphere_count, radii, xs, ys, zs, layer_map) &
             bind(C, name="ifdda_concentric_sphere_geometry_cuda")
            import :: c_int, c_float, c_double
            integer(c_int), value :: n, sphere_count
#ifdef USE_SIMP
            real(c_float), value :: cell_size, center
            real(c_float) :: radii(*), xs(*), ys(*), zs(*)
#else
            real(c_double), value :: cell_size, center
            real(c_double) :: radii(*), xs(*), ys(*), zs(*)
#endif
            integer(c_int) :: layer_map(*)
        end function ifdda_concentric_sphere_geometry_cuda
    end interface
#endif

    !--- Variables en entrée/sortie ---
    integer(ip), intent(in) :: nmax, nnnr, nmat, niso, numbersphere, numberspheremax
    integer(ip), intent(out) :: nbsphere, ndipole, nstop
    integer(ip), intent(inout) :: tabdip(nmax), tabnbs(nmax), nx, ny, nz, na
    real(wp), intent(in) :: k0, sldr
    real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, xg, yg, zg, rayons(numberspheremax)
    complex(wp), intent(in) :: epss(numberspheremax), epsanis(3,3,numberspheremax), eps0
    complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso)
    character(2), intent(in) :: methode
    character(3), intent(in) :: trope
    character(64), intent(out) :: infostr

    !--- Variables locales ---
    integer(ip) :: i, j, k, l, ii, jj, test, is, dddis, inv
    real(wp) :: x, y, z, ray, centre
    complex(wp) :: ctmp, eps, epsani(3,3), polaeps(3,3)
    character(LEN=100) :: datasetname
#ifndef USE_HDF5
    integer(ip), parameter :: hid_t=4
#endif
    integer(hid_t), intent(in) :: file_id, group_iddip    
    integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    integer(c_int) :: cuda_geometry_status
    integer(ip), allocatable :: cuda_layer_map(:)
    real(wp), allocatable :: cuda_mesh_x(:), cuda_mesh_y(:), cuda_mesh_z(:)
    integer(ip) :: flat_index
#endif

    !--- Initialisation ---
    nbsphere = 0
    ndipole = 0
    call polaepsto0(polarisa, epsilon, tabdip, niso, nmax)

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC)
    do i = 1, nmax
        tabnbs(i) = 0
    enddo
!$OMP ENDDO
!$OMP END PARALLEL

    dddis = 1
    inv = 1

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
    xg = xg * nmtom
    yg = yg * nmtom
    zg = zg * nmtom

    if (rayons(1) == zero) then
        infostr = 'spheres concentrics: 1st rayon=0'
        nstop = 1
        return
    endif

    do is = 1, numbersphere
        rayons(is) = rayons(is) * nmtom
        if (is >= 2) then
            if (rayons(is) - rayons(is-1) <= zero) then
                infostr = 'spheres concentrics: rayons non croissants'
                nstop = 1
                return
            endif
        endif
    enddo

    !--- Calcul des dimensions ---
    aretecube = 2.0_wp * rayons(numbersphere) / real(nnnr, wp)
    nx = nnnr
    ny = nnnr
    nz = nnnr
    centre = aretecube * real(nnnr, wp) / 2.0_wp

#ifdef IFDDA_CUDA_INVERSE_DRIVER
    if (na == -1 .or. na == 0) then
        allocate(cuda_mesh_x(nx*ny*nz), cuda_mesh_y(nx*ny*nz), &
                 cuda_mesh_z(nx*ny*nz), cuda_layer_map(nx*ny*nz))
        cuda_geometry_status = ifdda_concentric_sphere_geometry_cuda(nnnr, &
            aretecube, centre, numbersphere, rayons, cuda_mesh_x, cuda_mesh_y, &
            cuda_mesh_z, cuda_layer_map)
        if (cuda_geometry_status /= 0) then
            nstop = 1
            infostr = 'CUDA concentric-sphere geometry failed'
            return
        endif
    endif
#endif

    write(*,*) 'eps concentric', epss
    write(*,*) 'Box including the spheres concentriques', nx, ny, nz, aretecube, na, numbersphere

    !--- Construction des sphères concentriques ---
    if (na == -1) then
        do i = 1, nz
            do j = 1, ny
                do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    flat_index = (i - 1)*nx*ny + (j - 1)*nx + k
                    x = cuda_mesh_x(flat_index)
                    y = cuda_mesh_y(flat_index)
                    z = cuda_mesh_z(flat_index)
                    is = cuda_layer_map(flat_index)
#else
                    x = real(k - 1, wp) * aretecube + aretecube / 2.0_wp - centre
                    y = real(j - 1, wp) * aretecube + aretecube / 2.0_wp - centre
                    z = real(i - 1, wp) * aretecube + aretecube / 2.0_wp - centre
                    ray = sqrt(x*x + y*y + z*z)
#endif

                    if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z
                    if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y
                    if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x

                    ndipole = ndipole + 1
#ifndef IFDDA_CUDA_INVERSE_DRIVER
                    is = 0
                    if (ray <= rayons(1)) then
                        is = 1
                    else
                        do l = 2, numbersphere
                            if (ray <= rayons(l) .and. ray >= rayons(l-1)) is = l
                        enddo
                    endif
#endif

                    if (is /= 0) then
                        nbsphere = nbsphere + 1
                        Tabdip(ndipole) = nbsphere
                        Tabnbs(nbsphere) = is
                        xs(nbsphere) = x + xg
                        ys(nbsphere) = y + yg
                        zs(nbsphere) = z + zg
                        if (trope == 'iso') then
                            eps = epss(is)
                            call poladiff(aretecube, eps, eps0, k0, dddis, &
                                         methode, sldr, ctmp)
                            polarisa(nbsphere,1,1) = ctmp
                            epsilon(nbsphere,1,1) = eps
                        else
                            do ii = 1, 3
                                do jj = 1, 3
                                    epsani(ii,jj) = epsanis(ii,jj,is)
                                    epsilon(nbsphere,ii,jj) = epsani(ii,jj)
                                enddo
                            enddo
                            call polaepstens(aretecube, epsani, eps0, k0, &
                                             dddis, methode, inv, polaeps)
                            do ii = 1, 3
                                do jj = 1, 3
                                    polarisa(nbsphere,ii,jj) = polaeps(ii,jj)
                                enddo
                            enddo
                        endif
                    endif
                enddo
            enddo
        enddo
    elseif (na == 0) then
        do i = 1, nz
            do j = 1, ny
                do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    flat_index = (i - 1)*nx*ny + (j - 1)*nx + k
                    x = cuda_mesh_x(flat_index)
                    y = cuda_mesh_y(flat_index)
                    z = cuda_mesh_z(flat_index)
                    is = cuda_layer_map(flat_index)
#else
                    x = real(k - 1, wp) * aretecube + aretecube / 2.0_wp - centre
                    y = real(j - 1, wp) * aretecube + aretecube / 2.0_wp - centre
                    z = real(i - 1, wp) * aretecube + aretecube / 2.0_wp - centre
                    ray = sqrt(x*x + y*y + z*z)
#endif

                    if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z
                    if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y
                    if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x

                    ndipole = ndipole + 1
                    nbsphere = nbsphere + 1
                    Tabdip(ndipole) = nbsphere
#ifndef IFDDA_CUDA_INVERSE_DRIVER
                    is = 0
                    if (ray <= rayons(1)) then
                        is = 1
                    else
                        do l = 2, numbersphere
                            if (ray <= rayons(l) .and. ray >= rayons(l-1)) is = l
                        enddo
                    endif
#endif

                    if (is /= 0) then
                        Tabnbs(nbsphere) = is
                        xs(nbsphere) = x + xg
                        ys(nbsphere) = y + yg
                        zs(nbsphere) = z + zg
                        if (trope == 'iso') then
                            eps = epss(is)
                            call poladiff(aretecube, eps, eps0, k0, dddis, &
                                         methode, sldr, ctmp)
                            polarisa(nbsphere,1,1) = ctmp
                            epsilon(nbsphere,1,1) = eps
                        else
                            do ii = 1, 3
                                do jj = 1, 3
                                    epsani(ii,jj) = epsanis(ii,jj,is)
                                    epsilon(nbsphere,ii,jj) = epsani(ii,jj)
                                enddo
                            enddo
                            call polaepstens(aretecube, epsani, eps0, k0, &
                                             dddis, methode, inv, polaeps)
                            do ii = 1, 3
                                do jj = 1, 3
                                    polarisa(nbsphere,ii,jj) = polaeps(ii,jj)
                                enddo
                            enddo
                        endif
                    else
                        xs(nbsphere) = x + xg
                        ys(nbsphere) = y + yg
                        zs(nbsphere) = z + zg
                    endif
                enddo
            enddo
        enddo
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

end subroutine objetsphereconcentric
