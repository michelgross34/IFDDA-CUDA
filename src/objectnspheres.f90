subroutine objetnspheres(trope, epss, epsanis, numbersphere, &
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
        integer(c_int) function ifdda_nspheres_geometry_cuda(nx, ny, nz, &
             cell_size, x_min, y_min, z_min, sphere_count, fixed_sphere_count, parameters, &
             xs, ys, zs, sphere_map) bind(C, name="ifdda_nspheres_geometry_cuda")
            import :: c_int, c_float, c_double
            integer(c_int), value :: nx, ny, nz, sphere_count, fixed_sphere_count
#ifdef USE_SIMP
            real(c_float), value :: cell_size, x_min, y_min, z_min
            real(c_float) :: parameters(*), xs(*), ys(*), zs(*)
#else
            real(c_double), value :: cell_size, x_min, y_min, z_min
            real(c_double) :: parameters(*), xs(*), ys(*), zs(*)
#endif
            integer(c_int) :: sphere_map(*)
        end function ifdda_nspheres_geometry_cuda
    end interface
#endif

    !--- Variables en entrée/sortie ---
    integer(ip), intent(in) :: nmax, nnnr, nmat, niso, na, numbersphere, numberspheremax
    integer(ip), intent(out) :: nbsphere, ndipole, nstop
    integer(ip), intent(inout) :: tabdip(nmax), tabnbs(nmax), nx, ny, nz
    real(wp), intent(in) :: k0, sldr
    real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, xg(numberspheremax), yg(numberspheremax),&
         & zg(numberspheremax), rayons(numberspheremax)
    complex(wp), intent(in) :: epss(numberspheremax), epsanis(3,3,numberspheremax), eps0
    complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso)
    character(2), intent(in) :: methode
    character(3), intent(in) :: trope
    character(64), intent(out) :: infostr

    !--- Variables locales ---
    integer(ip) :: i, j, k, ii, jj, test, dddis, inv, is
    real(wp) :: x, y, z, xmin, xmax, ymin, ymax, zmin, zmax
    complex(wp) :: ctmp, eps, epsani(3,3), polaeps(3,3)
    character(LEN=100) :: datasetname
#ifndef USE_HDF5
    integer(ip), parameter :: hid_t=4
#endif
    integer(hid_t), intent(in) :: file_id, group_iddip
    integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    integer(c_int) :: cuda_geometry_status
    integer(ip), allocatable :: cuda_sphere_map(:)
    real(wp), allocatable :: cuda_mesh_x(:), cuda_mesh_y(:), cuda_mesh_z(:)
    real(wp), allocatable :: cuda_sphere_parameters(:)
    integer(ip) :: flat_index
#endif

    !--- Initialisation ---
    write(*,*) 'Begin of nspheres'
    nbsphere = 0
    ndipole = 0
    dddis = 1
    inv = 1

    call polaepsto0(polarisa,epsilon,tabdip,niso,nmax)

    xmax = -huge(1.0_wp)/100.0_wp
    xmin = huge(1.0_wp)/100.0_wp
    ymax = -huge(1.0_wp)/100.0_wp
    ymin = huge(1.0_wp)/100.0_wp
    zmax = -huge(1.0_wp)/100.0_wp
    zmin = huge(1.0_wp)/100.0_wp

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

    !--- Conversion des unités et calcul des limites ---
    do is = 1, numbersphere
        rayons(is) = rayons(is) * nmtom
        xg(is) = xg(is) * nmtom
        yg(is) = yg(is) * nmtom
        zg(is) = zg(is) * nmtom
        xmax = max(xmax, xg(is) + rayons(is))
        xmin = min(xmin, xg(is) - rayons(is))
        ymax = max(ymax, yg(is) + rayons(is))
        ymin = min(ymin, yg(is) - rayons(is))
        zmax = max(zmax, zg(is) + rayons(is))
        zmin = min(zmin, zg(is) - rayons(is))
        if (rayons(is) == zero) then
            nstop = 1
            infostr = 'object nspheres: radius=0!'
            return
        endif
    enddo

    if (numbersphere > numberspheremax) then
        infostr = 'number of spheres too large: limited to 20'
        nstop = 1
        return
    endif

    aretecube = max(zmax - zmin, ymax - ymin, xmax - xmin) / real(nnnr, wp)

    nx = nint((xmax - xmin) / aretecube)
    ny = nint((ymax - ymin) / aretecube)
    nz = nint((zmax - zmin) / aretecube)

#ifdef IFDDA_CUDA_INVERSE_DRIVER
    if (na == -1 .or. na == 0) then
        allocate(cuda_mesh_x(nx*ny*nz), cuda_mesh_y(nx*ny*nz), &
                 cuda_mesh_z(nx*ny*nz), cuda_sphere_map(nx*ny*nz), &
                 cuda_sphere_parameters(4*numbersphere))
        do is = 1, numbersphere
            cuda_sphere_parameters(4*is-3) = xg(is)
            cuda_sphere_parameters(4*is-2) = yg(is)
            cuda_sphere_parameters(4*is-1) = zg(is)
            cuda_sphere_parameters(4*is) = rayons(is)
        enddo
        cuda_geometry_status = ifdda_nspheres_geometry_cuda(nx, ny, nz, &
            aretecube, xmin, ymin, zmin, numbersphere, numbersphere, cuda_sphere_parameters, &
            cuda_mesh_x, cuda_mesh_y, cuda_mesh_z, cuda_sphere_map)
        if (cuda_geometry_status /= 0) then
            nstop = 1
            infostr = 'CUDA multiple-sphere geometry failed'
            return
        endif
    endif
#endif

    write(*,*) 'Box including the N spheres', nx, ny, nz, 'meshsize', aretecube
    write(*,*) 'Number of sphere', numbersphere
    write(*,*) 'X', xmin*1.0e9_wp, xmax*1.0e9_wp
    write(*,*) 'Y', ymin*1.0e9_wp, ymax*1.0e9_wp
    write(*,*) 'Z', zmin*1.0e9_wp, zmax*1.0e9_wp

    if (na == -1) then
        write(*,*) 'na', na
        do i = 1, nz
            do j = 1, ny
                do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    flat_index = (i - 1)*nx*ny + (j - 1)*nx + k
                    x = cuda_mesh_x(flat_index)
                    y = cuda_mesh_y(flat_index)
                    z = cuda_mesh_z(flat_index)
#else
                    x = xmin + real(k - 1, wp) * aretecube + aretecube / 2.0_wp
                    y = ymin + real(j - 1, wp) * aretecube + aretecube / 2.0_wp
                    z = zmin + real(i - 1, wp) * aretecube + aretecube / 2.0_wp
#endif

                    if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z
                    if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y
                    if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x

                    ndipole = ndipole + 1
                    test = 0
                    do is = 1, numbersphere
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                        if (cuda_sphere_map(ndipole) < 0) then
                            infostr = 'sphere are not distincts'
                            nstop = 1
                            return
                        endif
                        if (is == cuda_sphere_map(ndipole)) then
#else
                        if ((x - xg(is))**2 + (y - yg(is))**2 + (z - zg(is))**2 <= rayons(is)**2) then
#endif
                            test = test + 1
                            nbsphere = nbsphere + 1
                            Tabdip(ndipole) = nbsphere
                            Tabnbs(nbsphere) = is
                            xs(nbsphere) = x
                            ys(nbsphere) = y
                            zs(nbsphere) = z
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
                    if (test >= 2) then
                        infostr = 'sphere are not distincts'
                        nstop = 1
                        return
                    endif
                enddo
            enddo
        enddo
    elseif (na == 0) then
        write(*,*) 'na', na
        do i = 1, nz
            do j = 1, ny
                do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    flat_index = (i - 1)*nx*ny + (j - 1)*nx + k
                    x = cuda_mesh_x(flat_index)
                    y = cuda_mesh_y(flat_index)
                    z = cuda_mesh_z(flat_index)
#else
                    x = xmin + real(k - 1, wp) * aretecube + aretecube / 2.0_wp
                    y = ymin + real(j - 1, wp) * aretecube + aretecube / 2.0_wp
                    z = zmin + real(i - 1, wp) * aretecube + aretecube / 2.0_wp
#endif

                    if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z
                    if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y
                    if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x

                    ndipole = ndipole + 1
                    nbsphere = nbsphere + 1
                    Tabdip(ndipole) = nbsphere
                    test = 0
                    do is = 1, numbersphere
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                        if (cuda_sphere_map(ndipole) < 0) then
                            infostr = 'sphere are not distincts'
                            nstop = 1
                            return
                        endif
                        if (is == cuda_sphere_map(ndipole)) then
#else
                        if ((x - xg(is))**2 + (y - yg(is))**2 + (z - zg(is))**2 <= rayons(is)**2) then
#endif
                            test = test + 1
                            Tabnbs(nbsphere) = is
                            xs(nbsphere) = x
                            ys(nbsphere) = y
                            zs(nbsphere) = z
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
                    if (test == 0) then
                        xs(nbsphere) = x
                        ys(nbsphere) = y
                        zs(nbsphere) = z
                    elseif (test >= 2) then
                        infostr = 'sphere are not distincts'
                        nstop = 1
                        return
                    endif
                enddo
            enddo
        enddo
    else
        infostr = 'na should be equal to -1 or 0'
        nstop = 1
        return
    endif

    write(*,*) 'End of nspheres'
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

end subroutine objetnspheres
