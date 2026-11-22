subroutine objetparanxnynz(trope, eps, epsani, eps0, xs, ys, zs, k0, &
     aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, &
     nzm, nxmp, nymp, nzmp, methode, epsilon, polarisa, niso, sidex, &
     sidey, sidez, xg, yg, zg, na, sldr, nmat, file_id, group_iddip, infostr, nstop)
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
                cell_size, side_x, side_y, side_z, center_x, center_y, &
                center_z, xs, ys, zs, dipole_map) &
                bind(C, name="ifdda_cuboid_geometry_cuda")
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
    integer(ip), intent(in) :: nmax, nnnr, nxm, nym, nzm, nxmp, nymp, nzmp, nmat, niso, na
    integer(ip), intent(out) :: nbsphere, ndipole, nstop
    integer(ip), intent(inout) :: tabdip(nmax), nx, ny, nz
    real(wp), intent(in) :: k0, sldr
    real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, xg, yg, zg, sidex, sidey, sidez
    complex(wp), intent(inout) :: eps, epsani(3,3), eps0
    complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso)
    character(2), intent(in) :: methode
    character(3), intent(in) :: trope
    character(64), intent(out) :: infostr

    !--- Variables locales ---
    integer(ip) :: i, j, k, ii, jj, test, dddis, inv
    real(wp) :: x, y, z, xr, yr, zr, x1, x2, y1, y2, z1, z2, mat(3,3), side
    complex(wp) :: ctmp, polaeps(3,3)
    character(LEN=100) :: datasetname
#ifndef USE_HDF5
    integer(ip), parameter :: hid_t=4
#endif
    integer(hid_t), intent(in) :: file_id, group_iddip    
    integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    integer(c_int) :: cuda_geometry_status
    integer(ip) :: cuboid_count
#endif

    !--- Initialisation ---
    nbsphere = 0
    ndipole = 0
    call polaepsto0(polarisa, epsilon, tabdip, niso, nmax)
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

    write(*,*) 'object4', nx, nxm, nxmp
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
    if (nx <= 0 .or. ny <= 0 .or. nz <= 0) then
        nstop = 1
        infostr = 'object cuboid: invalid dimensions for CUDA'
        return
    endif
    if (nx > huge(cuboid_count) / ny) then
        nstop = 1
        infostr = 'object cuboid: dimensions exceed CUDA index range'
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
        real(xg,kind=wp), real(yg,kind=wp), real(zg,kind=wp), &
        xs, ys, zs, tabdip)
    if (cuda_geometry_status /= 0) then
        nstop = 1
        infostr = 'CUDA cuboid geometry generation failed'
        return
    endif
#endif

    !--- Construction du parallélépipède ---
    do i = 1, nz
        do j = 1, ny
            do k = 1, nx
#ifndef IFDDA_CUDA_INVERSE_DRIVER
                x = aretecube * (real(k, wp) - 0.5_wp) - sidex / 2.0_wp
                y = aretecube * (real(j, wp) - 0.5_wp) - sidey / 2.0_wp
                z = aretecube * (real(i, wp) - 0.5_wp) - sidez / 2.0_wp
#endif

#ifdef IFDDA_CUDA_INVERSE_DRIVER
                if (j == 1 .and. k == 1 .and. nmat == 0) &
                    write(22,*) zs((i-1)*nx*ny+1)
                if (i == 1 .and. k == 1 .and. nmat == 0) &
                    write(21,*) ys((j-1)*nx+1)
                if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) xs(k)
#else
                if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z + zg
                if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y + yg
                if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x + xg
#endif

                ndipole = ndipole + 1
                nbsphere = nbsphere + 1
#ifndef IFDDA_CUDA_INVERSE_DRIVER
                Tabdip(ndipole) = nbsphere
                xs(nbsphere) = x + xg
                ys(nbsphere) = y + yg
                zs(nbsphere) = z + zg
#endif

                if (trope == 'iso') then
                    call poladiff(aretecube, eps, eps0, k0, dddis, &
                                 methode, sldr, ctmp)
                    polarisa(nbsphere,1,1) = ctmp
                    epsilon(nbsphere,1,1) = eps
                else
                    call polaepstens(aretecube, epsani, eps0, k0, dddis, &
                                     methode, inv, polaeps)
                    do ii = 1, 3
                        do jj = 1, 3
                            epsilon(nbsphere,ii,jj) = epsani(ii,jj)
                            polarisa(nbsphere,ii,jj) = polaeps(ii,jj)
                        enddo
                    enddo
                endif
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

    write(*,*) 'Cuboid::nbsphere=', nbsphere, nx, ny, nz

end subroutine objetparanxnynz
