subroutine objetcube(trope, eps, epsani, eps0, xs, ys, zs, k0, aretecube, &
     tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, methode, &
     epsilon, polarisa, niso, side, xg, yg, zg, sldr, nmat, file_id, &
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
        integer(c_int) function ifdda_cube_geometry_cuda(subdivisions, side, &
                cell_size, center_x, center_y, center_z, xs, ys, zs, dipole_map) &
                bind(C, name="ifdda_cube_geometry_cuda")
            import :: c_int, c_float, c_double
            integer(c_int), value :: subdivisions
#ifdef USE_SIMP
            real(c_float), value :: side, cell_size, center_x, center_y, center_z
            real(c_float) :: xs(*), ys(*), zs(*)
#else
            real(c_double), value :: side, cell_size, center_x, center_y, center_z
            real(c_double) :: xs(*), ys(*), zs(*)
#endif
            integer(c_int) :: dipole_map(*)
        end function ifdda_cube_geometry_cuda
    end interface
#endif

    !--- Variables en entrée/sortie ---
    integer(ip), intent(in) :: nmax, nnnr, nmat, niso
    integer(ip), intent(out) :: nbsphere, ndipole, nstop
    integer(ip), intent(inout) :: tabdip(nmax) ,nx ,ny, nz
    real(wp), intent(in) :: k0, sldr
    real(wp), intent(inout) ::  side, xg, yg, zg, xs(nmax), ys(nmax), zs(nmax), aretecube
    complex(wp), intent(in) :: eps, epsani(3,3), eps0
    complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso)
    character(2), intent(in) :: methode
    character(3), intent(in) :: trope
    character(64), intent(out) :: infostr
  

    !--- Variables locales ---
    integer(ip) :: i, j, k, ii, jj, test, dddis, inv
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    integer(c_int) :: cuda_geometry_status
    integer(ip) :: cube_count
#endif
    real(wp) :: x, y, z, xc, yc, zc
    complex(wp) :: ctmp, polaeps(3,3)
    character(LEN=100) :: datasetname
#ifndef USE_HDF5
    integer(ip), parameter :: hid_t=4
#endif
    integer(hid_t), intent(in) :: file_id, group_iddip
    integer(ip) :: dim(4), error

    !--- Initialisation ---
    write(*,*) 'CUBE'

    nbsphere = 0
    ndipole = 0
    dddis = 1
    inv = 1
    
    call polaepsto0(polarisa,epsilon,tabdip,niso,nmax)

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

    side = side * nmtom
    write(*,*) 'objetcube::side=', side
    if (side == zero) then
        write(*,*) 'objetcube::side is equal to zero: Aborting', side
        nstop = 1
        infostr = 'object cube: side=0!'
        return
    endif

    xg = xg * nmtom
    yg = yg * nmtom
    zg = zg * nmtom

    !--- vérifie si on est bien multiple de 2 3 5 pour la discrétisation ---
    nx = nnnr
    ny = nnnr
    nz = nnnr
    !--- size of the subunit ---
    aretecube = side / real(nnnr, wp)

    xc = (side + aretecube) / 2.0_wp
    yc = (side + aretecube) / 2.0_wp
    zc = (side + aretecube) / 2.0_wp

#ifdef IFDDA_CUDA_INVERSE_DRIVER
    if (nnnr <= 0) then
        infostr = 'object cube: invalid discretization'
        nstop = 1
        return
    endif
    if (nnnr > huge(cube_count) / nnnr) then
        infostr = 'object cube: dimensions exceed CUDA index range'
        nstop = 1
        return
    endif
    cube_count = nnnr * nnnr
    if (cube_count > huge(cube_count) / nnnr) then
        infostr = 'object cube: dimensions exceed CUDA index range'
        nstop = 1
        return
    endif
    cube_count = cube_count * nnnr
    if (cube_count > nmax) then
        infostr = 'nmax parameter too small: increase nxm nym nzm'
        nstop = 1
        return
    endif
    cuda_geometry_status = ifdda_cube_geometry_cuda(int(nnnr,c_int), &
        real(side,kind=wp), real(aretecube,kind=wp), &
        real(xg,kind=wp), real(yg,kind=wp), real(zg,kind=wp), &
        xs, ys, zs, tabdip)
    if (cuda_geometry_status /= 0) then
        infostr = 'CUDA cube geometry generation failed'
        nstop = 1
        return
    endif
    ndipole = 0
    nbsphere = 0
#endif

    do i = 1, nnnr
        do j = 1, nnnr
            do k = 1, nnnr
#ifndef IFDDA_CUDA_INVERSE_DRIVER
                x = real(k, wp) * aretecube - xc
                y = real(j, wp) * aretecube - yc
                z = real(i, wp) * aretecube - zc
#endif

#ifdef IFDDA_CUDA_INVERSE_DRIVER
                if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) zs((i-1)*nnnr*nnnr+1)
                if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) ys((j-1)*nnnr+1)
                if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) xs(k)
                ndipole = ndipole + 1
                nbsphere = nbsphere + 1
                ii = (i-1)*nnnr*nnnr + (j-1)*nnnr + k
#else
                if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z + zg
                if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y + yg
                if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x + xg

                ndipole = ndipole + 1
                nbsphere = nbsphere + 1
                Tabdip(ndipole) = nbsphere
                xs(nbsphere) = x + xg
                ys(nbsphere) = y + yg
                zs(nbsphere) = z + zg
#endif
                if (trope == 'iso') then
                    call poladiff(aretecube, eps, eps0, k0, dddis, methode, &
                                 sldr, ctmp)
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
        write(*,*) 'Size of the parameter too small', ndipole, &
                   'should be smaller that', nmax
        write(*,*) 'Increase nxm nym and nzm'
        infostr = 'nmax parameter too small: increase nxm nym nzm'
        nstop = 1
        return
    endif

    if (nmat == 0) then
        do i = 1, nbsphere
            write(10,*) xs(nbsphere)
            write(11,*) ys(nbsphere)
            write(12,*) zs(nbsphere)
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

end subroutine objetcube
