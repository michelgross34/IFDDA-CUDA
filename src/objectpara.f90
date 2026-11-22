subroutine objetpara(trope, eps, epsani, eps0, xs, ys, zs, k0, aretecube, &
     tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, &
     methode, epsilon, polarisa, niso, sidex, sidey, sidez, xg, yg, zg, &
     phi, theta, psi, na, sldr, nmat, file_id, group_iddip, infostr, nstop)
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
        integer(c_int) function ifdda_rotated_cuboid_geometry_cuda(nx, ny, nz, &
                cell_size, extent_x, extent_y, extent_z, side_x, side_y, &
                side_z, center_x, center_y, center_z, inverse_rotation, &
                xs, ys, zs, inside_map) &
                bind(C, name="ifdda_rotated_cuboid_geometry_cuda")
            import :: c_int, c_float, c_double
            integer(c_int), value :: nx, ny, nz
#ifdef USE_SIMP
            real(c_float), value :: cell_size, extent_x, extent_y, extent_z
            real(c_float), value :: side_x, side_y, side_z
            real(c_float), value :: center_x, center_y, center_z
            real(c_float) :: inverse_rotation(*), xs(*), ys(*), zs(*)
#else
            real(c_double), value :: cell_size, extent_x, extent_y, extent_z
            real(c_double), value :: side_x, side_y, side_z
            real(c_double), value :: center_x, center_y, center_z
            real(c_double) :: inverse_rotation(*), xs(*), ys(*), zs(*)
#endif
            integer(c_int) :: inside_map(*)
        end function ifdda_rotated_cuboid_geometry_cuda
    end interface
#endif

    !--- Variables en entrée/sortie ---
    integer(ip), intent(in) :: nmax, nnnr, nxm, nym, nzm, nmat, niso, na
    integer(ip), intent(out) :: nbsphere, ndipole, nstop
    integer(ip), intent(inout) :: tabdip(nmax), nx, ny, nz
    real(wp), intent(in) :: k0, sldr
    real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, sidex, sidey, sidez, xg, yg, zg, phi, theta, psi
    complex(wp), intent(in) :: eps, epsani(3,3), eps0
    complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso)
    character(2), intent(in) :: methode
    character(3), intent(in) :: trope
    character(64), intent(out) :: infostr

    !--- Variables locales ---
    integer(ip) :: i, j, k, ii, jj, test, dddis, inv
    real(wp) :: x, y, z, xr, yr, zr, side, x1, x2, y1, y2, z1, z2, mat(3,3)
    complex(wp) :: ctmp, polaeps(3,3)
    character(LEN=100) :: datasetname
#ifndef USE_HDF5
    integer(ip), parameter :: hid_t=4
#endif
    integer(hid_t), intent(in) :: file_id, group_iddip    
    integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    integer(c_int) :: cuda_geometry_status
    integer(ip) :: geometry_count, linear_index
    logical :: cuda_inside
#endif

    !--- Initialisation ---
    write(*,*) 'object2'
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

    !--- Conversion des unités ---
    sidex = sidex * nmtom
    sidey = sidey * nmtom
    sidez = sidez * nmtom
    xg = xg * nmtom
    yg = yg * nmtom
    zg = zg * nmtom
    theta = theta * pi / 180.0_wp
    phi = phi * pi / 180.0_wp
    psi = psi * pi / 180.0_wp

    !--- Vérification des dimensions ---
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
    write(*,*) 'object'

    !--- Matrice de rotation ---
    mat(1,1) = cos(psi)*cos(phi) - sin(psi)*cos(theta)*sin(phi)
    mat(1,2) = -cos(psi)*sin(phi) - sin(psi)*cos(theta)*cos(phi)
    mat(1,3) = sin(psi)*sin(theta)

    mat(2,1) = sin(psi)*cos(phi) + cos(psi)*cos(theta)*sin(phi)
    mat(2,2) = -sin(psi)*sin(phi) + cos(psi)*cos(theta)*cos(phi)
    mat(2,3) = -cos(psi)*sin(theta)

    mat(3,1) = sin(theta)*sin(phi)
    mat(3,2) = sin(theta)*cos(phi)
    mat(3,3) = cos(theta)

    !--- Calcul des limites de la boîte englobante ---
    x1 = huge(1.0_wp)/100.0_wp
    x2 = -huge(1.0_wp)/100.0_wp
    y1 = huge(1.0_wp)/100.0_wp
    y2 = -huge(1.0_wp)/100.0_wp
    z1 = huge(1.0_wp)/100.0_wp
    z2 = -huge(1.0_wp)/100.0_wp

    !--- Calcul des 8 coins du parallélépipède ---
    x = -sidex / 2.0_wp
    y = -sidey / 2.0_wp
    z = -sidez / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = sidex / 2.0_wp
    y = -sidey / 2.0_wp
    z = -sidez / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = -sidex / 2.0_wp
    y = sidey / 2.0_wp
    z = -sidez / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = -sidex / 2.0_wp
    y = -sidey / 2.0_wp
    z = sidez / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = sidex / 2.0_wp
    y = sidey / 2.0_wp
    z = -sidez / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = sidex / 2.0_wp
    y = -sidey / 2.0_wp
    z = sidez / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = -sidex / 2.0_wp
    y = sidey / 2.0_wp
    z = sidez / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = sidex / 2.0_wp
    y = sidey / 2.0_wp
    z = sidez / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    !--- Calcul de la taille de la boîte ---
    side = max(x2 - x1, y2 - y1, z2 - z1)
    aretecube = side / real(nnnr, wp)

    ! The rotation matrix is orthogonal, so its inverse is its transpose.
    mat = transpose(mat)

    !--- Boîte au plus près de l'objet ---
    nx = int((x2 - x1) / aretecube)
    ny = int((y2 - y1) / aretecube)
    nz = int((z2 - z1) / aretecube)
    write(*,*) 'nnn', nx, ny, nz

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
    if (nx > huge(geometry_count) / ny) then
        nstop = 1
        infostr = 'object cuboid: dimensions exceed CUDA index range'
        return
    endif
    geometry_count = nx * ny
    if (geometry_count > huge(geometry_count) / nz) then
        nstop = 1
        infostr = 'object cuboid: dimensions exceed CUDA index range'
        return
    endif
    geometry_count = geometry_count * nz
    if (geometry_count > nmax) then
        nstop = 1
        infostr = 'nmax parameter too small: increase nxm nym nzm'
        return
    endif
    cuda_geometry_status = ifdda_rotated_cuboid_geometry_cuda(int(nx,c_int), &
        int(ny,c_int), int(nz,c_int), real(aretecube,kind=wp), &
        real(x2-x1,kind=wp), real(y2-y1,kind=wp), real(z2-z1,kind=wp), &
        real(sidex,kind=wp), real(sidey,kind=wp), real(sidez,kind=wp), &
        real(xg,kind=wp), real(yg,kind=wp), real(zg,kind=wp), mat, &
        xs, ys, zs, tabdip)
    if (cuda_geometry_status /= 0) then
        nstop = 1
        infostr = 'CUDA rotated cuboid geometry generation failed'
        return
    endif
#endif

    if (na == -1) then
        do i = 1, nz
            do j = 1, ny
                do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    linear_index = (i - 1) * ny * nx + (j - 1) * nx + k
                    x = xs(linear_index)
                    y = ys(linear_index)
                    z = zs(linear_index)
                    cuda_inside = tabdip(linear_index) /= 0
#else
                    x = -(x2 - x1) / 2.0_wp + aretecube * (real(k, wp) - 0.5_wp)
                    y = -(y2 - y1) / 2.0_wp + aretecube * (real(j, wp) - 0.5_wp)
                    z = -(z2 - z1) / 2.0_wp + aretecube * (real(i, wp) - 0.5_wp)
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
                    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
                    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
                    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
                    if (abs(xr) <= sidex / 2.0_wp .and. abs(yr) <= sidey / 2.0_wp .and. abs(zr) <= sidez / 2.0_wp) then
#endif
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    if (cuda_inside) then
#else
                    if (abs(xr) <= sidex / 2.0_wp .and. abs(yr) <= sidey / 2.0_wp .and. abs(zr) <= sidez / 2.0_wp) then
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
                    endif
                enddo
            enddo
        enddo
    elseif (na == 0) then
        do i = 1, nz
            do j = 1, ny
                do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    linear_index = (i - 1) * ny * nx + (j - 1) * nx + k
                    x = xs(linear_index)
                    y = ys(linear_index)
                    z = zs(linear_index)
                    cuda_inside = tabdip(linear_index) /= 0
#else
                    x = -(x2 - x1) / 2.0_wp + aretecube * (real(k, wp) - 0.5_wp)
                    y = -(y2 - y1) / 2.0_wp + aretecube * (real(j, wp) - 0.5_wp)
                    z = -(z2 - z1) / 2.0_wp + aretecube * (real(i, wp) - 0.5_wp)
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
                    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
                    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
                    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
                    if (abs(xr) <= sidex / 2.0_wp .and. abs(yr) <= sidey / 2.0_wp .and. abs(zr) <= sidez / 2.0_wp) then
#endif
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    if (cuda_inside) then
#else
                    if (abs(xr) <= sidex / 2.0_wp .and. abs(yr) <= sidey / 2.0_wp .and. abs(zr) <= sidez / 2.0_wp) then
#endif
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

    write(*,*) 'Cuboid::nbsphere=', nbsphere

end subroutine objetpara
