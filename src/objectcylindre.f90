subroutine objetcylindre(trope, eps, epsani, eps0, xs, ys, zs, k0, &
     aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, &
     nzm, methode, na, epsilon, polarisa, niso, rayon, hauteur, xg, yg, zg, &
     phi, theta, psi, sldr, nmat, file_id, group_iddip, infostr, nstop)
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
        integer(c_int) function ifdda_cylinder_geometry_cuda(nx, ny, nz, &
             cell_size, extent_x, extent_y, extent_z, radius, height, &
             m0, m1, m2, m3, m4, m5, m6, m7, m8, xs, ys, zs, inside_map) &
             bind(C, name="ifdda_cylinder_geometry_cuda")
            import :: c_int, c_float, c_double
            integer(c_int), value :: nx, ny, nz
#ifdef USE_SIMP
            real(c_float), value :: cell_size, extent_x, extent_y, extent_z
            real(c_float), value :: radius, height
            real(c_float), value :: m0, m1, m2, m3, m4, m5, m6, m7, m8
            real(c_float) :: xs(*), ys(*), zs(*)
#else
            real(c_double), value :: cell_size, extent_x, extent_y, extent_z
            real(c_double), value :: radius, height
            real(c_double), value :: m0, m1, m2, m3, m4, m5, m6, m7, m8
            real(c_double) :: xs(*), ys(*), zs(*)
#endif
            integer(c_int) :: inside_map(*)
        end function ifdda_cylinder_geometry_cuda
    end interface
#endif

    !--- Variables en entrée/sortie ---
    integer(ip), intent(in) :: nmax, nnnr, nxm, nym, nzm, nmat, niso, na
    integer(ip), intent(out) :: nbsphere, ndipole, nstop
    integer(ip), intent(inout) :: tabdip(nmax), nx, ny, nz
    real(wp), intent(in) :: k0, sldr
    real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, rayon, hauteur, xg, yg, zg, phi, theta, psi
    complex(wp), intent(in) :: eps, epsani(3,3), eps0
    complex(wp), intent(inout) :: epsilon(nmax, niso, niso), polarisa(nmax, niso, niso)
    character(2), intent(in) :: methode
    character(3), intent(in) :: trope
    character(64), intent(out) :: infostr


    !--- Variables locales ---
    integer(ip) :: i, j, k, ii, jj, test, dddis, inv
    real(wp) :: x, y, z, xr, yr, zr, demim, x1, x2, y1, y2, z1, z2, mat(3,3)
    complex(wp) :: ctmp, polaeps(3,3)
    character(LEN=100) :: datasetname
#ifndef USE_HDF5
    integer(ip), parameter :: hid_t=4
#endif
    integer(hid_t), intent(in) :: file_id, group_iddip
    integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    integer(c_int) :: cuda_geometry_status
    integer(ip) :: cylinder_count, linear_index
    logical :: cuda_inside
#endif

    !--- Initialisation ---
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

    rayon = rayon * nmtom
    hauteur = hauteur * nmtom
    write(*,*) 'cylinder radius =', rayon, 'height =', hauteur
    if (rayon == zero) then
        nstop = 1
        infostr = 'object cylinder: radius=0!'
        return
    endif
    if (hauteur == zero) then
        nstop = 1
        infostr = 'object cylinder: height=0!'
        return
    endif

    xg = xg * nmtom
    yg = yg * nmtom
    zg = zg * nmtom
    theta = theta * pi / 180.0_wp
    phi = phi * pi / 180.0_wp
    psi = psi * pi / 180.0_wp

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

    !--- Calcul des 8 coins du cylindre ---
    x = -rayon
    y = -rayon
    z = -hauteur / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = rayon
    y = -rayon
    z = -hauteur / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = -rayon
    y = rayon
    z = -hauteur / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = -rayon
    y = -rayon
    z = hauteur / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = rayon
    y = rayon
    z = -hauteur / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = rayon
    y = -rayon
    z = hauteur / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = -rayon
    y = rayon
    z = hauteur / 2.0_wp
    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
    x1 = min(x1, xr)
    x2 = max(x2, xr)
    y1 = min(y1, yr)
    y2 = max(y2, yr)
    z1 = min(z1, zr)
    z2 = max(z2, zr)

    x = rayon
    y = rayon
    z = hauteur / 2.0_wp
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
    demim = max(x2 - x1, y2 - y1, z2 - z1)
    aretecube = demim / real(nnnr, wp)

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
    if (na == -1 .or. na == 0) then
        if (nx <= 0 .or. ny <= 0 .or. nz <= 0 .or. nx > huge(cylinder_count) / ny) then
            nstop = 1
            infostr = 'object cylinder: invalid CUDA dimensions'
            return
        endif
        cylinder_count = nx * ny
        if (cylinder_count > huge(cylinder_count) / nz) then
            nstop = 1
            infostr = 'object cylinder: dimensions exceed CUDA index range'
            return
        endif
        cylinder_count = cylinder_count * nz
        if (cylinder_count > nmax) then
            nstop = 1
            infostr = 'nmax parameter too small: increase nxm nym nzm'
            return
        endif
        cuda_geometry_status = ifdda_cylinder_geometry_cuda(int(nx,c_int), &
             int(ny,c_int), int(nz,c_int), real(aretecube,kind=wp), &
             real(x2-x1,kind=wp), real(y2-y1,kind=wp), real(z2-z1,kind=wp), &
             real(rayon,kind=wp), real(hauteur,kind=wp), &
             real(mat(1,1),kind=wp), real(mat(1,2),kind=wp), &
             real(mat(1,3),kind=wp), real(mat(2,1),kind=wp), &
             real(mat(2,2),kind=wp), real(mat(2,3),kind=wp), &
             real(mat(3,1),kind=wp), real(mat(3,2),kind=wp), &
             real(mat(3,3),kind=wp), xs, ys, zs, tabdip)
        if (cuda_geometry_status /= 0) then
            nstop = 1
            infostr = 'CUDA cylinder geometry generation failed'
            return
        endif
    endif
#endif

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
                    x = -(x2 - x1) / 2.0_wp + aretecube * (real(k, wp) - 0.5_wp)
                    y = -(y2 - y1) / 2.0_wp + aretecube * (real(j, wp) - 0.5_wp)
                    z = -(z2 - z1) / 2.0_wp + aretecube * (real(i, wp) - 0.5_wp)
#endif

                    if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z + zg
                    if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y + yg
                    if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x + xg

                    ndipole = ndipole + 1

#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    if (cuda_inside) then
#else
                    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
                    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
                    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
                    if (xr*xr + yr*yr <= rayon*rayon .and. abs(zr) <= hauteur / 2.0_wp) then
#endif
                        nbsphere = nbsphere + 1
                        Tabdip(ndipole) = nbsphere
                        xs(nbsphere) = x + xg
                        ys(nbsphere) = y + yg
                        zs(nbsphere) = z + zg
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
                    linear_index = k + nx * (j - 1) + nx * ny * (i - 1)
                    x = xs(linear_index)
                    y = ys(linear_index)
                    z = zs(linear_index)
                    cuda_inside = tabdip(linear_index) /= 0
#else
                    x = -(x2 - x1) / 2.0_wp + aretecube * (real(k, wp) - 0.5_wp)
                    y = -(y2 - y1) / 2.0_wp + aretecube * (real(j, wp) - 0.5_wp)
                    z = -(z2 - z1) / 2.0_wp + aretecube * (real(i, wp) - 0.5_wp)
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
                    xr = mat(1,1)*x + mat(1,2)*y + mat(1,3)*z
                    yr = mat(2,1)*x + mat(2,2)*y + mat(2,3)*z
                    zr = mat(3,1)*x + mat(3,2)*y + mat(3,3)*z
                    if (xr*xr + yr*yr <= rayon*rayon .and. abs(zr) <= hauteur / 2.0_wp) then
#endif
                        xs(nbsphere) = x + xg
                        ys(nbsphere) = y + yg
                        zs(nbsphere) = z + zg
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
        write(*,*) 'Size of the parameter too small', ndipole, &
                   'should be smaller that', nmax
        write(*,*) 'Increase nxm nym and nzm'
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

end subroutine objetcylindre
