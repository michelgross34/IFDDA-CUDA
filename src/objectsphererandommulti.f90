subroutine objetsphererandommulti(trope, epss, epsanis, numbersphere, numberspheremax, xg, yg, zg, rayons, eps0, xs, ys, zs, k0,&
     & aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, methode, epsilon, polarisa, niso, sidex, sidey,&
     & sidez, xsphe, ysphe, zsphe, density, nseed, na, sldr, nmat, file_id, group_iddip, infostr, nstop)
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
             cell_size, x_min, y_min, z_min, sphere_count, fixed_count, &
             parameters, xs, ys, zs, sphere_map) &
             bind(C, name="ifdda_nspheres_geometry_cuda")
            import :: c_int, c_float, c_double
            integer(c_int), value :: nx, ny, nz, sphere_count, fixed_count
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
    integer(ip), intent(in) :: nmax, nnnr, nxm, nym, nzm, nmat, niso, nseed, numbersphere, numberspheremax
    integer(ip), intent(out) :: nbsphere, ndipole, nstop
    integer(ip), intent(inout) :: tabdip(nmax), nx, ny, nz, na
    real(wp), intent(in) :: k0, sldr
    real(wp), intent(inout) :: xs(nmax), ys(nmax), zs(nmax), aretecube, xsphe(nmax), ysphe(nmax), zsphe(nmax), density, sidex,&
         & sidey, sidez, rayons(numberspheremax), xg(numberspheremax), yg(numberspheremax), zg(numberspheremax)
    complex(wp), intent(inout) :: epss(numberspheremax), epsanis(3,3,numberspheremax), epsilon(nmax, niso, niso), polarisa(nmax,&
         & niso, niso), eps0
    character(2), intent(in) :: methode
    character(3), intent(in) :: trope
    character(64), intent(out) :: infostr

    !--- Variables locales ---
    integer(ip) :: i, j, k, ks, ii, jj, test, is, ns, nmaxite, ngraine, dddis, inv
    real(wp) :: x, y, z, xt, yt, zt, sunif, side, dist, volsphere, volspheremulti, xmax, xmin, ymax, ymin, zmax, zmin, tmp, radius,&
         & ran
    integer, allocatable :: seed(:) 
    integer :: seed_size    
    complex(wp) :: ctmp, polaeps(3,3), epsani(3,3), eps
    character(LEN=100) :: datasetname
#ifndef USE_HDF5
    integer(ip), parameter :: hid_t=4
#endif
    integer(hid_t), intent(in) :: file_id, group_iddip    
    integer(ip) :: dim(4), error
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    real(wp), allocatable :: cuda_mesh_x(:), cuda_mesh_y(:), cuda_mesh_z(:)
    real(wp), allocatable :: cuda_sphere_parameters(:)
    integer(c_int), allocatable :: cuda_sphere_map(:)
    integer(c_int) :: cuda_geometry_status, flat_index, cuda_sphere_count
#endif

 ! 1. Récupère la taille requise pour la graine
  call random_seed(size=seed_size)
  ! 2. Alloue un tableau de cette taille
  allocate(seed(seed_size))
  seed = nseed
  ! Initialise ran
  call random_seed(put=seed)
  

    !--- Initialisation ---
    nbsphere = 0
    ndipole = 0
    call polaepsto0(polarisa, epsilon, tabdip, niso, nmax)
    dddis = 1
    inv = 1
    ngraine = 4 * nseed + 1

    !--- Conversion des unités ---
    sidex = sidex * nmtom
    sidey = sidey * nmtom
    sidez = sidez * nmtom
    do i = 1, numbersphere
        xg(i) = xg(i) * nmtom
        yg(i) = yg(i) * nmtom
        zg(i) = zg(i) * nmtom
        rayons(i) = rayons(i) * nmtom
        if (rayons(i) >= sidex .or. rayons(i) >= sidey .or. rayons(i) >= sidez) then
            nstop = 1
            infostr = 'Radius larger than the side of the box'
            return
        endif
    enddo

    !--- Échange des valeurs de la première et dernière sphère ---
    tmp = rayons(numbersphere)
    rayons(numbersphere) = rayons(1)
    rayons(1) = tmp
    tmp = xg(numbersphere)
    xg(numbersphere) = xg(1)
    xg(1) = tmp
    tmp = yg(numbersphere)
    yg(numbersphere) = yg(1)
    yg(1) = tmp
    tmp = zg(numbersphere)
    zg(numbersphere) = zg(1)
    zg(1) = tmp
    ctmp = epss(numbersphere)
    epss(numbersphere) = epss(1)
    epss(1) = ctmp

    radius = rayons(1)

    !--- Calcul des limites de la boîte ---
    xmax = -huge(1.0_wp)/100.0_wp
    xmin = huge(1.0_wp)/100.0_wp
    ymax = -huge(1.0_wp)/100.0_wp
    ymin = huge(1.0_wp)/100.0_wp
    zmax = -huge(1.0_wp)/100.0_wp
    zmin = huge(1.0_wp)/100.0_wp
    do is = 2, numbersphere
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

    !--- Vérification que les sphères sont dans la boîte ---
    if (xmax > xg(1) + sidex / 2.0_wp) then
        infostr = 'Spheres not in the box x1'
        nstop = 1
        return
    elseif (xmin < xg(1) - sidex / 2.0_wp) then
        infostr = 'Spheres not in the box x2'
        nstop = 1
        return
    elseif (ymax > yg(1) + sidey / 2.0_wp) then
        infostr = 'Spheres not in the box y1'
        nstop = 1
        return
    elseif (ymin < yg(1) - sidey / 2.0_wp) then
        infostr = 'Spheres not in the box y2'
        nstop = 1
        return
    elseif (zmax > zg(1) + sidez / 2.0_wp) then
        infostr = 'Spheres not in the box z1'
        nstop = 1
        return
    elseif (zmin < zg(1) - sidez / 2.0_wp) then
        infostr = 'Spheres not in the box z2'
        nstop = 1
        return
    endif

    !--- Calcul des dimensions de la grille ---
    side = max(sidex, sidey, sidez)
    aretecube = side / real(nnnr, wp)
    nx = int(sidex / aretecube)
    ny = int(sidey / aretecube)
    nz = int(sidez / aretecube)

    do i = 1, numbersphere
        if (rayons(i) < aretecube) then
            nstop = 1
            infostr = 'Radius smaller than the meshsize'
            return
        endif
    enddo

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

    !--- Calcul du nombre de sphères aléatoires ---
    if (density >= 0.5_wp .and. density < 2.0_wp) then
        nstop = 1
        infostr = 'density too large!'
        return
    elseif (density <= zero) then
        nstop = 1
        infostr = 'density negative!!'
        return
    endif

    volsphere = 4.0_wp * pi / 3.0_wp * radius * radius * radius
    volspheremulti = zero
    do i = 2, numbersphere
        volspheremulti = volspheremulti + 4.0_wp * pi / 3.0_wp * rayons(i)**3
    enddo

    ns = int(density * (sidex * sidey * sidez - volspheremulti) / volsphere)
    write(*,*) 'density', ns, sidex, sidey, sidez, sidex * sidey * sidez - volspheremulti, volsphere, volspheremulti

    if (ns <= 1) then
        nstop = 1
        infostr = 'density too small!'
        return
    endif

    if (density > 2.0_wp) then
        ns = int(density)
        density = real(ns, wp) * volsphere / sidex / sidey / sidez
    endif

    !--- Tirage aléatoire des centres des sphères ---
    ks = 0
    nmaxite = max(1000, ns * 10)

    do i = 1, nmaxite
        !--- Tirage aléatoire dans la boîte ---
        call random_number(ran)
        xt = -sidex / 2.0_wp + radius + (sidex - 2.0_wp * radius) * ran !SUNIF(ngraine)
        call random_number(ran)
        yt = -sidey / 2.0_wp + radius + (sidey - 2.0_wp * radius) * ran ! SUNIF(ngraine)
        call random_number(ran)
        zt = -sidez / 2.0_wp + radius + (sidez - 2.0_wp * radius) * ran ! SUNIF(ngraine)

        !--- Vérifie que la sphère n'est pas dans une sphère multi ---
        do is = 2, numbersphere
            dist = sqrt((xt - xg(is))**2 + (yt - yg(is))**2 + (zt - zg(is))**2)
            if (dist < radius + rayons(is)) exit
        enddo

        !--- Test d'intersection ---
        if (ks >= 1) then
            do j = 1, ks
                dist = sqrt((xt - xsphe(j))**2 + (yt - ysphe(j))**2 + (zt - zsphe(j))**2)
                if (dist < radius * 2.0_wp) exit
            enddo
        endif

        ks = ks + 1
        xsphe(ks) = xt
        ysphe(ks) = yt
        zsphe(ks) = zt
        if (ks == ns) exit
    enddo

    write(*,*) 'Number of sphere', ks, na, nx, ny, nz

#ifdef IFDDA_CUDA_INVERSE_DRIVER
    if (na == -1 .or. na == 0) then
        cuda_sphere_count = numbersphere - 1 + ks
        allocate(cuda_mesh_x(nx*ny*nz), cuda_mesh_y(nx*ny*nz), &
                 cuda_mesh_z(nx*ny*nz), cuda_sphere_map(nx*ny*nz), &
                 cuda_sphere_parameters(4*cuda_sphere_count))
        do is = 2, numbersphere
            flat_index = is - 1
            cuda_sphere_parameters(4*flat_index-3) = xg(is)
            cuda_sphere_parameters(4*flat_index-2) = yg(is)
            cuda_sphere_parameters(4*flat_index-1) = zg(is)
            cuda_sphere_parameters(4*flat_index) = rayons(is)
        enddo
        do is = 1, ks
            flat_index = numbersphere - 1 + is
            cuda_sphere_parameters(4*flat_index-3) = xsphe(is)
            cuda_sphere_parameters(4*flat_index-2) = ysphe(is)
            cuda_sphere_parameters(4*flat_index-1) = zsphe(is)
            cuda_sphere_parameters(4*flat_index) = radius
        enddo
        cuda_geometry_status = ifdda_nspheres_geometry_cuda(nx, ny, nz, &
            aretecube, -sidex/2.0_wp, -sidey/2.0_wp, -sidez/2.0_wp, &
            cuda_sphere_count, numbersphere - 1, cuda_sphere_parameters, &
            cuda_mesh_x, cuda_mesh_y, cuda_mesh_z, cuda_sphere_map)
        if (cuda_geometry_status /= 0) then
            nstop = 1
            infostr = 'CUDA random multi-sphere geometry failed'
            return
        endif
    endif
#endif

    !--- Construction des sphères ---
    na = 0
    do i = 1, nz
        do j = 1, ny
            do k = 1, nx
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                flat_index = (i - 1)*nx*ny + (j - 1)*nx + k
                x = cuda_mesh_x(flat_index)
                y = cuda_mesh_y(flat_index)
                z = cuda_mesh_z(flat_index)
#else
                x = -sidex / 2.0_wp + aretecube * (real(k, wp) - 0.5_wp)
                y = -sidey / 2.0_wp + aretecube * (real(j, wp) - 0.5_wp)
                z = -sidez / 2.0_wp + aretecube * (real(i, wp) - 0.5_wp)
#endif

                if (j == 1 .and. k == 1 .and. nmat == 0) write(22,*) z + zg(1)
                if (i == 1 .and. k == 1 .and. nmat == 0) write(21,*) y + yg(1)
                if (j == 1 .and. i == 1 .and. nmat == 0) write(20,*) x + xg(1)

                ndipole = ndipole + 1
                nbsphere = nbsphere + 1
                Tabdip(ndipole) = nbsphere

                !--- Test des sphères multi ---
#ifdef IFDDA_CUDA_INVERSE_DRIVER
                test = cuda_sphere_map(flat_index)
                if (test < 0) then
                    infostr = 'spheres are not distincts'
                    nstop = 1
                    return
                elseif (test > 0 .and. test < numbersphere) then
                    is = test + 1
                    xs(nbsphere) = x + xg(1)
                    ys(nbsphere) = y + yg(1)
                    zs(nbsphere) = z + zg(1)
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
                    test = -1
                elseif (test >= numbersphere) then
                    test = test - numbersphere + 1
                endif
#else
                test = 0
                do is = 2, numbersphere
                    if ((x - xg(is))**2 + (y - yg(is))**2 + (z - zg(is))**2 <= rayons(is)**2) then
                        test = test + 1
                        xs(nbsphere) = x + xg(1)
                        ys(nbsphere) = y + yg(1)
                        zs(nbsphere) = z + zg(1)
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
                    infostr = 'spheres are not distincts'
                    nstop = 1
                    return
                endif
#endif

                !--- Test des petites sphères ---
                if (test == 0 .or. test > 0) then
#ifndef IFDDA_CUDA_INVERSE_DRIVER
                    do is = 1, ks
                        if ((x - xsphe(is))**2 + (y - ysphe(is))**2 + (z - zsphe(is))**2 <= radius**2) then
                            test = is
                        endif
                    enddo
#endif

#ifdef IFDDA_CUDA_INVERSE_DRIVER
                    if (test > 0) then
#else
                    if (test /= 0) then
#endif
                        xs(nbsphere) = x + xg(1)
                        ys(nbsphere) = y + yg(1)
                        zs(nbsphere) = z + zg(1)
                        if (trope == 'iso') then
                            call poladiff(aretecube, epss(1), eps0, k0, dddis, &
                                         methode, sldr, ctmp)
                            polarisa(nbsphere,1,1) = ctmp
                            epsilon(nbsphere,1,1) = epss(1)
                        else
                            do ii = 1, 3
                                do jj = 1, 3
                                    epsani(ii,jj) = epsanis(ii,jj,1)
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
                        xs(nbsphere) = x + xg(1)
                        ys(nbsphere) = y + yg(1)
                        zs(nbsphere) = z + zg(1)
                    endif
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
    deallocate(seed)

    write(*,*) 'Random sphere::nbsphere    =', nbsphere, ndipole
    write(*,*) 'density of sphere obtained =', real(ks, wp) * volsphere / sidex / sidey / sidez

end subroutine objetsphererandommulti
