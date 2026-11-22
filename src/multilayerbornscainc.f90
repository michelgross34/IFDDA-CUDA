subroutine multilayerbornscainc(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, theta, phi, xgaus, ygaus, zgaus, beam, epsilon, polarisa&
     &, niso, ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, imagex, imagey, imagez, FF0, FFloc, FF, plan2f, plan2b, nstop,&
     & infostr)
    use precision
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    use, intrinsic :: iso_c_binding, only: c_int,c_float,c_double, &
         c_float_complex,c_double_complex
#endif
    implicit none
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    interface
        integer(c_int) function ifdda_generate_multilayer_projected_green_cuda( &
             green,nfft2d,aretecube,k0,ux,uy,uz) result(status) &
             bind(C,name="ifdda_generate_multilayer_projected_green_cuda_")
            import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
            complex(c_float_complex), intent(out) :: green(*)
            complex(c_float_complex), intent(in) :: ux,uy,uz
            real(c_float), intent(in) :: aretecube,k0
#else
            complex(c_double_complex), intent(out) :: green(*)
            complex(c_double_complex), intent(in) :: ux,uy,uz
            real(c_double), intent(in) :: aretecube,k0
#endif
            integer(c_int), intent(in) :: nfft2d
        end function ifdda_generate_multilayer_projected_green_cuda
        integer(c_int) function ifdda_multilayerbornsca_layers_cuda( &
             image,local_field,polarizability,plane_map,green_key,nfft2d, &
             nx,ny,nz,nmax,niso,reflection,deltakx,deltaky,aretecube,k0,ux,uy,uz) &
             result(status) bind(C,name="ifdda_multilayerbornsca_layers_cuda_")
            import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
            complex(c_float_complex), intent(inout) :: image(*),local_field(*)
            complex(c_float_complex), intent(in) :: polarizability(*),green_key(*)
            complex(c_float_complex), intent(in) :: ux,uy,uz
            real(c_float), intent(in) :: deltakx,deltaky,aretecube,k0
#else
            complex(c_double_complex), intent(inout) :: image(*),local_field(*)
            complex(c_double_complex), intent(in) :: polarizability(*),green_key(*)
            complex(c_double_complex), intent(in) :: ux,uy,uz
            real(c_double), intent(in) :: deltakx,deltaky,aretecube,k0
#endif
            integer(c_int), intent(in) :: plane_map(*),nfft2d,nx,ny,nz,nmax,niso,reflection
        end function ifdda_multilayerbornsca_layers_cuda
        integer(c_int) function ifdda_multilayer_scalar_initialize_plane_cuda( &
             image,local_field,incident_dipoles,xs,ys,zs,ndipole,nfft2d, &
             nmax,kkm,jjm,xmin,ymin,first_z,aretecube,k0,E0,eps0,w0,ss,pp,theta, &
             phi,beam_type,conjugate_projection,ux,uy,uz) result(status) &
             bind(C,name="ifdda_multilayer_scalar_initialize_plane_cuda_")
            import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
            complex(c_float_complex), intent(inout) :: image(*),local_field(*)
            complex(c_float_complex), intent(in) :: incident_dipoles(*),E0,eps0,ux,uy,uz
            real(c_float), intent(in) :: xs(*),ys(*),zs(*),xmin,ymin,first_z
            real(c_float), intent(in) :: aretecube,k0,w0,ss,pp,theta,phi
#else
            complex(c_double_complex), intent(inout) :: image(*),local_field(*)
            complex(c_double_complex), intent(in) :: incident_dipoles(*),E0,eps0,ux,uy,uz
            real(c_double), intent(in) :: xs(*),ys(*),zs(*),xmin,ymin,first_z
            real(c_double), intent(in) :: aretecube,k0,w0,ss,pp,theta,phi
#endif
            integer(c_int), intent(in) :: ndipole,nfft2d,nmax,kkm,jjm,beam_type,conjugate_projection
        end function ifdda_multilayer_scalar_initialize_plane_cuda
    end interface
#endif

    !--- Variables en entrée ---
    integer(ip), intent(in) :: nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, ndipole, niso
    real(wp), intent(in) :: xs(nmax), ys(nmax), zs(nmax), aretecube, k0, w0, theta, phi, xgaus, ygaus, zgaus
    complex(wp), intent(in) :: E0, ss, pp, epsilon(nmax, niso, niso), polarisa(nmax, niso, niso)
    character(64), intent(in) :: beam

    !--- Variables en sortie ---
    integer(ip), intent(out) :: nstop
    character(64), intent(out) :: infostr
    complex(wp), intent(out) :: imagex(nfft2d*nfft2d), imagey(nfft2d*nfft2d), imagez(nfft2d*nfft2d)
    complex(wp), intent(inout) :: FF0(3*nmax), FFloc(3*nmax), FF(3*nmax)

    !--- Variables locales ---
    integer(ip) :: i, j, k, l, kk, ii, jj, ll, kkm, jjm, indice, nloin, comparaison, nxl, nyl, tabfft(nfft2d*nfft2d)
    real(wp) :: xmax, xmin, ymax, ymin, x, y, z, Emod, Emodmax, tol, tolc
    complex(wp) :: ctmp, Uincx, Uincy, Uincz, Ex, Ey, Ez, propaesplibre(3,3), eps0
    real(wp) :: kz, kx, ky, deltakx, deltaky, fac, fac2, kzc, r, xx0, yy0, zz0, x0, y0, z0
    integer*8 :: plan2f, plan2b, plan4b, plan4f
    complex(wp), allocatable :: FFTTENSOR(:), polarisation(:), champ(:)
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    integer(c_int) :: cuda_status,cuda_nfft2d,cuda_nx,cuda_ny,cuda_nz
    integer(c_int) :: cuda_nmax,cuda_niso,cuda_reflection
    integer(c_int) :: cuda_ndipole,cuda_kkm,cuda_jjm,cuda_beam_type,cuda_conjugate_projection
#endif

    !--- Constantes ---
    integer(ip), parameter :: FFTW_BACKWARD = +1, FFTW_FORWARD = -1, FFTW_ESTIMATE = 64
    fac = one / real(nfft2d*nfft2d, wp)
    eps0=zone
    !--- Vérification de la taille de la FFT ---
    if (nfft2d <= nx) then
        nstop = 1
        infostr = 'nfft2d too small'
        return
    end if
    if (nfft2d <= ny) then
        nstop = 1
        infostr = 'nfft2d too small'
        return
    end if

    deltakx = twopi / (real(nfft2d, wp) * aretecube)
    deltaky = twopi / (real(nfft2d, wp) * aretecube)

    if (deltakx >= k0 / 2.0_wp) then
        nstop = 1
        infostr = 'nfft2d too small'
        return
    end if
    if (deltaky >= k0 / 2.0_wp) then
        nstop = 1
        infostr = 'nfft2d too small'
        return
    end if
    if (nstop == 1) return

    write(*,*) 'int', k0/deltakx, theta*180.0_wp/pi
    write(*,*) 'beam', beam

    !--- Calcul de l'orientation du vecteur de polarisation ---
    x = zero
    y = zero
    z = zero
    if (beam(1:11) == 'pwavelinear') then
        call ondeplane(x, y, z, k0, eps0, E0, ss, pp, theta, phi, Uincx, Uincy, Uincz, nstop, infostr)
    else if (beam(1:13) == 'pwavecircular') then
        call ondecirce(x, y, z, k0, E0, ss, theta, phi, Uincx, Uincy, Uincz)
    else if (beam(1:15) == 'gparawavelinear') then
        call gaussianparalinear(xgaus, ygaus, zgaus, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Uincx, Uincy, Uincz&
             &, nstop, infostr)
    else if (beam(1:17) == 'gparawavecircular') then
        call gaussianparacirc(xgaus, ygaus, zgaus, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, eps0, E0, Uincx, Uincy, Uincz,&
             & nstop, infostr)
    else if (beam(1:11) == 'gwavelinear' .or. beam(1:14) == 'gfftwavelinear') then
        nloin = 0
        tol = 1.0e-3_wp
        call gaussianchamp(xgaus, ygaus, zgaus, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, E0, Uincx, Uincy, Uincz, tol,&
             & nloin, nstop, infostr)
    else if (beam(1:13) == 'gwavecircular' .or. beam(1:16) == 'gfftwavecircular') then
        nloin = 0
        tol = 1.0e-3_wp
        call gaussianchampcirc(xgaus, ygaus, zgaus, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, E0, Uincx, Uincy, Uincz, tol,&
             & nloin, nstop, infostr)
    else
        write(*,*) 'Beam not implemented for this approximation'
        nstop = 1
        infostr = 'Beam not implemented for this approximation'
        return
    end if

    !--- Normalisation du vecteur unitaire ---
    Emod = sqrt(abs(Uincx)**2 + abs(Uincy)**2 + abs(Uincz)**2)
    Uincx = Uincx / Emod
    Uincy = Uincy / Emod
    Uincz = Uincz / Emod
    write(*,*) 'beam', beam

    !--- Calcul de la fonction de Green scalaire ---
    nxl = nfft2d
    nyl = nfft2d
    fac2 = one / real(2*nfft2d * 2*nfft2d, wp)

    !--- Allocation des tableaux ---
    allocate(FFTTENSOR(1:2*nfft2d*2*nfft2d))
    allocate(polarisation(1:2*nfft2d*2*nfft2d))
    allocate(champ(1:2*nfft2d*2*nfft2d))

    !--- Définition des plans FFT ---
#ifdef USE_FFTW
#ifdef USE_SIMP
    call sfftw_plan_dft_2d(plan4b, 2*nfft2d, 2*nfft2d, champ, champ, FFTW_BACKWARD, FFTW_ESTIMATE)
    call sfftw_plan_dft_2d(plan4f, 2*nfft2d, 2*nfft2d, champ, champ, FFTW_FORWARD, FFTW_ESTIMATE)    
#else
    call dfftw_plan_dft_2d(plan4b, 2*nfft2d, 2*nfft2d, champ, champ, FFTW_BACKWARD, FFTW_ESTIMATE)
    call dfftw_plan_dft_2d(plan4f, 2*nfft2d, 2*nfft2d, champ, champ, FFTW_FORWARD, FFTW_ESTIMATE)
#endif    
#endif

    !--- Calcul de la fonction de Green ---
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    cuda_nfft2d = int(nfft2d,c_int)
    cuda_status = ifdda_generate_multilayer_projected_green_cuda(FFTTENSOR, &
         cuda_nfft2d,aretecube,k0,Uincx,Uincy,Uincz)
    if (cuda_status /= 0_c_int) then
        write(*,*) 'CUDA projected Green generation status:',cuda_status
        nstop = 1
        infostr = 'CUDA projected Green generation failed'
        deallocate(FFTTENSOR,polarisation,champ)
        return
    end if
#else
    x0 = zero
    y0 = zero
    z0 = zero
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ii, jj, indice, i, j, xx0, yy0, zz0, propaesplibre)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
    do jj = 1, 2*nfft2d
        do ii = 1, 2*nfft2d
            indice = ii + 2*nfft2d * (jj - 1)
            if (ii == nxl + 1 .or. jj == nyl + 1) then
                FFTTENSOR(indice) = zzero
            else
                if (ii > nxl) then
                    i = (ii - 1) - 2*nfft2d
                else
                    i = ii - 1
                end if
                if (jj > nyl) then
                    j = (jj - 1) - 2*nfft2d
                else
                    j = jj - 1
                end if
                xx0 = real(i, wp) * aretecube
                yy0 = real(j, wp) * aretecube
                zz0 = aretecube

                call propa_espace_libre(x0, y0, z0, xx0, yy0, zz0, k0, propaesplibre)
                FFTTENSOR(indice) = conjg(Uincx) * (propaesplibre(1,1) * Uincx + propaesplibre(1,2) * Uincy + propaesplibre(1,3) *&
                     & Uincz) + conjg(Uincy) * (propaesplibre(2,1) * Uincx + propaesplibre(2,2) * Uincy + propaesplibre(2,3) *&
                     & Uincz) + conjg(Uincz) * (propaesplibre(3,1) * Uincx + propaesplibre(3,2) * Uincy + propaesplibre(3,3) *&
                     & Uincz)
            end if
        end do
    end do
!$OMP END DO
!$OMP END PARALLEL

    write(*,*) 'End computation of the scalar Green function'

    !--- FFT de la fonction de Green ---
#ifdef USE_FFTW
#ifdef USE_SIMP
    call sfftw_execute_dft(plan4b, FFTTENSOR, FFTTENSOR)    
#else
    call dfftw_execute_dft(plan4b, FFTTENSOR, FFTTENSOR)
#endif
#else
    call fftsingletonz2d(FFTTENSOR, 2*nfft2d, 2*nfft2d, FFTW_BACKWARD)
#endif

    write(*,*) 'End FFT scalar Green function'
#endif

    !--- Calcul des limites x et y ---
    xmax = -huge(1.0_wp)/100.0_wp
    xmin = huge(1.0_wp)/100.0_wp
    ymax = -huge(1.0_wp)/100.0_wp
    ymin = huge(1.0_wp)/100.0_wp

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
!$OMP DO SCHEDULE(STATIC) REDUCTION(max:xmax, ymax) REDUCTION(min:xmin, ymin)
    do i = 1, ndipole
        xmax = max(xmax, xs(i))
        xmin = min(xmin, xs(i))
        ymax = max(ymax, ys(i))
        ymin = min(ymin, ys(i))
    end do
!$OMP END DO
!$OMP END PARALLEL

    kkm = (nfft2d - nx) / 2
    jjm = (nfft2d - ny) / 2

    !--- Calcul de Emodmax ---
    Emodmax = zero
    if (beam(1:11) == 'gwavelinear' .or. beam(1:14) == 'gfftwavelinear') then
        do j = 1, nfft2d
            do k = 1, nfft2d
                x = xmin + real(k - kkm - 1, wp) * aretecube
                y = ymin + real(j - jjm - 1, wp) * aretecube
                call gaussianparalinear(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Ex, Ey, Ez, nstop,&
                     & infostr)
                Emodmax = max(Emodmax, real(Ex*conjg(Ex) + Ey*conjg(Ey) + Ez*conjg(Ez),wp))
            end do
        end do
    else if (beam(1:13) == 'gwavecircular' .or. beam(1:16) == 'gfftwavecircular') then
        do j = 1, nfft2d
            do k = 1, nfft2d
                x = xmin + real(k - kkm - 1, wp) * aretecube
                y = ymin + real(j - jjm - 1, wp) * aretecube
                call gaussianparalinear(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Ex, Ey, Ez, nstop,&
                     & infostr)
                Emodmax = max(Emodmax, real(Ex*conjg(Ex) + Ey*conjg(Ey) + Ez*conjg(Ez),wp))
            end do
        end do
    end if

    write(*,*) 'int', k0/deltakx, theta*180.0_wp/pi
    write(*,*) 'beam', beam

    !--- Calcul de l'orientation du vecteur de polarisation ---
    x = zero
    y = zero
    z = zero
    if (beam(1:11) == 'pwavelinear') then
        call ondeplane(x, y, z, k0, eps0, E0, ss, pp, theta, phi, Uincx, Uincy, Uincz, nstop, infostr)
    else if (beam(1:13) == 'pwavecircular') then
        call ondecirce(x, y, z, k0, E0, ss, theta, phi, Uincx, Uincy, Uincz)
    else if (beam(1:15) == 'gparawavelinear') then
        call gaussianparalinear(xgaus, ygaus, zgaus, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Uincx, Uincy, Uincz&
             &, nstop, infostr)
    else if (beam(1:17) == 'gparawavecircular') then
        call gaussianparacirc(xgaus, ygaus, zgaus, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, eps0, E0, Uincx, Uincy, Uincz,&
             & nstop, infostr)
    else if (beam(1:11) == 'gwavelinear' .or. beam(1:14) == 'gfftwavelinear') then
        nloin = 0
        tol = 1.0e-3_wp
        call gaussianchamp(xgaus, ygaus, zgaus, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, E0, Uincx, Uincy, Uincz, tol,&
             & nloin, nstop, infostr)
    else if (beam(1:13) == 'gwavecircular' .or. beam(1:16) == 'gfftwavecircular') then
        nloin = 0
        tol = 1.0e-3_wp
        call gaussianchampcirc(xgaus, ygaus, zgaus, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, E0, Uincx, Uincy, Uincz, tol,&
             & nloin, nstop, infostr)
    end if

    !--- Normalisation du vecteur unitaire ---
    Emod = sqrt(abs(Uincx)**2 + abs(Uincy)**2 + abs(Uincz)**2)
    Uincx = Uincx / Emod
    Uincy = Uincy / Emod
    Uincz = Uincz / Emod

    l = 1
    z = zs(1)

    !--- Initialisation du premier z au champ incident ---
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    if (beam(1:11) == 'pwavelinear' .or. beam(1:13) == 'pwavecircular' .or. &
        beam(1:15) == 'gparawavelinear' .or. beam(1:17) == 'gparawavecircular') then
        cuda_ndipole = int(ndipole,c_int)
        cuda_nfft2d = int(nfft2d,c_int)
        cuda_nmax = int(nmax,c_int)
        cuda_kkm = int(kkm,c_int)
        cuda_jjm = int(jjm,c_int)
        cuda_beam_type = 0_c_int
        if (beam(1:13) == 'pwavecircular') cuda_beam_type = 1_c_int
        if (beam(1:15) == 'gparawavelinear') cuda_beam_type = 2_c_int
        if (beam(1:17) == 'gparawavecircular') cuda_beam_type = 3_c_int
        cuda_conjugate_projection = 1_c_int
        cuda_status = ifdda_multilayer_scalar_initialize_plane_cuda( &
             imagex,FFloc,FF0,xs,ys,zs,cuda_ndipole,cuda_nfft2d,cuda_nmax, &
             cuda_kkm,cuda_jjm,xmin,ymin,z,aretecube,k0,E0,eps0,w0,real(ss,wp), &
             real(pp,wp),theta,phi,cuda_beam_type,cuda_conjugate_projection, &
             Uincx,Uincy,Uincz)
        if (cuda_status /= 0_c_int) then
            write(*,*) 'CUDA scalar incident first-plane initialization status:',cuda_status
            nstop = 1
            infostr = 'CUDA scalar incident first-plane initialization failed'
            deallocate(FFTTENSOR,polarisation,champ)
            return
        end if
    else
#endif
    do j = 1, nfft2d
        do k = 1, nfft2d
            ii = k + (j - 1) * nfft2d
            x = xmin + real(k - kkm - 1, wp) * aretecube
            y = ymin + real(j - jjm - 1, wp) * aretecube
            if (comparaison(x, y, z, xs(l), ys(l), z, aretecube) == 1 .and. l <= ndipole) then
                tabfft(ii) = l
                kk = 3 * (l - 1)
                imagex(ii) = conjg(Uincx) * FF0(kk + 1) + conjg(Uincy) * FF0(kk + 2) + conjg(Uincz) * FF0(kk + 3)
                FFloc(kk + 1) = FF0(kk + 1)
                FFloc(kk + 2) = FF0(kk + 2)
                FFloc(kk + 3) = FF0(kk + 3)
                l = l + 1
            else
                tabfft(ii) = 0
                if (beam(1:11) == 'pwavelinear') then
                    call ondeplane(x, y, z, k0, eps0, E0, ss, pp, theta, phi, Ex, Ey, Ez, nstop, infostr)
                else if (beam(1:13) == 'pwavecircular') then
                    call ondecirce(x, y, z, k0, E0, ss, theta, phi, Ex, Ey, Ez)
                else if (beam(1:15) == 'gparawavelinear') then
                    call gaussianparalinear(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Ex, Ey, Ez, nstop,&
                         & infostr)
                else if (beam(1:17) == 'gparawavecircular') then
                    call gaussianparacirc(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, eps0, E0, Ex, Ey, Ez, nstop,&
                         & infostr)
                else if (beam(1:11) == 'gwavelinear' .or. beam(1:14) == 'gfftwavelinear') then
                    nloin = 0
                    tol = 1.0e-3_wp
                    call gaussianparalinear(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Ex, Ey, Ez, nstop,&
                         & infostr)
                    Emod = real(Ex*conjg(Ex) + Ey*conjg(Ey) + Ez*conjg(Ez),wp)
                    tolc = sqrt(Emod / Emodmax) * 10.0_wp
                    if (tolc >= tol) then
                        call gaussianchamp(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, E0, Ex, Ey, Ez, tol, nloin,&
                             & nstop, infostr)
                    end if
                else if (beam(1:13) == 'gwavecircular' .or. beam(1:16) == 'gfftwavecircular') then
                    nloin = 0
                    tol = 1.0e-3_wp
                    call gaussianparacirc(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, eps0, E0, Ex, Ey, Ez, nstop,&
                         & infostr)
                    Emod = real(Ex*conjg(Ex) + Ey*conjg(Ey) + Ez*conjg(Ez),wp)
                    tolc = sqrt(Emod / Emodmax) * 10.0_wp
                    if (tolc >= tol) then
                        call gaussianchampcirc(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, E0, Ex, Ey, Ez, tol, nloin,&
                             & nstop, infostr)
                    end if
                end if
                imagex(ii) = conjg(Uincx) * Ex + conjg(Uincy) * Ey + conjg(Uincz) * Ez
            end if
        end do
    end do
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    end if
#endif

    !--- Boucle sur les z ---
#ifdef IFDDA_CUDA_INVERSE_DRIVER
    cuda_nx = int(nx,c_int)
    cuda_ny = int(ny,c_int)
    cuda_nz = int(nz,c_int)
    cuda_nmax = int(nmax,c_int)
    cuda_niso = int(niso,c_int)
    cuda_reflection = 0_c_int
    cuda_status = ifdda_multilayerbornsca_layers_cuda(imagex,FFloc,polarisa, &
         tabfft,FFTTENSOR,cuda_nfft2d,cuda_nx,cuda_ny,cuda_nz,cuda_nmax, &
         cuda_niso,cuda_reflection,deltakx,deltaky,aretecube,k0,Uincx,Uincy,Uincz)
    if (cuda_status /= 0_c_int) then
        write(*,*) 'CUDA multilayer projected propagation status:',cuda_status
        nstop = 1
        infostr = 'CUDA multilayer projected propagation failed'
        deallocate(FFTTENSOR,polarisation,champ)
        return
    end if
#else
    do k = 2, nz

        !--- Calcul de la polarisation ---
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i, j, ii)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
        do j = 1, 2*nfft2d
            do i = 1, 2*nfft2d
                ii = i + (j - 1) * 2*nfft2d
                polarisation(ii) = zzero
            end do
        end do
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i, j, ii, jj, l)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
        do j = 1, nyl
            do i = 1, nxl
                ii = i + (j - 1) * 2*nfft2d
                jj = i + (j - 1) * nxl
                if (tabfft(jj) /= 0) then
                    l = nx * ny * (k - 1) + tabfft(jj)
                    polarisation(ii) = imagex(jj) * polarisa(l, 1, 1)
                end if
            end do
        end do
!$OMP END DO
!$OMP END PARALLEL

        !--- FFT de la polarisation ---
#ifdef USE_FFTW
#ifdef USE_SIMP
        call sfftw_execute_dft(plan4b, polarisation, polarisation)        
#else
        call dfftw_execute_dft(plan4b, polarisation, polarisation)
#endif
#else
        call fftsingletonz2d(polarisation, 2*nfft2d, 2*nfft2d, FFTW_BACKWARD)
#endif

        !--- Produit des deux FFT ---
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i, j, ii)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
        do j = 1, 2*nfft2d
            do i = 1, 2*nfft2d
                ii = i + (j - 1) * 2*nfft2d
                champ(ii) = polarisation(ii) * FFTTENSOR(ii)
            end do
        end do
!$OMP END DO
!$OMP END PARALLEL

        !--- FFT inverse pour obtenir le champ diffracté ---
#ifdef USE_FFTW
#ifdef USE_SIMP
        call sfftw_execute_dft(plan4f, champ, champ)        
#else    
        call dfftw_execute_dft(plan4f, champ, champ)
#endif
#else
        call fftsingletonz2d(champ, 2*nfft2d, 2*nfft2d, FFTW_FORWARD)
#endif

        !--- FFT à deux dimensions ---
#ifdef USE_FFTW
#ifdef USE_SIMP
        call sfftw_execute_dft(plan2b, imagex, imagex)
#else    
        call dfftw_execute_dft(plan2b, imagex, imagex)
#endif
#else
        call fftsingletonz2d(imagex, nfft2d, nfft2d, FFTW_BACKWARD)
#endif

        !--- Propagation du champ d'une couche ---
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i, j, indice, ii, jj, kx, ky, kz, ctmp)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
        do j = 1, nfft2d
            do i = 1, nfft2d
                indice = i + nfft2d * (j - 1)
                if (i > nfft2d / 2) then
                    ii = i - nfft2d - 1
                else
                    ii = i - 1
                end if
                if (j > nfft2d / 2) then
                    jj = j - nfft2d - 1
                else
                    jj = j - 1
                end if
                kx = deltakx * real(ii, wp)
                ky = deltaky * real(jj, wp)
                kz = k0**2 - kx**2 - ky**2
                if (kz >= zero) then
                    kz = sqrt(kz)
                    ctmp = exp(icomp * kz * aretecube)
                    imagex(indice) = imagex(indice) * ctmp
                else
                    imagex(indice) = zzero
                end if
            end do
        end do
!$OMP END DO
!$OMP END PARALLEL

        !--- FFT inverse ---
#ifdef USE_FFTW
#ifdef USE_SIMP
        call sfftw_execute_dft(plan2f, imagex, imagex)
#else
        call dfftw_execute_dft(plan2f, imagex, imagex)
#endif       
#else
        call fftsingletonz2d(imagex, nfft2d, nfft2d, FFTW_FORWARD)
#endif

        !--- Somme des deux champs ---
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i, j, indice, ii, ll, kk)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
        do j = 1, nyl
            do i = 1, nxl
                indice = i + (j - 1) * nxl
                ll = i + (j - 1) * 2*nfft2d
                imagex(indice) = imagex(indice) * fac + champ(ll) * fac2
                if (tabfft(indice) /= 0) then
                    ii = nx * ny * (k - 1) + tabfft(indice)
                    kk = 3 * (ii - 1)
                    FFloc(kk + 1) = imagex(indice) * Uincx
                    FFloc(kk + 2) = imagex(indice) * Uincy
                    FFloc(kk + 3) = imagex(indice) * Uincz
                end if
            end do
        end do
!$OMP END DO
!$OMP END PARALLEL
    end do

    !--- Libération des tableaux ---
#endif
    deallocate(FFTTENSOR)
    deallocate(polarisation)
    deallocate(champ)

end subroutine multilayerbornscainc
