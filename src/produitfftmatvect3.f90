subroutine produitfftmatvect3(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz,&
     & Tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, FFX, FFB, planb, planf)
    use precision
    implicit none

    !--- Variables en entrée/sortie ---
    integer(ip), intent(in) :: nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2
    integer(ip), intent(in) :: ntotal, ntotalm
    integer(ip) :: indice
    integer(ip), intent(in) :: Tabdip(nxm*nym*nzm)
    integer*8, intent(in) :: planf, planb
    complex(wp), intent(in) :: FFTTENSORxx((nxm+1)*(nym+1)*(nzm+1)), FFTTENSORxy((nxm+1)*(nym+1)*(nzm+1)), FFTTENSORxz((nxm+1)*(nym&
         &+1)*(nzm+1)), FFTTENSORyy((nxm+1)*(nym+1)*(nzm+1)), FFTTENSORyz((nxm+1)*(nym+1)*(nzm+1)), FFTTENSORzz((nxm+1)*(nym+1)&
         &*(nzm+1))
    complex(wp), intent(inout), dimension(nx2*ny2*nz2) :: vectx, vecty, vectz
    complex(wp), intent(in) :: FFX(3*nmax)
    complex(wp), intent(inout) :: FFB(3*nmax)

    !--- Variables locales ---
    integer(ip) :: ii, jj, kk, i, j, k, n, nxy, iii, jjj, kkk, nx1, nxy1
    real(wp) :: dntotal, signx, signy, signz
    complex(wp) :: ctmpx, ctmpy, ctmpz
    integer(ip) :: FFTW_FORWARD, FFTW_ESTIMATE, FFTW_BACKWARD

    !--- Initialisation ---
    FFTW_FORWARD = -1
    FFTW_BACKWARD = +1
    FFTW_ESTIMATE = 64
    nxy = nx * ny
    dntotal = real(ntotal, wp)
    nx1 = nx + 1
    nxy1 = nx1 * (ny + 1)

    !--- Initialisation des vecteurs ---
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(indice)
!$OMP DO SCHEDULE(STATIC)
    do indice = 1, ntotal
        vectx(indice) = zzero
        vecty(indice) = zzero
        vectz(indice) = zzero
    enddo
!$OMP ENDDO
!$OMP END PARALLEL

    !--- Remplissage des vecteurs ---
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ii, jj, i, j, k, indice)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(3)
    do k = 1, nz
        do j = 1, ny
            do i = 1, nx
                ii = i + nx * (j - 1) + nxy * (k - 1)
                if (Tabdip(ii) /= 0) then
                    indice = i + nx2 * (j - 1) + nxy2 * (k - 1)
                    jj = 3 * Tabdip(ii)
                    vectx(indice) = FFX(jj - 2)
                    vecty(indice) = FFX(jj - 1)
                    vectz(indice) = FFX(jj)
                endif
            enddo
        enddo
    enddo
!$OMP ENDDO
!$OMP END PARALLEL

    !--- FFT des vecteurs ---
#ifdef USE_FFTW
#ifdef USE_SIMP
    call sfftw_execute_dft(planb, vectx, vectx)
    call sfftw_execute_dft(planb, vecty, vecty)
    call sfftw_execute_dft(planb, vectz, vectz)    
#else
    call dfftw_execute_dft(planb, vectx, vectx)
    call dfftw_execute_dft(planb, vecty, vecty)
    call dfftw_execute_dft(planb, vectz, vectz)
#endif
#else
!$OMP PARALLEL DEFAULT(SHARED)
!$OMP SECTIONS
!$OMP SECTION
    call fftsingletonz3d(vectx, NX2, NY2, NZ2, FFTW_BACKWARD)
!$OMP SECTION
    call fftsingletonz3d(vecty, NX2, NY2, NZ2, FFTW_BACKWARD)
!$OMP SECTION
    call fftsingletonz3d(vectz, NX2, NY2, NZ2, FFTW_BACKWARD)
!$OMP END SECTIONS
!$OMP END PARALLEL
#endif

    !--- Produit matrice-vecteur ---
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(indice, ctmpx, ctmpy, ctmpz, kk, jj, ii, kkk, jjj, iii, i, signx, signy, signz)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(3)
    do kk = 1, nz2
        do jj = 1, ny2
            do ii = 1, nx2
               indice = ii + nx2 * (jj - 1) + nxy2 * (kk - 1)
                if (kk <= nz + 1) then
                    kkk = kk
                    signz = one
                else
                    kkk = nz2 - kk + 2
                    signz = -one
                endif
                if (jj <= ny + 1) then
                    jjj = jj
                    signy = one
                else
                    jjj = ny2 - jj + 2
                    signy = -one
                endif
                if (ii <= nx + 1) then
                    iii = ii
                    signx = one
                else
                    iii = nx2 - ii + 2
                    signx = -one
                endif
                i = iii + nx1 * (jjj - 1) + nxy1 * (kkk - 1)
                ctmpx = vectx(indice)
                ctmpy = vecty(indice)
                ctmpz = vectz(indice)
                vectx(indice)=FFTTENSORxx(i)*ctmpx + FFTTENSORxy(i) * signx * signy * ctmpy + FFTTENSORxz(i) * signx * signz * ctmpz
                vecty(indice)=FFTTENSORxy(i)*signx * signy * ctmpx + FFTTENSORyy(i) * ctmpy + FFTTENSORyz(i) * signy * signz * ctmpz
                vectz(indice)=FFTTENSORxz(i)*signx * signz * ctmpx + FFTTENSORyz(i) * signy * signz * ctmpy + FFTTENSORzz(i) * ctmpz
            enddo
        enddo
    enddo
!$OMP ENDDO
!$OMP END PARALLEL

    !--- FFT inverse ---
#ifdef USE_FFTW
#ifdef USE_SIMP
    call sfftw_execute_dft(planf, vectx, vectx)
    call sfftw_execute_dft(planf, vecty, vecty)
    call sfftw_execute_dft(planf, vectz, vectz)
#else
    call dfftw_execute_dft(planf, vectx, vectx)
    call dfftw_execute_dft(planf, vecty, vecty)
    call dfftw_execute_dft(planf, vectz, vectz)
#endif
#else
!$OMP PARALLEL DEFAULT(SHARED)
!$OMP SECTIONS
!$OMP SECTION
    call fftsingletonz3d(vectx, NX2, NY2, NZ2, FFTW_FORWARD)
!$OMP SECTION
    call fftsingletonz3d(vecty, NX2, NY2, NZ2, FFTW_FORWARD)
!$OMP SECTION
    call fftsingletonz3d(vectz, NX2, NY2, NZ2, FFTW_FORWARD)
!$OMP END SECTIONS
!$OMP END PARALLEL
#endif

    !--- Mise à jour de FFB ---
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(indice, jj, i, j, k, n)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(3)
    do k = 1, nz
        do j = 1, ny
            do i = 1, nx
                n = i + nx * (j - 1) + nxy * (k - 1)
                if (Tabdip(n) /= 0) then
                   indice = i + nx2 * (j - 1) + nxy2 * (k - 1)
                   jj = 3 * Tabdip(n)
                   FFB(jj - 2) = FFB(jj - 2) - vectx(indice) / dntotal
                   FFB(jj - 1) = FFB(jj - 1) - vecty(indice) / dntotal
                   FFB(jj) = FFB(jj) - vectz(indice) / dntotal
                endif
            enddo
        enddo
    enddo
!$OMP ENDDO
!$OMP END PARALLEL

end subroutine produitfftmatvect3
