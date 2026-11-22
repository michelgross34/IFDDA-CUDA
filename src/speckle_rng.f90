! Preserve the Fortran random stream used by speckle.f90 while moving the
! spectral field construction and FFTs to CUDA. Only random unit phases are
! produced here; no incident-field values are calculated on the CPU.
subroutine ifdda_prepare_speckle_phases(nfft, nfftmax, k0, numaperil, spacing, &
     speckseed, bidirectional, phase_positive, phase_negative)
  use precision
  implicit none
  integer(ip), intent(in) :: nfft, nfftmax, speckseed, bidirectional
  real(wp), intent(in) :: k0, numaperil, spacing
  complex(wp), intent(out) :: phase_positive(nfftmax*nfftmax)
  complex(wp), intent(out) :: phase_negative(nfftmax*nfftmax)
  integer(ip) :: half, nkx, nky, ii, jj, index
  integer :: seed_size
  integer, allocatable :: seed(:)
  real(wp) :: deltak, kmax, kx, ky, kp2, ran

  phase_positive = zzero
  phase_negative = zzero
  half = nfft / 2
  kmax = k0 * numaperil
  deltak = twopi / (real(nfft, wp) * spacing)

  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  seed = speckseed
  call random_seed(put=seed)
  do nkx = -half, half - 1
     ii = modulo(nkx, nfft) + 1
     do nky = -half, half - 1
        kx = deltak * real(nkx, wp)
        ky = deltak * real(nky, wp)
        kp2 = kx*kx + ky*ky
        if (kp2 <= kmax*kmax*0.999_wp) then
           call random_number(ran)
           jj = modulo(nky, nfft) + 1
           index = ii + nfft * (jj - 1)
           phase_positive(index) = exp(icomp * twopi * ran)
        end if
     end do
  end do

  if (bidirectional /= 0) then
     seed = speckseed + 1
     call random_seed(put=seed)
     do nkx = -half, half - 1
        ii = modulo(nkx, nfft) + 1
        do nky = -half, half - 1
           kx = deltak * real(nkx, wp)
           ky = deltak * real(nky, wp)
           kp2 = kx*kx + ky*ky
           if (kp2 <= kmax*kmax*0.999_wp) then
              call random_number(ran)
              jj = modulo(nky, nfft) + 1
              index = ii + nfft * (jj - 1)
              phase_negative(index) = exp(icomp * twopi * ran)
           end if
        end do
     end do
  end if
  deallocate(seed)
end subroutine ifdda_prepare_speckle_phases
