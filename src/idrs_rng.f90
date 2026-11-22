subroutine idrs_fill_random(ndim, ns, values)
  use precision
  implicit none
  integer, intent(in) :: ndim, ns
  complex(wp), intent(out) :: values(ndim, ns)
  character(len=32) :: seed_env
  integer :: status, read_status, seed_value, seed_size, i, j
  integer, allocatable :: seed_values(:)
  real(wp) :: alea_real, alea_imag

  call get_environment_variable('IFDDA_IDRS_SEED', seed_env, status=status)
  if (status == 0) then
     read(seed_env, *, iostat=read_status) seed_value
     if (read_status /= 0) then
        error stop 'Invalid IFDDA_IDRS_SEED: expected a decimal integer'
     end if
     call random_seed(size=seed_size)
     allocate(seed_values(seed_size))
     seed_values = seed_value
     call random_seed(put=seed_values)
     deallocate(seed_values)
  else
     call random_seed()
  end if

  do j = 1, ns
     do i = 1, ndim
        call random_number(alea_imag)
        call random_number(alea_real)
        values(i,j) = cmplx(alea_real, alea_imag, kind=wp)
     end do
  end do
end subroutine idrs_fill_random
