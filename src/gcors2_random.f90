subroutine gcors2_random_values(n, random_imag, random_real) bind(C, name="gcors2_random_values")
  use precision
  use iso_c_binding, only: c_int
  implicit none
  integer(c_int), value, intent(in) :: n
  real(wp), intent(out) :: random_imag(n), random_real(n)
  integer :: i

  ! Continue main0's seeded stream with the draw order used by Fortran GCORS2.
  do i = 1, n
     call random_number(random_imag(i))
     call random_number(random_real(i))
  end do
end subroutine gcors2_random_values
