module precision
  use, intrinsic :: iso_c_binding
  implicit none
#ifdef USE_SIMP
  integer, parameter :: wp = c_float
  character(len=*), parameter :: fmt_real = '(E22.15)'
#else
  integer, parameter :: wp = c_double
  character(len=*), parameter :: fmt_real = '(D22.15)'
#endif

#ifdef USE_INT64
  integer, parameter :: ip = c_long_long
#else
  integer, parameter :: ip = c_int
#endif
  
  ! Constantes réelles
  real(kind=wp), parameter :: zero = 0.0_wp
  real(kind=wp), parameter :: one  = 1.0_wp
  real(kind=wp), parameter :: two  = 2.0_wp
  real(kind=wp), parameter :: pi   = 3.14159265358979323846_wp
  real(kind=wp), parameter :: twopi  = 2.0_wp*pi
  real(kind=wp), parameter :: c   = 299792458_wp
  real(kind=wp), parameter :: quatpieps0  = one/(c*c*1.0e-7_wp)
  real(kind=wp), parameter :: degtorad  = pi/180.0_wp
  real(kind=wp), parameter :: radtodeg  = 180.0_wp/pi
  ! Keep SI units in both precisions: input lengths are in nm, internal
  ! lengths are in m, and cross sections are in m^2.
  real(kind=wp), parameter :: nmtom   = 1.0e-9_wp
  real(kind=wp), parameter :: nmtosurf = 1.0_wp
  
  ! Constantes complexes
  complex(kind=wp), parameter :: icomp  = cmplx(0.0_wp, 1.0_wp, kind=wp)
  complex(kind=wp), parameter :: ZZERO  = cmplx(0.0_wp, 0.0_wp, kind=wp)
  complex(kind=wp), parameter :: ZONE   = cmplx(1.0_wp, 0.0_wp, kind=wp)

end module precision
