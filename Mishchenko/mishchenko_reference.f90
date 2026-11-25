! Cross sections for the fixed-orientation NASA GISS EBCM.
module mishchenko_reference
  use, intrinsic :: iso_c_binding
  use, intrinsic :: ieee_arithmetic
  implicit none
  private
  public :: ifdda_mishchenko_v1
  real(c_double), parameter :: pi = acos(-1.0_c_double)
contains
  integer(c_int) function ifdda_mishchenko_v1(shape, radius, ratio, longest, mr, mi, theta, &
      ep_real, ep_imag, es_real, es_imag, sections, order, quadrature, variation) &
      bind(c, name="ifdda_mishchenko_v1") result(status)
    integer(c_int), value :: shape
    real(c_double), value :: radius, ratio, longest, mr, mi, theta
    real(c_double), value :: ep_real, ep_imag, es_real, es_imag
    real(c_double), intent(out) :: sections(3), variation
    integer(c_int), intent(out) :: order, quadrature
    complex(c_double_complex) :: ep, es
    real(c_double) :: norm, x, base(3), higher(3), finer(3), combined(3)
    integer :: n, q, dq, step, stable_steps, required_steps
    integer, parameter :: max_order = 48, max_quadrature = 400
    real(c_double), parameter :: tolerance = 1.0e-3_c_double

    sections = 0
    order = 0
    quadrature = 0
    variation = 0
    status = 1
    if (shape /= -1 .and. shape /= -2) return
    if (.not. all(ieee_is_finite([radius,ratio,longest,mr,mi,theta,ep_real,ep_imag,es_real,es_imag]))) return
    if (min(radius,ratio,longest) <= 0 .or. mr <= 0 .or. mi < 0 .or. theta < 0 .or. theta > 180) return
    ep = cmplx(ep_real, ep_imag, kind=c_double)
    es = cmplx(es_real, es_imag, kind=c_double)
    norm = abs(ep)**2 + abs(es)**2
    if (norm <= 0 .or. .not. ieee_is_finite(norm)) return
    ep = ep / sqrt(norm)
    es = es / sqrt(norm)
    if (mr == 1 .and. mi == 0) then
      status = 0
      return
    end if

    ! Work in units of the wavelength. Bound the legacy Bessel recurrences.
    x = 2*pi*longest
    if (x < 1.0e-5_c_double .or. x > max_order .or. x*hypot(mr,mi) > 850) return
    n = max(6, ceiling(x + 4*x**(1.0_c_double/3)) + 2)
    q = max(32, 3*n)
    step = 2
    required_steps = 1
    if (shape == -2) then
      ! The sharp rim can give a false plateau between nearby orders.
      step = 4
      required_steps = 2
    end if
    stable_steps = 0
    status = 2
    do while (n+step <= max_order .and. q+max(8,q/4) <= max_quadrature)
      dq = max(8,q/4)
      if (.not. cross_sections(n,q,base)) return
      if (.not. cross_sections(n+step,q,higher)) return
      if (.not. cross_sections(n,q+dq,finer)) return
      if (.not. cross_sections(n+step,q+dq,combined)) return
      variation = max(relative_change(base,higher), relative_change(base,finer), &
                      relative_change(combined,higher), relative_change(combined,finer))
      if (variation <= tolerance .and. ((mi > 0 .and. combined(3) >= 0) .or. &
          (mi == 0 .and. abs(combined(3)) <= tolerance*combined(1)))) then
        stable_steps = stable_steps + 1
      else
        stable_steps = 0
      end if
      if (stable_steps >= required_steps) then
        sections = combined
        if (mi == 0) sections(3) = 0
        order = n+step
        quadrature = 2*(q+dq)
        status = 0
        return
      end if
      n = n+step
      q = q+dq
    end do
  contains
    real(c_double) function relative_change(a,b) result(err)
      real(c_double), intent(in) :: a(3), b(3)
      err = maxval(abs(a(1:2)-b(1:2))/max(abs(a(1:2)),abs(b(1:2))))
    end function relative_change

    logical function cross_sections(n,q,cs) result(ok)
      integer, intent(in) :: n,q
      real(c_double), intent(out) :: cs(3)
      real(c_double) :: mu(n+2), w(n+2), out_theta, out_phi, azimuth_weight
      complex(c_double_complex) :: s11,s12,s21,s22
      integer :: core_status, i, j, nphi
      real(c_double), parameter :: lambda = 1.0_c_double, zero = 0.0_c_double

      ok = .false.
      call build_mish(shape,radius,ratio,lambda,mr,mi,n,q,core_status)
      if (core_status /= 0) return
      call ampl(n,lambda,theta,theta,zero,zero,zero,zero,s11,s12,s21,s22)
      cs(1) = 2*aimag(conjg(ep)*(s11*ep+s12*es) + conjg(es)*(s21*ep+s22*es))
      if (.not. ieee_is_finite(cs(1)) .or. cs(1) <= 0) return

      ! The azimuthal intensity contains Fourier orders up to 2*n.
      ! Gauss-Legendre integrates over cos(theta_s), with no sin(theta_s) factor.
      call gauss(n+2,0,0,mu,w)
      nphi = 4*n+4
      azimuth_weight = 2*pi/nphi
      cs(2) = 0
      do i=1,n+2
        out_theta = acos(mu(i))*180/pi
        do j=1,nphi
          out_phi = 360.0_c_double*(j-0.5_c_double)/nphi
          call ampl(n,lambda,theta,out_theta,zero,out_phi,zero,zero,s11,s12,s21,s22)
          cs(2) = cs(2) + w(i)*azimuth_weight*(abs(s11*ep+s12*es)**2 + abs(s21*ep+s22*es)**2)
        end do
      end do
      cs(3) = cs(1)-cs(2)
      ok = all(ieee_is_finite(cs)) .and. cs(2) > 0
    end function cross_sections
  end function ifdda_mishchenko_v1
end module mishchenko_reference
