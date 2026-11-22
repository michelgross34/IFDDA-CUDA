SUBROUTINE deriveFF(FF, FFder, nx, dx)
    use precision
    implicit none

    INTEGER, INTENT(IN) :: nx
    REAL(wp), INTENT(IN) :: dx
    COMPLEX(wp), INTENT(IN), DIMENSION(nx) :: FF
    COMPLEX(wp), INTENT(OUT), DIMENSION(nx) :: FFder

    INTEGER :: i, nordre, nderiv
    REAL(wp) :: pos(0:4), xderiv
    COMPLEX(wp) :: f(0:4), deriv

    nordre = 5
    nderiv = 1
    pos(0) = zero
    pos(1) = dx
    pos(2) = two * dx
    pos(3) = 3.0_wp * dx
    pos(4) = 4.0_wp * dx

    ! Calcul des dérivées aux bords
    f(0) = FF(1)
    f(1) = FF(2)
    f(2) = FF(3)
    f(3) = FF(4)
    f(4) = FF(5)

    ! Dérivée au point 1
    xderiv = pos(0)
    CALL derivative(nordre, nderiv, f, pos, dx, xderiv, deriv)
    FFder(1) = deriv

    ! Dérivée au point 2
    xderiv = pos(1)
    CALL derivative(nordre, nderiv, f, pos, dx, xderiv, deriv)
    FFder(2) = deriv

    ! Calcul des dérivées aux autres bords
    f(0) = FF(nx-4)
    f(1) = FF(nx-3)
    f(2) = FF(nx-2)
    f(3) = FF(nx-1)
    f(4) = FF(nx)

    ! Dérivée au point nx
    xderiv = pos(4)
    CALL derivative(nordre, nderiv, f, pos, dx, xderiv, deriv)
    FFder(nx) = deriv

    ! Dérivée au point nx-1
    xderiv = pos(3)
    CALL derivative(nordre, nderiv, f, pos, dx, xderiv, deriv)
    FFder(nx-1) = deriv

    ! Calcul des dérivées aux points intérieurs
    DO i = 3, nx-2
        f(0) = FF(i-2)
        f(1) = FF(i-1)
        f(2) = FF(i)
        f(3) = FF(i+1)
        f(4) = FF(i+2)
        xderiv = pos(2)
        CALL derivative(nordre, nderiv, f, pos, dx, xderiv, deriv)
        FFder(i) = deriv
    END DO
END SUBROUTINE deriveFF
