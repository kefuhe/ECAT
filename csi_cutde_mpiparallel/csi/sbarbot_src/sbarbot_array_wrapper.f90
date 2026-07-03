!-----------------------------------------------------------------------
! Array wrapper for computeDisplacementVerticalStrainVolume
! Loops over N observation points to avoid Python-level per-point overhead.
!
! Compile together with xlogy.f90, atan3.f90 and
! computeDisplacementVerticalStrainVolume.f90 into a shared library.
!-----------------------------------------------------------------------
SUBROUTINE computeDispVerticalStrainVolumeArray( &
    N, x1arr, x2arr, x3arr, q1, q2, q3, L, T, W, theta, &
    eps11p, eps12p, eps13p, eps22p, eps23p, eps33p, G, nu, &
    u1arr, u2arr, u3arr)

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: N
    REAL*8, INTENT(IN) :: x1arr(N), x2arr(N), x3arr(N)
    REAL*8, INTENT(IN) :: q1, q2, q3, L, T, W, theta
    REAL*8, INTENT(IN) :: eps11p, eps12p, eps13p, eps22p, eps23p, eps33p
    REAL*8, INTENT(IN) :: G, nu
    REAL*8, INTENT(OUT) :: u1arr(N), u2arr(N), u3arr(N)

    INTEGER :: i

    DO i = 1, N
        CALL computeDisplacementVerticalStrainVolume( &
            x1arr(i), x2arr(i), x3arr(i), q1, q2, q3, L, T, W, theta, &
            eps11p, eps12p, eps13p, eps22p, eps23p, eps33p, G, nu, &
            u1arr(i), u2arr(i), u3arr(i))
    END DO

END SUBROUTINE computeDispVerticalStrainVolumeArray
