
!------------------------------------------------------------------------
!> function atan3
!! computes atan2 with the required value at infinity
!------------------------------------------------------------------------
REAL*8 FUNCTION atan3(y,x)
  IMPLICIT NONE
  REAL*8, INTENT(IN) :: y,x

  REAL*8, PARAMETER :: PI = 3.141592653589793115997963468544185161_8

  IF (0 .EQ. x) THEN
     atan3=SIGN(PI/2,y)
  ELSE
     atan3=ATAN(y/x)
  END IF

END FUNCTION atan3

