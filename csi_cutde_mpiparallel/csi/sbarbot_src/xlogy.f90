
!------------------------------------------------------------------------
!> function xLogY
!! computes x*log(y) and enforces 0*log(0)=0 to avoid NaN
!------------------------------------------------------------------------
REAL*8 FUNCTION xLogy(x,y)
  IMPLICIT NONE
  REAL*8, INTENT(IN) :: x,y

  IF (0 .EQ. x) THEN
     xLogy=0._8
  ELSE
     xLogy=x*log(y)
  END IF

END FUNCTION xLogy

