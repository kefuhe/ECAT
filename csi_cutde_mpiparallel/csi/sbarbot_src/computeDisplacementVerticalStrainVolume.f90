!-----------------------------------------------------------------------
! Copyright (c) 2017 Sylvain Barbot
!
! This code and related code should be cited as:
!   Barbot S., J. D. P. Moore and V. Lambert, Displacement and Stress
!   Associated with Distributed Anelastic Deformation in a Half Space,
!   Bull. Seism. Soc. Am., 107(2), 10.1785/0120160237, 2017.
!
! This code is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This code is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License.
! If not, see <http://www.gnu.org/licenses/>.
!-----------------------------------------------------------------------

#define ACOSH(X) REAL(LOG(X+ZSQRT(CMPLX(X-1._8,0._8)*(X+1._8))))
#define ACOTH(X) 0.5_8*REAL(LOG(CMPLX(X+1,0._8))-LOG(CMPLX(X-1,0._8)))

!------------------------------------------------------------------------
!> subroutine ComputeDisplacementVerticalStrainVolume computes the
!! displacement field associated with deforming vertical strain volume
!! using the analytic solution of
!!
!!   Barbot S., J. D. P. Moore and V. Lambert, Displacement and Stress
!!   Associated with Distributed Anelastic Deformation in a Half Space,
!!   Bull. Seism. Soc. Am., 107(2), 10.1785/0120160237, 2017.
!!
!! considering the following geometry:
!!
!!
!!                      N (x1p)
!!                     /
!!                    /| strike (theta)          E (x2p)
!!        q1,q2,q3 ->@--------------------------+
!!                   |                        w |     +
!!                   |                        i |    /
!!                   |                        d |   / s
!!                   |                        t |  / s
!!                   |                        h | / e
!!                   |                          |/ n
!!                   +--------------------------+  k
!!                   :       l e n g t h       /  c
!!                   |                        /  i
!!                   :                       /  h
!!                   |                      /  t
!!                   :                     /
!!                   |                    +
!!                   Z (x3)
!!
!!
!! INPUT:
!! @param x1, x2, x3         northing, easting, and depth of the observation point
!!                           in unprimed system of coordinates.
!! @param q1, q2, q3         north, east and depth coordinates of the strain volume,
!! @param L, T, W            length, thickness, and width of the strain volume,
!! @param theta (radians)    strike of the strain volume,
!! @param epsijp             anelastic strain component 11, 12, 13, 22, 23 and 33 
!!                           in the strain volume in the system of reference tied to 
!!                           the strain volume (primed reference system),
!! @param G, nu              shear modulus and Poisson's ratio in the half space.
!!
!! OUTPUT:
!! ui                        displacement components in the unprimed reference system.
!!
!! \author Sylvain Barbot (21/02/17) - original fortran form
!------------------------------------------------------------------------
SUBROUTINE computeDisplacementVerticalStrainVolume( &
                        x1,x2,x3,q1,q2,q3,L,T,W,theta, &
                        eps11p,eps12p,eps13p,eps22p,eps23p,eps33p,G,nu, &
                        u1,u2,u3)

  IMPLICIT NONE

  REAL*8, INTENT(IN) :: x1,x2,x3,q1,q2,q3,L,T,W,theta
  REAL*8, INTENT(IN) :: eps11p,eps12p,eps13p,eps22p,eps23p,eps33p
  REAL*8, INTENT(IN) :: G,nu
  REAL*8, INTENT(OUT) :: u1,u2,u3
  
  REAL*8 :: lambda
  REAL*8 :: t1
  REAL*8 :: epskk
  REAL*8 :: eps11,eps12,eps13,eps22,eps23,eps33
  REAL*8 :: x1p,x2p
  REAL*8, EXTERNAL :: xlogy,atan3

  REAL*8, PARAMETER :: PI = 3.141592653589793115997963468544185161_8

  ! check valid parameters
  IF ((-1._8 .GT. nu) .OR. (0.5_8 .LT. nu)) THEN
     WRITE (0,'("error: -1<=nu<=0.5, nu=",ES9.2E2," given.")') nu
     STOP 1
  END IF

  IF (0 .GT. x3) THEN
     WRITE (0,'("error: observation depth (x3) must be positive")')
     STOP 1
  END IF

  IF (0 .GT. q3) THEN
     WRITE (0,'("error: source depth (q3) must be positive")')
     STOP 1
  END IF

  ! lame elastic parameters
  lambda=G*2._8*nu/(1-2*nu)

  ! isotropic strain
  epskk=eps11p+eps22p+eps33p

  ! rotate observation points to the strain-volume-centric system of coordinates
  x1p= (x1-q1)*DCOS(theta)+(x2-q2)*DSIN(theta)
  x2p=-(x1-q1)*DSIN(theta)+(x2-q2)*DCOS(theta)

  u1= IU1(L,   T/2,q3+W)-IU1(L,   -T/2,q3+W)+IU1(L,   -T/2,q3)-IU1(L,   T/2,q3) &
     -IU1(0._8,T/2,q3+W)+IU1(0._8,-T/2,q3+W)-IU1(0._8,-T/2,q3)+IU1(0._8,T/2,q3)
  u2= IU2(L,   T/2,q3+W)-IU2(L,   -T/2,q3+W)+IU2(L,   -T/2,q3)-IU2(L,   T/2,q3) &
     -IU2(0._8,T/2,q3+W)+IU2(0._8,-T/2,q3+W)-IU2(0._8,-T/2,q3)+IU2(0._8,T/2,q3)
  u3= IU3(L,   T/2,q3+W)-IU3(L,   -T/2,q3+W)+IU3(L,   -T/2,q3)-IU3(L,   T/2,q3) &
     -IU3(0._8,T/2,q3+W)+IU3(0._8,-T/2,q3+W)-IU3(0._8,-T/2,q3)+IU3(0._8,T/2,q3)

  ! rotate displacement field to reference system of coordinates
  t1=u1*DCOS(theta)-u2*DSIN(theta)
  u2=u1*DSIN(theta)+u2*DCOS(theta)
  u1=t1

CONTAINS

  !------------------------------------------------------------------------
  !> function r1
  !! computes the distance from the source at y1,y2,y3
  !------------------------------------------------------------------------
  REAL*8 FUNCTION r1(y1,y2,y3) 
    REAL*8, INTENT(IN) :: y1,y2,y3
  
    r1=sqrt((x1p-y1)**2+(x2p-y2)**2+(x3-y3)**2)
 
  END FUNCTION r1

  !------------------------------------------------------------------------
  !> function r2
  !! computes the distance from the image at y1,y2,-y3
  !------------------------------------------------------------------------
  REAL*8 FUNCTION r2(y1,y2,y3) 
    REAL*8, INTENT(IN) :: y1,y2,y3

    r2=sqrt((x1p-y1)**2+(x2p-y2)**2+(x3+y3)**2)

  END FUNCTION r2

  !---------------------------------------------------------------
  !> function IU1
  !! computes the indefinite integral U1 
  !---------------------------------------------------------------
  REAL*8 FUNCTION IU1(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    IU1=(lambda*epskk+2*G*eps11p)*J1123(y1,y2,y3) &
                      +2*G*eps12p*(J1223(y1,y2,y3)+J1113(y1,y2,y3)) &
                      +2*G*eps13p*(J1323(y1,y2,y3)+J1112(y1,y2,y3)) &
       +(lambda*epskk+2*G*eps22p)*J1213(y1,y2,y3) &
                      +2*G*eps23p*(J1212(y1,y2,y3)+J1313(y1,y2,y3)) &
       +(lambda*epskk+2*G*eps33p)*J1312(y1,y2,y3)

  END FUNCTION IU1

  !---------------------------------------------------------------
  !> function IU2
  !! computes the indefinite integral U2
  !---------------------------------------------------------------
  REAL*8 FUNCTION IU2(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    IU2=(lambda*epskk+2*G*eps11p)*J2123(y1,y2,y3) &
                      +2*G*eps12p*(J2223(y1,y2,y3)+J2113(y1,y2,y3)) &
                      +2*G*eps13p*(J2323(y1,y2,y3)+J2112(y1,y2,y3)) &
       +(lambda*epskk+2*G*eps22p)*J2213(y1,y2,y3) &
                      +2*G*eps23p*(J2212(y1,y2,y3)+J2313(y1,y2,y3)) &
       +(lambda*epskk+2*G*eps33p)*J2312(y1,y2,y3)

  END FUNCTION IU2

  !---------------------------------------------------------------
  !> function IU3
  !! computes the indefinite integral U3
  !---------------------------------------------------------------
  REAL*8 FUNCTION IU3(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    IU3=(lambda*epskk+2*G*eps11p)*J3123(y1,y2,y3) &
                      +2*G*eps12p*(J3223(y1,y2,y3)+J3113(y1,y2,y3)) &
                      +2*G*eps13p*(J3323(y1,y2,y3)+J3112(y1,y2,y3)) &
       +(lambda*epskk+2*G*eps22p)*J3213(y1,y2,y3) &
                      +2*G*eps23p*(J3212(y1,y2,y3)+J3313(y1,y2,y3)) &
       +(lambda*epskk+2*G*eps33p)*J3312(y1,y2,y3)

  END FUNCTION IU3

  !---------------------------------------------------------------
  !> function J1112
  !! computes the J integral J1112
  !---------------------------------------------------------------
  REAL*8 FUNCTION J1112(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J1112=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*x3*( &
          x1p-y1)*(x2p-y2)*y3*((x1p-y1)**2+(x3+y3)**2)**( &
          -1)-4*(-1._8+nu)*((-1)+2*nu)*(x3+y3)*atan3((x2p-y2),(x1p-y1)) &
          -x3*atan2(x3,x1p-y1)-3*x3* &
          atan2(3*x3,x1p-y1)+4*nu*x3*atan2(-nu*x3,x1p- &
          y1)+4*(-1._8+nu)*((-1)+2*nu)*(x3+y3)*atan2(lr2*(-x1p+y1),( &
          x2p-y2)*(x3+y3))-4*(-1._8+nu)*(x3-y3)*atan2(lr1*( &
          x3-y3),(x1p-y1)*(x2p-y2))+3*y3*atan2((-3)*y3, &
          x1p-y1)-y3*atan2(y3,x1p-y1)-4*nu*y3*atan2( &
          nu*y3,x1p-y1)-4*(-1._8+nu)*(x3+y3)*atan2(lr2*(x3+y3),( &
          x1p-y1)*(x2p-y2))+xLogy(-((-3)+4*nu)*(x1p- &
          y1),lr1+x2p-y2)+xLogy((5+4*nu*((-3)+2*nu))*(x1p-y1), &
          lr2+x2p-y2)+xLogy((-4)*(-1._8+nu)*(x2p-y2),lr1+x1p- &
          y1)+xLogy((-4)*(-1._8+nu)*(x2p-y2),lr2+x1p-y1))

  END FUNCTION J1112

  !---------------------------------------------------------------
  !> function J1113
  !! computes the J integral J1113
  !---------------------------------------------------------------
  REAL*8 FUNCTION J1113(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J1113=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*(x1p+( &
          -1)*y1)*((x1p-y1)**2+(x2p-y2)**2)**(-1)*(-((-1)+ &
          nu)*((-1)+2*nu)*lr2**2*(x3+y3)+(-1._8+nu)*((-1)+2*nu)*lr2* &
          y3*(2*x3+y3)+x3*((x1p-y1)**2+(x2p-y2)**2+x3*(x3+y3)) &
          )+x2p*atan2(-x2p,x1p-y1)-3*x2p*atan2(3*x2p,x1p- &
          y1)+4*nu*x2p*atan2(-nu*x2p,x1p-y1)-4*(-1._8+nu)*( &
          x2p-y2)*atan2(lr1*(x2p-y2),(x1p-y1)*(x3-y3) &
          )+4*(-1._8+nu)*(x2p-y2)*atan2(lr2*(x2p-y2),(x1p- &
          y1)*(x3+y3))+3*y2*atan2((-3)*y2,x1p-y1)-y2*atan2( &
          y2,x1p-y1)-4*nu*y2*atan2(nu*y2,x1p-y1)+xLogy((-1) &
          *((-3)+4*nu)*(x1p-y1),lr1+x3-y3)+xLogy(-(3 &
          -6*nu+4*nu**2)*(x1p-y1),lr2+x3+y3)+xLogy((-4)*(-1._8+nu)*( &
          x3-y3),lr1+x1p-y1)+xLogy(4*(-1._8+nu)*(x3+y3),lr2+x1p+( &
          -1)*y1))

  END FUNCTION J1113

  !---------------------------------------------------------------
  !> function J1123
  !! computes the J integral J1123
  !---------------------------------------------------------------
  REAL*8 FUNCTION J1123(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J1123=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*((-2)*lr2**(-1)*(( &
          x1p-y1)**2+(x2p-y2)**2)**(-1)*(x2p-y2)*((x1p+(-1) &
          *y1)**2+(x3+y3)**2)**(-1)*(x3*((x3**2+(x1p-y1)**2)*( &
          x3**2+(x1p-y1)**2+(x2p-y2)**2)+x3*(3*x3**2+2*(x1p+(-1) &
          *y1)**2+(x2p-y2)**2)*y3+3*x3**2*y3**2+x3*y3**3)-(( &
          -1)+nu)*((-1)+2*nu)*lr2**2*(x3+y3)*((x1p-y1)**2+(x3+y3) &
          **2)+(-1._8+nu)*((-1)+2*nu)*lr2*y3*(2*x3+y3)*((x1p-y1) &
          **2+(x3+y3)**2))+2*(-1._8+nu)*((-1)+2*nu)*(x1p-y1)*atan3((x1p-y1),(x2p-y2)) &
          +x1p*atan2(-x1p,x2p-y2) &
          -3*x1p*atan2(3*x1p,x2p-y2)+4*nu*x1p*atan2(-nu*x1p, &
          x2p-y2)+3*y1*atan2((-3)*y1,x2p-y2)-y1*atan2( &
          y1,x2p-y2)-4*nu*y1*atan2(nu*y1,x2p-y2)+2*((-1)+ &
          2*nu)*(x1p-y1)*atan2(lr1*(-x1p+y1),(x2p-y2)*(x3+ &
          (-1)*y3))+2*(1-2*nu)**2*(x1p-y1)*atan2(lr2*(-x1p+ &
          y1),(x2p-y2)*(x3+y3))+xLogy((-2)*x3,lr2-x2p+y2)+xLogy(( &
          -1)*((-3)+4*nu)*(x2p-y2),lr1+x3-y3)+xLogy(-(3+( &
          -6)*nu+4*nu**2)*(x2p-y2),lr2+x3+y3)+xLogy(-((-3)+4* &
          nu)*(x3-y3),lr1+x2p-y2)+xLogy(-(5+4*nu*((-3)+2* &
          nu))*(x3+y3),lr2+x2p-y2))

  END FUNCTION J1123

  !---------------------------------------------------------------
  !> function J2112
  !! computes the J integral J2112
  !---------------------------------------------------------------
  REAL*8 FUNCTION J2112(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J2112=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(-lr1+(1+8*(( &
          -1)+nu)*nu)*lr2-2*lr2**(-1)*x3*y3+xLogy((-4)*(-1._8+nu)*(( &
          -1)+2*nu)*(x3+y3),lr2+x3+y3))

  END FUNCTION J2112

  !---------------------------------------------------------------
  !> function J2113
  !! computes the J integral J2113
  !---------------------------------------------------------------
  REAL*8 FUNCTION J2113(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J2113=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*((x1p+ &
          (-1)*y1)**2+(x2p-y2)**2)**(-1)*(x2p-y2)*(-((-1)+ &
          nu)*((-1)+2*nu)*lr2**2*(x3+y3)+(-1._8+nu)*((-1)+2*nu)*lr2* &
          y3*(2*x3+y3)+x3*((x1p-y1)**2+(x2p-y2)**2+x3*(x3+y3)) &
          )+xLogy(-((-1)-2*nu+4*nu**2)*(x2p-y2),lr2+x3+y3)+ &
          xLogy(-x2p+y2,lr1+x3-y3))

  END FUNCTION J2113

  !---------------------------------------------------------------
  !> function J2123
  !! computes the J integral J2123
  !---------------------------------------------------------------
  REAL*8 FUNCTION J2123(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J2123=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*(x1p+( &
          -1)*y1)*((x1p-y1)**2+(x2p-y2)**2)**(-1)*(-((-1)+ &
          nu)*((-1)+2*nu)*lr2**2*(x3+y3)+(-1._8+nu)*((-1)+2*nu)*lr2* &
          y3*(2*x3+y3)+x3*((x1p-y1)**2+(x2p-y2)**2+x3*(x3+y3)) &
          )+xLogy(-((-1)-2*nu+4*nu**2)*(x1p-y1),lr2+x3+y3)+ &
          xLogy(-x1p+y1,lr1+x3-y3))

  END FUNCTION J2123

  !---------------------------------------------------------------
  !> function J3112
  !! computes the J integral J3112
  !---------------------------------------------------------------
  REAL*8 FUNCTION J3112(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J3112=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*((-2)*lr2**(-1)* &
          x3*(x2p-y2)*y3*(x3+y3)*((x1p-y1)**2+(x3+y3)**2)**( &
          -1)+4*(-1._8+nu)*((-1)+2*nu)*(x1p-y1)*atan3((x1p-y1),(x2p-y2)) &
          +4*(-1._8+nu)*((-1)+2*nu)*(x1p-y1)* &
          atan2(lr2*(-x1p+y1),(x2p-y2)*(x3+y3))+xLogy((-4)*((-1)+ &
          nu)*((-1)+2*nu)*(x2p-y2),lr2+x3+y3)+xLogy(x3-y3,lr1+ &
          x2p-y2)+xLogy(-x3-7*y3-8*nu**2*(x3+y3)+8*nu*( &
          x3+2*y3),lr2+x2p-y2))

  END FUNCTION J3112

  !---------------------------------------------------------------
  !> function J3113
  !! computes the J integral J3113
  !---------------------------------------------------------------
  REAL*8 FUNCTION J3113(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J3113=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(lr1+((-1)-8*(( &
          -1)+nu)*nu)*lr2-2*lr2**(-1)*x3*y3+2*((-3)+4*nu)*x3* &
          ACOTH(lr2**(-1)*(x3+y3))+xLogy(2*(3*x3+2*y3-6*nu*(x3+y3)+ &
          4*nu**2*(x3+y3)),lr2+x3+y3))

  END FUNCTION J3113

  !---------------------------------------------------------------
  !> function J3123
  !! computes the J integral J3123
  !---------------------------------------------------------------
  REAL*8 FUNCTION J3123(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J3123=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*x3*( &
          x1p-y1)*(x2p-y2)*y3*((x1p-y1)**2+(x3+y3)**2)**( &
          -1)+4*(-1._8+nu)*((-1)+2*nu)*(x3+y3)*atan3((x2p-y2),(x1p-y1)) &
          +4*((-1)+2*nu)*(nu*x3+(-1._8+nu)*y3)*atan2( &
          lr2*(x1p-y1),(x2p-y2)*(x3+y3))+xLogy(x1p-y1,lr1+x2p+ &
          (-1)*y2)+xLogy(-(1+8*(-1._8+nu)*nu)*(x1p-y1),lr2+x2p+( &
          -1)*y2))

  END FUNCTION J3123

  !---------------------------------------------------------------
  !> function J1212
  !! computes the J integral J1212
  !---------------------------------------------------------------
  REAL*8 FUNCTION J1212(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J1212=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(-lr1+(1+8*(( &
          -1)+nu)*nu)*lr2-2*lr2**(-1)*x3*y3+xLogy((-4)*(-1._8+nu)*(( &
          -1)+2*nu)*(x3+y3),lr2+x3+y3))

  END FUNCTION J1212

  !---------------------------------------------------------------
  !> function J1213
  !! computes the J integral J1213
  !---------------------------------------------------------------
  REAL*8 FUNCTION J1213(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J1213=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*((x1p+ &
          (-1)*y1)**2+(x2p-y2)**2)**(-1)*(x2p-y2)*(-((-1)+ &
          nu)*((-1)+2*nu)*lr2**2*(x3+y3)+(-1._8+nu)*((-1)+2*nu)*lr2* &
          y3*(2*x3+y3)+x3*((x1p-y1)**2+(x2p-y2)**2+x3*(x3+y3)) &
          )+xLogy(-((-1)-2*nu+4*nu**2)*(x2p-y2),lr2+x3+y3)+ &
          xLogy(-x2p+y2,lr1+x3-y3))

  END FUNCTION J1213

  !---------------------------------------------------------------
  !> function J1223
  !! computes the J integral J1223
  !---------------------------------------------------------------
  REAL*8 FUNCTION J1223(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J1223=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*(x1p+( &
          -1)*y1)*((x1p-y1)**2+(x2p-y2)**2)**(-1)*(-((-1)+ &
          nu)*((-1)+2*nu)*lr2**2*(x3+y3)+(-1._8+nu)*((-1)+2*nu)*lr2* &
          y3*(2*x3+y3)+x3*((x1p-y1)**2+(x2p-y2)**2+x3*(x3+y3)) &
          )+xLogy(-((-1)-2*nu+4*nu**2)*(x1p-y1),lr2+x3+y3)+ &
          xLogy(-x1p+y1,lr1+x3-y3))

  END FUNCTION J1223

  !---------------------------------------------------------------
  !> function J2212
  !! computes the J integral J2212
  !---------------------------------------------------------------
  REAL*8 FUNCTION J2212(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J2212=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*x3*( &
          x1p-y1)*(x2p-y2)*y3*((x2p-y2)**2+(x3+y3)**2)**( &
          -1)-4*(-1._8+nu)*((-1)+2*nu)*(x3+y3)*atan3((x1p-y1),(x2p-y2)) &
          -x3*atan2(x3,x1p-y1)-3*x3* &
          atan2(3*x3,x1p-y1)+4*nu*x3*atan2(-nu*x3,x1p- &
          y1)+4*(-1._8+nu)*((-1)+2*nu)*(x3+y3)*atan2(lr2*(-x2p+y2),( &
          x1p-y1)*(x3+y3))-4*(-1._8+nu)*(x3-y3)*atan2(lr1*( &
          x3-y3),(x1p-y1)*(x2p-y2))+3*y3*atan2((-3)*y3, &
          x1p-y1)-y3*atan2(y3,x1p-y1)-4*nu*y3*atan2( &
          nu*y3,x1p-y1)-4*(-1._8+nu)*(x3+y3)*atan2(lr2*(x3+y3),( &
          x1p-y1)*(x2p-y2))+xLogy((-4)*(-1._8+nu)*(x1p-y1), &
          lr1+x2p-y2)+xLogy((-4)*(-1._8+nu)*(x1p-y1),lr2+x2p- &
          y2)+xLogy(-((-3)+4*nu)*(x2p-y2),lr1+x1p-y1)+xLogy( &
          (5+4*nu*((-3)+2*nu))*(x2p-y2),lr2+x1p-y1))

  END FUNCTION J2212

  !---------------------------------------------------------------
  !> function J2213
  !! computes the J integral J2213
  !---------------------------------------------------------------
  REAL*8 FUNCTION J2213(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J2213=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*((-2)*lr2**(-1)*( &
          x1p-y1)*((x1p-y1)**2+(x2p-y2)**2)**(-1)*((x2p+(-1) &
          *y2)**2+(x3+y3)**2)**(-1)*(x3*((x3**2+(x2p-y2)**2)*( &
          x3**2+(x1p-y1)**2+(x2p-y2)**2)+x3*(3*x3**2+(x1p- &
          y1)**2+2*(x2p-y2)**2)*y3+3*x3**2*y3**2+x3*y3**3)-( &
          (-1)+nu)*((-1)+2*nu)*lr2**2*(x3+y3)*((x2p-y2)**2+(x3+y3) &
          **2)+(-1._8+nu)*((-1)+2*nu)*lr2*y3*(2*x3+y3)*((x2p-y2) &
          **2+(x3+y3)**2))+2*(-1._8+nu)*((-1)+2*nu)*(x2p-y2)*atan3((x2p-y2),(x1p-y1)) &
          +x2p*atan2(-x2p,x1p-y1) &
          -3*x2p*atan2(3*x2p,x1p-y1)+4*nu*x2p*atan2(-nu*x2p, &
          x1p-y1)+3*y2*atan2((-3)*y2,x1p-y1)-y2*atan2( &
          y2,x1p-y1)-4*nu*y2*atan2(nu*y2,x1p-y1)+2*((-1)+ &
          2*nu)*(x2p-y2)*atan2(lr1*(-x2p+y2),(x1p-y1)*(x3+ &
          (-1)*y3))+2*(1-2*nu)**2*(x2p-y2)*atan2(lr2*(-x2p+ &
          y2),(x1p-y1)*(x3+y3))+xLogy((-2)*x3,lr2-x1p+y1)+xLogy(( &
          -1)*((-3)+4*nu)*(x1p-y1),lr1+x3-y3)+xLogy(-(3+( &
          -6)*nu+4*nu**2)*(x1p-y1),lr2+x3+y3)+xLogy(-((-3)+4* &
          nu)*(x3-y3),lr1+x1p-y1)+xLogy(-(5+4*nu*((-3)+2* &
          nu))*(x3+y3),lr2+x1p-y1))

  END FUNCTION J2213

  !---------------------------------------------------------------
  !> function J2223
  !! computes the J integral J2223
  !---------------------------------------------------------------
  REAL*8 FUNCTION J2223(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J2223=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*((x1p+ &
          (-1)*y1)**2+(x2p-y2)**2)**(-1)*(x2p-y2)*(-((-1)+ &
          nu)*((-1)+2*nu)*lr2**2*(x3+y3)+(-1._8+nu)*((-1)+2*nu)*lr2* &
          y3*(2*x3+y3)+x3*((x1p-y1)**2+(x2p-y2)**2+x3*(x3+y3)) &
          )+x1p*atan2(-x1p,x2p-y2)-3*x1p*atan2(3*x1p,x2p- &
          y2)+4*nu*x1p*atan2(-nu*x1p,x2p-y2)-4*(-1._8+nu)*( &
          x1p-y1)*atan2(lr1*(x1p-y1),(x2p-y2)*(x3-y3) &
          )+4*(-1._8+nu)*(x1p-y1)*atan2(lr2*(x1p-y1),(x2p- &
          y2)*(x3+y3))+3*y1*atan2((-3)*y1,x2p-y2)-y1*atan2( &
          y1,x2p-y2)-4*nu*y1*atan2(nu*y1,x2p-y2)+xLogy((-1) &
          *((-3)+4*nu)*(x2p-y2),lr1+x3-y3)+xLogy(-(3 &
          -6*nu+4*nu**2)*(x2p-y2),lr2+x3+y3)+xLogy((-4)*(-1._8+nu)*( &
          x3-y3),lr1+x2p-y2)+xLogy(4*(-1._8+nu)*(x3+y3),lr2+x2p+( &
          -1)*y2))

  END FUNCTION J2223

  !---------------------------------------------------------------
  !> function J3212
  !! computes the J integral J3212
  !---------------------------------------------------------------
  REAL*8 FUNCTION J3212(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J3212=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*((-2)*lr2**(-1)* &
          x3*(x1p-y1)*y3*(x3+y3)*((x2p-y2)**2+(x3+y3)**2)**( &
          -1)+4*(-1._8+nu)*((-1)+2*nu)*(x2p-y2)*atan3((x2p-y2),(x1p-y1)) &
          +4*(-1._8+nu)*((-1)+2*nu)*(x2p-y2)* &
          atan2(lr2*(-x2p+y2),(x1p-y1)*(x3+y3))+xLogy((-4)*((-1)+ &
          nu)*((-1)+2*nu)*(x1p-y1),lr2+x3+y3)+xLogy(x3-y3,lr1+ &
          x1p-y1)+xLogy(-x3-7*y3-8*nu**2*(x3+y3)+8*nu*( &
          x3+2*y3),lr2+x1p-y1))

  END FUNCTION J3212

  !---------------------------------------------------------------
  !> function J3213
  !! computes the J integral J3213
  !---------------------------------------------------------------
  REAL*8 FUNCTION J3213(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J3213=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*x3*( &
          x1p-y1)*(x2p-y2)*y3*((x2p-y2)**2+(x3+y3)**2)**( &
          -1)+4*(-1._8+nu)*((-1)+2*nu)*(x3+y3)*atan3((x1p-y1),(x2p-y2)) &
          +4*((-1)+2*nu)*(nu*x3+(-1._8+nu)*y3)*atan2( &
          lr2*(x2p-y2),(x1p-y1)*(x3+y3))+xLogy(x2p-y2,lr1+x1p+ &
          (-1)*y1)+xLogy(-(1+8*(-1._8+nu)*nu)*(x2p-y2),lr2+x1p+( &
          -1)*y1))

  END FUNCTION J3213

  !---------------------------------------------------------------
  !> function J3223
  !! computes the J integral J3223
  !---------------------------------------------------------------
  REAL*8 FUNCTION J3223(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J3223=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(lr1+((-1)-8*(( &
          -1)+nu)*nu)*lr2-2*lr2**(-1)*x3*y3+2*((-3)+4*nu)*x3* &
          ACOTH(lr2**(-1)*(x3+y3))+xLogy(2*(3*x3+2*y3-6*nu*(x3+y3)+ &
          4*nu**2*(x3+y3)),lr2+x3+y3))

  END FUNCTION J3223

  !---------------------------------------------------------------
  !> function J1312
  !! computes the J integral J1312
  !---------------------------------------------------------------
  REAL*8 FUNCTION J1312(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J1312=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*x3*( &
          x2p-y2)*y3*(x3+y3)*((x1p-y1)**2+(x3+y3)**2)**(-1)+( &
          -4)*(-1._8+nu)*((-1)+2*nu)*(x1p-y1)*atan3((x1p-y1),(x2p-y2)) &
          +4*(-1._8+nu)*((-1)+2*nu)*(x1p-y1)* &
          atan2(lr2*(x1p-y1),(x2p-y2)*(x3+y3))+xLogy(4*(-1._8+nu) &
          *((-1)+2*nu)*(x2p-y2),lr2+x3+y3)+xLogy(x3-y3,lr1+x2p+( &
          -1)*y2)+xLogy((7+8*((-2)+nu)*nu)*x3+y3+8*(-1._8+nu)*nu*y3, &
          lr2+x2p-y2))

  END FUNCTION J1312

  !---------------------------------------------------------------
  !> function J1313
  !! computes the J integral J1313
  !---------------------------------------------------------------
  REAL*8 FUNCTION J1313(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J1313=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(lr1+lr2**(-1)*((7+ &
          8*((-2)+nu)*nu)*lr2**2+2*x3*y3)+ &
          2*((-3)+4*nu)*x3*ACOTH(lr2**(-1)*(x3+y3))+ &
          xLogy(2*(-3*x3-2*y3+6*nu*(x3+y3)-4*nu**2*(x3+y3)),lr2+x3+y3))

  END FUNCTION J1313

  !---------------------------------------------------------------
  !> function J1323
  !! computes the J integral J1323
  !---------------------------------------------------------------
  REAL*8 FUNCTION J1323(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J1323=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*((-2)*lr2**(-1)* &
          x3*(x1p-y1)*(x2p-y2)*y3*((x1p-y1)**2+(x3+y3) &
          **2)**(-1)-4*(-1._8+nu)*((-1)+2*nu)*(x3+y3)*atan3((x2p-y2),(x1p-y1)) &
          -4*(-1._8+nu)*((-3)*x3-y3+2* &
          nu*(x3+y3))*atan2(lr2*(x1p-y1),(x2p-y2)*(x3+y3))+ &
          xLogy(x1p-y1,lr1+x2p-y2)+xLogy((7+8*((-2)+nu)*nu)*(x1p+ &
          (-1)*y1),lr2+x2p-y2))

  END FUNCTION J1323

  !---------------------------------------------------------------
  !> function J2312
  !! computes the J integral J2312
  !---------------------------------------------------------------
  REAL*8 FUNCTION J2312(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J2312=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*x3*( &
          x1p-y1)*y3*(x3+y3)*((x2p-y2)**2+(x3+y3)**2)**(-1)+( &
          -4)*(-1._8+nu)*((-1)+2*nu)*(x2p-y2)*atan3((x2p-y2),(x1p-y1)) &
          +4*(-1._8+nu)*((-1)+2*nu)*(x2p-y2)* &
          atan2(lr2*(x2p-y2),(x1p-y1)*(x3+y3))+xLogy(4*(-1._8+nu) &
          *((-1)+2*nu)*(x1p-y1),lr2+x3+y3)+xLogy(x3-y3,lr1+x1p+( &
          -1)*y1)+xLogy((7+8*((-2)+nu)*nu)*x3+y3+8*(-1._8+nu)*nu*y3, &
          lr2+x1p-y1))

  END FUNCTION J2312

  !---------------------------------------------------------------
  !> function J2313
  !! computes the J integral J2313
  !---------------------------------------------------------------
  REAL*8 FUNCTION J2313(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J2313=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*((-2)*lr2**(-1)* &
          x3*(x1p-y1)*(x2p-y2)*y3*((x2p-y2)**2+(x3+y3) &
          **2)**(-1)-4*(-1._8+nu)*((-1)+2*nu)*(x3+y3)*atan3((x1p-y1),(x2p-y2)) &
          -4*(-1._8+nu)*((-3)*x3-y3+2* &
          nu*(x3+y3))*atan2(lr2*(x2p-y2),(x1p-y1)*(x3+y3))+ &
          xLogy(x2p-y2,lr1+x1p-y1)+xLogy((7+8*((-2)+nu)*nu)*(x2p+ &
          (-1)*y2),lr2+x1p-y1))

  END FUNCTION J2313

  !---------------------------------------------------------------
  !> function J2323
  !! computes the J integral J2323
  !---------------------------------------------------------------
  REAL*8 FUNCTION J2323(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J2323=(-(1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(lr1+lr2**(-1)*((7+ &
          8*((-2)+nu)*nu)*lr2**2+2*x3*y3)+ &
          2*((-3)+4*nu)*x3*ACOTH(lr2**(-1)*(x3+y3))+ &
          xLogy(2*((-3)*x3-2*y3+6*nu*(x3+y3)-4*nu**2*(x3+y3)),lr2+x3+y3))

    END FUNCTION J2323

  !---------------------------------------------------------------
  !> function J3312
  !! computes the J integral J3312
  !---------------------------------------------------------------
  REAL*8 FUNCTION J3312(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J3312=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*x3*( &
          x1p-y1)*(x2p-y2)*y3*((x1p-y1)**2+(x3+y3)**2)**( &
          -1)*((x2p-y2)**2+(x3+y3)**2)**(-1)*((x1p-y1)**2+(x2p+( &
          -1)*y2)**2+2*(x3+y3)**2)-3*x3*atan2(3*x3,x1p-y1) &
          -5*x3*atan2(5*x3,x2p-y2)+12*nu*x3*atan2((-3)*nu*x3,x2p+( &
          -1)*y2)+4*nu*x3*atan2(-nu*x3,x1p-y1)-8*nu**2* &
          x3*atan2(nu**2*x3,x2p-y2)+3*y3*atan2((-3)*y3,x1p- &
          y1)-5*y3*atan2(5*y3,x2p-y2)+12*nu*y3*atan2((-3)* &
          nu*y3,x2p-y2)-4*nu*y3*atan2(nu*y3,x1p-y1)-8* &
          nu**2*y3*atan2(nu**2*y3,x2p-y2)+2*((-1)+2*nu)*(x3+(-1) &
          *y3)*atan2(lr1*(-x3+y3),(x1p-y1)*(x2p-y2))+2*( &
          1-2*nu)**2*(x3+y3)*atan2(lr2*(x3+y3),(x1p-y1)*(x2p+(-1) &
          *y2))+xLogy(-((-3)+4*nu)*(x1p-y1),lr1+x2p-y2)+ &
          xLogy((5+4*nu*((-3)+2*nu))*(x1p-y1),lr2+x2p-y2)+ &
          xLogy(-((-3)+4*nu)*(x2p-y2),lr1+x1p-y1)+xLogy((5+ &
          4*nu*((-3)+2*nu))*(x2p-y2),lr2+x1p-y1))

  END FUNCTION J3312

  !---------------------------------------------------------------
  !> function J3313
  !! computes the J integral J3313
  !---------------------------------------------------------------
  REAL*8 FUNCTION J3313(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J3313=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*x3*( &
          x1p-y1)*y3*(x3+y3)*((x2p-y2)**2+(x3+y3)**2)**(-1)+5* &
          x2p*atan2((-5)*x2p,x1p-y1)-3*x2p*atan2(3*x2p,x1p-y1) &
          +4*nu*x2p*atan2(-nu*x2p,x1p-y1)-12*nu*x2p*atan2( &
          3*nu*x2p,x1p-y1)+8*nu**2*x2p*atan2(-nu**2*x2p,x1p+(-1) &
          *y1)-4*(-1._8+nu)*(x2p-y2)*atan2(lr1*(x2p-y2),(x1p+ &
          (-1)*y1)*(x3-y3))-8*(-1._8+nu)**2*(x2p-y2)* &
          atan2(lr2*(x2p-y2),(x1p-y1)*(x3+y3))+3*y2*atan2((-3) &
          *y2,x1p-y1)-5*y2*atan2(5*y2,x1p-y1)+12*nu*y2* &
          atan2((-3)*nu*y2,x1p-y1)-4*nu*y2*atan2(nu*y2,x1p+(-1) &
          *y1)-8*nu**2*y2*atan2(nu**2*y2,x1p-y1)+xLogy((-4)* &
          x3,lr2-x1p+y1)+xLogy((-4)*(-1._8+nu)*(x1p-y1),lr1+x3+(-1) &
          *y3)+xLogy((-8)*(-1._8+nu)**2*(x1p-y1),lr2+x3+y3)+xLogy((-1) &
          *((-3)+4*nu)*(x3-y3),lr1+x1p-y1)+xLogy((-7)*x3 &
          -5*y3+12*nu*(x3+y3)-8*nu**2*(x3+y3),lr2+x1p-y1))
        
  END FUNCTION J3313

  !---------------------------------------------------------------
  !> function J3323
  !! computes the J integral J3323
  !---------------------------------------------------------------
  REAL*8 FUNCTION J3323(y1,y2,y3)
    REAL*8, INTENT(IN) :: y1,y2,y3

    REAL*8 :: lr1,lr2

    lr1=r1(y1,y2,y3)
    lr2=r2(y1,y2,y3)

    J3323=((1._8/16._8))*(1-nu)**(-1)*PI**(-1)*G**(-1)*(2*lr2**(-1)*x3*( &
          x2p-y2)*y3*(x3+y3)*((x1p-y1)**2+(x3+y3)**2)**(-1)+5* &
          x1p*atan2((-5)*x1p,x2p-y2)-3*x1p*atan2(3*x1p,x2p-y2) &
          +4*nu*x1p*atan2(-nu*x1p,x2p-y2)-12*nu*x1p*atan2( &
          3*nu*x1p,x2p-y2)+8*nu**2*x1p*atan2(-nu**2*x1p,x2p+(-1) &
          *y2)-4*(-1._8+nu)*(x1p-y1)*atan2(lr1*(x1p-y1),(x2p+ &
          (-1)*y2)*(x3-y3))-8*(-1._8+nu)**2*(x1p-y1)* &
          atan2(lr2*(x1p-y1),(x2p-y2)*(x3+y3))+3*y1*atan2((-3) &
          *y1,x2p-y2)-5*y1*atan2(5*y1,x2p-y2)+12*nu*y1* &
          atan2((-3)*nu*y1,x2p-y2)-4*nu*y1*atan2(nu*y1,x2p+(-1) &
          *y2)-8*nu**2*y1*atan2(nu**2*y1,x2p-y2)+xLogy((-4)* &
          x3,lr2-x2p+y2)+xLogy((-4)*(-1._8+nu)*(x2p-y2),lr1+x3+(-1) &
          *y3)+xLogy((-8)*(-1._8+nu)**2*(x2p-y2),lr2+x3+y3)+xLogy((-1) &
          *((-3)+4*nu)*(x3-y3),lr1+x2p-y2)+xLogy((-7)*x3 &
          -5*y3+12*nu*(x3+y3)-8*nu**2*(x3+y3),lr2+x2p-y2))

  END FUNCTION J3323

END SUBROUTINE computeDisplacementVerticalStrainVolume

