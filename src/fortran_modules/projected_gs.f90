!program debug_pgs
!    implicit none
!    integer, parameter :: n = 4
!real (kind=8),dimension(n,n) :: M
!real (kind=8),dimension(n) :: q
!real (kind=8),dimension(n) :: x, init_guess
!data M / 1, 0, 0, -1, 0, 3, -1, 1, 2, 3, 2, 1, -1, 0, 0, 4 /
!q =(/ 1, 0, 2, -1/)
!init_guess =(/ 1, 1, 1, 1 /)
!x=0
!write(*,*) matmul(M,q)
!call pgs(M,q,init_guess,x,n)
!end

subroutine pgs(M,q,init_guess,x,n)
! Solve a Linear Complementary Problem using Projected Gauss Seidel
! This module uses one-based indexing
! Precision is reduced to REAL(4) because our epsilon is quite large
implicit none
integer n

real ,dimension(n,n) :: M
real ,dimension(n) :: q,w
real ,dimension(n) :: x, init_guess
real ,parameter  :: eps = 0.001
integer it,i
integer , parameter :: max_it = 10
real  :: r
!f2py intent(in) M,q,init_guess
!f2py intent(out) x
!f2py depend(n) x

! Initialization
x = init_guess
w = matmul(M,x)+q
it = 0

! Propagation
do while ((any(w < -eps) .or. abs(dot_product(w,x))> eps) .and. (it < max_it))
    it = it + 1
    do i=1,n
        r = -q(i)-dot_product(M(i,:),x) + M(i,i)*x(i)
        x(i) = max(0.,r/M(i,i))
    end do
    w = matmul(M,x)+q
end do
return
end
