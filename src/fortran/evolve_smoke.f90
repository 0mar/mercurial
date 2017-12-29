SUBROUTINE get_sparse_matrix(diff,velo_x,velo_y,nx,ny,dx,dy,dt,obstacles, a_val, a_row,a_col,nnz)
  ! Create a sparse matrix for a central difference in space, implicit euler in time discretization scheme 
  ! for the advection diffusion equation with constant coefficients
  IMPLICIT NONE
  ! Includes one layer for the boundary
  INTEGER nx,ny,max_nnz
  REAL (kind=8) :: diff,velo_x, velo_y
  REAL (kind=8) dx,dy,dt
  !REAL (kind=8), dimension(0:nx*ny-1,0:nx*ny-1) :: a_f
  INTEGER :: i,j,el_counter,el_index, self_el,nnz
  REAL (kind=8) :: l_up,l_down,l_left,l_right,l_self
  INTEGER, DIMENSION(0:nx-1,0:ny-1):: obstacles
  ! Number of nonzero elements: 5 for every cell, 1 for every virtual boundary
  ! Extension: Compute the nnz before this method
  ! The sparse matrices now are adapted for in the Jacobi method
  REAL (kind=8), DIMENSION(0:5*(nx-2)*(ny-2) + 2*(nx + ny)-4-1) :: a_val
  INTEGER , DIMENSION(0:5*(nx-2)*(ny-2) + 2*(nx + ny)-4-1) :: a_col,a_row
!f2py intent(in) diff,velo_x,velo_y,dx,dy,dt,obstacles
!f2py intent(out) a_val,a_row,a_col,nnz
!f2py depend(nx,ny) a_val,a_row,a_col,nnz


  max_nnz = 5*(nx-2)*(ny-2) + 2*(nx + ny)-4

  ! Coefficients
  l_up = -diff*dt/(dy*dy) + velo_y*dt/(2*dy)
  l_down = -diff*dt/(dy*dy) - velo_y*dt/(2*dy)
  l_left = -diff*dt/(dx*dx) - velo_x*dt/(2*dx)
  l_right = -diff*dt/(dx*dx) + velo_x*dt/(2*dx)
  l_self = 2 * diff*dt*(1/(dx*dx) + 1/(dy*dy)) + 1

  ! Start filling matrix
  el_counter = 0
  a_val = 0
  DO j = 0, ny - 1
    DO i = 0, nx - 1
      el_index = i + j * nx
      IF (i==0 .or. i==nx-1 .or. j==0 .or. j==ny-1 .or. obstacles(i,j)==1) THEN
        a_val(el_counter) = 1.d0
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index
        el_counter = el_counter + 1
      ELSE
        ! Self relation
        self_el = el_counter
        a_val(el_counter) = l_self
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index
        el_counter = el_counter + 1
        ! Down relation
        IF (obstacles(i,j-1)==0) THEN
          a_val(el_counter) = l_down
        ELSE
          a_val(self_el) = a_val(self_el) + l_down
        ENDIF
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index - nx
        el_counter = el_counter + 1
        ! Left relation
        IF (obstacles(i-1,j)==0) THEN
          a_val(el_counter) = l_left
        ELSE
          a_val(self_el) = a_val(self_el) + l_left
        ENDIF
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index - 1
        el_counter = el_counter + 1
        ! Right relation
        IF (obstacles(i+1,j)==0) THEN
          a_val(el_counter) = l_right
        ELSE
          a_val(self_el) = a_val(self_el) + l_right
        ENDIF
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index + 1
        el_counter = el_counter + 1
        ! Up relation
        IF (obstacles(i,j+1)==0) THEN
          a_val(el_counter) = l_up
        ELSE
          a_val(self_el) = a_val(self_el) + l_up
        ENDIF
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index + nx
        el_counter = el_counter + 1
      ENDIF
    ENDDO
  ENDDO
  nnz = el_counter

!   a_f = 0
!   do i=0,nnz-1
!       a_f(a_row(i),a_col(i)) = a_val(i)
!   enddo
!   do i=0,nx*ny-1
!       write(*, '(*(F7.3))')( a_f(i,j) ,j=0,nx*ny-1)
!   enddo
! CALL compress_row(nx*ny,nnz,a_row,a_crow) no need, lets compress D^-1 & R
END SUBROUTINE

SUBROUTINE iterate_jacobi(a_val,a_row,a_col,nnz,b,guess,obstacles,nx,ny,x)
  ! Perform jacobi iterations to solve the sparce system Ax=b
  ! with relevant elements coded in rel_els
  IMPLICIT NONE
  REAL (kind=8), DIMENSION(0:5*(nx-2)*(ny-2) + 2*(nx + ny)-4-1) :: a_val
  REAL (kind=8), DIMENSION(0:nnz-1) :: D_inv_val, R_val
  INTEGER , DIMENSION(0:5*(nx-2)*(ny-2) + 2*(nx + ny)-4-1) ::a_row, a_col
  INTEGER , DIMENSION(0:nnz-1) :: d_inv_row, d_inv_col,r_row, r_col
  INTEGER , DIMENSION(0:nx*ny-1) :: d_inv_crow,r_crow
  INTEGER, DIMENSION(0:nx-1,0:ny-1) :: obstacles
  INTEGER, DIMENSION(0:nx*ny-1) :: rel_els
  REAL (kind=8), DIMENSION(0:nx*ny-1) :: b,x,x_old,tmp,guess
  INTEGER nx,ny,nnz, iter, max_iter
  INTEGER  el_counter_D, el_counter_R, el_counter
  REAL (kind=8) ::  tol, error
!f2py intent(in) a_val,a_row,a_col,nnz,b,obstacles,init_guess
!f2py intent(out) x
!f2py depend(nx,ny) x

  tol = 0.00002
  max_iter = 10000

  ! Create the D^-1 and R matrices
  el_counter_D = 0
  el_counter_R = 0
  DO el_counter = 0, nnz - 1
     IF (A_row(el_counter) == A_col(el_counter)) THEN
        D_inv_val(el_counter_D) = 1.d0 / A_val(el_counter)
        D_inv_row(el_counter_D) = A_row(el_counter)
        D_inv_col(el_counter_D) = A_col(el_counter)
        el_counter_D = el_counter_D + 1
     ELSE
        R_val(el_counter_R) = A_val(el_counter)
        R_row(el_counter_R) = A_row(el_counter)
        R_col(el_counter_R) = A_col(el_counter)
        el_counter_R = el_counter_R + 1
     END IF
  END DO
  CALL compress_row(nx*ny, nx*ny, D_inv_row, D_inv_crow)
  CALL compress_row(nx*ny, nnz - nx*ny, R_row, R_crow)
  ! Compute the relevant entries from the obstacles
  rel_els = 1-reshape(obstacles,shape(rel_els))

  ! Apply the jacobi iteration
  x = guess
  x_old = guess - 1
  DO iter = 0, max_iter
    error = norm2((x - x_old)*rel_els)
    IF (error < tol) THEN
      EXIT
    ENDIF
    x_old = x
    CALL sparse_matmul(nx*ny,nnz,r_val,r_crow,r_col,x,tmp)
    CALL sparse_matmul(nx*ny,nnz,d_inv_val,d_inv_crow,d_inv_col,b-tmp,x)
  ENDDO

END SUBROUTINE
