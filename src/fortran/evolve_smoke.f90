PROGRAM evolve_smoke
  IMPLICIT NONE

END PROGRAM evolve_smoke

SUBROUTINE get_sparse_matrix(diff,velo_x,velo_y,nx,ny,dx,dy,dt,obstacles, a_val, a_row,a_col)
  IMPLICIT NONE
  ! Includes one layer for the boundary
  INTEGER nx,ny,nnz
  REAL (kind=8) :: diff,velo_x, velo_y
  REAL (kind=8) dx,dy,DATA
  INTEGER :: i,j,el_counter,el_index, self_el
  REAL (kind=8) :: l_up,l_down,l_left,l_right
  INTEGER, DIMENSION(nx,ny):: obstacles
  ! Number of nonzero elements: 5 for every cell, 1 for every virtual boundary
  REAL (kind=8), DIMENSION(0, 5*(nx-2)*(ny-2) + 2*(nx + ny)-4-1) :: a_val
  INTEGER (kind=8), DIMENSION(0, 5*(nx-2)*(ny-2) + 2*(nx + ny)-4) :: a_col,a_row
  INTEGER (kind=8), DIMENSION(0, nx*ny-1) :: a_crow
  nnz = 5*(nx-2)*(ny-2) + 2*(nx + ny)-4

  ! Coefficients
  l_up = diff*dt/(dy*dy) - v_y*dt/(2*dy)
  l_down = diff*dt/(dy*dy) + v_y*dt/(2*dy)
  l_left = diff*dt/(dx*dx) + v_x*dt/(2*dx)
  l_right = diff*dt/(dx*dx) - v_x*dt/(2*dx)
  l_self = -2*diff*dt*(1/(dx*dx) + 1/(dy*dy)) - 1

  ! Start filling matrix
  el_counter = 0
  out_a = 0
  DO i = 0, nx + 1
    DO j = 0, ny + 1
      el_index = i + j * nx
      IF (obstacles(i,j)==1) THEN
        out_a(el_counter) = 1.d0
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index
        el_counter = el_counter + 1
      ELSE
        ! Self relation
        self_el = el_counter
        out_a(el_counter) = l_self
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index
        el_counter = el_counter + 1
        ! Down relation
        IF (obstacles(i,j-1)==0) THEN
          out_a(el_counter) = l_down
        ELSE
          out_a(self_el) = out_a(self_el) + l_down
        ENDIF
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index - nx
        el_counter = el_counter + 1
        ! Left relation
        IF (obstacles(i-1,j)==0) THEN
          out_a(el_counter) = l_left
        ELSE
          out_a(self_el) = out_a(self_el) + l_left
        ENDIF
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index - 1
        el_counter = el_counter + 1
        ! Right relation
        IF (obstacles(i+1,j)==0) THEN
          out_a(el_counter) = l_right
        ELSE
          out_a(self_el) = out_a(self_el) + l_right
        ENDIF
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index + 1
        el_counter = el_counter + 1
        ! Up relation
        IF (obstacles(i,j+1)==0) THEN
          out_a(el_counter) = l_up
        ELSE
          out_a(self_el) = out_a(self_el) + l_up
        ENDIF
        a_row(el_counter) = el_index
        a_col(el_counter) = el_index + nx
        el_counter = el_counter + 1
      ENDIF
    ENDDO
  ENDDO

! CALL compress_row(nx*ny,nnz,a_row,a_crow) no need, lets compress D^-1 & R
END SUBROUTINE

SUBROUTINE iterate_jacobi(a_val,a_crow,a_col,b,x,nx,ny,rel_els)
  IMPLICIT NONE
  REAL (kind=8), DIMENSION(0, 5*(nx-2)*(ny-2) + 2*(nx + ny)-4-1) :: a_val, D_inv_val, R_val
  INTEGER (kind=8), DIMENSION(0, 5*(nx-2)*(ny-2) + 2*(nx + ny)-4-1) ::a_row, a_col,d_inv_row, d_inv_col,r_row, r_col
  INTEGER (kind=8), DIMENSION(0, nx*ny-1) :: a_crow,d_inv_crow,r_inv_crow
  REAL (kind=8), DIMENSION(0,nx*ny-1) :: b,x,x_old,tmp,rel_els
  INTEGER (kind=8) nx,ny,nnz, iter, max_iter
  INTEGER (kind=8) el_counter_D, el_counter_R, el_counter
  REAL tol, error

  tol = MIN(dx,dy)**2
  nnz = 5*(nx-2)*(ny-2) + 2*(nx + ny)-4
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

  ! Apply the jacobi iteration
  x_old = 0
  DO iter = 0, max_iter
    iter = iter + 1
    error = norm2((x - x_old)*rel_els)
    IF (error < tol) THEN
      EXIT
    ENDIF
    CALL sparse_matmul(nx*ny,nnz,r_val,r_crow,r_col,x,tmp)
    CALL sparse_matmul(nx*ny,nnz,d_inv_val,d_inv_crow,b-tmp,x)
  ENDDO

END SUBROUTINE
