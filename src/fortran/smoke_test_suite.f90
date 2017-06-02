PROGRAM evolve_smoke
  IMPLICIT NONE
    INTEGER, parameter ::nx=40
    INTEGER, parameter ::ny=40
    INTEGER :: nnz,i,j
    REAL (kind=8) :: diff,velo_x, velo_y
    REAL (kind=8) :: x,y,t,c
  REAL (kind=8), DIMENSION(0:nx*ny-1) :: b,u,rel_els, f,guess
    REAL (kind=8) dx,dy,dt
  INTEGER, DIMENSION(0:nx-1,0:ny-1):: obstacles
  REAL (kind=8), DIMENSION(0:5*(nx-2)*(ny-2) + 2*(nx + ny)-4-1) :: a_val
  INTEGER, DIMENSION(0:5*(nx-2)*(ny-2) + 2*(nx + ny)-4) :: a_col,a_row

  diff = 0.1
  velo_x = 0.9
  velo_y = 1.2
  dx = 0.1
  dy = 0.1
  dt = 0.2
  ! obstacles = reshape((/ 0, 0, 0, 0, 0, 0, 0, 0, 0 /), shape(obstacles))
  obstacles = 0
  obstacles(1,2) = 0

  c = 2
  do j=0,ny-1
      y = j*dy
      do i=0,nx-1
       x = i*dx 
        f(i+j*nx) = c*exp(-((x-nx*dx/2)**2+(y-ny*dy/2)**2)*c**2)
      enddo
  enddo
   call get_sparse_matrix(diff,velo_x,velo_y,nx,ny,dx,dy,dt,obstacles, a_val, a_row,a_col, nnz)
   u=0
   guess = 0
   call iterate_jacobi(a_val,a_row,a_col,nnz,b,guess,obstacles,nx,ny,x)
  write(*, '(*(F7.3))')( u(j) ,j=0,nx*ny-1)

END PROGRAM evolve_smoke
