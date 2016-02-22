! laplace
program pressure_modules
implicit none
integer, parameter :: nx = 40
integer, parameter :: ny = 40
real (kind=8), parameter :: hx = 1./(nx + 1) ! Change with real dx, real dy
real (kind=8), parameter :: hy = 1./(ny + 1)
real (kind=8), parameter :: dt = 0.05
real (kind=8), dimension(nx,ny) :: density, velo_x, velo_y
real (kind=8), dimension(nx*ny) :: out_pressure,flat_density
real (kind=8), dimension(nx*ny) :: b
!! Number of nonzero matrix elements: 5 for each interior point, minus the interior boundary
integer, parameter :: nnz = 5*(nx)*(ny) - 2*(nx) - 2*(ny)

!! CSR matrix
real (kind=8), dimension(nnz) :: a
integer, dimension(nnz) :: ja
integer, dimension(nx*ny+1) :: ia
real,parameter :: max_density = 2
! real (kind=8),external :: func
real :: start,finish
! integer :: i,j
call cpu_time(start)
! initialize matrix A and vector x:

call random_number(density)
density = density/3. + 1.80
flat_density = reshape(density,(/nx*ny/))
velo_x = 0.6
velo_y = -0.4
call create_sparse_stencil(density,velo_x,velo_y,nx,ny,hx,hy,a,ja,ia,b)
b = max_density - flat_density - b*dt
a = -a*dt
call sparse_pgs(a,ja,ia,nnz,b,nx*ny,out_pressure)
write(*,*) out_pressure
write(*,*) density
call cpu_time(finish)
print '("Time = ",f6.3," seconds.")',finish-start
 end program pressure_modules

subroutine compute_pressure(density,velo_x,velo_y,nx,ny,dx,dy,dt,max_density,out_pressure)
implicit none
integer nx,ny,nnz
real (kind=8), dimension(nx,ny) :: density, velo_x, velo_y
real (kind=8) dx,dy,dt
real (kind=8) max_density
real (kind=8), dimension(nx*ny) :: out_pressure,flat_density
real (kind=8), dimension(nx*ny) :: b
!f2py intent(in) density,velo_x,velo_y,dx,dy,dt,max_density
!f2py intent(out) out_pressure
!f2py depend(nx,ny) out_pressure

!! CSR matrix
real (kind=8), dimension(5*(nx)*(ny) - 2*(nx) - 2*(ny)) :: a
integer, dimension(5*(nx)*(ny) - 2*(nx) - 2*(ny)) :: ja
integer, dimension(nx*ny+1) :: ia
nnz = 5*(nx)*(ny) - 2*(nx) - 2*(ny)
flat_density = reshape(density,(/nx*ny/))
call create_sparse_stencil(density,velo_x,velo_y,nx,ny,dx,dy,a,ja,ia,b)
b = max_density - flat_density - b*dt
a = -a*dt
call sparse_pgs(a,ja,ia,nnz,b,nx*ny,out_pressure)

end subroutine


!! Init data structures
subroutine create_sparse_stencil(density,velo_x,velo_y,nx,ny,dx,dy,out_a,out_ja,out_ia,out_b)
! Create CSR matrices for the scheme in the thesis. 
implicit none
integer :: i,j,el
integer :: nx, ny
real (kind=8) :: dx,dy
real (kind=8), dimension(0:nx+1,0:ny+1) :: rho, vx, vy
real (kind=8), dimension(nx,ny) :: density, velo_x, velo_y
!! Number of nonzero matrix elements: 5 for each interior point, minus the interior boundary
integer :: nnz

!! COO matrix
real (kind=8), dimension(5*(nx)*(ny) - 2*(nx) - 2*(ny)) :: SM
integer, dimension(5*(nx)*(ny) - 2*(nx) - 2*(ny)) :: row,col

!! CSR matrix
real (kind=8), dimension(5*(nx)*(ny) - 2*(nx) - 2*(ny)) :: out_a
integer, dimension(5*(nx)*(ny) - 2*(nx) - 2*(ny)) :: out_ja
integer, dimension(nx*ny+1) :: out_ia

!! Right hand size
real (kind=8), dimension(nx*ny) :: out_b
!! Dirichlet boundary conditions
real, parameter :: p0 = 1.

nnz = 5*(nx)*(ny) - 2*(nx) - 2*(ny)
rho = 0
vx = 0
vy = 0

rho(1:nx,1:ny) = density
vx(1:nx,1:ny) = velo_x
vy(1:nx,1:ny) = velo_y
out_b = 0
el = 1

!! Start filling matrix
do i=1,nx
do j=1,ny
SM(el) = -(2*rho(i,j)/(dx*dx)+2*rho(i,j)/(dy*dy))
row(el) = i+(j-1)*nx
col(el) = i+(j-1)*nx
el = el + 1
out_b(i+(j-1)*nx) = - (rho(i-1,j)*vx(i-1,j) - rho(i+1,j)*vx(i+1,j))/(2*dx) &
                - (rho(i,j-1)*vy(i,j-1) - rho(i,j+1)*vy(i,j+1))/(2*dy)
if (i>1) then ! Has left point. Add this to the stencil
    SM(el) = -(rho(i-1,j) - rho(i+1,j))/(4*dx*dx) + rho(i,j)/(dx*dx)
    row(el) = i+(j-1)*nx
    col(el) = (i-1)+(j-1)*nx
    el = el+1
else ! Correct stencil with Diriclet boundary conditions
    out_b(i+(j-1)*nx) = out_b(i+(j-1)*nx)-p0*(-(rho(i-1,j) - rho(i+1,j))/(4*dx*dx) + rho(i,j)/(dx*dx))
end if

if (i<nx) then ! Has right point. Add this to the stencil
    SM(el) = (rho(i-1,j) - rho(i+1,j))/(4*dx*dx) + rho(i,j)/(dx*dx)
    row(el) = i+(j-1)*nx
    col(el) = i+1 + (j-1)*nx
    el = el + 1
else ! Correct stencil with Diriclet boundary conditions
    out_b(i+(j-1)*nx) = out_b(i+(j-1)*nx)- p0*((rho(i-1,j) - rho(i+1,j))/(4*dx*dx) + rho(i,j)/(dx*dx))
end if

if (j>1) then ! Has bottom point. Add this to the stencil
    SM(el) = -(rho(i,j-1) - rho(i,j+1))/(4*dy*dy) + rho(i,j)/(dy*dy)
    row(el) = i+(j-1)*nx
    col(el) = i + (j-2)*nx
    el = el + 1
else ! Correct stencil with Diriclet boundary conditions
    out_b(i+(j-1)*nx) = out_b(i+(j-1)*nx) - p0*(-(rho(i,j-1) - rho(i,j+1))/(4*dy*dy) + rho(i,j)/(dy*dy))
end if

if (j<ny) then ! Has top point. Add this to the stencil
    SM(el) = (rho(i,j-1) - rho(i,j+1))/(4*dy*dy) + rho(i,j)/(dy*dy)
    row(el) = i+(j-1)*nx
    col(el) = i + j*nx
    el = el + 1
else ! Correct stencil with Diriclet boundary conditions
    out_b(i+(j-1)*nx) = out_b(i+(j-1)*nx) - p0*(+(rho(i,j-1) - rho(i,j+1))/(4*dy*dy) + rho(i,j)/(dy*dy))
end if
end do
end do
!! Convert to CSR format ! Todo: Time to see if should be inplace
call coocsr(nx*ny,nnz,SM,row,col,out_a,out_ja,out_ia)
end subroutine

subroutine sparse_pgs(a,ja,ia,nnz,q,n,out_pressure)
    !! Use PGS to solve the system
implicit none
integer n,nnz
integer it,i,length,k
integer ::max_it = 1000
real (kind=8) :: eps = 0.001
real (kind=8) :: r,prod
real (kind=8), dimension(nnz) ::  a
integer, dimension (n+1) :: ia
integer, dimension (nnz) :: ja
real (kind=8), dimension(n) :: q
real (kind=8), dimension(n) :: diag,idiag
real (kind=8), dimension(n):: out_pressure
real (kind=8), dimension(n) :: density_overshoot

density_overshoot=0 !
     
     call amux(n,out_pressure,density_overshoot,a,ja,ia)
     density_overshoot = density_overshoot+q
     it = 0
     call getdia(n,n,0,a,ja,ia,length,diag,idiag,0)
     do while ((any(density_overshoot < -eps) .or. abs(dot_product(density_overshoot,out_pressure))>eps)&
          .and. (it < max_it))
     !write(*,*) "looping",it
  it = it+1
  do i=1,n
      prod = 0.0
      do k=ia(i),ia(i+1)-1
        prod = prod+a(k)*out_pressure(ja(k))
      end do
      !call amux(n,out_pressure,aux_prod,a,ja,ia)
      r = -q(i) - prod + diag(i)*out_pressure(i)
      out_pressure(i) = max(0.,r/diag(i))
  end do
  call amux(n,out_pressure,density_overshoot,a,ja,ia)
     density_overshoot = density_overshoot+q
     !write(*,*) "Score: ",dot_product(density_overshoot,out_pressure)
     end do
     end subroutine
     !write(*,*) "done at it ",it
     !write(*,*) "density ", density_overshoot
     !write(*,*) "pressure ", out_pressure

 
