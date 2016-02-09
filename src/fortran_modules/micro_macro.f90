subroutine comp_dens_velo(pos, velo, active, n_x, n_y, dx, dy,dens, v_x, v_y,n)
implicit none
integer n
!
! Compute the density and velocity field using the input values
integer (kind=8) :: n_x,n_y ! number of cells in x/y direction
integer (kind=8) :: i,j,k,range_ ! Looping variables
integer (kind=8) :: x_cell,y_cell ! Cell of pedestrian position
integer, dimension(0:n-1) :: active ! Whether pedestrian is active

real (kind=8):: dx,dy ! Cell size
real (kind=8), parameter :: eps = 0.0001
real (kind=8):: x_center,y_center ! cell center
real (kind=8),dimension(0:n_x-1,0:n_y-1) :: dens,v_x,v_y ! output
real (kind=8),dimension(0:n-1,0:1) :: pos,velo ! input
real (kind=8) :: dist,weight, h ! smoothing length
real (kind=8), external :: weight_function

!f2py intent(in) n_x,n_y,dx,dy
!f2py intent(out) dens,v_x,v_y
!f2py depend(n_x,n_y) dens,v_x,v_y
! Initialization
dens=0
v_x=0
v_y=0
h = sqrt(dx*dx+dy*dy)*3/4
range_ = int(2.*h/(min(dx,dy))+1) ! depends on smoothing length

!Computing interpolations
do k=0,n-1
    x_cell = int(pos(k,0)/dx)
    y_cell = int(pos(k,1)/dy)
    do i=x_cell-range_,x_cell+range_
        if (i>=0 .and. i<=n_x-1) then
            do j=y_cell-range_,y_cell+range_
                if (j>=0 .and. j<=n_y-1) then
                    ! Contributions of pedestrian k for cell (i,k)
                    x_center = (i+0.5)*dx
                    y_center = (j+0.5)*dy
                    dist = sqrt((pos(k,0)-x_center)*(pos(k,0)-x_center) +&
                    (pos(k,1)-y_center)*(pos(k,1)-y_center))
                    weight = weight_function(dist,h)*active(k)
                    dens(i,j) = dens(i,j) + weight
                    v_x(i,j) = v_x(i,j) + weight*velo(k,0)
                    v_y(i,j) = v_y(i,j) + weight*velo(k,1)
                end if
            end do
        end if
    end do
end do
v_x = v_x/(dens + eps)
v_y = v_y/(dens + eps)
return
end

real (kind=8) function weight_function(distance,h)
    ! Compute gaussian interpolation kernel
    implicit none

    real (kind=8),parameter :: pi =3.14159265359
    real (kind=8) h ! smoothing length
    real (kind=8) :: norm,weight,distance
    norm = 7.0/(4*pi*h*h)
    weight = max(1-distance/(2*h),0.)**4*(1+2*distance/h)
    weight_function = norm*weight
end function




