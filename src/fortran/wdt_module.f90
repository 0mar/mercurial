module wdt_module
    implicit none
    public :: exists, propagate_dist
    public :: nx,ny
    public :: LEFT, DOWN, RIGHT, UP, NUM_DIRS, KNOWN, UNKNOWN, CANDIDATE, NEW_CANDIDATE
    integer (kind=4) :: nx,ny
    real (kind=8) :: obstacle_value, unknown_value

    integer (kind=4), parameter :: LEFT = 0
    integer (kind=4), parameter :: DOWN = 1
    integer (kind=4), parameter :: RIGHT = 2
    integer (kind=4), parameter :: UP = 3
    integer (kind=4), parameter :: NUM_DIRS = 4

    integer (KIND=4), parameter :: KNOWN = 0
    integer (KIND=4), parameter :: UNKNOWN = 1
    integer (KIND=4), parameter :: CANDIDATE = 2
    integer (KIND=4), parameter :: NEW_CANDIDATE = 3

contains
    subroutine new_candidate_cells(cell_x,cell_y,wdt_field,n_x,n_y,cand_cells)
        implicit none
        ! Find new candidate cells
        integer (kind=4), intent(in) :: cell_x,cell_y,n_x,n_y
        real (kind=8), dimension(0:n_x-1,0:n_y-1), intent(in) :: wdt_field
        integer (kind=4), dimension(0:3,0:1), intent(out) :: cand_cells
        integer (kind=4) :: direction,nb_x,nb_y
        cand_cells = -1
        do direction=0,3
            nb_x = sign(mod(direction+1,2),direction-1) + cell_x
            nb_y = sign(mod(direction,2),direction-2) + cell_y
            if (exists(nb_x,nb_y)==1)  then
                if (wdt_field(nb_x,nb_y) == unknown_value) then
                    cand_cells(direction,0) = nb_x
                    cand_cells(direction,1) = nb_y
                end if
            end if
        end do
    end subroutine

    subroutine propagate_dist(cell_x,cell_y,wdt_field,costs_x,costs_y,nx,ny,out_pot)
    !   Compute the potential in a single cell with a first order upwind method
    implicit none
    integer (kind=4),intent(in) :: cell_x,cell_y,nx,ny
    real (kind=8) :: a,b,c,D ! parameters for upwind approximation
    integer (kind=4) :: normal_x,normal_y,face_index_x,face_index_y! Administration
    integer (kind=4) :: nb_cell_x,nb_cell_y! neighbour indices
    real (kind=8):: hor_potential,ver_potential,hor_cost,ver_cost

    integer (kind=4) :: direction

    real (kind=8), dimension(0:nx-1,0:ny-1), intent(in) :: wdt_field
    real (kind=8), dimension(0:nx,0:ny-1), intent(in) :: costs_x
    real (kind=8), dimension(0:nx-1,0:ny), intent(in) :: costs_y
    real (kind=8), dimension(0:3) :: neighbour_pots
    real (kind=8) :: pot,cost
    real (kind=8), intent(out) :: out_pot

    neighbour_pots=  (/ obstacle_value, obstacle_value, obstacle_value, obstacle_value /) 
    hor_cost = obstacle_value
    ver_cost = obstacle_value
        ! Find the minimal directions along a grid cell.
        ! Assume left and below are best, then overwrite with right and up if they are better
    do direction=0,3
            normal_x = sign(mod(direction+1,2),direction-1)
            normal_y = sign(mod(direction,2),direction-2)
            ! numerical direction
            nb_cell_x = cell_x + normal_x
            nb_cell_y = cell_y + normal_y
            if (exists(nb_cell_x,nb_cell_y) == 0) then
                cycle
            endif
            pot = wdt_field(nb_cell_x,nb_cell_y)
            ! potential in that neighbour field
            if (direction == RIGHT) then
                face_index_x = nb_cell_x
                face_index_y = nb_cell_y
                ! Unit cost values are defined w.r.t faces, not cells! So the indexing is different with right and up.
                cost = costs_x(face_index_x,face_index_y)
                ! Cost to go from there to here
            elseif (direction == UP) then
                face_index_x = nb_cell_x
                face_index_y = nb_cell_y
                ! Unit cost values are defined w.r.t faces, not cells! So the indexing is different with right and up.
                cost = costs_y(face_index_x,face_index_y)
            elseif (direction == LEFT) then
                face_index_x = nb_cell_x+1
                face_index_y = nb_cell_y
                cost = costs_x(face_index_x,face_index_y)
            elseif (direction == DOWN) then
                face_index_x = nb_cell_x
                face_index_y = nb_cell_y+1
                cost = costs_y(face_index_x,face_index_y)
            else
                !write(*,*) "Exception in compute_potential"
            endif
             
            neighbour_pots(direction) = pot + cost
            ! total potential
            if (neighbour_pots(direction) < neighbour_pots(mod(direction+2,4))) then
                if (mod(direction,2) == 0) then
                    hor_potential = pot
                    hor_cost = cost
                    !write(*,*) "Horizontal changed"
                    ! lowest in horizontal direction
                else
                    ver_potential = pot
                    ver_cost = cost
                    !write(*,*) "Vertical changed"
                    ! lowest in vertical direction
                endif
            endif
    end do

    if (hor_potential >= obstacle_value) then
        !write(*,*) "Horizontal obstacle_valueinite"
        a = 1. / (ver_cost * ver_cost)
        b = -2. * ver_potential / (ver_cost * ver_cost)
        c = (ver_potential / ver_cost) * (ver_potential / ver_cost) -1
    elseif (ver_potential >=obstacle_value) then
        !write(*,*) "Vertical obstacle_valueinite"
        a = 1. / (hor_cost * hor_cost)
        b = -2. * hor_potential / (hor_cost * hor_cost)
        c = (hor_potential / hor_cost) * (hor_potential / hor_cost) -1
    else
        !write(*,*) "All good"
        a = 1. / (hor_cost * hor_cost) + 1. / (ver_cost * ver_cost)
        b = -2. * (hor_potential / (hor_cost * hor_cost) + ver_potential / (ver_cost * ver_cost))
        c = (hor_potential / hor_cost) * (hor_potential / hor_cost) + (ver_potential / ver_cost) * (ver_potential / ver_cost) - 1
    endif
    D = b*b-4.*a*c
    out_pot = (-b + sqrt(D)) / (2.*a)
    end subroutine

    function exists(cell_x,cell_y) result(ex)
        implicit none
        ! Find out whether the cell exists given the dimensions of the scene
        integer (kind=4), intent(in) :: cell_x,cell_y
        integer (kind=4) :: ex
        if (cell_x < 0 .or. cell_y < 0 .or. cell_x>=nx .or. cell_y>=ny) then
            ex = 0
        else
            ex = 1
        end if
    end function exists
end module wdt_module

subroutine weighted_distance_transform(cost_field,wdt_field,n_x,n_y,obs_val)
    ! Todo: Somehow make sure I don't have to input the n_x/n_y in Python
use wdt_module
implicit none
!   Compute the potential in a single cell with a first order upwind method
!f2py intent(in) cost_field,obs_val,n_x,n_y
!f2py depend(n_x,n_y) cost_field,wdt_field
!f2py intent(out) wdt_field
integer (kind=4) :: n_x,n_y
real (kind=8):: obs_val

integer (kind=4), dimension(0:3,0:1) :: new_cand_cells
integer (kind=4), dimension(0:1) :: best_cell

integer (kind=4), allocatable, dimension(:,:) :: cand_heap
integer (kind=4), allocatable, dimension(:) :: indx
integer (kind=4) :: heap_capacity, heap_length, tree_length

integer (kind=4) :: i,j,l
real (kind=8), dimension(0:n_x-1,0:n_y-1) :: cost_field,wdt_field
real (kind=8), dimension(0:n_x,0:n_y-1) :: costs_x
real (kind=8), dimension(0:n_x-1,0:n_y) :: costs_y
integer (kind=4), dimension(0:n_x-1,0:n_y-1) :: cell_indicators
real (kind=8) :: dist

nx = n_x
ny = n_y
obstacle_value = obs_val
unknown_value = obs_val + 1
! Cost for moving along horizontal lines
costs_x = obstacle_value
costs_x(1:n_x-1,:) = (cost_field(1:n_x-1,:) + cost_field(0:n_x-2,:))/2
! Cost for moving along vertical lines
costs_y = obstacle_value
costs_y(:,1:n_y-1) = (cost_field(:,1:n_y-1) + cost_field(:,0:n_y-2))/2

!Initialize locations (known(exit/obstacles)/unknown)
wdt_field = unknown_value 
cell_indicators = UNKNOWN 

heap_length = 0
tree_length = 0
heap_capacity = (n_x+n_y)*1000 ! Todo: This number is pretty arbitrary. I think it should be hc = a*(nx+ny) + b*(#exits)
allocate(cand_heap(0:1,0:heap_capacity-1))
allocate(indx(0:heap_capacity-1))
call heap_init(indx,heap_capacity)

do j=0,n_y-1
    do i=0,n_x-1
        if (cost_field(i,j)==0) then
            ! No cost, so this is an exit
            wdt_field(i,j) = 0
            cell_indicators(i,j) = KNOWN
            call heap_insert(cand_heap,indx,heap_capacity,heap_length,tree_length,[i,j],wdt_field,n_x,n_y)
        elseif (cost_field(i,j)>=obstacle_value) then
            ! 'infinite cost', so this is an obstacle
            wdt_field(i,j) = obstacle_value
            cell_indicators(i,j) = KNOWN
        endif
    enddo
enddo

! Iteration of level set
do while (.true.)
    if (heap_length==0) then
        exit
    end if
    call heap_pop(cand_heap, indx, heap_capacity, heap_length, wdt_field, n_x, n_y, best_cell)

    call new_candidate_cells(best_cell(0),best_cell(1),wdt_field,n_x,n_y,new_cand_cells)
    do l=0,3
        i = new_cand_cells(l,0)
        if (i >=0) then
            j = new_cand_cells(l,1)
            call propagate_dist(i,j,wdt_field,costs_x,costs_y,n_x,n_y,dist)
            if (wdt_field(i,j) >= dist) then
                wdt_field(i,j) = dist
                call heap_insert(cand_heap,indx,heap_capacity,heap_length,tree_length,[i,j],wdt_field,n_x,n_y)
            end if
        end if
    end do
end do
deallocate(cand_heap)
deallocate(indx)

end subroutine

program test_wdt
    implicit none
    integer (kind=4), parameter :: n_x=400
    integer (kind=4), parameter :: n_y=500
    integer (kind=4) :: k
    real (kind=8), dimension(0:n_x-1,0:n_y-1) :: cost_field,wdt_field
    real (kind=8), parameter :: obstacle_value = 2000

    ! Example:
    !- - - -
    !* * * -
    !- * - -
    !- - - -

!    cost_field = 0.01
!    cost_field(1,1) = 0
!    cost_field(0:3,2) = 0
!    cost_field(3,1:3)= obstacle_value
!    cost_field(50,:) = obstacle_value
!    cost_field(50,250) = 0.01
    call weighted_distance_transform(cost_field,wdt_field,n_x,n_y,obstacle_value)
    k = 3
    open(k, file="output.dat", access="stream")
    write(k) wdt_field
    close(k)
end program

