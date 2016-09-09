!      program debug_cpf
!      integer (kind=8) , parameter :: n = 5
!      real (kind=8) :: s_x = 10
!      real (kind=8) :: s_y = 10
!      real (kind=8) :: min_dist = 1.5
!      real (kind=8), dimension(n,2) :: pos,corrects
!      integer, dimension(n) :: active
!      data pos / 1, 2.2, 3.4, 4.5, 6.7, 2.2, 2.5, 2, 7.4, 8.4/
!      active=1
!      call compute_potential_field(pos,s_x,s_y,active,min_dist,n,corrects)
!      write(*,*) corrects
!      end

subroutine compute_potential_field(cell_x,cell_y,n,pot_field,uf_left,uf_right,uf_up,uf_down,out_pot,inf)

!   Compute the potential in a single cell with a first order upwind method
implicit none
integer (kind=4) ::  n
!f2py intent(in) cell_x,cell_y,pot_field,uf_left,uf_right,uf_up,uf_down
!f2py intent(out) pot
!f2py depend(n) pot_field,uf_left,uf_right,uf_up,uf_down
!
! this subroutine uses 0 based indexing
!
real (kind=8) :: inf ! Domain size
real (kind=8) :: a,b,c,D,x_high ! parameters for upwind approximation
integer (kind=4) :: normal_x,normal_y,face_index_x,face_index_y ! Administration
integer (kind=4) :: nb_cell_x,nb_cell_y,! neighbour indices
real (kind=8):: hor_potential,ver_potential_hor_cost,ver_cost

integer (kind=4) :: direction
integer (kind=4), parameter :: LEFT = 0
integer (kind=4), parameter :: DOWN = 1
integer (kind=4), parameter :: RIGHT = 2
integer (kind=4), parameter :: UP = 3
real (kind=8), dimension(n,n) :: pot_field
real (kind=8), dimension(n-1,n) :: uf_left,uf_right
real (kind=8), dimension(n,n-1) :: uf_up,uf_down
real (kind=8), dimension(4) :: neighbour_pots
integer (kind=4) :: cell_x,cell_y
real (kind=8) :: pot, out_pot

data neighbour_pots / inf, inf, inf, inf / 
do direction=0,3

        normal_x = sign(mod(direction,2),direction-2)
        normal_y = sign(mod(direction+1,2),direction-2)
        nb_cell_x = cell_x + normal_x
        nb_cell_y = cell_y + normal_y
        if (exists(nb_cell_x,nb_cell_y,n,n) == 0) then
            continue
        end if
        pot = pot_field(nb_cell_x,nb_cell_y)
        if (direction == RIGHT) then
            face_index_x = nb_cell_x - 1
            face_index_y = nb_cell_y
            cost = uf_left(face_index_x,face_index_y)
        elseif (direction == UP) then
            face_index_x = nb_cell_x
            face_index_y = nb_cell_y - 1
            cost = uf_down(face_index_x,face_index_y)
        elseif (direction == LEFT) then
            face_index_x = nb_cell_x
            face_index_y = nb_cell_y
            cost = uf_left(face_index_x,face_index_y)
        elseif (direction == DOWN) then
            face_index_x = nb_cell_x
            face_index_y = nb_cell_y
            cost = uf_up(face_index_x,face_index_y)
        endif
        neighbour_pots[direction] + pot + cost
        if (neighbour_pots[direction] < neighbour_pots[mod(direction+2,4)] then
            if (mod(direction,2) == 0) then
                hor_potential = pot
                hor_cost = cost
            else
                ver_potential = pot
                ver_cost = cost
            endif
        endif
end do

if (hor_cost >= inf) then
    a = 1. / (ver_cost * ver_cost)
    b = -2. * ver_potential / (ver_cost * ver_cost)
    c = (ver_pot / ver_cost) * (ver_pot / ver_cost) -1
elseif (ver_cost >=inf) then
    a = 1. / (hor_cost * hor_cost)
    b = -2. * hor_potential / (hor_cost * hor_cost)
    c = (hor_pot / hor_cost) * (hor_pot / hor_cost) -1
else
    a = 1. / (hor_cost * hor_cost) + 1. / (ver_cost * ver_cost)
    b = -2. * (hor_potential / (hor_cost * hor_cost) + ver_pot / (ver_cost * ver_cost))
    c = (hor_pot / hor_cost) * (hor_pot / hor_cost) + (ver_pot / ver_cost) * (ver_pot / ver_cost) - 1
endif

D = b*b-4.*a*c
if (D<0) then
    write(*,*) "D < 0"
    D = 0
endif

out_pot = (-b + sqrt(D)) / (2.*a)
return
end

integer (kind=4) function exists(cell_x,cell_y,max_x,max_y)
! Find out whether the cell exists given the dimensions of the scene
integer (kind=4) :: cell_x,cell_y,max_x,max_y,ex
if (cell_x < 0 .or. cell_y < 0 .or. cell_x>=max_x .or. cell_y>=max_y) then
    ex = 0
    return
end if
ex = 1
return
end

