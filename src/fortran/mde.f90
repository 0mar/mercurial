!      program debug_mde
!      integer (kind=8) , parameter :: n = 5
!      real (kind=8) :: s_x = 10
!      real (kind=8) :: s_y = 10
!      real (kind=8) :: min_dist = 1.5
!      real (kind=8), dimension(n,2) :: pos,corrects
!      integer, dimension(n) :: active
!      data pos / 1, 2.2, 3.4, 4.5, 6.7, 2.2, 2.5, 2, 7.4, 8.4/
!      active=1
!      call compute_mde(pos,s_x,s_y,active,min_dist,n,corrects)
!      write(*,*) corrects
!      end

subroutine compute_mde(pos,s_x,s_y,active,min_dist,n,corrects)
! Correct almost colliding particles
implicit none
integer (kind=8) ::  n
!f2py intent(in) pos,s_x,s_y,active,min_dist
!f2py intent(out) corrects
!f2py depend(n) corrects
!
! this subroutine uses 1 based indexing
!
integer (kind=8) :: i,j,k,k2,l,max_val ! Looping variables
real (kind=8) :: s_x,s_y ! Domain size
integer (kind=8) :: n_x,n_y ! binning stuff
integer (kind=8) :: nb_i,nb_j ! neighbour indices
real (kind=8):: min_dist,dist 

integer, dimension(n) :: active ! Whether pedestrian is active
real (kind=8), parameter :: eps = 0.0001
real (kind=8),dimension(n,2) :: pos,corrects ! positions
integer (kind=8),dimension(n,2) :: cell_pos
real (kind=8) :: diff_x,diff_y

! Compute the distribution of particles 
integer(kind=8),dimension(:,:),allocatable :: count_arr
integer(kind=8),dimension(:,:,:),allocatable :: bin_arr
n_x = int(s_x/min_dist)+1
n_y = int(s_y/min_dist)+1
allocate(count_arr(n_x,n_y))
count_arr=0
cell_pos = int(pos/min_dist)+1
do k=1,n
    cell_pos(k, 1) = max(min(cell_pos(k, 1), n_x), 1)
    cell_pos(k, 2) = max(min(cell_pos(k, 2), n_y), 1)
    count_arr(cell_pos(k,1),cell_pos(k,2)) =&
    count_arr(cell_pos(k,1),cell_pos(k,2))+active(k)
end do
!write(*,*) count_arr

! Bin the particles 
max_val = maxval(count_arr)
allocate(bin_arr(n_x,n_y,max_val))
bin_arr = 0
do k=1,n
    if (active(k)==1) then
        bin_arr(cell_pos(k,1),cell_pos(k,2),&
        count_arr(cell_pos(k,1),cell_pos(k,2))) = k ! Fill the bins backwards
        count_arr(cell_pos(k,1),cell_pos(k,2)) =&
        count_arr(cell_pos(k,1),cell_pos(k,2))-1
    end if
end do

! Compute the interactions (+80% of time)
corrects = 0
do k=1,n
    if (active(k)==1) then
        do nb_i=-1,1
            i = nb_i + cell_pos(k,1)
            if (1<= i .and. i <=n_x) then
                do nb_j = -1,1 
                    j = nb_j + cell_pos(k,2)
                    if (1<= j .and. j <=n_y) then
                        do l=1,max_val
                            k2 = bin_arr(i,j,l)
                            if (k2==0) then
                                exit
                            endif
                            if (k2/=k .and. active(k2)==1) then
                                diff_x = pos(k,1) - pos(k2,1)
                                diff_y = pos(k,2) - pos(k2,2)
                                dist = sqrt(diff_x*diff_x + diff_y*diff_y)
                                corrects(k,1) = corrects(k,1) +&
                                max(min_dist-dist,0.)/(2*dist)*diff_x
                                corrects(k,2) = corrects(k,2) +&
                                max(min_dist-dist,0.)/(2*dist)*diff_y
                            end if
                        end do
                    end if
                end do
            end if
        end do
    end if
end do
return
end
