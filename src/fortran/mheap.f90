! Copyright (c) 2014, Daniel Pena
! All rights reserved.

! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:

! 1. Redistributions of source code must retain the above copyright notice, this
! list of conditions and the following disclaimer.
! 2. Redistributions in binary form must reproduce the above copyright notice,
! this list of conditions and the following disclaimer in the documentation
! and/or other materials provided with the distribution.

! this software is provided by the copyright holders and contributors "as is" and
! any express or implied warranties, including, but not limited to, the implied
! warranties of merchantability and fitness for a particular purpose are
! disclaimed. in no event shall the copyright owner or contributors be liable for
! any direct, indirect, incidental, special, exemplary, or consequential damages
! (including, but not limited to, procurement of substitute goods or services;
! loss of use, data, or profits; or business interruption) however caused and
! on any theory of liability, whether in contract, strict liability, or tort
! (including negligence or otherwise) arising in any way out of the use of this
! software, even if advised of the possibility of such damage.


!private
!public :: heap_init, heap_insert, heap_pop, heap_peek, heap_reheap
subroutine compare(node1, node2, field, nx, ny, smaller)
    integer (kind=4), dimension(0:1) :: node1, node2
    integer (kind=4) :: nx, ny
    real (kind=8), dimension(0:nx-1,0:ny-1) :: field
    logical :: smaller

    smaller = field(node1(0),node1(1)) < field(node2(0),node2(1))
end subroutine

subroutine heap_init(indx,cap)
  ! initializes the heap
  ! nmax  -  max size of the heap
  ! nlen  -  size of each node
  ! hpfun -  the heap function (provides comparison between two nodes' data)
  integer (kind=4), intent(in) :: cap
  integer (kind=4), dimension(0:cap-1) :: indx
  integer (kind=4) :: i
  do i = 0, cap -1
     indx(i)=i
  enddo
end subroutine heap_init

subroutine heap_insert(heap,indx,cap,heap_length,tree_length,node,field,nx,ny)
  ! insert a node into a heap. the resulting tree is re-heaped.
  !  input
  !        heap - the heap
  !        node - a double precision array, nlen long, which
  !               contains the node's information to be inserted.
  integer (kind=4), intent(in) :: cap
  integer (kind=4), intent(inout) :: heap_length, tree_length
  integer (kind=4), intent(inout), dimension(0:1,0:cap-1) :: heap
  integer (kind=4), intent(inout), dimension(0:cap-1) :: indx
  integer (kind=4), intent(in), dimension(0:1) :: node
  integer (kind=4), intent(in) :: nx, ny
  real (kind=8), intent(in), dimension(0:nx-1,ny-1) :: field

  integer (kind=4) :: k1, k2, il, ir
  logical :: smaller
  if (cap == heap_length) return

  ! add one element and copy node data to new element
  heap_length = heap_length + 1
  tree_length = tree_length + 1
  heap(:,indx(heap_length-1)) = node

  ! re-index the heap from the bottom up
  k2 = heap_length
  do while( k2 /= 1 )
     k1 = k2 / 2
     ir = indx(k2-1)
     il = indx(k1-1)
     call compare(heap(:,il), heap(:,ir), field,nx,ny,smaller)
     if (smaller) return
     call swapint(indx(k2-1), indx(k1-1))
     k2 = k2 / 2
  enddo
end subroutine heap_insert

subroutine heap_pop(heap, indx, cap, heap_length, field, nx, ny, node)
  ! retrieve the root element off the heap. the resulting tree is re-heaped.
  ! no data is deleted, thus the original
  !   input
  !        heap - the heap
  !   output
  !        node - the deleted node

  integer (kind=4), intent(in) :: cap
  integer (kind=4), intent(inout) :: heap_length
  integer (kind=4), intent(in), dimension(0:1,0:cap-1) :: heap
  integer (kind=4), intent(inout), dimension(0:cap-1) :: indx
  integer (kind=4), intent(out), dimension(0:1) :: node
  integer (kind=4), intent(in) :: nx, ny
  real (kind=8), intent(in), dimension(0:nx-1,ny-1) :: field

  if( heap_length == 0 ) then
      write(*,*) "Error: Nothing to pop"
  end if
  node = heap(:,indx(0))
  call swapint(indx(0), indx(heap_length-1))
  heap_length = heap_length-1

  call heap_grow(heap,indx, cap, heap_length, field, nx, ny, 1) !1 instead of 0 because of indexing artifact

end subroutine heap_pop

subroutine heap_peek( heap, indx,cap,heap_length,entr,node)
  ! access the k-th node of the heap

  integer (kind=4), intent(in) :: entr, cap
  integer (kind=4), intent(in) :: heap_length
  integer (kind=4), intent(in), dimension(0:1,0:cap-1) :: heap
  integer (kind=4), intent(in), dimension(0:cap-1) :: indx
  integer (kind=4), intent(out), dimension(0:1) :: node
  if (entr < 0 .or. entr >= heap_length .or. heap_length > cap) then
      write(*,*) "Error in peek"
      return
  endif
  node = heap(:,indx(entr))
end subroutine heap_peek

subroutine heap_grow(heap,indx, cap, heap_length, field, nx, ny, ktemp)
  ! forms a heap out of a tree. used privately by heap_reheap.
  ! the root node of the tree is stored in the location indx(ktemp).
  ! the first child node is in location indx(2*ktemp)...
  ! the next child node is in location indx(2*ktemp+1).
  ! this subroutines assumes each branch of the tree is itself a heap.
  integer (kind=4), intent(in) :: cap
  integer (kind=4), intent(inout) :: heap_length
  integer (kind=4), intent(in), dimension(0:1,0:cap-1) :: heap
  integer (kind=4), intent(inout), dimension(0:cap-1) :: indx
  integer (kind=4), intent(in) :: nx, ny
  real (kind=8), intent(in), dimension(0:nx-1,ny-1) :: field
  integer (kind=4) :: i, k, il, ir
  integer (kind=4) :: ktemp
  logical :: smaller

  if(cap == heap_length) then
      write(*,*) "Error in heap grow, capacity reached"
      return
  endif

  k = ktemp
  do while( 2*k <= heap_length )

     i = 2*k

     ! if there is more than one child node, find which is the smallest.
     if( 2*k /= heap_length ) then
        il = indx(2*k)
        ir = indx(2*k-1)
        call compare(heap(:,il), heap(:,ir), field,nx,ny,smaller)
        if(smaller) then
           i = i + 1
        endif
     endif

     ! if a child is larger than its parent, interchange them... this destroys
     ! the heap property, so the remaining elements must be re-heaped.
     il    = indx(k-1)
     ir    = indx(i-1)
     call compare(heap(:,il), heap(:,ir), field,nx,ny,smaller)
     if(smaller) return

     call swapint(indx(i-1), indx(k-1))

     k = i
  enddo

end subroutine heap_grow

subroutine heap_reheap(heap,indx, cap, heap_length, tree_length, field, nx, ny)
  ! builds the heap from the element data using the provided heap function.
  ! at exit, the root node satisfies the heap condition:
  !   hpfun( root_node, node ) = .true. for any other node
  !
  integer (kind=4), intent(in) :: cap
  integer (kind=4), intent(inout) :: heap_length, tree_length
  integer (kind=4), intent(in), dimension(0:1,0:cap-1) :: heap
  integer (kind=4), intent(inout), dimension(0:cap-1) :: indx
  integer (kind=4), intent(in) :: nx, ny
  real (kind=8), intent(in), dimension(0:nx-1,ny-1) :: field
  integer                      :: k

  heap_length = tree_length

  if (cap< heap_length) then
      write(*,*) "Exceeded heap capacity in reheaping"
      return
  end if

  do k = heap_length / 2, 1, -1
     call heap_grow(heap,indx, cap, heap_length, field, nx, ny, k)
  enddo
end subroutine heap_reheap

subroutine swapint( i, k )
  integer (kind=4), intent(inout):: i, k
  integer (kind=4) :: t
  t = i
  i = k
  k = t
end subroutine swapint
