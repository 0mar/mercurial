subroutine coocsr ( nrow, nnz, a, ir, jc, ao, jao, iao )

!*****************************************************************************80
!
!! COOCSR converts COO to CSR.
!
!  Discussion:
!
!    This routine converts a matrix that is stored in COO coordinate format
!    a, ir, jc into a CSR row general sparse ao, jao, iao format.
!
!  Modified:
!
!    07 January 2004
!
!  Author:
!
!    Youcef Saad
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) NROW, the row dimension of the matrix.
!
!    Input, integer ( kind = 4 ) NNZ, the number of nonzero elements.
!
! a,
! ir,
! jc    = matrix in coordinate format. a(k), ir(k), jc(k) store the nnz
!         nonzero elements of the matrix with a(k) = actual real value of
!         the elements, ir(k) = its row number and jc(k) = its column
!        number. The order of the elements is arbitrary.
!
! on return:
!
! ir       is destroyed
!
!    Output, real AO(*), JAO(*), IAO(NROW+1), the matrix in CSR
!    Compressed Sparse Row format.
!
  implicit none

  integer ( kind = 4 ) nrow

  real ( kind = 8 ) a(*)
  real ( kind = 8 ) ao(*)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) iad
  integer ( kind = 4 ) iao(nrow+1)
  integer ( kind = 4 ) ir(*)
  integer ( kind = 4 ) j
  integer ( kind = 4 ) jao(*)
  integer ( kind = 4 ) jc(*)
  integer ( kind = 4 ) k
  integer ( kind = 4 ) k0
  integer ( kind = 4 ) nnz
  real ( kind = 8 ) x

  iao(1:nrow+1) = 0
!
!  Determine the row lengths.
!
  do k = 1, nnz
    iao(ir(k)) = iao(ir(k)) + 1
  end do
!
!  The starting position of each row.
!
  k = 1
  do j = 1, nrow+1
     k0 = iao(j)
     iao(j) = k
     k = k + k0
  end do
!
!  Go through the structure once more.  Fill in output matrix.
!
  do k = 1, nnz
     i = ir(k)
     j = jc(k)
     x = a(k)
     iad = iao(i)
     ao(iad) = x
     jao(iad) = j
     iao(i) = iad + 1
  end do
!
!  Shift back IAO.
!
  do j = nrow, 1, -1
    iao(j+1) = iao(j)
  end do
  iao(1) = 1

  return
end
subroutine getdia ( nrow, ncol, job, a, ja, ia, len, diag, idiag, ioff )

!*****************************************************************************80
!
!! GETDIA extracts a given diagonal from a matrix stored in CSR format.
!
!  Discussion:
!
!    The output matrix may be transformed with the diagonal removed
!    from it if desired (as indicated by job.)
!
!    Our definition of a diagonal of matrix is a vector of length nrow
!    (always) which contains the elements in rows 1 to nrow of
!    the matrix that are contained in the diagonal offset by ioff
!    with respect to the main diagonal. If the diagonal element
!    falls outside the matrix then it is defined as a zero entry.
!    Thus the proper definition of diag(*) with offset ioff is
!
!    diag(k) = a(k,ioff+k) k = 1,2,...,nrow
!    with elements falling outside the matrix being defined as zero.
!
!  Modified:
!
!    07 January 2004
!
!  Author:
!
!    Youcef Saad
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) NROW, the row dimension of the matrix.
!
!    Input, integer ( kind = 4 ) NCOL, the column dimension of the matrix.
!
! job   = integer ( kind = 4 ). Job indicator.  If job = 0 then
!         the matrix a, ja, ia, is not altered on return.
!         if job/=1  then getdia will remove the entries
!         collected in diag from the original matrix.
!         This is done in place.
!
!    Input, real A(*), integer ( kind = 4 ) JA(*), IA(NROW+1), the matrix in CSR
!    Compressed Sparse Row format.
!
! ioff  = integer ( kind = 4 ),containing the offset of the wanted diagonal
!        the diagonal extracted is the one corresponding to the
!        entries a(i,j) with j-i = ioff.
!        thus ioff = 0 means the main diagonal
!
! on return:
!
! len   = number of nonzero elements found in diag.
!         (len <= min ( nrow, ncol-ioff ) - max ( 1, 1-ioff) + 1 )
!
! diag  = real array of length nrow containing the wanted diagonal.
!        diag contains the diagonal (a(i,j),j-i = ioff ) as defined
!         above.
!
! idiag = integer ( kind = 4 ) array of  length len, containing the poisitions
!         in the original arrays a and ja of the diagonal elements
!         collected in diag. A zero entry in idiag(i) means that
!         there was no entry found in row i belonging to the diagonal.
!
! a, ja,
!    ia = if job /= 0 the matrix is unchanged. otherwise the nonzero
!         diagonal entries collected in diag are removed from the
!         matrix. the structure is modified since the diagonal elements
!        are removed from a,ja,ia. Thus, the  returned matrix will
!         have len fewer elements if the diagonal is full.
!
  implicit none

  real ( kind = 8 ) a(*)
  real ( kind = 8 ) diag(*)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ia(*)
  integer ( kind = 4 ) idiag(*)
  integer ( kind = 4 ) iend
  integer ( kind = 4 ) ioff
  integer ( kind = 4 ) istart
  integer ( kind = 4 ) ja(*)
  integer ( kind = 4 ) job
  integer ( kind = 4 ) k
  integer ( kind = 4 ) kdiag
  integer ( kind = 4 ) ko
  integer ( kind = 4 ) kold
  integer ( kind = 4 ) len
  integer ( kind = 4 ) ncol
  integer ( kind = 4 ) nrow

  istart = max ( 0, -ioff )
  iend = min ( nrow, ncol-ioff )
  len = 0
  idiag(1:nrow) = 0
  diag(1:nrow) = 0.0D+00
!
!  Extract the diagonal elements.
!
  do i = istart+1, iend

     do k = ia(i), ia(i+1) -1
        if ( ja(k) - i == ioff ) then
           diag(i) = a(k)
           idiag(i) = k
           len = len + 1
           exit
        end if
     end do

  end do

  if ( job == 0 .or. len == 0 ) then
    return
  end if
!
!  Rewind the structure.
!
  ko = 0

  do i = istart+1, iend

    kold = ko
    kdiag = idiag(i)

    if ( kdiag /= 0 ) then

      do k = ia(i), ia(i+1)-1
        if ( ja(k) /= kdiag ) then
          ko = ko + 1
          a(ko) = a(k)
          ja(ko) = ja(k)
        end if
      end do
      ia(i) = kold + 1
    end if

  end do
!
!  Redefine IA(NROW+1).
!
  ia(nrow+1) = ko + 1

  return
end
subroutine amux ( n, x, y, a, ja, ia )

!*****************************************************************************80
!
!! AMUX multiplies a CSR matrix A times a vector.
!
!  Discussion:
!
!    This routine multiplies a matrix by a vector using the dot product form.
!    Matrix A is stored in compressed sparse row storage.
!
!  Modified:
!
!    07 January 2004
!
!  Author:
!
!    Youcef Saad
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the row dimension of the matrix.
!
!    Input, real X(*), and array of length equal to the column dimension
!    of A.
!
!    Input, real A(*), integer ( kind = 4 ) JA(*), IA(NROW+1), the matrix in CSR
!    Compressed Sparse Row format.
!
!    Output, real Y(N), the product A * X.
!
  implicit none

  integer ( kind = 4 ) n

  real ( kind = 8 ) a(*)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ia(*)
  integer ( kind = 4 ) ja(*)
  integer ( kind = 4 ) k
  real ( kind = 8 ) t
  real ( kind = 8 ) x(*)
  real ( kind = 8 ) y(n)

  do i = 1, n
!
!  Compute the inner product of row I with vector X.
!
    t = 0.0D+00
    do k = ia(i), ia(i+1)-1
      t = t + a(k) * x(ja(k))
    end do

    y(i) = t

  end do

  return
end

subroutine compress_row(n,nnz,coo_rows,compressed_rows)
    implicit none
    ! Input: COO matrix using an (unordered) value array, row array, and column array
    integer (kind=4), intent(in) :: n, nnz
    integer (kind=4), dimension(0:nnz-1), intent(in) :: coo_rows

    ! Output: CSR matrix using an ordered value array, compressed row array and column array
    integer (kind=4), dimension(0:n), intent(out) :: compressed_rows

    ! Subroutine parameters
    integer (kind=4) :: row, nnz_in_row, k

    ! Initialize output arrays
    compressed_rows = 0

    ! Determine nnz elements per row (leaving last element 0)
    do k=0,nnz-1
        compressed_rows(coo_rows(k)) = compressed_rows(coo_rows(k)) + 1
    end do

    ! Compress into nnz elements from previous rows
    ! (for more details, see Wikipedia/Sparse matrix/CSR)
    k=0
    do row=0,n
        nnz_in_row = compressed_rows(row)
        compressed_rows(row) = k
        k = k+nnz_in_row
    end do
end subroutine compress_row

subroutine sparse_matmul(n, nnz, a, ia, ja, x, b)
    implicit none

    ! Matrix parameters
    integer (kind=4), intent(in) :: n, nnz
    real (kind=8), dimension(0:nnz-1), intent(in) :: a
    integer (kind=4), dimension(0:n), intent(in) :: ia
    integer (kind=4), dimension(0:nnz-1), intent(in) :: ja

    ! Vectors
    real (kind=8), dimension(0:n-1), intent(in) :: x
    real (kind=8), dimension(0:n-1), intent(out) :: b

    ! Subroutine variables
    integer (kind=4) :: row, k
    real (kind=8) :: dotp
    b = 0

    do row = 0,n-1
        dotp = 0.d0
        do k=ia(row), ia(row+1)-1
            dotp = dotp + a(k) * x(ja(k))
        end do
        b(row) = dotp
    end do
end subroutine sparse_matmul
