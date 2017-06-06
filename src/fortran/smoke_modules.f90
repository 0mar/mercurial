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
