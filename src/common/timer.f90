module timer
  use iso_c_binding
  implicit none
  private
  public :: record_time

  type, bind(C) :: timespec
    integer(c_long) :: tv_sec
    integer(c_long) :: tv_nsec
  end type timespec

  interface
     function clock_gettime(clk_id, tp) bind(C, name="clock_gettime")
       import :: c_int, timespec
       integer(c_int), value :: clk_id
       type(timespec), intent(out) :: tp
       integer(c_int) :: clock_gettime
     end function clock_gettime
     function c_gethostname(name, len) bind(C, name="gethostname")
       import :: c_char, c_size_t, c_int
       character(kind=c_char), dimension(*) :: name
       integer(c_size_t), value :: len
       integer(c_int) :: c_gethostname
     end function c_gethostname
     function gettimeofday(tp, tzp) bind(C, name="gettimeofday")
       import :: c_int, c_ptr
       type(c_ptr), value :: tp
       type(c_ptr), value :: tzp
       integer(c_int) :: gettimeofday
     end function gettimeofday
  end interface

  type, bind(C) :: timeval
    integer(c_long) :: tv_sec
    integer(c_long) :: tv_usec
  end type timeval

contains

  subroutine record_time(cb, elapsed_ms_ptr) bind(C, name="record_time")
    use iso_c_binding
    type(C_FUNPTR), value :: cb
    procedure(), pointer :: fptr
    real(c_double), intent(out) :: elapsed_ms_ptr
    integer(c_int), parameter :: CLK(3) = [4_c_int, 1_c_int, 6_c_int] ! Linux RAW, Linux MONOTONIC, mac
    type(timespec) :: t0, t1
    type(timeval), target :: w0, w1
    real(8) :: elapsed_ns, elapsed_ms
    real(8) :: elapsed_ns_tv, elapsed_ms_tv
    integer :: i, rc0, rc1
    integer :: rate, cstart, cend
    integer :: ios, u
    character(len=64)  :: ts
    character(len=256) :: hostname
    character(len=256) :: branch

    call c_f_procpointer(cb, fptr)

    elapsed_ns = -1.0d0
    ! 高精度クロック探索
    do i = 1, size(CLK)
       rc0 = clock_gettime(CLK(i), t0)
       if (rc0 == 0) then
          call fptr()
          rc1 = clock_gettime(CLK(i), t1)
          if (rc1 == 0) then
             elapsed_ns = dble(t1%tv_sec - t0%tv_sec)*1.0d9 + dble(t1%tv_nsec - t0%tv_nsec)
             exit
          end if
       end if
    end do

    if (elapsed_ns < 0d0) then
       ! フォールバック: system_clock
       call system_clock(count_rate=rate)
       call system_clock(count=cstart)
       call fptr()
       call system_clock(count=cend)
       elapsed_ns = dble(cend - cstart)/dble(rate)*1.0d9
    end if

    elapsed_ms = elapsed_ns / 1.0d6

    ! 下3桁が常に 000 (マイクロ秒精度未満) なら gettimeofday で再取得
    if (abs(elapsed_ms - (floor(elapsed_ms*1000.0d0)/1000.0d0)) < 1.0d-9) then
       rc0 = gettimeofday(c_loc(w0), c_null_ptr)
       call fptr()
       rc1 = gettimeofday(c_loc(w1), c_null_ptr)
       if (rc0 == 0 .and. rc1 == 0) then
          elapsed_ns_tv = dble(w1%tv_sec - w0%tv_sec)*1.0d9 + dble(w1%tv_usec - w0%tv_usec)*1.0d3
          if (elapsed_ns_tv > 0d0) then
             elapsed_ms_tv = elapsed_ns_tv / 1.0d6
             ! より細かい差分が得られたら置き換え
             if (abs(elapsed_ms_tv - elapsed_ms) > 1.0d-6) elapsed_ms = elapsed_ms_tv
          end if
       end if
    end if

    if (elapsed_ms < 0d0) elapsed_ms = 0d0

    call get_timestamp(ts)
    call get_hostname(hostname)
    call get_git_branch(branch)

    write(*,'(A,F0.6," ms",A)') achar(27)//'[1;32mExecution time: ', elapsed_ms, achar(27)//'[0m'

    open(newunit=u, file='record.log', status='unknown', action='write', position='append', iostat=ios)
    if (ios == 0) then
       write(u,'(A,1X,A,1X,A,1X,A,F0.6," ms")') &
         '['//trim(ts)//']','Host: '//trim(hostname)//' |','Branch: '//trim(branch)//' |', &
         'Execution time:', elapsed_ms
       close(u)
    end if
    elapsed_ms_ptr = elapsed_ms
  end subroutine record_time

  subroutine get_timestamp(ts)
    character(len=*), intent(out) :: ts
    integer :: v(8)
    call date_and_time(values=v)
    write(ts,'(I4.4,"-",I2.2,"-",I2.2," ",I2.2,":",I2.2,":",I2.2)') v(1),v(2),v(3),v(5),v(6),v(7)
  end subroutine get_timestamp

  subroutine get_hostname(name)
    character(len=*), intent(out) :: name
    character(kind=c_char), dimension(256) :: buf
    integer(c_int) :: rc
    integer :: i
    integer(c_size_t) :: nlen
    name = 'unknown'
    buf = c_null_char
    nlen = size(buf, kind=c_size_t)
    rc = c_gethostname(buf, nlen)
    if (rc == 0) then
       name = ''
       do i = 1, size(buf)
          if (buf(i) == c_null_char) exit
          if (i <= len(name)) name(i:i) = transfer(buf(i), ' ')
       end do
       if (i > 1) name = adjustl(name(1:i-1))
    end if
  end subroutine get_hostname

  subroutine get_git_branch(branch)
    character(len=*), intent(out) :: branch
    integer :: u, ios, p
    character(len=512) :: line
    branch = 'unknown'
    open(newunit=u, file='.git/HEAD', status='old', action='read', iostat=ios)
    if (ios /= 0) return
    read(u,'(A)', iostat=ios) line
    close(u)
    if (ios /= 0) return
    p = index(line, 'refs/heads/')
    if (p > 0) then
       branch = adjustl(trim(line(p+11:)))
    else
       branch = 'detached'
    end if
  end subroutine get_git_branch

end module timer