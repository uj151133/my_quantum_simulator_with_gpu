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
  end interface

contains

  subroutine record_time(cb) bind(C, name="record_time")
    type(C_FUNPTR), value :: cb
    procedure(), pointer :: fptr
    type(timespec) :: t0, t1
    integer(c_int), parameter :: candidates(3) = [6_c_int, 4_c_int, 1_c_int]
    integer :: i, rc0, rc1, clk_id
    real(8) :: elapsed_ms
    integer :: rate, cstart, cend
    integer :: ios, u
    character(len=64)  :: ts
    character(len=256) :: hostname
    character(len=256) :: branch

    call c_f_procpointer(cb, fptr)

    clk_id = -1
    do i = 1, size(candidates)
       rc0 = clock_gettime(candidates(i), t0)
       if (rc0 == 0) then
          clk_id = candidates(i)
          exit
       end if
    end do

    if (clk_id /= -1) then
       call fptr()
       rc1 = clock_gettime(clk_id, t1)
       if (rc1 == 0) then
          elapsed_ms = dble(t1%tv_sec - t0%tv_sec) * 1000.0d0 + &
                       dble(t1%tv_nsec - t0%tv_nsec) / 1.0d6
       else
          elapsed_ms = 0.0d0
       end if
    else
       call system_clock(count_rate=rate)
       call system_clock(count=cstart)
       call fptr()
       call system_clock(count=cend)
       elapsed_ms = dble(cend - cstart) / dble(rate) * 1000.0d0
    end if

    if (elapsed_ms < 0d0) elapsed_ms = 0d0

    call get_timestamp(ts)
    call get_hostname(hostname)
    call get_git_branch(branch)

    write(*,'(A,F0.6,A)') achar(27)//'[1;32mExecution time: ', elapsed_ms, ' ms'//achar(27)//'[0m'

    open(newunit=u, file='record.log', status='unknown', action='write', position='append', iostat=ios)
    if (ios == 0) then
       write(u,'(A,1X,A,1X,A,1X,A,F0.6,A)') &
         '['//trim(ts)//']', 'Host: '//trim(hostname)//' |', 'Branch: '//trim(branch)//' |', &
         'Execution time: ', elapsed_ms, ' ms'
       close(u)
    end if
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