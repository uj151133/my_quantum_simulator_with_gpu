module timer
  use iso_c_binding
  implicit none
  private
  public :: record_time

  integer(c_int), parameter :: CLOCK_MONOTONIC = 6  ! macOS の MONOTONIC ID

  type, bind(C) :: timespec
    integer(c_long) :: tv_sec
    integer(c_long) :: tv_nsec
  end type timespec

  type, bind(C) :: timeval
    integer(c_long) :: tv_sec
    integer(c_long) :: tv_usec
  end type timeval

  interface
     function c_gethostname(name, len) bind(C, name="gethostname")
       import :: c_char, c_int
       character(kind=c_char), dimension(*) :: name
       integer(c_int), value :: len
       integer(c_int) :: c_gethostname
     end function c_gethostname

     function clock_gettime(clk_id, tp) bind(C, name="clock_gettime")
       import :: c_int, c_ptr
       integer(c_int), value :: clk_id
       type(c_ptr), value :: tp
       integer(c_int) :: clock_gettime
     end function clock_gettime

     function gettimeofday(tp, tzp) bind(C, name="gettimeofday")
       import :: c_int, c_ptr
       type(c_ptr), value :: tp
       type(c_ptr), value :: tzp
       integer(c_int) :: gettimeofday
     end function gettimeofday
  end interface

contains

  subroutine record_time(cb) bind(C, name="record_time")
    type(C_FUNPTR), value :: cb
    procedure(), pointer  :: fptr
    type(timespec), target :: m0, m1
    type(timeval),  target :: w0, w1
    integer :: rc_m0, rc_m1, rc_w0, rc_w1
    real(8) :: elapsed_ms
    real(8) :: sec_diff, nsec_diff
    integer :: ios, u
    character(len=64)  :: ts
    character(len=256) :: hostname
    character(len=256) :: branch

    call c_f_procpointer(cb, fptr)

    ! Monotonic 計測 (優先)
    rc_m0 = clock_gettime(CLOCK_MONOTONIC, c_loc(m0))
    call fptr()
    rc_m1 = clock_gettime(CLOCK_MONOTONIC, c_loc(m1))

    if (rc_m0 == 0 .and. rc_m1 == 0) then
       sec_diff  = dble(m1%tv_sec - m0%tv_sec)
       nsec_diff = dble(m1%tv_nsec - m0%tv_nsec)
       elapsed_ms = sec_diff * 1000.0d0 + nsec_diff / 1.0d6
    else
       ! フォールバック: gettimeofday
       rc_w0 = gettimeofday(c_loc(w0), c_null_ptr)
       call fptr()
       rc_w1 = gettimeofday(c_loc(w1), c_null_ptr)
       if (rc_w0 == 0 .and. rc_w1 == 0) then
          elapsed_ms = dble(w1%tv_sec - w0%tv_sec) * 1000.0d0 + &
                       dble(w1%tv_usec - w0%tv_usec) / 1000.0d0
       else
          elapsed_ms = 0.0d0
       end if
    end if

    if (elapsed_ms < 0.0d0) elapsed_ms = 0.0d0  ! 負値ガード

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
    integer :: rc, i
    name = 'unknown'
    buf = c_null_char
    rc = c_gethostname(buf, size(buf))
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