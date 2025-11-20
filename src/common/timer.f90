module timer
  use iso_c_binding
  implicit none
  private
  public :: record_time

  interface
     function c_gethostname(name, len) bind(C, name="gethostname")
       import :: c_char, c_int
       implicit none
       character(kind=c_char), dimension(*) :: name
       integer(c_int), value :: len
       integer(c_int) :: c_gethostname
     end function c_gethostname
  end interface

contains

  subroutine record_time(cb) bind(C, name="record_time")
    type(C_FUNPTR), value :: cb
    procedure(), pointer :: fptr
    integer :: c0, c1, rate
    real(8) :: elapsed_ms
    integer :: ios, u
    character(len=64)  :: ts
    character(len=255) :: hostname
    character(len=255) :: branch

    call c_f_procpointer(cb, fptr)

    call system_clock(count_rate=rate)
    call system_clock(count=c0)
    call fptr()
    call system_clock(count=c1)
    elapsed_ms = dble(c1 - c0) / dble(rate) * 1000.0d0

    call get_timestamp(ts)
    call get_hostname(hostname)
    call get_git_branch(branch)

    write(*,'(A,F0.3,A)') achar(27)//'[1;32mExecution time: ', elapsed_ms, ' ms'//achar(27)//'[0m'

    open(newunit=u, file='record.log', status='unknown', action='write', position='append', iostat=ios)
    if (ios == 0) then
      write(u,'(A,1X,A,1X,A,1X,A,F0.3,A)') &
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