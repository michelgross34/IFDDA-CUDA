PROGRAM main
  use precision
  use iso_fortran_env, only: int64
  implicit none
  integer :: geometry_case, beam_case, nrig_case, method_case, precon_case, init_case, option_case
  integer :: arg_index, illumination_case
  integer :: seed_size, seed_index
  integer, allocatable :: seed(:)
  character(len=32) :: geometry_arg, beam_arg, nrig_arg, method_arg, precon_arg, init_arg, option_arg
  character(len=64) :: chosen_geometry, cli_first, illumination_arg
  character(len=64) :: epsilon_override_env
  character(len=64) :: cli_arbitrary_file
  real(wp) :: epsilon_override_real, epsilon_override_imag
  integer(int64) :: geometry_mask_hash
  character(len=8) :: geometry_only_env
  character(len=32) :: runtime_key
  character(len=128) :: runtime_value, runtime_epsilon
  character(len=12) :: runtime_solver
  logical :: cli_grid_set, cli_grid_dimensions_set, cli_wavelength_set, cli_epsilon_set
  logical :: cli_tol_set, cli_maxit_set, cli_precon_set, cli_init_set
  logical :: cli_nrig_set, cli_nfft_set, cli_power_set, cli_nquad_set
  logical :: cli_section_set, cli_scattering_set, cli_force_set
  logical :: cli_torque_set, cli_energy_set, cli_local_set, cli_macro_set
  logical :: cli_theta_set, cli_phi_set, cli_pp_set, cli_ss_set, cli_waist_set, cli_seed_set
  logical :: cli_microscope_set, cli_micro_type_set, cli_side_set, cli_quick_lens_set
  logical :: cli_near_domain_set, cli_geometry_only_set, cli_save_mode_set, cli_quick_diff_set
  logical :: cli_na_set, cli_lens_z_set, cli_polarizability_set
  logical :: cli_radius_set, cli_side_length_set
  logical :: cli_dimensions_set, cli_angles_set, cli_density_set, cli_cell_size_set
  logical :: cli_force_density_set, cli_torque_density_set
  logical :: cli_isotropy_set
  logical :: cli_ntheta_set, cli_nphi_set, cli_gaussian_center_set, cli_geometry_center_set
  logical :: cli_illumination_na_set, cli_condenser_na_set, cli_inner_na_set
  logical :: cli_na_center_set, cli_magnification_set
  logical :: cli_padding_set
  logical :: cli_background_epsilon_set, cli_abc_type_set, cli_abc_nrbc_set, cli_abc_cos_set
  logical :: cli_read_field_set, cli_field_file_set, cli_object_file_set, cli_hdf5_file_set
  logical :: cli_arbitrary_file_set
  integer :: cli_grid, cli_grid_dimensions(3), cli_maxit, cli_precon, cli_init, cli_nrig, cli_nfft, cli_nquad
  integer :: cli_section, cli_scattering, cli_force, cli_torque, cli_energy, cli_local, cli_macro
  real(wp) :: cli_wavelength, cli_tolerance, cli_power, cli_epsilon_real, cli_epsilon_imag
  real(wp) :: cli_theta, cli_phi, cli_pp, cli_ss, cli_waist
  integer :: cli_seed
  integer :: cli_microscope, cli_micro_type, cli_side, cli_quick_lens, cli_near_domain
  integer :: cli_geometry_only, cli_save_mode, cli_quick_diff
  integer :: cli_force_density, cli_torque_density
  real(wp) :: cli_na, cli_lens_z
  real(wp) :: cli_radius, cli_side_length
  real(wp) :: cli_geometry_dimensions(3), cli_geometry_angles(3), cli_density, cli_cell_size
  character(len=2) :: cli_polarizability
  character(len=3) :: cli_isotropy
  character(len=64) :: cli_field_file, cli_object_file
  character(len=100) :: cli_hdf5_file
  integer :: cli_read_field
  integer :: cli_ntheta, cli_nphi, cli_abc_type
  integer :: cli_padding(3)
  real(wp) :: cli_gaussian_center(3), cli_geometry_center(3), cli_na_center(2)
  real(wp) :: cli_illumination_na, cli_condenser_na, cli_inner_na, cli_magnification
  real(wp) :: cli_background_epsilon, cli_abc_nrbc, cli_abc_cos

  ! Integer variables
  integer :: ii,jj,kk,i,j,k,nstop
  integer :: nlocal,nmacro,nsection,nsectionsca,nforce,nforced, &
             ntorque,ntorqued,nsens,nproche,nlecture,nquickdiffracte,nrig, nquad,nenergie,nmat,nprecon,ninitest

  ! Variables for the object
  integer :: nbsphere3,nbsphere,ndipole,test,numberobjetmax, numberobjet,ierror,ntypeabc
  parameter (numberobjetmax=2)
  integer :: nx,ny,nz,nx2,ny2,nxy2,nz2,nxm,nym,nzm,nxmp,nymp,nzmp, &
             ntotal,nmaxpp,nxym1,nzm1,nzms1
  integer :: subunit,nphi,ntheta
  real(wp) eps0i,lnrbc,lcos
  
  ! Variables for positions
  integer :: ng,speckseed
  real(wp) :: x0,y0,z0,x,y,z,xx0,yy0,zz0,rayon,density,side, &
                      sidex,sidey,sidez,hauteur,xgmulti(numberobjetmax), &
                      ygmulti(numberobjetmax),zgmulti(numberobjetmax), &
                      rayonmulti(numberobjetmax),demiaxea,demiaxeb,demiaxec, &
                      thetaobj,phiobj,psiobj,lc,hc,aretecube
  real(wp), allocatable :: xs(:),ys(:),zs(:)
  real(wp), allocatable :: xswf(:),yswf(:),zswf(:)
  real(wp) :: lambda,lambda10n,k0,k03,epi,epr

  ! Material variables
  integer :: niso
  real(wp) :: eps0
  complex(wp), allocatable :: polarisa(:,:,:),epsilon(:,:,:)
  complex(wp) :: epsmulti(numberobjetmax),epsanimulti(3,3,numberobjetmax)
  character(2) :: polarizability
  character(64) :: materiaumulti(numberobjetmax)
  character(64) :: materiau,object,beam,namefileobj,namefileinc,filereread
  character(3) :: trope


  ! Incident and local fields
  real(wp), allocatable :: incidentfield(:),localfield(:)
  real(wp), allocatable :: macroscopicfield(:)
  real(wp), allocatable :: forcex(:),forcey(:),forcez(:)
  real(wp), allocatable :: torquex(:),torquey(:),torquez(:)
  real(wp) :: forcexmulti(numberobjetmax),forceymulti(numberobjetmax), forcezmulti(numberobjetmax),torquexmulti(numberobjetmax),&
       & torqueymulti(numberobjetmax),torquezmulti(numberobjetmax)
  real(wp) :: ss,pp,theta,phi,I0
  integer :: nbinc
  real(wp) :: thetam(10),phim(10),ppm(10),ssm(10)
  complex(wp) :: E0m(10)
  complex(wp) :: Eloc(3),Em(3),E0
  complex(wp), allocatable :: macroscopicfieldx(:)
  complex(wp), allocatable :: macroscopicfieldy(:)
  complex(wp), allocatable :: macroscopicfieldz(:)
  complex(wp), allocatable :: localfieldx(:)
  complex(wp), allocatable :: localfieldy(:)
  complex(wp), allocatable :: localfieldz(:)
  complex(wp), allocatable :: incidentfieldx(:)
  complex(wp), allocatable :: incidentfieldy(:)
  complex(wp), allocatable :: incidentfieldz(:)
  complex(wp) :: propaesplibre(3,3)
  complex(wp), allocatable :: FF(:),FF0(:),FFloc(:)
  complex(wp), dimension(:,:,:), allocatable :: sdetnn
  complex(wp), dimension(:), allocatable :: FFprecon
  complex(wp), allocatable :: xr(:),xi(:)
  complex(wp), allocatable :: wrk(:,:)

  ! Green function
  integer, allocatable :: Tabdip(:),Tabmulti(:)
  integer :: indice
  complex(wp), allocatable :: FFTTENSORxx(:), FFTTENSORxy(:),FFTTENSORxz(:),FFTTENSORyy(:),FFTTENSORyz(:), FFTTENSORzz(:)

  ! Computation variables
  real(wp) :: forcet(3),forcem,forcemie
  real(wp) :: couplet(3),couplem
  real(wp) :: forcearrmax,torquearrmax
  complex(wp) :: Eder(3,3)
  integer :: iphi,itheta
  real(wp) :: MIECEXT,MIECABS,MIECSCA,GSCA,Cext,normal(3), deltatheta,deltaphi,Csca,Cscai,Cabs,gasym,thetas,phis, efficacite&
       &,efficaciteref,efficacitetrans
  complex(wp) :: ctmp

  ! Iterative method variables
  integer :: ldabi,nlar,nnnr,ncompte,NLIM,ndim,nou,maxit,nstat,nloop,STEPERR
  real(wp) :: NORM,TOL,norm1,norm2,tolinit,tol1,tempsreelmvp,tempsreeltotal
  real(wp) :: fieldmax,localmax
  complex(wp) :: ALPHA,BETA,GPETA,DZETA,R0RN

  ! Poynting vector
  integer :: nr,nrmax,nw,nwmax
  real(wp) :: Poyntinginc
  character(64) :: infostr

  ! New variables
  real(wp) :: P0,w0,xgaus,ygaus,zgaus
  character(12) :: methodeit
  real(wp) :: irra
  integer :: nloin,na

  ! Microscope variables
  integer :: nlentille,nobjet,nfft2d,nfft2d2,nquicklens,ntypemic,nside,npoynting
  real(wp), allocatable :: thetafield(:),phifield(:)
  real(wp), allocatable :: poyntingfield(:)
  real(wp), allocatable :: poyntingfieldpos(:)
  real(wp), allocatable :: poyntingfieldneg(:)
  integer, allocatable :: tabfft2(:)
  
  real(wp) :: kx,ky,kz,deltakx,deltaky,numaper,deltax,gross, numaperinc,numaperinc2,kcnax,kcnay,zlens,deltapoyntingx,deltapoyntingy&
       &, numaperil,psiinc,zstep,zinf,zsup
  real(wp), allocatable :: kxy(:),xy(:),kxypoynting(:)
  complex(wp), allocatable :: Eimagex(:),Eimagey(:),Eimagez(:),Eimageincx(:),Eimageincy(:) ,Eimageincz(:)
  complex(wp), allocatable :: Efourierx(:) ,Efouriery(:),Efourierz(:),Efourierincx(:),Efourierincy(:) ,Efourierincz(:)
  complex(wp), allocatable :: Ediffkzpos(:,:,:) ,Ediffkzneg(:,:,:)

  character(len=100) :: h5file
  integer :: alloc_size

! DATA INPUT
  lambda = 632.8_wp ! wavelength
  P0 = one ! power
  w0 = lambda * 10.0_wp ! waist
  eps0i=zero
  ntypeabc=0
  lnrbc=zero
  lcos=zero
  speckseed = 0
  ! Verified no-argument regression case: keep this baseline aligned across
  ! Fortran and CUDA builds: 32^3 sphere, plane wave, m=1.6+0i, GPBICG1,
  ! no preconditioner.
  geometry_case = 1
  beam_case = 200
  nrig_case = 0
  method_case = 24 ! GPBICG1 (same no-argument reference as the CUDA build)
  precon_case = 0
  init_case = 1
  option_case = 0
  geometry_arg = '1'
  beam_arg = '200'
  nrig_arg = '0'
  method_arg = '0'
  precon_arg = '0'
  init_arg = '1'
  option_arg = '0'
  cli_grid_set=.false.; cli_grid_dimensions_set=.false.
  cli_wavelength_set=.false.; cli_epsilon_set=.false.
  cli_tol_set=.false.; cli_maxit_set=.false.; cli_precon_set=.false.; cli_init_set=.false.
  cli_nrig_set=.false.; cli_nfft_set=.false.; cli_power_set=.false.; cli_nquad_set=.false.
  cli_section_set=.false.; cli_scattering_set=.false.; cli_force_set=.false.
  cli_torque_set=.false.; cli_energy_set=.false.; cli_local_set=.false.; cli_macro_set=.false.
  cli_theta_set=.false.; cli_phi_set=.false.; cli_pp_set=.false.; cli_ss_set=.false.
  cli_waist_set=.false.; cli_seed_set=.false.
  cli_microscope_set=.false.; cli_micro_type_set=.false.; cli_side_set=.false.
  cli_quick_lens_set=.false.; cli_near_domain_set=.false.; cli_geometry_only_set=.false.
  cli_save_mode_set=.false.; cli_quick_diff_set=.false.; cli_na_set=.false.
  cli_lens_z_set=.false.; cli_polarizability_set=.false.
  cli_radius_set=.false.; cli_side_length_set=.false.
  cli_dimensions_set=.false.; cli_angles_set=.false.; cli_density_set=.false.; cli_cell_size_set=.false.
  cli_force_density_set=.false.; cli_torque_density_set=.false.
  cli_isotropy_set=.false.; cli_isotropy='iso'
  cli_ntheta_set=.false.; cli_nphi_set=.false.
  cli_gaussian_center_set=.false.; cli_geometry_center_set=.false.
  cli_illumination_na_set=.false.; cli_condenser_na_set=.false.; cli_inner_na_set=.false.
  cli_na_center_set=.false.; cli_magnification_set=.false.
  cli_padding_set=.false.; cli_padding=0
  cli_background_epsilon_set=.false.; cli_abc_type_set=.false.
  cli_abc_nrbc_set=.false.; cli_abc_cos_set=.false.
  cli_read_field_set=.false.; cli_field_file_set=.false.
  cli_object_file_set=.false.; cli_hdf5_file_set=.false.
  cli_field_file=''; cli_object_file=''; cli_hdf5_file=''
  cli_arbitrary_file_set=.false.; cli_arbitrary_file=''
  runtime_solver=''; runtime_epsilon=''
  ! Legacy positional arguments are only read when the first token is not
  ! a named option. Otherwise e.g. --preconditioner 1 was misread as
  ! geometry=--preconditioner, beam=1 before the named-option parser ran.
  call get_command_argument(1, cli_first)
  if (len_trim(cli_first) == 0 .or. cli_first(1:2) /= '--') then
     if (command_argument_count() >= 1) call get_command_argument(1, geometry_arg)
     if (command_argument_count() >= 2) call get_command_argument(2, beam_arg)
     if (command_argument_count() >= 3) call get_command_argument(3, nrig_arg)
     if (command_argument_count() >= 4) call get_command_argument(4, method_arg)
     if (command_argument_count() >= 5) call get_command_argument(5, precon_arg)
     if (command_argument_count() >= 6) call get_command_argument(6, init_arg)
     if (command_argument_count() >= 7) call get_command_argument(7, option_arg)
  end if
  ! Named geometry option for the shared Fortran/C99/CPP/CUDA comparison.
  ! Keep all existing numeric positional arguments unchanged.
  call get_command_argument(1, cli_first)
  if (trim(cli_first)=='--help' .or. trim(cli_first)=='-h') then
     print '(A)', 'Usage: test0_double [GEOMETRY_ID BEAM_ID NRIG_ID ...]'
     print '(A)', '       test0_double --geometry NAME'
     print '(A)', '       test0_double --geometry NAME --illumination NAME'
     print '(A)', 'No-argument default: 32^3 sphere, pwavelinear, epsilon=1.6+0i, GPBICG1, preconditioner=0'
     print '(A)', 'Illuminations: pwavelinear pwavecircular wavelinearmulti antenna'
     print '(A)', '               gparawavelinear gparawavecircular gwavelinear gwavecircular'
     print '(A)', '               gfftwavelinear gfftwavecircular arbitrary greentensor'
     print '(A)', '               speckle demispeckle confocal demiconfocal'
     print '(A)', 'Geometry: concentricsphere sphere cube cuboid1 cuboid2 inhomosphere'
     print '(A)', '          inhomocuboid1 inhomocuboid2 ellipsoid nspheres cylinder'
     print '(A)', '          randomsphere1 randomsphere2 arbitrary randomspheremulti1'
     print '(A)', 'Options: --grid-size N --epsilon REAL,IMAG --wavelength NM --power W'
     print '(A)', '         --method-case ID (0..24) --option-case ID (0..28)'
     print '(A)', '         --solver NAME --preconditioner N --initial-guess N --nrig N'
     print '(A)', '         Solvers: gpbicg1 gpbicg2 gpbicgplus gpbicgsafe gpbicgar gpbicgar2'
     print '(A)', '                  bornserie (without preconditioner)'
     print '(A)', '                  gpbicgstarplus gpbicor gcors2 cors qmrcla ssbicgsafe2 tfqmr zcg pimzbicgstab qmrbicgstab1 qmrbicgstab2'
     print '(A)', '                  idrs2 idrs4 idrs8 bicglstab2 bicglstab4 bicglstab8'
     print '(A)', '                  gpbicgstabl2 gpbicgstabl4 gpbicgstabl8'
     print '(A)', '         --tolerance X --max-iterations N --nfft2d N --nquad N'
     print '(A)', '         --illumination NAME --theta DEG --phi DEG --pp X --ss X --waist NM'
     print '(A)', '         --scattering-theta-points N --scattering-phi-points N'
     print '(A)', '         --geometry-dimensions A,B,C --geometry-angles A,B,C --geometry-center X,Y,Z'
     print '(A)', '         --gaussian-center X,Y,Z --illumination-na X --condenser-na X --inner-na X'
     print '(A)', '         --na-center X,Y --magnification X --background-epsilon X'
     print '(A)', '         --abc-type N --abc-nrbc X --abc-cos X'
     print '(A)', '         --near-padding X,Y,Z (with --near-domain 2)'
     print '(A)', '         --grid-dimensions NX,NY,NZ for non-cubic meshes'
     stop
  end if
  read(geometry_arg, *, iostat=ierror) geometry_case
  if (ierror /= 0) geometry_case = 1
  read(beam_arg, *, iostat=ierror) beam_case
  if (ierror /= 0) beam_case = 200
  read(nrig_arg, *, iostat=ierror) nrig_case
  if (ierror /= 0 .or. nrig_case < 0 .or. nrig_case > 15) nrig_case = 0
  read(method_arg, *, iostat=ierror) method_case
  if (ierror /= 0 .or. method_case < 0 .or. method_case > 24) method_case = 0
  read(precon_arg, *, iostat=ierror) precon_case
  if (ierror /= 0 .or. precon_case < 0 .or. precon_case > 2) precon_case = 0
  read(init_arg, *, iostat=ierror) init_case
  if (ierror /= 0 .or. init_case < 0 .or. init_case > 5) init_case = 1
  read(option_arg, *, iostat=ierror) option_case
  if (ierror /= 0 .or. option_case < 0 .or. option_case > 28) option_case = 0
  chosen_geometry=''
  if (trim(cli_first)=='--geometry') then
     if (command_argument_count()<2) then
        write(*,*) 'Missing geometry after --geometry'
        stop 2
     end if
     call get_command_argument(2,chosen_geometry)
  else if (index(cli_first,'--geometry=')==1) then
     chosen_geometry=cli_first(12:)
  end if
  if (len_trim(chosen_geometry)>0) then
     ! A named geometry overrides only the geometry selection. Keep every
     ! other setting at the same defaults used by a no-argument run; later
     ! command-line options may override those defaults explicitly.
     select case (trim(chosen_geometry))
     case ('concentricsphere','concentric','0'); geometry_case=0
     case ('sphere','1'); geometry_case=1
     case ('cube','2'); geometry_case=2
     case ('cuboid1','3'); geometry_case=3
     case ('cuboid2','4'); geometry_case=4
     case ('inhomosphere','5'); geometry_case=5
     case ('inhomocuboid1','6'); geometry_case=6
     case ('inhomocuboid2','7'); geometry_case=7
     case ('ellipsoid','8'); geometry_case=8
     case ('nspheres','9'); geometry_case=9
     case ('cylinder','10'); geometry_case=10
     case ('randomsphere1','11'); geometry_case=11
     case ('randomsphere2','12'); geometry_case=12
     case ('arbitrary','13'); geometry_case=13
     case ('randomspheremulti1','randommulti','14'); geometry_case=14
     case default
        write(*,*) 'Unknown geometry: ',trim(chosen_geometry)
        stop 2
     end select
  else if (trim(cli_first)=='--geometry' .or. index(cli_first,'--geometry=')==1) then
     write(*,*) 'Empty geometry name'
     stop 2
  end if

  ! Optional named illumination. Numeric BEAM_ID remains supported above.
  ! The named option takes precedence so Fortran and CUDA tests can share it.
  illumination_case = -1
  arg_index = 1
  do while (arg_index <= command_argument_count())
     call get_command_argument(arg_index, illumination_arg)
     if (trim(illumination_arg) == '--illumination') then
        if (arg_index == command_argument_count()) then
           write(*,*) 'Missing illumination after --illumination'
           stop 2
        end if
        arg_index = arg_index + 1
        call get_command_argument(arg_index, illumination_arg)
        select case (trim(illumination_arg))
        case ('pwavelinear');       illumination_case = 200
        case ('pwavecircular');      illumination_case = 201
        case ('wavelinearmulti');    illumination_case = 202
        case ('antenna');            illumination_case = 203
        case ('gparawavelinear');    illumination_case = 204
        case ('gparawavecircular');  illumination_case = 205
        case ('gwavelinear');        illumination_case = 206
        case ('gwavecircular');      illumination_case = 207
        case ('arbitrary');          illumination_case = 208
        case ('gfftwavelinear');     illumination_case = 209
        case ('gfftwavecircular');   illumination_case = 210
        case ('greentensor');        illumination_case = 211
        case ('speckle');            illumination_case = 212
        case ('demispeckle');        illumination_case = 213
        case ('confocal');           illumination_case = 214
        case ('demiconfocal');       illumination_case = 215
        case default
           write(*,*) 'Unknown illumination: ',trim(illumination_arg)
           stop 2
        end select
     else if (index(illumination_arg,'--illumination=') == 1) then
        illumination_arg = illumination_arg(16:)
        select case (trim(illumination_arg))
        case ('pwavelinear');       illumination_case = 200
        case ('pwavecircular');      illumination_case = 201
        case ('wavelinearmulti');    illumination_case = 202
        case ('antenna');            illumination_case = 203
        case ('gparawavelinear');    illumination_case = 204
        case ('gparawavecircular');  illumination_case = 205
        case ('gwavelinear');        illumination_case = 206
        case ('gwavecircular');      illumination_case = 207
        case ('arbitrary');          illumination_case = 208
        case ('gfftwavelinear');     illumination_case = 209
        case ('gfftwavecircular');   illumination_case = 210
        case ('greentensor');        illumination_case = 211
        case ('speckle');            illumination_case = 212
        case ('demispeckle');        illumination_case = 213
        case ('confocal');           illumination_case = 214
        case ('demiconfocal');       illumination_case = 215
        case default
           write(*,*) 'Unknown illumination: ',trim(illumination_arg)
           stop 2
        end select
     end if
     arg_index = arg_index + 1
  end do
  if (illumination_case >= 0) beam_case = illumination_case

  ! Named runtime overrides shared with the CUDA driver.  Positional
  ! GEOMETRY_ID/BEAM_ID/... arguments remain accepted for older test scripts.
  arg_index=1
  do while (arg_index <= command_argument_count())
     runtime_key=''; runtime_value=''
     call get_command_argument(arg_index,runtime_key)
     if (index(runtime_key,'--') == 1) then
        if (trim(runtime_key)=='--geometry' .or. trim(runtime_key)=='--illumination' .or. &
            index(runtime_key,'--geometry=')==1 .or. index(runtime_key,'--illumination=')==1) then
           if(index(runtime_key,'=')==0)then
              arg_index=arg_index+2
           else
              arg_index=arg_index+1
           end if
           cycle
        end if
        if (index(runtime_key,'=') > 0) then
           runtime_value=runtime_key(index(runtime_key,'=')+1:)
           runtime_key=runtime_key(:index(runtime_key,'=')-1)
        else
           if (arg_index == command_argument_count()) then
              write(*,*) 'Missing value after ',trim(runtime_key)
              stop 2
           end if
           arg_index=arg_index+1
           call get_command_argument(arg_index,runtime_value)
        end if
        ierror=0
        select case(trim(runtime_key))
        case('--method-case')
           read(runtime_value,*,iostat=ierror) method_case
           if(ierror==0)then
              if(method_case<0.or.method_case>24)ierror=1
           end if
        case('--option-case')
           read(runtime_value,*,iostat=ierror) option_case
           if(ierror==0)then
              if(option_case<0.or.option_case>28)ierror=1
           end if
        case('--solver')
           select case(trim(runtime_value))
           case('gpbicg','gpbicg1'); runtime_solver='GPBICG1'
           case('bornserie'); runtime_solver='BORNSERIE'
           case('gpbicg2'); runtime_solver='GPBICG2'
           case('gpbicgplus'); runtime_solver='GPBICGplus'
           case('gpbicgsafe'); runtime_solver='GPBICGsafe'
           case('gpbicgar'); runtime_solver='GPBICGAR'
           case('gpbicgar2'); runtime_solver='GPBICGAR2'
           case('gpbicgstarplus','bicgstarplus'); runtime_solver='BICGSTARPLUS'
           case('gpbicor'); runtime_solver='GPBICOR'
           case('gcors2'); runtime_solver='GCORS2'
           case('cors'); runtime_solver='CORS'
           case('qmrcla','pimzqmr'); runtime_solver='QMRCLA'
           case('ssbicgsafe2'); runtime_solver='SSBICGSAFE2'
           case('tfqmr'); runtime_solver='TFQMR'
           case('zcg'); runtime_solver='CG'
           case('pimzbicgstab'); runtime_solver='BICGSTAB'
           case('qmrbicgstab','qmrbicgstab1'); runtime_solver='QMRBICGSTAB1'
           case('qmrbicgstab2'); runtime_solver='QMRBICGSTAB2'
           case('idrs2'); runtime_solver='IDRS2'
           case('idrs4'); runtime_solver='IDRS4'
           case('idrs8'); runtime_solver='IDRS8'
           case('bicglstab2'); runtime_solver='BICGLSTAB2'
           case('bicglstab4'); runtime_solver='BICGLSTAB4'
           case('bicglstab8'); runtime_solver='BICGLSTAB8'
           case('gpbicgstabl2'); runtime_solver='GPBICGSTABL2'
           case('gpbicgstabl4'); runtime_solver='GPBICGSTABL4'
           case('gpbicgstabl8'); runtime_solver='GPBICGSTABL8'
           case default
              write(*,*) 'Unknown solver: ',trim(runtime_value)
              stop 2
           end select
        case('--grid-size'); read(runtime_value,*,iostat=ierror) cli_grid; cli_grid_set=ierror==0
        case('--grid-dimensions')
           read(runtime_value,*,iostat=ierror)cli_grid_dimensions
           cli_grid_dimensions_set=ierror==0
        case('--wavelength'); read(runtime_value,*,iostat=ierror) cli_wavelength; cli_wavelength_set=ierror==0
        case('--epsilon')
           do i=1,len_trim(runtime_value)
              if(runtime_value(i:i)==',')runtime_value(i:i)=' '
           end do
           read(runtime_value,*,iostat=ierror)cli_epsilon_real,cli_epsilon_imag
           cli_epsilon_set=ierror==0
        case('--tolerance'); read(runtime_value,*,iostat=ierror) cli_tolerance; cli_tol_set=ierror==0
        case('--max-iterations'); read(runtime_value,*,iostat=ierror) cli_maxit; cli_maxit_set=ierror==0
        case('--preconditioner'); read(runtime_value,*,iostat=ierror) cli_precon; cli_precon_set=ierror==0
        case('--initial-guess'); read(runtime_value,*,iostat=ierror) cli_init; cli_init_set=ierror==0
        case('--nrig'); read(runtime_value,*,iostat=ierror) cli_nrig; cli_nrig_set=ierror==0
        case('--nfft2d'); read(runtime_value,*,iostat=ierror) cli_nfft; cli_nfft_set=ierror==0
        case('--power'); read(runtime_value,*,iostat=ierror) cli_power; cli_power_set=ierror==0
        case('--nquad'); read(runtime_value,*,iostat=ierror) cli_nquad; cli_nquad_set=ierror==0
        case('--cross-section'); read(runtime_value,*,iostat=ierror) cli_section; cli_section_set=ierror==0
        case('--scattering'); read(runtime_value,*,iostat=ierror) cli_scattering; cli_scattering_set=ierror==0
        case('--force'); read(runtime_value,*,iostat=ierror) cli_force; cli_force_set=ierror==0
        case('--torque'); read(runtime_value,*,iostat=ierror) cli_torque; cli_torque_set=ierror==0
        case('--energy'); read(runtime_value,*,iostat=ierror) cli_energy; cli_energy_set=ierror==0
        case('--local-field'); read(runtime_value,*,iostat=ierror) cli_local; cli_local_set=ierror==0
        case('--macro-field'); read(runtime_value,*,iostat=ierror) cli_macro; cli_macro_set=ierror==0
        case('--theta'); read(runtime_value,*,iostat=ierror) cli_theta; cli_theta_set=ierror==0
        case('--phi'); read(runtime_value,*,iostat=ierror) cli_phi; cli_phi_set=ierror==0
        case('--pp'); read(runtime_value,*,iostat=ierror) cli_pp; cli_pp_set=ierror==0
        case('--ss'); read(runtime_value,*,iostat=ierror) cli_ss; cli_ss_set=ierror==0
        case('--waist'); read(runtime_value,*,iostat=ierror) cli_waist; cli_waist_set=ierror==0
        case('--speckle-seed'); read(runtime_value,*,iostat=ierror) cli_seed; cli_seed_set=ierror==0
        case('--microscope'); read(runtime_value,*,iostat=ierror) cli_microscope; cli_microscope_set=ierror==0
        case('--microscope-type'); read(runtime_value,*,iostat=ierror) cli_micro_type; cli_micro_type_set=ierror==0
        case('--microscope-side'); read(runtime_value,*,iostat=ierror) cli_side; cli_side_set=ierror==0
        case('--quick-lens'); read(runtime_value,*,iostat=ierror) cli_quick_lens; cli_quick_lens_set=ierror==0
        case('--near-domain'); read(runtime_value,*,iostat=ierror) cli_near_domain; cli_near_domain_set=ierror==0
        case('--near-padding')
           read(runtime_value,*,iostat=ierror)cli_padding
           cli_padding_set=ierror==0
        case('--geometry-only'); read(runtime_value,*,iostat=ierror) cli_geometry_only; cli_geometry_only_set=ierror==0
        case('--save-mode'); read(runtime_value,*,iostat=ierror) cli_save_mode; cli_save_mode_set=ierror==0
        case('--quick-diffraction'); read(runtime_value,*,iostat=ierror) cli_quick_diff; cli_quick_diff_set=ierror==0
        case('--na'); read(runtime_value,*,iostat=ierror) cli_na; cli_na_set=ierror==0
        case('--lens-z'); read(runtime_value,*,iostat=ierror) cli_lens_z; cli_lens_z_set=ierror==0
        case('--polarizability'); cli_polarizability=runtime_value(1:2); cli_polarizability_set=.true.
        case('--radius'); read(runtime_value,*,iostat=ierror)cli_radius; cli_radius_set=ierror==0
        case('--side-length'); read(runtime_value,*,iostat=ierror)cli_side_length; cli_side_length_set=ierror==0
        case('--geometry-dimensions')
           read(runtime_value,*,iostat=ierror)cli_geometry_dimensions
           cli_dimensions_set=ierror==0
        case('--geometry-angles')
           read(runtime_value,*,iostat=ierror)cli_geometry_angles
           cli_angles_set=ierror==0
        case('--geometry-density'); read(runtime_value,*,iostat=ierror)cli_density; cli_density_set=ierror==0
        case('--cell-size'); read(runtime_value,*,iostat=ierror)cli_cell_size; cli_cell_size_set=ierror==0
        case('--scattering-theta-points'); read(runtime_value,*,iostat=ierror)cli_ntheta; cli_ntheta_set=ierror==0
        case('--scattering-phi-points'); read(runtime_value,*,iostat=ierror)cli_nphi; cli_nphi_set=ierror==0
        case('--gaussian-center')
           read(runtime_value,*,iostat=ierror)cli_gaussian_center
           cli_gaussian_center_set=ierror==0
        case('--geometry-center')
           read(runtime_value,*,iostat=ierror)cli_geometry_center
           cli_geometry_center_set=ierror==0
        case('--illumination-na'); read(runtime_value,*,iostat=ierror)cli_illumination_na; cli_illumination_na_set=ierror==0
        case('--condenser-na'); read(runtime_value,*,iostat=ierror)cli_condenser_na; cli_condenser_na_set=ierror==0
        case('--inner-na'); read(runtime_value,*,iostat=ierror)cli_inner_na; cli_inner_na_set=ierror==0
        case('--na-center')
           read(runtime_value,*,iostat=ierror)cli_na_center
           cli_na_center_set=ierror==0
        case('--magnification'); read(runtime_value,*,iostat=ierror)cli_magnification; cli_magnification_set=ierror==0
        case('--background-epsilon'); read(runtime_value,*,iostat=ierror)cli_background_epsilon; cli_background_epsilon_set=ierror==0
        case('--abc-type'); read(runtime_value,*,iostat=ierror)cli_abc_type; cli_abc_type_set=ierror==0
        case('--abc-nrbc'); read(runtime_value,*,iostat=ierror)cli_abc_nrbc; cli_abc_nrbc_set=ierror==0
        case('--abc-cos'); read(runtime_value,*,iostat=ierror)cli_abc_cos; cli_abc_cos_set=ierror==0
        case('--force-density'); read(runtime_value,*,iostat=ierror)cli_force_density; cli_force_density_set=ierror==0
        case('--torque-density'); read(runtime_value,*,iostat=ierror)cli_torque_density; cli_torque_density_set=ierror==0
        case('--isotropy')
           if(trim(runtime_value)=='iso'.or.trim(runtime_value)=='ani')then
              cli_isotropy=trim(runtime_value)
              cli_isotropy_set=.true.
           else
              ierror=1
           end if
        case('--read-field'); read(runtime_value,*,iostat=ierror)cli_read_field; cli_read_field_set=ierror==0
        case('--field-file')
           if(len_trim(runtime_value)<1.or.len_trim(runtime_value)>63)then
              ierror=1
           else
              cli_field_file=trim(runtime_value)
              cli_field_file_set=.true.
           end if
        case('--object-file')
           if(len_trim(runtime_value)<1.or.len_trim(runtime_value)>63)then
              ierror=1
           else
              cli_object_file=trim(runtime_value)
              cli_object_file_set=.true.
           end if
        case('--hdf5-file')
           if(len_trim(runtime_value)<1.or.len_trim(runtime_value)>99)then
              ierror=1
           else
              cli_hdf5_file=trim(runtime_value)
              cli_hdf5_file_set=.true.
           end if
        case('--arbitrary-file')
           if(len_trim(runtime_value)<1.or.len_trim(runtime_value)>63)then
              ierror=1
           else
              cli_arbitrary_file=trim(runtime_value)
              cli_arbitrary_file_set=.true.
           end if
        case default
           write(*,*) 'Unknown option: ',trim(runtime_key)
           stop 2
        end select
        if(ierror/=0) then
           if(trim(runtime_key)/='--solver') then
              write(*,*) 'Invalid value for ',trim(runtime_key),': ',trim(runtime_value)
              stop 2
           end if
        end if
     end if
     arg_index=arg_index+1
  end do

!********************************************************
! Define the polarizability
!********************************************************

! polarizability='CM'  !Clausius Mossotti
  polarizability = 'RR' ! Clausius Mossotti with radiative reaction
  geometry_only_env=''
  call get_environment_variable('IFDDA_TEST_POLARIZABILITY',geometry_only_env,status=ierror)
  if(ierror==0 .and. (trim(geometry_only_env)=='CM' .or. &
                     trim(geometry_only_env)=='RR')) polarizability=geometry_only_env(1:2)
  if(cli_polarizability_set)then
     select case(cli_polarizability)
     case('CM','RR','PS','GB','LA','LR','LS','FG')
        polarizability=cli_polarizability
     case default
        stop 'invalid polarizability'
     end select
  end if
! polarizability='LA'  !Polarizability defines by Lakthakia
! polarizability='LR'  !Polarizability defines by Draine
! polarizability='GB'  !Polarizability with first Mie coefficient
! polarizability='PS'  !Polarizability for a sphere with local correction
! polarizability='FG'  !Filtered Green function

!********************************************************
! End polarizability
!********************************************************

!********************************************************
! Define the incident wave
!********************************************************
   ! The no-argument regression case uses the same default as case 200.
   beam='pwavelinear'
  select case (beam_case)
  case (200); beam = 'pwavelinear'
  case (201); beam = 'pwavecircular'
  case (202); beam = 'wavelinearmulti'
  case (203); beam = 'antenna'
  case (204); beam = 'gparawavelinear'
  case (205); beam = 'gparawavecircular'
  case (206); beam = 'gwavelinear'
  case (207); beam = 'gwavecircular'
  case (208); beam = 'arbitrary'
  case (209); beam = 'gfftwavelinear'
  case (210); beam = 'gfftwavecircular'
  case (211); beam = 'greentensor'
  case (212); beam = 'speckle'
  case (213); beam = 'demispeckle'
  case (214); beam = 'confocal'
  case (215); beam = 'demiconfocal'
  end select
! beam='wavelinearmulti'    !Multiple linear plane wave
! beam='antenna'            !Antenna define by dipolar emitter
! beam='pwavecircular'      !Circular plane wave
! beam='gparawavecircular'  !Linear Gaussian wave with paraxial approximation
! beam='gparawavelinear'    !Circular Gaussian wave with paraxial approximation
! beam='gwavelinear'        !Linear Gaussian wave rigourous
! beam='gwavecircular'      !Circular Gaussian wave rigourous
! beam='arbitrary'          !Arbitrary wave defines by the user
! beam='gfftwavecircular'   !Circular Gaussian wave rigourous for the first plane and propagates with FFT
! beam='gfftwavelinear'     !Linear Gaussian wave rigourous for the first plane and propagates with FFT
! beam='greentensor'        !Green tensor inside the object for one position
! beam='speckle'            !Illumination with a speckle
! beam='demispeckle'        !Illumination with a half speckle kz>0
! beam='confocal'           !Illumination with a confocal
! beam='demiconfocal'       !Illumination with a half confocal kz>0

  if (beam(1:11) == 'pwavelinear') then
     theta = zero
     phi = zero
     pp = zero
     ss = one
  else if (beam(1:15) == 'wavelinearmulti') then
     nbinc = 1
     thetam(1) = zero
     phim(1) = zero
     ppm(1) = zero
     ssm(1) = one
     E0m(1) = zone
  else if (beam(1:7) == 'antenna') then
     ! Source dipolaire a 300 nm du centre de la sphere testee (R=100 nm).
     ! -HUGE(one) produisait des debordements numeriques dans les distances.
     xgaus = -300.0_wp
     ygaus = zero
     zgaus = zero
     theta = zero
     phi = zero
  else if (beam(1:11) == 'greentensor') then
     xgaus = zero
     ygaus = zero
     zgaus = zero
  else if (beam(1:13) == 'pwavecircular') then
     theta = zero
     phi = zero
     ss = one
  else if (beam(1:11) == 'gwavelinear') then
     theta = zero
     phi = zero
     pp = zero
     ss = one
     xgaus = zero
     ygaus = zero
     zgaus = zero
  else if (beam(1:13) == 'gwavecircular') then
     theta = zero
     phi = zero
     ss = one
     xgaus = zero
     ygaus = zero
     zgaus = zero
  else if (beam(1:14) == 'gfftwavelinear') then
     theta = zero
     phi = zero
     pp = zero
     ss = one
     xgaus = zero
     ygaus = zero
     zgaus = zero
  else if (beam(1:16) == 'gfftwavecircular') then
     theta = zero
     phi = zero
     ss = one
     xgaus = zero
     ygaus = zero
     zgaus = zero
  else if (beam(1:15) == 'gparawavelinear') then
     theta = zero
     phi = zero
     pp = zero
     ss = one
     xgaus = zero
     ygaus = zero
     zgaus = zero
  else if (beam(1:17) == 'gparawavecircular') then
     theta = zero
     phi = zero
     ss = one
     xgaus = zero
     ygaus = zero
     zgaus = zero
  else if (beam(1:7) == 'speckle') then
     xgaus = zero
     ygaus = zero
     zgaus = zero
     speckseed = 0
     numaperil = one
  else if (beam(1:11) == 'demispeckle') then
     xgaus = zero
     ygaus = zero
     zgaus = zero
     speckseed = 0
     numaperil = one
  else if (beam(1:8) == 'confocal') then
     xgaus = zero
     ygaus = zero
     zgaus = zero
     numaperil = one
  else if (beam(1:12) == 'demiconfocal') then
     xgaus = zero
     ygaus = zero
     zgaus = zero
     numaperil = one
  else if (beam(1:9) == 'arbitrary') then
     namefileinc = 'incarbitrary.in'
  end if
!*******************************************************
! End incident wave
!*******************************************************
  if(cli_theta_set)theta=cli_theta
  if(cli_phi_set)phi=cli_phi
  if(cli_pp_set)pp=cli_pp
  if(cli_ss_set)ss=cli_ss
  if(cli_waist_set)w0=cli_waist
  if(cli_gaussian_center_set)then
     xgaus=cli_gaussian_center(1)
     ygaus=cli_gaussian_center(2)
     zgaus=cli_gaussian_center(3)
  end if
  if(cli_seed_set)speckseed=cli_seed
  if(cli_arbitrary_file_set)namefileinc=cli_arbitrary_file

!********************************************************
! Define the object
!********************************************************
  nxm = 32
  nym = 32
  nzm = 32
  nnnr = 32
  numberobjet = 1
  ! Geometry generators consume the center arrays even when a case does not
  ! set an explicit center (notably randomsphere1 and randomsphere2).
  xgmulti = zero
  ygmulti = zero
  zgmulti = zero
  materiaumulti(1) = 'xx'
  materiaumulti(2) = 'xx'
  object = 'sphere' ! Default object when the executable is run without arguments.
  select case (geometry_case)
  case (0); object = 'concentricsphere'
  case (1); object = 'sphere'
  case (2); object = 'cube'
  case (3); object = 'cuboid1'
  case (4); object = 'cuboid2'
  case (5); object = 'inhomosphere'
  case (6); object = 'inhomocuboid1'
  case (7); object = 'inhomocuboid2'
  case (8); object = 'ellipsoid'
  case (9); object = 'nspheres'
  case (10); object = 'cylinder'
  case (11); object = 'randomsphere1'
  case (12); object = 'randomsphere2'
  case (13); object = 'arbitrary'
  case (14); object = 'randomspheremulti1'
  end select
  if (len_trim(chosen_geometry)>0) &
     write(*,'(A,A)') '[IF-DDA] Geometry selected: ',trim(object)
! object='inhomosphere'     !Inhomogeneous sphere
! object='cube'             !Cube
! object='cuboid1'          !Cuboid with side given
! object='cuboid2'          !Cuboid with nx,ny,nz,aretecube given
! object='inhomocuboid1'    !Inhomogeneous cuboid with side given
! object='inhomocuboid2'    !Inhomogeneous cuboid with nx,ny,nz,aretecube given
! object='ellipsoid'        !Ellipsoid
! object='nspheres'         !Multiple spheres with same radius
! object='cylinder'         !Cylinder
! object='concentricsphere' !Concentric sphere
! object='randomsphere1'    !Random spheres in box with side given
! object='randomsphere2'    !Random spheres in box with nx,ny,nz,aretecube given
! object='arbitrary'        !Arbitrary object

  if (object(1:6) == 'sphere') then
     numberobjet = 1
     rayonmulti(1) = 500.0_wp
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     materiaumulti(1) = 'xx'
     ! Run the remaining targeted sphere regressions on a 32^3 mesh.
     nnnr = 32
     rayonmulti(1) = 100.0_wp
     ! Default sphere refractive index: m = 1.6 + 0.0i.
     epsmulti(1) = cmplx(1.6_wp, 0.0_wp, kind=wp)
     nxm=nnnr
     nym=nnnr
     nzm=nnnr    
  else if (object(1:13) == 'inhomosphere') then
     numberobjet = 1
     rayonmulti(1) = 800.0_wp
     lc = lambda
     hc = 0.2_wp
     materiaumulti(1) = 'xx'
     ng = 2
     nnnr = 32
     nxm = 32; nym = 32; nzm = 32
     epsmulti(1) = cmplx(1.2_wp, 5.0e-2_wp, kind=wp)
  else if (object(1:13) == 'inhomocuboid1') then
     numberobjet = 1
     sidex = 1600.0_wp
     sidey = 1600.0_wp
     sidez = 1600.0_wp
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     lc = lambda
     hc = 0.2_wp
     materiaumulti(1) = 'xx'
     ng = 2
     nnnr = 32
     nxm = 32; nym = 32; nzm = 32
     epsmulti(1) = cmplx(1.2_wp, 5.0e-2_wp, kind=wp)
  else if (object(1:13) == 'inhomocuboid2') then
     numberobjet = 1
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     aretecube = 25.0_wp
     lc = lambda
     hc = 0.2_wp
     materiaumulti(1) = 'xx'
     ng = 2
     epsmulti(1) = (2.25_wp,0.0_wp)
  else if (object(1:13) == 'randomsphere1') then
     numberobjet = 1
     rayonmulti(1) = 150.0_wp
     density = 0.2_wp
     sidex = 2000.0_wp
     sidey = 2000.0_wp
     sidez = 2000.0_wp
     materiaumulti(1) = 'xx'
     ng = 2
     nnnr = 20
     epsmulti(1) = (2.25_wp,0.05_wp) ! Weak absorption, avoids zero-extinction cancellation in nrig=2
  else if (object(1:13) == 'randomsphere2') then
     numberobjet = 1
     rayonmulti(1) = 100.0_wp
     density = 0.2_wp
     aretecube = 25.0_wp
     materiaumulti(1) = 'xx'
     ng = 2
     nnnr = 20
     epsmulti(1) = (2.25_wp,0.05_wp) ! Same material for F90/C99 comparison
  else if (object(1:18) == 'randomspheremulti1') then
     numberobjet = 2
     rayonmulti(1) = 120.0_wp
     rayonmulti(2) = 120.0_wp
     xgmulti(1) = -250.0_wp
     xgmulti(2) = 250.0_wp
     density = 0.2_wp
     sidex = 2000.0_wp
     sidey = 2000.0_wp
     sidez = 2000.0_wp
     materiaumulti(1) = 'xx'
     materiaumulti(2) = 'xx'
     ng = 2
     nnnr = 32
     epsmulti(1) = cmplx(1.2_wp, 5.0e-2_wp, kind=wp)
     epsmulti(2) = (2.25_wp,0.0_wp)
  else if (object(1:4) == 'cube') then
     numberobjet = 1
     side = 100.0_wp
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     materiaumulti(1) = 'xx'
     nnnr = 10
     epsmulti(1) = (2.0_wp, 0.5_wp)
  else if (object(1:7) == 'cuboid1') then
     numberobjet = 1
     sidex = 20.0_wp
     sidey = 40.0_wp
     sidez = 10.0_wp
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     materiaumulti(1) = 'xx'
     nnnr = 10
     phiobj = zero
     thetaobj = zero
     psiobj = zero
     epsmulti(1) = (2.0_wp, 0.5_wp)
     nnnr = 20
  else if (object(1:7) == 'cuboid2') then
     numberobjet = 1
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     materiaumulti(1) = 'xx'
     aretecube = 25.0_wp
     epsmulti(1) = (2.0_wp, 0.5_wp)
  else if (object(1:8) == 'nspheres') then
     numberobjet = 2
     nxm = 32
     nym = 32
     nzm = 32
     materiaumulti(1) = 'xx'
     materiaumulti(2) = 'xx'
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     rayonmulti(1) = 10.0_wp
     epsmulti(1) = (2.25_wp, zero)
     xgmulti(2) = 30.0_wp
     ygmulti(2) = zero
     zgmulti(2) = zero
     rayonmulti(2) = 15.0_wp
     epsmulti(2) = (1.25_wp, zero)
     nnnr = 20
  else if (object(1:9) == 'ellipsoid') then
     demiaxea = 10.0_wp
     demiaxeb = 20.0_wp
     demiaxec = 40.0_wp
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     thetaobj = 45.0_wp
     phiobj = 45.0_wp
     psiobj = zero
     nnnr = 32
     epsmulti(1) = cmplx(1.2_wp, 5.0e-2_wp, kind=wp)
  else if (object(1:8) == 'cylinder') then
     rayonmulti(1) = 10.0_wp
     hauteur = 20.0_wp
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     thetaobj = 45.0_wp
     phiobj = 45.0_wp
     psiobj = zero
     nnnr = 20
     epsmulti(1) = (2.25_wp,0.0_wp)
  else if (object(1:16) == 'concentricsphere') then
     numberobjet = 2
     nxm = 32
     nym = 32
     nzm = 32
     materiaumulti(1) = 'xx'
     materiaumulti(2) = 'xx'
     rayonmulti(1) = 2.5_wp
     epsmulti(1) = (2.25_wp, zero)
     xgmulti(1) = zero
     ygmulti(1) = zero
     zgmulti(1) = zero
     rayonmulti(2) = 5.0_wp
     epsmulti(2) = (1.25_wp, zero)
     thetaobj = zero
     phiobj = zero
     nnnr = 32
  else if (object(1:9) == 'arbitrary') then
     numberobjet = 1
     namefileobj = 'arbit.in'
     if(cli_object_file_set)namefileobj=cli_object_file
     open(15, file=namefileobj, status='old', iostat=ierror)
     if (ierror /= 0) then
        write(*,*) 'bad namefile for arbitrary'
        stop
     end if
     read(15,*) nx, ny, nz
     read(15,*) aretecube
     rewind(15)
     close(15)

     if (nx > nxm .or. ny > nym .or. nz > nzm) then
        write(*,*) 'Size of the table too small'
        stop
     end if
  end if

  ! Validation-only relative permittivity override for matched CPU/CUDA runs.
  ! The environment value is epsilon_real,epsilon_imag (not refractive index).
  epsilon_override_env=''
  call get_environment_variable('IFDDA_TEST_GEOMETRY_EPSILON', &
       epsilon_override_env,status=ierror)
  if (ierror == 0 .and. len_trim(epsilon_override_env) > 0) then
     read(epsilon_override_env,*,iostat=ierror) &
          epsilon_override_real,epsilon_override_imag
     if (ierror /= 0) stop 'Invalid IFDDA_TEST_GEOMETRY_EPSILON'
     do ii=1,numberobjet
        epsmulti(ii)=cmplx(epsilon_override_real,epsilon_override_imag,kind=wp)
     end do
  end if

  if(cli_grid_set) then
     if(cli_grid<8 .or. cli_grid>256) stop 'grid-size must be 8..256'
     nnnr=cli_grid
     nxm=cli_grid
     nym=cli_grid
     nzm=cli_grid
  end if
  if(cli_grid_dimensions_set)then
     if(any(cli_grid_dimensions<8).or.any(cli_grid_dimensions>256)) &
          stop 'grid-dimensions values must be 8..256'
     nxm=cli_grid_dimensions(1)
     nym=cli_grid_dimensions(2)
     nzm=cli_grid_dimensions(3)
     nnnr=minval(cli_grid_dimensions)
  end if
  if(cli_epsilon_set) then
     do ii=1,numberobjet
        epsmulti(ii)=cmplx(cli_epsilon_real,cli_epsilon_imag,kind=wp)
     end do
  end if
  ! Validation fixture: deliberately different, lossy materials for each
  ! object; the same main0.f90 source is linked against both backends.
  geometry_only_env=''
  call get_environment_variable('IFDDA_TEST_DISTINCT_COMPLEX_MATERIALS',geometry_only_env,status=ierror)
  if (ierror == 0 .and. trim(geometry_only_env) == '1') then
     do ii=1,numberobjet
        epsmulti(ii)=cmplx(1.35_wp + 0.25_wp*real(ii-1,wp), &
                           0.04_wp + 0.015_wp*real(ii-1,wp),kind=wp)
     end do
  end if
  if(cli_radius_set)then
     if(cli_radius<=0)stop 'radius must be positive'
     rayonmulti(1)=cli_radius
  end if
  if(cli_side_length_set)then
     if(cli_side_length<=0)stop 'side-length must be positive'
     side=cli_side_length
     sidex=cli_side_length
     sidey=cli_side_length
     sidez=cli_side_length
  end if
  if(cli_dimensions_set)then
     select case(geometry_case)
     case(3,6)
        if(any(cli_geometry_dimensions<=0))stop 'cuboid dimensions must be positive'
        sidex=cli_geometry_dimensions(1)
        sidey=cli_geometry_dimensions(2)
        sidez=cli_geometry_dimensions(3)
     case(8)
        if(any(cli_geometry_dimensions<=0))stop 'ellipsoid semiaxes must be positive'
        demiaxea=cli_geometry_dimensions(1)
        demiaxeb=cli_geometry_dimensions(2)
        demiaxec=cli_geometry_dimensions(3)
     case(10)
        if(any(cli_geometry_dimensions(1:2)<=0))stop 'cylinder radius and height must be positive'
        rayonmulti(1)=cli_geometry_dimensions(1)
        hauteur=cli_geometry_dimensions(2)
     case default
        stop '--geometry-dimensions needs cuboid1, inhomocuboid1, ellipsoid or cylinder'
     end select
  end if
  if(cli_angles_set)then
     thetaobj=cli_geometry_angles(1)
     phiobj=cli_geometry_angles(2)
     psiobj=cli_geometry_angles(3)
  end if
  if(cli_density_set)then
     if(cli_density<=0)stop 'geometry-density must be positive'
     density=cli_density
  end if
  if(cli_cell_size_set)then
     if(cli_cell_size<=0)stop 'cell-size must be positive'
     aretecube=cli_cell_size
  end if
  if(cli_geometry_center_set)then
     xgmulti(1)=cli_geometry_center(1)
     ygmulti(1)=cli_geometry_center(2)
     zgmulti(1)=cli_geometry_center(3)
  end if
  if(cli_wavelength_set)lambda=cli_wavelength
  if(cli_power_set)P0=cli_power

  if (numberobjet > numberobjetmax) then
     write(*,*) 'redimensionner numberobjetmax', numberobjet, numberobjetmax
     stop
  end if
!*******************************************************
! End object
!*******************************************************

  if (polarizability == 'PS') then
     niso = 3
  else
     niso = 1
  end if
  geometry_only_env=''
  call get_environment_variable('IFDDA_TEST_GEOMETRY_ANIS',geometry_only_env,status=ierror)
  if(ierror==0 .and. trim(geometry_only_env)=='1') niso=3
  ! Anisotropy requires the complete 3x3 constitutive tensor, not the
  ! scalar (niso=1) allocation used for isotropic polarizabilities.
  if(cli_isotropy_set.and.cli_isotropy=='ani') niso=3

  nfft2d = 128
  ! Validation-only FFT dimension override to match large-mesh CUDA/Fortran
  ! optical-force cases; ordinary coverage fixtures retain nfft2d=128.
  geometry_only_env=''
  call get_environment_variable('IFDDA_TEST_NFFT2D',geometry_only_env,status=ierror)
  if(ierror==0.and.len_trim(geometry_only_env)>0)then
     read(geometry_only_env,*,iostat=ierror)nfft2d
     if(ierror/=0.or.nfft2d<128.or.nfft2d>1024.or. &
          iand(nfft2d,nfft2d-1)/=0)stop 'Invalid IFDDA_TEST_NFFT2D'
  endif
  if(cli_nfft_set)then
     if(cli_nfft<16.or.cli_nfft>1024.or.iand(cli_nfft,cli_nfft-1)/=0) &
          stop 'nfft2d must be a power of two in 16..1024'
     nfft2d=cli_nfft
  endif
  nlar = 42
  select case (method_case)
  case (15); nlar = 12
  case (16, 19); nlar = 18
  case (17); nlar = 30
  case (18); nlar = 8
  case (20); nlar = 20
  ! GPBICGSTABL(L) requires at least NLAR=4*L+8 columns:
  ! L=2/4/8 => 16/24/40. Keep the default 42 for all three,
  ! matching the C99 coverage driver's allocated work array.
  end select
  if(len_trim(runtime_solver)>0)methodeit=runtime_solver
  ! The CLI controls far-field grid dimensions before allocation.
  ntheta = 0
  nphi = 0
  if(cli_scattering_set.and.cli_scattering==1)then
     ntheta=64
     nphi=64
  end if
  if(cli_ntheta_set)ntheta=cli_ntheta
  if(cli_nphi_set)nphi=cli_nphi
  alloc_size = max((ntheta+1)*max(nphi,1), nfft2d * nfft2d)
  allocate(thetafield(alloc_size))
  allocate(phifield(alloc_size))
  allocate(poyntingfield(alloc_size))
  allocate(poyntingfieldpos(alloc_size))
  allocate(poyntingfieldneg(alloc_size))
  allocate(xs(nxm*nym*nzm), ys(nxm*nym*nzm), zs(nxm*nym*nzm))
  allocate(xswf(nxm*nym*nzm), yswf(nxm*nym*nzm), zswf(nxm*nym*nzm))
  allocate(polarisa(nxm*nym*nzm, niso, niso), epsilon(nxm*nym*nzm, niso, niso))
  allocate(incidentfield(nxm*nym*nzm), localfield(nxm*nym*nzm))
  allocate(macroscopicfield(nxm*nym*nzm))
  allocate(forcex(nxm*nym*nzm), forcey(nxm*nym*nzm))
  allocate(forcez(nxm*nym*nzm))
  allocate(torquex(nxm*nym*nzm), torquey(nxm*nym*nzm))
  allocate(torquez(nxm*nym*nzm))
  allocate(macroscopicfieldx(nxm*nym*nzm))
  allocate(macroscopicfieldy(nxm*nym*nzm))
  allocate(macroscopicfieldz(nxm*nym*nzm))
  allocate(localfieldx(nxm*nym*nzm), localfieldy(nxm*nym*nzm))
  allocate(localfieldz(nxm*nym*nzm))
  allocate(incidentfieldx(nxm*nym*nzm), incidentfieldy(nxm*nym*nzm))
  allocate(incidentfieldz(nxm*nym*nzm))
  allocate(FF(3*nxm*nym*nzm), FF0(3*nxm*nym*nzm))
  allocate(FFloc(3*nxm*nym*nzm))
  allocate(Tabdip(nxm*nym*nzm), Tabmulti(nxm*nym*nzm))
  ntotal = (nxm+1)*(nym+1)*(nzm+1)
  allocate(FFTTENSORxx(ntotal), FFTTENSORxy(ntotal))
  allocate(FFTTENSORxz(ntotal))
  allocate(FFTTENSORyy(ntotal), FFTTENSORyz(ntotal))
  allocate(FFTTENSORzz(ntotal))
  allocate(xr(3*nxm*nym*nzm), xi(3*nxm*nym*nzm))
  allocate(wrk(3*nxm*nym*nzm, nlar))
  allocate(tabfft2(nfft2d))
  allocate(kxy(nfft2d), xy(nfft2d), kxypoynting(nfft2d))
  allocate(Eimagex(nfft2d*nfft2d), Eimagey(nfft2d*nfft2d))
  allocate(Eimagez(nfft2d*nfft2d), Eimageincx(nfft2d*nfft2d))
  allocate(Eimageincy(nfft2d*nfft2d), Eimageincz(nfft2d*nfft2d))
  allocate(Efourierx(nfft2d*nfft2d), Efouriery(nfft2d*nfft2d))
  allocate(Efourierz(nfft2d*nfft2d), Efourierincx(nfft2d*nfft2d))
  allocate(Efourierincy(nfft2d*nfft2d), Efourierincz(nfft2d*nfft2d))
  allocate(Ediffkzpos(nfft2d, nfft2d, 3), Ediffkzneg(nfft2d, nfft2d, 3))

!********************************************************
! Define if object is isotropic or not
!********************************************************
  trope = 'iso'
  geometry_only_env=''
  call get_environment_variable('IFDDA_TEST_GEOMETRY_ANIS',geometry_only_env,status=ierror)
  if(ierror==0 .and. trim(geometry_only_env)=='1') trope='ani'
  if(cli_isotropy_set)trope=cli_isotropy
! trope='ani'
  if (trope(1:3) == 'ani') then
     ! Deterministic genuinely anisotropic, absorbing and symmetric tensor.
     ! The previous validation driver filled three equal diagonal entries,
     ! which was only an isotropic material stored in a 3x3 array.
     ! Distinct axes and nonzero complex off-diagonal couplings exercise
     ! the general 3x3 CUDA/Fortran material and polarizability paths.
     epsanimulti=cmplx(0.0_wp,0.0_wp,kind=wp)
     do ii=1,numberobjet
        epsanimulti(1,1,ii)=epsmulti(ii)
        epsanimulti(2,2,ii)=epsmulti(ii)*0.90_wp
        epsanimulti(3,3,ii)=epsmulti(ii)*1.15_wp
        epsanimulti(1,2,ii)=epsmulti(ii)*0.04_wp
        epsanimulti(2,1,ii)=epsanimulti(1,2,ii)
        epsanimulti(1,3,ii)=epsmulti(ii)*0.02_wp
        epsanimulti(3,1,ii)=epsanimulti(1,3,ii)
        epsanimulti(2,3,ii)=epsmulti(ii)*0.03_wp
        epsanimulti(3,2,ii)=epsanimulti(2,3,ii)
     end do
     write(*,*) 'CDMLIB_ANISOTROPIC_TENSOR', epsanimulti(:,:,1)
  end if

!*******************************************************
! Define the iterative method used
!*******************************************************
  methodeit = 'GPBICG1'
  select case (method_case)
  case (24); methodeit = 'GPBICG1'
  case (1); methodeit = 'GPBICG2'
  case (2); methodeit = 'GPBICGplus'
  case (3); methodeit = 'GPBICGsafe'
  case (4); methodeit = 'GPBICGAR'
  case (5); methodeit = 'GPBICGAR2'
  case (6); methodeit = 'BICGSTARPLUS'
  case (7); methodeit = 'GPBICOR'
  case (8); methodeit = 'CORS'
  case (9); methodeit = 'QMRCLA'
  case (10); methodeit = 'TFQMR'
  case (11); methodeit = 'CG'
  case (12); methodeit = 'BICGSTAB'
  case (13); methodeit = 'QMRBICGSTAB1'
  case (14); methodeit = 'QMRBICGSTAB2'
  case (15); methodeit = 'IDRS2'
  case (16); methodeit = 'IDRS4'
  case (17); methodeit = 'IDRS8'
  case (18); methodeit = 'BICGLSTAB2'
  case (19); methodeit = 'BICGLSTAB4'
  case (20); methodeit = 'BICGLSTAB8'
  case (21); methodeit = 'GPBICGSTABL2'
  case (22); methodeit = 'GPBICGSTABL4'
  case (23); methodeit = 'GPBICGSTABL8'
  end select
  if(len_trim(runtime_solver)>0)methodeit=runtime_solver
! methodeit='GPBICG2'
! methodeit='GPBICGplus'
! methodeit='GPBICGsafe'
! methodeit='GPBICGAR'
! methodeit='GPBICGAR2'
! methodeit='BICGSTARPLUS'
! methodeit='GPBICOR'
! methodeit='CORS'
! methodeit='QMRCLA'
! methodeit='TFQMR'
! methodeit='CG'
! methodeit='BICGSTAB'
! methodeit='QMRBICGSTAB1'
! methodeit='QMRBICGSTAB2'
! methodeit='IDRS2'
! methodeit='IDRS4' !attention changer nlar
! methodeit='IDRS8' !attention changer nlar
! methodeit='BICGLSTABL2'
! methodeit='BICGLSTABL4' !attention changer nlar
! methodeit='BICGLSTABL8' !attention changer nlar
! methodeit='GPBICGSTABL2'
! methodeit='GPBICGSTABL4' !attention changer nlar
! methodeit='GPBICGSTABL8' !attention changer nlar

! Define the tolerance of the iterative method
  tolinit = 1.0e-4_wp ! Tolerance for the iterative method
  nlim = 1000 ! Maximum of iteration for the iterative method
  nprecon = precon_case ! 0 none, 1 Chan left, 2 Chan right
  ninitest = init_case ! 0 null, 1 Born, 2 Rytov, 3 BPM, 4/5 special scalar modes
  ! nrig controls the 3*nzm tensor dimensions below, so it must be
  ! initialized before sdetnn and FFprecon are allocated.
  nrig = nrig_case
  if(cli_tol_set)tolinit=cli_tolerance
  if(cli_maxit_set)nlim=cli_maxit
  if(cli_precon_set)nprecon=cli_precon
  if(cli_init_set)ninitest=cli_init
  if(cli_nrig_set)nrig=cli_nrig
  if(cli_nquad_set)nquad=cli_nquad
  if (nprecon < 0 .or. nprecon > 2) then
     write(*, '(A,I0,A)') 'Unsupported preconditioner ', nprecon, &
          '; supported values are 0 (none), 1 (Chan left), and 2 (Chan right).'
     stop 2
  end if
  if(tolinit<=0.or.nlim<1.or.ninitest<0.or.ninitest>5.or. &
     nrig<0.or.nrig>15) &
       stop 'invalid iterative solver option'

  if (nprecon == 0) then
     nxym1 = 1
     nzm1 = 1
     nzms1 = 1
  else
     if (nrig == 0) then
        nxym1 = nxm * nym
        nzm1 = nzm
        nzms1 = 3 * nzm
     else
        nxym1 = nxm * nym
        nzm1 = nzm
        nzms1 = nzm
     end if
  end if

  allocate(sdetnn(nzms1, nzms1, nxym1))
  allocate(FFprecon(3*nxym1*nzm1))
!*******************************************************
! End iterative method used
!*******************************************************

!********************************************************
! define all the options
!********************************************************
  nobjet = 0 ! 1 compute only the position of the
  geometry_only_env=''
  call get_environment_variable('IFDDA_TEST_GEOMETRY_ONLY',geometry_only_env,status=ierror)
  if (ierror==0 .and. trim(geometry_only_env)=='1') then
     nobjet=1
     write(*,*) '[IF-DDA] Geometry-only verification (no iterative solve)'
  end if
             ! dipole, all the other options are
             ! disabled.

! nproche adjust the size of near field domain. Near field (0)
! inside the object, (1) inside a cuboid which contains the object,
! (2) inside the boxnx+2*nxmp,ny+2*nymp,nz+2*nzmp
  nproche = 0
  ! Match the Fortran coverage driver for every named geometry.
  ! --near-domain 0 still explicitly selects compact object storage.
  if (len_trim(chosen_geometry) > 0) nproche = 1

  nxmp = 0 ! if nproche=2 used then the addsize along x : nx+2*nxmp
  nymp = 0 ! if nproche=2 used then the addsize along y : ny+2*nymp
  nzmp = 0 ! if nproche=2 used then the addsize along z : nz+2*nzmp
  nlocal = 1 ! 0 do not compute the local field, 1 compute the local field
  nmacro = 1 ! 0 do not compute the macroscopic field, 1 compute the macroscopic field

! 1 reread or create a file which contains the local field at each
! position. Avoid to compute again the local field if the
! configuration is the same, i.e. keep the same local field.
  nlecture = 0
  filereread = 'toto' ! name fo the file if reread the local field.
  if(cli_read_field_set)nlecture=cli_read_field
  if(cli_field_file_set)filereread=cli_field_file

! nrig adjust the ways used to compute the near field. (0) compute
! rigorously the near field in solving the near field equation, (1)
! use renormalized Born approximation, (2) use Born approximation,
! (3) use Born series at order 1, (4) renormalized Rytov, (5) Rytov,
! (6) BPM, (7) renormalized BPM.
  ! nrig was initialized before allocating the preconditioner buffers.

  nforce = 0 ! (0) Do not compute or (1) compute the optical force.
  nforced = 0 ! (0) Do not compute or (1) compute the density of optical force.
  ntorque = 0 ! (0) Do not compute or (1) compute the optical torque.
  ntorqued = 0 ! (0) Do not compute or (1) compute the density of optical torque.

  nsection = 1 ! 0 do not compute the cross section, 1 compute the cross section.
  nsectionsca = 0 ! 1: calcul C_sca, Poynting and g with radiating dipole.
  nquickdiffracte = 0 ! 0 compute far field classically, 1 compute far field with FFT.
  nside = 0 ! compute microscope in reflexion (1), or transmission (0).
  nlentille = 0 ! Compute microscopy.
  nquicklens = 0 ! Compute microscopy with FFT (1) or wihtout FFT (0).
  numaper = 0.9_wp ! Numerical aperture for the microscope.
  zlens = 200.0_wp ! Position of lens.
  ntypemic = 0 ! Type of microscope: O Holographic, 1 Bright field, 2 Dark field, 3 schiren, &
               ! 4 Dark field2, 5 dark field experimental, 6 confocal
  gross = 100.0_wp ! Manyfing factor for the microscope
  numaperinc = 0.9_wp ! Numerical aperture for the condenser lens.
  numaperinc2 = zero ! Numerical aperture for central aperture
  kcnax = zero ! X position of the NA
  kcnay = zero ! Y position of the NA
  nenergie = 0 ! 0 Do not compute energy, 1 compute energy conservation.
  if (beam_case /= 0 .and. beam_case /= 200 .and. beam_case /= 201) nsection = 0

  nmat = 1 ! 1 Do not save the data, 0 save the data in mat file, &
  geometry_only_env=''
  call get_environment_variable('IFDDA_TEST_GEOMETRY_ASCII',geometry_only_env,status=ierror)
  if(ierror==0 .and. trim(geometry_only_env)=='1') nmat=0
           ! 2 save the data in one hdf5 file.
  h5file = 'ifdda.h5' ! name of the hdf5 file
  if(cli_hdf5_file_set)h5file=cli_hdf5_file

  nquad = 0 ! 0 -> 5 define the level of integration of the Green tensor.

  ! Additional branch-coverage selectors. Both reference and C99 builds use
  ! this same driver, so each selector exercises an identical cdmlib call.
  select case (option_case)
  case (1); nlocal = 0
  case (2); nmacro = 0
  case (3); nsection = 0
  case (4); nsectionsca = 1; ntheta = 12; nphi = 24
  case (5); nsectionsca = 1; nquickdiffracte = 1
  case (6); nenergie = 1
  case (7); nquad = 1
  case (8); nproche = 1
  case (9); nproche = 2; nnnr = 6; nxmp = 1; nymp = 1; nzmp = 1
  case (10); nforce = 1
  case (11); nforce = 1; ntorque = 1
  case (12); nlentille = 1; ntypemic = 0; nquicklens = 0
  case (13); nlentille = 1; ntypemic = 0; nquicklens = 1
  case (14); nlentille = 1; ntypemic = 0; nquicklens = 0; nside = 1
  case (15); nlentille = 1; ntypemic = 1
  case (16); nlentille = 1; ntypemic = 2
  case (17); nlentille = 1; ntypemic = 3
  case (18); nlentille = 1; ntypemic = 4
  case (19); nlentille = 1; ntypemic = 5
  case (20); nlentille = 1; ntypemic = 6
  case (21); nlentille = 1; ntypemic = 7
  case (22); nquad = 2
  case (23); nquad = 3
  case (24); nquad = 4
  case (25); nquad = 5
  case (26); nforce = 1; nforced = 1
  case (27); nforce = 1; ntorque = 1; ntorqued = 1
  case (28); nlecture = 1; filereread = 'coverage-nlecture'
  end select

  if(cli_nquad_set)nquad=cli_nquad
  if(nquad<0.or.nquad>5) stop 'invalid Green integration level'
  if(cli_section_set)nsection=cli_section
  if(cli_scattering_set)then
     nsectionsca=cli_scattering
     if(nsectionsca==1)then
        ntheta=64
        nphi=64
     end if
  end if
  if(cli_force_set)nforce=cli_force
  if(cli_force_density_set)nforced=cli_force_density
  if(cli_torque_set)ntorque=cli_torque
  if(cli_torque_density_set)ntorqued=cli_torque_density
  if(cli_energy_set)nenergie=cli_energy
  if(cli_local_set)nlocal=cli_local
  if(cli_macro_set)nmacro=cli_macro
  if(cli_microscope_set)nlentille=cli_microscope
  if(cli_micro_type_set)ntypemic=cli_micro_type
  if(cli_side_set)nside=cli_side
  if(cli_quick_lens_set)nquicklens=cli_quick_lens
  if(cli_near_domain_set)nproche=cli_near_domain
  if(cli_padding_set)then
     nxmp=cli_padding(1)
     nymp=cli_padding(2)
     nzmp=cli_padding(3)
     if(min(nxm-2*nxmp,nym-2*nymp,nzm-2*nzmp)<8) &
          stop 'near-padding leaves fewer than 8 interior cells on an axis'
     nnnr=min(nnnr,nxm-2*nxmp,nym-2*nymp,nzm-2*nzmp)
  end if
  if(cli_geometry_only_set)nobjet=cli_geometry_only
  if(cli_save_mode_set)nmat=cli_save_mode
  if(cli_quick_diff_set)nquickdiffracte=cli_quick_diff
  if(cli_na_set)numaper=cli_na
  if(cli_lens_z_set)zlens=cli_lens_z
  if(cli_polarizability_set)then
     select case(cli_polarizability)
     case('CM','RR','PS','GB','LA','LR','LS','FG')
        polarizability=cli_polarizability
     case default
        stop 'invalid polarizability'
     end select
  end if
  if(nsection<0.or.nsection>1.or.nsectionsca<0.or.nsectionsca>1.or. &
     nforce<0.or.nforce>1.or.ntorque<0.or.ntorque>1.or.nenergie<0.or.nenergie>1.or. &
     nforced<0.or.nforced>1.or.ntorqued<0.or.ntorqued>1.or. &
     nlocal<0.or.nlocal>1.or.nmacro<0.or.nmacro>1.or.nlecture<0.or.nlecture>1) &
       stop 'boolean options must be 0 or 1'

!*******************************************************
! Confocal microscope
!*******************************************************
  if (ntypemic == 6) then
     zstep = 100.0_wp ! step of the z scan to create the z-stack
     zinf = -1000.0_wp ! lower z of the s-stack
     zsup = 1000.0_wp ! higher z of the s-stack

! use previous varaibles
     zlens = zstep
     kcnax = zinf
     kcnay = zsup
  end if
  if(cli_ntheta_set)ntheta=cli_ntheta
  if(cli_nphi_set)nphi=cli_nphi
  if(cli_illumination_na_set)numaperil=cli_illumination_na
  if(cli_condenser_na_set)numaperinc=cli_condenser_na
  if(cli_inner_na_set)numaperinc2=cli_inner_na
  if(cli_na_center_set)then
     kcnax=cli_na_center(1)
     kcnay=cli_na_center(2)
  end if
  if(cli_magnification_set)gross=cli_magnification
  if(cli_lens_z_set)zlens=cli_lens_z
  if(cli_background_epsilon_set)eps0i=cli_background_epsilon
  if(cli_abc_type_set)ntypeabc=cli_abc_type
  if(cli_abc_nrbc_set)lnrbc=cli_abc_nrbc
  if(cli_abc_cos_set)lcos=cli_abc_cos
  if(cli_ntheta_set.and.cli_ntheta<1)stop 'scattering-theta-points must be positive'
  if(cli_padding_set.and.any(cli_padding<0))stop 'near-padding values must be nonnegative'
  if(cli_nphi_set.and.cli_nphi<1)stop 'scattering-phi-points must be positive'
  if((cli_ntheta_set.or.cli_nphi_set).and.(ntheta<1.or.nphi<1)) &
       stop 'set both scattering point counts to positive values'
  if(cli_illumination_na_set.and.(numaperil<0.or.numaperil>1))stop 'illumination-na must be in [0,1]'
  if(cli_condenser_na_set.and.(numaperinc<0.or.numaperinc>1))stop 'condenser-na must be in [0,1]'
  if(cli_inner_na_set.and.(numaperinc2<0.or.numaperinc2>1))stop 'inner-na must be in [0,1]'
  if(cli_magnification_set.and.gross<=0)stop 'magnification must be positive'
  if(cli_padding_set.and.nproche/=2)stop 'near-padding requires --near-domain 2'
!*******************************************************

!*******************************************************
! End options
!*******************************************************

! compute size when meshsize is given
  if (object(1:13) == 'inhomocuboid2' .or. object(1:7) == 'cuboid2') then
     nx = nxm - 2 * nxmp
     ny = nym - 2 * nymp
     nz = nzm - 2 * nzmp
  else
     nx = nnnr
     ny = nnnr
     nz = nnnr
     if (nx + 2 * nxmp > nxm) then
        write(*,*) 'pb with size: increase nxm'
        stop
     end if
     if (ny + 2 * nymp > nym) then
        write(*,*) 'pb with size: increase nym'
        stop
     end if
     if (nz + 2 * nzmp > nzm) then
        write(*,*) 'pb with size: increase nzm'
        stop
     end if
  end if


  ! Initialize outputs so that rejected/incomplete paths cannot produce
  ! misleading uninitialized values in the machine-readable record.
  nbsphere = 0
  ndipole = 0
  Tabdip = 0
  nstop = 0
  ncompte = 0
  nloop = 0
  tol1 = zero
  Cext = zero
  Csca = zero
  Cabs = zero
  Cscai = zero
  forcem = zero
  couplem = zero
  efficacite = zero
  forcex = zero
  forcey = zero
  forcez = zero
  torquex = zero
  torquey = zero
  torquez = zero
  incidentfield = zero
  localfield = zero
  macroscopicfield = zero
  incidentfieldx = zzero
  incidentfieldy = zzero
  incidentfieldz = zzero
  localfieldx = zzero
  localfieldy = zzero
  localfieldz = zzero
  macroscopicfieldx = zzero
  macroscopicfieldy = zzero
  macroscopicfieldz = zzero

  ! Fix the GNU Fortran RNG stream for stochastic solvers such as IDRS/GCORS2.
  ! Geometry generators that reseed internally are tested separately.
  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  do seed_index = 1, seed_size
     seed(seed_index) = 13579 + 104729 * seed_index
  end do
  call random_seed(put=seed)
  deallocate(seed)

  call cdmlib( &
                                ! input file cdm.in
       lambda, eps0i,ntypeabc,lnrbc,lcos, beam, object, trope, niso, &
       materiaumulti, nnnr, tolinit, nlim, methodeit, polarizability, &
       nprecon, ninitest, nquad, nlecture, filereread, nmat, h5file, &
                                ! output file cdm.out
       nlocal, nmacro, nsection, nsectionsca, nquickdiffracte, nrig, &
       nforce, nforced, ntorque, ntorqued, nproche, nlentille, nquicklens, &
       nenergie, nobjet, &
                                ! cube, sphere (includes multiple)
       density, side, sidex, sidey, sidez, hauteur, &
       numberobjet, rayonmulti, xgmulti, ygmulti, zgmulti, &
       epsmulti, epsanimulti, lc, hc, ng, &
                                ! ellipsoid + arbitrary
       demiaxea, demiaxeb, demiaxec, thetaobj, phiobj, psiobj, &
       namefileobj, &
                                ! planewavecircular.in / planewavelinear.in files
       theta, phi, pp, ss, P0, w0, xgaus, ygaus, zgaus, speckseed, &
       namefileinc, &
                                ! ondeplane multiple
       thetam, phim, ppm, ssm, E0m, nbinc, &
                                ! return info string
       infostr, nstop, &
                                ! return scalar results
       nbsphere, ndipole, aretecube, &
       lambda10n, k0, tol1, tempsreelmvp, tempsreeltotal, ncompte, nloop, &
       efficacite, efficaciteref, efficacitetrans, &
       Cext, Cabs, Csca, Cscai, gasym, irra, E0, &
       forcet, forcem, &
       couplet, couplem, &
       nxm, nym, nzm, nxmp, nymp, nzmp, nmaxpp, nxym1, nzm1, nzms1, &
       incidentfield, localfield, macroscopicfield, &
       xs, ys, zs, xswf, yswf, zswf, &
       ntheta, nphi, thetafield, phifield, poyntingfield, &
       kxypoynting, poyntingfieldpos, poyntingfieldneg, &
       forcex, forcey, forcez, forcexmulti, forceymulti, forcezmulti, &
       torquex, torquey, torquez, torquexmulti, torqueymulti, torquezmulti, &
       incidentfieldx, incidentfieldy, incidentfieldz, &
       localfieldx, localfieldy, localfieldz, &
       macroscopicfieldx, macroscopicfieldy, macroscopicfieldz, &
       polarisa, epsilon, &
       nfft2d, npoynting, deltapoyntingx, deltapoyntingy, &
       Eimagex, Eimagey, Eimagez, Eimageincx, Eimageincy, &
       Eimageincz, Efourierx, Efouriery, Efourierz, Efourierincx, &
       Efourierincy, Efourierincz, kxy, xy, numaper, numaperinc, &
       numaperinc2, kcnax, kcnay, numaperil, gross, zlens, psiinc, &
       ntypemic, nside, &
                                ! tableaux utilises que dans cdmlib
                                ! taille double complex (3*nxm*nym*nzm)
       FF, FF0, FFloc, xr, xi, FFprecon, &
                                ! taille double complex (3*nxm*nym*nzm,12)
       wrk, nlar, &
                                ! taille double complex (8*nxm*nym*nzm)
       FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, &
        FFTTENSORzz,  &
                                ! taille double complex (nfft2d,nfft2d,3)
       Ediffkzpos, Ediffkzneg, sdetnn, &
                                ! taille entier (nxm*nym*nzm)
       Tabdip, Tabmulti, tabfft2 )
  ! Optional PML diagnostic: compare every *masked* incident-field vector
  ! after returning from cdmlib, not only its maximum or final sections.
  geometry_only_env=''
  call get_environment_variable('IFDDA_TEST_DUMP_PML_FIELD',geometry_only_env,status=ierror)
  if (ierror == 0 .and. trim(geometry_only_env) == '1' .and. nstop == 0 .and. ntypeabc == 1) then
     open(unit=904,file='ifdda_pml_field_debug.txt',status='replace',action='write',iostat=ierror)
     if (ierror == 0) then
        write(904,'(I0)') nbsphere
        do i=1,nbsphere
           k=Tabdip(i)
           if (k < 1 .or. 3*k > size(FF0)) then
              write(*,*) 'INVALID_PML_TABDIP',i,k
              exit
           end if
           write(904,'(2(I0,1X),*(ES25.16E3,1X))') i,k, &
                (real(FF0(3*(k-1)+j),wp),aimag(FF0(3*(k-1)+j)),j=1,3)
        end do
        close(904)
        write(*,*) 'CDMLIB_PML_FIELD_DUMP ifdda_pml_field_debug.txt'
     end if
  end if
  ! Optional common-driver diagnostic: identical Fortran source dumps the
  ! post-calculation dipole map, coordinates and all epsilon/alpha tensors.
  ! Both the Fortran and C++/CUDA executables use this exact same writer.
  geometry_only_env=''
  call get_environment_variable('IFDDA_TEST_DUMP_MATERIAL',geometry_only_env,status=ierror)
  if (ierror == 0 .and. trim(geometry_only_env) == '1') then
     open(unit=903,file='ifdda_material_debug.txt',status='replace',action='write',iostat=ierror)
     if (ierror == 0) then
        write(903,'(3(I0,1X))') nbsphere,ndipole,niso
        do i=1,min(nbsphere,size(epsilon,1))
           write(903,'(2(I0,1X),*(ES25.16E3,1X))') i,Tabdip(i),xs(i),ys(i),zs(i), &
                ((real(epsilon(i,j,k),wp),aimag(epsilon(i,j,k)), &
                  real(polarisa(i,j,k),wp),aimag(polarisa(i,j,k)),j=1,niso),k=1,niso)
        end do
        close(903)
        write(*,*) 'CDMLIB_MATERIAL_DUMP ifdda_material_debug.txt'
     end if
  end if
  write(*,'(A,2(I0,1X))') 'CDMLIB_GEOMETRY_RESULT ',nbsphere,ndipole
  ii=0
  geometry_mask_hash=0_int64
  do i=1,min(nbsphere,size(epsilon,1))
     if(abs(epsilon(i,1,1))>0.0_wp .and. &
        abs(epsilon(i,1,1)-cmplx(1.0_wp,0.0_wp,kind=wp))>1.0e-8_wp) then
        ii=ii+1
        geometry_mask_hash=modulo(geometry_mask_hash*131_int64+1_int64,1000000007_int64)
     else
        geometry_mask_hash=modulo(geometry_mask_hash*131_int64,1000000007_int64)
     end if
  end do
  write(*,'(A,I0)') 'CDMLIB_MATERIAL_RESULT ',ii
  write(*,'(A,I0)') 'CDMLIB_MATERIAL_HASH ',geometry_mask_hash
  fieldmax = zero
  localmax = zero
  if (nstop == 0) then
     fieldmax = max(maxval(abs(incidentfieldx)), maxval(abs(incidentfieldy)), &
                    maxval(abs(incidentfieldz)))
     localmax = max(maxval(abs(localfieldx)), maxval(abs(localfieldy)), &
                    maxval(abs(localfieldz)))
  end if
  forcearrmax = max(maxval(abs(forcex)), maxval(abs(forcey)), maxval(abs(forcez)))
  torquearrmax = max(maxval(abs(torquex)), maxval(abs(torquey)), maxval(abs(torquez)))
  write(*,*) "CDMLIB_RESULT", nstop, ncompte, nloop, Cext, Csca, Cabs, tol1, &
       ndipole, fieldmax, localmax, forcem, couplem, efficacite, forcearrmax, torquearrmax

  deallocate(sdetnn)
  deallocate(FFprecon)
  deallocate(thetafield,phifield)
  deallocate(poyntingfield,poyntingfieldneg,poyntingfieldpos)
  deallocate(xs, ys, zs, xswf, yswf,zswf)
  deallocate(polarisa, epsilon)
  deallocate(incidentfield, localfield, macroscopicfield)
  deallocate(forcex, forcey, forcez)
  deallocate(torquex, torquey, torquez)
  deallocate(macroscopicfieldx, macroscopicfieldy)
  deallocate(macroscopicfieldz)
  deallocate(localfieldx, localfieldy, localfieldz)
  deallocate(incidentfieldx, incidentfieldy, incidentfieldz)
  deallocate(FFloc)
  deallocate(FF0)
  deallocate(FF)  
  deallocate(Tabdip, Tabmulti)
  deallocate(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz)
  deallocate(FFTTENSORyy, FFTTENSORyz, FFTTENSORzz)
  deallocate(xr, xi)
  deallocate(wrk)
  deallocate(tabfft2)
  deallocate(kxy,xy,kxypoynting)
  deallocate(Eimagex,Eimagey)
  deallocate(Eimagez,Eimageincx)
  deallocate(Eimageincy,Eimageincz)
  deallocate(Efourierx,Efouriery)
  deallocate(Efourierz,Efourierincx)
  deallocate(Efourierincy,Efourierincz)
  deallocate(Ediffkzpos,Ediffkzneg)

  
  if (nstop == 0) then
     write(*,*) '***********************************************'
     write(*,*) 'Computation finished without problem:'
     write(*,*) '***********************************************'
  else
     write(*,*) '***********************************************'
     write(*,*) 'Programm finished with problem:'
     write(*,*) infostr
     write(*,*) '***********************************************'
  end if

END PROGRAM main
