subroutine cdmlib( &
                                ! input file cdm.in
     lambda,eps0i,ntypeabc,lnrbc,lcos,beam,object,trope,niso, &
     materiaumulti, nnnr, tolinit, nlim, methodeit, polarizability, nprecon, ninitest, nquad, nlecture, filereread, nmat, h5file, &
                                !output file cdm.out
     nlocal, nmacro, nsection, nsectionsca, nquickdiffracte, nrig, &
     nforce, nforced, ntorque, ntorqued, nproche, nlentille, nquicklens, &
     nenergie, nobjet, &
                                ! cube, sphere (includes multiple)
     density, side, sidex, sidey, sidez, hauteur, &
     numberobjet, rayonmulti, xgmulti, ygmulti, zgmulti, &
     epsmulti, epsanimulti, lc, hc, ng, &
                                ! ellipsoid+arbitrary
     demiaxea, demiaxeb, demiaxec, thetaobj, phiobj, psiobj, &
     namefileobj, &
                                ! planewavecircular.in / planewavelinear.in files
     theta, phi, pp, ss, P0, w0, xgaus, ygaus, zgaus, speckseed, &
     namefileinc, &
                                ! ondeplane multiple
     thetam, phim, ppm, ssm, E0m, nbinc, &
                                ! return info stringf
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
     torquex, torquey, torquez, torquexmulti, torqueymulti, &
     torquezmulti, &
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
     FF, FF0, FFloc, xr, xi, FFprecon, &
     wrk, nlar, &
     FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, &
     FFTTENSORzz,  Ediffkzpos, Ediffkzneg, sdetnn, &
     Tabdip, Tabmulti, tabfft2 &
     )
  use, intrinsic :: ISO_FORTRAN_ENV
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use precision
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  use, intrinsic :: iso_c_binding, only: c_int, c_char, c_float, c_double, &
       c_float_complex, c_double_complex
#endif
#ifdef USE_HDF5
  use HDF5
#endif

  implicit none

#ifdef IFDDA_CUDA_INVERSE_DRIVER
  interface
     subroutine ifdda_set_initial_guess_mode_cuda(mode) &
          bind(C,name="ifdda_set_initial_guess_mode_all_cuda_")
       import :: c_int
       integer(c_int), value, intent(in) :: mode
     end subroutine ifdda_set_initial_guess_mode_cuda
     subroutine ifdda_compute_cross_sections_cuda(incident,polarization, &
          local_field,ndipoles,k0,intensity,extinction,absorption,scattering,status) &
          bind(C,name="ifdda_compute_cross_sections_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: incident(*),polarization(*),local_field(*)
       real(c_float), intent(in) :: k0,intensity
       real(c_float), intent(out) :: extinction,absorption,scattering
#else
       complex(c_double_complex), intent(in) :: incident(*),polarization(*),local_field(*)
       real(c_double), intent(in) :: k0,intensity
       real(c_double), intent(out) :: extinction,absorption,scattering
#endif
       integer(c_int), intent(in) :: ndipoles
       integer(c_int), intent(out) :: status
     end subroutine ifdda_compute_cross_sections_cuda
     subroutine clear_device_result_vector_alias_cuda(field) &
          bind(C,name="clear_device_result_vector_alias")
       import :: c_float_complex,c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: field(*)
#else
       complex(c_double_complex), intent(in) :: field(*)
#endif
     end subroutine clear_device_result_vector_alias_cuda
     subroutine ifdda_incident_arbitrary_cuda(xs,ys,zs,cube_size,output, &
          nxm,nym,nzm,dipole_count,direct_device_consumer,status, &
          namefile,info) &
          bind(C,name="ifdda_incident_arbitrary_cuda_")
       import :: c_int,c_char,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),cube_size
       complex(c_float_complex), intent(out) :: output(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),cube_size
       complex(c_double_complex), intent(out) :: output(*)
#endif
       integer(c_int), intent(in) :: nxm,nym,nzm,dipole_count,direct_device_consumer
       integer(c_int), intent(out) :: status
       character(kind=c_char), intent(in) :: namefile(*)
       character(kind=c_char), intent(out) :: info(*)
     end subroutine ifdda_incident_arbitrary_cuda
     subroutine ifdda_direct_scattering_cuda(field,xs,ys,zs,ndipoles, &
          ntheta,nphi,nfft,k0,theta,phi,intensity,const_poynting, &
          angular_poynting,theta_values,phi_values,positive_spectrum, &
          negative_spectrum,kx_values,asymmetry,scattering,status) &
          bind(C,name="ifdda_direct_scattering_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: field(*)
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),k0,theta,phi,intensity,const_poynting
       real(c_float), intent(out) :: angular_poynting(*),theta_values(*),phi_values(*)
       real(c_float), intent(out) :: positive_spectrum(*),negative_spectrum(*),kx_values(*)
       real(c_float), intent(out) :: asymmetry,scattering
#else
       complex(c_double_complex), intent(in) :: field(*)
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),k0,theta,phi,intensity,const_poynting
       real(c_double), intent(out) :: angular_poynting(*),theta_values(*),phi_values(*)
       real(c_double), intent(out) :: positive_spectrum(*),negative_spectrum(*),kx_values(*)
       real(c_double), intent(out) :: asymmetry,scattering
#endif
       integer(c_int), intent(in) :: ndipoles,ntheta,nphi,nfft
       integer(c_int), intent(out) :: status
     end subroutine ifdda_direct_scattering_cuda
     subroutine ifdda_quick_diffraction_cuda(nx,ny,nz,nfft,fft_map,k0, &
          xs,ys,zs,spacing,field,ntheta,nphi,intensity,theta,phi, &
          const_poynting,imax,deltakx,deltaky,positive,negative,kx, &
          angular_poynting,theta_values,phi_values,asymmetry,scattering,status) &
          bind(C,name="ifdda_quick_diffraction_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
       integer(c_int), intent(in) :: nx,ny,nz,nfft
       integer(c_int), intent(inout) :: fft_map(*)
       integer(c_int), intent(in) :: ntheta,nphi
       integer(c_int), intent(out) :: imax,status
#ifdef USE_SIMP
       real(c_float), intent(in) :: k0,xs(*),ys(*),zs(*),spacing,intensity,theta,phi,const_poynting
       real(c_float), intent(out) :: deltakx,deltaky
       complex(c_float_complex), intent(in) :: field(*)
       real(c_float), intent(out) :: positive(*),negative(*),kx(*)
       real(c_float), intent(out) :: angular_poynting(*),theta_values(*),phi_values(*)
       real(c_float), intent(out) :: asymmetry,scattering
#else
       real(c_double), intent(in) :: k0,xs(*),ys(*),zs(*),spacing,intensity,theta,phi,const_poynting
       real(c_double), intent(out) :: deltakx,deltaky
       complex(c_double_complex), intent(in) :: field(*)
       real(c_double), intent(out) :: positive(*),negative(*),kx(*)
       real(c_double), intent(out) :: angular_poynting(*),theta_values(*),phi_values(*)
       real(c_double), intent(out) :: asymmetry,scattering
#endif
     end subroutine ifdda_quick_diffraction_cuda
     subroutine ifdda_energy_conservation_plane_cuda(nx,ny,nz,nfft,fft_map, &
          k0,xs,ys,zs,spacing,epsilon,e0,ss,pp,theta,phi,circular,field, &
          imax,dkx,dky,efficiency,reflectivity,transmissivity, &
          incident_flux_w,reflected_flux_w,transmitted_flux_w, &
          energy_outside_percent,status) &
          bind(C,name="ifdda_energy_conservation_plane_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
       integer(c_int), intent(in) :: nx,ny,nz,nfft,circular
       integer(c_int), intent(inout) :: fft_map(*)
       integer(c_int), intent(out) :: imax,status
#ifdef USE_SIMP
       real(c_float), intent(in) :: k0,xs(*),ys(*),zs(*),spacing,ss,pp,theta,phi
       complex(c_float_complex), intent(in) :: epsilon,e0,field(*)
       real(c_float), intent(out) :: dkx,dky,efficiency,reflectivity,transmissivity
       real(c_float), intent(out) :: incident_flux_w,reflected_flux_w,transmitted_flux_w
       real(c_float), intent(out) :: energy_outside_percent
#else
       real(c_double), intent(in) :: k0,xs(*),ys(*),zs(*),spacing,ss,pp,theta,phi
       complex(c_double_complex), intent(in) :: epsilon,e0,field(*)
       real(c_double), intent(out) :: dkx,dky,efficiency,reflectivity,transmissivity
       real(c_double), intent(out) :: incident_flux_w,reflected_flux_w,transmitted_flux_w
       real(c_double), intent(out) :: energy_outside_percent
#endif
     end subroutine ifdda_energy_conservation_plane_cuda
     subroutine ifdda_apply_pml_mask_cuda(xs,ys,zs,epsilon,k0,pml_work, &
          mask_type,dipole_count,spacing,field,mask,status) &
          bind(C,name="ifdda_apply_pml_mask_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),k0,spacing
       complex(c_float_complex), intent(in) :: epsilon
       complex(c_float_complex), intent(inout) :: field(*)
       real(c_float), intent(out) :: mask(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),k0,spacing
       complex(c_double_complex), intent(in) :: epsilon
       complex(c_double_complex), intent(inout) :: field(*)
       real(c_double), intent(out) :: mask(*)
#endif
       integer(c_int), intent(in) :: pml_work(*),mask_type,dipole_count
       integer(c_int), intent(out) :: status
     end subroutine ifdda_apply_pml_mask_cuda
     subroutine ifdda_compute_optical_force_cuda(polarization,derivatives, &
          ndipoles,mesh_size,scale,force_x,force_y,force_z,force_total,status) &
          bind(C,name="ifdda_compute_optical_force_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: polarization(*),derivatives(*)
       real(c_float), intent(in) :: scale
       real(c_float), intent(out) :: force_x(*),force_y(*),force_z(*),force_total(3)
#else
       complex(c_double_complex), intent(in) :: polarization(*),derivatives(*)
       real(c_double), intent(in) :: scale
       real(c_double), intent(out) :: force_x(*),force_y(*),force_z(*),force_total(3)
#endif
       integer(c_int), intent(in) :: ndipoles,mesh_size
       integer(c_int), intent(out) :: status
     end subroutine ifdda_compute_optical_force_cuda
     subroutine ifdda_compute_optical_torque_cuda(local_field,xs,ys,zs, &
          ndipoles,k03,scale,center,torque_x,torque_y,torque_z, &
          torque_total,status) &
          bind(C,name="ifdda_compute_optical_torque_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: local_field(*)
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),k03,scale
       real(c_float), intent(out) :: torque_x(*),torque_y(*),torque_z(*),torque_total(3)
       real(c_float), intent(out) :: center(3)
#else
       complex(c_double_complex), intent(in) :: local_field(*)
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),k03,scale
       real(c_double), intent(out) :: torque_x(*),torque_y(*),torque_z(*),torque_total(3)
       real(c_double), intent(out) :: center(3)
#endif
       integer(c_int), intent(in) :: ndipoles
       integer(c_int), intent(out) :: status
     end subroutine ifdda_compute_optical_torque_cuda
     subroutine ifdda_sum_optical_force_by_object_cuda(object_ids, &
          ndipoles,nobjects,sum_x,sum_y,sum_z,status) &
          bind(C,name="ifdda_sum_optical_force_by_object_cuda_")
       import :: c_int,c_float,c_double
#ifdef USE_SIMP
       real(c_float), intent(out) :: sum_x(*),sum_y(*),sum_z(*)
#else
       real(c_double), intent(out) :: sum_x(*),sum_y(*),sum_z(*)
#endif
       integer(c_int), intent(in) :: object_ids(*),ndipoles,nobjects
       integer(c_int), intent(out) :: status
     end subroutine ifdda_sum_optical_force_by_object_cuda
     subroutine ifdda_sum_optical_torque_by_object_cuda(object_ids, &
          ndipoles,nobjects,sum_x,sum_y,sum_z,status) &
          bind(C,name="ifdda_sum_optical_torque_by_object_cuda_")
       import :: c_int,c_float,c_double
#ifdef USE_SIMP
       real(c_float), intent(out) :: sum_x(*),sum_y(*),sum_z(*)
#else
       real(c_double), intent(out) :: sum_x(*),sum_y(*),sum_z(*)
#endif
       integer(c_int), intent(in) :: object_ids(*),ndipoles,nobjects
       integer(c_int), intent(out) :: status
     end subroutine ifdda_sum_optical_torque_by_object_cuda
     subroutine ifdda_derive_incident_field_cuda(field,nx,ny,nz,nmax, &
          spacing,axis,derivative,status) &
          bind(C,name="ifdda_derive_incident_field_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: field(*)
       complex(c_float_complex), intent(out) :: derivative(*)
       real(c_float), intent(in) :: spacing
#else
       complex(c_double_complex), intent(in) :: field(*)
       complex(c_double_complex), intent(out) :: derivative(*)
       real(c_double), intent(in) :: spacing
#endif
       integer(c_int), intent(in) :: nx,ny,nz,nmax,axis
       integer(c_int), intent(out) :: status
     end subroutine ifdda_derive_incident_field_cuda
     subroutine ifdda_prepare_optical_force_fft_cuda(polarization,dipole_map, &
          nx,ny,nz,nx2,ny2,nz2,use_map,status) &
          bind(C,name="ifdda_prepare_optical_force_fft_cuda_")
       import :: c_int,c_float_complex,c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: polarization(*)
#else
       complex(c_double_complex), intent(in) :: polarization(*)
#endif
       integer(c_int), intent(in) :: dipole_map(*),nx,ny,nz,nx2,ny2,nz2,use_map
       integer(c_int), intent(out) :: status
     end subroutine ifdda_prepare_optical_force_fft_cuda
     subroutine ifdda_compute_force_green_derivative_cuda(spacing,k0,nx,ny,nz, &
          nx2,ny2,nz2,nxy2,axis,output,status) &
          bind(C,name="ifdda_compute_force_green_derivative_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: spacing,k0
       complex(c_float_complex), intent(out) :: output(*)
#else
       real(c_double), intent(in) :: spacing,k0
       complex(c_double_complex), intent(out) :: output(*)
#endif
       integer(c_int), intent(in) :: nx,ny,nz,nx2,ny2,nz2,nxy2,axis
       integer(c_int), intent(out) :: status
     end subroutine ifdda_compute_force_green_derivative_cuda
     subroutine ifdda_plane_wave_derivatives_cuda(xs,ys,zs,tabdip, &
          coordinate_count,dipole_count,k0,e0,ss,pp,theta,phi,circular, &
          output,status) bind(C,name="ifdda_plane_wave_derivatives_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),k0,ss,pp,theta,phi
       complex(c_float_complex), intent(in) :: e0
       complex(c_float_complex), intent(out) :: output(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),k0,ss,pp,theta,phi
       complex(c_double_complex), intent(in) :: e0
       complex(c_double_complex), intent(out) :: output(*)
#endif
       integer(c_int), intent(in) :: tabdip(*),coordinate_count,dipole_count,circular
       integer(c_int), intent(out) :: status
     end subroutine ifdda_plane_wave_derivatives_cuda
     subroutine ifdda_multi_plane_wave_derivatives_cuda(xs,ys,zs,tabdip, &
          coordinate_count,dipole_count,k0,e0m,pp,thetam,phim,plane_count, &
          output,status) bind(C,name="ifdda_multi_plane_wave_derivatives_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),k0,pp(*),thetam(*),phim(*)
       complex(c_float_complex), intent(in) :: e0m(*)
       complex(c_float_complex), intent(out) :: output(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),k0,pp(*),thetam(*),phim(*)
       complex(c_double_complex), intent(in) :: e0m(*)
       complex(c_double_complex), intent(out) :: output(*)
#endif
       integer(c_int), intent(in) :: tabdip(*),coordinate_count,dipole_count,plane_count
       integer(c_int), intent(out) :: status
     end subroutine ifdda_multi_plane_wave_derivatives_cuda
     subroutine ifdda_paraxial_gaussian_derivatives_cuda(xs,ys,zs,tabdip, &
          coordinate_count,dipole_count,x0,y0,z0,theta,phi,w0,k0,ss,pp,e0, &
          circular,output,status) bind(C,name="ifdda_paraxial_gaussian_derivatives_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),x0,y0,z0,theta,phi,w0,k0,ss,pp
       complex(c_float_complex), intent(in) :: e0
       complex(c_float_complex), intent(out) :: output(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),x0,y0,z0,theta,phi,w0,k0,ss,pp
       complex(c_double_complex), intent(in) :: e0
       complex(c_double_complex), intent(out) :: output(*)
#endif
       integer(c_int), intent(in) :: tabdip(*),coordinate_count,dipole_count,circular
       integer(c_int), intent(out) :: status
     end subroutine ifdda_paraxial_gaussian_derivatives_cuda
     subroutine ifdda_rigorous_gaussian_derivatives_cuda(xs,ys,zs,tabdip, &
          coordinate_count,dipole_count,x0,y0,z0,theta,phi,w0,k0,ss,pp,e0, &
          tolerance,nloin,circular,output,status) &
          bind(C,name="ifdda_rigorous_gaussian_derivatives_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),x0,y0,z0,theta,phi,w0,k0,ss,pp,tolerance
       complex(c_float_complex), intent(in) :: e0
       complex(c_float_complex), intent(out) :: output(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),x0,y0,z0,theta,phi,w0,k0,ss,pp,tolerance
       complex(c_double_complex), intent(in) :: e0
       complex(c_double_complex), intent(out) :: output(*)
#endif
       integer(c_int), intent(in) :: tabdip(*),coordinate_count,dipole_count,nloin,circular
       integer(c_int), intent(out) :: status
     end subroutine ifdda_rigorous_gaussian_derivatives_cuda
     subroutine ifdda_antenna_derivatives_cuda(xs,ys,zs,tabdip, &
          coordinate_count,dipole_count,xdip,ydip,zdip,theta,phi,aretecube,k0,e0, &
          output,status) bind(C,name="ifdda_antenna_derivatives_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),xdip,ydip,zdip,theta,phi,aretecube,k0
       complex(c_float_complex), intent(in) :: e0
       complex(c_float_complex), intent(out) :: output(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),xdip,ydip,zdip,theta,phi,aretecube,k0
       complex(c_double_complex), intent(in) :: e0
       complex(c_double_complex), intent(out) :: output(*)
#endif
       integer(c_int), intent(in) :: tabdip(*),coordinate_count,dipole_count
       integer(c_int), intent(out) :: status
     end subroutine ifdda_antenna_derivatives_cuda
     subroutine ifdda_arbitrary_derivatives_cuda(xs,ys,zs,tabdip, &
          coordinate_count,dipole_count,filename,filename_length,output,status) &
          bind(C,name="ifdda_arbitrary_derivatives_cuda_")
       import :: c_char,c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*)
       complex(c_float_complex), intent(out) :: output(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*)
       complex(c_double_complex), intent(out) :: output(*)
#endif
       character(kind=c_char), intent(in) :: filename(*)
       integer(c_int), intent(in) :: tabdip(*),coordinate_count,dipole_count,filename_length
       integer(c_int), intent(out) :: status
     end subroutine ifdda_arbitrary_derivatives_cuda
     subroutine ifdda_plane_wave_field_cuda(xs,ys,zs,count,k0,eps0,e0, &
          ss,pp,theta,phi,circular,field,status) &
          bind(C,name="ifdda_plane_wave_field_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),k0,ss,pp,theta,phi
       complex(c_float_complex), intent(in) :: eps0,e0
       complex(c_float_complex), intent(out) :: field(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),k0,ss,pp,theta,phi
       complex(c_double_complex), intent(in) :: eps0,e0
       complex(c_double_complex), intent(out) :: field(*)
#endif
       integer(c_int), intent(in) :: count,circular
       integer(c_int), intent(out) :: status
     end subroutine ifdda_plane_wave_field_cuda
     subroutine ifdda_multi_plane_wave_field_cuda(xs,ys,zs,count,k0,e0,e0m, &
          ssm,ppm,thetam,phim,plane_count,field,status) &
          bind(C,name="ifdda_multi_plane_wave_field_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),k0,ssm(*),ppm(*),thetam(*),phim(*)
       complex(c_float_complex), intent(in) :: e0,e0m(*)
       complex(c_float_complex), intent(out) :: field(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),k0,ssm(*),ppm(*),thetam(*),phim(*)
       complex(c_double_complex), intent(in) :: e0,e0m(*)
       complex(c_double_complex), intent(out) :: field(*)
#endif
       integer(c_int), intent(in) :: count,plane_count
       integer(c_int), intent(out) :: status
     end subroutine ifdda_multi_plane_wave_field_cuda
     subroutine ifdda_paraxial_gaussian_field_cuda(xs,ys,zs,count,e0,eps0, &
          ss,pp,theta,phi,x0,y0,z0,w0,k0,circular,field,status) &
          bind(C,name="ifdda_paraxial_gaussian_field_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),ss,pp,theta,phi,x0,y0,z0,w0,k0
       complex(c_float_complex), intent(in) :: e0,eps0
       complex(c_float_complex), intent(out) :: field(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),ss,pp,theta,phi,x0,y0,z0,w0,k0
       complex(c_double_complex), intent(in) :: e0,eps0
       complex(c_double_complex), intent(out) :: field(*)
#endif
       integer(c_int), intent(in) :: count,circular
       integer(c_int), intent(out) :: status
     end subroutine ifdda_paraxial_gaussian_field_cuda
     subroutine ifdda_rigorous_gaussian_field_cuda(xs,ys,zs,count,x0,y0,z0, &
          theta,phi,w0,k0,ss,pp,e0,tolerance,circular,nloin,field,status) &
          bind(C,name="ifdda_rigorous_gaussian_field_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),x0,y0,z0,theta,phi,w0,k0,ss,pp,tolerance
       complex(c_float_complex), intent(in) :: e0
       complex(c_float_complex), intent(out) :: field(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),x0,y0,z0,theta,phi,w0,k0,ss,pp,tolerance
       complex(c_double_complex), intent(in) :: e0
       complex(c_double_complex), intent(out) :: field(*)
#endif
       integer(c_int), intent(in) :: count,circular,nloin
       integer(c_int), intent(out) :: status
     end subroutine ifdda_rigorous_gaussian_field_cuda
     subroutine ifdda_antenna_field_cuda(xs,ys,zs,count,xdip,ydip,zdip, &
          aretecube,k0,e0,theta,phi,reference_x,reference_y,reference_z, &
          field,reference_field,status) bind(C,name="ifdda_antenna_field_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: xs(*),ys(*),zs(*),xdip,ydip,zdip,aretecube,k0
       real(c_float), intent(in) :: theta,phi,reference_x,reference_y,reference_z
       complex(c_float_complex), intent(in) :: e0
       complex(c_float_complex), intent(out) :: field(*),reference_field(*)
#else
       real(c_double), intent(in) :: xs(*),ys(*),zs(*),xdip,ydip,zdip,aretecube,k0
       real(c_double), intent(in) :: theta,phi,reference_x,reference_y,reference_z
       complex(c_double_complex), intent(in) :: e0
       complex(c_double_complex), intent(out) :: field(*),reference_field(*)
#endif
       integer(c_int), intent(in) :: count
       integer(c_int), intent(out) :: status
     end subroutine ifdda_antenna_field_cuda
     subroutine ifdda_speckle_field_cuda(nx,ny,nz,dipole_count,nfft,k0, &
          xx0,yy0,zz0,psi,spacing,power,aperture,e0,phase_positive, &
          phase_negative,xs,ys,zs,bidirectional,field,irradiance,status) &
          bind(C,name="ifdda_speckle_field_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: k0,xx0,yy0,zz0,psi,spacing,power,aperture
       real(c_float), intent(in) :: xs(*),ys(*),zs(*)
       complex(c_float_complex), intent(inout) :: e0
       complex(c_float_complex), intent(in) :: phase_positive(*),phase_negative(*)
       complex(c_float_complex), intent(out) :: field(*)
       real(c_float), intent(out) :: irradiance
#else
       real(c_double), intent(in) :: k0,xx0,yy0,zz0,psi,spacing,power,aperture
       real(c_double), intent(in) :: xs(*),ys(*),zs(*)
       complex(c_double_complex), intent(inout) :: e0
       complex(c_double_complex), intent(in) :: phase_positive(*),phase_negative(*)
       complex(c_double_complex), intent(out) :: field(*)
       real(c_double), intent(out) :: irradiance
#endif
       integer(c_int), intent(in) :: nx,ny,nz,dipole_count,nfft,bidirectional
       integer(c_int), intent(out) :: status
     end subroutine ifdda_speckle_field_cuda
     subroutine ifdda_confocal_field_cuda(nx,ny,nz,dipole_count,nfft,k0, &
          xx0,yy0,zz0,psi,spacing,power,aperture,e0,xs,ys,zs, &
          bidirectional,field,irradiance,status) &
          bind(C,name="ifdda_confocal_field_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       real(c_float), intent(in) :: k0,xx0,yy0,zz0,psi,spacing,power,aperture
       real(c_float), intent(in) :: xs(*),ys(*),zs(*)
       complex(c_float_complex), intent(inout) :: e0
       complex(c_float_complex), intent(out) :: field(*)
       real(c_float), intent(out) :: irradiance
#else
       real(c_double), intent(in) :: k0,xx0,yy0,zz0,psi,spacing,power,aperture
       real(c_double), intent(in) :: xs(*),ys(*),zs(*)
       complex(c_double_complex), intent(inout) :: e0
       complex(c_double_complex), intent(out) :: field(*)
       real(c_double), intent(out) :: irradiance
#endif
       integer(c_int), intent(in) :: nx,ny,nz,dipole_count,nfft,bidirectional
       integer(c_int), intent(out) :: status
     end subroutine ifdda_confocal_field_cuda
     subroutine ifdda_renormalized_born_cuda(incident,polarizability, &
          local_field,polarized_field,nmax,niso,ndipoles,nstop) &
          bind(C,name="ifdda_renormalized_born_all_cuda_")
       import :: c_int, c_float, c_double, c_float_complex, c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: incident(*),polarizability(*)
       complex(c_float_complex), intent(out) :: local_field(*),polarized_field(*)
#else
       complex(c_double_complex), intent(in) :: incident(*),polarizability(*)
       complex(c_double_complex), intent(out) :: local_field(*),polarized_field(*)
#endif
       integer(c_int), intent(in) :: nmax,niso,ndipoles
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_renormalized_born_cuda
     subroutine ifdda_born_cuda(incident,epsilon,polarizability, &
          local_field,polarized_field,nmax,niso,ndipoles,aretecube,k0,nstop) &
          bind(C,name="ifdda_born_all_cuda_")
       import :: c_int, c_float, c_double, c_float_complex, c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: incident(*),epsilon(*),polarizability(*)
       complex(c_float_complex), intent(out) :: local_field(*),polarized_field(*)
       real(c_float), intent(in) :: aretecube,k0
#else
       complex(c_double_complex), intent(in) :: incident(*),epsilon(*),polarizability(*)
       complex(c_double_complex), intent(out) :: local_field(*),polarized_field(*)
       real(c_double), intent(in) :: aretecube,k0
#endif
       integer(c_int), intent(in) :: nmax,niso,ndipoles
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_born_cuda
     subroutine ifdda_vector_copy_cuda(input,output,count,negate,nstop) &
          bind(C,name="ifdda_vector_copy_all_cuda_")
       import :: c_int, c_float_complex, c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: input(*)
       complex(c_float_complex), intent(out) :: output(*)
#else
       complex(c_double_complex), intent(in) :: input(*)
       complex(c_double_complex), intent(out) :: output(*)
#endif
       integer(c_int), intent(in) :: count,negate
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_vector_copy_cuda
     subroutine ifdda_vector_zero_cuda(output,count,nstop) &
          bind(C,name="ifdda_vector_zero_all_cuda_")
       import :: c_int, c_float_complex, c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(out) :: output(*)
#else
       complex(c_double_complex), intent(out) :: output(*)
#endif
       integer(c_int), intent(in) :: count
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_vector_zero_cuda
     subroutine ifdda_vector_negate_cuda(values,count,nstop) &
          bind(C,name="ifdda_vector_negate_all_cuda_")
       import :: c_int, c_float_complex, c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(inout) :: values(*)
#else
       complex(c_double_complex), intent(inout) :: values(*)
#endif
       integer(c_int), intent(in) :: count
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_vector_negate_cuda
     subroutine ifdda_rytov_macro_prepare_cuda(incident,epsilon, &
          polarizability,background_epsilon,macroscopic_local_field, &
          delta_field,negative_polarized_field,nmax,niso,ndipoles, &
          aretecube,k0,nstop) bind(C,name="ifdda_rytov_macro_prepare_all_cuda_")
       import :: c_int, c_float, c_double, c_float_complex, c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: incident(*),epsilon(*),polarizability(*)
       complex(c_float_complex), intent(in) :: background_epsilon
       complex(c_float_complex), intent(out) :: macroscopic_local_field(*)
       complex(c_float_complex), intent(out) :: delta_field(*),negative_polarized_field(*)
       real(c_float), intent(in) :: aretecube,k0
#else
       complex(c_double_complex), intent(in) :: incident(*),epsilon(*),polarizability(*)
       complex(c_double_complex), intent(in) :: background_epsilon
       complex(c_double_complex), intent(out) :: macroscopic_local_field(*)
       complex(c_double_complex), intent(out) :: delta_field(*),negative_polarized_field(*)
       real(c_double), intent(in) :: aretecube,k0
#endif
       integer(c_int), intent(in) :: nmax,niso,ndipoles
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_rytov_macro_prepare_cuda
     subroutine ifdda_local_macro_batch_cuda(epsilon,input_field, &
          output_x,output_y,output_z,field_magnitude,nmax,niso,ndipoles, &
          aretecube,k0,nsens,nstop) &
          bind(C,name="ifdda_local_macro_batch_cuda_")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(in) :: epsilon(*),input_field(*)
       complex(c_float_complex), intent(out) :: output_x(*),output_y(*),output_z(*)
       real(c_float), intent(out) :: field_magnitude(*)
       real(c_float), intent(in) :: aretecube,k0
#else
       complex(c_double_complex), intent(in) :: epsilon(*),input_field(*)
       complex(c_double_complex), intent(out) :: output_x(*),output_y(*),output_z(*)
       real(c_double), intent(out) :: field_magnitude(*)
       real(c_double), intent(in) :: aretecube,k0
#endif
       integer(c_int), intent(in) :: nmax,niso,ndipoles,nsens
       integer(c_int), intent(out) :: nstop
     end subroutine ifdda_local_macro_batch_cuda
     subroutine ifdda_generate_vector_green_cuda(gxx,gxy,gxz,gyy,gyz,gzz, &
          nx,ny,nz,nxm,nym,nzm,nx2,ny2,nz2,aretecube,k0,eps0,medium,nquad,filtered,status) &
          bind(C,name="ifdda_generate_vector_green_cuda")
       import :: c_int,c_float,c_double,c_float_complex,c_double_complex
#ifdef USE_SIMP
       complex(c_float_complex), intent(out) :: gxx(*),gxy(*),gxz(*),gyy(*),gyz(*),gzz(*)
       complex(c_float_complex), intent(in) :: eps0
       real(c_float), intent(in) :: aretecube,k0
#else
       complex(c_double_complex), intent(out) :: gxx(*),gxy(*),gxz(*),gyy(*),gyz(*),gzz(*)
       complex(c_double_complex), intent(in) :: eps0
       real(c_double), intent(in) :: aretecube,k0
#endif
       integer(c_int), intent(in) :: nx,ny,nz,nxm,nym,nzm,nx2,ny2,nz2
       integer(c_int), intent(in) :: medium,nquad,filtered
       integer(c_int), intent(out) :: status
     end subroutine ifdda_generate_vector_green_cuda
     subroutine ifdda_reset_cuda_transfer_stats() bind(C,name="ifdda_reset_cuda_transfer_stats")
     end subroutine ifdda_reset_cuda_transfer_stats
     subroutine ifdda_print_cuda_transfer_stats() bind(C,name="ifdda_print_cuda_transfer_stats")
     end subroutine ifdda_print_cuda_transfer_stats
  end interface
#endif

  ! integer
  integer(ip) :: ii, jj, kk, ll, i, j, k, cnt, nstop, cntwf, kkm, jjm, iim, ntmp
  integer(ip) :: nlocal, nmacro, nsection, nsectionsca, nforce, nforced
  integer(ip) :: nquickdiffracte, ntorque, ntorqued, nsens, nproche, nprochefft
  integer(ip) :: nlecture, nlecture1, long, long1, ierror, nrig, nquicklens, nquad
  integer(ip) :: nenergie, nmat, ntypemic, nside, npolainc, nprecon, ninitest, ntypeabc, ntypemask, nabc, nabs, pmlwrk(5)
  integer(ip) :: imaxinc, nbitermic
  logical :: cuda_scalar_guess, cuda_vector_direct_guess
  real(c_double) :: cuda_green_norm
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  integer(c_int) :: cuda_nmax,cuda_niso,cuda_ndipoles,cuda_nstop,cuda_count,cuda_negate
  integer(c_int) :: cuda_arbitrary_status,cuda_arbitrary_direct
   integer(c_int) :: cuda_green_status,cuda_green_filtered,cuda_initial_guess_mode
   integer(c_int) :: cuda_scalar_initial_mode
  integer :: cuda_bpm_trace_status
  integer(c_int) :: cuda_energy_imax,cuda_energy_circular,cuda_energy_status
#ifdef USE_SIMP
  real(c_float) :: cuda_aretecube,cuda_k0
#else
  real(c_double) :: cuda_aretecube,cuda_k0
#endif
#endif
  integer(ip), parameter :: nmasque = 100
  complex(wp) :: masque(nmasque*nmasque)
  ! variables for the object
  integer(ip) :: nbsphere3, nbsphere, ndipole, test, testgreen, numberobjet, is, ng, speckseed
  integer(ip) :: nx, ny, nz, nx2, ny2, nxy2, nz2, nxm, nym, nzm, nxmp, nymp, nzmp
  integer(ip) ::  nxm2, nym2, nzm2, nxym2, nxmpp, nympp, nzmpp, nmaxpp, nxym1
  integer(ip) :: nzm1, nzms1
  integer(ip) :: subunit, nsubunit, comparaison
  ! definition of the size for the code
  integer(ip) :: nmax
  integer(ip) :: ntotal, ntotalm
  
  ! variables for the positions
  real(wp) :: x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, rayon, side
  real(wp) :: sidex, sidey, sidez, hauteur, density, xgmulti(numberobjet)
  real(wp) :: ygmulti(numberobjet), zgmulti(numberobjet)
  real(wp) :: rayonmulti(numberobjet), demiaxea, demiaxeb, demiaxec, thetaobj
  real(wp) :: phiobj, psiobj, constpoynting
  real(wp) :: aretecube, const
  integer(ip) :: iphi, itheta, nphi, ntheta
  real(wp), dimension(nxm*nym*nzm) :: xs, ys, zs
  real(wp), dimension(nxm*nym*nzm) :: xswf, yswf, zswf

  real(wp) :: lambda, lambda10n, k0, k02, k03, epi, epr

  ! variables for the material
  integer(ip) :: niso
  real(wp) :: eps0i, sldr,indiceimag,lnrbc,lcos
  complex(wp) :: eps0
  complex(wp), dimension(nxm*nym*nzm, niso, niso) :: polarisa, epsilon
  complex(wp) :: eps, epsani(3, 3), epsmulti(numberobjet)
  complex(wp) :: epsanimulti(3, 3, numberobjet)
  character(len=2) :: polarizability
  character(len=64), dimension(numberobjet) :: materiaumulti
  character(len=64) :: materiau, object, beam, namefileobj, namefileinc
  character(len=64) :: filereread, filereread1
  character(len=3) :: trope, file1

  ! variables for the incident field and local field
  real(wp), dimension(nxm*nym*nzm) :: incidentfield, localfield, macroscopicfield
  real(wp), dimension(nxm*nym*nzm*nforce) :: forcex, forcey, forcez
  real(wp), dimension(nxm*nym*nzm*ntorque) :: torquex, torquey, torquez
  real(wp) :: forcexmulti(numberobjet), forceymulti(numberobjet), forcezmulti(numberobjet)
  real(wp) :: torquexmulti(numberobjet), torqueymulti(numberobjet), torquezmulti(numberobjet)
  integer(ip) :: nbinc
  real(wp) :: thetam(10), phim(10), ppm(10), ssm(10)
  complex(wp) :: E0m(10)
  real(wp) :: ss, pp, theta, phi, I0, Emod, tmp, Emod11, Emod22, Emod12, Emod21
  complex(wp) :: Eloc(3), Em(3), E0, Emx, Emy, Emz
  complex(wp) :: Uincx, Uincy, Uincz, pinc(3)
  complex(wp), dimension(nxm*nym*nzm) :: macroscopicfieldx, macroscopicfieldy, macroscopicfieldz
  complex(wp), dimension(nxm*nym*nzm) :: localfieldx, localfieldy, localfieldz
  complex(wp), dimension(nxm*nym*nzm) :: incidentfieldx, incidentfieldy, incidentfieldz
  complex(wp), dimension(3*nxm*nym*nzm) :: FF, FF0, FFloc

  ! double complex, dimension(3*nzm,3*nzm,nxm*nym) :: sdetnn
  complex(wp), dimension(nzms1, nzms1, nxym1) :: sdetnn
  complex(wp), dimension(3*nxym1*nzm1) :: FFprecon

  ! Green function
  integer(ip), dimension(nxm*nym*nzm) :: Tabdip, Tabmulti
  integer(ip) :: indice, indicex, indicey, ntotalf
  complex(wp), dimension((nxm+1)*(nym+1)*(nzm+1)) :: FFTTENSORxx, FFTTENSORxy, FFTTENSORxz
  complex(wp), dimension((nxm+1)*(nym+1)*(nzm+1)) :: FFTTENSORyy, FFTTENSORyz, FFTTENSORzz
  complex(wp), dimension(:), allocatable :: vectx, vecty, vectz
  real(wp),  dimension(:), allocatable ::masquenrbc


  real(wp) :: forcet(3), forcem, forcemie
  real(wp) :: couplet(3), couplem, xg, yg, zg, lc, hc, couplemie
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  real(wp) :: cuda_torque_center(3)
#endif
  complex(wp) :: Eder(3, 3)
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  complex(wp), dimension(:,:), allocatable :: incident_plane_derivatives
  integer(c_int) :: cuda_plane_derivative_circular
#endif

  ! computation of the cross section
  integer(ip) :: imaxk0, npoynting
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  integer(c_int) :: cuda_quick_status
  integer(c_int) :: cuda_quick_nx, cuda_quick_ny, cuda_quick_nz
  integer(c_int) :: cuda_quick_nfft, cuda_quick_imax
  integer(c_int) :: cuda_quick_ntheta, cuda_quick_nphi
#endif
  real(wp) :: Cext, normal(3), deltapoyntingx, deltapoyntingy, Csca, Cscai, Cabs, gasym,deltatheta,deltaphi
  real(wp) :: thetas, phis, u1, u2, MIECEXT, MIECABS, MIECSCA, GSCA
  complex(wp) :: ctmp, ctmp1

  ! variables for the iterative method
  integer(ip) :: ldabi, nlar
  integer(ip) :: nnnr, ncompte
  integer(ip) :: NLIM, nloop
  real(wp) :: NORM, TOL, tolinit, tol1

  complex(wp), dimension(3*nxm*nym*nzm) :: xr, xi
  complex(wp), dimension(3*nxm*nym*nzm, nlar) :: wrk

  ! Poynting vector and energy conservation
  integer(ip) :: i2, j2, ii2, jj2
  real(wp) :: Poyntinginc, efficacite, efficaciteref, efficacitetrans
  real(wp) :: energy_flux_incident, energy_flux_reflected, energy_flux_transmitted
  real(wp) :: energy_outside_percent

  ! Info string
  character(len=64) :: infostr, messagetemps

  ! nouvelle variable a passer en argument d'entree
  ! power et diametre
  real(wp) :: P0, w0, xgaus, ygaus, zgaus, xdip, ydip, zdip
  character(len=12) :: methodeit, colorw

  ! nouvelle variable de sortie Irra
  real(wp) :: irra

  ! nouvelle variable
  integer(ip) :: nloin

  ! variable pour avoir l'image a travers la lentille
  integer(ip) :: nlentille, nobjet, nfft2d, nfft2d2
  real(wp) :: kx, ky, kz, deltakx, deltaky, numaper, numaperinc
  real(wp) :: numaperinc2, numaperil, deltax, gross, zlens, sintmp, costmp, u(3), v(3)
  real(wp) :: kcnax, kcnay, deltakinc, psiinc
  integer(ip) :: tabfft2(nfft2d)
  real(wp), dimension(max((ntheta+1)*nphi, nfft2d*nfft2d)) :: thetafield, phifield, poyntingfield
  real(wp), dimension(max((ntheta+1)*nphi, nfft2d*nfft2d)) :: poyntingfieldpos, poyntingfieldneg
  complex(wp) :: Ediffkzpos(nfft2d, nfft2d, 3), Ediffkzneg(nfft2d, nfft2d, 3), zfocus
  real(wp) :: kxy(nfft2d), xy(nfft2d), kxypoynting(nfft2d)
  complex(wp) :: Eimagex(nfft2d*nfft2d), Eimagey(nfft2d*nfft2d), Eimagez(nfft2d*nfft2d)
  complex(wp) :: Eimageincx(nfft2d*nfft2d), Eimageincy(nfft2d*nfft2d), Eimageincz(nfft2d*nfft2d)
  complex(wp) :: Efourierx(nfft2d*nfft2d), Efouriery(nfft2d*nfft2d), Efourierz(nfft2d*nfft2d)
  complex(wp) :: Efourierincx(nfft2d*nfft2d), Efourierincy(nfft2d*nfft2d), Efourierincz(nfft2d*nfft2d)
  integer(ip) :: dddis, inv
  real(wp) :: t1, t2, ti, tf, tempsreelmvp, tempsreeltotal

  ! variables pour le temps et les plans pour FFTW
  character(len=8) :: date
  character(len=10) :: time
  character(len=5) :: zzone
  integer(ip) :: values(8), values2(8), valuesi(8), valuesf(8)
  integer(ip) :: iret, omp_get_max_threads, howmany, istride, idist, ostride, odist
  integer(8) :: planf, planb, plan2f, plan2b, planfn, planbn
  integer(8) :: planXb, planYb, planZb, planXf, planYf, planZf
  integer(ip) :: FFTW_FORWARD, FFTW_ESTIMATE, FFTW_BACKWARD, FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE, flagfftw
  integer(ip) :: n(1), inembed(1), onembed(1)

  character(len=40) :: name
  character(len=100) :: h5file
  character(len=100) :: datasetname
  character(len=100) :: writecol
  integer(ip) :: debug
  integer :: error
  character(len=120) :: PROG_STRING

#ifndef USE_HDF5
  integer(ip), parameter :: hid_t = 4
#endif
  integer(hid_t) :: file_id
  integer(hid_t) :: group_idopt, group_idmic, group_idnf, group_idof, group_idff, group_iddip, group_idver
  integer(ip) :: dim(4)

  real(wp) :: errmod,errpha,phase,phaseref,phasetrans

  nstop = 0
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  ! Legacy FFTW plan arguments are kept for the CUDA leaf ABI, but CUDA/cuFFT
  ! owns all Fourier plans. Never pass uninitialized plan handles downstream.
  planf = 0_8
  planb = 0_8
  plan2f = 0_8
  plan2b = 0_8
  planXf = 0_8
  planXb = 0_8
  planYf = 0_8
  planYb = 0_8
  planZf = 0_8
  planZb = 0_8
  call ifdda_reset_cuda_transfer_stats()
#endif
  call date_and_time(date, time, zzone, valuesi)
  call cpu_time(ti)




#ifdef USE_FFTW
#ifdef USE_SIMP
  call sfftw_init_threads(iret)
#else
  call dfftw_init_threads(iret)    
#endif    
  if (iret == 0) then
     infostr = 'Unlikely error during thread initialization'
     nstop = 1
     return
  end if
#ifdef USE_SIMP
  call sfftw_plan_with_nthreads(omp_get_max_threads())
#else
  call dfftw_plan_with_nthreads(omp_get_max_threads())
#endif

#endif

  ! change nside to meet the right value to get the sign of kz
  if (nside == 1) nside = -1
  if (nside == 0) nside = 1
  ! FF0 : champ incident
  ! FF  : dipole
  ! FFloc : champ local

  PROG_STRING = "cdm "
#ifdef CDMVERSION
  PROG_STRING = trim(PROG_STRING) // ' ' // CDMVERSION
#endif
  ! #ifdef DEBUG
  ! PROG_STRING = trim(PROG_STRING) // ' debug  '
  ! #endif
#ifdef USE_FFTW
  PROG_STRING = trim(PROG_STRING) // ' with FFTW '
#endif
#ifdef USE_HDF5
  PROG_STRING = trim(PROG_STRING) // ' with HDF5 '
#endif
#ifdef USE_SIMP
  PROG_STRING = trim(PROG_STRING) // ' simple precision '
#else
  PROG_STRING = trim(PROG_STRING) // ' double precision '    
#endif

  ! VÃ©rifier si le type par dÃ©faut est 32 ou 64 bits
    if (kind(ntypeabc) == INT32) then
       PROG_STRING = trim(PROG_STRING) // ' integer 32 bits '  
    else if (kind(ntypeabc) == INT64) then
       PROG_STRING = trim(PROG_STRING) // ' integer 64 bits '  
    else
       print *, "integer not defined (KIND =", kind(ntypeabc), ")"
       return
    end if
    write(*,*) 'kind',KIND(ip)
  
  writecol = '********* compilation option ***************'  
  write(*, *) trim(PROG_STRING),' size wrk', nlar
  writecol = '*************** INPUT DATA ***************'
  colorw = 'red'
  write(*, *) '******************************************'
  call writecolor(writecol, colorw)
  write(*, *) '******************************************'
  write(*, *) 'Wavelength           : ', lambda, 'nm'
  write(*, *) 'Beam                 : ', trim(beam)
  write(*, *) 'Object               : ', trim(object)
  write(*, *) 'Isotropy             : ', trope
  write(*, *) 'Discretization       : ', nnnr
  write(*, *) 'Size of the box      : ', nxm, nym, nzm
  write(*, *) 'Iterative method     : ', methodeit
  write(*, *) 'Preconditionner      : ', nprecon, '(0 no precon.)'
  write(*, *) 'Initial guess        : ', ninitest, '(0 null, 1 Born)'
  write(*, *) 'Tolerance asked      : ', tolinit
  write(*, *) 'Maximum of iteration : ', nlim
  write(*, *) 'Rigorous or approx.  : ', nrig, '(0 rigourous)'
  write(*, *) 'Local field          : ', nlocal, '(1 compute)'
  write(*, *) 'Macroscopic field    : ', nmacro, '(1 compute)'
  write(*, *) 'Cross section        : ', nsection, 'Csca', nsectionsca, '(1 compute)'
  write(*, *) 'Poynting asym. fac.  : ', nquickdiffracte, '(1 compute)'
  write(*, *) 'Energy conservation  : ', nenergie, '(1 compute)'
  write(*, *) 'Microscopy           : ', nlentille, 'Type', ntypemic

  if (ntypemic == 0) then
     write(*, *) 'NA for hollographic mic. : ', numaper
  else if (ntypemic == 1) then
     write(*, *) 'NA BF mic.               : ', numaper
     write(*, *) 'NA condenser BF mic.     : ', numaperinc
     ! dark field avec anneau
  else if (ntypemic == 2 .or. ntypemic == 5) then
     write(*, *) 'NA for DF mic.           : ', numaper
     write(*, *) 'NA condenser DF mic.     : ', numaperinc, numaperinc2
     ! Schieren
  else if (ntypemic == 3) then
     write(*, *) 'NA for Schieren mic.     : ', numaper
     write(*, *) 'NA condenser Schieren    : ', numaperinc
     write(*, *) 'kx center condenser      : ', kcnax
     write(*, *) 'ky center condenser      : ', kcnay
  else if (ntypemic == 4) then
     write(*, *) 'NA for DF mic.           : ', numaper
     write(*, *) 'NA condenser DF mic.     : ', numaperinc
  else if (ntypemic == 6) then
     write(*, *) 'NA for confocal mic.     : ', numaper
     numaperinc = numaper
     ! condenser=objectif pour un confocal
  else if (ntypemic == 7) then
     write(*, *) 'NA for confocal mic.     : ', numaper
     numaperinc = numaper
     ! condenser=objectif pour un confocal
  end if

  write(*, *) 'Side study           : ', nside, '(1 kz>0, -1 kz<0)'
  write(*, *) 'Quick Lens           : ', nquicklens, '(1 compute with FFT)'
  write(*, *) 'Focal plane position : ', zlens, 'nm'
  write(*, *) 'Optical force        : ', nforce, nforced, '(1 compute)'
  write(*, *) 'Optical torque       : ', ntorque, ntorqued, '(1 compute)'
  write(*, *) 'Near field           : ', nproche
  write(*, *) 'Additional side      : ', nxmp, nymp, nzmp
  write(*, *) 'Polarizability       : ', polarizability
  write(*, *) 'Integration Green    : ', nquad, '(0 no integration)'
  write(*, *) 'Write  file          : ', nmat, '0 ascii file: 1 no file: 2 hdf5 file'

  if (nmat == 2) then
#ifdef USE_HDF5
     if (file_id /= 0) then
        write(*, *) 'Close HFD5 if not done', file_id
        call h5gclose_f(group_idopt, error)
        call h5gclose_f(group_iddip, error)
        call h5gclose_f(group_idff, error)
        call h5gclose_f(group_idmic, error)
        call h5gclose_f(group_idof, error)
        call h5gclose_f(group_idnf, error)
        call h5gclose_f(group_idver, error)
        call hdf5close(file_id)
     end if
     debug = 1

     call hdf5create(h5file, file_id)
     write(*, *) 'h5 file created      : ', trim(h5file)
     write(*, *) 'file_id              : ', file_id
     call h5gcreate_f(file_id, "Option", group_idopt, error)
     call h5gcreate_f(file_id, "Object", group_iddip, error)
     call h5gcreate_f(file_id, "Far Field", group_idff, error)
     call h5gcreate_f(file_id, "Microscopy", group_idmic, error)
     call h5gcreate_f(file_id, "Optical Force", group_idof, error)
     call h5gcreate_f(file_id, "Near Field", group_idnf, error)
     call h5gcreate_f(file_id, "Version of the code", group_idver, error)
#endif

#ifndef USE_HDF5
     write(*, *) 'To use HDF5 you should compile with the option'
     infostr = 'To use HDF5 you should compile with the option'
     nstop = 1
     return
#endif

  end if

  if (nxm == 0 .or. nym == 0 .or. nzm == 0) then
     infostr = 'One of the number of subunit is equal to zero'
     nstop = -1
     return
  end if

  ! arret de suite si nnnr trop petit par rapport a n*m
  if (object(1:7) /= 'cuboid2' .and. object(1:13) /= 'inhomocuboid2' &
       .and. object(1:13) /= 'randomsphere2' .and. object(1:9) /= 'arbitrary') then
     if (nnnr > min(nxm, nym, nzm)) then
        infostr = 'nxm nym or nzm smaller than discretization'
        nstop = 1
        return
     end if
  end if

  materiau = materiaumulti(1)
  ! Liste des arrets si le choix des options est mal faits.
  ! si rien n'est demandÃ©
  testgreen = 0
  if (beam(1:11) == 'greentensor') then
     nmacro = 1
     testgreen = 1
     nlecture = 0
  end if
  if (nobjet == 0 .and. nlocal == 0 .and. nmacro == 0 &
       .and. nsection == 0 .and. nsectionsca == 0 &
       .and. nforce == 0 .and. nforced == 0 &
       .and. ntorque == 0 .and. ntorqued == 0 &
       .and. nlentille == 0 .and. nenergie == 0) then
     infostr = 'No calculation requested!'
     nstop = 1
     return
  end if
  ! liste des arrets pour emissivitÃ©
  if (nenergie == 1) then
     if (beam(1:7) == 'antenna' .or. beam(1:9) == 'arbitrary') then
        infostr = 'Can not compute emissivity with this beam'
        nstop = 1
        return
     end if
     if (abs(theta) >= 90.0_wp) then
        nstop = 1
        infostr = 'Illumination on the wrong side with the options'
        return
     end if
     if (abs(theta) > 90.0_wp .and. abs(theta) >= 45.0_wp) then
        nstop = 1
        infostr = 'Angle too strong: Imprecise calculation'
        return
     end if
  end if
  ! liste des arrets pour lentille
  if (nlentille == 1) then
     if (beam(1:7) == 'antenna' .or. beam(1:9) == 'arbitrary') then
        infostr = 'Can not compute total field with this beam'
        nstop = 1
        return
     end if
     if (numaper <= zero .or. numaper > one) then
        nstop = 1
        infostr = 'Problem with numerical aperture!'
        return
     end if

     if (abs(gross) < one) then
        nstop = 1
        infostr = 'Problem with magnification!'
        return
     end if

     if (abs(theta) >= 90.0_wp) then
        nstop = 1
        infostr = 'Illumination on the wrong side with the options'
        return
     end if
     ! change le signe si diffracte vers l'arriÃ¨re
     zlens = zlens * nmtom * real(nside, wp)
     write(*,*) 'zlens',zlens,nmtom,real(nside, wp)
  end if

  ! liste des arrets pour section efficace

  if (nsectionsca == 1) then
     if (abs(theta) >= 45.0_wp) then
        nstop = 1
        infostr = 'Angle too strong: Imprecise calculation with FFT'
        return
     end if
  end if

  if (nsection == 1) then
     if (beam(1:11) /= 'pwavelinear' .and. beam(1:13) /= 'pwavecircular') then
        nstop = 1
        infostr = 'Cross section only with plane wave'
        return
     end if
  end if

  ! calculation size parameter initialization
  nmax = nxm * nym * nzm
  ntotalm = 8 * nxm * nym * nzm
  ntotalf = (nxm + 1) * (nym + 1) * (nzm + 1)

  if (3.d0* real(nxm,wp)*real(nym,wp)*real(nzm,wp) > 2147483647.0_wp) then
     infostr = 'Number of subunits too large for integer in 32 bits'
     nstop = 1
     return
  endif


  ldabi = 3 * nxm * nym * nzm
  xg = xgmulti(1)
  yg = ygmulti(1)
  zg = zgmulti(1)
  rayon = rayonmulti(1)
  eps = epsmulti(1)
  epsani = epsanimulti(:, :, 1)
  ! pas assez discrÃ©tisÃ©
  if (nmax < 8) then
     nstop = 1
     infostr = 'Check Nxm Nym and Nzm!'
     return
  end if
  if (object(1:7) /= 'cuboid2' .and. object(1:13) /= 'inhomocuboid2' &
       .and. object(1:13) /= 'randomsphere2' .and. object(1:9) /= 'arbitrary') then
     if (nnnr < 2 .and. object(1:9) /= 'arbitrary') then
        nstop = 1
        write(*, *) 'There is no discretization!'
        infostr = 'There is no discretization!'
        return
     end if
  else
     if (nxm <= 1) then
        nstop = 1
        write(*, *) 'Discretization follow x too small'
        infostr = 'Discretization follow x too small'
        return
     end if
     if (nym <= 1) then
        nstop = 1
        write(*, *) 'Discretization follow y too small'
        infostr = 'Discretization follow y too small'
        return
     end if
     if (nzm <= 1) then
        nstop = 1
        write(*, *) 'Discretization follow z too small'
        infostr = 'Discretization follow z too small'
        return
     end if
  end if
  ! arret pour tolerance dans la mÃ©thode itÃ©rative
  if (tolinit < 1.0e-12_wp) then
     infostr = 'Tolerance for the iterative method too small!'
     nstop = 1
     return
  end if
  if (tolinit > 0.1_wp) then
     infostr = 'Tolerance  for the iterative method too large!'
     nstop = 1
     return
  end if

  ! initialization of the data
  eps0 = zone
  infostr = 'STARTED!'

  ! transform the input data in meter
  lambda = lambda * nmtom
  w0 = w0 * nmtom

  if (w0 <= zero) then
     nstop = 1
     infostr = 'Waist=0!'
     return
  end if
  if (P0 <= zero) then
     nstop = 1
     infostr = 'Power=0!'
     return
  end if
  ! open the output file:

  if (nmat == 0) then
     ! Intensity of the incident field
     open(36, file='incidentfieldx.mat')
     open(37, file='incidentfieldy.mat')
     open(38, file='incidentfieldz.mat')
     open(39, file='incidentfield.mat')
     ! Intensity of the local field
     open(40, file='localfieldx.mat')
     open(41, file='localfieldy.mat')
     open(42, file='localfieldz.mat')
     open(43, file='localfield.mat')
     ! Intensity of the macroscopic field
     open(44, file='macroscopicfieldx.mat')
     open(45, file='macroscopicfieldy.mat')
     open(46, file='macroscopicfieldz.mat')
     open(47, file='macroscopicfield.mat')
     ! Intensity of the incident field
     open(136, file='incidentfieldxwf.mat')
     open(137, file='incidentfieldywf.mat')
     open(138, file='incidentfieldzwf.mat')
     open(139, file='incidentfieldwf.mat')
     ! Intensity of the local field
     open(140, file='localfieldxwf.mat')
     open(141, file='localfieldywf.mat')
     open(142, file='localfieldzwf.mat')
     open(143, file='localfieldwf.mat')
     ! Intensity of the macroscopic field
     open(144, file='macroscopicfieldxwf.mat')
     open(145, file='macroscopicfieldywf.mat')
     open(146, file='macroscopicfieldzwf.mat')
     open(147, file='macroscopicfieldwf.mat')
     ! Far field discretization
     open(67, file='xwf.mat')
     open(68, file='ywf.mat')
     open(69, file='zwf.mat')
     ! save the Poynting vecteur
     open(50, file='poynting.mat')
     open(51, file='theta.mat')
     open(52, file='phi.mat')
     ! pour FFT
     open(53, file='poyntingpos.mat')
     open(54, file='poyntingneg.mat')
     open(55, file='kx.mat')
     open(56, file='ky.mat')

     ! save the density of the optical force
     open(60, file='forcex.mat')
     open(61, file='forcey.mat')
     open(62, file='forcez.mat')
     ! save the density of optical torque
     open(63, file='torquex.mat')
     open(64, file='torquey.mat')
     open(65, file='torquez.mat')
     ! save the Green tensor
     open(70, file='greenxx.mat')
     open(71, file='greenyx.mat')
     open(72, file='greenzx.mat')
     open(73, file='greenxy.mat')
     open(74, file='greenyy.mat')
     open(75, file='greenzy.mat')
     open(76, file='greenxz.mat')
     open(77, file='greenyz.mat')
     open(78, file='greenzz.mat')
  end if

  ! compute the relative permittivity fron a database versus the
  ! wavelength of illumination
  ! write(*,*) 'Relative permittivity',eps,materiau(1:2),lambda
  ! passe arbitrary car epsilon pas dÃ©fini encore, car objet pas fait
  if (object(1:9) == 'arbitrary') goto 111

  do k = 1, numberobjet
     materiau = materiaumulti(k)
     if (materiau(1:2) /= 'xx') then
        call interpdielec(lambda, materiau, epr, epi, infostr, nstop)
        if (nstop == 1) return
        epsmulti(k) = (epr * zone + icomp * epi)
        write(*, *) 'Relative permittivity', eps, materiau(1:2)
     else
        if (trope(1:3) == 'iso') then
           write(*, *) 'Relative permittivity: ', eps
        else
           do i = 1, 3
              do j = 1, 3
                 write(*, *) 'Relative permittivity', epsanimulti(i, j, k), i, j, k
              end do
           end do
        end if
     end if
  end do

  ! test pour voir si la permittivitÃ© a un sens
  do k = 1, numberobjet
     if (trope(1:3) == 'iso') then
        if (real(epsmulti(k),wp) == one .and. aimag(epsmulti(k)) == zero) then
           infostr = 'Object is in vacuum'
           nstop = 1
        else if (abs(epsmulti(k)) == zero) then
           infostr = 'No relative permittivity given'
           nstop = 1
        end if
     else
        tmp = zero
        do ii = 1, 3
           do jj = 1, 3
              tmp = tmp + abs(epsani(ii, jj))
           end do
        end do
        if (tmp == zero) then
           infostr = 'No relative permittivity given'
           nstop = 1
        end if
        ! Pour la pola tensorielle uniquement CR et CM disponible
        if (polarizability /= 'RR' .and. polarizability /= 'CM') then
           infostr = 'Polarizability should be CM or RR with anisotropy'
           nstop = 1
        end if
     end if
  end do
  if (nstop == 1) return

  eps = epsmulti(1)
  materiau = materiaumulti(1)

  ! wavenumber
111 k0 = 2.0_wp * pi / lambda
  k03 = k0 * k0 * k0
  ! look  for compute near field with FFT
  ! dÃ©crÃ©ment de 1 de nproche pour faciliter le code C
  nproche = nproche - 1
  nabc=0
  if (nproche == 2) then
     nabc=1
     nproche=0
     !verification si tout est pret
     if (object(1:6) == 'sphere' .or. object(1:12) == 'inhomosphere' .or. object(1:13) == 'inhomocuboid1') then
        write(*,*) 'Object OK for MDDA'
     else
        nstop = 1
        infostr='object not done for MDDA'
        return
     endif
     
     if (beam(1:11) == 'pwavelinear' .or. beam(1:15) == 'gparawavelinear' .or. beam(1:17) == 'gparawavecircular') then
        write(*,*) 'Beam OK for MDDA'
     else
        nstop = 1
        infostr='Beam not done for MDDA'
        return
     endif
     !ne permet pas de preocnditionner
     if ( nprecon == 0) then
        write(*,*) 'No preconditionner OK for MDDA'
     else
        nstop = 1
        infostr='No preconditionner with MDDA'
        return
     endif
     if (nforce == 1 .or. ntorque == 1) then
        nstop = 1
        infostr='Not recommended to calculate optical forces with MDDA'
        return
     endif
  endif

  ! passe objet dans boite si utilise FFT
  if (nlecture == 1 .and. nproche == -1) nproche = 0
  if (nenergie == 1 .and. nproche == -1) nproche = 0
  if (nquickdiffracte == 1 .and. nproche == -1) nproche = 0
  if (nquicklens == 1 .and. nproche == -1) nproche = 0
  if ((nrig == 6 .or. nrig == 7 .or. nrig == 10 .or. nrig == 11 .or. nrig == 12 .or. nrig == 13 .or. &
       nrig == 14 .or. nrig == 15) &
       .and. nproche == -1) nproche = 0
  if (beam(1:8) == 'gfftwave' .and. nproche == -1) nproche = 0
  if (beam(1:7) == 'speckle' .and. nproche == -1) nproche = 0
  if (beam(1:11) == 'demispeckle' .and. nproche == -1) nproche = 0
  if (beam(1:8) == 'confocal' .and. nproche == -1) nproche = 0
  if (beam(1:12) == 'demiconfocal' .and. nproche == -1) nproche = 0
  if (ninitest == 2 .or. ninitest == 3) nproche = 0
  if (nquicklens + nquickdiffracte >= 1) then
     if (nfft2d < max(nxm, nym)) then
        infostr = 'Increase size of FFT: larger than discretization'
        nstop = 1
        return
     end if
  end if
  ! fait plus que objet dans le cas multisphere; pb reprÃ©sentation
  if (object(1:13) == 'randomsphere1' .or. object(1:13) == 'randomsphere2' .or. &
       object(1:18) == 'randomspheremulti1') then
     if (nproche == -1) nproche = 0
  end if

  if (nprecon /= 0) then
     if (methodeit(1:6) == 'QMRCLA') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        ! QMRCLA with Chan preconditioning is supported by the full-wave CUDA
        ! driver. Both optimized and standard Green products are admitted here;
        ! validation determines which combinations are numerically stable.
        if (nrig /= 0 .and. nrig /= 8 .and. nrig /= 9) then
           nstop = 1
           infostr = 'QMRCLA CUDA supports full-wave and scalar modes only'
           return
        end if
#else
        ! The Fortran QMRCLA branches do not implement Chan preconditioning.
        nstop = 1
        infostr = 'iterative method not implemented with preconditionner'
        return
#endif
     end if
  end if

  nprochefft = 0
  if (nproche >= 1) then
     nprochefft = nproche
     nproche = 0
  end if
  write(*, *) 'Maximum of subunit   : ', nmax
  write(*, *) 'Near field options   : ', nproche, 'Wide field :', nprochefft
  writecol = '***********END INPUT DATA ****************'
  colorw = 'red'
  call writecolor(writecol, colorw)
  write(*, *) ' '

  writecol = '************ BEGIN OBJECT ****************'
  colorw = 'blue'
  write(*, *) '******************************************'
  call writecolor(writecol, colorw)
  write(*, *) '******************************************'

  if (polarizability == 'PS' .and. object(1:6) /= 'sphere') then
     nstop = 1
     infostr = 'PS only with a sphere'
     return
  end if

  if (polarizability == 'LS') then
     if (beam(1:11) == 'pwavelinear') then
        call sldrr(ss, pp, theta, phi, sldr, nstop, infostr)
     else
        write(*, *) 'Beam unknown'
        infostr = 'LS polarizability only with plane wave'
        nstop = 1
        return
     end if
  end if

  if (nabc.eq.1) then
     nproche=1
     eps0=zone+icomp*eps0i
  endif
  do i=1,nxm*nym*nzm
     tabmulti(i)=zero
  enddo

  ! Built the object
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  ! One shared CUDA geometry selector for this Fortran driver and cdmlib.cpp.
  ! The selector uses the caller-owned CPU ABI arrays; it builds the GPU
  ! occupancy map, publishes the CPU tabdip, and runs the CUDA material stage.
  call ifdda_geometry_dispatch_cuda(object, trope, polarizability, namefileobj, infostr, &
       eps, epsani, epsmulti, epsanimulti, numberobjet, &
       xgmulti, ygmulti, zgmulti, rayonmulti, xswf, yswf, zswf, density, &
       xg, yg, zg, side, sidex, sidey, sidez, hauteur, rayon, &
       demiaxea, demiaxeb, demiaxec, lnrbc, lcos, lc, hc, &
       phiobj, thetaobj, psiobj, ntypeabc, pmlwrk, ng, &
       eps0, xs, ys, zs, k0, lambda, aretecube, tabdip, tabmulti, &
       nnnr, nmax, nxm, nym, nzm, nxmp, nymp, nzmp, &
       nbsphere, ndipole, nx, ny, nz, nproche, niso, &
       epsilon, polarisa, macroscopicfieldx, sldr, nmat, &
       file_id, group_iddip, nstop)
#else
  if (object(1:6) == 'sphere') then
     numberobjet = 1
     call objetsphere(trope, eps, epsani, eps0, lnrbc, lcos, ntypeabc, xs, ys, zs, k0, aretecube, tabdip, tabmulti, pmlwrk, &
          & nnnr, nmax, nbsphere, ndipole, nx, ny, nz, polarizability, nproche, epsilon, polarisa, niso, rayon, xg, yg, zg, sldr&
          &, nmat, file_id, group_iddip, infostr, nstop)
  else if (object(1:12) == 'inhomosphere') then
     numberobjet = 1
     if (trope /= 'iso') then
        nstop = 1
        infostr = 'Permittivity not scalar for inhomogenous sphere'
        write(*, *) 'Permittivity not scalar for inhomogenous sphere'
        return
     end if

     call objetsphereinhomo(eps, eps0, lnrbc, lcos, ntypeabc, xs, ys, zs, k0, aretecube, tabdip, tabmulti, pmlwrk, nnnr, nmax,&
          & nbsphere, ndipole, nx, ny, nz, polarizability , nproche, epsilon, polarisa, niso, rayon, lc, hc, ng, macroscopicfieldx,&
          & sldr, nmat, file_id, group_iddip, infostr, nstop)

     macroscopicfieldx = zzero

  else if (object(1:13) == 'inhomocuboid1') then
     numberobjet = 1
     if (trope /= 'iso') then
        nstop = 1
        infostr = 'Permittivity not scalar for inhomogenous cuboid'
        write(*, *) 'Permittivity not scalar for inhomogenous cuboid'
        return
     end if

     call objetparainhomo(eps, eps0, lnrbc, lcos, ntypeabc,  xs, ys, zs, k0, aretecube, tabdip, tabmulti, pmlwrk, nnnr, nmax,&
          & nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, polarizability, epsilon, polarisa, niso, sidex, sidey, sidez, xg, yg, zg,&
          & lc, hc, ng, macroscopicfieldx, nproche, sldr, nmat, file_id, group_iddip, infostr, nstop)

     macroscopicfieldx = zzero

  else if (object(1:13) == 'inhomocuboid2') then
     numberobjet = 1
     if (trope /= 'iso') then
        nstop = 1
        infostr = 'Permittivity not scalar for inhomogenous cuboid'
        write(*, *) 'Permittivity not scalar for inhomogenous cuboid'
        return
     end if

     call objetparainhomonxnynz(eps, eps0, xs, ys, zs, k0, aretecube, &
          tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, &
          nxmp, nymp, nzmp, polarizability, epsilon, polarisa, niso, &
          sidex, sidey, sidez, xg, yg, zg, lc, hc, ng, macroscopicfieldx, &
          nproche, sldr, nmat, file_id, group_iddip, infostr, nstop)

     macroscopicfieldx = zzero

  else if (object(1:13) == 'randomsphere1') then
     numberobjet = 1

     call objetsphererandom(trope, eps, epsani, eps0, xs, ys, zs, k0, &
          aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, &
          nym, nzm, polarizability, epsilon, polarisa, niso, sidex, &
          sidey, sidez, xg, yg, zg, xswf, yswf, zswf, rayon, density, ng, &
          nproche, sldr, nmat, file_id, group_iddip, infostr, nstop)

  else if (object(1:13) == 'randomsphere2') then
     numberobjet = 1

     call objetsphererandomnxnynz(trope, eps, epsani, eps0, xs, ys, zs, k0, &
          aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, &
          nym, nzm, nxmp, nymp, nzmp, polarizability, epsilon, polarisa, niso, &
          sidex, sidey, sidez, xg, yg, zg, xswf, yswf, zswf, rayon, &
          density, ng, nproche, sldr, nmat, file_id, group_iddip, infostr, nstop)
  else if (object(1:18) == 'randomspheremulti1') then
     call objetsphererandommulti(trope, epsmulti, epsanimulti, &
          numberobjet, numberobjet, xgmulti, ygmulti, zgmulti, &
          rayonmulti, eps0, xs, ys, zs, k0, aretecube, tabdip, nnnr, nmax, &
          nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, polarizability, &
          epsilon, polarisa, niso, sidex, sidey, sidez, xswf, yswf, &
          zswf, density, ng, nproche, sldr, nmat, file_id, group_iddip, &
          infostr, nstop)

  else if (object(1:4) == 'cube') then
  
     write(*, *) 'cube:side', side
     numberobjet = 1
     call objetcube(trope, eps, epsani, eps0, xs, ys, zs, k0, aretecube, &
          tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, polarizability, &
          epsilon, polarisa, niso, side, xg, yg, zg, sldr, nmat, &
          file_id, group_iddip, infostr, nstop)
  else if (object(1:7) == 'cuboid1') then
     write(*, *) 'cuboid:sidex,sidey,sizez ', sidex, sidey, sidez
     numberobjet = 1
     call objetpara(trope, eps, epsani, eps0, xs, ys, zs, k0, aretecube, &
          tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, &
          polarizability, epsilon, polarisa, niso, sidex, sidey, &
          sidez, xg, yg, zg, phiobj, thetaobj, psiobj, nproche, sldr, &
          nmat, file_id, group_iddip, infostr, nstop)

  else if (object(1:7) == 'cuboid2') then
     numberobjet = 1
     call objetparanxnynz(trope, eps, epsani, eps0, xs, ys, zs, k0, &
          aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, &
          nym, nzm, nxmp, nymp, nzmp, polarizability, epsilon, polarisa, niso, &
          sidex, sidey, sidez, xg, yg, zg, nproche, sldr, nmat, &
          file_id, group_iddip, infostr, nstop)
  else if (object(1:9) == 'ellipsoid') then
     numberobjet = 1
     call objetellipse(trope, eps, epsani, eps0, xs, ys, zs, k0, aretecube, &
          tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, &
          polarizability, nproche, epsilon, polarisa, niso, &
          demiaxea, demiaxeb, demiaxec, xg, yg, zg, phiobj, thetaobj, &
          psiobj, sldr, nmat, file_id, group_iddip, infostr, nstop)
  else if (object(1:8) == 'nspheres') then
     call objetnspheres(trope, epsmulti, epsanimulti, numberobjet, &
          numberobjet, xgmulti, ygmulti, zgmulti, rayonmulti, eps0, xs, &
          ys, zs, k0, aretecube, tabdip, tabmulti, nnnr, nmax, nbsphere, &
          ndipole, nx, ny, nz, polarizability, nproche, epsilon, polarisa, niso, &
          sldr, nmat, file_id, group_iddip, infostr, nstop)

  else if (object(1:8) == 'cylinder') then
     numberobjet = 1
     call objetcylindre(trope, eps, epsani, eps0, xs, ys, zs, k0, aretecube, &
          tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, nxm, nym, nzm, &
          polarizability, nproche, epsilon, polarisa, niso, rayon, &
          hauteur, xg, yg, zg, phiobj, thetaobj, psiobj, sldr, nmat, &
          file_id, group_iddip, infostr, nstop)
  else if (object(1:16) == 'concentricsphere') then
     call objetsphereconcentric(trope, epsmulti, epsanimulti, &
          numberobjet, numberobjet, xg, yg, zg, rayonmulti, eps0, xs, ys, &
          zs, k0, aretecube, tabdip, tabmulti, nnnr, nmax, nbsphere, &
          ndipole, nx, ny, nz, polarizability, nproche, epsilon, polarisa, niso, &
          sldr, nmat, file_id, group_iddip, infostr, nstop)
  else if (object(1:9) == 'arbitrary') then
     numberobjet = 1
     call objetarbitrary(trope, eps, epsani, eps0, xs, ys, zs, k0, &
          aretecube, tabdip, nnnr, nmax, nbsphere, ndipole, nx, ny, nz, &
          polarizability, namefileobj, nproche, epsilon, polarisa, niso, &
          sldr, nmat, file_id, group_iddip, infostr, nstop)

     if (aretecube / lambda <= 0.01_wp) then
        infostr = 'Meshsize too small'
        nstop = 1
        return
     end if

  else
     write(*, *) 'Object unknown'
     infostr = 'Object unknown!'
     nstop = 1
     return
  end if
#endif
  write(*, *) 'Object under study created       : ', trim(object)
  if (nstop == 1) return
  ! if (nnnr*nnnr*nnnr.gt.nmax) then
  ! nstop=1
  ! write(*,*) 'nxm nym and nzm too small'
  ! infostr
  ! \$    ='nxm nym and nzm too small compare to the discretization'
  ! return
  ! endif

  if (nx > nxm .or. ny > nym .or. nz > nzm) then
     nstop = 1
     infostr = 'Dimension Problem of the Box : Box too small!'
     write(*, *) 'dimension Problem', nx, nxm, ny, nym, nz, nzm
     return
  end if
  if (nx <= 1 .or. ny <= 1 .or. nz <= 1) then
     nstop = 1
     infostr = 'Dimension Problem nx ny nz too small!'
     write(*, *) 'dimension Problem', nx, nxm, ny, nym, nz, nzm
     return
  end if
  if (nstop == 1) return
  if (nstop == -1) then
     infostr = 'Calculation cancelled after object created!'
     return
  end if

  ! ecriture dans fichiers du epsilon
  if (nmat == 0) then
     open(66, file='epsilon.mat')
     if (trope == 'iso') then
        do i = 1, ndipole
           k = tabdip(i)
           if (k /= 0) then
              write(66, *) real(epsilon(k, 1, 1),wp), aimag(epsilon(k, 1, 1))
           else
              write(66, *) one, zero
           end if
        end do
     else
        do i = 1, ndipole
           k = tabdip(i)
           if (k /= 0) then
              do ii = 1, 3
                 do jj = 1, 3
                    write(66, *) real(epsilon(k, ii, jj),wp), aimag(epsilon(k, ii, jj))
                 end do
              end do
           else
              do ii = 1, 3
                 do jj = 1, 3
                    if (ii == jj) then
                       write(66, *) one, zero
                    else
                       write(66, *) zero, zero
                    end if
                 end do
              end do
           end if
        end do
     end if
     ! fin ecriture du epsilon
     close(66)
  else if (nmat == 2) then
     dim(1) = 1
     dim(2) = 1
     datasetname = 'nx'
     call hdf5write_int(group_iddip, datasetname, nx, dim)
     datasetname = 'ny'
     call hdf5write_int(group_iddip, datasetname, ny, dim)
     datasetname = 'nz'
     call hdf5write_int(group_iddip, datasetname, nz, dim)

     if (trope == 'iso') then
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, ndipole
           k = tabdip(i)
           if (k /= 0) then
              wrk(i, 1) = epsilon(k, 1, 1)
           else
              wrk(i, 1) = zone
           end if
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
        dim(1) = ndipole
        dim(2) = nmax * 3
        datasetname = 'Epsilon real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 1)), dim)
        datasetname = 'Epsilon imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 1)), dim)
     else
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, ndipole
           k = tabdip(i)
           if (k /= 0) then
              wrk(i, 1) = epsilon(k, 1, 1)
              wrk(i, 2) = epsilon(k, 2, 2)
              wrk(i, 3) = epsilon(k, 3, 3)
              wrk(i, 4) = epsilon(k, 1, 2)
              wrk(i, 5) = epsilon(k, 1, 3)
              wrk(i, 6) = epsilon(k, 2, 1)
              wrk(i, 7) = epsilon(k, 2, 3)
              wrk(i, 8) = epsilon(k, 3, 1)
              wrk(i, 9) = epsilon(k, 3, 2)
           else
              wrk(i, 1) = zone
              wrk(i, 2) = zone
              wrk(i, 3) = zone
              wrk(i, 4) = zzero
              wrk(i, 5) = zzero
              wrk(i, 6) = zzero
              wrk(i, 7) = zzero
              wrk(i, 8) = zzero
              wrk(i, 9) = zzero
           end if
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
        dim(1) = ndipole
        dim(2) = nmax * 3
        datasetname = 'epsilon xx real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 1)), dim)
        datasetname = 'epsilon xx imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 1)), dim)
        datasetname = 'epsilon yy real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 2)), dim)
        datasetname = 'epsilon yy imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 2)), dim)
        datasetname = 'epsilon zz real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 3)), dim)
        datasetname = 'epsilon zz imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 3)), dim)
        datasetname = 'epsilon xy real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 4)), dim)
        datasetname = 'epsilon xy imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 4)), dim)
        datasetname = 'epsilon xz real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 5)), dim)
        datasetname = 'epsilon xz imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 5)), dim)
        datasetname = 'epsilon yx real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 6)), dim)
        datasetname = 'epsilon yx imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 6)), dim)
        datasetname = 'epsilon yz real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 7)), dim)
        datasetname = 'epsilon yz imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 7)), dim)
        datasetname = 'epsilon zx real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 8)), dim)
        datasetname = 'epsilon zx imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 8)), dim)
        datasetname = 'epsilon zy real part'
        call hdf5write1d(group_iddip, datasetname, real(wrk(:, 9)), dim)
        datasetname = 'epsilon zy imaginary part'
        call hdf5write1d(group_iddip, datasetname, aimag(wrk(:, 9)), dim)
     end if
  end if

  lambda10n = lambda / 10.0_wp / abs(sqrt(eps))
  if (aretecube > 2.5_wp * lambda10n) then
     nstop = 1
     infostr = 'Meshsize larger than lambda/4!'
     return
  end if

  write(*, *) 'Number of subunits for the object :', nbsphere
  write(*, *) 'Number of subunits for the mesh   :', ndipole
  write(*, *) 'Size of the mesh size             :', aretecube
  write(*, *) 'Number of subunits along x y and z:', nx, ny, nz
  write(*, *) 'Lambda/(10n)                      :', lambda / 10.0_wp / abs(sqrt(eps))
  writecol = '************** END OBJECT ****************'
  colorw = 'blue'
  call writecolor(writecol, colorw)
  write(*, *) ' '

  if (beam(1:11) /= 'pwavelinear' .and. beam(1:13) /= 'pwavecircular' &
       .and. beam(1:7) /= 'antenna' .and. beam(1:11) /= 'gwavelinear' &
       .and. beam(1:13) /= 'gwavecircular' .and. beam(1:14) /= 'gfftwavelinear' &
       .and. beam(1:16) /= 'gfftwavecircular' .and. beam(1:15) /= 'gparawavelinear' &
       .and. beam(1:17) /= 'gparawavecircular' .and. beam(1:15) /= 'wavelinearmulti' &
       .and. beam(1:11) /= 'greentensor' .and. beam(1:7) /= 'speckle' &
       .and. beam(1:11) /= 'demispeckle' .and. beam(1:8) /= 'confocal' &
       .and. beam(1:12) /= 'demiconfocal' .and. beam(1:9) /= 'arbitrary') then
     write(*, *) 'Beam unknown'
     infostr = 'Beam unknown!'
     nstop = 1
     return
  end if
  ! met numaperil a zero si on utilise pas le confocal ou le speckle
  if (beam(1:7) /= 'speckle' .and. beam(1:11) /= 'demispeckle' &
       .and. beam(1:8) /= 'confocal' .and. beam(1:12) /= 'demiconfocal') then
     numaperil = zero
  end if

  ! crÃ© le fichier de data pour connaitre les options pour matlab

  open(900, file='inputmatlab.mat')
  write(900, *) nproche
  write(900, *) nlocal
  write(900, *) nmacro
  write(900, *) nsection
  write(900, *) nsectionsca
  write(900, *) nquickdiffracte
  write(900, *) nforce
  write(900, *) nforced
  write(900, *) ntorque
  write(900, *) ntorqued
  write(900, *) nlentille
  write(900, *) nquicklens
  write(900, *) nphi
  write(900, *) ntheta + 1
  if (trope == 'iso') write(900, *) 0
  if (trope == 'ani') write(900, *) 1
  write(900, *) nfft2d
  write(900, *) k0
  write(900, *) numaper
  write(900, *) nprochefft
  write(900, *) nobjet
  write(900, *) ntypemic
  write(900, *) nside
  write(900, *) nmat
  write(900, *) numaperinc
  write(900, *) numaperinc2
  write(900, *) kcnax
  write(900, *) kcnay
  write(900, *) testgreen
  write(900, *) numaperil

  close(900)
  if (nmat == 2) then
     open(901, file='filenameh5')
     write(901, *) h5file
     close(901)
     dim(1) = 1
     dim(2) = 1

     datasetname = 'nproche'
     call hdf5write_int(group_idopt, datasetname, nproche, dim)
     datasetname = 'nlocal'
     call hdf5write_int(group_idopt, datasetname, nlocal, dim)
     datasetname = 'nmacro'
     call hdf5write_int(group_idopt, datasetname, nmacro, dim)
     datasetname = 'nsection'
     call hdf5write_int(group_idopt, datasetname, nsection, dim)
     datasetname = 'nsectionsca'
     call hdf5write_int(group_idopt, datasetname, nsectionsca, dim)
     datasetname = 'nquickdiffracte'
     call hdf5write_int(group_idopt, datasetname, nquickdiffracte, dim)
     datasetname = 'nforce'
     call hdf5write_int(group_idopt, datasetname, nforce, dim)
     datasetname = 'nforced'
     call hdf5write_int(group_idopt, datasetname, nforced, dim)
     datasetname = 'ntorque'
     call hdf5write_int(group_idopt, datasetname, ntorque, dim)
     datasetname = 'ntorqued'
     call hdf5write_int(group_idopt, datasetname, ntorqued, dim)
     datasetname = 'nlentille'
     call hdf5write_int(group_idopt, datasetname, nlentille, dim)
     datasetname = 'nquicklens'
     call hdf5write_int(group_idopt, datasetname, nquicklens, dim)
     datasetname = 'nphi'
     call hdf5write_int(group_idopt, datasetname, nphi, dim)
     datasetname = 'ntheta'
     call hdf5write_int(group_idopt, datasetname, ntheta + 1, dim)

     if (trope == 'iso') then
        datasetname = 'iso'
        i = 0
        call hdf5write_int(group_idopt, datasetname, i, dim)
     end if

     if (trope == 'ani') then
        datasetname = 'iso'
        i = 1
        call hdf5write_int(group_idopt, datasetname, i, dim)
     end if
     datasetname = 'nfft2d'
     call hdf5write_int(group_idopt, datasetname, nfft2d, dim)
     datasetname = 'k0'
     call hdf5write(group_idopt, datasetname, k0, dim)
     datasetname = 'numaper'
     call hdf5write(group_idopt, datasetname, numaper, dim)
     datasetname = 'nprochefft'
     call hdf5write_int(group_idopt, datasetname, nprochefft, dim)
     datasetname = 'nobjet'
     call hdf5write_int(group_idopt, datasetname, nobjet, dim)
     datasetname = 'ntypemic'
     call hdf5write_int(group_idopt, datasetname, ntypemic, dim)
     datasetname = 'nside'
     call hdf5write_int(group_idopt, datasetname, nside, dim)
     datasetname = 'nmat'
     call hdf5write_int(group_idopt, datasetname, nmat, dim)
     datasetname = 'numaperinc'
     call hdf5write(group_idopt, datasetname, numaperinc, dim)
     datasetname = 'numaperinc2'
     call hdf5write(group_idopt, datasetname, numaperinc2, dim)
     datasetname = 'kcx'
     call hdf5write(group_idopt, datasetname, kcnax, dim)
     datasetname = 'kcy'
     call hdf5write(group_idopt, datasetname, kcnay, dim)
     datasetname = 'testgreen'
     call hdf5write_int(group_idopt, datasetname, testgreen, dim)
     datasetname = 'numaperil'
     call hdf5write(group_idopt, datasetname, numaperil, dim)

     ! ajoute qqs donnÃ©es pour le hdf5
     datasetname = 'rigorous'
     call hdf5write_int(group_idopt, datasetname, nrig, dim)
     datasetname = 'preconditionner'
     call hdf5write_int(group_idopt, datasetname, nprecon, dim)
     datasetname = 'initial guess'
     call hdf5write_int(group_idopt, datasetname, ninitest, dim)
     datasetname = 'maximum of iteration'
     call hdf5write_int(group_idopt, datasetname, nlim, dim)
     datasetname = 'tolerance iterative method'
     call hdf5write(group_idopt, datasetname, tolinit, dim)

  end if

  ! ne fait que l'objet
  if (nobjet == 1.0_wp) then
     infostr = 'Dipole calculation completed!'
     if (nmat == 2) then
#ifdef USE_HDF5
        dim(1) = 1
        dim(2) = 1
        nx = 0
        datasetname = trim(PROG_STRING)
        call hdf5write_int(group_idver, datasetname, nx, dim)
        call h5gclose_f(group_idopt, error)
        call h5gclose_f(group_iddip, error)
        call h5gclose_f(group_idff, error)
        call h5gclose_f(group_idmic, error)
        call h5gclose_f(group_idof, error)
        call h5gclose_f(group_idnf, error)
        call h5gclose_f(group_idver, error)
        call hdf5close(file_id)
        file_id = 0
#else
        write(*, *) "No HDF5!"
#endif
     end if
     return
  end if

  ! multiplication by a factor 2: Toeplitz matrix transformed in a
  ! circulant matrix with a doble size.
  if (nabc == 1 .and. nlentille == 0 .and. nforce == 0 .and. ntorque == 0) then
     indiceimag=aimag(sqrt(eps0))
     if (indiceimag <= tiny(indiceimag)) then
        ! With a lossless background the attenuation length is infinite.
        ! Keep the standard doubled Toeplitz grid instead of converting an
        ! infinity to an integer (which overflows to a negative extent).
        nabs=max(nx,ny,nz)
     else
        nabs=(-log(0.001_wp)/k0/indiceimag)/aretecube
     endif

     write(*,*) 'optimisation',indiceimag,aretecube,nabs
     if (nabs.ge.nx) then
        nx2=2*nx
     else
        nx2=nx+nabs
        call factorto235(nx2)
        if (nx2.gt.2*nx) nx2=2*nx
     endif
     if (nabs.ge.ny) then
        ny2=2*ny
     else
        ny2=ny+nabs
        call factorto235(ny2)
        if (ny2.gt.2*ny) ny2=2*ny
     endif
     if (nabs.ge.nz) then
        nz2=2*nz
     else
        nz2=nz+nabs
        call factorto235(nz2)
        if (nz2.gt.2*nz) nz2=2*nz
     endif
  else
     nx2=2*nx
     ny2=2*ny
     nz2=2*nz
  endif

  write(*,*) 'nx',nx,'nx2',nx2
  write(*,*) 'ny',ny,'ny2',ny2
  write(*,*) 'nz',nz,'nz2',nz2

  if ( real(nx2,wp)*real(ny2,wp)*real(nz2,wp) > 2147483647.0_wp) then
     infostr = 'Number of subunits too large for integer in 32 bits'
     nstop = 1
     return
  endif


  ntotal=nx2 * ny2 *nz2
  write(*,*) 'ntotal',ntotal
  allocate(vectx(ntotal))
  allocate(vecty(ntotal))
  allocate(vectz(ntotal))



  nbsphere3=3*nbsphere
  nxy2=nx2*ny2
  

  ! calcul les plans pour la FFT3D
  FFTW_FORWARD = -1
  FFTW_BACKWARD = +1

  FFTW_ESTIMATE = 64
  FFTW_MEASURE = 0
  FFTW_PATIENT = 32
  FFTW_EXHAUSTIVE = 8
  ! comment calculer les plans pour FFTW

#ifdef USE_FFTW

#if defined(USE_CUDA) || defined(IFDDA_CUDA_INVERSE_DRIVER)
  flagfftw = FFTW_ESTIMATE
#else
  if (nx * ny * nz <= 1000000) then
     flagfftw = FFTW_ESTIMATE
  else
     flagfftw = FFTW_MEASURE
  end if
#endif
  write(*, *) 'CREATE THE PLAN FOR FFT WITH FLAG=', flagfftw
#ifdef USE_SIMP
  call sfftw_plan_dft_3d(planb, nx2, ny2, nz2, vectx, vectx, FFTW_BACKWARD, flagFFTW)
  call sfftw_plan_dft_3d(planf, nx2, ny2, nz2, vectx, vectx, FFTW_FORWARD, flagFFTW)
#else
  call dfftw_plan_dft_3d(planb, nx2, ny2, nz2, vectx, vectx, FFTW_BACKWARD, flagFFTW)
  call dfftw_plan_dft_3d(planf, nx2, ny2, nz2, vectx, vectx, FFTW_FORWARD, flagFFTW)
#endif


  write(*, *) 'PLAN IN 3D'

  ! FFT X
  n(1) = nx2
  inembed(1) = nx2
  onembed(1) = nx2
  ii=1
#ifdef USE_SIMP
  call sfftw_plan_many_dft(planXb, ii, n, ny2 * nz, vectx(1:ny * nx2), inembed, ii, nx2, vectx(1:ny * nx2), onembed, ii, nx2,&
       & FFTW_BACKWARD, flagFFTW)
  call sfftw_plan_many_dft(planXf, ii, n, ny2 * nz, vectx(1:ny * nx2), inembed, ii, nx2, vectx(1:ny * nx2), onembed, ii, nx2,&
       & FFTW_FORWARD, flagFFTW)
#else
  call dfftw_plan_many_dft(planXb, ii, n, ny2 * nz, vectx(1:ny * nx2), inembed, ii, nx2, vectx(1:ny * nx2), onembed, ii, nx2,&
       & FFTW_BACKWARD, flagFFTW)
  call dfftw_plan_many_dft(planXf, ii, n, ny2 * nz, vectx(1:ny * nx2), inembed, ii, nx2, vectx(1:ny * nx2), onembed, ii, nx2,&
       & FFTW_FORWARD, flagFFTW)
#endif


  ! FFT 1-D le long de Y
  n(1) = ny2
  inembed(1) = ny2
  onembed(1) = ny2
#ifdef USE_SIMP
  call sfftw_plan_many_dft(planYb, ii, n, nx, vectx(1:nx2 * ny2), inembed, nx2, ii, vectx(1:nx2 * ny2), onembed, nx2, ii,&
       & FFTW_BACKWARD, flagFFTW)
  call sfftw_plan_many_dft(planYf, ii, n, nx, vectx(1:nx2 * ny2), inembed, nx2, ii, vectx(1:nx2 * ny2), onembed, nx2, ii,&
       & FFTW_FORWARD, flagFFTW)
#else
  call dfftw_plan_many_dft(planYb, ii, n, nx, vectx(1:nx2 * ny2), inembed, nx2, ii, vectx(1:nx2 * ny2), onembed, nx2, ii,&
       & FFTW_BACKWARD, flagFFTW)
  call dfftw_plan_many_dft(planYf, ii, n, nx, vectx(1:nx2 * ny2), inembed, nx2, ii, vectx(1:nx2 * ny2), onembed, nx2, ii,&
       & FFTW_FORWARD, flagFFTW)
#endif  

  ! FFT Z
  n(1) = nz2
  inembed(1) = nz2
  onembed(1) = nz2
#ifdef USE_SIMP
  call sfftw_plan_many_dft(planZb, ii, n, nx2 * ny2, vectx, inembed, nx2 * ny2, ii, vectx, onembed, nx2 * ny2, ii, FFTW_BACKWARD,&
       & flagFFTW)
  call sfftw_plan_many_dft(planZf, ii, n, nx2 * ny2, vectx, inembed, nx2 * ny2, ii, vectx, onembed, nx2 * ny2, ii, FFTW_FORWARD,&
       & flagFFTW) 
#else
  call dfftw_plan_many_dft(planZb, ii, n, nx2 * ny2, vectx, inembed, nx2 * ny2, ii, vectx, onembed, nx2 * ny2, ii, FFTW_BACKWARD,&
       & flagFFTW)
  call dfftw_plan_many_dft(planZf, ii, n, nx2 * ny2, vectx, inembed, nx2 * ny2, ii, vectx, onembed, nx2 * ny2, ii, FFTW_FORWARD,&
       & flagFFTW)
#endif 
  write(*, *) 'PLAN IN 1D'
  ! calcul les plans pour la FFT2D
#ifdef USE_SIMP
  call sfftw_plan_dft_2d(plan2b, nfft2d, nfft2d, Eimagex, Eimagex, FFTW_BACKWARD, flagFFTW)
  call sfftw_plan_dft_2d(plan2f, nfft2d, nfft2d, Eimagex, Eimagex, FFTW_FORWARD, flagFFTW)
#else
  call dfftw_plan_dft_2d(plan2b, nfft2d, nfft2d, Eimagex, Eimagex, FFTW_BACKWARD, flagFFTW)
  call dfftw_plan_dft_2d(plan2f, nfft2d, nfft2d, Eimagex, Eimagex, FFTW_FORWARD, flagFFTW)
#endif     
  write(*, *) 'PLAN IN 2D'
#endif

  writecol = '******** COMPUTE INCIDENT  FIELD *********'
  colorw = 'pink'
  write(*, *) '******************************************'
  call writecolor(writecol, colorw)
  write(*, *) '******************************************'
  write(*, *) 'Beam used  : ', trim(beam)

  if (beam(1:11) == 'pwavelinear') then
     write(*, *) 'theta      : ', theta
     write(*, *) 'phi        : ', phi
     write(*, *) 'pola.      : ', pp

     if (nenergie == 1) w0 = real(nfft2d, wp) * aretecube / sqrt(pi)
     if (ntypemic == 0 .and. nlentille == 1) call calculangle(theta, phi, k0, aretecube, nfft2d)
     ! compute E0
     call irradiance(P0, w0, E0, irra)
     I0 = abs(E0)**2
     norm=zero
     tmp=zero
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_plane_wave_field_cuda(xs,ys,zs,int(nbsphere,c_int),k0,eps0,E0, &
          ss,pp,theta,phi,0_c_int,FF0,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA linear plane-wave illumination failed'
        return
     endif
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC) REDUCTION(+:norm,tmp)
     do i=1,nbsphere
        norm=norm+abs(FF0(3*i-2))**2+abs(FF0(3*i-1))**2+abs(FF0(3*i))**2
        if (tabmulti(i) == -1) then
           FF0(3*i-2)=zzero
           FF0(3*i-1)=zzero
           FF0(3*i)=zzero
        endif
        tmp=tmp+abs(FF0(3*i-2))**2+abs(FF0(3*i-1))**2+abs(FF0(3*i))**2
     enddo
     !$OMP ENDDO
     !$OMP END PARALLEL
#else
     
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC) REDUCTION(+:norm)
     do i = 1, nbsphere
        call ondeplane(xs(i), ys(i), zs(i), k0, eps0, E0, ss, pp, theta, phi, FF0(3 * i - 2), FF0(3 * i - 1), FF0(3 * i), nstop&
             &, infostr)
        norm=norm+abs(FF0(3 * i - 2))**2+abs(FF0(3 * i - 1))**2+abs(FF0(3 * i ))**2
        if (tabmulti(i) == -1) then
           FF0(3*i-2)=zzero
           FF0(3*i-1)=zzero
           FF0(3*i)=zzero
        endif
        tmp=tmp+abs(FF0(3 * i - 2))**2+abs(FF0(3 * i - 1))**2+abs(FF0(3 * i ))**2        
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif

  else if (beam(1:13) == 'pwavecircular') then
     write(*, *) 'theta=', theta
     write(*, *) 'phi=', phi
     write(*, *) 'pola=', ss
     ! compute E0
     if (nenergie == 1) w0 = real(nfft2d, wp) * aretecube / sqrt(pi)
     call irradiance(P0, w0, E0, irra)
     I0 = abs(E0)**2
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_plane_wave_field_cuda(xs,ys,zs,int(nbsphere,c_int),k0,eps0,E0, &
          ss,pp,theta,phi,1_c_int,FF0,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA circular plane-wave illumination failed'
        return
     endif
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere
        call ondecirce(xs(i), ys(i), zs(i), k0, E0, ss, theta, phi, FF0(3 * i - 2), FF0(3 * i - 1), FF0(3 * i))
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif
  else if (beam(1:15) == 'wavelinearmulti') then
     if (nenergie == 1) w0 = real(nfft2d, wp) * aretecube / sqrt(pi)
     call irradiance(P0, w0, E0, irra)
     I0 = abs(E0)**2
     tmp = zero
     do i = 1, nbinc
        tmp = tmp + abs(E0m(i))**2.0_wp
        if (abs(E0m(i)) == zero) then
           infostr = 'Magnitude equal to zero in incident beam'
           nstop = 1
           return
        end if
     end do
     tmp = sqrt(abs(E0)**2.0_wp / tmp)
     do i = 1, nbinc
        E0m(i) = E0m(i) * tmp
     end do

     do i = 1, nbinc
        write(*, *) 'theta=', thetam(i)
        write(*, *) 'phi=', phim(i)
        write(*, *) 'pola=', ppm(i)
        write(*, *) 'E0=', E0m(i)
     end do
     ! compute E0
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_multi_plane_wave_field_cuda(xs,ys,zs,int(nbsphere,c_int),k0,E0, &
          E0m,ssm,ppm,thetam,phim,int(nbinc,c_int),FF0,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA multiple plane-wave illumination failed'
        return
     endif
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere
        call ondeplanemulti(xs(i), ys(i), zs(i), k0, E0m, ssm, ppm, thetam, phim, nbinc, FF0(3 * i - 2), FF0(3 * i - 1), FF0(3 &
             &* i), nstop, infostr)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif
  else if (beam(1:7) == 'antenna') then
     E0 = sqrt(3.0_wp * quatpieps0 / (k0**4.0_wp) * P0) / quatpieps0 * zone
     xdip = xgaus * nmtom
     ydip = ygaus * nmtom
     zdip = zgaus * nmtom
     call dipoleinside(xdip, ydip, zdip, xs, ys, zs, aretecube, nmax, nbsphere)
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_antenna_field_cuda(xs,ys,zs,int(nbsphere,c_int),xdip,ydip, &
          zdip,aretecube,k0,E0,theta,phi,xgmulti(1),ygmulti(1),zgmulti(1), &
          FF0,Em,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA antenna illumination failed'
        return
     endif
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere
        call dipoleinc(xdip, ydip, zdip, theta, phi, xs(i), ys(i), zs(i), aretecube, k0, E0, FF0(3 * i - 2), FF0(3 * i - 1),&
             & FF0(3 * i), nstop, infostr)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
     ! compute the intenisty at the first object location
     call dipoleinc(xdip, ydip, zdip, theta, phi, xgmulti(1), ygmulti(1), zgmulti(1), aretecube, k0, E0, Em(1), Em(2), Em(3),&
          & nstop, infostr)
#endif
     I0 = abs(Em(1))**2.0_wp + abs(Em(2))**2.0_wp + abs(Em(3))**2.0_wp
  else if (beam(1:11) == 'gwavelinear') then
     write(*, *) 'theta=', theta
     write(*, *) 'phi=', phi
     write(*, *) 'pola=', pp

     xgaus = xgaus * nmtom
     ygaus = ygaus * nmtom
     zgaus = zgaus * nmtom

     nloin = 0
     tol = tolinit
     call gaussianpuissance(P0, irra, w0, k0, E0)
     I0 = abs(E0)**2
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_rigorous_gaussian_field_cuda(xs,ys,zs,int(nbsphere,c_int), &
          xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,pp,E0,tol,0_c_int, &
          int(nloin,c_int),FF0,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA rigorous Gaussian illumination failed'
        return
     endif
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere
        call gaussianchamp(xs(i), ys(i), zs(i), xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, E0, FF0(3 * i - 2), FF0(3 *&
             & i - 1), FF0(3 * i), tol, nloin, nstop, infostr)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif
  else if (beam(1:13) == 'gwavecircular') then
     write(*, *) 'theta=', theta
     write(*, *) 'phi=', phi
     write(*, *) 'pola=', ss

     xgaus = xgaus * nmtom
     ygaus = ygaus * nmtom
     zgaus = zgaus * nmtom

     nloin = 0
     tol = tolinit
     call gaussianpuissance(P0, irra, w0, k0, E0)
     I0 = abs(E0)**2
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_rigorous_gaussian_field_cuda(xs,ys,zs,int(nbsphere,c_int), &
          xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,pp,E0,tol,1_c_int, &
          int(nloin,c_int),FF0,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA circular rigorous Gaussian illumination failed'
        return
     endif
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere
        call gaussianchampcirc(xs(i), ys(i), zs(i), xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, E0, FF0(3 * i - 2), FF0(3 *&
             & i - 1), FF0(3 * i), tol, nloin, nstop, infostr)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif
  else if (beam(1:8) == 'gfftwave') then
     write(*, *) 'incident wave with FFT'
     write(*, *) 'theta=', theta
     write(*, *) 'phi=', phi
     write(*, *) 'pola=', pp

     xgaus = xgaus * nmtom
     ygaus = ygaus * nmtom
     zgaus = zgaus * nmtom

     call gaussianpuissance(P0, irra, w0, k0, E0)
     I0 = abs(E0)**2
     call gaussianchampfft(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, theta, phi, xgaus, ygaus, zgaus, beam, ndipole, nx, ny, nz&
          &, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, Eimagez, FF0, tolinit, plan2f, plan2b, nstop, infostr)

  else if (beam(1:15) == 'gparawavelinear') then
     write(*, *) 'theta=', theta
     write(*, *) 'phi=', phi
     write(*, *) 'pola=', pp

     call gaussianpuissancepara(w0, P0, irra, E0)

     xgaus = xgaus * nmtom
     ygaus = ygaus * nmtom
     zgaus = zgaus * nmtom

     I0 = abs(E0)**2
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_paraxial_gaussian_field_cuda(xs,ys,zs,int(nbsphere,c_int), &
          E0,eps0,ss,pp,theta,phi,xgaus,ygaus,zgaus,w0,k0,0_c_int,FF0,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA paraxial Gaussian illumination failed'
        return
     endif
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere
        call gaussianparalinear(xs(i), ys(i), zs(i), xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, FF0(3 * i - 2),&
             & FF0(3 * i - 1), FF0(3 * i), nstop, infostr)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif
  else if (beam(1:17) == 'gparawavecircular') then
     write(*, *) 'theta=', theta
     write(*, *) 'phi=', phi
     write(*, *) 'pola=', ss

     xgaus = xgaus * nmtom
     ygaus = ygaus * nmtom
     zgaus = zgaus * nmtom

     call gaussianpuissancepara(w0, P0, irra, E0)
     I0 = abs(E0)**2
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_paraxial_gaussian_field_cuda(xs,ys,zs,int(nbsphere,c_int), &
          E0,eps0,ss,pp,theta,phi,xgaus,ygaus,zgaus,w0,k0,1_c_int,FF0,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA circular paraxial Gaussian illumination failed'
        return
     endif
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere
        call gaussianparacirc(xs(i), ys(i), zs(i), xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, eps0, E0, FF0(3 * i - 2), FF0(3 * i&
             & - 1), FF0(3 * i), nstop, infostr)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif
  else if (beam(1:7) == 'speckle') then
     E0 = one
     ! psi=pp

     xgaus = xgaus * nmtom
     ygaus = ygaus * nmtom
     zgaus = zgaus * nmtom
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_prepare_speckle_phases(nfft2d,nfft2d,k0,numaperil, &
          aretecube,speckseed,1_ip,Eimagex,Eimagey)
     call ifdda_speckle_field_cuda(int(nx,c_int),int(ny,c_int), &
          int(nz,c_int),int(ndipole,c_int),int(nfft2d,c_int),k0, &
          xgaus,ygaus,zgaus,pp,aretecube,P0,numaperil,E0,Eimagex, &
          Eimagey,xs,ys,zs,1_c_int,FF0,irra,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        select case (cuda_green_status)
        case (2); infostr='nfft2d too small for confocal'
        case (3); infostr='object larger than the FFT box along x'
        case (4); infostr='object larger than the FFT box along y'
        case (5); infostr='CUDA speckle grid dimensions mismatch'
        case default; infostr='CUDA speckle illumination failed'
        end select
        return
     endif
#else
     call speckle(k0, xgaus, ygaus, zgaus, pp, E0, xs, ys, zs, FF0, &
          aretecube, nx, ny, nz, ndipole, nmax, nfft2d, nfft2d, Eimagex, &
          Eimagey, Eimagez, speckseed, P0, irra, numaperil, nstop, &
          infostr, plan2b)
#endif
     if (nstop == 1) return
  else if (beam(1:11) == 'demispeckle') then
     E0 = one
     ! psi=pp

     xgaus = xgaus * nmtom
     ygaus = ygaus * nmtom
     zgaus = zgaus * nmtom
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_prepare_speckle_phases(nfft2d,nfft2d,k0,numaperil, &
          aretecube,speckseed,0_ip,Eimagex,Eimagey)
     call ifdda_speckle_field_cuda(int(nx,c_int),int(ny,c_int), &
          int(nz,c_int),int(ndipole,c_int),int(nfft2d,c_int),k0, &
          xgaus,ygaus,zgaus,pp,aretecube,P0,numaperil,E0,Eimagex, &
          Eimagey,xs,ys,zs,0_c_int,FF0,irra,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        select case (cuda_green_status)
        case (2); infostr='nfft2d too small for confocal'
        case (3); infostr='object larger than the FFT box along x'
        case (4); infostr='object larger than the FFT box along y'
        case (5); infostr='CUDA speckle grid dimensions mismatch'
        case default; infostr='CUDA demi-speckle illumination failed'
        end select
        return
     endif
#else
     ! demispeckle accumulates each z-plane into FF0; its intent(out) dummy
     ! must start at zero, just as the CUDA adapter's persistent field buffer does.
     FF0=zzero
     call demispeckle(k0, xgaus, ygaus, zgaus, pp, E0, xs, ys, zs, FF0, &
          aretecube, nx, ny, nz, ndipole, nmax, nfft2d, nfft2d, Eimagex, &
          Eimagey, Eimagez, speckseed, P0, irra, numaperil, nstop, infostr, &
          plan2b)
#endif
     if (nstop == 1) return
  else if (beam(1:8) == 'confocal') then
     E0 = one
     ! psi=pp

     xgaus = xgaus * nmtom
     ygaus = ygaus * nmtom
     zgaus = zgaus * nmtom
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_confocal_field_cuda(int(nx,c_int),int(ny,c_int), &
          int(nz,c_int),int(ndipole,c_int),int(nfft2d,c_int),k0, &
          xgaus,ygaus,zgaus,pp,aretecube,P0,numaperil,E0,xs,ys,zs, &
          1_c_int,FF0,irra,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA confocal illumination failed'
        return
     endif
#else
     call confocal(k0, xgaus, ygaus, zgaus, pp, E0, xs, ys, zs, FF0, &
          aretecube, nx, ny, nz, ndipole, nmax, nfft2d, nfft2d, Eimagex, &
          Eimagey, Eimagez, P0, irra, numaperil, nstop, infostr, &
          plan2b)
#endif
     if (nstop == 1) return
  else if (beam(1:12) == 'demiconfocal') then

     write(*, *) 'confocal demi'
     E0 = one
     ! psi=pp

     xgaus = xgaus * nmtom
     ygaus = ygaus * nmtom
     zgaus = zgaus * nmtom
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_confocal_field_cuda(int(nx,c_int),int(ny,c_int), &
          int(nz,c_int),int(ndipole,c_int),int(nfft2d,c_int),k0, &
          xgaus,ygaus,zgaus,pp,aretecube,P0,numaperil,E0,xs,ys,zs, &
          0_c_int,FF0,irra,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA demi-confocal illumination failed'
        return
     endif
#else
     call demiconfocal(k0, xgaus, ygaus, zgaus, pp, E0, xs, ys, zs, FF0, &
          aretecube, nx, ny, nz, ndipole, nmax, nfft2d, nfft2d, Eimagex, &
          Eimagey, Eimagez, P0, irra, numaperil, nstop, infostr, &
          plan2b)
#endif
     if (nstop == 1) return

  else if (beam(1:9) == 'arbitrary') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     ! The following CPU phase still reads FF0 to populate incidentfield and
     ! write the incident-field output files. Keep the host result valid until
     ! that phase is ported; the interpolation itself is already on the GPU.
     cuda_arbitrary_direct=0_c_int
     call ifdda_incident_arbitrary_cuda(xs,ys,zs,aretecube,FF0, &
          int(nxm,c_int),int(nym,c_int),int(nzm,c_int), &
          int(nbsphere,c_int),cuda_arbitrary_direct,cuda_arbitrary_status, &
          namefileinc,infostr)
     nstop=int(cuda_arbitrary_status,ip)
#else
     call incidentarbitrary(xs, ys, zs, aretecube, FF0, nxm, nym, nzm, nbsphere, nstop, namefileinc, infostr)
#endif

     if (nstop == 1) return

  end if

  !     ajoutte le masque si pml
  if (nabc.eq.1) then
     allocate(masquenrbc(nmax))
     write(*,*) 'begin mask for MDDA'
     !         if (ntypeabc.eq.0.or.ntypeabc.eq.1) then
     !            ntypemask=0
     !         else
     ntypemask=1
     !         endif
     !le masque recupÃ¨re la diffÃ©rence d'amplitude sur E0 du fait de l'onde exponentielle.     
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_apply_pml_mask_cuda(xs,ys,zs,eps0,k0,int(pmlwrk,c_int), &
          int(ntypemask,c_int),int(nbsphere,c_int),aretecube,FF0, &
          masquenrbc,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA PML mask application failed'
        return
     endif
#else
     call masquepml(xs,ys,zs,E0,eps0,k0,FF0,nmax,aretecube,nbsphere,pmlwrk,ntypemask,masquenrbc)
#endif
     write(*,*) 'End mask'
  endif


301 if (beam(1:11) == 'greentensor') then
     ! initialise pour le calcul de la fonction de Green
     if (testgreen == 1) then
        xdip = xgaus * nmtom
        ydip = ygaus * nmtom
        zdip = zgaus * nmtom
        write(*, *) 'Position of the dipole', xdip, ydip, zdip
        call dipoleinsidep(xdip, ydip, zdip, xs, ys, zs, aretecube, nmax, nbsphere, k)
        write(*, *) 'Position of the dipole=subunit', xdip, ydip, zdip
        if (k == 1) then
           nstop = 1
           infostr = 'Point outside the object under study'
           return
        end if
        nrig = 0
        nforce = 0
        ntorque = 0
        nforced = 0
        ntorqued = 0
        nsection = 0
        nsectionsca = 0
        nquickdiffracte = 0
        nlentille = 0
        nquicklens = 0
        nenergie = 0
        nlocal = 0
        nmacro = 1
        pinc(1) = zone
        pinc(2) = zzero
        pinc(3) = zzero
     else if (testgreen == 2) then
        pinc(1) = zzero
        pinc(2) = zone
        pinc(3) = zzero
     else if (testgreen == 3) then
        pinc(1) = zzero
        pinc(2) = zzero
        pinc(3) = zone
     end if
     testgreen = testgreen + 1

     call dipoleincp_batch(xdip, ydip, zdip, pinc, xs, ys, zs, nbsphere, &
          aretecube, k0, FF0, nstop, infostr)

  end if

  ! ecriture dans .mat et creation de incidentifeld et composantes.
  subunit = 0
  if (nmat == 0) then
     do i = 1, ndipole
        k = tabdip(i)
        if (k /= 0) then
           subunit = subunit + 1
           incidentfieldx(subunit) = FF0(3 * k - 2)
           incidentfieldy(subunit) = FF0(3 * k - 1)
           incidentfieldz(subunit) = FF0(3 * k)
           incidentfield(subunit) = sqrt(real(FF0(3 * k - 2) &
                * conjg(FF0(3 * k - 2)) + FF0(3 * k - 1) * conjg(FF0(3 * k - 1)) &
                + FF0(3 * k) * conjg(FF0(3 * k))))
           write(36, *) real(incidentfieldx(subunit),wp), aimag(incidentfieldx(subunit))
           write(37, *) real(incidentfieldy(subunit),wp), aimag(incidentfieldy(subunit))
           write(38, *) real(incidentfieldz(subunit),wp), aimag(incidentfieldz(subunit))
           write(39, *) incidentfield(subunit)
        else
           write(36, *) zero, zero
           write(37, *) zero, zero
           write(38, *) zero, zero
           write(39, *) zero
        end if
     end do
  else if (nmat == 2) then

     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, ndipole
        k = tabdip(i)
        if (k /= 0) then
           incidentfieldx(k) = FF0(3 * k - 2)
           incidentfieldy(k) = FF0(3 * k - 1)
           incidentfieldz(k) = FF0(3 * k)
           incidentfield(k) = sqrt(real(FF0(3 * k - 2) * conjg(FF0(3 * k - 2)) &
                + FF0(3 * k - 1) * conjg(FF0(3 * k - 1)) + FF0(3 * k) * conjg(FF0(3 * k)),wp))
           wrk(i, 1) = FF0(3 * k - 2)
           wrk(i, 2) = FF0(3 * k - 1)
           wrk(i, 3) = FF0(3 * k)
           wrk(i, 4) = incidentfield(k)
        else
           wrk(i, 1) = zzero
           wrk(i, 2) = zzero
           wrk(i, 3) = zzero
           wrk(i, 4) = zzero
        end if
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
     dim(1) = ndipole
     dim(2) = nmax * 3
     datasetname = 'Incident field modulus'
     call hdf5write1d(group_idnf, datasetname, real(wrk(:, 4)), dim)
     datasetname = 'Incident field x component real part'
     call hdf5write1d(group_idnf, datasetname, real(Wrk(:, 1)), dim)
     datasetname = 'Incident field x component imaginary part'
     call hdf5write1d(group_idnf, datasetname, aimag(Wrk(:, 1)), dim)
     datasetname = 'Incident field y component real part'
     call hdf5write1d(group_idnf, datasetname, real(Wrk(:, 2)), dim)
     datasetname = 'Incident field y component imaginary part'
     call hdf5write1d(group_idnf, datasetname, aimag(Wrk(:, 2)), dim)
     datasetname = 'Incident field z component real part'
     call hdf5write1d(group_idnf, datasetname, real(Wrk(:, 3)), dim)
     datasetname = 'Incident field z component imaginary part'
     call hdf5write1d(group_idnf, datasetname, aimag(Wrk(:, 3)), dim)

  else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, ndipole
        k = tabdip(i)
        if (k /= 0) then
           incidentfieldx(k) = FF0(3 * k - 2)
           incidentfieldy(k) = FF0(3 * k - 1)
           incidentfieldz(k) = FF0(3 * k)
           incidentfield(k) = sqrt(real(FF0(3 * k - 2) * conjg(FF0(3 * k - 2)) &
                + FF0(3 * k - 1) * conjg(FF0(3 * k - 1)) + FF0(3 * k) * conjg(FF0(3 * k)),wp))
        end if
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
  end if
  ! cLose intensity of the incident field
  close(36)
  close(37)
  close(38)
  close(39)

  if (nstop == 1) return
  if (nstop == -1) then
     infostr = 'Calculation cancelled after incident field!'
     return
  end if

  write(*, *) 'Power      : ', P0, 'W'
  write(*, *) 'Irradiance : ', irra, 'W/m2'
  write(*, *) 'Field      : ', E0, 'V/m'

  writecol = '********** END INCIDENT FIELD ***********'
  colorw = 'pink'
  call writecolor(writecol, colorw)
  write(*, *) ' '

  if (nlecture == 1) then
     write(*, *) 'Reread or write the dipoles', nlecture
     call relecture(aretecube, lambda, nrig, nfft2d, beam, object, trope, &
          nnnr, tolinit, density, side, sidex, sidey, sidez, hauteur, &
          numberobjet, rayonmulti, xgmulti, ygmulti, zgmulti, &
          epsmulti, epsanimulti, lc, hc, ng, demiaxea, demiaxeb, &
          demiaxec, thetaobj, phiobj, psiobj, namefileobj, theta, phi, &
          pp, ss, thetam, phim, ppm, ssm, E0m, nbinc, P0, w0, xgaus, &
          ygaus, zgaus, speckseed, numaperil, namefileinc, numberobjet, &
          polarizability, nquad, filereread, nlecture1, nx, ny, nz, nstop, &
          infostr)
     write(*, *) 'reread', nlecture1, nstop, infostr
     if (nstop == 1) return
     ! reread the local field
     if (nlecture1 == 1) then
        write(*, *) 'reread the dipoles'
        file1 = '.lf'
        long = len(trim(filereread))
        long1 = len(trim(file1))
        filereread1 = filereread(1:long) // file1(1:long1)
        open(1000, file=filereread1, status='old', iostat=ierror, form='unformatted')
        if (ierror /= 0) then
           nstop = 1
           infostr = 'file for read local field do not exist!'
           return
        else
           do i = 1, nbsphere3
              read(1000) FF(i), FFloc(i)
           end do
        end if
        close(1000)
        goto 1000
     end if
  end if

  if (ninitest == 4 .and. nrig == 8) ninitest = 1
  if (ninitest == 4 .and. nrig == 9) ninitest = 1

  if (ninitest == 4 .and. nrig == 0) then
     nrig = 8
     ntmp = nprecon
     nprecon = 0
  end if
  if (ninitest == 5 .and. nrig == 0) then
     nrig = 8
     ntmp = nprecon
     nprecon = 0
  end if






  ! if the computation asked is rigourous then compute the Green
  ! function
333 if (nrig == 0 .or. nrig == 3 .or. nrig == 4 .or. nrig == 5) then

     call primefactor(nx, 1, test)
     call primefactor(ny, 2, test)
     call primefactor(nz, 3, test)

     writecol = '**** BEGIN COMPUTATION GREEN FUNCTION ****'
     colorw = 'green'
     write(*, *) '******************************************'
     call writecolor(writecol, colorw)
     write(*, *) '******************************************'
     ! Compute the Green fonction in free space
     if (nabc == 1) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
         cuda_green_filtered=0
         call ifdda_generate_vector_green_cuda(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
              FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,nx,ny,nz,nxm,nym,nzm,nx2,ny2,nz2, &
              aretecube,k0,eps0,1_c_int,nquad,cuda_green_filtered,cuda_green_status)
         if (cuda_green_status /= 0) then
            nstop=1
            infostr='CUDA medium Green tensor generation failed'
            return
         end if
#else
        if (nquad == 0) then
           call greencalculfftc(nx,ny,nz,nx2,ny2,nz2,nxy2,nxm,nym,nzm,aretecube,k0,eps0,ntypeabc,theta,phi,tolinit ,FFTTENSORxx &
                &,FFTTENSORxy ,FFTTENSORxz ,FFTTENSORyy,FFTTENSORyz ,FFTTENSORzz,vectx,vecty ,vectz ,planb)
        else
           call greencalculquadfftc(nx,ny,nz,nx2,ny2,nz2,nxy2,nxm ,nym,nzm,nquad,aretecube,k0,eps0,ntypeabc,theta,phi ,tolinit &
                &,FFTTENSORxx ,FFTTENSORxy ,FFTTENSORxz ,FFTTENSORyy,FFTTENSORyz ,FFTTENSORzz,vectx,vecty ,vectz ,planb)
        endif
#endif
     else
        if (nquad == 0) then
           if (polarizability /= 'FG') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              cuda_green_filtered=0
              call ifdda_generate_vector_green_cuda(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
                   FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,nx,ny,nz,nxm,nym,nzm,nx2,ny2,nz2, &
                   aretecube,k0,eps0,0_c_int,0_c_int,cuda_green_filtered,cuda_green_status)
              if (cuda_green_status /= 0) then
                 nstop=1
                 infostr='CUDA vector Green tensor generation failed'
                 return
              end if
#else
              call greencalculfft(nx, ny, nz, nx2, ny2, nz2, nxy2, nxm, nym, nzm, aretecube, k0, FFTTENSORxx, FFTTENSORxy,&
                   & FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz, planb)
#endif
           else
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              cuda_green_filtered=1
              call ifdda_generate_vector_green_cuda(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
                   FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,nx,ny,nz,nxm,nym,nzm,nx2,ny2,nz2, &
                   aretecube,k0,eps0,0_c_int,0_c_int,cuda_green_filtered,cuda_green_status)
              if (cuda_green_status /= 0) then
                 nstop=1
                 infostr='CUDA filtered Green tensor generation failed'
                 return
              end if
#else
              call greenfilcalculfft(nx, ny, nz, nx2, ny2, nz2, nxy2, nxm, nym, nzm, aretecube, k0, FFTTENSORxx, FFTTENSORxy,&
                   & FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz, planb)
#endif
           end if
        else
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           cuda_green_filtered=0
           call ifdda_generate_vector_green_cuda(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
                FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,nx,ny,nz,nxm,nym,nzm,nx2,ny2,nz2, &
                aretecube,k0,eps0,0_c_int,nquad,cuda_green_filtered,cuda_green_status)
           if (cuda_green_status /= 0) then
              nstop=1
              infostr='CUDA quadrature Green tensor generation failed'
              return
           end if
#else
           call greencalculquadfft(nx, ny, nz, nx2, ny2, nz2, nxy2, nxm, nym, nzm, nquad, tolinit, aretecube, k0, FFTTENSORxx,&
                & FFTTENSORxy, FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz, planb)
#endif
        end if
     end if

     writecol = '***** END COMPUTATION GREEN FUNCTION *****'
     colorw = 'green'
     call writecolor(writecol, colorw)
     write(*, *) ' '
     ! Compute the incident field at each subunit of the object
     if (nstop == -1) then
        infostr = 'Calculation cancelled after FFT Green function!'
        return
     end if
  end if

  writecol = '***** BEGIN TO SOLVE LINEAR SYSTEM *******'
  colorw = 'pink'
  write(*, *) '******************************************'
  call writecolor(writecol, colorw)
  write(*, *) '******************************************'

  call cpu_time(t1)
  call date_and_time(date, time, zzone, values)

  if (nrig == 0) then

     cuda_vector_direct_guess = .false.
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_vector_direct_guess = &
          (nprecon == 0 .or. nprecon == 1 .or. nprecon == 2) .and. &
          (ninitest == 0 .or. ninitest == 1 .or. ninitest == 2 .or. &
          ninitest == 3) .and. ( &
          methodeit(1:7) == 'GPBICG1' .or. methodeit(1:7) == 'GPBICG2' .or. &
          methodeit(1:10) == 'GPBICGplus' .or. methodeit(1:10) == 'GPBICGsafe' .or. &
          methodeit(1:9) == 'GPBICGAR2' .or. methodeit(1:7) == 'GPBICOR' .or. &
          methodeit(1:8) == 'GPBICGAR' .or. methodeit(1:4) == 'CORS' .or. &
          methodeit(1:12) == 'QMRBICGSTAB1' .or. methodeit(1:12) == 'QMRBICGSTAB2' .or. &
          methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
          methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. &
          methodeit(1:2) == 'CG' .or. methodeit(1:12) == 'BICGSTARPLUS' .or. &
          methodeit(1:12) == 'SSBICGSAFE2' .or. methodeit(1:6) == 'QMRCLA' .or. &
          methodeit(1:4) == 'IDRS' .or. methodeit(1:6) == 'GCORS2' .or. &
          methodeit(1:9) == 'BORNSERIE' )
     ! The BPM field can be written directly into the resident Xi allocation
     ! shared by the standard Krylov drivers. GCORS2/GPBICOR own a separate
     ! state layout, so keep their established host hand-off for this guess.
     if (ninitest == 3 .and. &
         (methodeit(1:6) == 'GCORS2' .or. methodeit(1:7) == 'GPBICOR')) &
          cuda_vector_direct_guess = .false.
     ! The diagnostic BPM path deliberately copies layer state back to the
     ! host; do not tell the solver that XI is already resident in that mode.
     if (ninitest == 3) then
        call get_environment_variable('IFDDA_BPM_TRACE', &
             status=cuda_bpm_trace_status)
        if (cuda_bpm_trace_status == 0) cuda_vector_direct_guess = .false.
     end if
     ! The non-optimized Fortran left-Chan path keeps XI equal to the
     ! caller-provided initial iterate. Let the resident CUDA driver apply
     ! only P^-1 to the RHS; do not replace XI with its Hegedus-normalized
     ! Born iterate.
     if (nprecon == 1 .and. (ninitest == 0 .or. ninitest == 1)) &
          cuda_vector_direct_guess = .false.
#ifdef USE_SIMP
     ! The Green-tensor incident field can exceed the dynamic range of the
     ! single-precision resident Born initializer.  Its normalized host
     ! iterate is still valid, so use the established upload path here.
     if (beam(1:11) == 'greentensor') cuda_vector_direct_guess = .false.
#endif
     cuda_initial_guess_mode = -1_c_int
     if (cuda_vector_direct_guess .and. (nprecon == 0 .or. nprecon == 1 .or. nprecon == 2) .and. &
          (ninitest == 0 .or. ninitest == 1)) &
          cuda_initial_guess_mode = int(ninitest,c_int)
     ! The non-preconditioned optimized driver also requires the resident
     ! Rytov initializer (mode 2); otherwise the host Rytov branch above is
     ! skipped while the solver receives mode -1 and an uninitialized Xi.
     if (cuda_vector_direct_guess .and. (nprecon == 0 .or. nprecon == 1) .and. ninitest == 2) &
          cuda_initial_guess_mode = 2_c_int
     if (cuda_vector_direct_guess .and. nprecon == 2 .and. ninitest == 2) &
          cuda_initial_guess_mode = 3_c_int
     ! Mode 3 is the CUDA driver's Rytov initializer. BPM already constructs
     ! XI through beampropagation, so mode 4 tells the driver to preserve and
     ! upload that completed iterate instead of rebuilding it as Rytov.
     if (cuda_vector_direct_guess .and. ninitest == 3) &
          cuda_initial_guess_mode = 4_c_int
#ifdef USE_SIMP
     if (beam(1:11) == 'greentensor' .and. .not. cuda_vector_direct_guess) &
          cuda_initial_guess_mode = -2_c_int
#endif
     call ifdda_set_initial_guess_mode_cuda(cuda_initial_guess_mode)
#endif

     ! Compute the local field at each subunit position by solving Ax=b
     ! as an initial guess the incident field
     ! initilization for solve Ax=b
     tol = tolinit
     ncompte = 0
     nloop = 0
     if (nproche == -1) then
        if (ninitest == 0 .and. .not. cuda_vector_direct_guess) then
           !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, nbsphere3
              xi(i) = zzero
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL
        else if (ninitest == 1 .and. .not. cuda_vector_direct_guess) then
           !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, nbsphere3
              xi(i) = FF0(i)
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL
        end if

        if (nprecon == 0) then
#ifdef USE_CUDA
           call inverserig_cuda(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz,&
                & Tabdip, nlar, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nz2 , nbsphere, XI, XR, wrk, FF, FF0,&
                & FFloc, polarisa, methodeit, trope, tol, tol1, nloop, ncompte, nlim, nstop, infostr)
           if (nstop == 1) return
#else
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           if (methodeit(1:7) == 'GPBICG1' .or. &
               methodeit(1:7) == 'GPBICG2' .or. &
               methodeit(1:10) == 'GPBICGplus' .or. &
               methodeit(1:10) == 'GPBICGsafe' .or. &
               methodeit(1:9) == 'GPBICGAR2' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
               methodeit(1:8) == 'GPBICGAR' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
               methodeit(1:4) == 'CORS' .or. &
               methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
                methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
                methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. &
               methodeit(1:2) == 'CG' .or. &
               methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
                methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
               methodeit(1:12) == 'QMRBICGSTAB1' .or. &
               methodeit(1:12) == 'QMRBICGSTAB2' .or. &
               methodeit(1:12) == 'BICGSTARPLUS' .or. &
               methodeit(1:12) == 'SSBICGSAFE2' .or. &
               methodeit(1:9) == 'BICGLSTAB' .or. &
               methodeit(1:11) == 'GPBICGSTABL' .or. &
               methodeit(1:6) == 'QMRCLA' .or. &
               methodeit(1:4) == 'IDRS' .or. &
               methodeit(1:6) == 'GCORS2' .or. &
               methodeit(1:9) == 'BORNSERIE') then
#ifdef USE_SIMP
              cuda_green_norm = 0.0_c_double
              if (beam(1:11) == 'greentensor' .and. ninitest == 1 .and. &
                  .not. cuda_vector_direct_guess) then
                 do i = 1, nbsphere3
                    cuda_green_norm = cuda_green_norm + &
                         real(FF0(i),kind=c_double) * real(FF0(i),kind=c_double) + &
                         real(aimag(FF0(i)),kind=c_double) * real(aimag(FF0(i)),kind=c_double)
                 end do
                 cuda_green_norm = sqrt(cuda_green_norm)
                 do i = 1, nbsphere3
                    FF0(i) = FF0(i) / cuda_green_norm
                    XI(i) = XI(i) / cuda_green_norm
                 end do
              end if
#endif
              call inverserig_all_cuda_driver(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
                   FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz, &
                   Tabdip, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                   nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, nbsphere3, XI, XR, wrk, &
                   FF, FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, &
                   nlim, planXf, planXb, planYf, planYb, planZf, planZb, planf, planb, &
                   0, merge(int(cuda_initial_guess_mode), -1, &
                   cuda_vector_direct_guess), nprecon, nstop, infostr)
#ifdef USE_SIMP
              if (cuda_green_norm > 0.0_c_double) then
                 do i = 1, nbsphere3
                    FF0(i) = FF0(i) * cuda_green_norm
                    FF(i) = FF(i) * cuda_green_norm
                    FFloc(i) = FFloc(i) * cuda_green_norm
                 end do
              end if
#endif
           else
              nstop = 1
              infostr = 'Unsupported solver in CUDA inverserig driver'
#endif
#ifndef IFDDA_CUDA_INVERSE_DRIVER
           call inverserig(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
                FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, &
                vectz, Tabdip, ntotalm, ntotal, ldabi, nlar, nmax, &
                ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, &
                nbsphere, nbsphere3, XI, XR, wrk, FF, FF0, FFloc, polarisa, &
                niso, methodeit, tol, tol1, nloop, ncompte, nlim, &
                planXf, planXb, planYf, planYb, planZf, planZb, planf, &
                planb, nstop, infostr)
#endif
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           endif
           if (nstop == 1) return
#endif
#endif
        else if (nprecon == 1) then
           call inverserigpg(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
                FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, &
                vectz, tabdip, ntotalm, ntotal, ldabi, nlar, nmax, &
                ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, &
                nbsphere, nbsphere3, XI, XR, wrk, FF, FF0, FFloc, &
                FFprecon, polarisa, niso, methodeit, tol, tol1, &
                nloop, ncompte, nlim, k0, aretecube, localfieldx, &
                localfieldy, localfieldz, macroscopicfieldx, &
                macroscopicfieldy, macroscopicfieldz, sdetnn, planf, &
                planb, nstop, infostr)

        else if (nprecon == 2) then
           call inverserigpd(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
                FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, &
                vectz, tabdip, ntotalm, ntotal, ldabi, nlar, nmax, &
                ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, &
                nbsphere, nbsphere3, XI, XR, wrk, FF, FF0, FFloc, &
                FFprecon, polarisa, niso, methodeit, tol, tol1, &
                nloop, ncompte, nlim, k0, aretecube, localfieldx, &
                localfieldy, localfieldz, macroscopicfieldx, &
                macroscopicfieldy, macroscopicfieldz, sdetnn, planf, &
                planb, nstop, infostr)

        end if

     else
        if (ninitest == 0 .and. .not. cuda_vector_direct_guess) then
           !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, nbsphere3
              xi(i) = zzero
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL
        else if (ninitest == 1 .and. .not. cuda_vector_direct_guess) then
           !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, nbsphere3
              xi(i) = FF0(i)
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL
        ! The CPU fallback constructs the Rytov iterate here.  When the
        ! CUDA driver accepts the requested guess, it builds this iterate on
        ! the device, so keep this host-side work out of the CUDA path.
        else if (ninitest == 2 .and. .not. cuda_vector_direct_guess) then

           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, nbsphere3
              xr(i) = zzero
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL

           if (niso == 1) then
              ! Isotropic polarizability stores one complex value per dipole.
              ! The 3x3 tensor entries below exist only for niso == 3.
              !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
              !$OMP DO SCHEDULE(STATIC)
              do i = 1, nbsphere
                 k = 3 * (i - 1)
                 xi(k + 1) = -polarisa(i, 1, 1) * FF0(k + 1)
                 xi(k + 2) = -polarisa(i, 1, 1) * FF0(k + 2)
                 xi(k + 3) = -polarisa(i, 1, 1) * FF0(k + 3)
              end do
              !$OMP ENDDO
              !$OMP END PARALLEL
           else
              !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
              !$OMP DO SCHEDULE(STATIC)
              do i = 1, nbsphere
                 k = 3 * (i - 1)
                 xi(k + 1) = -polarisa(i, 1, 1) * FF0(k + 1) - polarisa(i, 1, 2) &
                      * FF0(k + 2) - polarisa(i, 1, 3) * FF0(k + 3)
                 xi(k + 2) = -polarisa(i, 2, 1) * FF0(k + 1) - polarisa(i, 2, 2) &
                      * FF0(k + 2) - polarisa(i, 2, 3) * FF0(k + 3)
                 xi(k + 3) = -polarisa(i, 3, 1) * FF0(k + 1) - polarisa(i, 3, 2) &
                      * FF0(k + 2) - polarisa(i, 3, 3) * FF0(k + 3)
              end do
              !$OMP ENDDO
              !$OMP END PARALLEL
           end if

           call produitfftmatvect3(FFTTENSORxx, FFTTENSORxy, &
                FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, &
                vectx, vecty, vectz, Tabdip, ntotalm, ntotal, nmax, &
                ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, XI, &
                XR, planf, planb)
           call rytov_vector_field(FF0, xr, xi, nbsphere, nstop)
           if (nstop /= 0) then
              infostr = 'Rytov vector field is outside numerical range'
              return
           endif
        else if (ninitest == 3) then
           ! Paraxial Gaussian beams are supported by beampropagation, just as
           ! in the standalone BPM branches (nrig=6/7).
           if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti') then
              nstop = 1.0_wp
              infostr = 'BPM not possible with this beam'
              return
           end if

           call beampropagation(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, &
                theta, phi, xgaus, ygaus, zgaus, beam, epsilon, niso, &
                ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, &
                Eimagey, Eimagez, FF0, xi, FF, plan2f, plan2b, nstop, infostr, &
                merge(1,0,cuda_vector_direct_guess))
#ifndef IFDDA_CUDA_INVERSE_DRIVER
           write(*, *) 'bpm', xi(nbsphere:nbsphere + 10)
#endif
           if (nstop == 1) return

        end if

        if (nprecon == 0) then
#ifdef USE_CUDA

           if (methodeit(1:10) == 'BORNMSERIE') call calculgamma(wrk,polarisa,nbsphere,eps0i,aretecube,nlar,nmax)
           call inverserigopt_cuda(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty,&
                & vectz, nlar, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nz2, nbsphere, XI, XR, wrk, FF, FF0, FFloc ,&
                & polarisa, methodeit, trope, tol, tol1, nloop, ncompte, nlim, nstop, infostr)

#else
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           if (methodeit(1:7) == 'GPBICG1' .or. &
               methodeit(1:7) == 'GPBICG2' .or. &
               methodeit(1:10) == 'GPBICGplus' .or. &
               methodeit(1:10) == 'GPBICGsafe' .or. &
               methodeit(1:9) == 'GPBICGAR2' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
               methodeit(1:8) == 'GPBICGAR' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
               methodeit(1:4) == 'CORS' .or. &
               methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
                methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
                methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. &
               methodeit(1:2) == 'CG' .or. &
               methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
                methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
               methodeit(1:12) == 'QMRBICGSTAB1' .or. &
               methodeit(1:12) == 'QMRBICGSTAB2' .or. &
               methodeit(1:12) == 'BICGSTARPLUS' .or. &
               methodeit(1:12) == 'SSBICGSAFE2' .or. &
               methodeit(1:9) == 'BICGLSTAB' .or. &
               methodeit(1:11) == 'GPBICGSTABL' .or. &
               methodeit(1:6) == 'QMRCLA' .or. &
               methodeit(1:4) == 'IDRS' .or. &
               methodeit(1:6) == 'GCORS2' .or. &
               methodeit(1:9) == 'BORNSERIE') then
#ifdef USE_SIMP
              cuda_green_norm = 0.0_c_double
              if (beam(1:11) == 'greentensor' .and. ninitest == 1 .and. &
                  .not. cuda_vector_direct_guess) then
                 do i = 1, nbsphere3
                    cuda_green_norm = cuda_green_norm + &
                         real(FF0(i),kind=c_double) * real(FF0(i),kind=c_double) + &
                         real(aimag(FF0(i)),kind=c_double) * real(aimag(FF0(i)),kind=c_double)
                 end do
                 cuda_green_norm = sqrt(cuda_green_norm)
                 do i = 1, nbsphere3
                    FF0(i) = FF0(i) / cuda_green_norm
                    XI(i) = XI(i) / cuda_green_norm
                 end do
              end if
#endif
              call inverserig_all_cuda_driver(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
                   FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz, &
                   Tabdip, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                   nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, nbsphere3, XI, XR, wrk, &
                   FF, FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, &
                   nlim, planXf, planXb, planYf, planYb, planZf, planZb, planf, planb, &
                   1, merge(int(cuda_initial_guess_mode), -1, &
                   cuda_vector_direct_guess), nprecon, nstop, infostr)
#ifdef USE_SIMP
              if (cuda_green_norm > 0.0_c_double) then
                 do i = 1, nbsphere3
                    FF0(i) = FF0(i) * cuda_green_norm
                    FF(i) = FF(i) * cuda_green_norm
                    FFloc(i) = FFloc(i) * cuda_green_norm
                 end do
              end if
#endif
           else
              nstop = 1
              infostr = 'Unsupported solver in CUDA inverserigopt driver'
           endif
#else
           call inverserigopt(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz,&
                & ntotalm, ntotal, ldabi, nlar, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, nbsphere3, XI, XR,&
                & wrk, FF, FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, planXf, planXb, planYf, planYb,&
                & planZf, planZb, planf, planb, nstop, infostr)
#endif

#endif

        else if (nprecon == 1) then
           call inverserigoptpg(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
                FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, &
                vectz, tabdip, ndipole, ntotalm, ntotal, ldabi, nlar, nmax, &
                nxm, nym, nzm, &
                nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, nbsphere3, XI, XR, &
                wrk, FF, FF0, FFloc, FFprecon, polarisa, niso, &
                methodeit, tol, tol1, nloop, ncompte, nlim, k0, &
                aretecube, localfieldx, localfieldy, localfieldz, &
                macroscopicfieldx, macroscopicfieldy, macroscopicfieldz, sdetnn, planf, &
                planb, nstop, infostr)
        else if (nprecon == 2) then
           call inverserigoptpd(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
                FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, &
                vectz, tabdip, ndipole, ntotalm, ntotal, ldabi, nlar, nmax, &
                nxm, nym, nzm, &
                nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, nbsphere3, XI, XR, &
                wrk, FF, FF0, FFloc, FFprecon, polarisa, niso, &
                methodeit, tol, tol1, nloop, ncompte, nlim, k0, &
                aretecube, localfieldx, localfieldy, localfieldz, &
                macroscopicfieldx, macroscopicfieldy, macroscopicfieldz, sdetnn, planf, &
                planb, nstop, infostr)

        end if
     end if
     if (nstop == 1) return

     if (tol1 >= tolinit) then
        nstop = 1
        infostr = 'Converge do not reach!'
        write(*, *) 'Converge do not reach!', tol1, 'Tol Asked', tolinit
        return
     end if

     if (nstop == -1) then
        infostr = 'Calculation cancelled after iterative method!'
        return
     end if

  else if (nrig == 1) then
     ! **************************************
     ! Renormalized Born approximation field
     ! **************************************
     write(*, *) 'Renormalized Born approximation'
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_nmax = int(nmax,c_int)
     cuda_niso = int(niso,c_int)
     cuda_ndipoles = int(nbsphere,c_int)
     call ifdda_renormalized_born_cuda(FF0,polarisa,FFloc,FF, &
          cuda_nmax,cuda_niso,cuda_ndipoles,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop = 1
        infostr = 'CUDA renormalized Born field calculation failed'
        return
     end if
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere3
        FFloc(i) = FF0(i)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
     if (niso == 1) then
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k,ii,jj)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           k = 3 * (i - 1)
           FF(k + 1) = polarisa(i, 1, 1) * FFloc(k + 1)
           FF(k + 2) = polarisa(i, 1, 1) * FFloc(k + 2)
           FF(k + 3) = polarisa(i, 1, 1) * FFloc(k + 3)
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
     else
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k,ii,jj)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           k = 3 * (i - 1)
           FF(k + 1) = polarisa(i, 1, 1) * FFloc(k + 1) + polarisa(i, 1, 2) &
                * FFloc(k + 2) + polarisa(i, 1, 3) * FFloc(k + 3)
           FF(k + 2) = polarisa(i, 2, 1) * FFloc(k + 1) + polarisa(i, 2, 2) &
                * FFloc(k + 2) + polarisa(i, 2, 3) * FFloc(k + 3)
           FF(k + 3) = polarisa(i, 3, 1) * FFloc(k + 1) + polarisa(i, 3, 2) &
                * FFloc(k + 2) + polarisa(i, 3, 3) * FFloc(k + 3)
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
     end if
#endif
  else if (nrig == 2) then
     ! **************************************
     ! Born approximation field
     ! **************************************
     write(*, *) 'Born approximation'
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_nmax = int(nmax,c_int)
     cuda_niso = int(niso,c_int)
     cuda_ndipoles = int(nbsphere,c_int)
     cuda_aretecube = real(aretecube,kind(cuda_aretecube))
     cuda_k0 = real(k0,kind(cuda_k0))
     call ifdda_born_cuda(FF0,epsilon,polarisa,FFloc,FF, &
          cuda_nmax,cuda_niso,cuda_ndipoles,cuda_aretecube,cuda_k0,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop = 1
        infostr = 'CUDA Born field calculation failed'
        return
     end if
#else
     nsens = -1
     const=aretecube*aretecube*aretecube/4.0_wp/pi
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(k,kk,Em,Eloc,epsani)
     !$OMP DO SCHEDULE(STATIC)
     do k = 1, nbsphere
        kk = 3 * (k - 1)
        Em(1) = FF0(kk + 1)
        Em(2) = FF0(kk + 2)
        Em(3) = FF0(kk + 3)
        call copiani(epsilon, epsani, nmax, niso, k)
        call local_macro(Eloc, Em, epsani, aretecube, k0, nsens)
        FFloc(kk + 1) = Eloc(1)
        FFloc(kk + 2) = Eloc(2)
        FFloc(kk + 3) = Eloc(3) 
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)
#endif
  else if (nrig == 3) then
     ! renormalized Born ordre 1
     write(*, *) 'Born series order 1'
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_count = int(nbsphere3,c_int)
     cuda_negate = 0_c_int
     call ifdda_vector_copy_cuda(FF0,xr,cuda_count,cuda_negate,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop = 1
        infostr = 'CUDA Born-series incident copy failed'
        return
     end if
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere3
        xr(i) = FF0(i)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif

     call produitpolavect(polarisa, niso, nmax, nbsphere, xr, xi)
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_count = int(3*nbsphere,c_int)
     call ifdda_vector_negate_cuda(xi,cuda_count,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop = 1
        infostr = 'CUDA Born-series sign change failed'
        return
     end if
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere * 3
        xi(i) = -xi(i)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif

     call produitfftmatvect3(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
          FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz, &
          Tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, &
          nz, nx2, ny2, nxy2, nz2, XI, XR, planf, planb)

     call produitpolavect(polarisa, niso, nmax, nbsphere, xr, FF)
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_count = int(nbsphere3,c_int)
     cuda_negate = 0_c_int
     call ifdda_vector_copy_cuda(xr,FFloc,cuda_count,cuda_negate,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop = 1
        infostr = 'CUDA Born-series local field copy failed'
        return
     end if
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere
        k = 3 * (i - 1)
        FFloc(k + 1) = xr(k + 1)
        FFloc(k + 2) = xr(k + 2)
        FFloc(k + 3) = xr(k + 3)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif

  else if (nrig == 4) then
     ! Rytov renormalize
     write(*, *) 'Rytov approximation on local field'
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_count = int(nbsphere3,c_int)
     call ifdda_vector_zero_cuda(xr,cuda_count,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop = 1
        infostr = 'CUDA Rytov vector initialization failed'
        return
     end if
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere3
        xr(i) = zzero
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif

     call produitpolavect(polarisa, niso, nmax, nbsphere, FF0, xi)
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_count = int(3*nbsphere,c_int)
     call ifdda_vector_negate_cuda(xi,cuda_count,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop = 1
        infostr = 'CUDA Rytov polarization sign change failed'
        return
     end if
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere * 3
        xi(i) = -xi(i)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif

     call produitfftmatvect3(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
          FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, vectz, &
          Tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, &
          nz, nx2, ny2, nxy2, nz2, XI, XR, planf, planb)

     block
       character(8) :: env
       integer :: status, idx, off, j, nlarge, printed
       real(wp) :: threshold, qrmax
       complex(wp) :: exponent
       call get_environment_variable('IFDDA_RYTOV_DIAG', env, status=status)
       if (status == 0 .and. trim(env) == '1') then
         qrmax = -huge(1.0_wp)
         nlarge = 0
         printed = 0
         do idx=1,nbsphere
           off=3*(idx-1)
           threshold=sqrt(real(sum(FF0(off+1:off+3)*conjg(FF0(off+1:off+3))),wp))*1.0e-6_wp
           do j=1,3
             if (abs(FF0(off+j)) <= threshold) cycle
             exponent=xr(off+j)/FF0(off+j)
             qrmax=max(qrmax,real(exponent,wp))
             if (real(exponent,wp) > 40.0_wp) then
               nlarge=nlarge+1
               if (printed < 5) then
                 write(*,*) 'RYTOV_DIAG nrig=4 dipole component=',idx,j,&
                      ' inc=',FF0(off+j),' delta=',xr(off+j),' exponent=',exponent
                 printed=printed+1
               endif
             endif
           enddo
         enddo
         write(*,*) 'RYTOV_DIAG nrig=4 max_Re_exponent=',qrmax,' above40=',nlarge
       endif
     end block

     ! Componentwise Rytov exp(delta_j/Eref_j) is singular for weak
     ! longitudinal components of vector FFT beams. Project the perturbation
     ! onto the full reference polarization and retain transverse Born terms.
     call rytov_vector_field(FF0, xr, FFloc, nbsphere, nstop)
     if (nstop /= 0) then
        infostr = 'Rytov vector field is outside numerical range'
        return
     endif
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  else if (nrig == 5) then
     ! Rytov
     write(*, *) 'Rytov approximation on macroscopic field'
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_nmax = int(nmax,c_int)
     cuda_niso = int(niso,c_int)
     cuda_ndipoles = int(nbsphere,c_int)
     cuda_aretecube = real(aretecube,kind(cuda_aretecube))
     cuda_k0 = real(k0,kind(cuda_k0))
     call ifdda_rytov_macro_prepare_cuda(FF0,epsilon,polarisa,eps0, &
          FF,xr,xi,cuda_nmax,cuda_niso,cuda_ndipoles,cuda_aretecube, &
          cuda_k0,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop = 1
        infostr = 'CUDA Rytov macroscopic field preparation failed'
        return
     end if
#else
     if (niso == 1) then
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           k = 3 * (i - 1)
           xr(k + 1) = -FF0(k + 1) * (epsilon(i, 1, 1) - eps0) / 3.0_wp
           xr(k + 2) = -FF0(k + 2) * (epsilon(i, 1, 1) - eps0) / 3.0_wp
           xr(k + 3) = -FF0(k + 3) * (epsilon(i, 1, 1) - eps0) / 3.0_wp
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
     else
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           k = 3 * (i - 1)
           xr(k + 1) = -FF0(k + 1) * (epsilon(i, 1, 1) - eps0) / 3.0_wp
           xr(k + 2) = -FF0(k + 2) * (epsilon(i, 2, 2) - eps0) / 3.0_wp
           xr(k + 3) = -FF0(k + 3) * (epsilon(i, 3, 3) - eps0) / 3.0_wp
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
     end if
     nsens = -1

     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k,Em,Eloc,ii,jj,epsani)
     !$OMP DO SCHEDULE(STATIC)
     do i = 1, nbsphere
        k = 3 * (i - 1)
        Em(1) = FF0(k + 1)
        Em(2) = FF0(k + 2)
        Em(3) = FF0(k + 3)
        call copiani(epsilon, epsani, nmax, niso, i)
        call local_macro(Eloc, Em, epsani, aretecube, k0, nsens)
        FF(k + 1) = Eloc(1)
        FF(k + 2) = Eloc(2)
        FF(k + 3) = Eloc(3)
        if (niso == 1) then
           xi(k + 1) = -polarisa(i, 1, 1) * FF(k + 1)
           xi(k + 2) = -polarisa(i, 1, 1) * FF(k + 2)
           xi(k + 3) = -polarisa(i, 1, 1) * FF(k + 3)
        else
           xi(k + 1) = -polarisa(i, 1, 1) * FF(k + 1) - polarisa(i, 1, 2) * FF(k + 2) &
                - polarisa(i, 1, 3) * FF(k + 3)
           xi(k + 2) = -polarisa(i, 2, 1) * FF(k + 1) - polarisa(i, 2, 2) * FF(k + 2) &
                - polarisa(i, 2, 3) * FF(k + 3)
           xi(k + 3) = -polarisa(i, 3, 1) * FF(k + 1) - polarisa(i, 3, 2) * FF(k + 2) &
                - polarisa(i, 3, 3) * FF(k + 3)
        end if
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif

     call produitfftmatvect3(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, &
          FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, vectx, vecty, &
          vectz, Tabdip, ntotalm, ntotal, nmax, ndipole, nxm, nym, nzm, nx, ny, &
          nz, nx2, ny2, nxy2, nz2, XI, XR, planf, planb)

     block
       character(8) :: env
       integer :: status, idx, off, j, nlarge, printed
       real(wp) :: threshold, qrmax
       complex(wp) :: exponent
       call get_environment_variable('IFDDA_RYTOV_DIAG', env, status=status)
       if (status == 0 .and. trim(env) == '1') then
         qrmax = -huge(1.0_wp)
         nlarge = 0
         printed = 0
         do idx=1,nbsphere
           off=3*(idx-1)
           threshold=sqrt(real(sum(FF(off+1:off+3)*conjg(FF(off+1:off+3))),wp))*1.0e-6_wp
           do j=1,3
             if (abs(FF(off+j)) <= threshold) cycle
             exponent=xr(off+j)/FF(off+j)
             qrmax=max(qrmax,real(exponent,wp))
             if (real(exponent,wp) > 40.0_wp) then
               nlarge=nlarge+1
               if (printed < 5) then
                 write(*,*) 'RYTOV_DIAG nrig=5 dipole component=',idx,j,&
                      ' macro=',FF(off+j),' delta=',xr(off+j),' exponent=',exponent
                 printed=printed+1
               endif
             endif
           enddo
         enddo
         write(*,*) 'RYTOV_DIAG nrig=5 max_Re_exponent=',qrmax,' above40=',nlarge
       endif
     end block

     call rytov_vector_field(FF, xr, FFloc, nbsphere, nstop)
     if (nstop /= 0) then
        infostr = 'Rytov vector field is outside numerical range'
        return
     endif
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  else if (nrig == 6) then
     ! Beam propagation method
     write(*, *) 'Beam propagation method'

     if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti') then
        nstop = 1.0_wp
        infostr = 'BPM not possible with this beam'
        return
     end if

     call beampropagationmacro(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, &
          theta, phi, xgaus, ygaus, zgaus, beam, epsilon, niso, ndipole, nx, &
          ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, Eimagez, &
          FF0, FFloc, FF, plan2f, plan2b, nstop, infostr)

     if (nstop == 1) return
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  else if (nrig == 7) then
     ! Beam propagation method
     write(*, *) 'renormalized Beam propagation method'

     if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti') then
        nstop = 1.0_wp
        infostr = 'BPM not possible with this beam'
        return
     end if

     call beampropagation(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, theta, &
          phi, xgaus, ygaus, zgaus, beam, epsilon, niso, ndipole, nx, &
          ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, Eimagez, FF0, &
          FFloc, FF, plan2f, plan2b, nstop, infostr, 0)

     if (nstop == 1) return
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  else if (nrig == 8) then
     write(*, *) 'Scalar approximation Uinc.GUinv'
     if (trope == 'ani') then
        infostr = 'Anisotropy can not be used with scalar approximation'
        nstop = 1
        return
     end if
     if (beam(1:11) /= 'pwavelinear' .and. beam(1:13) /= 'pwavecircular' &
          .and. beam(1:15) /= 'gparawavelinear' .and. beam(1:17) /= 'gparawavecircular' &
          .and. beam(1:14) /= 'gfftwavelinear' .and. beam(1:16) /= 'gfftwavecircular') then
        infostr = 'Scalar appr. for plane wave or paraxial gaussian'
        nstop = 1
        return
     end if
     ! calcule le vecteur unitaire de polarisation incidente
     ! Approximation scalaire
     x = zero
     y = zero
     z = zero
     if (beam(1:11) == 'pwavelinear') then
        call ondeplane(x, y, z, k0, eps0, E0, ss, pp, theta, phi, Uincx, Uincy, Uincz, nstop, infostr)
     else if (beam(1:13) == 'pwavecircular') then
        call ondecirce(x, y, z, k0, E0, ss, theta, phi, Uincx, Uincy, Uincz)
     else if (beam(1:15) == 'gparawavelinear') then
        call gaussianparalinear(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Uincx, Uincy, Uincz, nstop,&
             & infostr)
     else if (beam(1:17) == 'gparawavecircular') then
        call gaussianparacirc(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, eps0, E0, Uincx, Uincy, Uincz, nstop, infostr)
     else if (beam(1:14) == 'gfftwavelinear') then
        call gaussianparalinear(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Uincx, Uincy, Uincz, nstop,&
             & infostr)
     else if (beam(1:16) == 'gfftwavecircular') then
        call gaussianparacirc(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, eps0, E0, Uincx, Uincy, Uincz, nstop, infostr)
     end if

     Emod = sqrt(abs(Uincx)**2.0_wp + abs(Uincy)**2.0_wp + abs(Uincz)**2.0_wp)
     Uincx = Uincx / Emod
     Uincy = Uincy / Emod
     Uincz = Uincz / Emod

     ! calcul la fonction de Green scalaire avec sa FFT
     call greencalculscalairefft(nx, ny, nz, nx2, ny2, nz2, nxy2, nxm, nym, nzm, aretecube, k0, Uincx, Uincy, Uincz, vectx, planb, ntotalm, nstop, infostr)
     if (nstop == 1) return

     ! The historical inverserigoptscalaire always starts from the projected
     ! incident field, including outer initialization selectors 4 and 5.
     cuda_scalar_initial_mode=ninitest
     if (ninitest == 4 .or. ninitest == 5) cuda_scalar_initial_mode=1
     cuda_scalar_guess = .false.
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_scalar_guess = (ninitest == 0 .or. ninitest == 1 .or. &
          ninitest == 2 .or. ninitest == 3 .or. ninitest == 4 .or. &
          ninitest == 5) .and. &
          (nprecon == 0 .or. ninitest == 4 .or. &
          ((nprecon == 1 .or. nprecon == 2) .and. (nrig == 8 .or. nrig == 9))) .and. &
          methodeit(1:6) /= 'GCORS2' .and. methodeit(1:9) /= 'BORNSERIE' .and. &
          (methodeit(1:7) == 'GPBICG1' .or. methodeit(1:7) == 'GPBICG2' .or. &
          methodeit(1:10) == 'GPBICGplus' .or. methodeit(1:10) == 'GPBICGsafe' .or. &
          methodeit(1:8) == 'GPBICGAR' .or. methodeit(1:9) == 'GPBICGAR2' .or. &
          methodeit(1:7) == 'GPBICOR' .or. methodeit(1:12) == 'BICGSTARPLUS' .or. &
          methodeit(1:11) == 'SSBICGSAFE2' .or. methodeit(1:4) == 'CORS' .or. &
          methodeit(1:12) == 'QMRBICGSTAB1' .or. methodeit(1:12) == 'QMRBICGSTAB2' .or. &
          methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
          methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. &
          methodeit(1:2) == 'CG' .or. methodeit(1:4) == 'IDRS' .or. &
          methodeit(1:6) == 'QMRCLA')
#endif
     if (ninitest == 0 .and. .not. cuda_scalar_guess) then
        !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           xi(i) = zzero
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
     else if (ninitest == 1 .and. .not. cuda_scalar_guess) then
        !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i,ii)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           ii = 3 * (i - 1)
           xi(i) = FF0(ii + 1) * conjg(Uincx) + FF0(ii + 2) * conjg(Uincy) + FF0(ii + 3) * conjg(Uincz)
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL

     else if (ninitest == 4) then
        if (.not. cuda_scalar_guess) then
           !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, nbsphere
              xi(i) = zzero
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL
        end if
        tmp = tolinit
        tolinit = 0.01_wp
        nprecon = 0
     end if
     tol = tolinit
     ncompte = 0
     nloop = 0
     if (nproche == -1) then
        if (nprecon == 0) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           if (methodeit(1:7) == 'GPBICG1' .or. methodeit(1:7) == 'GPBICG2' .or. &
               methodeit(1:10) == 'GPBICGplus' .or. methodeit(1:10) == 'GPBICGsafe' .or. &
               methodeit(1:8) == 'GPBICGAR' .or. methodeit(1:9) == 'GPBICGAR2' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
               methodeit(1:12) == 'BICGSTARPLUS' .or. methodeit(1:11) == 'SSBICGSAFE2' .or. &
               methodeit(1:4) == 'CORS' .or. methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
               methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
               methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. &
               methodeit(1:2) == 'CG' .or. methodeit(1:4) == 'IDRS' .or. &
               methodeit(1:6) == 'QMRCLA') then
              call inverserigscalaire_cuda_driver(vectx, vecty, Tabdip, &
                   macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                   ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                   nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                   FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, &
                   ncompte, nlim, nprecon, nrig, ninitest, 0, k0, aretecube, &
                   planf, planb, nstop, infostr)
           else
              call inverserigscalaire(vectx, vecty, Tabdip, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm,&
                   & ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, FF0,&
                   & FFloc, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, planf, planb, nstop, infostr)
           endif
#else
           call inverserigscalaire(vectx, vecty, Tabdip, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm, ntotal&
                &, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, FF0,&
                & FFloc, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, planf, planb, nstop, infostr)
#endif
        else if (nprecon == 1) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           if ((nrig == 8 .or. nrig == 9) .and. &
               (methodeit(1:7) == 'GPBICG1' .or. methodeit(1:7) == 'GPBICG2' .or. &
                methodeit(1:10) == 'GPBICGplus' .or. methodeit(1:10) == 'GPBICGsafe' .or. &
                methodeit(1:8) == 'GPBICGAR' .or. methodeit(1:9) == 'GPBICGAR2' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
                methodeit(1:12) == 'BICGSTARPLUS' .or. methodeit(1:11) == 'SSBICGSAFE2' .or. &
                methodeit(1:4) == 'CORS' .or. methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
                methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
                methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. methodeit(1:2) == 'CG' .or. &
                methodeit(1:4) == 'IDRS' .or. methodeit(1:6) == 'QMRCLA')) then
              call inverserigscalaire_cuda_driver(vectx, vecty, Tabdip, &
                   macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                   ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                   nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                   FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, &
                   ncompte, nlim, nprecon, nrig, ninitest, 0, k0, aretecube, &
                   planf, planb, nstop, infostr)
           else
              call inverserigscalairepg(vectx, vecty, Tabdip, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm,&
                   & ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF,&
                   & FF0, FFloc, FFprecon, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, k0, aretecube, localfieldx,&
                   & sdetnn, planf, planb, nrig, nstop, infostr)
           endif
#else
           call inverserigscalairepg(vectx, vecty, Tabdip, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm,&
                & ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF,&
                & FF0, FFloc, FFprecon, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, k0, aretecube, localfieldx,&
                & sdetnn, planf, planb, nrig, nstop, infostr)
#endif
        else if (nprecon == 2) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           if ((nrig == 8 .or. nrig == 9) .and. &
               (methodeit(1:7) == 'GPBICG1' .or. methodeit(1:7) == 'GPBICG2' .or. &
                methodeit(1:10) == 'GPBICGplus' .or. methodeit(1:10) == 'GPBICGsafe' .or. &
                methodeit(1:8) == 'GPBICGAR' .or. methodeit(1:9) == 'GPBICGAR2' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
                methodeit(1:12) == 'BICGSTARPLUS' .or. methodeit(1:11) == 'SSBICGSAFE2' .or. methodeit(1:4) == 'CORS' .or. methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
                methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
                methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. methodeit(1:2) == 'CG' .or. &
                methodeit(1:4) == 'IDRS' .or. methodeit(1:6) == 'QMRCLA')) then
              call inverserigscalaire_cuda_driver(vectx, vecty, Tabdip, &
                   macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                   ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                   nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                   FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, &
                   ncompte, nlim, nprecon, nrig, ninitest, 0, k0, aretecube, &
                   planf, planb, nstop, infostr)
           else
              call inverserigscalairepd(vectx, vecty, Tabdip, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm,&
                   & ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF,&
                   & FF0, FFloc, FFprecon, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, k0, aretecube, localfieldx,&
                   & sdetnn, planf, planb, nrig, nstop, infostr)
           endif
#else
           call inverserigscalairepd(vectx, vecty, Tabdip, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm,&
                & ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF,&
                & FF0, FFloc, FFprecon, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, k0, aretecube, localfieldx,&
                & sdetnn, planf, planb, nrig, nstop, infostr)
#endif
        end if
     else

        if (ninitest == 2 .and. .not. cuda_scalar_guess) then
           !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i,ii)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, nbsphere
              ii = 3 * (i - 1)
              xr(i) = zzero
              FF(i) = FF0(ii + 1) * conjg(Uincx) + FF0(ii + 2) * conjg(Uincy) + FF0(ii + 3) * conjg(Uincz)
              xi(i) = -polarisa(i, 1, 1) * FF(i)
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL

           call produitfftmatvectoptscalaire(vectx, vecty, ntotalm, ntotal, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2,&
                & XI, XR, planf, planb)

           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, nbsphere
              xi(i) = FF(i) * exp(xr(i) / FF(i))
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL
        else if (ninitest == 3) then
           if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti' .or. &
                beam(1:15) == 'gparawavelinear' .or. beam(1:17) == 'gparawavecircular') then
              nstop = 1.0_wp
              infostr = 'BPM not possible with this beam'
              return
           end if

           call beampropagation(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, theta, phi, xgaus, ygaus, zgaus, beam, epsilon, niso,&
                & ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, Eimagez, FF0, xr, FF, plan2f, plan2b, nstop,&
                & infostr, merge(2,0,cuda_scalar_guess .and. ninitest == 3))
           if (nstop == 1) return
           if (.not. cuda_scalar_guess) then
              !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i,ii)
              !$OMP DO SCHEDULE(STATIC)
              do i = 1, nbsphere
                 ii = 3 * (i - 1)
                 xi(i) = xr(ii + 1) * conjg(Uincx) + xr(ii + 2) * conjg(Uincy) + xr(ii + 3) * conjg(Uincz)
              end do
              !$OMP ENDDO
              !$OMP END PARALLEL
           end if
        end if

         if (nprecon == 0) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
          call inverserigscalaire_cuda_driver(vectx, vecty, Tabdip, &
                 macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                 ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                 nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                 FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, &
                  ncompte, nlim, nprecon, nrig, cuda_scalar_initial_mode, 1, k0, aretecube, planf, planb, &
                 nstop, infostr)
#else
            call inverserigoptscalaire(vectx, vecty, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm, ntotal,&
                 & ldabi, nlar, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, FF0, FFloc,&
                 & polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, planf, planb, nstop, infostr)
#endif
         else if (nprecon == 1 .or. nprecon == 2) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
            call inverserigscalaire_cuda_driver(vectx, vecty, Tabdip, &
                 macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                 ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                 nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                 FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, &
                 ncompte, nlim, nprecon, nrig, ninitest, 1, k0, aretecube, planf, planb, &
                 nstop, infostr)
#else
            if (nprecon == 1) then
               call inverserigscalaireoptpg(vectx, vecty, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm, ntotal,&
                    & ldabi, nlar, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, FF0, FFloc,&
                    & FFprecon, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, k0, aretecube, localfieldx, sdetnn, planf,&
                    & planb, nrig, nstop, infostr)
            else
               call inverserigscalaireoptpd(vectx, vecty, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm, ntotal,&
                    & ldabi, nlar, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, FF0, FFloc,&
                    & FFprecon, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, k0, aretecube, localfieldx, sdetnn, planf,&
                    & planb, nrig, nstop, infostr)
            endif
#endif
         end if

     end if
     if (nstop == 1) return

     if (tol1 >= tolinit) then
        nstop = 1
        infostr = 'Converge do not reach!'
        write(*, *) 'Converge do not reach!', tol1, 'Tol Asked', tolinit
        return
     end if

     if (nstop == -1) then
        infostr = 'Calculation cancelled after iterative method!'
        return
     end if

     if (ninitest == 4) then
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere * 3
           xi(i) = FFloc(i)
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
        nrig = 0
        nprecon = ntmp
        tolinit = tmp
        goto 333
     end if

  else if (nrig == 9) then
     write(*, *) 'Scalar approximation G=e(ikr)/r'
     if (trope == 'ani') then
        infostr = 'Anisotropy can not be used with scalar approximation'
        nstop = 1
        return
     end if
     if (beam(1:11) /= 'pwavelinear' .and. beam(1:13) /= 'pwavecircular' &
          .and. beam(1:15) /= 'gparawavelinear' .and. beam(1:17) /= 'gparawavecircular' &
          .and. beam(1:14) /= 'gfftwavelinear' .and. beam(1:16) /= 'gfftwavecircular') then
        infostr = 'Scalar appr. for plane wave or paraxial gaussian'
        nstop = 1
        return
     end if
     ! calcule le vecteur unitaire de polarisation incidente
     ! Approximation scalaire
     x = zero
     y = zero
     z = zero

     if (beam(1:11) == 'pwavelinear') then
        call ondeplane(x, y, z, k0, eps0, E0, ss, pp, theta, phi, Uincx, Uincy, Uincz, nstop, infostr)
     else if (beam(1:13) == 'pwavecircular') then
        call ondecirce(x, y, z, k0, E0, ss, theta, phi, Uincx, Uincy, Uincz)
     else if (beam(1:15) == 'gparawavelinear') then
        call gaussianparalinear(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Uincx, Uincy, Uincz, nstop,&
             & infostr)
     else if (beam(1:17) == 'gparawavecircular') then
        call gaussianparacirc(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, eps0, E0, Uincx, Uincy, Uincz, nstop, infostr)
     else if (beam(1:14) == 'gfftwavelinear') then
        call gaussianparalinear(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, pp, eps0, E0, Uincx, Uincy, Uincz, nstop,&
             & infostr)
     else if (beam(1:16) == 'gfftwavecircular') then
        call gaussianparacirc(x, y, z, xgaus, ygaus, zgaus, theta, phi, w0, k0, ss, eps0, E0, Uincx, Uincy, Uincz, nstop, infostr)
     end if

     Emod = sqrt(abs(Uincx)**2.0_wp + abs(Uincy)**2.0_wp + abs(Uincz)**2.0_wp)
     Uincx = Uincx / Emod
     Uincy = Uincy / Emod
     Uincz = Uincz / Emod
     ! calcul la fonction de Green scalaire avec sa FFT
     call greencalculscalairefft2(nx, ny, nz, nx2, ny2, nz2, nxy2, nxm, nym, nzm, aretecube, k0, Uincx, Uincy, Uincz, vecty, planb, ntotalm, nstop, infostr)
     if (nstop == 1) return

     tol = tolinit
     ncompte = 0
     nloop = 0
     cuda_scalar_guess = .false.
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_scalar_guess = (ninitest == 0 .or. ninitest == 1 .or. ninitest == 2 .or. &
          (ninitest == 3 .and. (nrig == 8 .or. nrig == 9))) .and. &
          (nprecon == 0 .or. ((nprecon == 1 .or. nprecon == 2) .and. &
          (nrig == 8 .or. nrig == 9))) .and. &
          methodeit(1:6) /= 'GCORS2' .and. methodeit(1:9) /= 'BORNSERIE' .and. &
          (methodeit(1:7) == 'GPBICG1' .or. methodeit(1:7) == 'GPBICG2' .or. &
          methodeit(1:10) == 'GPBICGplus' .or. methodeit(1:10) == 'GPBICGsafe' .or. &
          methodeit(1:8) == 'GPBICGAR' .or. methodeit(1:9) == 'GPBICGAR2' .or. &
          methodeit(1:7) == 'GPBICOR' .or. methodeit(1:12) == 'BICGSTARPLUS' .or. &
          methodeit(1:11) == 'SSBICGSAFE2' .or. methodeit(1:4) == 'CORS' .or. &
          methodeit(1:12) == 'QMRBICGSTAB1' .or. methodeit(1:12) == 'QMRBICGSTAB2' .or. &
          methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
          methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. &
          methodeit(1:2) == 'CG' .or. methodeit(1:4) == 'IDRS' .or. &
          methodeit(1:6) == 'QMRCLA')
#endif
     if (ninitest == 0 .and. .not. cuda_scalar_guess) then
        !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           xi(i) = zzero
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
     else if (ninitest == 1 .and. .not. cuda_scalar_guess) then
        !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i,ii)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           ii = 3 * (i - 1)
           xi(i) = FF0(ii + 1) * conjg(Uincx) + FF0(ii + 2) * conjg(Uincy) + FF0(ii + 3) * conjg(Uincz)
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL

     else if (ninitest == 2 .and. .not. cuda_scalar_guess) then
        ! Exponential scalar initial guess, as in the nrig=8 branch.
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,ii)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           ii = 3 * (i - 1)
           xr(i) = zzero
           FF(i) = FF0(ii + 1) * conjg(Uincx) + FF0(ii + 2) * conjg(Uincy) + FF0(ii + 3) * conjg(Uincz)
           xi(i) = -polarisa(i, 1, 1) * FF(i)
        enddo
        !$OMP ENDDO
        !$OMP END PARALLEL
        call produitfftmatvectoptscalaire(vecty, vectx, ntotalm, ntotal, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, &
             & XI, XR, planf, planb)
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           if (abs(FF(i)) == zero) then
              xi(i) = zzero
           else
              xi(i) = FF(i) * exp(xr(i) / FF(i))
           endif
        enddo
        !$OMP ENDDO
        !$OMP END PARALLEL

     else if (ninitest == 5) then
        !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere
           xi(i) = zzero
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
        tmp = tolinit
        tolinit = 0.01_wp
        nprecon = 0
     end if
     tol = tolinit
     ncompte = 0
     nloop = 0
     if (nproche == -1) then
        if (nprecon == 0) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           if (methodeit(1:7) == 'GPBICG1' .or. methodeit(1:7) == 'GPBICG2' .or. &
               methodeit(1:10) == 'GPBICGplus' .or. methodeit(1:10) == 'GPBICGsafe' .or. &
               methodeit(1:8) == 'GPBICGAR' .or. methodeit(1:9) == 'GPBICGAR2' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
               methodeit(1:12) == 'BICGSTARPLUS' .or. methodeit(1:11) == 'SSBICGSAFE2' .or. methodeit(1:4) == 'CORS' .or. methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
                methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
                methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. methodeit(1:2) == 'CG' .or. &
                methodeit(1:4) == 'IDRS' .or. methodeit(1:6) == 'QMRCLA') then
              call inverserigscalaire_cuda_driver(vecty, vectx, Tabdip, &
                   macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                   ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                   nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                   FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, &
                   ncompte, nlim, nprecon, nrig, ninitest, 0, k0, aretecube, planf, planb, &
                   nstop, infostr)
           else
              call inverserigscalaire(vecty, vectx, Tabdip, &
                   macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, &
                   Uincz, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, &
                   nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, &
                   wrk, FF, FF0, FFloc, polarisa, niso, methodeit, tol, tol1, &
                   nloop, ncompte, nlim, planf, planb, nstop, infostr)
           endif
#else
           call inverserigscalaire(vecty, vectx, Tabdip, &
                macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, &
                Uincz, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, &
                nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, &
                wrk, FF, FF0, FFloc, polarisa, niso, methodeit, tol, tol1, &
                nloop, ncompte, nlim, planf, planb, nstop, infostr)
#endif
        else if (nprecon == 1) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           if ((nrig == 8 .or. nrig == 9) .and. &
               (methodeit(1:7) == 'GPBICG1' .or. methodeit(1:7) == 'GPBICG2' .or. &
                methodeit(1:10) == 'GPBICGplus' .or. methodeit(1:10) == 'GPBICGsafe' .or. &
                methodeit(1:8) == 'GPBICGAR' .or. methodeit(1:9) == 'GPBICGAR2' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
                methodeit(1:12) == 'BICGSTARPLUS' .or. methodeit(1:11) == 'SSBICGSAFE2' .or. methodeit(1:4) == 'CORS' .or. methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
                methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
               methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. methodeit(1:2) == 'CG' .or. &
                 methodeit(1:4) == 'IDRS' .or. methodeit(1:6) == 'QMRCLA')) then
              call inverserigscalaire_cuda_driver(vecty, vectx, Tabdip, &
                   macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                   ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                   nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                   FF0, FFloc, polarisa, niso, methodeit, tol, tol1, &
                   nloop, ncompte, nlim, nprecon, nrig, ninitest, 0, k0, aretecube, &
                   planf, planb, nstop, infostr)
           else
              call inverserigscalairepg(vecty, vectx, Tabdip, &
                   macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, &
                   Uincz, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, &
                   nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, &
                   wrk, FF, FF0, FFloc, FFprecon, polarisa, niso, &
                   methodeit, tol, tol1, nloop, ncompte, nlim, k0, &
                   aretecube, localfieldx, sdetnn, planf, planb, nrig, nstop, &
                   infostr)
           endif
#else
           call inverserigscalairepg(vecty, vectx, Tabdip, &
                macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, &
                Uincz, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, &
                nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, &
                wrk, FF, FF0, FFloc, FFprecon, polarisa, niso, &
                methodeit, tol, tol1, nloop, ncompte, nlim, k0, &
                aretecube, localfieldx, sdetnn, planf, planb, nrig, nstop, &
                infostr)
#endif

        else if (nprecon == 2) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           if ((nrig == 8 .or. nrig == 9) .and. &
               (methodeit(1:7) == 'GPBICG1' .or. methodeit(1:7) == 'GPBICG2' .or. &
                methodeit(1:10) == 'GPBICGplus' .or. methodeit(1:10) == 'GPBICGsafe' .or. &
                methodeit(1:8) == 'GPBICGAR' .or. methodeit(1:9) == 'GPBICGAR2' .or. &
               methodeit(1:7) == 'GPBICOR' .or. &
                methodeit(1:12) == 'BICGSTARPLUS' .or. methodeit(1:11) == 'SSBICGSAFE2' .or. methodeit(1:4) == 'CORS' .or. methodeit(1:12) == 'QMRBICGSTAB1' .or. &
                methodeit(1:12) == 'QMRBICGSTAB2' .or. &
                methodeit(1:8) == 'BICGSTAB' .or. methodeit(1:9) == 'BICGLSTAB' .or. &
                methodeit(1:11) == 'GPBICGSTABL' .or. methodeit(1:5) == 'TFQMR' .or. methodeit(1:2) == 'CG' .or. &
                 methodeit(1:4) == 'IDRS' .or. methodeit(1:6) == 'QMRCLA')) then
              call inverserigscalaire_cuda_driver(vecty, vectx, Tabdip, &
                   macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                   ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                   nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                   FF0, FFloc, polarisa, niso, methodeit, tol, tol1, &
                   nloop, ncompte, nlim, nprecon, nrig, ninitest, 0, k0, aretecube, &
                   planf, planb, nstop, infostr)
           else
              call inverserigscalairepd(vecty, vectx, Tabdip, &
                   macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, &
                   Uincz, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, &
                   nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, &
                   wrk, FF, FF0, FFloc, FFprecon, polarisa, niso, &
                   methodeit, tol, tol1, nloop, ncompte, nlim, k0, &
                   aretecube, localfieldx, sdetnn, planf, planb, nrig, nstop, &
                   infostr)
           endif
#else
           call inverserigscalairepd(vecty, vectx, Tabdip, &
                macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, &
                Uincz, ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, &
                nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, &
                wrk, FF, FF0, FFloc, FFprecon, polarisa, niso, &
                methodeit, tol, tol1, nloop, ncompte, nlim, k0, &
                aretecube, localfieldx, sdetnn, planf, planb, nrig, nstop, &
                infostr)
#endif
        end if

     else
        if (ninitest == 3) then
           if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti' .or. &
                beam(1:15) == 'gparawavelinear' .or. beam(1:17) == 'gparawavecircular') then
              nstop = 1.0_wp
              infostr = 'BPM not possible with this beam'
              return
           end if

           call beampropagation(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, theta, phi, xgaus, ygaus, zgaus, beam, epsilon, niso,&
                & ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, Eimagez, FF0, xr, FF, plan2f, plan2b, nstop,&
                & infostr, merge(2,0,cuda_scalar_guess .and. ninitest == 3))
           if (nstop == 1) return
           !$OMP PARALLEL DEFAULT(SHARED)  PRIVATE(i,ii)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, nbsphere
              ii = 3 * (i - 1)
              xi(i) = xr(ii + 1) * conjg(Uincx) + xr(ii + 2) * conjg(Uincy) + xr(ii + 3) * conjg(Uincz)
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL
        end if
        if (nprecon == 0) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
            call inverserigscalaire_cuda_driver(vecty, vectx, Tabdip, &
                 macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                 ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                 nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                 FF0, FFloc, polarisa, niso, methodeit, tol, tol1, nloop, &
                 ncompte, nlim, nprecon, nrig, ninitest, 1, k0, aretecube, planf, planb, &
                 nstop, infostr)
#else
           call inverserigoptscalaire(vecty, vectx, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm, ntotal,&
                & ldabi, nlar, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, FF0, FFloc,&
                & polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, planf, planb, nstop, infostr)
#endif
        else if (nprecon == 1) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
            call inverserigscalaire_cuda_driver(vecty, vectx, Tabdip, &
                 macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                 ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                 nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                 FF0, FFloc, polarisa, niso, methodeit, tol, tol1, &
                 nloop, ncompte, nlim, nprecon, nrig, ninitest, 1, k0, aretecube, &
                 planf, planb, nstop, infostr)
#else
           call inverserigscalaireoptpg(vecty, vectx, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm, ntotal,&
                & ldabi, nlar, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, FF0, FFloc,&
                & FFprecon, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, k0, aretecube, localfieldx, sdetnn, planf,&
                & planb, nrig, nstop, infostr)
#endif
        else if (nprecon == 2) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
            call inverserigscalaire_cuda_driver(vecty, vectx, Tabdip, &
                 macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, &
                 ntotalm, ntotal, ldabi, nlar, nmax, ndipole, nxm, nym, nzm, &
                 nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, &
                 FF0, FFloc, polarisa, niso, methodeit, tol, tol1, &
                 nloop, ncompte, nlim, nprecon, nrig, ninitest, 1, k0, aretecube, &
                 planf, planb, nstop, infostr)
#else
           call inverserigscalaireoptpd(vecty, vectx, macroscopicfieldx, macroscopicfieldy, Uincx, Uincy, Uincz, ntotalm, ntotal,&
                & ldabi, nlar, nmax, nxm, nym, nzm, nx, ny, nz, nx2, ny2, nxy2, nz2, nbsphere, XI, XR, wrk, FF, FF0, FFloc,&
                & FFprecon, polarisa, niso, methodeit, tol, tol1, nloop, ncompte, nlim, k0, aretecube, localfieldx, sdetnn, planf,&
                & planb, nrig, nstop, infostr)
#endif
        end if
     end if
     if (nstop == 1) return

     if (.not. ieee_is_finite(tol1)) then
        nstop = 1
        infostr = 'Non-finite scalar solver residual'
        return
     endif

     if (tol1 >= tolinit) then
        nstop = 1
        infostr = 'Converge do not reach!'
        write(*, *) 'Converge do not reach!', tol1, 'Tol Asked', tolinit
        return
     end if

     if (nstop == -1) then
        infostr = 'Calculation cancelled after iterative method!'
        return
     end if

     if (ninitest == 5) then
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
        !$OMP DO SCHEDULE(STATIC)
        do i = 1, nbsphere * 3
           xi(i) = FFloc(i)
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
        nrig = 0
        nprecon = ntmp
        tolinit = tmp
        goto 333
     end if

  else if (nrig == 10) then

     ! multi layer Born
     write(*, *) 'Multi layer Born scalar'

     if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti') then
        nstop = 1.0_wp
        infostr = 'MLB not possible with this beam'
        return
     end if

     call multilayerbornsca(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, theta, phi, xgaus, ygaus, zgaus, beam, epsilon, polarisa,&
          & niso, ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, Eimagez, FF0, FFloc, FF, plan2f, plan2b,&
          & nstop , infostr)

     if (nstop == 1) return
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  else if (nrig == 11) then

     ! multi layer Born
     write(*, *) 'Multi layer Born scalar with reflexion'

     if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti') then
        nstop = 1.0_wp
        infostr = 'MLB not possible with this beam'
        return
     end if

     call multilayerbornscaref(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, &
          theta, phi, xgaus, ygaus, zgaus, beam, epsilon, polarisa, niso, &
          ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, &
          Eimagez, FF0, FFloc, FF, plan2f, plan2b, nstop, infostr)

     if (nstop == 1) return
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  else if (nrig == 12) then

     ! multi layer Born
     write(*, *) 'Multi layer Born inc'

     if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti') then
        nstop = 1.0_wp
        infostr = 'MLB not possible with this beam'
        return
     end if

     call multilayerbornscainc(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, &
          theta, phi, xgaus, ygaus, zgaus, beam, epsilon, polarisa, niso, &
          ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, &
          Eimagez, FF0, FFloc, FF, plan2f, plan2b, nstop, infostr)

     if (nstop == 1) return
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  else if (nrig == 13) then

     ! multi layer Born
     write(*, *) 'Multi layer Born inc'

     if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti') then
        nstop = 1.0_wp
        infostr = 'MLB not possible with this beam'
        return
     end if

     call multilayerbornscaincref(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, &
          theta, phi, xgaus, ygaus, zgaus, beam, epsilon, polarisa, niso, &
          ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, &
          Eimagez, FF0, FFloc, FF, plan2f, plan2b, nstop, infostr)

     if (nstop == 1) return
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  else if (nrig == 14) then

     ! multi layer Born
     write(*, *) 'Multi layer Born vectorial'

     if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti') then
        nstop = 1.0_wp
        infostr = 'MLB not possible with this beam'
        return
     end if

     call multilayerbornvect(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, &
          theta, phi, xgaus, ygaus, zgaus, beam, epsilon, polarisa, niso, &
          ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, &
          Eimagez, FF0, FFloc, FF, plan2f, plan2b, nstop, infostr)

     if (nstop == 1) return
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  else if (nrig == 15) then

     ! multi layer Born
     write(*, *) 'Multi layer Born vectorial with reflexion'

     if (beam(1:7) == 'antenna' .or. beam(1:15) == 'wavelinearmulti') then
        nstop = 1.0_wp
        infostr = 'MLB not possible with this beam'
        return
     end if

     call multilayerbornvectref(xs, ys, zs, aretecube, k0, w0, E0, ss, pp, &
          theta, phi, xgaus, ygaus, zgaus, beam, epsilon, polarisa, niso, &
          ndipole, nx, ny, nz, nxm, nym, nzm, nmax, nfft2d, Eimagex, Eimagey, &
          Eimagez, FF0, FFloc, FF, plan2f, plan2b, nstop, infostr)

     if (nstop == 1) return
     call produitpolavect(polarisa, niso, nmax, nbsphere, FFloc, FF)

  end if

  ! The greentensor mode revisits label 301 for all three dipole
  ! orientations.  Keep the FFT work buffers and plans alive until
  ! the last orientation has been computed.
  if (beam(1:11) /= 'greentensor' .or. testgreen >= 4) then
  ! Destroy FFTW plans before releasing their associated work buffers.
  !detruit les plans 3D
#ifdef USE_FFTW
#ifdef USE_SIMP
  call sfftw_destroy_plan(planf)
  call sfftw_destroy_plan(planb)
#else 
  call dfftw_destroy_plan(planf)
  call dfftw_destroy_plan(planb)
#endif
#endif

  deallocate(vectx)
  deallocate(vecty)
  deallocate(vectz)
  end if

  
  if (nlecture == 1 .and. nlecture1 == 0) then
     file1 = '.lf'
     long = len(trim(filereread))
     long1 = len(trim(file1))
     filereread1 = filereread(1:long) // file1(1:long1)
     open(1000, file=filereread1, status='new', form='unformatted')
     do i = 1, nbsphere3
        write(1000) FF(i), FFloc(i)
     end do
     close(1000)
  end if

  call cpu_time(t2)
  call date_and_time(date, time, zzone, values2)
  messagetemps = ' to solve Ax=b '
  call calculatedate(values2, values, t2, t1, messagetemps, tempsreelmvp)
  write(*, *) 'Method iterative used             : ', methodeit
  write(*, *) 'Tolerance obtained                : ', tol1
  write(*, *) 'Tolerance asked                   : ', tolinit
  write(*, *) 'Number of product Ax needs        : ', ncompte
  write(*, *) 'Number of iterations              : ', nloop
  writecol = '******** END SOLVE LINEAR SYSTEM *********'
  colorw = 'pink'
  call writecolor(writecol, colorw)
  write(*, *) ' '

  
  ! compute the near field with FFT
1000 if (nprochefft >= 1) then
     ! compute the FFT in all the box

     call calculwidefield(nmat, nrig, nx, ny, nz, nx2, ny2, nz2, nbsphere, subunit, nmaxpp, nxmp, nymp, nzmp, nquad, nxm, nym, nzm,&
          & nfft2d, ntotalm , nmax, speckseed, irra, numaperil, aretecube, k0, lambda, theta, phi, pp, ss, P0, w0, xdip, ydip, zdip&
          & , xgaus, ygaus, zgaus, tolinit, E0, E0m, ssm, ppm, thetam, phim, nbinc, xs, ys, zs, xswf, yswf, zswf, xi, xr, FFloc, FF&
          &, FF0, incidentfieldx, incidentfieldy, incidentfieldz, incidentfield, Eimagex, Eimagey, Eimagez, polarisa, niso, epsilon&
          &, beam, infostr, namefileinc, FFTTENSORxx, FFTTENSORxy, FFTTENSORxz, FFTTENSORyy, FFTTENSORyz, FFTTENSORzz, plan2f,&
          & plan2b, group_idnf, nstop)

     if (nlocal == 1) then
        if (nmat == 0) then
           do i = 1, subunit
              ii = 3 * (i - 1)
              localfieldx(i) = xr(ii + 1)
              localfieldy(i) = xr(ii + 2)
              localfieldz(i) = xr(ii + 3)
              localfield(i) = sqrt(real(xr(ii + 1) * conjg(xr(ii + 1)) &
                   + xr(ii + 2) * conjg(xr(ii + 2)) + xr(ii + 3) * conjg(xr(ii + 3)),wp))
              write(140, *) real(xr(ii + 1),wp), aimag(xr(ii + 1))
              write(141, *) real(xr(ii + 2),wp), aimag(xr(ii + 2))
              write(142, *) real(xr(ii + 3),wp), aimag(xr(ii + 3))
              write(143, *) localfield(i)
           end do
        else
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,ii)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, subunit
              ii = 3 * (i - 1)
              localfieldx(i) = xr(ii + 1)
              localfieldy(i) = xr(ii + 2)
              localfieldz(i) = xr(ii + 3)
              localfield(i) = sqrt(real(xr(ii + 1) * conjg(xr(ii + 1)) + xr(ii + 2) * conjg(xr(ii + 2)) + xr(ii + 3) * conjg(xr(ii &
                   &+ 3)),wp))
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL

           if (nmat == 2) then
              dim(1) = subunit
              dim(2) = nmax
              datasetname = 'Local field modulus wf'
              call hdf5write1d(group_idnf, datasetname, localfield, dim)
              datasetname = 'Local field x component real part wf'
              call hdf5write1d(group_idnf, datasetname, real(localfieldx), dim)
              datasetname = 'Local field x component imaginary part wf'
              call hdf5write1d(group_idnf, datasetname, aimag(localfieldx), dim)
              datasetname = 'Local field y component real part wf'
              call hdf5write1d(group_idnf, datasetname, real(localfieldy), dim)
              datasetname = 'Local field y component imaginary part wf'
              call hdf5write1d(group_idnf, datasetname, aimag(localfieldy), dim)
              datasetname = 'Local field z component real part wf'
              call hdf5write1d(group_idnf, datasetname, real(localfieldz), dim)
              datasetname = 'Local field z component imaginary part wf'
              call hdf5write1d(group_idnf, datasetname, aimag(localfieldz), dim)
           end if

        end if
        ! Close Intensity of the local field wide field
        close(140)
        close(141)
        close(142)
        close(143)
     end if
     if (nmacro == 1) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        cuda_nmax = int(nmax,c_int)
        cuda_niso = int(niso,c_int)
        cuda_ndipoles = int(subunit,c_int)
        cuda_aretecube = real(aretecube,kind(cuda_aretecube))
        cuda_k0 = real(k0,kind(cuda_k0))
        cuda_nstop = 0_c_int
        call ifdda_local_macro_batch_cuda(polarisa,xr,macroscopicfieldx, &
             macroscopicfieldy,macroscopicfieldz,macroscopicfield, &
             cuda_nmax,cuda_niso,cuda_ndipoles,cuda_aretecube,cuda_k0, &
             1_c_int,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop = 1
           infostr = 'CUDA local-to-macroscopic field conversion failed'
           return
        end if
        if (nmat == 0) then
           do i = 1, subunit
              write(144, *) real(macroscopicfieldx(i)), aimag(macroscopicfieldx(i))
              write(145, *) real(macroscopicfieldy(i)), aimag(macroscopicfieldy(i))
              write(146, *) real(macroscopicfieldz(i)), aimag(macroscopicfieldz(i))
              write(147, *) macroscopicfield(i)
           end do
        else if (nmat == 2) then
           dim(1) = subunit
           dim(2) = nmax
           datasetname = 'Macroscopic field modulus wf'
           call hdf5write1d(group_idnf, datasetname, macroscopicfield, dim)
           datasetname = 'Macroscopic field x component real part wf'
           call hdf5write1d(group_idnf, datasetname, real(macroscopicfieldx), dim)
           datasetname = 'Macroscopic field x component imaginary part wf'
           call hdf5write1d(group_idnf, datasetname, aimag(macroscopicfieldx), dim)
           datasetname = 'Macroscopic field y component real part wf'
           call hdf5write1d(group_idnf, datasetname, real(macroscopicfieldy), dim)
           datasetname = 'Macroscopic field y component imaginary part wf'
           call hdf5write1d(group_idnf, datasetname, aimag(macroscopicfieldy), dim)
           datasetname = 'Macroscopic field z component real part wf'
           call hdf5write1d(group_idnf, datasetname, real(macroscopicfieldz), dim)
           datasetname = 'Macroscopic field z component imaginary part wf'
           call hdf5write1d(group_idnf, datasetname, aimag(macroscopicfieldz), dim)
        end if
#else
        nsens = 1
        if (nmat == 0) then
           do i = 1, subunit
              ii = 3 * (i - 1)
              Eloc(1) = xr(ii + 1)
              Eloc(2) = xr(ii + 2)
              Eloc(3) = xr(ii + 3)
              call copiani(polarisa, epsani, nmax, niso, i)
              ! pour l'instant faut au niveau du epsilon pour le champ macro
              call local_macro(Eloc, Em, epsani, aretecube, k0, nsens)
              macroscopicfieldx(i) = Em(1)
              macroscopicfieldy(i) = Em(2)
              macroscopicfieldz(i) = Em(3)
              macroscopicfield(i) = sqrt(real(Em(1) * conjg(Em(1)) + Em(2) * conjg(Em(2)) + Em(3) * conjg(Em(3)),wp))
              write(144, *) real(Em(1)), aimag(Em(1))
              write(145, *) real(Em(2)), aimag(Em(2))
              write(146, *) real(Em(3)), aimag(Em(3))
              write(147, *) macroscopicfield(i)
           end do
        else
           nsens = 1
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,ii,jj,Eloc,epsani,Em)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, subunit
              ii = 3 * (i - 1)
              Eloc(1) = xr(ii + 1)
              Eloc(2) = xr(ii + 2)
              Eloc(3) = xr(ii + 3)
              ! pour l'instant faut au niveau du epsilonpour le champ macro
              call copiani(polarisa, epsani, nmax, niso, i)
              call local_macro(Eloc, Em, epsani, aretecube, k0, nsens)
              macroscopicfieldx(i) = Em(1)
              macroscopicfieldy(i) = Em(2)
              macroscopicfieldz(i) = Em(3)
              macroscopicfield(i) = sqrt(real(Em(1) * conjg(Em(1)) + Em(2) * conjg(Em(2)) + Em(3) * conjg(Em(3)),wp))
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL
           if (nmat == 2) then
              dim(1) = subunit
              dim(2) = nmax
              datasetname = 'Macroscopic field modulus wf'
              call hdf5write1d(group_idnf, datasetname, macroscopicfield, dim)
              datasetname = 'Macroscopic field x component real part wf'
              call hdf5write1d(group_idnf, datasetname, real(macroscopicfieldx), dim)
              datasetname = 'Macroscopic field x component imaginary part wf'
              call hdf5write1d(group_idnf, datasetname, aimag(macroscopicfieldx), dim)
              datasetname = 'Macroscopic field y component real part wf'
              call hdf5write1d(group_idnf, datasetname, real(macroscopicfieldy), dim)
              datasetname = 'Macroscopic field y component imaginary part wf'
              call hdf5write1d(group_idnf, datasetname, aimag(macroscopicfieldy), dim)
              datasetname = 'Macroscopic field z component real part wf'
              call hdf5write1d(group_idnf, datasetname, real(macroscopicfieldz), dim)
              datasetname = 'Macroscopic field z component imaginary part wf'
              call hdf5write1d(group_idnf, datasetname, aimag(macroscopicfieldz), dim)
           end if
        end if
#endif
        ! Intensity of the macroscopic field wide field
        close(144)
        close(145)
        close(146)
        close(147)
     end if
     write(*, *) '*************** END LARGE NEAR  FIELD ***********'
     write(*, *) ' '
  else
     ! ***************************************************
     ! fin de condition sur le champ proche large
     ! debut calcul du champ local et macro dans juste l'objet
     if (nlocal == 1) then
        if (nmat == 0) then
           do i = 1, ndipole
              k = tabdip(i)
              if (k /= 0) then
                 ii = 3 * (k - 1)
                 localfieldx(k) = FFloc(ii + 1)
                 localfieldy(k) = FFloc(ii + 2)
                 localfieldz(k) = FFloc(ii + 3)
                 localfield(k) = sqrt(real(FFloc(ii + 1) * conjg(FFloc(ii + 1)) + FFloc(ii + 2) * conjg(FFloc(ii + 2)) + FFloc(ii +&
                      & 3) * conjg(FFloc(ii + 3)),wp))
                 write(40, *) real(FFloc(ii + 1)), aimag(FFloc(ii + 1))
                 write(41, *) real(FFloc(ii + 2)), aimag(FFloc(ii + 2))
                 write(42, *) real(FFloc(ii + 3)), aimag(FFloc(ii + 3))
                 write(43, *) localfield(k)
              else
                 write(40, *) zero, zero
                 write(41, *) zero, zero
                 write(42, *) zero, zero
                 write(43, *) zero
              end if
           end do
        else
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k,ii)
           !$OMP DO SCHEDULE(STATIC)
           do i = 1, ndipole
              k = tabdip(i)
              if (k /= 0) then
                 ii = 3 * (k - 1)
                 localfieldx(k) = FFloc(ii + 1)
                 localfieldy(k) = FFloc(ii + 2)
                 localfieldz(k) = FFloc(ii + 3)
                 localfield(k) = sqrt(real(FFloc(ii + 1) * conjg(FFloc(ii + 1)) + FFloc(ii + 2) * conjg(FFloc(ii + 2)) + FFloc(ii +&
                      & 3) * conjg(FFloc(ii + 3)),wp))
              end if
           end do
           !$OMP ENDDO
           !$OMP END PARALLEL

           if (nmat == 2) then
              !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
              !$OMP DO SCHEDULE(STATIC)
              do i = 1, ndipole
                 k = tabdip(i)
                 if (k /= 0) then
                    wrk(i, 1) = localfieldx(k)
                    wrk(i, 2) = localfieldy(k)
                    wrk(i, 3) = localfieldz(k)
                    wrk(i, 4) = sqrt(real(localfieldx(k) * conjg(localfieldx(k)) + localfieldy(k) * conjg(localfieldy(k)) +&
                         & localfieldz(k) * conjg(localfieldz(k)),wp))
                 else
                    wrk(i, 1) = zzero
                    wrk(i, 2) = zzero
                    wrk(i, 3) = zzero
                    wrk(i, 4) = zzero
                 end if
              end do
              !$OMP ENDDO
              !$OMP END PARALLEL
              dim(1) = ndipole
              dim(2) = nmax * 3
              datasetname = 'Local field modulus'
              call hdf5write1d(group_idnf, datasetname, real(wrk(:, 4)), dim)
              datasetname = 'Local field x component real part'
              call hdf5write1d(group_idnf, datasetname, real(Wrk(:, 1)), dim)
              datasetname = 'Local field x component imaginary part'
              call hdf5write1d(group_idnf, datasetname, aimag(Wrk(:, 1)), dim)
              datasetname = 'Local field y component real part'
              call hdf5write1d(group_idnf, datasetname, real(Wrk(:, 2)), dim)
              datasetname = 'Local field y component imaginary part'
              call hdf5write1d(group_idnf, datasetname, aimag(Wrk(:, 2)), dim)
              datasetname = 'Local field z component real part'
              call hdf5write1d(group_idnf, datasetname, real(Wrk(:, 3)), dim)
              datasetname = 'Local field z component imaginary part'
              call hdf5write1d(group_idnf, datasetname, aimag(Wrk(:, 3)), dim)
           end if
        end if
        ! Close Intensity of the local field
        close(40)
        close(41)
        close(42)
        close(43)
     end if
     ! compute and save the macroscopic field
     if (nmacro == 1) then

        call savemacroscopicfield(nmat, ndipole, nmax, nlar, testgreen, &
             tabdip, aretecube, k0, FFloc, epsilon, niso, &
             macroscopicfield, macroscopicfieldx, macroscopicfieldy, &
             macroscopicfieldz, wrk, group_idnf)
        ! Close Intensity of the macroscopic field

        if (testgreen == 2) goto 301
        if (testgreen == 3) goto 301
        if (testgreen == 4) goto 302
     end if
  end if

  if (nsection == 1 .or. nsectionsca == 1) then
     if (object(1:6) == 'sphere') then
        call CALLBHMIE(one, eps, rayon, lambda, MIECEXT, MIECABS, MIECSCA, GSCA)
     end if
  end if
  if (nabc == 1) then
     write(*,*) 'compute the new area'

     call rerangechamppola(FFloc,epsilon,polarisa,niso,aretecube,k0 ,FF,xs,ys,zs ,xswf,yswf, zswf,tabdip,tabmulti,masquenrbc,nx,ny &
          &,nz,nxm,nym ,nzm ,ndipole,nbsphere,nmax,trope,polarizability,sldr, nproche)

     deallocate(masquenrbc)
     call ondeplane(0.0_wp,0.0_wp,zs(1),k0,eps0,E0,ss,pp,theta,phi, Emx,Emy,Emz,nstop,infostr)

     if (pp == 0) then
        phase=atan2(aimag(Emy),real(Emy))
     elseif (pp == 1) then
        phase=atan2(aimag(Emx),real(Emx))
     else
        write(*,*) 'problem with polarization'
        stop
     endif
     ctmp=2.0_wp*sqrt(eps0)/(one+sqrt(eps0))
     phasetrans=atan2(aimag(ctmp),real(ctmp))
     I0=abs(E0)**2.0_wp

     eps0=zone
     call ondeplane(0.0_wp,0.0_wp,zs(1),k0,eps0,E0,ss,pp ,theta,phi,Emx,Emy,Emz,nstop,infostr)
     if (pp.eq.0) then
        phaseref=atan2(aimag(Emy) ,real(Emy))
     elseif (pp.eq.1) then
        phaseref=atan2(aimag(Emx),real(Emx))
     else
        write(*,*) 'problem with polarization'
        stop
     endif
     ctmp=exp(-icomp*(phaseref-phase-phasetrans))

     !     $OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)   
     !     $OMP DO SCHEDULE(STATIC)
     do i=1,nbsphere
        call ondeplane(xs(i),ys(i),zs(i),k0,eps0,E0,ss,pp ,theta,phi,FF0(3*i-2),FF0(3*i-1),FF0(3*i),nstop ,infostr)
        FF0(3*i-2)=FF0(3*i-2)*ctmp
        FF0(3*i-1)=FF0(3*i-1)*ctmp
        FF0(3*i)=FF0(3*i)*ctmp
     enddo
     !     $OMP ENDDO 
     !     $OMP END PARALLEL         

#ifdef IFDDA_CUDA_INVERSE_DRIVER
     ! PML compaction changes FFloc and FF on the host, and the vacuum
     ! reference above replaces FF0. Do not reuse the solver's GPU aliases.
     call clear_device_result_vector_alias_cuda(FF0)
     call clear_device_result_vector_alias_cuda(FF)
     call clear_device_result_vector_alias_cuda(FFloc)
#endif

     !     calcul erreur relative ente onde plane et champ dans la boite.
     errmod=zero
     errpha=zero
     k=0

     do i=1,nbsphere
        k=k+1

        errmod=errmod+abs(sqrt(abs(FF0(3*i-2))**2.0_wp +abs(FF0(3*i-1))**2.0_wp+abs(FF0(3*i))**2.0_wp) -sqrt(abs(FFloc(3 *i-2)) &
             &**2.0_wp +abs(FFloc(3*i -1))**2.0_wp+abs(FFloc(3 *i))**2.0_wp))
        errpha=errpha+sqrt(abs(FF0(3*i-2)-FFloc(3*i-2)) **2.0_wp +abs(FF0(3*i-1)-FFloc(3*i-1))**2.0_wp +abs(FF0(3  *i) -FFloc(3*i))&
             & **2.0_wp)
     enddo
!     write(*,*)  'erreur module',errmod/real(k,wp)/abs(E0),k,nbsphere
!     write(*,*)  'erreur phasee',errpha/real(k,wp)/abs(E0)

  endif





  if (nsection == 1) then
     ! Compute the extinction cross section and absorbing cross section
     writecol = '*********** COMPUTE CROSS SECTION ********'
     colorw = 'red'
     write(*, *) '******************************************'
     call writecolor(writecol, colorw)
     write(*, *) '******************************************'
     Cext = zero
     Cabs = zero

#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_ndipoles=int(nbsphere,c_int)
     call ifdda_compute_cross_sections_cuda(FF0,FF,FFloc,cuda_ndipoles,k0,I0, &
          Cext,Cabs,Csca,cuda_green_status)
     if (cuda_green_status /= 0) then
        nstop=1
        infostr='CUDA cross section reduction failed'
        return
     endif
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,kk)
     !$OMP DO SCHEDULE(STATIC) REDUCTION(+:Cext,Cabs)
     do i = 1, nbsphere
        kk = 3 * (i - 1)
        Cext = Cext + aimag(conjg(FF0(kk + 1)) * FF(kk + 1) + conjg(FF0(kk + 2)) * FF(kk + 2) + conjg(FF0(kk + 3)) * FF(kk + 3))
        Cabs = Cabs + aimag(FF(kk + 1) * conjg(FFloc(kk + 1)) + FF(kk + 2) * conjg(FFloc(kk + 2)) + FF(kk + 3) * conjg(FFloc(kk +&
             & 3))) - 2.0_wp / 3.0_wp * k03 * real(FF(kk + 1) * conjg(FF(kk + 1)) + FF(kk + 2) * conjg(FF(kk + 2)) + FF(kk + 3) *&
             & conjg(FF(kk + 3)),wp)
     end do
     !$OMP ENDDO
     !$OMP END PARALLEL
     Cext = 4.0_wp * pi * k0 * Cext / I0
     Cabs = 4.0_wp * pi * k0 * Cabs / I0

     ! compute the scattering cross section
     Csca = Cext - Cabs
#endif

     if (Cabs <= Cext * 1.0e-12_wp) then
        Cabs = zero
     end if
     if (nstop == -1) then
        infostr = 'Calculation cancelled after cross section!'
        return
     end if
     write(*, *) 'Cext (DDA)  =  ', Cext, 'm2'
     write(*, *) 'Csca (DDA)  =  ', Csca, 'm2'
     if (Cabs <= Cext * 1.0e-12_wp) then
        Cabs = zero
        write(*, *) 'Cabs = ', Cabs, 'm2'
     else
        write(*, *) 'Cabs = ', Cabs, 'm2'
     end if

     if (object(1:6) == 'sphere') then
        write(*, *) 'Cext (Mie)  =  ', MIECEXT, 'm2'
        write(*, *) 'Csca (Mie)  =  ', MIECSCA, 'm2'
        write(*, *) 'Comparison with Mie s series'
        write(*, *) 'Cext error in %', (Cext - MIECEXT) / MIECEXT * 100.0_wp
        write(*, *) 'Csca error in %', (Csca - MIECSCA) / MIECSCA * 100.0_wp
        ! write(200,*) dreal(eps),(Cext-MIECEXT)/MIECEXT*100.d0
        ! \$           ,MIECEXT
        if (Cabs /= zero) then
           write(*, *) 'Cabs error in %', (Cabs - MIECABS) / MIECABS * 100.0_wp
        end if
        writecol = '********** END CROSS SECTION *************'
        colorw = 'red'
        call writecolor(writecol, colorw)
        write(*, *) ' '
     end if
  end if

  ! compute the scattering cross section by integration in far field
  ! and compute the Poynting vector along the normal in CGS n.P=c/(8
  ! pi) Re(ExH^*)=c/(8 pi)|E|^2

  if (nsectionsca == 1) then
     writecol = '***** COMPUTE Csca g AND POYNTING ********'
     colorw = 'blue'
     write(*, *) '******************************************'
     call writecolor(writecol, colorw)
     write(*, *) '******************************************'

     if (nquickdiffracte == 0) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        write(*, *) 'Direct far-field and Poynting calculation on CUDA'
        if (ntheta <= 10 .or. nphi <= 20) then
           infostr = 'ntheta and nphi too small!'
           nstop = 1
           return
        end if
        cnt = (ntheta + 1) * nphi
        deltatheta = pi / real(ntheta, wp)
        deltaphi = twopi / real(nphi, wp)
        constpoynting = k0 * k0 * k0 * k0 * c / 8.0_wp / pi * quatpieps0
        deltapoyntingx = k0 / real(min(ntheta - 1, nphi - 1) / 2, wp)
        deltapoyntingy = deltapoyntingx
        ll = int(k0 / deltapoyntingx)
        npoynting = 2 * ll + 1
        cuda_ndipoles = int(nbsphere,c_int)
        cuda_nstop = 1
        call ifdda_direct_scattering_cuda(FF,xs,ys,zs,cuda_ndipoles, &
             int(ntheta,c_int),int(nphi,c_int),int(nfft2d,c_int),k0, &
             theta,phi,I0,constpoynting,poyntingfield,thetafield,phifield, &
             poyntingfieldpos,poyntingfieldneg,kxypoynting,gasym,Cscai,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop = 1
           infostr = 'CUDA direct scattering calculation failed'
           return
        end if
        if (nmat == 0) then
           do i=1,cnt
              write(50,*) poyntingfield(i)
              write(51,*) thetafield(i)
              write(52,*) phifield(i)
           end do
           do ii=-ll,ll
              write(55,*) kxypoynting(ii+ll+1)
              do jj=-ll,ll
                 k = (ii+ll+1) + npoynting * (jj+ll)
                 write(53,*) poyntingfieldpos(k)
                 write(54,*) poyntingfieldneg(k)
              end do
           end do
           do jj=-ll,ll
              write(56,*) kxypoynting(jj+ll+1)
           end do
        end if
        if (nmat == 2) then
           dim(1) = cnt
           dim(2) = max((ntheta + 1) * nphi, nfft2d * nfft2d)
           datasetname = 'Poynting'
           call hdf5write1d(group_idff, datasetname, poyntingfield, dim)
           datasetname = 'theta for Poynting'
           call hdf5write1d(group_idff, datasetname, thetafield, dim)
           datasetname = 'phi for Poynting'
           call hdf5write1d(group_idff, datasetname, phifield, dim)
           dim(1) = npoynting
           dim(2) = nfft2d
           datasetname = 'kx Poynting'
           call hdf5write1d(group_idff, datasetname, kxypoynting, dim)
           datasetname = 'ky Poynting'
           call hdf5write1d(group_idff, datasetname, kxypoynting, dim)
           dim(1) = npoynting * npoynting
           dim(2) = nfft2d * nfft2d
           datasetname = 'Poynting positive'
           call hdf5write1d(group_idff, datasetname, poyntingfieldpos, dim)
           datasetname = 'Poynting negative'
           call hdf5write1d(group_idff, datasetname, poyntingfieldneg, dim)
        end if
        close(50)
        close(51)
        close(52)
        close(53)
        close(54)
        close(55)
        close(56)
        if (beam(1:11) /= 'pwavelinear' .and. beam(1:13) /= 'pwavecircular') then
           gasym = zero
           Cscai = zero
        else
           write(*, *) 'Csca =', Cscai, 'm2'
           write(*, *) 'g    =', gasym
           if (object(1:6) == 'sphere') then
              write(*, *) 'Comparison with Mie s series'
              write(*, *) 'Csca error in %', (Cscai - MIECSCA) / MIECSCA * 100.0_wp
              write(*, *) 'g error in %', (gasym - GSCA) / GSCA * 100.0_wp
           end if
        end if
#else
        ! compute the diffracted field in summing the dipole
        write(*, *) 'Slow method with discretization:', ntheta, nphi
        if (ntheta <= 10 .or. nphi <= 20) then
           infostr = 'ntheta and nphi too small!'
           nstop = 1
           return
        end if

        deltatheta = pi / real(ntheta, wp)
        deltaphi = twopi / real(nphi, wp)
        Cscai = zero
        Poyntinginc = zero
        gasym = zero
        cnt = 0
        constpoynting = k0 * k0 * k0 * k0 * c / 8.0_wp / pi * quatpieps0
        deltapoyntingx = k0 / real(min(ntheta - 1, nphi - 1) / 2, wp)
        deltapoyntingy = k0 / real(min(ntheta - 1, nphi - 1) / 2, wp)

        do itheta = 0, ntheta
           thetas = deltatheta * real(itheta, wp)
           do iphi = 0, nphi - 1
              cnt = cnt + 1
              phis = deltaphi * real(iphi, wp)
              normal(1) = sin(thetas) * cos(phis)
              normal(2) = sin(thetas) * sin(phis)
              normal(3) = cos(thetas)
              Emx = zzero
              Emy = zzero
              Emz = zzero
              !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,kk,ctmp,ctmp1)
              !$OMP DO SCHEDULE(STATIC) REDUCTION(+:Emx,Emy,Emz)
              do i = 1, nbsphere
                 kk = 3 * (i - 1)
                 ctmp = exp(-icomp * k0 * (normal(1) * xs(i) + normal(2) * ys(i) + normal(3) * zs(i)))
                 ctmp1 = normal(1) * FF(kk + 1) + normal(2) * FF(kk + 2) + normal(3) * FF(kk + 3)
                 Emx = Emx + ctmp * (FF(kk + 1) - ctmp1 * normal(1))
                 Emy = Emy + ctmp * (FF(kk + 2) - ctmp1 * normal(2))
                 Emz = Emz + ctmp * (FF(kk + 3) - ctmp1 * normal(3))
              end do
              !$OMP ENDDO
              !$OMP END PARALLEL
              Emod = real(Emx * conjg(Emx) + Emy * conjg(Emy) + Emz * conjg(Emz),wp)
              if (nmat == 0) then
                 write(50, *) Emod * k0 * k0 * k0 * k0 * c / 8.0_wp / pi * quatpieps0
                 write(51, *) thetas * 180.0_wp / pi
                 write(52, *) phis * 180.0_wp / pi
              end if
              poyntingfield(cnt) = Emod * constpoynting
              thetafield(cnt) = thetas * 180.0_wp / pi
              phifield(cnt) = phis * 180.0_wp / pi
              Cscai = Cscai + deltaphi * deltatheta * sin(thetas) * Emod
              gasym = gasym + deltaphi * deltatheta * sin(thetas) * Emod &
                   * (normal(1) * sin(theta * pi / 180.0_wp) * cos(phi * pi / 180.0_wp) &
                   + normal(2) * sin(theta * pi / 180.0_wp) * sin(phi * pi / 180.0_wp) &
                   + normal(3) * cos(theta * pi / 180.0_wp))
           end do
        end do

        if (nmat == 2) then
           dim(1) = cnt
           dim(2) = max((ntheta + 1) * nphi, nfft2d * nfft2d)
           datasetname = 'Poynting'
           call hdf5write1d(group_idff, datasetname, poyntingfield, dim)
           datasetname = 'theta for Poynting'
           call hdf5write1d(group_idff, datasetname, thetafield, dim)
           datasetname = 'phi for Poynting'
           call hdf5write1d(group_idff, datasetname, phifield, dim)

        end if

        ll = int(k0 / deltapoyntingx)
        npoynting = 2 * ll + 1

        do ii = -ll, ll
           kxypoynting(ii + ll + 1) = deltapoyntingx * real(ii, wp)
           do jj = -ll, ll
              k = (ii + ll + 1) + npoynting * (jj + ll)
              kx = deltapoyntingx * real(ii, wp)
              ky = deltapoyntingy * real(jj, wp)

              normal(1) = kx / k0
              normal(2) = ky / k0
              if (k0 * k0 - kx * kx - ky * ky > zero) then
                 normal(3) = sqrt(one - normal(1) * normal(1) - normal(2) &
                      * normal(2))
                 Emx = zzero
                 Emy = zzero
                 Emz = zzero
                 Uincx = zzero
                 Uincy = zzero
                 Uincz = zzero
                 !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,kk,ctmp,ctmp1)
                 !$OMP DO SCHEDULE(STATIC) REDUCTION(+:Emx,Emy,Emz,Uincx,Uincy,Uincz)
                 do i = 1, nbsphere
                    kk = 3 * (i - 1)
                    ctmp = exp(-icomp * k0 * (normal(1) * xs(i) + normal(2) * ys(i) + normal(3) * zs(i)))
                    ctmp1 = normal(1) * FF(kk + 1) + normal(2) * FF(kk + 2) + normal(3) * FF(kk + 3)
                    Emx = Emx + ctmp * (FF(kk + 1) - ctmp1 * normal(1))
                    Emy = Emy + ctmp * (FF(kk + 2) - ctmp1 * normal(2))
                    Emz = Emz + ctmp * (FF(kk + 3) - ctmp1 * normal(3))
                    ctmp = exp(-icomp * k0 * (normal(1) * xs(i) + normal(2) * ys(i) - normal(3) * zs(i)))
                    ctmp1 = normal(1) * FF(kk + 1) + normal(2) * FF(kk + 2) - normal(3) * FF(kk + 3)
                    Uincx = Uincx + ctmp * (FF(kk + 1) - ctmp1 * normal(1))
                    Uincy = Uincy + ctmp * (FF(kk + 2) - ctmp1 * normal(2))
                    Uincz = Uincz + ctmp * (FF(kk + 3) + ctmp1 * normal(3))
                 end do
                 !$OMP ENDDO
                 !$OMP END PARALLEL
                 poyntingfieldpos(k) = real(Emx * conjg(Emx) + Emy * conjg(Emy) + Emz * conjg(Emz),wp) * constpoynting
                 poyntingfieldneg(k) = real(Uincx * conjg(Uincx) + Uincy * conjg(Uincy) + Uincz * conjg(Uincz),wp) * constpoynting
              else
                 poyntingfieldpos(k) = zero
                 poyntingfieldneg(k) = zero
              end if
              if (nmat == 0) then
                 if (jj == 0) write(55, *) kx
                 if (ii == 0) write(56, *) ky
                 write(53, *) poyntingfieldpos(k)
                 write(54, *) poyntingfieldneg(k)
              end if

           end do
        end do

        ! save the Poynting vecteur
        close(50)
        close(51)
        close(52)
        close(53)
        close(54)
        close(55)
        close(56)

        if (nmat == 2) then
           dim(1) = 2 * ll + 1
           dim(2) = nfft2d
           datasetname = 'kx Poynting'
           call hdf5write1d(group_idff, datasetname, kxypoynting, dim)
           datasetname = 'ky Poynting'
           call hdf5write1d(group_idff, datasetname, kxypoynting, dim)

           dim(1) = npoynting * npoynting
           dim(2) = nfft2d * nfft2d
           datasetname = 'Poynting positive'
           call hdf5write1d(group_idff, datasetname, poyntingfieldpos, dim)
           datasetname = 'Poynting negative'
           call hdf5write1d(group_idff, datasetname, poyntingfieldneg, dim)
        end if

        gasym = gasym / cscai
        Cscai = Cscai * k03 * k0 / I0

        ! recalcul le poynting pour le mettre Ã  deux dimensions.

        if (beam(1:11) /= 'pwavelinear' .and. beam(1:13) /= 'pwavecircular') then
           gasym = zero
           Cscai = zero
        else
           write(*, *) 'Csca =', Cscai, 'm2'
           write(*, *) 'g    =', gasym
           if (object(1:6) == 'sphere') then
              write(*, *) 'Comparison with Mie s series'
              write(*, *) 'Csca error in %', (Cscai - MIECSCA) / MIECSCA * 100.0_wp
              write(*, *) 'g error in %', (gasym - GSCA) / GSCA * 100.0_wp
           end if
        end if
#endif
     else
        ! compute the diffracted field with FFT
        write(*, *) 'Diffracted field: Quick method with FFT'
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        cuda_quick_nx = int(nx,c_int)
        cuda_quick_ny = int(ny,c_int)
        cuda_quick_nz = int(nz,c_int)
        cuda_quick_nfft = int(nfft2d,c_int)
        cuda_quick_ntheta = int(ntheta,c_int)
        cuda_quick_nphi = int(nphi,c_int)
        constpoynting = k0*k0*k0*k0*c/8.0_wp/pi*quatpieps0
        call ifdda_quick_diffraction_cuda(cuda_quick_nx,cuda_quick_ny, &
             cuda_quick_nz,cuda_quick_nfft,tabfft2,k0,xs,ys,zs,aretecube, &
             FF,cuda_quick_ntheta,cuda_quick_nphi,I0,theta,phi, &
             constpoynting,cuda_quick_imax,deltakx,deltaky, &
             poyntingfieldpos,poyntingfieldneg,kxypoynting,poyntingfield, &
             thetafield,phifield,gasym,Cscai,cuda_quick_status)
        imaxk0 = int(cuda_quick_imax,ip)
        nstop = int(cuda_quick_status,ip)
        if (nstop /= 0) then
           infostr = 'CUDA quick diffraction failed'
           return
        end if
        write(*, *) '[CUDA] quick diffraction FFT, Csca, g and Poynting computed on GPU'
#else
        call diffractefft2d(nx, ny, nz, nxm, nym, nzm, nfft2d, tabfft2, k0, xs, ys, zs, aretecube, Efourierx, Efouriery, Efourierz,&
             & FF, imaxk0, deltakx, deltaky, Ediffkzpos, Ediffkzneg, plan2f, plan2b, nstop, infostr)
#endif
        write(*, *) 'End Quick method with FFT'
        if (nstop == 1) return
        ! put the field in file with the right angles
        write(*, *) 'Number of point in the numerical aperture', 2 * imaxk0 + 1
        deltapoyntingx = deltakx
        deltapoyntingy = deltaky
        constpoynting = k0 * k0 * k0 * k0 * c / 8.0_wp / pi * quatpieps0
        npoynting = 2 * imaxk0 + 1

#ifndef IFDDA_CUDA_INVERSE_DRIVER
        do i = -imaxk0, imaxk0
           kxypoynting(i+imaxk0+1)=dble(i)*deltakx
           do j = -imaxk0, imaxk0
              k=(i+imaxk0+1)+npoynting*(j+imaxk0)
              kx = real(i, wp) * deltakx
              ky = real(j, wp) * deltaky
       
              if (k0*k0-kx*kx-ky*ky >=  zero) then
                 ii = imaxk0 + i + 1
                 jj = imaxk0 + j + 1
                 poyntingfieldpos(k)=(abs(Ediffkzpos(ii,jj,1))**2.0_wp+abs(Ediffkzpos(ii,jj,2))**2.0_wp+abs(Ediffkzpos(ii,jj,3)) &
                      &**2.0_wp) *constpoynting
                 poyntingfieldneg(k)=(abs(Ediffkzneg(ii,jj,1))**2.0_wp+abs(Ediffkzneg(ii,jj,2))**2.0_wp+abs(Ediffkzneg(ii,jj ,3)) &
                      &**2.0_wp) *constpoynting
              else
                 poyntingfieldpos(k)=zero
                 poyntingfieldneg(k)=zero      
              end if
              if (nmat == 0) then
                 if (j == 0) write(55, *) kx
                 if (i == 0) write(56, *) ky
                 write(53, *) poyntingfieldpos(k)
                 write(54, *) poyntingfieldneg(k)
              end if
           end do
        end do

        ! save the Poynting vecteur
        close(53)
        close(54)
        close(55)
        close(56)
        write(*, *) 'rr'
        if (nmat == 2) then

           dim(1) = 2 * imaxk0 + 1
           dim(2) = nfft2d
           datasetname = 'kx Poynting'
           call hdf5write1d(group_idff, datasetname, kxypoynting, dim)
           datasetname = 'ky Poynting'
           call hdf5write1d(group_idff, datasetname, kxypoynting, dim)

           dim(1) = (2 * imaxk0 + 1)**2
           dim(2) = nfft2d * nfft2d
           datasetname = 'Poynting positive'
           call hdf5write1d(group_idff, datasetname, poyntingfieldpos, dim)
           datasetname = 'Poynting negative'
           call hdf5write1d(group_idff, datasetname, poyntingfieldneg, dim)
        end if
        ! save  the Poynting vecteur in kx ky
        close(53)
        close(54)
        close(55)
        close(56)

        call computegcfft2d(imaxk0, deltakx, deltaky, k0, I0, theta, phi, nfft2d, Ediffkzpos, Ediffkzneg, gasym, Cscai)

        if (beam(1:11) /= 'pwavelinear' .and. beam(1:13) /= 'pwavecircular') then
           gasym = zero
           Cscai = zero
        else
           if (object(1:6) == 'sphere') then
              write(*, *) 'Comparison with Mie s series'
              write(*, *) 'Csca error in %', (Cscai - MIECSCA) / MIECSCA * 100.0_wp
              write(*, *) 'g error in %', (gasym - GSCA) / GSCA * 100.0_wp
           end if
        end if
        write(*, *) 'rr'
        deltatheta = pi / real(ntheta, wp)
        deltaphi = 2.0_wp * pi / real(nphi, wp)
        cnt = 0
        ! calcul Poynting par interpolation en thta phi, avec les points
        ! calcules en kx,ky
        do itheta = 0, ntheta
           thetas = deltatheta * real(itheta, wp)
           do iphi = 0, nphi - 1
              cnt = cnt + 1
              phis = deltaphi * real(iphi, wp)
              if (nmat == 0) then
                 write(51, *) thetas * 180.0_wp / pi
                 write(52, *) phis * 180.0_wp / pi
              end if

              thetafield(cnt) = thetas * 180.0_wp / pi
              phifield(cnt) = phis * 180.0_wp / pi

              kx = k0 * sin(thetas) * cos(phis)
              ky = k0 * sin(thetas) * sin(phis)
              do i = -imaxk0, imaxk0
                 if (kx >= real(i, wp) * deltakx) i2 = i
              end do
              do j = -imaxk0, imaxk0
                 if (ky >= real(j, wp) * deltakx) j2 = j
              end do
              ii = imaxk0 + i2 + 1
              jj = imaxk0 + j2 + 1

              test = 0
              if (thetas <= pi / 2.0_wp) then
                 Emod11 = abs(Ediffkzpos(ii, jj, 1))**2 + abs(Ediffkzpos(ii, jj, 2))**2 + abs(Ediffkzpos(ii, jj, 3))**2
                 if (Emod11 == zero) test = test + 1
                 Emod22 = abs(Ediffkzpos(ii + 1, jj + 1, 1))**2 + abs(Ediffkzpos(ii + 1, jj + 1, 2))**2 + abs(Ediffkzpos(ii + 1, jj&
                      & + 1, 3))**2
                 if (Emod22 == zero) test = test + 1
                 Emod21 = abs(Ediffkzpos(ii + 1, jj, 1))**2 + abs(Ediffkzpos(ii + 1, jj, 2))**2 + abs(Ediffkzpos(ii + 1, jj, 3))**2
                 if (Emod21 == zero) test = test + 1
                 Emod12 = abs(Ediffkzpos(ii, jj + 1, 1))**2 + abs(Ediffkzpos(ii, jj + 1, 2))**2 + abs(Ediffkzpos(ii, jj + 1, 3))**2
                 if (Emod12 == zero) test = test + 1
                 if (test == 0) then
                    Emod = (Emod21 - Emod11) * (kx - deltakx * real(i2, wp)) / deltakx &
                         + (Emod12 - Emod11) * (ky - deltakx * real(j2, wp)) / deltakx &
                         + (Emod11 + Emod22 - Emod12 - Emod21) * (kx - deltakx * real(i2, wp)) &
                         / deltakx * (ky - deltakx * real(j2, wp)) / deltakx + Emod11
                 else
                    Emod = (Emod11 + Emod12 + Emod21 + Emod22) / real(4 - test, wp)
                 end if
              else
                 Emod11 = abs(Ediffkzneg(ii, jj, 1))**2 + abs(Ediffkzneg(ii, jj, 2))**2 + abs(Ediffkzneg(ii, jj, 3))**2
                 if (Emod11 == zero) test = test + 1
                 Emod22 = abs(Ediffkzneg(ii + 1, jj + 1, 1))**2 + abs(Ediffkzneg(ii + 1, jj + 1, 2))**2 + abs(Ediffkzneg(ii + 1, jj&
                      & + 1, 3))**2
                 if (Emod22 == zero) test = test + 1
                 Emod21 = abs(Ediffkzneg(ii + 1, jj, 1))**2 + abs(Ediffkzneg(ii + 1, jj, 2))**2 + abs(Ediffkzneg(ii + 1, jj, 3))**2
                 if (Emod21 == zero) test = test + 1
                 Emod12 = abs(Ediffkzneg(ii, jj + 1, 1))**2 + abs(Ediffkzneg(ii, jj + 1, 2))**2 + abs(Ediffkzneg(ii, jj + 1, 3))**2
                 if (Emod12 == zero) test = test + 1
                 if (test == 0) then
                    Emod = (Emod21 - Emod11) * (kx - deltakx * real(i2, wp)) / deltakx &
                         + (Emod12 - Emod11) * (ky - deltakx * real(j2, wp)) / deltakx &
                         + (Emod11 + Emod22 - Emod12 - Emod21) * (kx - deltakx * real(i2, wp)) &
                         / deltakx * (ky - deltakx * real(j2, wp)) / deltakx + Emod11
                 else
                    Emod = (Emod11 + Emod12 + Emod21 + Emod22) / real(4 - test, wp)
                 end if
              end if
              poyntingfield(cnt) = Emod * k0 * k0 * k0 * k0 * c / 8.0_wp / pi * quatpieps0
              if (nmat == 0) write(50, *) Emod * k0 * k0 * k0 * k0 * c / 8.0_wp / pi * quatpieps0

           end do
        end do
        write(*, *) 'rr'
        ! save the Poynting vecteur
        close(50)
        close(51)
        close(52)
        if (nmat == 2) then
           dim(1) = cnt
           dim(2) = max((ntheta + 1) * nphi, nfft2d * nfft2d)
           datasetname = 'Poynting'
           call hdf5write1d(group_idff, datasetname, poyntingfield, dim)
           datasetname = 'theta for Poynting'
           call hdf5write1d(group_idff, datasetname, thetafield, dim)
           datasetname = 'phi for Poynting'
           call hdf5write1d(group_idff, datasetname, phifield, dim)
        end if
#else
        cnt = (ntheta + 1) * nphi
        if (nmat == 0) then
           do i = 1, npoynting
              write(55,*) kxypoynting(i)
              write(56,*) kxypoynting(i)
           end do
           do i = -imaxk0, imaxk0
              do j = -imaxk0, imaxk0
                 k = i + imaxk0 + 1 + npoynting * (j + imaxk0)
                 write(53,*) poyntingfieldpos(k)
                 write(54,*) poyntingfieldneg(k)
              end do
           end do
           do i = 1, cnt
              write(50,*) poyntingfield(i)
              write(51,*) thetafield(i)
              write(52,*) phifield(i)
           end do
        end if
        close(50)
        close(51)
        close(52)
        close(53)
        close(54)
        close(55)
        close(56)

        if (nmat == 2) then
           dim(1) = npoynting
           dim(2) = nfft2d
           datasetname = 'kx Poynting'
           call hdf5write1d(group_idff, datasetname, kxypoynting, dim)
           datasetname = 'ky Poynting'
           call hdf5write1d(group_idff, datasetname, kxypoynting, dim)
           dim(1) = npoynting * npoynting
           dim(2) = nfft2d * nfft2d
           datasetname = 'Poynting positive'
           call hdf5write1d(group_idff, datasetname, poyntingfieldpos, dim)
           datasetname = 'Poynting negative'
           call hdf5write1d(group_idff, datasetname, poyntingfieldneg, dim)
           dim(1) = cnt
           dim(2) = max(cnt, nfft2d * nfft2d)
           datasetname = 'Poynting'
           call hdf5write1d(group_idff, datasetname, poyntingfield, dim)
           datasetname = 'theta for Poynting'
           call hdf5write1d(group_idff, datasetname, thetafield, dim)
           datasetname = 'phi for Poynting'
           call hdf5write1d(group_idff, datasetname, phifield, dim)
        end if
        if (beam(1:11) /= 'pwavelinear' .and. beam(1:13) /= 'pwavecircular') then
           gasym = zero
           Cscai = zero
        else
           write(*,*) 'Csca =',Cscai,'m2'
           write(*,*) 'g    =',gasym
           if (object(1:6) == 'sphere') then
              write(*,*) 'Comparison with Mie s series'
              write(*,*) 'Csca error in %',(Cscai-MIECSCA)/MIECSCA*100.0_wp
              write(*,*) 'g error in %',(gasym-GSCA)/GSCA*100.0_wp
           end if
        end if
#endif
     end if
     writecol = '*** END COMPUTATION Csca g AND POYNTING **'
     colorw = 'blue'
     call writecolor(writecol, colorw)
     write(*, *) ' '
  end if

  if (nforce == 1) then
     writecol = '******** COMPUTE OPTICAL FORCE ***********'
     colorw = 'green'
     write(*, *) '******************************************'
     call writecolor(writecol, colorw)
     write(*, *) '******************************************',nx,ny,nz
     ! Begin the computation of the force
     ! Compute the FFT of the dipole
     nx2=2*nx
     ny2=2*ny
     nz2=2*nz
     ntotal=nx2*ny2*nz2
     nxy2=nx2*ny2

     ! FFTW planning reads the address of vectx.  The original work
     ! buffers were released after solving Ax=b; allocate the force
     ! buffers before creating either plan.
#ifndef IFDDA_CUDA_INVERSE_DRIVER
     allocate(vectx(ntotalm))
     allocate(vecty(ntotalm))
     allocate(vectz(ntotalm))
     
#ifdef USE_SIMP
     call sfftw_plan_dft_3d(planb, nx2, ny2, nz2, vectx, vectx, FFTW_BACKWARD, flagFFTW)
     call sfftw_plan_dft_3d(planf, nx2, ny2, nz2, vectx, vectx, FFTW_FORWARD, flagFFTW)
#else
     call dfftw_plan_dft_3d(planb, nx2, ny2, nz2, vectx, vectx, FFTW_BACKWARD, flagFFTW)
     call dfftw_plan_dft_3d(planf, nx2, ny2, nz2, vectx, vectx, FFTW_FORWARD, flagFFTW)
#endif
#else
     cuda_count=0
     if (nproche == -1) cuda_count=1
     call ifdda_prepare_optical_force_fft_cuda(FF,Tabdip,int(nx,c_int),int(ny,c_int), &
          int(nz,c_int),int(nx2,c_int),int(ny2,c_int),int(nz2,c_int), &
          cuda_count,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop=1
        infostr='CUDA force FFT preparation failed'
        return
     endif
#endif
     ! traite faisceau particulier en premier
     if (beam(1:9) == 'arbitrary') then

        if (nx < 5 .or. ny < 5 .or. nz < 5) then
           infostr = 'Discretization too small for the derivative'
           nstop = 1
           return
        end if
#ifndef IFDDA_CUDA_INVERSE_DRIVER
        call incidentarbitrarydercreate(namefileinc)
#endif
     else if (beam(1:8) == 'gfftwave' .or. beam(1:7) == 'speckle' .or. beam(1:11) == 'demispeckle' .or. beam(1:8) == 'confocal'&
          & .or. beam(1:12) == 'demiconfocal') then

        test = 1

#ifdef IFDDA_CUDA_INVERSE_DRIVER
        call ifdda_derive_incident_field_cuda(FF0,int(nx,c_int),int(ny,c_int), &
             int(nz,c_int),int(nmax,c_int),aretecube,int(test,c_int),xi,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA incident-field derivative failed'
           return
        endif
#else
        call derivechamp(nx, ny, nz, nmax, aretecube, FF0, test, xi)
#endif
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,ii)
        !$OMP DO  SCHEDULE(STATIC)
        do i = 1, ndipole
           ii = 3 * (i - 1)
           wrk(i, 1) = xi(ii + 1)
           wrk(i, 2) = xi(ii + 2)
           wrk(i, 3) = xi(ii + 3)
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL

        test = 2
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        call ifdda_derive_incident_field_cuda(FF0,int(nx,c_int),int(ny,c_int), &
             int(nz,c_int),int(nmax,c_int),aretecube,int(test,c_int),xi,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA incident-field derivative failed'
           return
        endif
#else
        call derivechamp(nx, ny, nz, nmax, aretecube, FF0, test, xi)
#endif
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,ii)
        !$OMP DO  SCHEDULE(STATIC)
        do i = 1, ndipole
           ii = 3 * (i - 1)
           wrk(i, 4) = xi(ii + 1)
           wrk(i, 5) = xi(ii + 2)
           wrk(i, 6) = xi(ii + 3)
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL

        test = 3
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        call ifdda_derive_incident_field_cuda(FF0,int(nx,c_int),int(ny,c_int), &
             int(nz,c_int),int(nmax,c_int),aretecube,int(test,c_int),xi,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA incident-field derivative failed'
           return
        endif
#else
        call derivechamp(nx, ny, nz, nmax, aretecube, FF0, test, xi)
#endif
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,ii)
        !$OMP DO  SCHEDULE(STATIC)
        do i = 1, ndipole
           ii = 3 * (i - 1)
           wrk(i, 7) = xi(ii + 1)
           wrk(i, 8) = xi(ii + 2)
           wrk(i, 9) = xi(ii + 3)
        end do
        !$OMP ENDDO
        !$OMP END PARALLEL
     end if

#ifdef IFDDA_CUDA_INVERSE_DRIVER
     if (beam(1:11) == 'pwavelinear' .or. beam(1:13) == 'pwavecircular') then
        allocate(incident_plane_derivatives(9,ndipole))
        cuda_plane_derivative_circular=0
        if (beam(1:13) == 'pwavecircular') cuda_plane_derivative_circular=1
        call ifdda_plane_wave_derivatives_cuda(xs,ys,zs,Tabdip, &
             int(nxm*nym*nzm,c_int),int(ndipole,c_int),k0,E0,ss,pp,theta,phi, &
             cuda_plane_derivative_circular,incident_plane_derivatives,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA plane-wave derivative batch failed'
           return
        endif
     endif
     if (beam(1:15) == 'wavelinearmulti') then
        allocate(incident_plane_derivatives(9,ndipole))
        call ifdda_multi_plane_wave_derivatives_cuda(xs,ys,zs,Tabdip, &
             int(nxm*nym*nzm,c_int),int(ndipole,c_int),k0,E0m,ppm,thetam,phim, &
             int(nbinc,c_int),incident_plane_derivatives,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA multi-plane derivative batch failed'
           return
        endif
     endif
     if (beam(1:15) == 'gparawavelinear' .or. beam(1:17) == 'gparawavecircular') then
        allocate(incident_plane_derivatives(9,ndipole))
        cuda_plane_derivative_circular=0
        if (beam(1:17) == 'gparawavecircular') cuda_plane_derivative_circular=1
        call ifdda_paraxial_gaussian_derivatives_cuda(xs,ys,zs,Tabdip, &
             int(nxm*nym*nzm,c_int),int(ndipole,c_int),xgaus,ygaus,zgaus, &
             theta,phi,w0,k0,ss,pp,E0,cuda_plane_derivative_circular, &
             incident_plane_derivatives,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA paraxial Gaussian derivative batch failed'
           return
        endif
     endif
     if (beam(1:11) == 'gwavelinear' .or. beam(1:13) == 'gwavecircular') then
        allocate(incident_plane_derivatives(9,ndipole))
        cuda_plane_derivative_circular=0
        if (beam(1:13) == 'gwavecircular') cuda_plane_derivative_circular=1
        call ifdda_rigorous_gaussian_derivatives_cuda(xs,ys,zs,Tabdip, &
             int(nxm*nym*nzm,c_int),int(ndipole,c_int),xgaus,ygaus,zgaus, &
             theta,phi,w0,k0,ss,pp,E0,tol,int(nloin,c_int), &
             cuda_plane_derivative_circular,incident_plane_derivatives,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA rigorous Gaussian derivative batch failed'
           return
        endif
     endif
     if (beam(1:7) == 'antenna') then
        allocate(incident_plane_derivatives(9,ndipole))
        call ifdda_antenna_derivatives_cuda(xs,ys,zs,Tabdip, &
             int(nxm*nym*nzm,c_int),int(ndipole,c_int),xdip,ydip,zdip, &
             theta,phi,aretecube,k0,E0,incident_plane_derivatives,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA antenna derivative batch failed'
           return
        endif
     endif
     if (beam(1:9) == 'arbitrary') then
        allocate(incident_plane_derivatives(9,ndipole))
        call ifdda_arbitrary_derivatives_cuda(xs,ys,zs,Tabdip, &
             int(nxm*nym*nzm,c_int),int(ndipole,c_int),namefileinc, &
             int(len_trim(namefileinc),c_int),incident_plane_derivatives,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA arbitrary derivative batch failed'
           return
        endif
     endif
#endif

#ifndef IFDDA_CUDA_INVERSE_DRIVER
     if (nproche == -1) then
        call calculforcefft(nx, ny, nz, nx2, ny2, nz2, nxy2, nxm, nym, nzm, tabdip, vectx, vecty, vectz, FF, planb)
     else
        call calculforcefftopt(nx, ny, nz, nx2, ny2, nz2, nxy2, nxm, nym, nzm, vectx, vecty, vectz, FF, planb)
     end if
#endif

     ! Compute the x derivative of the local field
     test = 1

#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_compute_force_green_derivative_cuda(aretecube,k0,int(nx,c_int), &
          int(ny,c_int),int(nz,c_int),int(nx2,c_int),int(ny2,c_int), &
          int(nz2,c_int),int(nxy2,c_int),int(test,c_int),FF0,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop=1
        infostr='CUDA force Green derivative failed'
        return
     endif
#else
     call derivativefield2(aretecube, k0, nx, ny, nz, nx2, ny2, nz2, nxy2, ntotal, ntotalm, nmax, vectx, vecty, vectz, test, FF0,&
          & planb, planf)
#endif

     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,ii,j,Eder)
     !$OMP DO  SCHEDULE(STATIC)
     do i=1,ndipole
        ii=3*i
        if (Tabdip(i).ne.0) then
           j=Tabdip(i)
           if (beam(1:11) == 'pwavelinear' .or. beam(1:13) == 'pwavecircular' .or. &
                beam(1:15) == 'wavelinearmulti') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              if (beam(1:15) == 'wavelinearmulti') then
                 call ondeplanedmulti(xs(j),ys(j),zs(j),k0,E0m,ssm,ppm,thetam,phim,nbinc,test,Eder,nstop,infostr)
              elseif (beam(1:11) == 'pwavelinear') then
                 call ondeplaned(xs(j),ys(j),zs(j),k0,E0,ss,pp,theta,phi,test,Eder)
              else
                 call ondecirced(xs(j),ys(j),zs(j),k0,E0,ss,theta,phi,test,Eder)
              endif
#endif
           elseif (beam(1:15) == 'wavelinearmulti') then
              call ondeplanedmulti(xs(j),ys(j),zs(j),k0,E0m,ssm,ppm,thetam,phim,nbinc,test,Eder,nstop,infostr)
           elseif (beam(1:7) == 'antenna') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
              wrk(j,4)=Eder(1,2)
              wrk(j,5)=Eder(2,2)
              wrk(j,6)=Eder(3,2)
              wrk(j,7)=Eder(1,3)
              wrk(j,8)=Eder(2,3)
              wrk(j,9)=Eder(3,3)
#else
              call dipoleincder(xdip,ydip,zdip,theta,phi,xs(j),ys(j),zs(j),aretecube,k0,E0,Eder,test,nstop,infostr)
#endif
           elseif(beam(1:11) == 'gwavelinear') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              call  gaussianchampd(xs(j),ys(j),zs(j),xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,pp,E0,Eder,tol,nloin)
#endif
              wrk(j,4)=Eder(1,2)
              wrk(j,5)=Eder(2,2)
              wrk(j,6)=Eder(3,2)
              wrk(j,7)=Eder(1,3)
              wrk(j,8)=Eder(2,3)
              wrk(j,9)=Eder(3,3)
           elseif(beam(1:13) == 'gwavecircular') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              call  gaussianchampdcirc(xs(j),ys(j),zs(j),xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,E0,Eder,tol,nloin)
#endif
              wrk(j,4)=Eder(1,2)
              wrk(j,5)=Eder(2,2)
              wrk(j,6)=Eder(3,2)
              wrk(j,7)=Eder(1,3)
              wrk(j,8)=Eder(2,3)
              wrk(j,9)=Eder(3,3)
           elseif(beam(1:15) == 'gparawavelinear') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              call  gaussianparalineard(xs(j),ys(j),zs(j),xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,pp,E0,Eder,nstop,infostr)
#endif
              wrk(j,4)=Eder(1,2)
              wrk(j,5)=Eder(2,2)
              wrk(j,6)=Eder(3,2)
              wrk(j,7)=Eder(1,3)
              wrk(j,8)=Eder(2,3)
              wrk(j,9)=Eder(3,3)
           elseif(beam(1:17) == 'gparawavecircular') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              call  gaussianparacircd(xs(j),ys(j),zs(j),xgaus,ygaus,zgaus,theta,phi,w0,k0,ss,E0,Eder,nstop,infostr)
#endif
              wrk(j,4)=Eder(1,2)
              wrk(j,5)=Eder(2,2)
              wrk(j,6)=Eder(3,2)
              wrk(j,7)=Eder(1,3)
              wrk(j,8)=Eder(2,3)
              wrk(j,9)=Eder(3,3)
           elseif (beam(1:8) == 'gfftwave' .or. beam(1:7)  == 'speckle' .or. beam(1:11) == 'demispeckle' .or. &
                beam(1:8) == 'confocal' .or. beam(1:12) == 'demiconfocal') then
              Eder(1,1)=wrk(j,1)
              Eder(2,1)=wrk(j,2)
              Eder(3,1)=wrk(j,3)
           elseif (beam(1:9) == 'arbitrary') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
              wrk(j,4)=Eder(1,2)
              wrk(j,5)=Eder(2,2)
              wrk(j,6)=Eder(3,2)
              wrk(j,7)=Eder(1,3)
              wrk(j,8)=Eder(2,3)
              wrk(j,9)=Eder(3,3)
#else
              !$omp critical                  
              call incidentarbitraryder1(test,xs(j),ys(j),zs(j),namefileinc,Eder)
              !$omp end critical                  
#endif
           endif
           wrk(j,1)=Eder(1,1)+FF0(ii-2)
           wrk(j,2)=Eder(2,1)+FF0(ii-1)
           wrk(j,3)=Eder(3,1)+FF0(ii)
        endif
     enddo
     !$OMP ENDDO
     !$OMP END PARALLEL

     test=2
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_compute_force_green_derivative_cuda(aretecube,k0,int(nx,c_int), &
          int(ny,c_int),int(nz,c_int),int(nx2,c_int),int(ny2,c_int), &
          int(nz2,c_int),int(nxy2,c_int),int(test,c_int),FF0,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop=1
        infostr='CUDA force Green derivative failed'
        return
     endif
#else
     call derivativefield2(aretecube,k0,nx,ny,nz,nx2,ny2,nz2,nxy2,ntotal,ntotalm,nmax,vectx,vecty,vectz,test,FF0,planb,planf)
#endif

     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,ii,j,Eder)
     !$OMP DO  SCHEDULE(STATIC)
     do i=1,ndipole
        ii=3*i
        if (Tabdip(i).ne.0) then
           j=Tabdip(i)
           if (beam(1:11) == 'pwavelinear' .or. beam(1:13) == 'pwavecircular' .or. &
                beam(1:15) == 'wavelinearmulti') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              if (beam(1:15) == 'wavelinearmulti') then
                 call ondeplanedmulti(xs(j),ys(j),zs(j),k0,E0m,ssm,ppm,thetam,phim,nbinc,test,Eder,nstop,infostr)
              elseif (beam(1:11) == 'pwavelinear') then
                 call ondeplaned(xs(j),ys(j),zs(j),k0,E0,ss,pp,theta,phi,test,Eder)
              else
                 call ondecirced(xs(j),ys(j),zs(j),k0,E0,ss,theta,phi,test,Eder)
              endif
#endif
           elseif (beam(1:15) == 'wavelinearmulti') then
              call ondeplanedmulti(xs(j),ys(j),zs(j),k0,E0m,ssm,ppm,thetam,phim,nbinc,test,Eder,nstop,infostr)
           elseif (beam(1:7) == 'antenna') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              call dipoleincder(xdip,ydip,zdip,theta,phi,xs(j),ys(j),zs(j),aretecube,k0,E0,Eder,test,nstop,infostr)
#endif
           elseif(beam(1:11) == 'gwavelinear') then
              Eder(1,2)=wrk(j,4)
              Eder(2,2)=wrk(j,5)
              Eder(3,2)=wrk(j,6)
           elseif(beam(1:13) == 'gwavecircular') then
              Eder(1,2)=wrk(j,4)
              Eder(2,2)=wrk(j,5)
              Eder(3,2)=wrk(j,6)
           elseif(beam(1:15) == 'gparawavelinear') then
              Eder(1,2)=wrk(j,4)
              Eder(2,2)=wrk(j,5)
              Eder(3,2)=wrk(j,6)
           elseif(beam(1:17) == 'gparawavecircular') then
              Eder(1,2)=wrk(j,4)
              Eder(2,2)=wrk(j,5)
              Eder(3,2)=wrk(j,6)
           elseif (beam(1:8) == 'gfftwave' .or. beam(1:7)  == 'speckle' .or. beam(1:11) == 'demispeckle' .or. &
                beam(1:8) == 'confocal' .or. beam(1:12) == 'demiconfocal') then
              Eder(1,2)=wrk(j,4)
              Eder(2,2)=wrk(j,5)
              Eder(3,2)=wrk(j,6)
           elseif (beam(1:9) == 'arbitrary') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              !$omp critical                 
              call incidentarbitraryder1(test,xs(j),ys(j),zs(j),namefileinc,Eder)
              !$omp end critical                   
#endif
           endif
           j=j+nmax
           wrk(j,1)=Eder(1,2)+FF0(ii-2)
           wrk(j,2)=Eder(2,2)+FF0(ii-1)
           wrk(j,3)=Eder(3,2)+FF0(ii)
        endif
     enddo
     !$OMP ENDDO
     !$OMP END PARALLEL

     !     Compute the z derivative of the local field
     test=3
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     call ifdda_compute_force_green_derivative_cuda(aretecube,k0,int(nx,c_int), &
          int(ny,c_int),int(nz,c_int),int(nx2,c_int),int(ny2,c_int), &
          int(nz2,c_int),int(nxy2,c_int),int(test,c_int),FF0,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop=1
        infostr='CUDA force Green derivative failed'
        return
     endif
#else
     call derivativefield2(aretecube,k0,nx,ny,nz,nx2,ny2,nz2,nxy2,ntotal,ntotalm,nmax,vectx,vecty,vectz,test,FF0,planb,planf)
#endif

     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,ii,j,Eder)
     !$OMP DO  SCHEDULE(STATIC)
     do i=1,ndipole
        ii=3*i
        if (Tabdip(i).ne.0) then
           j=Tabdip(i)
           if (beam(1:11) == 'pwavelinear' .or. beam(1:13) == 'pwavecircular' .or. &
                beam(1:15) == 'wavelinearmulti') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              if (beam(1:15) == 'wavelinearmulti') then
                 call ondeplanedmulti(xs(j),ys(j),zs(j),k0,E0m,ssm,ppm,thetam,phim,nbinc,test,Eder,nstop,infostr)
              elseif (beam(1:11) == 'pwavelinear') then
                 call ondeplaned(xs(j),ys(j),zs(j),k0,E0,ss,pp,theta,phi,test,Eder)
              else
                 call ondecirced(xs(j),ys(j),zs(j),k0,E0,ss,theta,phi,test,Eder)
              endif
#endif
           elseif (beam(1:15) == 'wavelinearmulti') then
              call ondeplanedmulti(xs(j),ys(j),zs(j),k0,E0m,ssm,ppm,thetam,phim,nbinc,test,Eder,nstop,infostr)
           elseif (beam(1:7) == 'antenna') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              call dipoleincder(xdip,ydip,zdip,theta,phi,xs(j),ys(j),zs(j),aretecube,k0,E0,Eder,test,nstop,infostr)
#endif
           elseif(beam(1:11) == 'gwavelinear') then
              Eder(1,3)=wrk(j,7)
              Eder(2,3)=wrk(j,8)
              Eder(3,3)=wrk(j,9)
           elseif(beam(1:13) == 'gwavecircular') then
              Eder(1,3)=wrk(j,7)
              Eder(2,3)=wrk(j,8)
              Eder(3,3)=wrk(j,9)
           elseif(beam(1:15) == 'gparawavelinear') then
              Eder(1,3)=wrk(j,7)
              Eder(2,3)=wrk(j,8)
              Eder(3,3)=wrk(j,9)
           elseif(beam(1:17) == 'gparawavecircular') then
              Eder(1,3)=wrk(j,7)
              Eder(2,3)=wrk(j,8)
              Eder(3,3)=wrk(j,9)
           elseif (beam(1:8) == 'gfftwave' .or. beam(1:7)  == 'speckle' .or. beam(1:11) == 'demispeckle' .or. &
                beam(1:8) == 'confocal' .or. beam(1:12) == 'demiconfocal') then
              Eder(1,3)=wrk(j,7)
              Eder(2,3)=wrk(j,8)
              Eder(3,3)=wrk(j,9)
           elseif (beam(1:9) == 'arbitrary') then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
              Eder=reshape(incident_plane_derivatives(:,i),[3,3])
#else
              !$omp critical                  
              call incidentarbitraryder1(test,xs(j),ys(j),zs(j),namefileinc,Eder)
              !$omp end critical                  
#endif
           endif
           j=j+2*nmax
           wrk(j,1)=Eder(1,3)+FF0(ii-2)
           wrk(j,2)=Eder(2,3)+FF0(ii-1)
           wrk(j,3)=Eder(3,3)+FF0(ii)
        endif
     enddo
     !$OMP ENDDO
     !$OMP END PARALLEL

#ifdef IFDDA_CUDA_INVERSE_DRIVER
     if (allocated(incident_plane_derivatives)) deallocate(incident_plane_derivatives)
#endif

     !     Compute the optical force from the derivative of the local field and the dipole
#ifndef IFDDA_CUDA_INVERSE_DRIVER
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC)
     do i=1,nbsphere3
        FF0(i)=conjg(FF(i))
     enddo
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif
     forcet(1)=zero
     forcet(2)=zero
     forcet(3)=zero

#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_ndipoles=int(nbsphere,c_int)
     cuda_nmax=int(nmax,c_int)
     call ifdda_compute_optical_force_cuda(FF,wrk,cuda_ndipoles,cuda_nmax, &
          quatpieps0,forcex,forcey,forcez,forcet,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop=1
        infostr='CUDA optical force calculation failed'
        return
     end if
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k,j)
     !$OMP DO SCHEDULE(STATIC) REDUCTION(+:forcet)
     do i=1,nbsphere
        k=3*(i-1)
        forcex(i)=0.5_wp*real(FF0(k+1)*wrk(i,1)+FF0(k+2)*wrk(i,2)+FF0(k+3)*wrk(i,3),wp)*quatpieps0
        j=i+nmax
        forcey(i)=0.5_wp*real(FF0(k+1)*wrk(j,1)+FF0(k+2)*wrk(j,2)+FF0(k+3)*wrk(j,3),wp)*quatpieps0
        j=i+2*nmax
        forcez(i)=0.5_wp*real(FF0(k+1)*wrk(j,1)+FF0(k+2)*wrk(j,2)+FF0(k+3)*wrk(j,3),wp)*quatpieps0
        forcet(1)=forcet(1)+forcex(i)
        forcet(2)=forcet(2)+forcey(i)
        forcet(3)=forcet(3)+forcez(i)
     enddo
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif

     if (nstop == -1) then
        infostr = 'Calculation cancelled at the optical force!'
        return
     endif

     !     Save the density of force
     if (nforced == 1) then
        if (nmat == 0) then
           do i=1,ndipole
              k=tabdip(i)
              if (k.ne.0) then
                 write(60,*) forcex(k)
                 write(61,*) forcey(k)
                 write(62,*) forcez(k)
              else
                 write(60,*) zero
                 write(61,*) zero
                 write(62,*) zero
              endif
           enddo
           close(60)
           close(61)
           close(62)
        elseif (nmat == 2) then
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
           !$OMP DO SCHEDULE(STATIC)
           do i=1,ndipole
              k=tabdip(i)
              if (k.ne.0) then
                 wrk(i,1)=forcex(k)
                 wrk(i,2)=forcey(k)
                 wrk(i,3)=forcez(k)
              else
                 wrk(i,1)=zzero
                 wrk(i,2)=zzero
                 wrk(i,3)=zzero
              endif
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
           dim(1)=ndipole
           dim(2)=nmax*3
           datasetname='Optical force x component'
           call hdf5write1d(group_idof,datasetname,real(wrk(:,1)),dim)
           datasetname='Optical force y component'
           call hdf5write1d(group_idof,datasetname,real(wrk(:,2)),dim)
           datasetname='Optical force z component'
           call hdf5write1d(group_idof,datasetname,real(wrk(:,3)),dim)
        endif
     endif

     forcem=sqrt(forcet(1)*forcet(1)+forcet(2)*forcet(2)+forcet(3)*forcet(3))
     write(*,*) 'Optical force x      : ',forcet(1),'N'
     write(*,*) 'Optical force y      : ',forcet(2),'N'
     write(*,*) 'Optical force z      : ',forcet(3),'N'
     write(*,*) 'Modulus of the force : ',forcem,'N'

     if (nsection == 1.or.nsectionsca == 1) then
        if (object(1:6) == 'sphere') then
           forcemie=(MIECEXT-GSCA*MIECSCA)/8.0_wp/pi*I0*quatpieps0
           write(*,*) 'Force with Mie',forcemie
           write(*,*) 'Error with Mie in %',100.0_wp*(forcemie-forcem)/forcemie
        endif
     endif

     !     Calcul sur des forces et couple sur differents objets si presents
     if (numberobjet.gt.1) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        cuda_ndipoles=int(nbsphere,c_int)
        cuda_nstop=1
        call ifdda_sum_optical_force_by_object_cuda(tabmulti,cuda_ndipoles, &
             int(numberobjet,c_int),forcexmulti,forceymulti,forcezmulti,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA force grouping failed'
           return
        end if
#else
        forcexmulti=zero
        forceymulti=zero
        forcezmulti=zero
        do i=1,nbsphere
           is=tabmulti(i)
           forcexmulti(is)=forcexmulti(is)+forcex(i)
           forceymulti(is)=forceymulti(is)+forcey(i)
           forcezmulti(is)=forcezmulti(is)+forcez(i)
        enddo
#endif
     endif
     if (numberobjet.ge.2) then
        do is=1,numberobjet
           write(*,*) 'Optical force for the',is,'object'
           write(*,*) 'optical force x',forcexmulti(is),'N'
           write(*,*) 'optical force y',forceymulti(is),'N'
           write(*,*) 'optical force z',forcezmulti(is),'N'
        enddo
     endif

#ifndef IFDDA_CUDA_INVERSE_DRIVER
#ifdef USE_FFTW
#ifdef USE_SIMP
     call sfftw_destroy_plan(planf)
     call sfftw_destroy_plan(planb)
#else 
     call dfftw_destroy_plan(planf)
     call dfftw_destroy_plan(planb)
#endif
#endif
     deallocate(vectx)
     deallocate(vecty)
     deallocate(vectz)
#endif
     writecol='*********** END OPTICAL FORCE ***********'
     colorw='green'
     call writecolor(writecol,colorw)
     write(*,*) ' '
  endif

  if (ntorque*nforce == 1) then
     writecol='********** COMPUTE OPTICAL TORQUE ********'
     colorw='blue'
     write(*,*) '******************************************'
     call writecolor(writecol,colorw)
     write(*,*) '******************************************'
     !     Compute the optical torque on the particle
     couplet(1)=zero
     couplet(2)=zero
     couplet(3)=zero
     !     Compute the center of gravity of the object
     xg=zero
     yg=zero
     zg=zero
#ifndef IFDDA_CUDA_INVERSE_DRIVER
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
     !$OMP DO SCHEDULE(STATIC) REDUCTION(+:xg,yg,zg)
     do i=1,nbsphere
        xg=xg+xs(i)
        yg=yg+ys(i)
        zg=zg+zs(i)
     enddo
     !$OMP ENDDO
     !$OMP END PARALLEL

     xg=xg/real(nbsphere,wp)
     yg=yg/real(nbsphere,wp)
     zg=zg/real(nbsphere,wp)
#endif
     !     gamma=r x F+1/2 Re(p^* x p / alpha_0)
#ifdef IFDDA_CUDA_INVERSE_DRIVER
     cuda_ndipoles=int(nbsphere,c_int)
     call ifdda_compute_optical_torque_cuda(FFloc,xs,ys,zs, &
          cuda_ndipoles,k03,quatpieps0,cuda_torque_center,torquex,torquey,torquez, &
          couplet,cuda_nstop)
     if (cuda_nstop /= 0) then
        nstop=1
        infostr='CUDA optical torque calculation failed'
        return
     end if
     xg=cuda_torque_center(1)
     yg=cuda_torque_center(2)
     zg=cuda_torque_center(3)
#else
     !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,kk,ii,Em)
     !$OMP DO SCHEDULE(STATIC) REDUCTION(+:couplet)
     do i=1,nbsphere
        kk=3*(i-1)
        do ii=1,3
           Em(ii)=2.0_wp/3.0_wp*icomp*k03*FF(kk+ii)+FFloc(kk+ii)
        enddo
        torquex(i)=0.5_wp*real(FF0(kk+2)*Em(3)-FF0(kk+3)*Em(2),wp)*quatpieps0+((ys(i)-yg)*forcez(i)-(zs(i)-zg)*forcey(i))
        torquey(i)=0.5_wp*real(-FF0(kk+1)*Em(3)+FF0(kk+3)*Em(1),wp)*quatpieps0+((zs(i)-zg)*forcex(i)-(xs(i)-xg)*forcez(i))
        torquez(i)=0.5_wp*real(FF0(kk+1)*Em(2)-FF0(kk+2)*Em(1),wp)*quatpieps0+((xs(i)-xg)*forcey(i)-(ys(i)-yg)*forcex(i))
        couplet(1)=couplet(1)+torquex(i)
        couplet(2)=couplet(2)+torquey(i)
        couplet(3)=couplet(3)+torquez(i)
     enddo
     !$OMP ENDDO
     !$OMP END PARALLEL
#endif
     write(*,*) 'Position of the center of gravity :'
     write(*,*) 'xg=',xg
     write(*,*) 'yg=',yg
     write(*,*) 'zg=',zg

     !     Save the density of torque
     if (nstop == -1) then
        infostr = 'Calculation cancelled at the optical torque!'
        return
     endif
     if (ntorqued == 1) then
        if (nmat == 0) then
           do i=1,ndipole
              k=tabdip(i)
              if (k.ne.0) then
                 write(63,*) torquex(k)
                 write(64,*) torquey(k)
                 write(65,*) torquez(k)
              else
                 write(63,*) zero
                 write(64,*) zero
                 write(65,*) zero
              endif
           enddo
           close(63)
           close(64)
           close(65)
        elseif (nmat == 2) then
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,k)
           !$OMP DO SCHEDULE(STATIC)
           do i=1,ndipole
              k=tabdip(i)
              if (k.ne.0) then
                 wrk(i,1)=torquex(k)
                 wrk(i,2)=torquey(k)
                 wrk(i,3)=torquez(k)
              else
                 wrk(i,1)=zzero
                 wrk(i,2)=zzero
                 wrk(i,3)=zzero
              endif
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
           dim(1)=ndipole
           dim(2)=nmax*3
           datasetname='Optical torque x component'
           call hdf5write1d(group_idof,datasetname,real(wrk(:,1)),dim)
           datasetname='Optical torque y component'
           call hdf5write1d(group_idof,datasetname,real(wrk(:,2)),dim)
           datasetname='Optical torque z component'
           call hdf5write1d(group_idof,datasetname,real(wrk(:,3)),dim)
        endif
     endif

     couplem=sqrt(couplet(1)*couplet(1)+couplet(2)*couplet(2)+couplet(3)*couplet(3))
     write(*,*) 'optical torque x',couplet(1),'N/m'
     write(*,*) 'optical torque y',couplet(2),'N/m'
     write(*,*) 'optical torque z',couplet(3),'N/m'
     write(*,*) 'Modulus of the optical torque',couplem,'N/m'

     if (nsection == 1.or.nsectionsca == 1) then
        if (object(1:6) == 'sphere' .and. beam(1:13) == 'pwavecircular') then
           couplemie=MIECABS/8.0_wp/k0/pi*I0*quatpieps0
           write(*,*) 'Torque with Mie',couplemie,'N/m'
           write(*,*) 'Error with Mie in %',100.0_wp*(couplemie-couplem)/couplemie
        endif
     endif

     !     Calcul sur des forces et couple sur differents objets si presents
     if (numberobjet.gt.1) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        cuda_ndipoles=int(nbsphere,c_int)
        cuda_nstop=1
        call ifdda_sum_optical_torque_by_object_cuda(tabmulti,cuda_ndipoles, &
             int(numberobjet,c_int),torquexmulti,torqueymulti,torquezmulti,cuda_nstop)
        if (cuda_nstop /= 0) then
           nstop=1
           infostr='CUDA torque grouping failed'
           return
        end if
#else
        torquexmulti=zero
        torqueymulti=zero
        torquezmulti=zero
        do i=1,nbsphere
           is=tabmulti(i)
           torquexmulti(is)=torquexmulti(is)+torquex(i)
           torqueymulti(is)=torqueymulti(is)+torquey(i)
           torquezmulti(is)=torquezmulti(is)+torquez(i)
        enddo
#endif
     endif
     if (numberobjet.ge.2) then
        do is=1,numberobjet
           write(*,*) 'Optical torque for the',is,'object'
           write(*,*) 'optical torque x',torquexmulti(is),'N/m'
           write(*,*) 'optical torque y',torqueymulti(is),'N/m'
           write(*,*) 'optical torque z',torquezmulti(is),'N/m'
        enddo
     endif
     writecol='******* END OPTICAL TORQUE ***************'
     colorw='blue'
     call writecolor(writecol,colorw)
     write(*,*) ' '
  endif

  if (nenergie == 1) then
     writecol='******** BEGIN ENERGY CONSERVATION *******'
     colorw='cyan'
     write(*,*) '******************************************'
     call writecolor(writecol,colorw)
     write(*,*) '******************************************'
     if (beam(1:7) == 'speckle' .or. beam(1:11) == 'demispeckle' .or.beam(1:8) == 'confocal' .or. beam(1:12) == 'demiconfocal')&
          & then
        call diffractefft2denergieconfocal(speckseed,numaperil,nx,ny,nz,nxm,nym,nzm,nfft2d,tabfft2,k0,xs,ys,zs,E0,ss,pp,theta&
             &,phi, xgaus,ygaus,zgaus,w0,aretecube,Efourierx,Efouriery,Efourierz,FF,imaxk0,deltakx,deltaky,Ediffkzpos &
             &,Ediffkzneg,beam, efficacite,efficaciteref,efficacitetrans,nsectionsca,nquickdiffracte,plan2f,plan2b,nside,nstop &
             &,infostr)
     else
#ifdef IFDDA_CUDA_INVERSE_DRIVER
        if (beam(1:11) == 'pwavelinear' .or. beam(1:13) == 'pwavecircular') then
           cuda_energy_circular=0_c_int
           if (beam(1:13) == 'pwavecircular') cuda_energy_circular=1_c_int
           call ifdda_energy_conservation_plane_cuda(int(nx,c_int),int(ny,c_int), &
                int(nz,c_int),int(nfft2d,c_int),tabfft2,k0,xs,ys,zs,aretecube, &
                eps0,E0,ss,pp,theta,phi,cuda_energy_circular,FF, &
                cuda_energy_imax,deltakx,deltaky,efficacite,efficaciteref, &
                efficacitetrans,energy_flux_incident,energy_flux_reflected, &
                energy_flux_transmitted,energy_outside_percent,cuda_energy_status)
           write(*,*) 'Incident flux         :',energy_flux_incident,'W'
           write(*,*) 'Reflected flux        :',energy_flux_reflected,'W'
           write(*,*) 'Transmitted flux      :',energy_flux_transmitted,'W'
           write(*,*) 'Total flux            :',energy_flux_reflected+energy_flux_transmitted,'W'
           write(*,*) 'Energy outside the NA (%):',energy_outside_percent
           if (cuda_energy_status == 2_c_int) then
              nstop=1
              infostr='Energy: Beam too inclined or waist or nfft too small'
              return
           elseif (cuda_energy_status /= 0_c_int) then
              nstop=1
              infostr='CUDA plane-wave energy flux failed'
              return
           endif
           imaxk0=int(cuda_energy_imax,ip)
           write(*,*) '[CUDA] energy flux, incident FFT and scattered FFT computed on GPU'
        else
           call diffractefft2denergie(nx,ny,nz,nxm,nym,nzm,nfft2d,tabfft2,k0,xs,ys,zs,E0,ss,pp,theta,phi,thetam,phim,ppm,ssm,E0m,&
                & nbinc,xdip,ydip,zdip,xgaus,ygaus,zgaus,w0,aretecube,tol,Efourierx,Efouriery,Efourierz,FF,imaxk0,deltakx,deltaky,&
                & Ediffkzpos,Ediffkzneg,beam,efficacite,efficaciteref,efficacitetrans,nsectionsca,nquickdiffracte,plan2f,plan2b&
                &,nstop,infostr)
        endif
#else
        call diffractefft2denergie(nx,ny,nz,nxm,nym,nzm,nfft2d,tabfft2,k0,xs,ys,zs,E0,ss,pp,theta,phi,thetam,phim,ppm,ssm,E0m,&
             & nbinc,xdip,ydip,zdip,xgaus,ygaus,zgaus,w0,aretecube,tol,Efourierx,Efouriery,Efourierz,FF,imaxk0,deltakx,deltaky,&
             & Ediffkzpos,Ediffkzneg,beam,efficacite,efficaciteref,efficacitetrans,nsectionsca,nquickdiffracte,plan2f,plan2b&
             &,nstop,infostr)
#endif
     endif
     if (nstop == 1) return
     write(*,*) 'Conservation of energy :',efficacite
     write(*,*) 'Absorptivity           :',one-efficacite
     write(*,*) 'Reflextivity           :',efficaciteref
     write(*,*) 'Transmittivity         :',efficacitetrans
     writecol='******** END ENERGY CONSERVATION *********'
     colorw='cyan'
     call writecolor(writecol,colorw)
     write(*,*) ' '
  endif

  if (nlentille == 1) then
     writecol='********** BEGIN MICROSCOPY **************'
     colorw='pink'
     write(*,*) '******************************************'
     call writecolor(writecol,colorw)
     write(*,*) '******************************************'
     write(*,*) 'Microscopy with NA    :',numaper
     write(*,*) 'Magnifying factor     :',gross
     write(*,*) 'Type of Microscopy    :',ntypemic,'(0 holography)'
     write(*,*) 'Focal plane position  :',zlens*real(nside,wp),'nm'

     write(*,*) 'nstop',nstop
     
     nfft2d2=nfft2d/2
     numaper=numaper*k0

  

     
     !     Refait calcul de la FFT de Green si option champproche choisie et microscope non holographique

     if (ntypemic.ne.0) then
        write(*,*) 'recompute the Green function'
        nx2=2*nx
        ny2=2*ny
        nz2=2*nz     
        nxy2=nx2*ny2
        allocate(vectx(nx2*ny2*nz2))
        allocate(vecty(nx2*ny2*nz2))
        allocate(vectz(nx2*ny2*nz2))
#ifndef IFDDA_CUDA_INVERSE_DRIVER
#ifdef USE_SIMP
        call sfftw_plan_dft_3d(planb, nx2, ny2, nz2, vectx, vectx, FFTW_BACKWARD, flagFFTW)
        call sfftw_plan_dft_3d(planf, nx2, ny2, nz2, vectx, vectx, FFTW_FORWARD, flagFFTW)
#else
        call dfftw_plan_dft_3d(planb, nx2, ny2, nz2, vectx, vectx, FFTW_BACKWARD, flagFFTW)
        call dfftw_plan_dft_3d(planf, nx2, ny2, nz2, vectx, vectx, FFTW_FORWARD, flagFFTW)
#endif        
#endif
        if (nquad == 0) then
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           call ifdda_generate_vector_green_cuda(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
                FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,nx,ny,nz,nxm,nym,nzm,nx2,ny2,nz2, &
                aretecube,k0,eps0,int(nabc,c_int),0_c_int,0_c_int,cuda_green_status)
           if (cuda_green_status /= 0) then
              nstop=1
              infostr='CUDA vector Green tensor generation failed'
              return
           end if
#else
           call greencalculfft(nx,ny,nz,nx2,ny2,nz2,nxy2,nxm,nym,nzm,aretecube,k0,FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, FFTTENSORyy &
                &,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz,planb)
#endif
        else
#ifdef IFDDA_CUDA_INVERSE_DRIVER
           cuda_green_filtered=0
           call ifdda_generate_vector_green_cuda(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz, &
                FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,nx,ny,nz,nxm,nym,nzm,nx2,ny2,nz2, &
                aretecube,k0,eps0,int(nabc,c_int),nquad,cuda_green_filtered,cuda_green_status)
           if (cuda_green_status /= 0) then
              nstop=1
              infostr='CUDA quadrature Green tensor generation failed'
              return
           end if
#else
           call greencalculquadfft(nx,ny,nz,nx2,ny2,nz2,nxy2,nxm,nym,nzm,nquad,tolinit,aretecube,k0,FFTTENSORxx,FFTTENSORxy,&
                & FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz,planb)
#endif
        endif
     endif
        !     Recalcul la pola
        if (nprochefft == 1.and.ntypemic.ne.0) then       
        dddis=1
        inv=1
        if (trope == 'iso') then
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,eps,ctmp,ii,jj)
           !$OMP DO SCHEDULE(STATIC)
           do i=1,ndipole
              eps=epsilon(i,1,1)
              call poladiff(aretecube,eps,eps0,k0,dddis,polarizability,sldr,ctmp)
              do ii=1,3
                 do jj=1,3
                    polarisa(i,ii,jj)=zzero
                 enddo
                 polarisa(i,ii,ii)=ctmp
              enddo
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
        else
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,epsani,ii,jj,Eder)
           !$OMP DO SCHEDULE(STATIC)
           do i=1,ndipole
              do ii=1,3
                 do jj=1,3
                    epsani(ii,jj)=epsilon(i,ii,jj)
                 enddo
              enddo
              call polaepstens(aretecube,epsani,eps0,k0,dddis,polarizability,inv,Eder)
              do ii=1,3
                 do jj=1,3
                    polarisa(i,ii,jj)=Eder(ii,jj)
                 enddo
              enddo
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
        endif
     endif

     if (ntypemic == 1 .or. ntypemic == 2 .or. ntypemic == 3 .or. ntypemic == 5) then
        call micromasque(masque,nmasque,imaxinc,imaxk0,nbitermic,ntypemic,nfft2d,deltakinc,deltakx,numaper,numaperinc, &
             numaperinc2,kcnax,kcnay,aretecube,k0,xs,ys,nxm,nym,nzm,nbsphere,nquicklens,nstop,infostr)
        if (nstop == 1) return
        write(*,*) 'side',nside
        if (nside == 1) then
           write(*,*) 'side',nside
           call microsbf(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz,ntotalm, &
                ntotal,ldabi,nlar,nmax,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,ndipole,nbsphere,nbsphere3,nproche,nrig,nfft2d, &
                tabfft2,tabdip,XI,XR,wrk,FF,FF0,FFloc,polarisa,niso,epsilon,methodeit,trope,tolinit,tol1,nloop,ncompte,nlim, &
                xs,ys,zs,aretecube,numaper,numaperinc,numaperinc2,npolainc,nquicklens,eps0,k0,P0,irra,w0,gross,zlens,Eimagex, &
                Eimagey,Eimagez,Efourierx,Efouriery,Efourierz,Eimageincx,Eimageincy,Eimageincz,Efourierincx,Efourierincy, &
                Efourierincz,Ediffkzpos,Ediffkzneg,kxy,xy,nside,masque,nmasque,imaxinc,imaxk0,nbitermic,deltakinc,deltakx, &
                ntypemic,planXf,planXb,planYf,planYb,planZf,planZb,planf,planb,plan2f,plan2b,nmat,file_id,group_idmic,nstop&
                &,infostr)
        else
           call microsbfneg(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz,ntotalm, &
                ntotal,ldabi,nlar,nmax,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,ndipole,nbsphere,nbsphere3,nproche,nrig,nfft2d, &
                tabfft2,tabdip,XI,XR,wrk,FF,FF0,FFloc,polarisa,niso,epsilon,methodeit,trope,tolinit,tol1,nloop,ncompte,nlim, &
                xs,ys,zs,aretecube,numaper,numaperinc,numaperinc2,npolainc,nquicklens,eps0,k0,P0,irra,w0,gross,zlens,Eimagex, &
                Eimagey,Eimagez,Efourierx,Efouriery,Efourierz,Ediffkzpos,Ediffkzneg,kxy,xy,nside,masque,nmasque,imaxinc, &
                imaxk0,nbitermic,deltakinc,deltakx,ntypemic,planXf,planXb,planYf,planYb,planZf,planZb,planf,planb,plan2f,plan2b&
                &, nmat,file_id,group_idmic,nstop,infostr)
        endif
        if (nstop == 1) return
     elseif (ntypemic == 4) then
        if (nside == 1) then
           call microsdf(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz,ntotalm, &
                ntotal,ldabi,nlar,nmax,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,ndipole,nbsphere,nbsphere3,nproche,nrig,nfft2d, &
                tabfft2,tabdip,XI,XR,wrk,FF,FF0,FFloc,polarisa,niso,epsilon,methodeit,trope,tolinit,tol1,nloop,ncompte,nlim, &
                xs,ys,zs,aretecube,numaper,numaperinc,npolainc,nquicklens,eps0,k0,P0,irra,w0,gross,zlens,Eimagex,Eimagey, &
                Eimagez,Efourierx,Efouriery,Efourierz,Eimageincx,Eimageincy,Eimageincz,Efourierincx,Efourierincy,Efourierincz, &
                Ediffkzpos,Ediffkzneg,kxy,xy,nside,planXf,planXb,planYf,planYb,planZf,planZb,planf,planb,plan2f,plan2b,nmat, &
                file_id,group_idmic,nstop,infostr)
        else
           call microsdfneg(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz,ntotalm, &
                ntotal,ldabi,nlar,nmax,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,ndipole,nbsphere,nbsphere3,nproche,nrig,nfft2d, &
                tabfft2,tabdip,XI,XR,wrk,FF,FF0,FFloc,polarisa,niso,epsilon,methodeit,trope,tolinit,tol1,nloop,ncompte,nlim, &
                xs,ys,zs,aretecube,numaper,numaperinc,npolainc,nquicklens,eps0,k0,P0,irra,w0,gross,zlens,Eimagex,Eimagey, &
                Eimagez,Efourierx,Efouriery,Efourierz,Ediffkzpos,Ediffkzneg,kxy,xy,nside,planXf,planXb,planYf,planYb,planZf, &
                planZb,planf,planb,plan2f,plan2b,nmat,file_id,group_idmic,nstop,infostr)
        endif
        if (nstop == 1) return
     elseif (ntypemic == 6) then
        call micromasque(masque,nmasque,imaxinc,imaxk0,nbitermic,ntypemic,nfft2d,deltakinc,deltakx,numaper,numaperinc, &
             numaperinc2,kcnax,kcnay,aretecube,k0,xs,ys,nxm,nym,nzm,nbsphere,nquicklens,nstop,infostr)
        if (nstop == 1) return
        call microsconfocal(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz,ntotalm, &
             ntotal,ldabi,nlar,nmax,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,ndipole,nbsphere,nbsphere3,nproche,nrig,nfft2d, &
             tabfft2,tabdip,XI,XR,wrk,FF,FF0,FFloc,polarisa,niso,epsilon,methodeit,trope,tolinit,tol1,nloop,ncompte,nlim, &
             xs,ys,zs,aretecube,numaper,numaperinc,numaperinc2,psiinc,nquicklens,eps0,k0,P0,irra,w0,gross,zlens,kcnax,kcnay, &
             Eimagex,Eimagey,Eimagez,Efourierx,Efouriery,Efourierz,Eimageincx,Eimageincy,Eimageincz,Efourierincx,Efourierincy, &
             Efourierincz,Ediffkzpos,Ediffkzneg,kxy,xy,nside,masque,nmasque,imaxinc,imaxk0,nbitermic,deltakinc,deltakx,ntypemic&
             &, planXf,planXb,planYf,planYb,planZf,planZb,planf,planb,plan2f,plan2b,nmat,file_id,group_idmic,nstop,infostr)
     elseif (ntypemic == 7) then
        call micromasque(masque,nmasque,imaxinc,imaxk0,nbitermic,ntypemic,nfft2d,deltakinc,deltakx,numaper,numaperinc, &
             numaperinc2,kcnax,kcnay,aretecube,k0,xs,ys,nxm,nym,nzm,nbsphere,nquicklens,nstop,infostr)
        if (nstop == 1) return
        call microsconfocalfluo(FFTTENSORxx,FFTTENSORxy,FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,FFTTENSORzz,vectx,vecty,vectz, &
             ntotalm,ntotal,ldabi,nlar,nmax,nxm,nym,nzm,nx,ny,nz,nx2,ny2,nxy2,nz2,ndipole,nbsphere,nbsphere3,nproche,nrig, &
             nfft2d,tabfft2,tabdip,XI,XR,wrk,FF,FF0,FFloc,polarisa,niso,epsilon,methodeit,trope,tolinit,tol1,nloop,ncompte, &
             nlim,xs,ys,zs,aretecube,numaper,numaperinc,numaperinc2,psiinc,nquicklens,eps0,k0,P0,irra,w0,gross,zlens,kcnax, &
             kcnay,Eimagex,Eimagey,Eimagez,Efourierx,Efouriery,Efourierz,Eimageincx,Eimageincy,Eimageincz,Efourierincx, &
             Efourierincy,Efourierincz,Ediffkzpos,Ediffkzneg,kxy,xy,nside,masque,nmasque,imaxinc,imaxk0,nbitermic,deltakinc, &
             deltakx,ntypemic,planXf,planXb,planYf,planYb,planZf,planZb,planf,planb,plan2f,plan2b,nmat,file_id,group_idmic, &
             nstop,infostr)
     else
        if (nquicklens == 0) then
           write(*,*) 'Slow method for holographic microscope'
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i=1,nfft2d*nfft2d
              Efourierx(i)=zzero
              Efouriery(i)=zzero
              Efourierz(i)=zzero
              Eimageincx(i)=zzero
              Eimageincy(i)=zzero
              Eimageincz(i)=zzero
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
           write(*,*) 'Step size delta k : ',2.0_wp*pi/(real(nfft2d,wp)*aretecube),'m-1'
           k=0
222        deltakx=2.0_wp*pi/(real(nfft2d,wp)*aretecube)/real(2**k,wp)
           imaxk0=nint(k0/deltakx)+1
           if (imaxk0.le.20) then
              k=k+1
              write(*,*) 'change delta k :',deltakx,'m-1',k
              goto 222
           endif
           write(*,*) 'Final delta k',deltakx,'m-1'
           deltaky=deltakx
           deltax=2.0_wp*pi/real(nfft2d,wp)/deltakx
           !     Appelle routine pour calculer le champ incident.
           if (nside == 1) then
              call  diffractefft2dinc(speckseed,nfft2d,k0,E0,ss,pp,theta,phi,thetam,phim,ppm,ssm,E0m,nbinc,xdip,ydip,zdip, &
                   xgaus,ygaus,zgaus,numaperil,w0,aretecube,tol,Efourierx,Efouriery,Efourierz,imaxk0,deltakx,deltaky, &
                   Ediffkzneg,numaper,beam,plan2f,plan2b,nstop,infostr)
              if (nstop == 1) return
           elseif (nside == -1) then
              call  diffractefft2dinckzneg(speckseed,nfft2d,k0,E0,ss,pp,theta,phi,thetam,phim,ppm,ssm,E0m,nbinc,xdip,ydip, &
                   zdip,xgaus,ygaus,zgaus,numaperil,w0,aretecube,tol,Efourierx,Efouriery,Efourierz,imaxk0,deltakx,deltaky, &
                   Ediffkzneg,numaper,beam,plan2f,plan2b,nstop,infostr)
              if (nstop == 1) return
           endif
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i=1,nfft2d*nfft2d
              Efourierx(i)=zzero
              Efouriery(i)=zzero
              Efourierz(i)=zzero
              Efourierincx(i)=zzero
              Efourierincy(i)=zzero
              Efourierincz(i)=zzero
              Eimageincx(i)=zzero
              Eimageincy(i)=zzero
              Eimageincz(i)=zzero
              Eimagex(i)=zzero
              Eimagey(i)=zzero
              Eimagez(i)=zzero
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
           do i=-imaxk0,imaxk0
              if (i.ge.0) then
                 indicex=i+1
              else
                 indicex=nfft2d+i+1
              endif
              kx=deltakx*real(i,wp)
              do j=-imaxk0,imaxk0
                 if (j.ge.0) then
                    indicey=j+1
                 else
                    indicey=nfft2d+j+1
                 endif
                 ky=deltaky*real(j,wp)
                 kz=one
                 indice=indicex+nfft2d*(indicey-1)
                 if (sqrt(kx*kx+ky*ky).le.numaper) then
                    kz=sqrt(k0*k0-kx*kx-ky*ky)
                    zfocus=exp(icomp*kz*zlens)
                    !     Compute theta and phi assuming z axis of the lens
                    normal(1)=kx/k0
                    normal(2)=ky/k0
                    normal(3)=sqrt(one-normal(1)*normal(1)-normal(2)*normal(2))*real(nside,wp)
                    Emx=zzero
                    Emy=zzero
                    Emz=zzero
                    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ii,kk,ctmp,ctmp1)
                    !$OMP DO SCHEDULE(STATIC) REDUCTION(+:Emx,Emy,Emz)
                    do ii=1,nbsphere
                       kk=3*(ii-1)
                       ctmp=exp(-icomp*k0*(normal(1)*xs(ii)+normal(2)*ys(ii)+normal(3)*zs(ii)))
                       ctmp1=normal(1)*FF(kk+1)+normal(2)*FF(kk+2)+normal(3)*FF(kk+3)
                       Emx=Emx+ctmp*(FF(kk+1)-ctmp1*normal(1))
                       Emy=Emy+ctmp*(FF(kk+2)-ctmp1*normal(2))
                       Emz=Emz+ctmp*(FF(kk+3)-ctmp1*normal(3))
                    enddo
                    !$OMP ENDDO
                    !$OMP END PARALLEL
                    ctmp=-2.0_wp*pi*icomp*kz
                    kk=i+nfft2d2+1+nfft2d*(j+nfft2d2)
                    ii=imaxk0+i+1
                    jj=imaxk0+j+1
                    Efourierx(kk)=Emx*k0*k0/ctmp*zfocus
                    Efouriery(kk)=Emy*k0*k0/ctmp*zfocus
                    Efourierz(kk)=Emz*k0*k0/ctmp*zfocus
                    Efourierincx(kk)=Efourierx(kk)+Ediffkzneg(ii,jj,1)*zfocus
                    Efourierincy(kk)=Efouriery(kk)+Ediffkzneg(ii,jj,2)*zfocus
                    Efourierincz(kk)=Efourierz(kk)+Ediffkzneg(ii,jj,3)*zfocus
                    !     Calcul de l'axe de rotation et effectue la rotation
                    u(1)=kx/k0
                    u(2)=ky/k0
                    u(3)=sqrt(one-u(1)*u(1)-u(2)*u(2))*real(nside,wp)
                    v(1)=-kx/k0/gross
                    v(2)=-ky/k0/gross
                    v(3)=sqrt(one-v(1)*v(1)-v(2)*v(2))*real(nside,wp)
                    u1=u(2)*v(3)-u(3)*v(2)
                    u2=-u(1)*v(3)+u(3)*v(1)
                    costmp=u(1)*v(1)+u(2)*v(2)+u(3)*v(3)
                    sintmp=sqrt(u1*u1+u2*u2)
                    if (sintmp == zero) then
                       Eimagex(indice)=Efourierx(kk)
                       Eimagey(indice)=Efouriery(kk)
                       Eimagez(indice)=Efourierz(kk)
                       Eimageincx(indice)=Efourierincx(kk)
                       Eimageincy(indice)=Efourierincy(kk)
                       Eimageincz(indice)=Efourierincz(kk)
                    else
                       u1=u1/sintmp
                       u2=u2/sintmp
                       tmp=sqrt(u(3)/v(3))
                       !     Rotation de theta plus theta'
                       Eimagex(indice)=((u1*u1+(one-u1*u1)*costmp)*Efourierx(kk)+u1*u2*(one-costmp)*Efouriery(kk)+u2*sintmp&
                            &*Efourierz(kk))*tmp
                       Eimagey(indice)=(u1*u2*(one-costmp)*Efourierx(kk)+(u2*u2+(one-u2*u2)*costmp)*Efouriery(kk)-u1*sintmp&
                            &*Efourierz(kk))*tmp
                       Eimagez(indice)=(-u2*sintmp*Efourierx(kk)+u1*sintmp*Efouriery(kk)+costmp*Efourierz(kk))*tmp
                       Eimageincx(indice)=((u1*u1+(one-u1*u1)*costmp)*Efourierincx(kk)+u1*u2*(one-costmp)*Efourierincy(kk)+u2&
                            &*sintmp*Efourierincz(kk))*tmp
                       Eimageincy(indice)=(u1*u2*(one-costmp)*Efourierincx(kk)+(u2*u2+(one-u2*u2)*costmp)*Efourierincy(kk)-u1&
                            &*sintmp*Efourierincz(kk))*tmp
                       Eimageincz(indice)=(-u2*sintmp*Efourierincx(kk)+u1*sintmp*Efourierincy(kk)+costmp*Efourierincz(kk))*tmp
                    endif
                 endif
                 if (i == -nfft2d2.or.j == -nfft2d2) then
                    Eimageincx(indice)=zzero
                    Eimageincy(indice)=zzero
                    Eimageincz(indice)=zzero
                 endif
              enddo
           enddo
           write(*,*) 'Number of point in NA',2*imaxk0+1
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i=-nfft2d2,nfft2d2-1
              xy(i+nfft2d2+1)=deltax*real(i,wp)*gross
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
           do i=-imaxk0,imaxk0
              kxy(i+imaxk0+1)=deltakx*real(i,wp)/k0
           enddo
           if (nmat == 0) then
              open(400,file='kxfourier.mat')
              open(401,file='ximage.mat')
              do i=-nfft2d2,nfft2d2-1
                 write(401,*) xy(i+nfft2d2+1)
              enddo
              do i=-imaxk0,imaxk0
                 write(400,*) kxy(i+imaxk0+1)
              enddo
              close(400)
              close(401)
              open(300,file='fourier.mat')
              open(301,file='fourierx.mat')
              open(302,file='fouriery.mat')
              open(303,file='fourierz.mat')
              open(304,file='fourierinc.mat')
              open(305,file='fourierincx.mat')
              open(306,file='fourierincy.mat')
              open(307,file='fourierincz.mat')
              do j=-imaxk0,imaxk0
                 do i=-imaxk0,imaxk0
                    k=(i+nfft2d2+1)+nfft2d*(j+nfft2d2)
                    write(300,*) sqrt(real(Efourierx(k)*conjg(Efourierx(k))+Efouriery(k)*conjg(Efouriery(k))+Efourierz(k) &
                         &*conjg(Efourierz(k)),wp))
                    write(301,*) real(Efourierx(k),wp),aimag(Efourierx(k))
                    write(302,*) real(Efouriery(k),wp),aimag(Efouriery(k))
                    write(303,*) real(Efourierz(k),wp),aimag(Efourierz(k))
                    write(304,*) sqrt(real(Efourierincx(k)*conjg(Efourierincx(k))+Efourierincy(k)*conjg(Efourierincy(k)) &
                         &+Efourierincz(k)*conjg(Efourierincz(k)),wp))
                    write(305,*) real(Efourierincx(k),wp),aimag(Efourierincx(k))
                    write(306,*) real(Efourierincy(k),wp),aimag(Efourierincy(k))
                    write(307,*) real(Efourierincz(k),wp),aimag(Efourierincz(k))
                 enddo
              enddo
              close(300)
              close(301)
              close(302)
              close(303)
              close(304)
              close(305)
              close(306)
              close(307)
           elseif (nmat == 2) then
              dim(1)=nfft2d
              dim(2)=nfft2d
              datasetname='x Image'
              call hdf5write1d(group_idmic,datasetname,xy,dim)
              dim(1)=2*imaxk0+1
              dim(2)=nfft2d
              datasetname='kx Fourier'
              call hdf5write1d(group_idmic,datasetname,kxy,dim)
              k=1
              name='Fourier'
              call writehdf5mic(Efourierx,Efouriery,Efourierz,nfft2d,imaxk0,Ediffkzpos,k,name,group_idmic)
              name='Fourier+incident'
              call writehdf5mic(Efourierincx,Efourierincy,Efourierincz,nfft2d,imaxk0,Ediffkzpos,k,name,group_idmic)
           endif
        else
           write(*,*) 'Quick FFT method for holographic microscopy'
           if (nsectionsca*nquickdiffracte == 0.and.nenergie == 0) then
              !     Calcul le champ diffractÃ© par l'objet dans le domaine de Fourier par FFT
              call diffractefft2dlens(nx,ny,nz,nxm,nym,nzm,nfft2d,tabfft2,k0,xs,ys,zs,aretecube,Efourierx,Efouriery, &
                   Efourierz,FF,imaxk0,deltakx,deltaky,Ediffkzpos,numaper,nside,plan2f,plan2b,nstop,infostr)
              if (nstop == 1) return
              if (nside == -1) then
                 !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,kx,ky,ii,jj)
                 !$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
                 do i=-imaxk0,imaxk0
                    do j=-imaxk0,imaxk0
                       kx=real(i,wp)*deltakx
                       ky=real(j,wp)*deltaky
                       ii=imaxk0+i+1
                       jj=imaxk0+j+1
                       Ediffkzneg(ii,jj,1)=Ediffkzpos(ii,jj,1)
                       Ediffkzneg(ii,jj,2)=Ediffkzpos(ii,jj,2)
                       Ediffkzneg(ii,jj,3)=Ediffkzpos(ii,jj,3)
                    enddo
                 enddo
                 !$OMP ENDDO
                 !$OMP END PARALLEL
              endif
           elseif (nsectionsca*nquickdiffracte == 1.and.nenergie == 0) then
              !     DÃ©jÃ  calcule le champ diffractÃ© par FFT et le passe en Fourier en divisant par 2 i pi kz ey met en dehors d'ON a
              !     zero.
              k02=k0*k0
              if (nside == 1) then
                 !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,kx,ky,kz,ctmp,ii,jj)
                 !$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
                 do i=-imaxk0,imaxk0
                    do j=-imaxk0,imaxk0
                       kx=real(i,wp)*deltakx
                       ky=real(j,wp)*deltaky
                       ii=imaxk0+i+1
                       jj=imaxk0+j+1
                       if (kx*kx+ky*ky.ge.numaper*numaper) then
                          Ediffkzpos(ii,jj,1)=zzero
                          Ediffkzpos(ii,jj,2)=zzero
                          Ediffkzpos(ii,jj,3)=zzero
                       else
                          kz=sqrt(k0*k0-kx*kx-ky*ky)
                          ctmp=-2.0_wp*pi*icomp*kz
                          Ediffkzpos(ii,jj,1)=Ediffkzpos(ii,jj,1)*k02/ctmp
                          Ediffkzpos(ii,jj,2)=Ediffkzpos(ii,jj,2)*k02/ctmp
                          Ediffkzpos(ii,jj,3)=Ediffkzpos(ii,jj,3)*k02/ctmp
                       endif
                    enddo
                 enddo
                 !$OMP ENDDO
                 !$OMP END PARALLEL
              else
                 !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,kx,ky,kz,ctmp,ii,jj)
                 !$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
                 do i=-imaxk0,imaxk0
                    do j=-imaxk0,imaxk0
                       kx=real(i,wp)*deltakx
                       ky=real(j,wp)*deltaky
                       ii=imaxk0+i+1
                       jj=imaxk0+j+1
                       if (kx*kx+ky*ky.ge.numaper*numaper) then
                          Ediffkzneg(ii,jj,1)=zzero
                          Ediffkzneg(ii,jj,2)=zzero
                          Ediffkzneg(ii,jj,3)=zzero
                       else
                          kz=sqrt(k0*k0-kx*kx-ky*ky)
                          ctmp=-2.0_wp*pi*icomp*kz
                          Ediffkzneg(ii,jj,1)=Ediffkzneg(ii,jj,1)*k02/ctmp
                          Ediffkzneg(ii,jj,2)=Ediffkzneg(ii,jj,2)*k02/ctmp
                          Ediffkzneg(ii,jj,3)=Ediffkzneg(ii,jj,3)*k02/ctmp
                       endif
                    enddo
                 enddo
                 !$OMP ENDDO
                 !$OMP END PARALLEL
              endif
           else
              !     Met Ã  zero ce qui en dehors de l'AN: Ediffkzpos et Ediffkzneg dÃ©jÃ  calcule dans le calcul de l'energie.
              if (nside == 1) then
                 !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,kx,ky,ii,jj)
                 !$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
                 do i=-imaxk0,imaxk0
                    do j=-imaxk0,imaxk0
                       kx=real(i,wp)*deltakx
                       ky=real(j,wp)*deltaky
                       ii=imaxk0+i+1
                       jj=imaxk0+j+1
                       if (kx*kx+ky*ky.ge.numaper*numaper) then
                          Ediffkzpos(ii,jj,1)=zzero
                          Ediffkzpos(ii,jj,2)=zzero
                          Ediffkzpos(ii,jj,3)=zzero
                       endif
                    enddo
                 enddo
                 !$OMP ENDDO
                 !$OMP END PARALLEL
              else
                 !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,kx,ky,ii,jj)
                 !$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
                 do i=-imaxk0,imaxk0
                    do j=-imaxk0,imaxk0
                       kx=real(i,wp)*deltakx
                       ky=real(j,wp)*deltaky
                       ii=imaxk0+i+1
                       jj=imaxk0+j+1
                       if (kx*kx+ky*ky.ge.numaper*numaper) then
                          Ediffkzneg(ii,jj,1)=zzero
                          Ediffkzneg(ii,jj,2)=zzero
                          Ediffkzneg(ii,jj,3)=zzero
                       endif
                    enddo
                 enddo
                 !$OMP ENDDO
                 !$OMP END PARALLEL
              endif
           endif
           !     Calcul de Ediffkzpos ou Ediffkzneg de l'objet fini
           deltax=2.0_wp*pi/real(nfft2d,wp)/deltakx
           write(*,*) 'Size of FFT           : ',nfft2d
           write(*,*) 'Step size delta k     : ',deltakx,'m-1'
           write(*,*) 'Step size delta x     : ',deltax,'m'
           write(*,*) 'Number of point in NA : ',2*imaxk0+1
           !     Calcul le champ incident
           if (nenergie == 0) then
              !     Calcule le champ incident dans le plan de Fourier si pas fait energie et le met dans Ediffkzneg
              if (nside == 1) then
                 call  diffractefft2dinc(speckseed,nfft2d,k0,E0,ss,pp,theta,phi,thetam,phim,ppm,ssm,E0m,nbinc,xdip,ydip,zdip, &
                      xgaus,ygaus,zgaus,numaperil,w0,aretecube,tol,Efourierx,Efouriery,Efourierz,imaxk0,deltakx,deltaky, &
                      Ediffkzneg,numaper,beam,plan2f,plan2b,nstop,infostr)
                 if (nstop == 1) return
              elseif (nside == -1) then
                 call  diffractefft2dinckzneg(speckseed,nfft2d,k0,E0,ss,pp,theta,phi,thetam,phim,ppm,ssm,E0m,nbinc,xdip,ydip, &
                      zdip,xgaus,ygaus,zgaus,numaperil,w0,aretecube,tol,Efourierx,Efouriery,Efourierz,imaxk0,deltakx,deltaky, &
                      Ediffkzpos,numaper,beam,plan2f,plan2b,nstop,infostr)
                 if (nstop == 1) return
              endif
           else
              !     Met le champ incident dans Ediffkzneg si dÃ©jÃ  calcule avec l'energie (dans Efourier)
              if (nside == 1) then
                 !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,kx,ky,ii,jj,indicex,indicey,indice)
                 !$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
                 do i=-imaxk0,imaxk0
                    do j=-imaxk0,imaxk0
                       kx=real(i,wp)*deltakx
                       ky=real(j,wp)*deltaky
                       ii=imaxk0+i+1
                       jj=imaxk0+j+1
                       Ediffkzneg(ii,jj,1)=zzero
                       Ediffkzneg(ii,jj,2)=zzero
                       Ediffkzneg(ii,jj,3)=zzero
                       if (kx*kx+ky*ky.lt.numaper*numaper) then
                          if (i.ge.0) then
                             indicex=i+1
                          else
                             indicex=nfft2d+i+1
                          endif
                          if (j.ge.0) then
                             indicey=j+1
                          else
                             indicey=nfft2d+j+1
                          endif
                          indice=indicex+nfft2d*(indicey-1)
                          Ediffkzneg(ii,jj,1)=Efourierx(indice)
                          Ediffkzneg(ii,jj,2)=Efouriery(indice)
                          Ediffkzneg(ii,jj,3)=Efourierz(indice)
                       endif
                    enddo
                 enddo
                 !$OMP ENDDO
                 !$OMP END PARALLEL
              elseif (nside == -1) then
                 !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,kx,ky,ii,jj,indicex,indicey,indice)
                 !$OMP DO SCHEDULE(DYNAMIC) COLLAPSE(2)
                 do i=-imaxk0,imaxk0
                    do j=-imaxk0,imaxk0
                       kx=real(i,wp)*deltakx
                       ky=real(j,wp)*deltaky
                       ii=imaxk0+i+1
                       jj=imaxk0+j+1
                       Ediffkzpos(ii,jj,1)=zzero
                       Ediffkzpos(ii,jj,2)=zzero
                       Ediffkzpos(ii,jj,3)=zzero
                       if (kx*kx+ky*ky.lt.numaper*numaper) then
                          if (i.ge.0) then
                             indicex=i+1
                          else
                             indicex=nfft2d+i+1
                          endif
                          if (j.ge.0) then
                             indicey=j+1
                          else
                             indicey=nfft2d+j+1
                          endif
                          indice=indicex+nfft2d*(indicey-1)
                          Ediffkzpos(ii,jj,1)=Efourierx(indice)
                          Ediffkzpos(ii,jj,2)=Efouriery(indice)
                          Ediffkzpos(ii,jj,3)=Efourierz(indice)
                       endif
                    enddo
                 enddo
                 !$OMP ENDDO
                 !$OMP END PARALLEL
              endif
           endif
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,kx)
           !$OMP DO SCHEDULE(STATIC)
           do i=-nfft2d2,nfft2d2-1
              xy(i+nfft2d2+1)=deltax*real(i,wp)*gross
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,kx)
           !$OMP DO SCHEDULE(STATIC)
           do i=-imaxk0,imaxk0
              kx=deltakx*real(i,wp)
              kxy(i+imaxk0+1)=kx/k0
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
           if (nmat == 0) then
              open(400,file='kxfourier.mat')
              open(401,file='ximage.mat')
              do i=-nfft2d2,nfft2d2-1
                 write(401,*) xy(i+nfft2d2+1)
              enddo
              do i=-imaxk0,imaxk0
                 write(400,*) kxy(i+imaxk0+1)
              enddo
              close(400)
              close(401)
           elseif (nmat == 2) then
              dim(1)=2*imaxk0+1
              dim(2)=nfft2d
              datasetname='kx Fourier'
              call hdf5write1d(group_idmic,datasetname,kxy,dim)
              dim(1)=nfft2d
              dim(2)=nfft2d
              datasetname='x Image'
              call hdf5write1d(group_idmic,datasetname,xy,dim)
           endif
           !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
           !$OMP DO SCHEDULE(STATIC)
           do i=1,nfft2d*nfft2d
              Efourierx(i)=zzero
              Efouriery(i)=zzero
              Efourierz(i)=zzero
              Efourierincx(i)=zzero
              Efourierincy(i)=zzero
              Efourierincz(i)=zzero
              Eimageincx(i)=zzero
              Eimageincy(i)=zzero
              Eimageincz(i)=zzero
              Eimagex(i)=zzero
              Eimagey(i)=zzero
              Eimagez(i)=zzero
           enddo
           !$OMP ENDDO
           !$OMP END PARALLEL
           !     Calcul en transmission et reflexion en meme temps
           if (nside == 1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,indicex,indicey,indice,kx,ky,kz,zfocus,u1,u2,sintmp,costmp,tmp,u,v,ii,jj,kk)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
              do i=-imaxk0,imaxk0
                 do j=-imaxk0,imaxk0
                    if (i.ge.0) then
                       indicex=i+1
                    else
                       indicex=nfft2d+i+1
                    endif
                    if (j.ge.0) then
                       indicey=j+1
                    else
                       indicey=nfft2d+j+1
                    endif
                    indice=indicex+nfft2d*(indicey-1)
                    kx=deltakx*real(i,wp)
                    ky=deltakx*real(j,wp)
                    if (kx*kx+ky*ky.lt.numaper*numaper) then
                       kz=k0*k0-kx*kx-ky*ky
                       kz=sqrt(kz)
                       zfocus=exp(icomp*kz*zlens)
                       ii=imaxk0+i+1
                       jj=imaxk0+j+1
                       kk=i+nfft2d2+1+nfft2d*(j+nfft2d2)
                       Efourierx(kk)=Ediffkzpos(ii,jj,1)*zfocus
                       Efouriery(kk)=Ediffkzpos(ii,jj,2)*zfocus
                       Efourierz(kk)=Ediffkzpos(ii,jj,3)*zfocus
                       Efourierincx(kk)=(Ediffkzpos(ii,jj,1)+Ediffkzneg(ii,jj,1))*zfocus
                       Efourierincy(kk)=(Ediffkzpos(ii,jj,2)+Ediffkzneg(ii,jj,2))*zfocus
                       Efourierincz(kk)=(Ediffkzpos(ii,jj,3)+Ediffkzneg(ii,jj,3))*zfocus
                       u(1)=kx/k0
                       u(2)=ky/k0
                       u(3)=real(nside,wp)*sqrt(one-u(1)*u(1)-u(2)*u(2))
                       v(1)=-kx/k0/gross
                       v(2)=-ky/k0/gross
                       v(3)=real(nside,wp)*sqrt(one-v(1)*v(1)-v(2)*v(2))
                       u1=u(2)*v(3)-u(3)*v(2)
                       u2=-u(1)*v(3)+u(3)*v(1)
                       costmp=u(1)*v(1)+u(2)*v(2)+u(3)*v(3)
                       sintmp=sqrt(u1*u1+u2*u2)
                       if (sintmp == zero) then
                          Eimagex(indice)=Efourierx(kk)
                          Eimagey(indice)=Efouriery(kk)
                          Eimagez(indice)=Efourierz(kk)
                          Eimageincx(indice)=Efourierincx(kk)
                          Eimageincy(indice)=Efourierincy(kk)
                          Eimageincz(indice)=Efourierincz(kk)
                       else
                          u1=u1/sintmp
                          u2=u2/sintmp
                          tmp=sqrt(u(3)/v(3))
                          Eimageincx(indice)=((u1*u1+(one-u1*u1)*costmp)*Efourierincx(kk)+u1*u2*(one-costmp)*Efourierincy(kk)&
                               &+u2*sintmp*Efourierincz(kk))*tmp
                          Eimageincy(indice)=(u1*u2*(one-costmp)*Efourierincx(kk)+(u2*u2+(one-u2*u2)*costmp)*Efourierincy(kk) &
                               &-u1*sintmp*Efourierincz(kk))*tmp
                          Eimageincz(indice)=(-u2*sintmp*Efourierincx(kk)+u1*sintmp*Efourierincy(kk)+costmp*Efourierincz(kk))&
                               &*tmp
                          Eimagex(indice)=((u1*u1+(one-u1*u1)*costmp)*Efourierx(kk)+u1*u2*(one-costmp)*Efouriery(kk)+u2*sintmp&
                               &*Efourierz(kk))*tmp
                          Eimagey(indice)=(u1*u2*(one-costmp)*Efourierx(kk)+(u2*u2+(one-u2*u2)*costmp)*Efouriery(kk)-u1*sintmp&
                               &*Efourierz(kk))*tmp
                          Eimagez(indice)=(-u2*sintmp*Efourierx(kk)+u1*sintmp*Efouriery(kk)+costmp*Efourierz(kk))*tmp
                       endif
                    endif
                 enddo
              enddo
              !$OMP ENDDO
              !$OMP END PARALLEL
           elseif (nside == -1) then
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i,j,indicex,indicey,indice,kx,ky,kz,zfocus,u1,u2,sintmp,costmp,tmp,u,v,ii,jj,kk)
!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
              do i=-imaxk0,imaxk0
                 do j=-imaxk0,imaxk0
                    if (i.ge.0) then
                       indicex=i+1
                    else
                       indicex=nfft2d+i+1
                    endif
                    if (j.ge.0) then
                       indicey=j+1
                    else
                       indicey=nfft2d+j+1
                    endif
                    indice=indicex+nfft2d*(indicey-1)
                    kx=deltakx*real(i,wp)
                    ky=deltakx*real(j,wp)
                    if (kx*kx+ky*ky.lt.numaper*numaper) then
                       kz=k0*k0-kx*kx-ky*ky
                       kz=sqrt(kz)
                       zfocus=exp(icomp*kz*zlens)
                       ii=imaxk0+i+1
                       jj=imaxk0+j+1
                       kk=i+nfft2d2+1+nfft2d*(j+nfft2d2)
                       Efourierx(kk)=Ediffkzneg(ii,jj,1)*zfocus
                       Efouriery(kk)=Ediffkzneg(ii,jj,2)*zfocus
                       Efourierz(kk)=Ediffkzneg(ii,jj,3)*zfocus
                       Efourierincx(kk)=(Ediffkzpos(ii,jj,1)+Ediffkzneg(ii,jj,1))*zfocus
                       Efourierincy(kk)=(Ediffkzpos(ii,jj,2)+Ediffkzneg(ii,jj,2))*zfocus
                       Efourierincz(kk)=(Ediffkzpos(ii,jj,3)+Ediffkzneg(ii,jj,3))*zfocus
                       u(1)=kx/k0
                       u(2)=ky/k0
                       u(3)=real(nside,wp)*sqrt(one-u(1)*u(1)-u(2)*u(2))
                       v(1)=-kx/k0/gross
                       v(2)=-ky/k0/gross
                       v(3)=real(nside,wp)*sqrt(one-v(1)*v(1)-v(2)*v(2))
                       u1=u(2)*v(3)-u(3)*v(2)
                       u2=-u(1)*v(3)+u(3)*v(1)
                       costmp=u(1)*v(1)+u(2)*v(2)+u(3)*v(3)
                       sintmp=sqrt(u1*u1+u2*u2)
                       if (sintmp == zero) then
                          Eimagex(indice)=Efourierx(kk)
                          Eimagey(indice)=Efouriery(kk)
                          Eimagez(indice)=Efourierz(kk)
                          Eimageincx(indice)=Efourierincx(kk)
                          Eimageincy(indice)=Efourierincy(kk)
                          Eimageincz(indice)=Efourierincz(kk)
                       else
                          u1=u1/sintmp
                          u2=u2/sintmp
                          tmp=sqrt(u(3)/v(3))
                          Eimageincx(indice)=((u1*u1+(one-u1*u1)*costmp)*Efourierincx(kk)+u1*u2*(one-costmp)*Efourierincy(kk)&
                               &+u2*sintmp*Efourierincz(kk))*tmp
                          Eimageincy(indice)=(u1*u2*(one-costmp)*Efourierincx(kk)+(u2*u2+(one-u2*u2)*costmp)*Efourierincy(kk)&
                               &-u1*sintmp*Efourierincz(kk))*tmp
                          Eimageincz(indice)=(-u2*sintmp*Efourierincx(kk)+u1*sintmp*Efourierincy(kk)+costmp*Efourierincz(kk))&
                               &*tmp
                          Eimagex(indice)=((u1*u1+(one-u1*u1)*costmp)*Efourierx(kk)+u1*u2*(one-costmp)*Efouriery(kk)+u2*sintmp&
                               &*Efourierz(kk))*tmp
                          Eimagey(indice)=(u1*u2*(one-costmp)*Efourierx(kk)+(u2*u2+(one-u2*u2)*costmp)*Efouriery(kk)-u1*sintmp&
                               &*Efourierz(kk))*tmp
                          Eimagez(indice)=(-u2*sintmp*Efourierx(kk)+u1*sintmp*Efouriery(kk)+costmp*Efourierz(kk))*tmp
                       endif
                    endif
                 enddo
              enddo
              !$OMP ENDDO
              !$OMP END PARALLEL
           endif
           if (nmat == 0) then
              open(300,file='fourier.mat')
              open(301,file='fourierx.mat')
              open(302,file='fouriery.mat')
              open(303,file='fourierz.mat')
              open(304,file='fourierinc.mat')
              open(305,file='fourierincx.mat')
              open(306,file='fourierincy.mat')
              open(307,file='fourierincz.mat')
              do j=-imaxk0,imaxk0
                 do i=-imaxk0,imaxk0
                    k=(i+nfft2d2+1)+nfft2d*(j+nfft2d2)
                    write(300,*) sqrt(real(Efourierx(k)*conjg(Efourierx(k))+Efouriery(k)*conjg(Efouriery(k))+Efourierz(k) &
                         &*conjg(Efourierz(k)),wp))
                    write(301,*) real(Efourierx(k),wp),aimag(Efourierx(k))
                    write(302,*) real(Efouriery(k),wp),aimag(Efouriery(k))
                    write(303,*) real(Efourierz(k),wp),aimag(Efourierz(k))
                    write(304,*) sqrt(real(Efourierincx(k)*conjg(Efourierincx(k))+Efourierincy(k)*conjg(Efourierincy(k))&
                         &+Efourierincz(k)*conjg(Efourierincz(k)),wp))
                    write(305,*) real(Efourierincx(k),wp),aimag(Efourierincx(k))
                    write(306,*) real(Efourierincy(k),wp),aimag(Efourierincy(k))
                    write(307,*) real(Efourierincz(k),wp),aimag(Efourierincz(k))
                 enddo
              enddo
              close(300)
              close(301)
              close(302)
              close(303)
              close(304)
              close(305)
              close(306)
              close(307)
           elseif (nmat == 2) then
              k=1
              name='Fourier'
              call writehdf5mic(Efourierx,Efouriery,Efourierz,nfft2d,imaxk0,Ediffkzpos,k,name,group_idmic)
              name='Fourier+incident'
              call writehdf5mic(Efourierincx,Efourierincy,Efourierincz,nfft2d,imaxk0,Ediffkzpos,k,name,group_idmic)
           endif
        endif
        write(*,*) 'Compute the images through the microscope'
        call fouriertoimage(deltakx,deltaky,gross,Eimagex,Eimagey,Eimagez,Eimageincx,Eimageincy,Eimageincz,nfft2D,nfft2d2&
             &,plan2b,plan2f)
        !     Calcul intensite de l'image
        if (nmat == 0) then
           open(301,file='image.mat')
           open(302,file='imagex.mat')
           open(303,file='imagey.mat')
           open(304,file='imagez.mat')
           open(305,file='imageinc.mat')
           open(306,file='imageincx.mat')
           open(307,file='imageincy.mat')
           open(308,file='imageincz.mat')
           do i=1,nfft2D*nfft2D
              write(301,*) sqrt(real(Eimagex(i)*conjg(Eimagex(i))+Eimagey(i)*conjg(Eimagey(i))+Eimagez(i)*conjg(Eimagez(i)),wp))
              write(302,*) real(Eimagex(i),wp),aimag(Eimagex(i))
              write(303,*) real(Eimagey(i),wp),aimag(Eimagey(i))
              write(304,*) real(Eimagez(i),wp),aimag(Eimagez(i))
              write(305,*) sqrt(real(Eimageincx(i)*conjg(Eimageincx(i))+Eimageincy(i)*conjg(Eimageincy(i))+Eimageincz(i) &
                   &*conjg(Eimageincz(i)),wp))
              write(306,*) real(Eimageincx(i),wp),aimag(Eimageincx(i))
              write(307,*) real(Eimageincy(i),wp),aimag(Eimageincy(i))
              write(308,*) real(Eimageincz(i),wp),aimag(Eimageincz(i))
           enddo
           close(301)
           close(302)
           close(303)
           close(304)
           close(305)
           close(306)
           close(307)
           close(308)
        elseif (nmat == 2) then
           k=0
           name='Image'
           call writehdf5mic(Eimagex,Eimagey,Eimagez,nfft2d,imaxk0,Ediffkzpos,k,name,group_idmic)
           name='Image+incident'
           call writehdf5mic(Eimageincx,Eimageincy,Eimageincz,nfft2d,imaxk0,Ediffkzpos,k,name,group_idmic)
        endif
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i)
        !$OMP DO SCHEDULE(STATIC)
        do i=1,nfft2d
           kxy(i)=deltakx*real(i-nfft2d2-1,wp)/k0
        enddo
        !$OMP ENDDO
        !$OMP END PARALLEL
        write(*,*) 'End image through the microscope'
        writecol='************ END MICROSCOPY **************'
        colorw='pink'
        call writecolor(writecol,colorw)
        write(*,*) ' '
     endif
  endif

  !     Output file
302 if (nmat == 2) then
#ifdef USE_HDF5
     !     Fermeture du fichier hdf5
     dim(1)=1
     dim(2)=1
     nx=0
     datasetname=trim(PROG_STRING)
     call hdf5write_int(group_idver,datasetname,nx,dim)
     CALL h5gclose_f(group_idopt,error)
     CALL h5gclose_f(group_iddip,error)
     CALL h5gclose_f(group_idff,error)
     CALL h5gclose_f(group_idmic,error)
     CALL h5gclose_f(group_idof,error)
     CALL h5gclose_f(group_idnf,error)
     CALL h5gclose_f(group_idver,error)
     call hdf5close(file_id)
     file_id=0
     write(*,*) 'Close HDF5 files'
#else
     write(*,*) 'No HDF5 files'
#endif
     write(*,*) ' '
  endif

  if (nstop == -1) then
     infostr = 'Calculation cancelled at the end!'
     return
  endif

  call date_and_time(date,time,zzone,valuesf)
  call cpu_time(tf)
  messagetemps=' to execute the code '
  write(*,*) '**************** CPU TIME ****************'
  call calculatedate(valuesf,valuesi,tf,ti,messagetemps,tempsreeltotal)
  write(*,*) '******************************************'
  infostr='COMPLETED!'
  writecol='*************** END **********************'
  colorw='red'
  call writecolor(writecol,colorw)
#ifdef IFDDA_CUDA_INVERSE_DRIVER
  call ifdda_print_cuda_transfer_stats()
#endif

end subroutine cdmlib
