cmake_minimum_required(VERSION 3.20)

# MSVC-only build: build main0.cpp + cdmlib.cpp in float and double precision.
# Fortran-driver variants are built by MINGW and WSL.
project(if_dda_cuda LANGUAGES CXX)

if(NOT MSVC)
    message(FATAL_ERROR "Use CUDA/MINGW/CMakeLists.txt for GCC/MinGW builds.")
endif()

# Keep old CLion CMake caches from re-enabling the Fortran hybrid targets.
# Those targets belong to CUDA/MINGW, not to the pure MSVC configuration.
set(IFDDA_BUILD_HYBRID_FORTRAN_CUDA OFF CACHE BOOL
    "Fortran main + CUDA C++ hybrid (disabled in MSVC-only build)" FORCE)
set(IFDDA_BUILD_HYBRID_F90_CUDA_LEAF OFF CACHE BOOL
    "Fortran cdmlib + CUDA leaves hybrid (disabled in MSVC-only build)" FORCE)

option(IFDDA_CUDA_CHAN_INITIAL_RESIDENT
    "Keep the standard right-Chan initial iterate on the CUDA device" ON)
option(IFDDA_TEMPORARY_FORTRAN_FALLBACKS
    "Temporarily link unported routines from ../cdmlib for runtime validation" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# When CLion starts the Ninja generator without a Visual Studio developer
# environment, cl.exe is found but its standard headers and import libraries
# are not: INCLUDE and LIB are empty. Supply those search paths explicitly;
# do not alter the C++/CUDA sources or rely on a particular shell startup.
if(MSVC)
    get_filename_component(_ifdda_msvc_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_ifdda_msvc_toolset "${_ifdda_msvc_bin_dir}/../../.." ABSOLUTE)
    if(NOT EXISTS "${_ifdda_msvc_toolset}/include/vector")
        message(FATAL_ERROR "Cannot locate MSVC C++ headers alongside ${CMAKE_CXX_COMPILER}")
    endif()

    set(_ifdda_sdk_root "C:/Program Files (x86)/Windows Kits/10")
    file(GLOB _ifdda_sdk_include_versions LIST_DIRECTORIES TRUE
        "${_ifdda_sdk_root}/Include/10.*")
    list(SORT _ifdda_sdk_include_versions COMPARE NATURAL ORDER DESCENDING)
    set(_ifdda_sdk_version "")
    foreach(_candidate IN LISTS _ifdda_sdk_include_versions)
        get_filename_component(_version "${_candidate}" NAME)
        if(EXISTS "${_candidate}/ucrt/stdio.h" AND
           EXISTS "${_ifdda_sdk_root}/Lib/${_version}/um/x64/kernel32.lib")
            set(_ifdda_sdk_version "${_version}")
            break()
        endif()
    endforeach()
    if(NOT _ifdda_sdk_version)
        message(FATAL_ERROR "A complete x64 Windows 10 SDK was not found")
    endif()

    include_directories(SYSTEM
        "${_ifdda_msvc_toolset}/include"
        "${_ifdda_sdk_root}/Include/${_ifdda_sdk_version}/ucrt"
        "${_ifdda_sdk_root}/Include/${_ifdda_sdk_version}/shared"
        "${_ifdda_sdk_root}/Include/${_ifdda_sdk_version}/um"
        "${_ifdda_sdk_root}/Include/${_ifdda_sdk_version}/winrt")
    link_directories(
        "${_ifdda_msvc_toolset}/lib/x64"
        "${_ifdda_sdk_root}/Lib/${_ifdda_sdk_version}/ucrt/x64"
        "${_ifdda_sdk_root}/Lib/${_ifdda_sdk_version}/um/x64")
    message(STATUS "MSVC x64 headers/libs: ${_ifdda_msvc_toolset}; Windows SDK ${_ifdda_sdk_version}")
endif()

get_filename_component(IFDDA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(IFDDA_CUDA_SOURCE_DIR "${IFDDA_CUDA_ROOT}/src" ABSOLUTE)
get_filename_component(IF_DDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(IFDDA_CPP_MAIN "${IFDDA_CUDA_SOURCE_DIR}/main0.cpp")
if(NOT EXISTS "${IFDDA_CPP_MAIN}")
    message(FATAL_ERROR "C++ driver not found: ${IFDDA_CPP_MAIN}")
endif()

# Compile the C++ scientific translation and CUDA-backed helper routines.
# random_bridge_c.cpp provides a portable C++ RNG when built by MSVC.
add_subdirectory("${IFDDA_CUDA_ROOT}/Scattnlay" Scattnlay EXCLUDE_FROM_ALL)
set(IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/random_bridge_c.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/cdmlib.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/cdmlib_param.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/rytov_vector_field.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/calculforcefft_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/derivativefield2_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/espace_libre_derive_mat_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/diffractefft2d_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/computegcfft2d_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/inverserigoptscalaire_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/inverserigscalaireoptpd_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/inverserigscalaireoptpg_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/abcchi_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/produitmatvectfft_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/produitpolavect_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/produitfftmatvectopt_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/produitfftmatvectscalaire_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/calculgreenscalairefft_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/multilayerbornsca_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/calculatedate_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/green_tensor_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/geometry_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/dipole_map_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/object_all.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/geometry_dispatch_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/geometry_sphere_pml_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/inhomogeneous_material_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/polaepsto0_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/polarizability_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/interpdielec_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gaussian_power_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gpbicg_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gpbicgplus_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gpbicgsafe_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gpbicgar_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gpbicgar2_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gpbicor_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/cors_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/tfqmr_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/bicgstarplus_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/zcg_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/pimzbicgstab_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/qmrbicgstab_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/idrs_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/idrs_rng_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gpbicgstabl_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/bicglstab_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/pimzqmr_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/ssbicgsafe2_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gcors2_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gcors2_random_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/speckle_rng_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/inverserig_all_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/inverserig_legacy_bridge_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/chan_right_inverse_driver_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/chan_inverse_driver_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/inverserigscalaire_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/inverserigscalaire_abi_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/preconditionneur_apply_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/preconditionneur_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/rerangechamppola_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/preconditionneurscal_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/cuda_utils.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/cuda_data.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/beampropagation_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/beampropagationmacro_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/gaussianfft_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/incident_field_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/incidentarbitrary_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/rytov_vector_field_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/local_macro_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/cross_sections_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/scattering_results_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/mie_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/micromasque_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/microsdf_fields_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/microsdf_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/hdf5_noop_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/quick_diffraction_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/energy_conservation_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/optical_force_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/derivechamp_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/pml_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/speckle_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/confocal_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/microsconfocal_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/microsbf_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/microsconfocal_legacy_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/microsconfocalfluo_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/widefield_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/calculwidefield_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/savemacroscopicfield_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/comparaisonxyz_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/incidentarbitrarydercreate_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/incidentarbitraryder1_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/incidentarbitrarypos_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/relecture_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/diffractefft2dinckzneg_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/diffractefft2dlens_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/diffractefft2denergie_CUDA.cpp")
list(APPEND IFDDA_CUDA_CPP_SOURCES
    "${IFDDA_CUDA_SOURCE_DIR}/diffractefft2denergieconfocal_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/diffractefft2dinc_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/microsbfneg_CUDA.cpp")

# FFTW and OpenBLAS runtime libraries are provided as Windows DLLs.
if(WIN32)
    set(IFDDA_RUNTIME_BIN_DIR "E:/msys64/mingw64/bin" CACHE PATH
        "MinGW runtime DLL directory")
    set(IFDDA_FFTW_ROOT "${IF_DDA_ROOT}/fftw" CACHE PATH "FFTW bundle" FORCE)
    set(IFDDA_OPENBLAS_LIBRARY "E:/msys64/mingw64/lib/libopenblas.dll.a"
        CACHE FILEPATH "MinGW OpenBLAS import library" FORCE)
    set(IFDDA_FFTW_DOUBLE_LIBRARY "${IFDDA_FFTW_ROOT}/libfftw3.dll.a"
        CACHE FILEPATH "FFTW double import library" FORCE)
    set(IFDDA_FFTW_DOUBLE_THREADS_LIBRARY "${IFDDA_FFTW_ROOT}/libfftw3_omp.dll.a"
        CACHE FILEPATH "FFTW double OpenMP import library" FORCE)
    set(IFDDA_FFTW_FLOAT_LIBRARY "${IFDDA_FFTW_ROOT}/libfftw3f.dll.a"
        CACHE FILEPATH "FFTW single import library" FORCE)
    set(IFDDA_FFTW_FLOAT_THREADS_LIBRARY "${IFDDA_FFTW_ROOT}/libfftw3f_omp.dll.a"
        CACHE FILEPATH "FFTW single OpenMP import library" FORCE)
else()
    find_library(IFDDA_OPENBLAS_LIBRARY NAMES openblas REQUIRED)
    find_library(IFDDA_FFTW_DOUBLE_LIBRARY NAMES fftw3 REQUIRED)
    find_library(IFDDA_FFTW_DOUBLE_THREADS_LIBRARY NAMES fftw3_omp REQUIRED)
    find_library(IFDDA_FFTW_FLOAT_LIBRARY NAMES fftw3f REQUIRED)
    find_library(IFDDA_FFTW_FLOAT_THREADS_LIBRARY NAMES fftw3f_omp REQUIRED)
endif()

foreach(_library IFDDA_OPENBLAS_LIBRARY
        IFDDA_FFTW_DOUBLE_LIBRARY IFDDA_FFTW_DOUBLE_THREADS_LIBRARY
        IFDDA_FFTW_FLOAT_LIBRARY IFDDA_FFTW_FLOAT_THREADS_LIBRARY)
    if(NOT EXISTS "${${_library}}")
        message(FATAL_ERROR "Required library not found: ${${_library}}")
    endif()
endforeach()

# cdmlib.cpp relies on a transitive <vector> include under MinGW.  Make the
# dependency explicit for MSVC through this target's compiler configuration,
# without changing the common C++ source file.
set_source_files_properties("${IFDDA_CUDA_SOURCE_DIR}/cdmlib.cpp"
    PROPERTIES COMPILE_OPTIONS "/FIvector")

function(add_ifdda_cuda_variant precision)
    set(_library "ifdda_cuda_${precision}")
    set(_executable "test_main0_cpp_cdmlib_cpp_cuda_${precision}")


    if(precision STREQUAL "float")
        set(_precision_defs USE_SIMP PRODUITMATVECTFFT_SINGLE_PRECISION=1)
        set(_fftw_main "${IFDDA_FFTW_FLOAT_LIBRARY}")
        set(_fftw_omp "${IFDDA_FFTW_FLOAT_THREADS_LIBRARY}")
        set(_fftw_dlls libfftw3f-3.dll libfftw3f_omp-3.dll)
    else()
        set(_precision_defs PRODUITMATVECTFFT_SINGLE_PRECISION=0)
        set(_fftw_main "${IFDDA_FFTW_DOUBLE_LIBRARY}")
        set(_fftw_omp "${IFDDA_FFTW_DOUBLE_THREADS_LIBRARY}")
        set(_fftw_dlls libfftw3-3.dll libfftw3_omp-3.dll)
    endif()

    add_library(${_library} STATIC ${IFDDA_CUDA_CPP_SOURCES})
    ifdda_enable_scattnlay(${_library})

    # Visual Studio may compile several translation units against one /Zi PDB.
    # Serialize writes to that PDB (MSVC C1041 when /m or /MP is active).
    target_compile_options(${_library} PRIVATE /FS)


    target_include_directories(${_library} PUBLIC
        "${IFDDA_CUDA_SOURCE_DIR}"
        "${IFDDA_CUDA_SOURCE_DIR}" "${IF_DDA_ROOT}/cdmlib_C99" "${IF_DDA_ROOT}/fftw")
    target_compile_definitions(${_library} PUBLIC USE_FFTW ${_precision_defs})
    target_link_libraries(${_library} PUBLIC
        "${IFDDA_OPENBLAS_LIBRARY}" "${_fftw_omp}" "${_fftw_main}")


    target_compile_definitions(${_library} PRIVATE IFDDA_CUDA_INVERSE_DRIVER)

    if(IFDDA_CUDA_CHAN_INITIAL_RESIDENT)
        target_compile_definitions(${_library} PRIVATE
            IFDDA_CUDA_CHAN_INITIAL_RESIDENT)
    endif()
    target_include_directories(${_library} PUBLIC "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(${_library} PUBLIC
        CUDA::cudart CUDA::cufft CUDA::cublas
        "ifdda_cuda_kernels_${precision}")

    add_executable(${_executable} "${IFDDA_CPP_MAIN}")

    target_link_libraries(${_executable} PRIVATE ${_library})

    if(WIN32)
        # The runtime DLL is loaded from beside the executable. Track it as a
        # link dependency so a go.bat rebuild refreshes the copied runtime DLL.
        set_property(TARGET ${_executable} APPEND PROPERTY LINK_DEPENDS
            "${IFDDA_KERNEL_DLL_${precision}}")
    endif()



    if(WIN32)
        set(_runtime_dlls ${_fftw_dlls}
            libopenblas.dll libgfortran-5.dll libgomp-1.dll
            libgcc_s_seh-1.dll libquadmath-0.dll libwinpthread-1.dll)
        list(APPEND _runtime_dlls cudart64_110.dll cufft64_10.dll cublas64_11.dll)
        add_custom_command(TARGET ${_executable} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${IFDDA_KERNEL_DLL_${precision}}" "$<TARGET_FILE_DIR:${_executable}>")
        foreach(_dll IN LISTS _runtime_dlls)
            if(_dll MATCHES "^libfftw")
                set(_dll_source "${IFDDA_FFTW_ROOT}/${_dll}")
            elseif(_dll MATCHES "^(cudart|cufft|cublas)")
                set(_dll_source "${CUDAToolkit_BIN_DIR}/${_dll}")
            else()
                set(_dll_source "${IFDDA_RUNTIME_BIN_DIR}/${_dll}")
            endif()
            if(NOT EXISTS "${_dll_source}")
                message(FATAL_ERROR "Runtime DLL not found: ${_dll_source}")
            endif()
            add_custom_command(TARGET ${_executable} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_dll_source}" "$<TARGET_FILE_DIR:${_executable}>")
        endforeach()
    endif()
endfunction()

if(NOT WIN32)
    message(FATAL_ERROR "CUDA DLL targets require Windows and CUDA/go.bat outputs")
endif()
if(NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
endif()
find_package(CUDAToolkit 11.8 REQUIRED)
if(NOT CUDAToolkit_VERSION MATCHES "^11\\.8(\\.|$)")
    message(FATAL_ERROR "CUDA targets require CUDA 11.8.x")
endif()

foreach(_precision double float)
    set(_kernel_dll_default
        "${CMAKE_CURRENT_LIST_DIR}/../src/kernel_${_precision}.dll")
    
    set(_kernel_implib_default
        "${CMAKE_CURRENT_LIST_DIR}/../src/kernel_${_precision}.lib")
    set(IFDDA_KERNEL_DLL_${_precision} "${_kernel_dll_default}" CACHE FILEPATH
        "CUDA wrapper DLL for ${_precision} precision")
    set(IFDDA_KERNEL_IMPLIB_${_precision} "${_kernel_implib_default}" CACHE FILEPATH
        "CUDA wrapper import library for ${_precision} precision")
    if(NOT EXISTS "${IFDDA_KERNEL_DLL_${_precision}}" OR
       NOT EXISTS "${IFDDA_KERNEL_IMPLIB_${_precision}}")
        message(FATAL_ERROR
            "CUDA ${_precision} DLL/import library missing. Run CUDA/go.bat; expected:\n"
            "  ${IFDDA_KERNEL_DLL_${_precision}}\n"
            "  ${IFDDA_KERNEL_IMPLIB_${_precision}}")
    endif()
    add_library(ifdda_cuda_kernels_${_precision} SHARED IMPORTED GLOBAL)
    set_target_properties(ifdda_cuda_kernels_${_precision} PROPERTIES
        IMPORTED_LOCATION "${IFDDA_KERNEL_DLL_${_precision}}"
        IMPORTED_IMPLIB "${IFDDA_KERNEL_IMPLIB_${_precision}}"
        INTERFACE_INCLUDE_DIRECTORIES "${IFDDA_CUDA_SOURCE_DIR};${CUDAToolkit_INCLUDE_DIRS}")
endforeach()

add_ifdda_cuda_variant(double)
add_ifdda_cuda_variant(float)

message(STATUS "MSVC executables: test_main0_cpp_cdmlib_cpp_cuda_double, test_main0_cpp_cdmlib_cpp_cuda_float (main0.cpp + cdmlib.cpp)")

# Build and deploy the accelerated Mishchenko reference alongside all IF-DDA executables.
include("${CMAKE_CURRENT_LIST_DIR}/../Mishchenko/IFDDAIntegration.cmake")
ifdda_configure_mishchenko(IFDDA_WINDOWS_TMATRIX)
if(IFDDA_WINDOWS_TMATRIX)
    ifdda_attach_windows_references("${CMAKE_CURRENT_SOURCE_DIR}")
endif()
