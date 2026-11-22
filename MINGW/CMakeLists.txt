cmake_minimum_required(VERSION 3.20)

# CMake builds only CUDA-backed executables and their ordinary Fortran/C++ host
# code. The CUDA wrappers are built separately through go.bat as imported
# DLL/import-library pairs; kernel.cu is never a CMake source.
project(if_dda_cuda LANGUAGES CXX)

# Keep every generated executable at the selected build root (for example,
# CUDA/MINGW/cmake-build-debug), including targets declared in subdirectories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

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

get_filename_component(IFDDA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(IFDDA_CUDA_SOURCE_DIR "${IFDDA_CUDA_ROOT}/src" ABSOLUTE)
get_filename_component(IF_DDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(IFDDA_CPP_MAIN "${IFDDA_CUDA_SOURCE_DIR}/main0.cpp")
if(NOT EXISTS "${IFDDA_CPP_MAIN}")
    message(FATAL_ERROR "C++ driver not found: ${IFDDA_CPP_MAIN}")
endif()

# Compile the local Fortran entry/orchestrator, precision definition, and
# CUDA-aware Fortran ABI dispatchers. The RNG bridge is C++: with MinGW/g++ it
# calls libgfortran's native random_seed/random_number ABI, preserving the
# exact Fortran stream for CUDA stochastic paths.
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
    "${IFDDA_CUDA_SOURCE_DIR}/object_all.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/geometry_dispatch_CUDA.cpp"
    "${IFDDA_CUDA_SOURCE_DIR}/dipole_map_CUDA.cpp"
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

# Use the MinGW-compatible BLAS and FFTW dependencies.
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


    target_include_directories(${_library} PUBLIC
        "${IFDDA_CUDA_SOURCE_DIR}"
        "${IFDDA_CUDA_SOURCE_DIR}" "${IF_DDA_ROOT}/cdmlib_C99" "${IF_DDA_ROOT}/fftw")
    target_compile_definitions(${_library} PUBLIC USE_FFTW ${_precision_defs})
    target_link_libraries(${_library} PUBLIC
        "${IFDDA_OPENBLAS_LIBRARY}" "${_fftw_omp}" "${_fftw_main}" "E:/msys64/mingw64/lib/libgfortran.dll.a")


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
        "${IFDDA_CUDA_ROOT}/src/kernel_${_precision}.dll")
    
    set(_kernel_implib_default
        "${IFDDA_CUDA_ROOT}/src/kernel_${_precision}.lib")
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

enable_language(Fortran)
    enable_language(Fortran)
    add_library(ifdda_hybrid_precision_double STATIC
        "${IFDDA_CUDA_SOURCE_DIR}/mod_precision.f90")
    set_target_properties(ifdda_hybrid_precision_double PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/hybrid_fortran_modules")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/hybrid_fortran_modules")
    # Real CUDA/main0.f90 entry point with the C++ cdmlib.cpp orchestrator
    # and the CUDA/C++ leaf implementations (including objetsphere_).
    add_executable(test_main0_f90_cdmlib_cpp_cuda_double
        "${IFDDA_CUDA_SOURCE_DIR}/main0.f90")
    set_target_properties(test_main0_f90_cdmlib_cpp_cuda_double PROPERTIES
        Fortran_PREPROCESS ON)
    target_compile_options(test_main0_f90_cdmlib_cpp_cuda_double PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>)
    target_include_directories(test_main0_f90_cdmlib_cpp_cuda_double PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/hybrid_fortran_modules")
    target_link_libraries(test_main0_f90_cdmlib_cpp_cuda_double PRIVATE
        ifdda_hybrid_precision_double ifdda_cuda_double)
    add_dependencies(test_main0_f90_cdmlib_cpp_cuda_double test_main0_cpp_cdmlib_cpp_cuda_double)

    # The identical main0.f90 is also required for a genuine float reference
    # comparison; isolate the USE_SIMP precision module from the double one.
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/hybrid_fortran_float_modules")
    add_library(ifdda_hybrid_precision_float STATIC
        "${IFDDA_CUDA_SOURCE_DIR}/mod_precision.f90")
    set_target_properties(ifdda_hybrid_precision_float PROPERTIES
        Fortran_PREPROCESS ON
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/hybrid_fortran_float_modules")
    target_compile_definitions(ifdda_hybrid_precision_float PUBLIC USE_SIMP)
    target_include_directories(ifdda_hybrid_precision_float PUBLIC
        "${CMAKE_CURRENT_BINARY_DIR}/hybrid_fortran_float_modules")
    add_executable(test_main0_f90_cdmlib_cpp_cuda_float
        "${IFDDA_CUDA_SOURCE_DIR}/main0.f90")
    set_target_properties(test_main0_f90_cdmlib_cpp_cuda_float PROPERTIES
        Fortran_PREPROCESS ON)
    target_compile_options(test_main0_f90_cdmlib_cpp_cuda_float PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>)
    target_compile_definitions(test_main0_f90_cdmlib_cpp_cuda_float PRIVATE USE_SIMP)
    target_link_libraries(test_main0_f90_cdmlib_cpp_cuda_float PRIVATE
        ifdda_hybrid_precision_float ifdda_cuda_float)
    add_dependencies(test_main0_f90_cdmlib_cpp_cuda_float test_main0_cpp_cdmlib_cpp_cuda_float)

message(STATUS "CUDA executables: six main0/cdmlib combinations, simple et double precision")



add_subdirectory("${IFDDA_CUDA_ROOT}/hybrid_fortran_cuda" "${CMAKE_CURRENT_BINARY_DIR}/hybrid_fortran_cuda")
foreach(_precision IN ITEMS double float)
    add_dependencies(test_main0_f90_cdmlib_f90_cuda_${_precision}
        test_main0_cpp_cdmlib_cpp_cuda_${_precision})
endforeach()
