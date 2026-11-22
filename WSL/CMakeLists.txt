cmake_minimum_required(VERSION 3.22)

if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR
        "CUDA/WSL requires CLion configured with a Linux/WSL toolchain. "
        "Select the WSL toolchain and regenerate the CMake profile; "
        "do not reuse a Windows CMake cache.")
endif()

# CLion: select the WSL toolchain, then open CUDA/WSL as a CMake project.
# Unlike CUDA/MINGW, this builds kernel.cu with Linux nvcc and never imports
# precompiled Windows DLLs or links FFTW into the CUDA targets.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH
        "CUDA compiler inside WSL")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING
        "CUDA SM architectures (70 for NVIDIA TITAN V)")
endif()
project(ifdda_cuda_wsl LANGUAGES CXX Fortran CUDA)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Use CLion with a Linux/WSL toolchain for CUDA/WSL")
endif()
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

get_filename_component(IFDDA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(IFDDA_CUDA_SOURCE_DIR "${IFDDA_CUDA_ROOT}/src" ABSOLUTE)
get_filename_component(IF_DDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(IFDDA_WSL_BUILD_HYBRID ON)
set(IFDDA_WSL_BUILD_CPP_CDMLIB ON)
option(IFDDA_CUDA_CHAN_INITIAL_RESIDENT
    "Keep the standard right-Chan initial iterate on the CUDA device" ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX Fortran)
find_library(IFDDA_OPENBLAS_LIBRARY NAMES openblas REQUIRED)

# Keep this list aligned with CUDA/CMakeLists.txt. In the hybrid build,
# cdmlib.cpp is excluded: CUDA/cdmlib.f90 supplies the same entry point.
set(IFDDA_CUDA_CPP_SOURCES
    random_bridge_c.cpp
    cdmlib.cpp
    calculforcefft_CUDA.cpp
    derivativefield2_CUDA.cpp
    espace_libre_derive_mat_CUDA.cpp
    diffractefft2d_CUDA.cpp
    computegcfft2d_CUDA.cpp
    inverserigoptscalaire_CUDA.cpp
    inverserigscalaireoptpd_CUDA.cpp
    inverserigscalaireoptpg_CUDA.cpp
    abcchi_CUDA.cpp
    objetarbitrary_CUDA.cpp
    produitmatvectfft_CUDA.cpp
    produitpolavect_CUDA.cpp
    produitfftmatvectopt_CUDA.cpp
    produitfftmatvectscalaire_CUDA.cpp
    calculgreenscalairefft_CUDA.cpp
    multilayerbornsca_CUDA.cpp
    calculatedate_CUDA.cpp
    deltakroutine_CUDA.cpp
    green_tensor_CUDA.cpp
    geometry_CUDA.cpp
    object_all.cpp
    geometry_dispatch_CUDA.cpp
    dipole_map_CUDA.cpp
    objectsphere_CUDA.cpp
    objectsphereconcentric_CUDA.cpp
    objectnspheres_CUDA.cpp
    objectsphererandom_CUDA.cpp
    objectsphererandomnxnynz_CUDA.cpp
    objectsphererandommulti_CUDA.cpp
    objectparanxnynz_CUDA.cpp
    objectpara_CUDA.cpp
    objectcylindre_CUDA.cpp
    inhomogeneous_material_CUDA.cpp
    objectparainhomonxnynz_CUDA.cpp
    objectparainhomo_CUDA.cpp
    objectsphereinhomo_CUDA.cpp
    objectellipse_CUDA.cpp
    polaepsto0_CUDA.cpp
    polarizability_CUDA.cpp
    interpdielec_CUDA.cpp
    gaussian_power_CUDA.cpp
    gpbicg_CUDA.cpp
    gpbicgplus_CUDA.cpp
    gpbicgsafe_CUDA.cpp
    gpbicgar_CUDA.cpp
    gpbicgar2_CUDA.cpp
    gpbicor_CUDA.cpp
    cors_CUDA.cpp
    tfqmr_CUDA.cpp
    bicgstarplus_CUDA.cpp
    zcg_CUDA.cpp
    pimzbicgstab_CUDA.cpp
    qmrbicgstab_CUDA.cpp
    idrs_CUDA.cpp
    idrs_rng_CUDA.cpp
    gpbicgstabl_CUDA.cpp
    bicglstab_CUDA.cpp
    pimzqmr_CUDA.cpp
    ssbicgsafe2_CUDA.cpp
    gcors2_CUDA.cpp
    gcors2_random_CUDA.cpp
    speckle_rng_CUDA.cpp
    inverserig_all_CUDA.cpp
    inverserig_legacy_bridge_CUDA.cpp
    chan_right_inverse_driver_CUDA.cpp
    chan_inverse_driver_CUDA.cpp
    inverserigscalaire_CUDA.cpp
    inverserigscalaire_abi_CUDA.cpp
    preconditionneur_apply_CUDA.cpp
    preconditionneur_CUDA.cpp
    rerangechamppola_CUDA.cpp
    preconditionneurscal_CUDA.cpp
    cuda_utils.cpp
    cuda_data.cpp
    beampropagation_CUDA.cpp
    beampropagationmacro_CUDA.cpp
    gaussianfft_CUDA.cpp
    incident_field_CUDA.cpp
    incidentarbitrary_CUDA.cpp
    rytov_vector_field_CUDA.cpp
    local_macro_CUDA.cpp
    cross_sections_CUDA.cpp
    scattering_results_CUDA.cpp
    mie_CUDA.cpp
    micromasque_CUDA.cpp
    microsdf_fields_CUDA.cpp
    microsdf_CUDA.cpp
    hdf5_noop_CUDA.cpp
    quick_diffraction_CUDA.cpp
    energy_conservation_CUDA.cpp
    optical_force_CUDA.cpp
    derivechamp_CUDA.cpp
    pml_CUDA.cpp
    speckle_CUDA.cpp
    confocal_CUDA.cpp
    microsconfocal_CUDA.cpp
    microsbf_CUDA.cpp
    microsconfocal_legacy_CUDA.cpp
    microsconfocalfluo_CUDA.cpp
    widefield_CUDA.cpp
    calculwidefield_CUDA.cpp
    savemacroscopicfield_CUDA.cpp
    comparaisonxyz_CUDA.cpp
    incidentarbitrarydercreate_CUDA.cpp
    incidentarbitraryder1_CUDA.cpp
    incidentarbitrarypos_CUDA.cpp
    relecture_CUDA.cpp
    diffractefft2dinckzneg_CUDA.cpp
    diffractefft2dlens_CUDA.cpp
    diffractefft2denergie_CUDA.cpp
    diffractefft2denergieconfocal_CUDA.cpp
    diffractefft2dinc_CUDA.cpp
    microsdfneg_CUDA.cpp
    microsbfneg_CUDA.cpp)
list(TRANSFORM IFDDA_CUDA_CPP_SOURCES PREPEND "${IFDDA_CUDA_SOURCE_DIR}/")
foreach(_source IN LISTS IFDDA_CUDA_CPP_SOURCES)
    if(NOT EXISTS "${_source}")
        message(FATAL_ERROR "Missing CUDA C++ source: ${_source}")
    endif()
endforeach()

function(ifdda_wsl_precision_settings target precision)
    if(precision STREQUAL "float")
        target_compile_definitions(${target} PRIVATE
            USE_SIMP PRODUITMATVECTFFT_SINGLE_PRECISION=1)
    else()
        target_compile_definitions(${target} PRIVATE
            PRODUITMATVECTFFT_SINGLE_PRECISION=0)
    endif()
endfunction()

foreach(_precision IN ITEMS double float)
    # Build a native .so from the current kernel source in each configuration.
    # Debug -G enables device debugging; Release keeps CUDA optimizations.
    add_library(ifdda_cuda_kernels_${_precision} SHARED
        "${IFDDA_CUDA_SOURCE_DIR}/kernel.cu")
    ifdda_wsl_precision_settings(ifdda_cuda_kernels_${_precision} ${_precision})
    set_target_properties(ifdda_cuda_kernels_${_precision} PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared CUDA_SEPARABLE_COMPILATION OFF)
    target_include_directories(ifdda_cuda_kernels_${_precision} PUBLIC
        "${IFDDA_CUDA_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
    target_compile_options(ifdda_cuda_kernels_${_precision} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)
    target_link_libraries(ifdda_cuda_kernels_${_precision} PUBLIC
        CUDA::cudart CUDA::cufft CUDA::cublas)

    if(IFDDA_WSL_BUILD_HYBRID)
        set(_module_dir "${CMAKE_CURRENT_BINARY_DIR}/fortran_modules_${_precision}")
        file(MAKE_DIRECTORY "${_module_dir}")

        add_library(ifdda_main_f90_cdmlib_f90_core_${_precision} STATIC
            "${IFDDA_CUDA_SOURCE_DIR}/cdmlib.f90"
            "${IFDDA_CUDA_SOURCE_DIR}/mod_precision.f90")
        set_target_properties(ifdda_main_f90_cdmlib_f90_core_${_precision} PROPERTIES
            Fortran_PREPROCESS ON Fortran_MODULE_DIRECTORY "${_module_dir}")
        ifdda_wsl_precision_settings(ifdda_main_f90_cdmlib_f90_core_${_precision} ${_precision})
        target_compile_definitions(ifdda_main_f90_cdmlib_f90_core_${_precision} PRIVATE
            IFDDA_CUDA_INVERSE_DRIVER)
        target_compile_options(ifdda_main_f90_cdmlib_f90_core_${_precision} PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>)
        target_include_directories(ifdda_main_f90_cdmlib_f90_core_${_precision} PUBLIC
            "${_module_dir}")
        target_link_libraries(ifdda_main_f90_cdmlib_f90_core_${_precision} PUBLIC
            OpenMP::OpenMP_Fortran)

        set(_leaf_sources ${IFDDA_CUDA_CPP_SOURCES})
        list(REMOVE_ITEM _leaf_sources "${IFDDA_CUDA_SOURCE_DIR}/cdmlib.cpp")
        add_library(ifdda_main_f90_cdmlib_f90_cuda_leaf_${_precision} STATIC
            ${_leaf_sources}
            "${IFDDA_CUDA_SOURCE_DIR}/rytov_vector_field.cpp")
        ifdda_wsl_precision_settings(ifdda_main_f90_cdmlib_f90_cuda_leaf_${_precision} ${_precision})
        target_compile_definitions(ifdda_main_f90_cdmlib_f90_cuda_leaf_${_precision} PRIVATE
            IFDDA_CUDA_INVERSE_DRIVER IFDDA_WSL_ABI_INT64_PLAN)
        target_include_directories(ifdda_main_f90_cdmlib_f90_cuda_leaf_${_precision} PUBLIC
            "${IFDDA_CUDA_SOURCE_DIR}"
            "${IF_DDA_ROOT}/cdmlib_C99" "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(ifdda_main_f90_cdmlib_f90_cuda_leaf_${_precision} PUBLIC
            ifdda_cuda_kernels_${_precision} "${IFDDA_OPENBLAS_LIBRARY}"
            OpenMP::OpenMP_CXX)

        set(_main_f90_cdmlib_f90_exe "test_main_f90_cdmlib_f90_cuda_${_precision}")
        add_executable(${_main_f90_cdmlib_f90_exe} "${IFDDA_CUDA_SOURCE_DIR}/main0.f90")
        set_target_properties(${_main_f90_cdmlib_f90_exe} PROPERTIES Fortran_PREPROCESS ON)
        ifdda_wsl_precision_settings(${_main_f90_cdmlib_f90_exe} ${_precision})
        target_include_directories(${_main_f90_cdmlib_f90_exe} PRIVATE "${_module_dir}")
        target_compile_options(${_main_f90_cdmlib_f90_exe} PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>)
        target_link_libraries(${_main_f90_cdmlib_f90_exe} PRIVATE
            ifdda_main_f90_cdmlib_f90_core_${_precision}
            ifdda_main_f90_cdmlib_f90_cuda_leaf_${_precision})
    endif()

    # Two independent entry points sharing the C++ cdmlib implementation:
    # main0.f90 -> cdmlib.cpp, and main0.cpp -> cdmlib.cpp.
    # Do not link cdmlib.f90: it defines the same cdmlib_ symbol.
    if(IFDDA_WSL_BUILD_CPP_CDMLIB)
        set(_cpp_cdmlib_sources ${IFDDA_CUDA_CPP_SOURCES})
        list(APPEND _cpp_cdmlib_sources
            "${IFDDA_CUDA_SOURCE_DIR}/cdmlib_param.cpp"
            "${IFDDA_CUDA_SOURCE_DIR}/rytov_vector_field.cpp")
        add_library(ifdda_cdmlib_cpp_cuda_${_precision} STATIC
            ${_cpp_cdmlib_sources})
        ifdda_wsl_precision_settings(ifdda_cdmlib_cpp_cuda_${_precision} ${_precision})
        target_compile_definitions(ifdda_cdmlib_cpp_cuda_${_precision} PRIVATE
            IFDDA_CUDA_INVERSE_DRIVER IFDDA_WSL_ABI_INT64_PLAN)
        target_include_directories(ifdda_cdmlib_cpp_cuda_${_precision} PUBLIC
            "${IFDDA_CUDA_SOURCE_DIR}"
            "${IF_DDA_ROOT}/cdmlib_C99" "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(ifdda_cdmlib_cpp_cuda_${_precision} PUBLIC
            ifdda_cuda_kernels_${_precision} "${IFDDA_OPENBLAS_LIBRARY}"
            OpenMP::OpenMP_CXX OpenMP::OpenMP_Fortran)

        # main0.f90 needs the Fortran precision module but not cdmlib.f90.
        set(_cpp_module_dir
            "${CMAKE_CURRENT_BINARY_DIR}/fortran_cpp_modules_${_precision}")
        file(MAKE_DIRECTORY "${_cpp_module_dir}")
        add_library(ifdda_cpp_main_precision_${_precision} STATIC
            "${IFDDA_CUDA_SOURCE_DIR}/mod_precision.f90")
        set_target_properties(ifdda_cpp_main_precision_${_precision} PROPERTIES
            Fortran_PREPROCESS ON Fortran_MODULE_DIRECTORY "${_cpp_module_dir}")
        ifdda_wsl_precision_settings(ifdda_cpp_main_precision_${_precision} ${_precision})
        target_include_directories(ifdda_cpp_main_precision_${_precision} PUBLIC
            "${_cpp_module_dir}")

        set(_fortran_cpp_exe "test_main0_f90_cdmlib_cpp_cuda_${_precision}")
        add_executable(${_fortran_cpp_exe}
            "${IFDDA_CUDA_SOURCE_DIR}/main0.f90")
        set_target_properties(${_fortran_cpp_exe} PROPERTIES
            Fortran_PREPROCESS ON)
        ifdda_wsl_precision_settings(${_fortran_cpp_exe} ${_precision})
        target_compile_options(${_fortran_cpp_exe} PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>)
        target_link_libraries(${_fortran_cpp_exe} PRIVATE
            ifdda_cpp_main_precision_${_precision}
            ifdda_cdmlib_cpp_cuda_${_precision})

        set(_cpp_cpp_exe "test_main0_cpp_cdmlib_cpp_cuda_${_precision}")
        add_executable(${_cpp_cpp_exe}
            "${IFDDA_CUDA_SOURCE_DIR}/main0.cpp")
        ifdda_wsl_precision_settings(${_cpp_cpp_exe} ${_precision})
        target_link_libraries(${_cpp_cpp_exe} PRIVATE
            ifdda_cdmlib_cpp_cuda_${_precision})
    endif()
endforeach()

if(IFDDA_CUDA_CHAN_INITIAL_RESIDENT)
    foreach(_precision IN ITEMS double float)
        if(IFDDA_WSL_BUILD_HYBRID)
            target_compile_definitions(ifdda_main_f90_cdmlib_f90_core_${_precision} PRIVATE
                IFDDA_CUDA_CHAN_INITIAL_RESIDENT)
            target_compile_definitions(ifdda_main_f90_cdmlib_f90_cuda_leaf_${_precision} PRIVATE
                IFDDA_CUDA_CHAN_INITIAL_RESIDENT)
        endif()
        if(IFDDA_WSL_BUILD_CPP_CDMLIB)
            target_compile_definitions(ifdda_cdmlib_cpp_cuda_${_precision} PRIVATE
                IFDDA_CUDA_CHAN_INITIAL_RESIDENT)
        endif()
    endforeach()
endif()

message(STATUS "WSL CUDA toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "WSL CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "WSL hybrid Fortran/CUDA: ${IFDDA_WSL_BUILD_HYBRID}")
message(STATUS "WSL executable set: six main0/cdmlib combinations, float and double")
