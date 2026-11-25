# Keep the tested hybrid structure: Fortran main0 + cdmlib + precision,
# with CUDA C++ leaf routines (no duplicate Fortran leaf definitions).
# Compile separate double/float targets so Fortran module kinds and the
# CUDA ABI precision always match; never share their .mod files.
set(_cuda_leaf ${IFDDA_CUDA_CPP_SOURCES})
list(FILTER _cuda_leaf EXCLUDE REGEX "/cdmlib\\.cpp$")

function(add_ifdda_hybrid_f90_cuda_variant precision)
  if(precision STREQUAL "float")
    set(_precision_definitions USE_SIMP PRODUITMATVECTFFT_SINGLE_PRECISION=1)
    set(_core ifdda_hybrid_f90_core_float)
    set(_leaf ifdda_hybrid_f90_cuda_leaf_float)
    set(_modules "${CMAKE_CURRENT_BINARY_DIR}/modules_float")
  else()
    set(_precision_definitions PRODUITMATVECTFFT_SINGLE_PRECISION=0)
    # Preserve existing double-precision CMake target names.
    set(_core ifdda_hybrid_f90_core)
    set(_leaf ifdda_hybrid_f90_cuda_leaf)
    set(_modules "${CMAKE_CURRENT_BINARY_DIR}/modules")
  endif()
  set(_exe "test_main0_f90_cdmlib_f90_cuda_${precision}")

  file(MAKE_DIRECTORY "${_modules}")
  add_library(${_core} STATIC
    "${CMAKE_CURRENT_LIST_DIR}/../src/cdmlib.f90"
    "${CMAKE_CURRENT_LIST_DIR}/../src/mod_precision.f90")
  set_target_properties(${_core} PROPERTIES
    Fortran_PREPROCESS ON Fortran_MODULE_DIRECTORY "${_modules}")
  target_include_directories(${_core} PUBLIC "${_modules}")
  target_compile_options(${_core} PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-fopenmp>
    $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>)
  target_compile_definitions(${_core} PRIVATE
    ${_precision_definitions} IFDDA_CUDA_INVERSE_DRIVER
    IFDDA_CUDA_CHAN_INITIAL_RESIDENT)

  add_library(${_leaf} STATIC ${_cuda_leaf})
  if(TARGET ifdda_scattnlay)
    ifdda_enable_scattnlay(${_leaf})
  endif()
  target_include_directories(${_leaf} PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/../src"
    "${IF_DDA_ROOT}/cdmlib_C99" "${CUDAToolkit_INCLUDE_DIRS}")
  target_compile_definitions(${_leaf} PRIVATE
    ${_precision_definitions} IFDDA_CUDA_INVERSE_DRIVER
    IFDDA_CUDA_CHAN_INITIAL_RESIDENT)
  target_link_libraries(${_leaf} PUBLIC
    "${IFDDA_OPENBLAS_LIBRARY}" CUDA::cudart CUDA::cufft CUDA::cublas
    "ifdda_cuda_kernels_${precision}"
    "E:/msys64/mingw64/lib/libgfortran.dll.a")

  add_executable(${_exe} "${CMAKE_CURRENT_LIST_DIR}/../src/main0.f90")
  set_target_properties(${_exe} PROPERTIES Fortran_PREPROCESS ON)
  target_compile_definitions(${_exe} PRIVATE ${_precision_definitions})
  target_include_directories(${_exe} PRIVATE "${_modules}")
  target_compile_options(${_exe} PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-fopenmp>
    $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>)
  target_link_libraries(${_exe} PRIVATE ${_core} ${_leaf} gomp)
  add_custom_command(TARGET ${_exe} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${IFDDA_KERNEL_DLL_${precision}}" "$<TARGET_FILE_DIR:${_exe}>")
endfunction()

add_ifdda_hybrid_f90_cuda_variant(double)
add_ifdda_hybrid_f90_cuda_variant(float)
