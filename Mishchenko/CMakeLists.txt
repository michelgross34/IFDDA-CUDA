cmake_minimum_required(VERSION 3.20)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(IFDDAMishchenko LANGUAGES CXX)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(nasa_dir "${CMAKE_CURRENT_SOURCE_DIR}/NASA")
set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(core "${work_dir}/mish_core.f" "${work_dir}/lpd.f")
set(wrapper "${CMAKE_CURRENT_SOURCE_DIR}/mishchenko_reference.f90")
set(exports "${CMAKE_CURRENT_SOURCE_DIR}/ifdda_mishchenko.def")
set(fortran_flags -std=legacy -fallow-argument-mismatch)
add_custom_command(OUTPUT ${core}
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/prepare_core.py" --output-dir "${work_dir}"
    DEPENDS prepare_core.py "${nasa_dir}/ampld.lp.f" "${nasa_dir}/ampld.par.f" "${nasa_dir}/lpd.f"
    VERBATIM)

if(MSVC)
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "Select an x64 MSVC toolchain for the GNU Fortran DLL.")
    endif()
    find_program(IFDDA_MISHCHENKO_GFORTRAN NAMES gfortran REQUIRED
        DOC "64-bit Windows GNU Fortran compiler for the Mishchenko DLL")
    execute_process(COMMAND "${IFDDA_MISHCHENKO_GFORTRAN}" -dumpmachine
        OUTPUT_VARIABLE fortran_target OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE fortran_status)
    if(NOT fortran_status EQUAL 0 OR NOT fortran_target MATCHES "^x86_64.*mingw")
        message(FATAL_ERROR "IFDDA_MISHCHENKO_GFORTRAN must target x86_64 Windows (MinGW-w64).")
    endif()
    if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
    set(out "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
    set(dll "${out}/ifdda_mishchenko.dll")
    set(implib "${out}/ifdda_mishchenko.lib")
    add_custom_command(OUTPUT "${dll}" "${implib}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${out}"
        COMMAND "${IFDDA_MISHCHENKO_GFORTRAN}" -O2 ${fortran_flags}
            "-I${nasa_dir}" "-J${out}" -shared -static -static-libgfortran -static-libgcc
            -o "${dll}" ${core} "${wrapper}" "${exports}"
        COMMAND "${CMAKE_AR}" /nologo /machine:x64 "/def:${exports}" "/out:${implib}"
        DEPENDS ${core} "${wrapper}" "${exports}" "${nasa_dir}/ampld.par.f"
        VERBATIM)
    add_custom_target(ifdda_mishchenko_shared ALL DEPENDS "${dll}" "${implib}")
    add_library(ifdda_mishchenko_link SHARED IMPORTED)
    foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES CMAKE_BUILD_TYPE)
        string(TOUPPER "${config}" config_upper)
        set_target_properties(ifdda_mishchenko_link PROPERTIES
            "IMPORTED_LOCATION_${config_upper}" "${CMAKE_CURRENT_BINARY_DIR}/${config}/ifdda_mishchenko.dll"
            "IMPORTED_IMPLIB_${config_upper}" "${CMAKE_CURRENT_BINARY_DIR}/${config}/ifdda_mishchenko.lib")
    endforeach()
    add_dependencies(ifdda_mishchenko_link ifdda_mishchenko_shared)
else()
    enable_language(Fortran)
    if(NOT CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR "Select GNU Fortran for the NASA legacy sources.")
    endif()
    add_library(ifdda_mishchenko_shared SHARED ${core} "${wrapper}")
    set_target_properties(ifdda_mishchenko_shared PROPERTIES OUTPUT_NAME ifdda_mishchenko)
    target_include_directories(ifdda_mishchenko_shared PRIVATE "${nasa_dir}")
    target_compile_options(ifdda_mishchenko_shared PRIVATE ${fortran_flags})
    if(MINGW)
        target_sources(ifdda_mishchenko_shared PRIVATE "${exports}")
        target_link_options(ifdda_mishchenko_shared PRIVATE -static -static-libgfortran -static-libgcc)
        set_target_properties(ifdda_mishchenko_shared PROPERTIES PREFIX "")
    endif()
    add_library(ifdda_mishchenko_link ALIAS ifdda_mishchenko_shared)
endif()

add_executable(ifdda_mishchenko_sections sections_main.cpp)
target_compile_features(ifdda_mishchenko_sections PRIVATE cxx_std_17)
target_link_libraries(ifdda_mishchenko_sections PRIVATE ifdda_mishchenko_link)
if(MSVC)
    set_target_properties(ifdda_mishchenko_sections PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${out}")
elseif(MINGW)
    target_link_options(ifdda_mishchenko_sections PRIVATE -static)
else()
    set_target_properties(ifdda_mishchenko_sections PROPERTIES BUILD_RPATH_USE_ORIGIN ON)
endif()
