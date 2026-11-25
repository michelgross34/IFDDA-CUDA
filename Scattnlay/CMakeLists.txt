cmake_minimum_required(VERSION 3.18)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(IFDDAScattnlay LANGUAGES CXX)
endif()

add_library(ifdda_scattnlay STATIC concentric_mie.cpp)
target_compile_features(ifdda_scattnlay PRIVATE cxx_std_20)
set_target_properties(ifdda_scattnlay PROPERTIES
    OUTPUT_NAME ifdda_scattnlay_static
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

add_library(ifdda_scattnlay_shared SHARED scattnlay_c.cpp)
target_link_libraries(ifdda_scattnlay_shared PRIVATE ifdda_scattnlay)
target_compile_features(ifdda_scattnlay_shared PRIVATE cxx_std_11)
target_compile_definitions(ifdda_scattnlay_shared PRIVATE IFDDA_SCATTNLAY_EXPORTS)
target_include_directories(ifdda_scattnlay_shared PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(ifdda_scattnlay_shared PROPERTIES
    OUTPUT_NAME ifdda_scattnlay
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
if(WIN32)
    set_target_properties(ifdda_scattnlay_shared PROPERTIES PREFIX "")
endif()

function(ifdda_enable_scattnlay target)
    set_property(SOURCE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/mie_CUDA.cpp"
        TARGET_DIRECTORY ${target} PROPERTY COMPILE_DEFINITIONS IFDDA_WITH_SCATTNLAY)
    target_link_libraries(${target} PRIVATE ifdda_scattnlay)
endfunction()

add_executable(ifdda_scattnlay_sections sections_main.cpp)
target_link_libraries(ifdda_scattnlay_sections PRIVATE ifdda_scattnlay)
target_compile_features(ifdda_scattnlay_sections PRIVATE cxx_std_11)
if(MINGW)
    target_link_options(ifdda_scattnlay_shared PRIVATE -static)
    target_link_options(ifdda_scattnlay_sections PRIVATE -static)
endif()
