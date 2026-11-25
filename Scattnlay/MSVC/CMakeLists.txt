cmake_minimum_required(VERSION 3.18)
project(IFDDAScattnlayMSVC LANGUAGES CXX)

if(NOT MSVC)
    message(FATAL_ERROR "Select the MSVC toolchain.")
endif()
add_subdirectory(".." scattnlay)
