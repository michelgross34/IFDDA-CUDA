cmake_minimum_required(VERSION 3.18)
project(IFDDAScattnlayWSL LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Configure this project under WSL or Linux.")
endif()
add_subdirectory(".." scattnlay)
