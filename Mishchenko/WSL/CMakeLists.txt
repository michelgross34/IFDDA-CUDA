cmake_minimum_required(VERSION 3.20)
project(IFDDAMishchenkoWSL LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Configure this project under WSL or Linux.")
endif()
add_subdirectory(".." mishchenko)
