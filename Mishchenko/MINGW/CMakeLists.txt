cmake_minimum_required(VERSION 3.20)
project(IFDDAMishchenkoMINGW LANGUAGES CXX)

if(NOT MINGW)
    message(FATAL_ERROR "Select the MinGW-w64 toolchain.")
endif()
add_subdirectory(".." mishchenko)
