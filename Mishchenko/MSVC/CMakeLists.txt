cmake_minimum_required(VERSION 3.20)
project(IFDDAMishchenkoMSVC LANGUAGES CXX)

if(NOT MSVC)
    message(FATAL_ERROR "Select the MSVC toolchain.")
endif()
add_subdirectory(".." mishchenko)
