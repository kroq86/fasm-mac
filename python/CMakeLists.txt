cmake_minimum_required(VERSION 3.18)
project(eml_sr_python LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/bindings.cpp)
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../fasm/apps/eml_sr")
target_compile_features(_core PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(_core PRIVATE /O2)
else()
    target_compile_options(_core PRIVATE -O2)
endif()

install(TARGETS _core LIBRARY DESTINATION eml_sr)
