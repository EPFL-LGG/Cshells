cmake_minimum_required(VERSION 3.1)
project(OffscreenRenderer)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OFFSCREEN_RENDERER_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${OFFSCREEN_RENDERER_ROOT}/cmake)

# Configurable options
option(USE_OSMESA "Use the software rasterization library OSMesa instead of a GPU-accelerated EGL/CGL context" OFF)

# Color diagnostics
add_definitions(-fdiagnostics-color=always)

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings (shared libraries) to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src/OffscreenRenderer)

if (TARGET offscreen_renderer)
    add_subdirectory(src/python_bindings)
endif()
