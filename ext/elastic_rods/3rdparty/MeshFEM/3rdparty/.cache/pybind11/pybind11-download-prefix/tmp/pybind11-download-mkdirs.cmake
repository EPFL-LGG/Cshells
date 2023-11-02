# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/cmake/../3rdparty/pybind11"
  "/home/qbeck/Research/Cshells/build/pybind11-build"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/pybind11/pybind11-download-prefix"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/pybind11/pybind11-download-prefix/tmp"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/pybind11/pybind11-download-prefix/src"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp${cfgdir}") # cfgdir has leading slash
endif()
