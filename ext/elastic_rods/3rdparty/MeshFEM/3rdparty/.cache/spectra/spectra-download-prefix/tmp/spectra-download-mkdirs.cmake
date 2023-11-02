# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/cmake/../3rdparty/spectra"
  "/home/qbeck/Research/Cshells/build/spectra-build"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/spectra/spectra-download-prefix"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/spectra/spectra-download-prefix/tmp"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/spectra/spectra-download-prefix/src/spectra-download-stamp"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/spectra/spectra-download-prefix/src"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/spectra/spectra-download-prefix/src/spectra-download-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/spectra/spectra-download-prefix/src/spectra-download-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/3rdparty/.cache/spectra/spectra-download-prefix/src/spectra-download-stamp${cfgdir}") # cfgdir has leading slash
endif()
