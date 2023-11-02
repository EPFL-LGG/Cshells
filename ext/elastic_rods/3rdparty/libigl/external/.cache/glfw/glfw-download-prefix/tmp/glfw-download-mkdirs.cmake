# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/cmake/../external/glfw"
  "/home/qbeck/Research/Cshells/build/glfw-build"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/glfw/glfw-download-prefix"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/glfw/glfw-download-prefix/tmp"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/glfw/glfw-download-prefix/src/glfw-download-stamp"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/glfw/glfw-download-prefix/src"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/glfw/glfw-download-prefix/src/glfw-download-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/glfw/glfw-download-prefix/src/glfw-download-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/glfw/glfw-download-prefix/src/glfw-download-stamp${cfgdir}") # cfgdir has leading slash
endif()
