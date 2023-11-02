# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/cmake/../external/imgui"
  "/home/qbeck/Research/Cshells/build/imgui-build"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/imgui/imgui-download-prefix"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/imgui/imgui-download-prefix/tmp"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/imgui/imgui-download-prefix/src/imgui-download-stamp"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/imgui/imgui-download-prefix/src"
  "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/imgui/imgui-download-prefix/src/imgui-download-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/imgui/imgui-download-prefix/src/imgui-download-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/external/.cache/imgui/imgui-download-prefix/src/imgui-download-stamp${cfgdir}") # cfgdir has leading slash
endif()
