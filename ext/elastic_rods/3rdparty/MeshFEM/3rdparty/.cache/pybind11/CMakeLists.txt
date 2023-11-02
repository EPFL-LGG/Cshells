# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 3.1)

project(pybind11-download NONE)

include(ExternalProject)
ExternalProject_Add(pybind11-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/pybind/pybind11.git;GIT_TAG;v2.6.1
                    SOURCE_DIR          "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/cmake/../3rdparty/pybind11"
                    BINARY_DIR          "/home/qbeck/Research/Cshells/build/pybind11-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
