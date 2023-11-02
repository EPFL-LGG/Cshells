# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 3.1)

project(spectra-download NONE)

include(ExternalProject)
ExternalProject_Add(spectra-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/yixuan/spectra.git;GIT_TAG;ec27cfd2210a9b2322825c4cb8e5d47f014e1ac3
                    SOURCE_DIR          "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/cmake/../3rdparty/spectra"
                    BINARY_DIR          "/home/qbeck/Research/Cshells/build/spectra-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
