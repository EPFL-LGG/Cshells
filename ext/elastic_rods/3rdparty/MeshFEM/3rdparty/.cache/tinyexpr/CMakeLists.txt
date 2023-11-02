# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 3.1)

project(tinyexpr-download NONE)

include(ExternalProject)
ExternalProject_Add(tinyexpr-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/codeplea/tinyexpr.git;GIT_TAG;ffb0d41b13e5f8d318db95feb071c220c134fe70
                    SOURCE_DIR          "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/cmake/../3rdparty/tinyexpr"
                    BINARY_DIR          "/home/qbeck/Research/Cshells/build/tinyexpr-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
