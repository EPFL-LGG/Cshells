# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 3.1)

project(triangle-download NONE)

include(ExternalProject)
ExternalProject_Add(triangle-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/triangle.git;GIT_TAG;d6761dd691e2e1318c83bf7773fea88d9437464a
                    SOURCE_DIR          "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/cmake/../3rdparty/triangle"
                    BINARY_DIR          "/home/qbeck/Research/Cshells/build/triangle-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
