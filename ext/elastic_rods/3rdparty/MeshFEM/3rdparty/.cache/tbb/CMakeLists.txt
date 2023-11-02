# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 3.1)

project(tbb-download NONE)

include(ExternalProject)
ExternalProject_Add(tbb-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/wjakob/tbb.git;GIT_TAG;141b0e310e1fb552bdca887542c9c1a8544d6503
                    SOURCE_DIR          "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/cmake/../3rdparty/tbb"
                    BINARY_DIR          "/home/qbeck/Research/Cshells/build/tbb-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
