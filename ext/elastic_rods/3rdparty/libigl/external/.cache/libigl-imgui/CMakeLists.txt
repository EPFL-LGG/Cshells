# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(libigl-imgui-download NONE)

include(ExternalProject)
ExternalProject_Add(libigl-imgui-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/libigl-imgui.git;GIT_TAG;a37e6e59e72fb07bd787dc7e90f72b9e1928dae7
                    SOURCE_DIR          "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/cmake/../external/libigl-imgui"
                    BINARY_DIR          "/home/qbeck/Research/Cshells/build/libigl-imgui-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
