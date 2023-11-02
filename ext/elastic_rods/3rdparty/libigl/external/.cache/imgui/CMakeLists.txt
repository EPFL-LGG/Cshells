# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(imgui-download NONE)

include(ExternalProject)
ExternalProject_Add(imgui-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/ocornut/imgui.git;GIT_TAG;bc6ac8b2aee0614debd940e45bc9cd0d9b355c86
                    SOURCE_DIR          "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/cmake/../external/imgui"
                    BINARY_DIR          "/home/qbeck/Research/Cshells/build/imgui-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
