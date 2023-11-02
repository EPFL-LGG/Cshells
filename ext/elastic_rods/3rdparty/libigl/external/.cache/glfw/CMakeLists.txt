# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(glfw-download NONE)

include(ExternalProject)
ExternalProject_Add(glfw-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/glfw/glfw.git;GIT_TAG;53c8c72c676ca97c10aedfe3d0eb4271c5b23dba
                    SOURCE_DIR          "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/libigl/cmake/../external/glfw"
                    BINARY_DIR          "/home/qbeck/Research/Cshells/build/glfw-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
