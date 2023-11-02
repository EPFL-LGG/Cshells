# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 3.1)

project(eigen-download NONE)

include(ExternalProject)
ExternalProject_Add(eigen-download
                    GIT_CONFIG advice.detachedHead=false;URL;https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz;URL_MD5;9e30f67e8531477de4117506fe44669b
                    SOURCE_DIR          "/home/qbeck/Research/Cshells/ext/elastic_rods/3rdparty/MeshFEM/cmake/../3rdparty/eigen"
                    BINARY_DIR          "/home/qbeck/Research/Cshells/build/eigen-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
