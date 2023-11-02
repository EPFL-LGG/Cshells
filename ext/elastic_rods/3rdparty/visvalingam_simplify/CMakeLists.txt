cmake_minimum_required(VERSION 3.1)
project(VisvalingamSimplify)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

add_subdirectory(src)
