if(KNITRO_FOUND)
    # Add your project files
    file(GLOB SRCFILES *.cpp)
    add_executable(xshell_editor xshell_editor.cc)
    target_link_libraries(xshell_editor RodLinkages igl::core igl::opengl_glfw igl::opengl_glfw_imgui)
    target_include_directories(xshell_editor PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/LBFGSpp/include)
    set_target_properties(xshell_editor PROPERTIES CXX_STANDARD 14)
    set_target_properties(xshell_editor PROPERTIES CXX_STANDARD_REQUIRED ON)
else()
    message(STATUS "LibIGL not found; xshell editor won't be built")
endif()
