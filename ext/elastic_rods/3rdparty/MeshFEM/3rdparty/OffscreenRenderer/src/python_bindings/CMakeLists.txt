include(FetchContent)

# Allow the user's PYTHON_EXECUTABLE environment variable influence the python
# we compile for (without making them pass `-DPYTHON_EXECUTABLE=...` to `cmake`
if (DEFINED ENV{PYTHON_EXECUTABLE})
    set(PYTHON_EXECUTABLE $ENV{PYTHON_EXECUTABLE})
endif()

# Download pybind dependency when pybind11 is missing
if (NOT TARGET pybind11::module)
    message("Offscreen renderer doesn't have pybind11...")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.6.1
    )
    FetchContent_MakeAvailable(pybind11)
    # add_subdirectory(${pybind11_SOURCE_DIR} pybind11_bin)
endif()

pybind11_add_module(_offscreen_renderer offscreen_renderer.cc)
set_target_properties(_offscreen_renderer PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_SOURCE_DIR}/python>)
target_link_libraries(_offscreen_renderer PUBLIC offscreen_renderer)
