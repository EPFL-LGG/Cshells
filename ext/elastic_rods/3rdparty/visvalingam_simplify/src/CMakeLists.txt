add_library(VisvalingamSimplify visvalingam_algorithm.cpp)
target_include_directories(VisvalingamSimplify PUBLIC .)
target_compile_features(${PROJECT_NAME} PUBLIC ${CXX11_FEATURES})

set_target_properties(VisvalingamSimplify PROPERTIES CXX_STANDARD 11)
set_target_properties(VisvalingamSimplify PROPERTIES CXX_STANDARD_REQUIRED ON)
