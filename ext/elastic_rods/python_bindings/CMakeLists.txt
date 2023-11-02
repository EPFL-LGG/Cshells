pybind11_add_module(elastic_rods elastic_rods.cc)
set_target_properties(elastic_rods PROPERTIES CXX_STANDARD 14)
set_target_properties(elastic_rods PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(elastic_rods PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
target_link_libraries(elastic_rods PUBLIC ElasticRods RodLinkages intel_pybind_14_hack)

pybind11_add_module(vector_operations vector_operations.cc)
set_target_properties(vector_operations PROPERTIES CXX_STANDARD 14)
set_target_properties(vector_operations PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(vector_operations PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
target_link_libraries(vector_operations PUBLIC MeshFEM intel_pybind_14_hack)

if(Knitro_FOUND)
    pybind11_add_module(linkage_optimization linkage_optimization.cc)
    set_target_properties(linkage_optimization PROPERTIES CXX_STANDARD 14)
    set_target_properties(linkage_optimization PROPERTIES CXX_STANDARD_REQUIRED ON)
    set_target_properties(linkage_optimization PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
    target_link_libraries(linkage_optimization PUBLIC ElasticRods RodLinkages intel_pybind_14_hack)
endif()

# Create a `ElasticRods` python module that just sets the correct paths for
# importing ElasticRods's individual pybind11 modules
set(PYTHON_DEPLOY_DIR "${CMAKE_SOURCE_DIR}/python/ElasticRods")
file(MAKE_DIRECTORY "${PYTHON_DEPLOY_DIR}")
configure_file(${PROJECT_SOURCE_DIR}/python/init_template.py "${PYTHON_DEPLOY_DIR}/__init__.py")
