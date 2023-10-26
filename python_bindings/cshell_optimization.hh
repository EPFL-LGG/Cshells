#include "../ext/elastic_rods/python_bindings/linkage_optimization.hh"
#include "../src/AverageAngleCShellOptimization.hh"
#include "../src/AverageAngleLinkage.hh"
#include "../src/AverageAngleSurfaceAttractedLinkage.hh"

template<typename Real_>
using AASAL_T = AverageAngleSurfaceAttractedLinkage_T<Real_>;
using AASAL   = AASAL_T<Real>;

template<template<typename> class Object>
void bindAverageAngleCShellOptimization(py::module &m, const std::string &typestr) {
    using AACO = AverageAngleCShellOptimization<Object>;
    using LO  = LinkageOptimization<Object>;
    std::string pyclass_name = std::string("AverageAngleCShellOptimization_") + typestr;
    py::class_<AACO, LO>(m, pyclass_name.c_str())
    .def(py::init<Object<Real> &, Object<Real> &, const NewtonOptimizerOptions &, Real, int, bool, bool, bool, const std::vector<size_t> &, const std::vector<size_t> &>(), py::arg("flat_linkage"), py::arg("deployed_linkage"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, 
         py::arg("allowFlatActuation") = true, py::arg("optimizeTargetAngle") = true, py::arg("fixDeployedVars") = true, py::arg("additionalFixedFlatVars") = std::vector<size_t>(), py::arg("additionalFixedDeployedVars") = std::vector<size_t>())
    .def("J",                        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&AACO::J),              py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
    .def("J_target",                 py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&AACO::J_target),                      py::arg("params"))
    .def("c",                        py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&AACO::c),                             py::arg("params"))
    .def("angle_constraint",         py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&AACO::angle_constraint),              py::arg("params"))
    .def("gradp_J",                  py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&AACO::gradp_J),        py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
    .def("gradp_J_target",           py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&AACO::gradp_J_target),                py::arg("params"))
    .def("gradp_c",                  py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&AACO::gradp_c),                       py::arg("params"))
    .def("gradp_angle_constraint",   py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&AACO::gradp_angle_constraint),        py::arg("params"))
    
    .def("get_w_x",        &AACO::get_w_x)
    .def("get_w_lambda",   &AACO::get_w_lambda)
    .def("get_delta_x2d",  &AACO::get_delta_x2d)
    .def("get_delta_x3d",  &AACO::get_delta_x3d)
    .def("get_delta_w_x",  &AACO::get_delta_w_x)
    .def("get_delta_w_lambda", &AACO::get_delta_w_lambda)

    .def("get_l0",  &AACO::get_l0)
    .def("get_rl0", &AACO::get_rl0)
    .def("get_E0",  &AACO::get_E0)
    .def("set_l0",  &AACO::set_l0,  py::arg("l0"))
    .def("set_rl0", &AACO::set_rl0, py::arg("rl0"))
    .def("set_E0",  &AACO::set_E0,  py::arg("E0"))

    .def("pushforward", &AACO::pushforward, py::arg("params"), py::arg("delta_p"))
    .def("setTargetSurface",              &AACO::setTargetSurface,           py::arg("V"), py::arg("F"))
    .def("loadTargetSurface",             &AACO::loadTargetSurface,          py::arg("path"))
    .def("saveTargetSurface",             &AACO::saveTargetSurface,          py::arg("path"))
    .def("scaleJointWeights",             &AACO::scaleJointWeights, py::arg("jointPosWeight"), py::arg("featureMultiplier") = 1.0, py::arg("additional_feature_pts") = std::vector<size_t>())
    .def("scaleFeatureJointWeights",      &AACO::scaleFeatureJointWeights, py::arg("jointPosWeight"), py::arg("featureMultiplier") = 1.0, py::arg("feature_pts") = std::vector<size_t>())
    .def("apply_hess",                    &AACO::apply_hess, py::arg("params"), py::arg("delta_p"), py::arg("coeff_J") = 1., py::arg("coeff_c") = 0., py::arg("coeff_angle_constraint") = 0., py::arg("energyType") = OptEnergyType::Full)
    .def("numFullParams",                 &AACO::numFullParams)
    .def("getTargetAngle",                &AACO::getTargetAngle)
    .def("setTargetAngle",                &AACO::setTargetAngle, py::arg("alpha_t"))
    .def("getEquilibriumOptions",         &AACO::getEquilibriumOptions)
    .def("setEquilibriumOptions",         &AACO::setEquilibriumOptions, py::arg("options"))
    .def("getDeploymentOptions",          &AACO::getDeploymentOptions)
    .def("getFixedFlatVars",              &AACO::getFixedFlatVars)
    .def("getFixedDeployedVars",          &AACO::getFixedDeployedVars)
    .def("getBaseLinkage",                &AACO::getBaseLinkage,               py::return_value_policy::reference)
    .def("getLinesearchBaseLinkage",      &AACO::getLinesearchBaseLinkage,     py::return_value_policy::reference)
    .def("getDeployedLinkage",            &AACO::getDeployedLinkage,           py::return_value_policy::reference)
    .def("getLinesearchDeployedLinkage",  &AACO::getLinesearchDeployedLinkage, py::return_value_policy::reference)
    .def("getLinesearchDesignParameters", &AACO::getLinesearchDesignParameters)
    .def("getFullDesignParameters",       &AACO::getFullDesignParameters)
    .def("getOptimizeTargetAngle",        &AACO::getOptimizeTargetAngle)
    .def("setOptimizeTargetAngle",        &AACO::setOptimizeTargetAngle,    py::arg("optimize"))
    .def("setHoldClosestPointsFixed",     &AACO::setHoldClosestPointsFixed, py::arg("hold"))
    .def("get_target_surface_fitter",     &AACO::get_target_surface_fitter)
    .def("reflectTargetSurface",          &AACO::reflectTargetSurface, py::arg("jointID"))
    .def("getTargetJointsPosition",       &AACO::getTargetJointsPosition)
    .def("setTargetJointsPosition",       &AACO::setTargetJointsPosition, py::arg("jointPosition"))
    .def("getEpsMinAngleConstraint",      &AACO::getEpsMinAngleConstraint)
    .def("constructTargetSurface",        &AACO::constructTargetSurface, py::arg("loop_subdivisions"), py::arg("num_extension_layers"), py::arg("scale_factors"))
    .def("CShellOptimize", &AACO::optimize, py::arg("alg"), py::arg("num_steps"), py::arg("trust_region_scale"), py::arg("optimality_tol"), py::arg("update_viewer"), py::arg("minRestLen") = -1, py::arg("applyAngleConstraint") = true, py::arg("applyFlatnessConstraint") = true)
    ;
}
