#ifndef LINKAGEOPTIMIZATIONBINDING_HH
#define LINKAGEOPTIMIZATIONBINDING_HH

#include <MeshFEM/Geometry.hh>

#include "../SurfaceAttractedLinkage.hh"
#include "../DesignOptimizationTerms.hh"
#include "../TargetSurfaceFitterMesh.hh"

#include "../RegularizationTerms.hh"
#include "../LinkageOptimization.hh"
#include "../XShellOptimization.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
#include <sstream>
namespace py = pybind11;

template<typename Real_>
using SAL_T = SurfaceAttractedLinkage_T<Real_>;
using SAL   = SAL_T<Real>;

template<typename T>
std::string hexString(T val) {
    std::ostringstream ss;
    ss << std::hex << val;
    return ss.str();
}

template<template<typename> class Object>
struct LinkageOptimizationTrampoline : public LinkageOptimization<Object> {
    // Inherit the constructors.
    using LinkageOptimization<Object>::LinkageOptimization;

    Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::VectorXd, // Return type.
            LinkageOptimization<Object>, // Parent class.
            gradp_J, // Name of the function in C++.
            params, opt_eType// Arguments.
        );
    }

    Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, Real coeff_c = 0.0, Real coeff_angle_constraint = 0.0, OptEnergyType opt_eType = OptEnergyType::Full) override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::VectorXd, // Return type.
            LinkageOptimization<Object>, // Parent class.
            apply_hess, // Name of the function in C++.
            params, delta_p, coeff_J, coeff_c, coeff_angle_constraint, opt_eType// Arguments.
        );
    }
    void setLinkageInterleavingType(InterleavingType new_type) override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            setLinkageInterleavingType, // Name of the function in C++.
            new_type// Arguments.
        );
    }
    void commitLinesearchLinkage() override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            commitLinesearchLinkage, // Name of the function in C++.
            // No Arguments.
        );
    }
    void setEquilibriumOptions(const NewtonOptimizerOptions &eopts) override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            setEquilibriumOptions, // Name of the function in C++.
            eopts// Arguments.
        );
    }
    NewtonOptimizerOptions getEquilibriumOptions() const override {
        PYBIND11_OVERRIDE_PURE(
            NewtonOptimizerOptions, // Return type.
            LinkageOptimization<Object>, // Parent class.
            getEquilibriumOptions, // Name of the function in C++.
            // No Arguments.
        );
    }
    void setGamma(Real val) override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            setGamma, // Name of the function in C++.
            val// Arguments.
        );
    }
    Real getGamma() const override {
        PYBIND11_OVERRIDE_PURE(
            Real, // Return type.
            LinkageOptimization<Object>, // Parent class.
            getGamma, // Name of the function in C++.
            // No Arguments.
        );
    }

    void m_forceEquilibriumUpdate() override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            m_forceEquilibriumUpdate, // Name of the function in C++.
            // No Arguments.
        );
    }
    bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &params) override {
        PYBIND11_OVERRIDE_PURE(
            bool, // Return type.
            LinkageOptimization<Object>, // Parent class.
            m_updateEquilibria, // Name of the function in C++.
            params// Arguments.
        );
    }
    void m_updateClosestPoints() override {
        PYBIND11_OVERRIDE_PURE(
            void, // Return type.
            LinkageOptimization<Object>, // Parent class.
            m_updateClosestPoints, // Name of the function in C++.
            // No Arguments.
        );
    }
    bool m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params, const OptEnergyType /*opt_eType=OptEnergyType::Full*/) override {
        PYBIND11_OVERRIDE_PURE(
            bool, // Return type.
            LinkageOptimization<Object>, // Parent class.
            m_updateAdjointState, // Name of the function in C++.
            params// Arguments.
        );
    }

};

template<template<typename> class Object>
void bindLinkageOptimization(py::module &m, const std::string &typestr) {
    using LO = LinkageOptimization<Object>;
    using LTO = LinkageOptimizationTrampoline<Object>;
    std::string pyclass_name = std::string("LinkageOptimization_") + typestr;
    py::class_<LO, LTO>(m, pyclass_name.c_str())
    .def(py::init<Object<Real> &, const NewtonOptimizerOptions &, Real, Real, Real, Real>(), py::arg("baseLinkage"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("E0") = 1.0, py::arg("l0") = 1.0, py::arg("rl0") = 1.0, py::arg("rk0") = 1.0)
    .def("newPt",          &LO::newPt, py::arg("params"))
    .def("params",         &LO::params)
    .def("J",              py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&LO::J),              py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
    .def("J_target",       py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&LO::J_target),       py::arg("params"))
    .def("J_regularization", &LO::J_regularization)
    .def("J_smoothing",      &LO::J_smoothing)
    .def("apply_hess_J",   &LO::apply_hess_J, py::arg("params"), py::arg("delta_p"), py::arg("energyType") = OptEnergyType::Full)
    .def("apply_hess_c",   &LO::apply_hess_J, py::arg("params"), py::arg("delta_p"), py::arg("energyType") = OptEnergyType::Full)
    .def("numParams",       &LO::numParams)
    .def("get_l0",          &LO::get_l0)
    .def("get_rl0",         &LO::get_rl0)
    .def("get_rk0",         &LO::get_rk0)
    .def("get_E0",          &LO::get_E0)
    .def("invalidateAdjointState",     &LO::invalidateAdjointState)
    .def("restKappaSmoothness", &LO::restKappaSmoothness)
    .def_readwrite("prediction_order", &LO::prediction_order)
    .def_property("beta",  &LO::getBeta , &LO::setBeta )
    .def_property("gamma", &LO::getGamma, &LO::setGamma)
    .def_property("rl_regularization_weight", &LO::getRestLengthMinimizationWeight, &LO::setRestLengthMinimizationWeight)
    .def_property("smoothing_weight",         &LO::getRestKappaSmoothingWeight,     &LO::setRestKappaSmoothingWeight)
    .def_readonly("target_surface_fitter",    &LO::target_surface_fitter)
    .def("getTargetSurfaceVertices",          &LO::getTargetSurfaceVertices)
    .def("getTargetSurfaceFaces",             &LO::getTargetSurfaceFaces)
    .def("getTargetSurfaceNormals",           &LO::getTargetSurfaceNormals)
    .def_readwrite("objective", &LO::objective, py::return_value_policy::reference)
    ;
}

template<template<typename> class Object>
void bindXShellOptimization(py::module &m, const std::string &typestr) {
    using XO = XShellOptimization<Object>;
    using LO = LinkageOptimization<Object>;
    std::string pyclass_name = std::string("XShellOptimization_") + typestr;
    py::class_<XO, LO>(m, pyclass_name.c_str())
    .def(py::init<Object<Real> &, Object<Real> &, const NewtonOptimizerOptions &, Real, int, bool, bool, bool>(), py::arg("flat_linkage"), py::arg("deployed_linkage"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, 
         py::arg("allowFlatActuation") = true, py::arg("optimizeTargetAngle") = true, py::arg("fixDeployedVars") = true)
    .def("J",                      py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&XO::J),              py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
    .def("J_target",               py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&XO::J_target),                      py::arg("params"))
    .def("c",                      py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&XO::c),                             py::arg("params"))
    .def("angle_constraint",       py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&XO::angle_constraint),              py::arg("params"))
    .def("gradp_J",                py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, OptEnergyType>(&XO::gradp_J),        py::arg("params"), py::arg("energyType") = OptEnergyType::Full)
    .def("gradp_J_target",         py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&XO::gradp_J_target),                py::arg("params"))
    .def("gradp_angle_constraint", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&XO::gradp_angle_constraint),        py::arg("params"))
    .def("gradp_c",                &XO::gradp_c,        py::arg("params"))
    .def("get_w_x",                &XO::get_w_x)
    .def("get_w_lambda",           &XO::get_w_lambda)
    .def("get_y",                  &XO::get_y)

    .def("get_s_x",            &XO::get_s_x)
    .def("get_delta_x3d",      &XO::get_delta_x3d)
    .def("get_delta_x2d",      &XO::get_delta_x2d)
    .def("get_delta_w_x",      &XO::get_delta_w_x)
    .def("get_delta_w_lambda", &XO::get_delta_w_lambda)
    .def("get_delta_s_x",      &XO::get_delta_s_x)

    .def("pushforward",                  &XO::pushforward, py::arg("params"), py::arg("delta_p"))
    .def("apply_hess",                   &XO::apply_hess, py::arg("params"), py::arg("delta_p"), py::arg("coeff_J") = 1., py::arg("coeff_c") = 0., py::arg("coeff_angle_constraint") = 0., py::arg("energyType") = OptEnergyType::Full)
    .def("setTargetSurface",             &XO::setTargetSurface,           py::arg("V"), py::arg("F"))
    .def("loadTargetSurface",            &XO::loadTargetSurface,          py::arg("path"))
    .def("saveTargetSurface",            &XO::saveTargetSurface,          py::arg("path"))
    .def("scaleJointWeights",            &XO::scaleJointWeights, py::arg("jointPosWeight"), py::arg("featureMultiplier") = 1.0, py::arg("additional_feature_pts") = std::vector<size_t>())
    .def("numFullParams",                &XO::numFullParams)
    .def("getTargetAngle",               &XO::getTargetAngle)
    .def("setTargetAngle",               &XO::setTargetAngle, py::arg("alpha_t"))
    .def("getEquilibriumOptions",         &XO::getEquilibriumOptions)
    .def("getDeploymentOptions",          &XO::getDeploymentOptions)
    .def("getFixedFlatVars",              &XO::getFixedFlatVars)
    .def("getFixedDeployedVars",          &XO::getFixedDeployedVars)
    .def("getLinesearchBaseLinkage",      &XO::getLinesearchBaseLinkage,     py::return_value_policy::reference)
    .def("getLinesearchDeployedLinkage",  &XO::getLinesearchDeployedLinkage, py::return_value_policy::reference)
    .def("getLinesearchDesignParameters", &XO::getLinesearchDesignParameters)
    .def("getFullDesignParameters",       &XO::getFullDesignParameters)
    .def("getOptimizeTargetAngle",        &XO::getOptimizeTargetAngle)
    .def("setOptimizeTargetAngle",        &XO::setOptimizeTargetAngle,    py::arg("optimize"))
    .def("setHoldClosestPointsFixed",     &XO::setHoldClosestPointsFixed, py::arg("hold"))
    .def("getTargetJointsPosition",       &XO::getTargetJointsPosition)
    .def("setTargetJointsPosition",       &XO::setTargetJointsPosition, py::arg("jointPosition"))
    .def("getEpsMinAngleConstraint",      &XO::getEpsMinAngleConstraint)
    .def("constructTargetSurface", &XO::constructTargetSurface, py::arg("loop_subdivisions"), py::arg("num_extension_layers"), py::arg("scale_factors"))
    .def("XShellOptimize", &XO::optimize, py::arg("alg"), py::arg("num_steps"), py::arg("trust_region_scale"), py::arg("optimality_tol"), py::arg("update_viewer"), py::arg("minRestLen") = -1, py::arg("applyAngleConstraint") = true, py::arg("applyFlatnessConstraint") = true)
    ;
}

#endif /* end of include guard: LINKAGEOPTIMIZATIONBINDING_HH */
