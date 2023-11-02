#include "linkage_optimization.hh"
PYBIND11_MODULE(linkage_optimization, m) {
    py::module::import("MeshFEM");
    py::module::import("py_newton_optimizer");
    py::module::import("elastic_rods");
    m.doc() = "Linkage Optimization Codebase";

    py::module detail_module = m.def_submodule("detail");

    py::enum_<OptEnergyType>(m, "OptEnergyType")
        .value("Full",           OptEnergyType::Full           )
        .value("ElasticBase",    OptEnergyType::ElasticBase    )
        .value("ElasticDeployed",OptEnergyType::ElasticDeployed)
        .value("Target",         OptEnergyType::Target         )
        .value("Regularization", OptEnergyType::Regularization )
        .value("Smoothing",      OptEnergyType::Smoothing      )
        ;

    using TSF = TargetSurfaceFitter;
    py::class_<TSF>(m, "TargetSurfaceFitter")
        .def(py::init<>())
        .def("saveTargetSurface", &TSF::saveTargetSurface, py::arg("surf_path"))
        .def("loadTargetSurface", &TSF::loadTargetSurface, py::arg("linkage"), py::arg("surf_path"))
        .def("objective",       &TSF::objective,             py::arg("linkage"))
        .def("gradient",        &TSF::gradient,              py::arg("linkage"))
        .def("numSamplePoints", &TSF::numSamplePoints<Real>, py::arg("linkage"))

        .def("getUseCenterline", &TSF::getUseCenterline)
        .def("setUseCenterline", &TSF::setUseCenterline<Real>,    py::arg("linkage"), py::arg("useCenterline"), py::arg("jointPosWeight"), py::arg("jointPosValence2Multiplier") = 1.0)
        .def("setTargetJointPosVsTargetSurfaceTradeoff", &TSF::setTargetJointPosVsTargetSurfaceTradeoff<Real>, py::arg("linkage"), py::arg("jointPosWeight"), py::arg("valence2Multiplier") = 1.0)
        .def("scaleJointWeights", &TSF::scaleJointWeights<Real>,  py::arg("linkage"), py::arg("jointPosWeight"), py::arg("valence2Multiplier") = 1.0, py::arg("additional_feature_pts") = std::vector<size_t>())
        .def("scaleFeatureJointWeights", &TSF::scaleFeatureJointWeights<Real>, py::arg("linkage"), py::arg("jointPosWeight"), py::arg("featureMultiplier") = 1.0, py::arg("feature_pts") = std::vector<size_t>())
        .def_readwrite("holdClosestPointsFixed", &TSF::holdClosestPointsFixed)

        .def_readonly("W_diag_joint_pos",                      &TSF::W_diag_joint_pos)
        .def_readonly("Wsurf_diag_linkage_sample_pos",         &TSF::Wsurf_diag_linkage_sample_pos)
        .def_readonly("joint_pos_tgt",                         &TSF::joint_pos_tgt)

        .def_property_readonly("V", [](const TSF &tsf) { return tsf.getV(); })
        .def_property_readonly("F", [](const TSF &tsf) { return tsf.getF(); })
        .def_property_readonly("N", [](const TSF &tsf) { return tsf.getN(); })
        .def_readonly("linkage_closest_surf_pts",              &TSF::linkage_closest_surf_pts)
        .def_readonly("linkage_closest_surf_pt_sensitivities", &TSF::linkage_closest_surf_pt_sensitivities)
        .def_readonly("linkage_closest_surf_tris",             &TSF::linkage_closest_surf_tris)
        .def_readonly("holdClosestPointsFixed",                &TSF::holdClosestPointsFixed)
        ;

    using RT_SAL = RegularizationTerm<SAL>;
    py::class_<RT_SAL, std::shared_ptr<RT_SAL>>(m, "RegularizationTerm_SAL")
        .def("energy", &RT_SAL::energy)
        .def_readwrite("weight", &RT_SAL::weight)
        ;

    using RT_RL = RegularizationTerm<RodLinkage>;
    py::class_<RT_RL, std::shared_ptr<RT_RL>>(m, "RegularizationTerm_RL")
        .def("energy", &RT_RL::energy)
        .def_readwrite("weight", &RT_RL::weight)
        ;

    using RCS_SAL = RestCurvatureSmoothing<SAL>;
    py::class_<RCS_SAL, RT_SAL, std::shared_ptr<RCS_SAL>>(m, "RestCurvatureSmoothing_SAL")
        .def(py::init<const SAL &>(), py::arg("linkage"))
        ;
    
    using RCS_RL = RestCurvatureSmoothing<RodLinkage>;
    py::class_<RCS_RL, RT_RL, std::shared_ptr<RCS_RL>>(m, "RestCurvatureSmoothing_RL")
        .def(py::init<const RodLinkage &>(), py::arg("linkage"))
        ;

    using RLM_SAL = RestLengthMinimization<SAL>;
    py::class_<RLM_SAL, RT_SAL, std::shared_ptr<RLM_SAL>>(m, "RestLengthMinimization_SAL")
        .def(py::init<const SAL &>(), py::arg("linkage"))
        ;

    using RLM_RL = RestLengthMinimization<RodLinkage>;
    py::class_<RLM_RL, RT_RL, std::shared_ptr<RLM_RL>>(m, "RestLengthMinimization_RL")
        .def(py::init<const RodLinkage &>(), py::arg("linkage"))
        ;

    using DOT_SAL = DesignOptimizationTerm<SAL_T>;
    py::class_<DOT_SAL, std::shared_ptr<DOT_SAL>>(m, "DesignOptimizationTerm_SAL")
        .def("value",  &DOT_SAL::value)
        .def("update", &DOT_SAL::update)
        .def("grad"  , &DOT_SAL::grad  )
        .def("grad_x", &DOT_SAL::grad_x)
        .def("grad_p", &DOT_SAL::grad_p)
        .def("computeGrad",      &DOT_SAL::computeGrad)
        .def("computeDeltaGrad", &DOT_SAL::computeDeltaGrad, py::arg("delta_xp"))
        ;

    using DOT_RL = DesignOptimizationTerm<RodLinkage_T>;
    py::class_<DOT_RL, std::shared_ptr<DOT_RL>>(m, "DesignOptimizationTerm_RL")
        .def("value",  &DOT_RL::value)
        .def("update", &DOT_RL::update)
        .def("grad"  , &DOT_RL::grad  )
        .def("grad_x", &DOT_RL::grad_x)
        .def("grad_p", &DOT_RL::grad_p)
        .def("computeGrad",      &DOT_RL::computeGrad)
        .def("computeDeltaGrad", &DOT_RL::computeDeltaGrad, py::arg("delta_xp"))
        ;

    using DOOT_SAL = DesignOptimizationObjectiveTerm<SAL_T>;
    py::class_<DOOT_SAL, DOT_SAL, std::shared_ptr<DOOT_SAL>>(m, "DesignOptimizationObjectiveTerm_SAL")
        .def_readwrite("weight", &DOOT_SAL::weight)
        ;

    using DOOT_RL = DesignOptimizationObjectiveTerm<RodLinkage_T>;
    py::class_<DOOT_RL, DOT_RL, std::shared_ptr<DOOT_RL>>(m, "DesignOptimizationObjectiveTerm_RL")
        .def_readwrite("weight", &DOOT_RL::weight)
        ;

    using EEO_SAL = ElasticEnergyObjective<SAL_T>;
    py::class_<EEO_SAL, DOOT_SAL, std::shared_ptr<EEO_SAL>>(m, "ElasticEnergyObjective_SAL")
        .def(py::init<const SAL &>(), py::arg("surface_attracted_linkage"))
        .def_property("useEnvelopeTheorem", &EEO_SAL::useEnvelopeTheorem, &EEO_SAL::setUseEnvelopeTheorem)
        ;

    using EEO_RL = ElasticEnergyObjective<RodLinkage_T>;
    py::class_<EEO_RL, DOOT_RL, std::shared_ptr<EEO_RL>>(m, "ElasticEnergyObjective_RL")
        .def(py::init<const RodLinkage &>(), py::arg("rod_linkage"))
        .def_property("useEnvelopeTheorem", &EEO_RL::useEnvelopeTheorem, &EEO_RL::setUseEnvelopeTheorem)
        ;

    using TFO_SAL = TargetFittingDOOT<SAL_T>;
    py::class_<TFO_SAL, DOOT_SAL, std::shared_ptr<TFO_SAL>>(m, "TargetFittingDOOT_SAL")
        .def(py::init<const SAL &, TargetSurfaceFitter &>(), py::arg("surface_attracted_linkage"), py::arg("targetSurfaceFitter"))
        ;

    using TFO_RL = TargetFittingDOOT<RodLinkage_T>;
    py::class_<TFO_RL, DOOT_RL, std::shared_ptr<TFO_RL>>(m, "TargetFittingDOOT_RL")
        .def(py::init<const RodLinkage &, TargetSurfaceFitter &>(), py::arg("rod_linkage"), py::arg("targetSurfaceFitter"))
        ;

    using RCSD_SAL = RegularizationTermDOOWrapper<SAL_T, RestCurvatureSmoothing>; //Might be wrong
    py::class_<RCSD_SAL, DOT_SAL, std::shared_ptr<RCSD_SAL>>(m, "RestCurvatureSmoothingDOOT_SAL")
        .def(py::init<const SAL &>(),          py::arg("surface_attracted_linkage"))
        .def(py::init<std::shared_ptr<RCS_SAL>>(), py::arg("restCurvatureRegTerm"))
        .def_property("weight", [](const RCSD_SAL &r) { return r.weight; }, [](RCSD_SAL &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using RCSD_RL = RegularizationTermDOOWrapper<RodLinkage_T, RestCurvatureSmoothing>; //Might be wrong
    py::class_<RCSD_RL, DOT_RL, std::shared_ptr<RCSD_RL>>(m, "RestCurvatureSmoothingDOOT_RL")
        .def(py::init<const RodLinkage &>(),          py::arg("rod_linkage"))
        .def(py::init<std::shared_ptr<RCS_RL>>(), py::arg("restCurvatureRegTerm"))
        .def_property("weight", [](const RCSD_RL &r) { return r.weight; }, [](RCSD_RL &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using RLMD_SAL = RegularizationTermDOOWrapper<SAL_T, RestLengthMinimization>; //Might be wrong
    py::class_<RLMD_SAL, DOT_SAL, std::shared_ptr<RLMD_SAL>>(m, "RestLengthMinimizationDOOT_SAL")
        .def(py::init<const SAL &>(),          py::arg("surface_attracted_linkage"))
        .def(py::init<std::shared_ptr<RLM_SAL>>(), py::arg("restLengthMinimizationTerm"))
        .def_property("weight", [](const RLMD_SAL &r) { return r.weight; }, [](RLMD_SAL &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using RLMD_RL = RegularizationTermDOOWrapper<RodLinkage_T, RestLengthMinimization>; //Might be wrong
    py::class_<RLMD_RL, DOT_RL, std::shared_ptr<RLMD_RL>>(m, "RestLengthMinimizationDOOT_RL")
        .def(py::init<const RodLinkage &>(),          py::arg("rod_linkage"))
        .def(py::init<std::shared_ptr<RLM_RL>>(), py::arg("restLengthMinimizationTerm"))
        .def_property("weight", [](const RLMD_RL &r) { return r.weight; }, [](RLMD_RL &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using OEType = OptEnergyType;
    using DOO_SAL  = DesignOptimizationObjective<SAL_T, OEType>;
    using TR_SAL   = DOO_SAL::TermRecord;
    using TPtr_SAL = DOO_SAL::TermPtr;
    py::class_<DOO_SAL> doo_SAL(m, "DesignOptimizationObjective_SAL");

    py::class_<TR_SAL>(doo_SAL, "DesignOptimizationObjectiveTermRecord_SAL")
        .def(py::init<const std::string &, OEType, std::shared_ptr<DOT_SAL>>(),  py::arg("name"), py::arg("type"), py::arg("term"))
        .def_readwrite("name", &TR_SAL::name)
        .def_readwrite("type", &TR_SAL::type)
        .def_readwrite("term", &TR_SAL::term)
        .def("__repr__", [](const TR_SAL *trp) {
                const auto &tr = *trp;
                return "TermRecord " + tr.name + " at " + hexString(trp) + " with weight " + std::to_string(tr.term->getWeight()) + " and value " + std::to_string(tr.term->unweightedValue());
        })
        ;

    doo_SAL.def(py::init<>())
       .def("update",         &DOO_SAL::update)
       .def("grad",           &DOO_SAL::grad, py::arg("type") = OEType::Full)
       .def("values",         &DOO_SAL::values)
       .def("weightedValues", &DOO_SAL::weightedValues)
       .def("value",  py::overload_cast<OEType>(&DOO_SAL::value, py::const_), py::arg("type") = OEType::Full)
       .def("computeGrad",      &DOO_SAL::computeGrad, py::arg("type") = OEType::Full)
       .def("computeDeltaGrad", &DOO_SAL::computeDeltaGrad, py::arg("delta_xp"), py::arg("type") = OEType::Full)
       .def_readwrite("terms",  &DOO_SAL::terms, py::return_value_policy::reference)
       .def("add", py::overload_cast<const std::string &, OEType, TPtr_SAL      >(&DOO_SAL::add),  py::arg("name"), py::arg("type"), py::arg("term"))
       .def("add", py::overload_cast<const std::string &, OEType, TPtr_SAL, Real>(&DOO_SAL::add),  py::arg("name"), py::arg("type"), py::arg("term"), py::arg("weight"))
       // More convenient interface for adding multiple terms at once
       .def("add", [](DOO_SAL &o, const std::list<std::tuple<std::string, OEType, TPtr_SAL>> &terms) {
                    for (const auto &t : terms)
                        o.add(std::get<0>(t), std::get<1>(t), std::get<2>(t));
               })
       ;

    using DOO_RL = DesignOptimizationObjective<RodLinkage_T, OEType>;
    using TR_RL   = DOO_RL::TermRecord;
    using TPtr_RL = DOO_RL::TermPtr;
    py::class_<DOO_RL> doo_RL(m, "DesignOptimizationObjective_RL");

    py::class_<TR_RL>(doo_RL, "DesignOptimizationObjectiveTermRecord_RL")
        .def(py::init<const std::string &, OEType, std::shared_ptr<DOT_RL>>(),  py::arg("name"), py::arg("type"), py::arg("term"))
        .def_readwrite("name", &TR_RL::name)
        .def_readwrite("type", &TR_RL::type)
        .def_readwrite("term", &TR_RL::term)
        .def("__repr__", [](const TR_RL *trp) {
                const auto &tr = *trp;
                return "TermRecord " + tr.name + " at " + hexString(trp) + " with weight " + std::to_string(tr.term->getWeight()) + " and value " + std::to_string(tr.term->unweightedValue());
        })
        ;

    doo_RL.def(py::init<>())
       .def("update",         &DOO_RL::update)
       .def("grad",           &DOO_RL::grad, py::arg("type") = OEType::Full)
       .def("values",         &DOO_RL::values)
       .def("weightedValues", &DOO_RL::weightedValues)
       .def("value",  py::overload_cast<OEType>(&DOO_RL::value, py::const_), py::arg("type") = OEType::Full)
        .def("computeGrad",     &DOO_RL::computeGrad, py::arg("type") = OEType::Full)
       .def("computeDeltaGrad", &DOO_RL::computeDeltaGrad, py::arg("delta_xp"), py::arg("type") = OEType::Full)
       .def_readwrite("terms",  &DOO_RL::terms, py::return_value_policy::reference)
       .def("add", py::overload_cast<const std::string &, OEType, TPtr_RL      >(&DOO_RL::add),  py::arg("name"), py::arg("type"), py::arg("term"))
       .def("add", py::overload_cast<const std::string &, OEType, TPtr_RL, Real>(&DOO_RL::add),  py::arg("name"), py::arg("type"), py::arg("term"), py::arg("weight"))
       // More convenient interface for adding multiple terms at once
       .def("add", [](DOO_RL &o, const std::list<std::tuple<std::string, OEType, TPtr_RL>> &terms) {
                    for (const auto &t : terms)
                        o.add(std::get<0>(t), std::get<1>(t), std::get<2>(t));
               })
       ;

    ////////////////////////////////////////////////////////////////////////////////
    // Linkage Optimization Base Class
    ////////////////////////////////////////////////////////////////////////////////
    py::enum_<OptAlgorithm>(m, "OptAlgorithm")
        .value("NEWTON_CG", OptAlgorithm::NEWTON_CG)
        .value("BFGS",      OptAlgorithm::BFGS     )
        ;


    bindLinkageOptimization<RodLinkage_T>(detail_module,       "RodLinkage");
    bindLinkageOptimization<SAL_T>(detail_module, "SurfaceAttractedLinkage");
    ////////////////////////////////////////////////////////////////////////////////
    // XShell Optimization
    ////////////////////////////////////////////////////////////////////////////////
    bindXShellOptimization<RodLinkage_T>(detail_module,       "RodLinkage");
    bindXShellOptimization<SAL_T>(detail_module, "SurfaceAttractedLinkage");
    m.def("XShellOptimization", [](RodLinkage &flat, RodLinkage &deployed, const NewtonOptimizerOptions &eopts, Real minAngleConstraint, int pinJoint, bool allowFlatActuation, bool optimizeTargetAngle, bool fixDeployedVars) {
          // Uncomment this part once XShell Optimization is tested with Surface Attraction Linkage.
          // // Note: while py::cast is not yet documented in the official documentation,
          // // it accepts the return_value_policy as discussed in:
          // //      https://github.com/pybind/pybind11/issues/1201
          // // by setting the return value policy to take_ownership, we can avoid
          // // memory leaks and double frees regardless of the holder type for XShellOptimization_*.
          // try {
          //   auto &sl = dynamic_cast<SurfaceAttractedLinkage &>(linkage);
          //   return py::cast(new XShellOptimization<SAL_T>(sl, input_surface_path, useCenterline, equilibrium_options, pinJoint, useFixedJoint, fixedVars), py::return_value_policy::take_ownership);
          // }
          // catch (...) {
          return py::cast(new XShellOptimization<RodLinkage_T>(flat, deployed, eopts, minAngleConstraint, pinJoint, allowFlatActuation, optimizeTargetAngle, fixDeployedVars), py::return_value_policy::take_ownership);
          // }
    },  py::arg("flat"), py::arg("deployed"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true, py::arg("optimizeTargetAngle") = true, py::arg("fixDeployedVars") = true);

    ////////////////////////////////////////////////////////////////////////////////
    // Benchmarking
    ////////////////////////////////////////////////////////////////////////////////
    m.def("benchmark_reset", &BENCHMARK_RESET);
    m.def("benchmark_start_timer_section", &BENCHMARK_START_TIMER_SECTION, py::arg("name"));
    m.def("benchmark_stop_timer_section",  &BENCHMARK_STOP_TIMER_SECTION,  py::arg("name"));
    m.def("benchmark_start_timer",         &BENCHMARK_START_TIMER,         py::arg("name"));
    m.def("benchmark_stop_timer",          &BENCHMARK_STOP_TIMER,          py::arg("name"));
    m.def("benchmark_report", [](bool includeMessages) {
            // py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
            if (includeMessages) BENCHMARK_REPORT(); else BENCHMARK_REPORT_NO_MESSAGES();
        },
        py::arg("include_messages") = false)
        ;
}
