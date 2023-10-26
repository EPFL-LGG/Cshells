#include "cshell_optimization.hh"

PYBIND11_MODULE(cshell_optimization, m) {
    py::module::import("MeshFEM");
    py::module::import("py_newton_optimizer");
    py::module::import("elastic_rods");
    py::module::import("linkage_optimization");
    m.doc() = "CShell Optimization Codebase";

    py::module detail_module = m.def_submodule("detail");

    bindLinkageOptimization<AverageAngleLinkage_T>(detail_module,   "AverageAngleLinkage");
    bindLinkageOptimization<AASAL_T>(detail_module, "AverageAngleSurfaceAttractedLinkage");

    ////////////////////////////////////////////////////////////////////////////////
    // Average Angle CShell Optimization
    ////////////////////////////////////////////////////////////////////////////////

    bindAverageAngleCShellOptimization<AverageAngleLinkage_T>(detail_module,   "AverageAngleLinkage");
    bindAverageAngleCShellOptimization<AASAL_T>(detail_module, "AverageAngleSurfaceAttractedLinkage");
    m.def("AverageAngleCShellOptimization", [](AverageAngleLinkage &flat, AverageAngleLinkage &deployed, const NewtonOptimizerOptions &eopts, Real minAngleConstraint, int pinJoint, bool allowFlatActuation, bool optimizeTargetAngle, bool fixDeployedVars, const std::vector<size_t> &additionalFixedFlatVars, const std::vector<size_t> &additionalFixedDeployedVars) {
        return py::cast(new AverageAngleCShellOptimization<AverageAngleLinkage_T>(flat, deployed, eopts, minAngleConstraint, pinJoint, allowFlatActuation, optimizeTargetAngle, fixDeployedVars, additionalFixedFlatVars, additionalFixedDeployedVars), py::return_value_policy::take_ownership);
    },  py::arg("flat"), py::arg("deployed"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true, py::arg("optimizeTargetAngle") = true, py::arg("fixDeployedVars") = true, py::arg("additionalFixedFlatVars") = std::vector<size_t>(), py::arg("additionalFixedDeployedVars") = std::vector<size_t>());
    m.def("AverageAngleCShellOptimizationSAL", [](AASAL &flat, AASAL &deployed, const NewtonOptimizerOptions &eopts, Real minAngleConstraint, int pinJoint, bool allowFlatActuation, bool optimizeTargetAngle, bool fixDeployedVars, const std::vector<size_t> &additionalFixedFlatVars, const std::vector<size_t> &additionalFixedDeployedVars) {
        return py::cast(new AverageAngleCShellOptimization<AASAL_T>(flat, deployed, eopts, minAngleConstraint, pinJoint, allowFlatActuation, optimizeTargetAngle, fixDeployedVars, additionalFixedFlatVars, additionalFixedDeployedVars), py::return_value_policy::take_ownership);
    },  py::arg("flat"), py::arg("deployed"), py::arg("equilibrium_options") = NewtonOptimizerOptions(), py::arg("minAngleConstraint") = 0, py::arg("pinJoint") = -1, py::arg("allowFlatActuation") = true, py::arg("optimizeTargetAngle") = true, py::arg("fixDeployedVars") = true, py::arg("additionalFixedFlatVars") = std::vector<size_t>(), py::arg("additionalFixedDeployedVars") = std::vector<size_t>());

    ////////////////////////////////////////////////////////////////////////////////
    // Benchmarking
    ////////////////////////////////////////////////////////////////////////////////
    m.def("benchmark_reset", &BENCHMARK_RESET);
    m.def("benchmark_start_timer_section", &BENCHMARK_START_TIMER_SECTION, py::arg("name"));
    m.def("benchmark_stop_timer_section",  &BENCHMARK_STOP_TIMER_SECTION,  py::arg("name"));
    m.def("benchmark_start_timer",         &BENCHMARK_START_TIMER,         py::arg("name"));
    m.def("benchmark_stop_timer",          &BENCHMARK_STOP_TIMER,          py::arg("name"));
    m.def("benchmark_report", [](bool includeMessages) {
            py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
            if (includeMessages) BENCHMARK_REPORT(); else BENCHMARK_REPORT_NO_MESSAGES();
        },
        py::arg("include_messages") = false)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Bind some terms (see linkage_optimization.cc in the elastic_rods repo)
    ////////////////////////////////////////////////////////////////////////////////

    using RT_AASAL = RegularizationTerm<AASAL>;
    py::class_<RT_AASAL, std::shared_ptr<RT_AASAL>>(m, "RegularizationTerm_AASAL")
        .def("energy", &RT_AASAL::energy)
        .def_readwrite("weight", &RT_AASAL::weight)
        ;

    using RT_AA = RegularizationTerm<AverageAngleLinkage>;
    py::class_<RT_AA, std::shared_ptr<RT_AA>>(m, "RegularizationTerm_AA")
        .def("energy", &RT_AA::energy)
        .def_readwrite("weight", &RT_AA::weight)
        ;

    using RCS_AASAL = RestCurvatureSmoothing<AASAL>;
    py::class_<RCS_AASAL, RT_AASAL, std::shared_ptr<RCS_AASAL>>(m, "RestCurvatureSmoothing_AASAL")
        .def(py::init<const AASAL &>(), py::arg("linkage"))
        ;
    
    using RCS_AA = RestCurvatureSmoothing<AverageAngleLinkage>;
    py::class_<RCS_AA, RT_AA, std::shared_ptr<RCS_AA>>(m, "RestCurvatureSmoothing_AA")
        .def(py::init<const AverageAngleLinkage &>(), py::arg("linkage"))
        ;

    using RLM_AASAL = RestLengthMinimization<AASAL>;
    py::class_<RLM_AASAL, RT_AASAL, std::shared_ptr<RLM_AASAL>>(m, "RestLengthMinimization_AASAL")
        .def(py::init<const AASAL &>(), py::arg("linkage"))
        ;

    using RLM_AA = RestLengthMinimization<AverageAngleLinkage>;
    py::class_<RLM_AA, RT_AA, std::shared_ptr<RLM_AA>>(m, "RestLengthMinimization_AA")
        .def(py::init<const AverageAngleLinkage &>(), py::arg("linkage"))
        ;

    using DOT_AASAL = DesignOptimizationTerm<AASAL_T>;
    py::class_<DOT_AASAL, std::shared_ptr<DOT_AASAL>>(m, "DesignOptimizationTerm_AASAL")
        .def("value",  &DOT_AASAL::value)
        .def("update", &DOT_AASAL::update)
        .def("grad"  , &DOT_AASAL::grad  )
        .def("grad_x", &DOT_AASAL::grad_x)
        .def("grad_p", &DOT_AASAL::grad_p)
        .def("computeGrad",      &DOT_AASAL::computeGrad)
        .def("computeDeltaGrad", &DOT_AASAL::computeDeltaGrad, py::arg("delta_xp"))
        ;

    using DOT_AA = DesignOptimizationTerm<AverageAngleLinkage_T>;
    py::class_<DOT_AA, std::shared_ptr<DOT_AA>>(m, "DesignOptimizationTerm_AA")
        .def("value",  &DOT_AA::value)
        .def("update", &DOT_AA::update)
        .def("grad"  , &DOT_AA::grad  )
        .def("grad_x", &DOT_AA::grad_x)
        .def("grad_p", &DOT_AA::grad_p)
        .def("computeGrad",      &DOT_AA::computeGrad)
        .def("computeDeltaGrad", &DOT_AA::computeDeltaGrad, py::arg("delta_xp"))
        ;

    using DOOT_AASAL = DesignOptimizationObjectiveTerm<AASAL_T>;
    py::class_<DOOT_AASAL, DOT_AASAL, std::shared_ptr<DOOT_AASAL>>(m, "DesignOptimizationObjectiveTerm_AASAL")
        .def_readwrite("weight", &DOOT_AASAL::weight)
        ;

    using DOOT_AA = DesignOptimizationObjectiveTerm<AverageAngleLinkage_T>;
    py::class_<DOOT_AA, DOT_AA, std::shared_ptr<DOOT_AA>>(m, "DesignOptimizationObjectiveTerm_AA")
        .def_readwrite("weight", &DOOT_AA::weight)
        ;

    using EEO_AASAL = ElasticEnergyObjective<AASAL_T>;
    py::class_<EEO_AASAL, DOOT_AASAL, std::shared_ptr<EEO_AASAL>>(m, "ElasticEnergyObjective_AASAL")
        .def(py::init<const AASAL &>(), py::arg("surface_attracted_linkage"))
        .def_property("useEnvelopeTheorem", &EEO_AASAL::useEnvelopeTheorem, &EEO_AASAL::setUseEnvelopeTheorem)
        ;

    using EEO_AA = ElasticEnergyObjective<AverageAngleLinkage_T>;
    py::class_<EEO_AA, DOOT_AA, std::shared_ptr<EEO_AA>>(m, "ElasticEnergyObjective_AA")
        .def(py::init<const AverageAngleLinkage &>(), py::arg("rod_linkage"))
        .def_property("useEnvelopeTheorem", &EEO_AA::useEnvelopeTheorem, &EEO_AA::setUseEnvelopeTheorem)
        ;

    using CFO_AASAL = ContactForceObjective<AASAL_T>;
    py::class_<CFO_AASAL, DOOT_AASAL, std::shared_ptr<CFO_AASAL>>(m, "ContactForceObjective_AASAL")
        .def(py::init<const AASAL &>(), py::arg("surface_attracted_linkage"))
        .def_property(             "normalWeight", &CFO_AASAL::getNormalWeight,              &CFO_AASAL::setNormalWeight)
        .def_property(         "tangentialWeight", &CFO_AASAL::getTangentialWeight,          &CFO_AASAL::setTangentialWeight)
        .def_property(             "torqueWeight", &CFO_AASAL::getTorqueWeight,              &CFO_AASAL::setTorqueWeight)
        .def_property(     "boundaryNormalWeight", &CFO_AASAL::getBoundaryNormalWeight,      &CFO_AASAL::setBoundaryNormalWeight)
        .def_property( "boundaryTangentialWeight", &CFO_AASAL::getBoundaryTangentialWeight,  &CFO_AASAL::setBoundaryTangentialWeight)
        .def_property(     "boundaryTorqueWeight", &CFO_AASAL::getBoundaryTorqueWeight,      &CFO_AASAL::setBoundaryTorqueWeight)
        .def_property("normalActivationThreshold", &CFO_AASAL::getNormalActivationThreshold, &CFO_AASAL::setNormalActivationThreshold)
        .def("jointForces", [](const CFO_AASAL &cfo) { return cfo.jointForces(); })
        ;

    using CFO_AA= ContactForceObjective<AverageAngleLinkage_T>;
    py::class_<CFO_AA, DOOT_AA, std::shared_ptr<CFO_AA>>(m, "ContactForceObjective_AA")
        .def(py::init<const AverageAngleLinkage &>(), py::arg("rod_linkage"))
        .def_property(             "normalWeight", &CFO_AA::getNormalWeight,              &CFO_AA::setNormalWeight)
        .def_property(         "tangentialWeight", &CFO_AA::getTangentialWeight,          &CFO_AA::setTangentialWeight)
        .def_property(             "torqueWeight", &CFO_AA::getTorqueWeight,              &CFO_AA::setTorqueWeight)
        .def_property(     "boundaryNormalWeight", &CFO_AA::getBoundaryNormalWeight,      &CFO_AA::setBoundaryNormalWeight)
        .def_property( "boundaryTangentialWeight", &CFO_AA::getBoundaryTangentialWeight,  &CFO_AA::setBoundaryTangentialWeight)
        .def_property(     "boundaryTorqueWeight", &CFO_AA::getBoundaryTorqueWeight,      &CFO_AA::setBoundaryTorqueWeight)
        .def_property("normalActivationThreshold", &CFO_AA::getNormalActivationThreshold, &CFO_AA::setNormalActivationThreshold)
        .def("jointForces", [](const CFO_AA &cfo) { return cfo.jointForces(); })
        ;

    using TFO_AASAL = TargetFittingDOOT<AASAL_T>;
    py::class_<TFO_AASAL, DOOT_AASAL, std::shared_ptr<TFO_AASAL>>(m, "TargetFittingDOOT_AASAL")
        .def(py::init<const AASAL &, TargetSurfaceFitter &>(), py::arg("surface_attracted_linkage"), py::arg("targetSurfaceFitter"))
        ;

    using TFO_AA = TargetFittingDOOT<AverageAngleLinkage_T>;
    py::class_<TFO_AA, DOOT_AA, std::shared_ptr<TFO_AA>>(m, "TargetFittingDOOT_AA")
        .def(py::init<const AverageAngleLinkage &, TargetSurfaceFitter &>(), py::arg("rod_linkage"), py::arg("targetSurfaceFitter"))
        ;

    using RCSD_AASAL = RegularizationTermDOOWrapper<AASAL_T, RestCurvatureSmoothing>; //Might be wrong
    py::class_<RCSD_AASAL, DOT_AASAL, std::shared_ptr<RCSD_AASAL>>(m, "RestCurvatureSmoothingDOOT_AASAL")
        .def(py::init<const AASAL &>(),          py::arg("surface_attracted_linkage"))
        .def(py::init<std::shared_ptr<RCS_AASAL>>(), py::arg("restCurvatureRegTerm"))
        .def_property("weight", [](const RCSD_AASAL &r) { return r.weight; }, [](RCSD_AASAL &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using RCSD_AA = RegularizationTermDOOWrapper<AverageAngleLinkage_T, RestCurvatureSmoothing>; //Might be wrong
    py::class_<RCSD_AA, DOT_AA, std::shared_ptr<RCSD_AA>>(m, "RestCurvatureSmoothingDOOT_AA")
        .def(py::init<const AverageAngleLinkage &>(),          py::arg("rod_linkage"))
        .def(py::init<std::shared_ptr<RCS_AA>>(), py::arg("restCurvatureRegTerm"))
        .def_property("weight", [](const RCSD_AA &r) { return r.weight; }, [](RCSD_AA &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using RLMD_AASAL = RegularizationTermDOOWrapper<AASAL_T, RestLengthMinimization>; //Might be wrong
    py::class_<RLMD_AASAL, DOT_AASAL, std::shared_ptr<RLMD_AASAL>>(m, "RestLengthMinimizationDOOT_AASAL")
        .def(py::init<const AASAL &>(),          py::arg("surface_attracted_linkage"))
        .def(py::init<std::shared_ptr<RLM_AASAL>>(), py::arg("restLengthMinimizationTerm"))
        .def_property("weight", [](const RLMD_AASAL &r) { return r.weight; }, [](RLMD_AASAL &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using RLMD_AA = RegularizationTermDOOWrapper<AverageAngleLinkage_T, RestLengthMinimization>; //Might be wrong
    py::class_<RLMD_AA, DOT_AA, std::shared_ptr<RLMD_AA>>(m, "RestLengthMinimizationDOOT_AA")
        .def(py::init<const AverageAngleLinkage &>(),          py::arg("rod_linkage"))
        .def(py::init<std::shared_ptr<RLM_AA>>(), py::arg("restLengthMinimizationTerm"))
        .def_property("weight", [](const RLMD_AA &r) { return r.weight; }, [](RLMD_AA &r, Real w) { r.weight = w; }) // Needed since we inherit from DOT instead of DOOT (to share weight with RestCurvatureSmoothing)
        ;

    using OEType = OptEnergyType;
    using DOO_AASAL  = DesignOptimizationObjective<AASAL_T, OEType>;
    using TR_AASAL   = DOO_AASAL::TermRecord;
    using TPtr_AASAL = DOO_AASAL::TermPtr;
    py::class_<DOO_AASAL> doo_AASAL(m, "DesignOptimizationObjective_AASAL");

    py::class_<TR_AASAL>(doo_AASAL, "DesignOptimizationObjectiveTermRecord_AASAL")
        .def(py::init<const std::string &, OEType, std::shared_ptr<DOT_AASAL>>(),  py::arg("name"), py::arg("type"), py::arg("term"))
        .def_readwrite("name", &TR_AASAL::name)
        .def_readwrite("type", &TR_AASAL::type)
        .def_readwrite("term", &TR_AASAL::term)
        .def("__repr__", [](const TR_AASAL *trp) {
                const auto &tr = *trp;
                return "TermRecord " + tr.name + " at " + hexString(trp) + " with weight " + std::to_string(tr.term->getWeight()) + " and value " + std::to_string(tr.term->unweightedValue());
        })
        ;

    doo_AASAL.def(py::init<>())
       .def("update",         &DOO_AASAL::update)
       .def("grad",           &DOO_AASAL::grad, py::arg("type") = OEType::Full)
       .def("values",         &DOO_AASAL::values)
       .def("weightedValues", &DOO_AASAL::weightedValues)
       .def("value",  py::overload_cast<OEType>(&DOO_AASAL::value, py::const_), py::arg("type") = OEType::Full)
       .def("computeGrad",      &DOO_AASAL::computeGrad, py::arg("type") = OEType::Full)
       .def("computeDeltaGrad", &DOO_AASAL::computeDeltaGrad, py::arg("delta_xp"), py::arg("type") = OEType::Full)
       .def_readwrite("terms",  &DOO_AASAL::terms, py::return_value_policy::reference)
       .def("add", py::overload_cast<const std::string &, OEType, TPtr_AASAL      >(&DOO_AASAL::add),  py::arg("name"), py::arg("type"), py::arg("term"))
       .def("add", py::overload_cast<const std::string &, OEType, TPtr_AASAL, Real>(&DOO_AASAL::add),  py::arg("name"), py::arg("type"), py::arg("term"), py::arg("weight"))
       // More convenient interface for adding multiple terms at once
       .def("add", [](DOO_AASAL &o, const std::list<std::tuple<std::string, OEType, TPtr_AASAL>> &terms) {
                    for (const auto &t : terms)
                        o.add(std::get<0>(t), std::get<1>(t), std::get<2>(t));
               })
       ;

    using DOO_AA = DesignOptimizationObjective<AverageAngleLinkage_T, OEType>;
    using TR_AA   = DOO_AA::TermRecord;
    using TPtr_AA = DOO_AA::TermPtr;
    py::class_<DOO_AA> doo_AA(m, "DesignOptimizationObjective_AA");

    py::class_<TR_AA>(doo_AA, "DesignOptimizationObjectiveTermRecord_AA")
        .def(py::init<const std::string &, OEType, std::shared_ptr<DOT_AA>>(),  py::arg("name"), py::arg("type"), py::arg("term"))
        .def_readwrite("name", &TR_AA::name)
        .def_readwrite("type", &TR_AA::type)
        .def_readwrite("term", &TR_AA::term)
        .def("__repr__", [](const TR_AA *trp) {
                const auto &tr = *trp;
                return "TermRecord " + tr.name + " at " + hexString(trp) + " with weight " + std::to_string(tr.term->getWeight()) + " and value " + std::to_string(tr.term->unweightedValue());
        })
        ;

    doo_AA.def(py::init<>())
       .def("update",         &DOO_AA::update)
       .def("grad",           &DOO_AA::grad, py::arg("type") = OEType::Full)
       .def("values",         &DOO_AA::values)
       .def("weightedValues", &DOO_AA::weightedValues)
       .def("value",  py::overload_cast<OEType>(&DOO_AA::value, py::const_), py::arg("type") = OEType::Full)
        .def("computeGrad",     &DOO_AA::computeGrad, py::arg("type") = OEType::Full)
       .def("computeDeltaGrad", &DOO_AA::computeDeltaGrad, py::arg("delta_xp"), py::arg("type") = OEType::Full)
       .def_readwrite("terms",  &DOO_AA::terms, py::return_value_policy::reference)
       .def("add", py::overload_cast<const std::string &, OEType, TPtr_AA      >(&DOO_AA::add),  py::arg("name"), py::arg("type"), py::arg("term"))
       .def("add", py::overload_cast<const std::string &, OEType, TPtr_AA, Real>(&DOO_AA::add),  py::arg("name"), py::arg("type"), py::arg("term"), py::arg("weight"))
       // More convenient interface for adding multiple terms at once
       .def("add", [](DOO_AA &o, const std::list<std::tuple<std::string, OEType, TPtr_AA>> &terms) {
                    for (const auto &t : terms)
                        o.add(std::get<0>(t), std::get<1>(t), std::get<2>(t));
               })
       ;


}
