#include <iostream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <memory>
#include "../ElasticRod.hh"
#include "../RodLinkage.hh"
#include "../PeriodicRod.hh"
#include "../SurfaceAttractedLinkage.hh"
#include "../compute_equilibrium.hh"
#include "../restlen_solve.hh"
#include "../design_parameter_solve.hh"
#include "../knitro_solver.hh"
#include "../linkage_deformation_analysis.hh"
#include "../DeploymentPathAnalysis.hh"

#include "../CrossSection.hh"
#include "../cross_sections/Custom.hh"
#include "../CrossSectionMesh.hh"

#include "LinkageTerminalEdgeSensitivity.hh"

#include "visualization.hh"

#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
namespace py = pybind11;

template <typename T>
std::string to_string_with_precision(const T &val, const int n = 6) {
    std::ostringstream ss;
    ss << std::setprecision(n) << val;
    return ss.str();

}
// Conversion of std::tuple to and from a py::tuple, since pybind11 doesn't seem to provide this...
template<typename... Args, size_t... Idxs>
py::tuple to_pytuple_helper(const std::tuple<Args...> &args, std::index_sequence<Idxs...>) {
    return py::make_tuple(std::get<Idxs>(args)...);
}

template<typename... Args>
py::tuple to_pytuple(const std::tuple<Args...> &args) {
    return to_pytuple_helper(args, std::make_index_sequence<sizeof...(Args)>());
}

template<class OutType>
struct FromPytupleImpl;

template<typename... Args>
struct FromPytupleImpl<std::tuple<Args...>> {
    template<size_t... Idxs>
    static auto run_helper(const py::tuple &t, std::index_sequence<Idxs...>) {
        return std::make_tuple((t[Idxs].cast<Args>())...);
    }
    static auto run(const py::tuple &t) {
        if (t.size() != sizeof...(Args)) throw std::runtime_error("Mismatched tuple size for py::tuple to std::tuple conversion.");
        return run_helper(t, std::make_index_sequence<sizeof...(Args)>());
    }
};

template<class OutType>
OutType from_pytuple(const py::tuple &t) {
    return FromPytupleImpl<OutType>::run(t);
}

// Hack around a limitation of pybind11 where we cannot specify argument passing policies and
// pybind11 tries to make a copy if the passed instance is not already registered:
//      https://github.com/pybind/pybind11/issues/1200
// We therefore make our Python callback interface use a raw pointer to forbid this copy (which
// causes an error since NewtonProblem is not copyable).
using PyCallbackFunction = std::function<void(NewtonProblem *, size_t)>;
CallbackFunction callbackWrapper(const PyCallbackFunction &pcb) {
    return [pcb](NewtonProblem &p, size_t i) -> void { if (pcb) pcb(&p, i); };
}

template<typename Object>
void bindDesignParameterProblem(py::module &m, const std::string &typestr) {
    using DPP = DesignParameterProblem<Object>;
    std::string pyclass_name = std::string("DesignParameterProblem_") + typestr;
    py::class_<DPP, NewtonProblem>(m, pyclass_name.c_str())
    .def(py::init<Object &>())
    .def("set_regularization_weight",  &DPP::set_regularization_weight, py::arg("weight"))
    .def("set_smoothing_weight",       &DPP::set_smoothing_weight,      py::arg("weight"))
    .def("set_gamma",                  &DPP::set_gamma,                 py::arg("new_gamma"))
    .def("elasticEnergyWeight",        &DPP::elasticEnergyWeight)
    .def("setCustomIterationCallback", [](DPP &dpp, const PyCallbackFunction &pcb) { dpp.setCustomIterationCallback(callbackWrapper(pcb)); })
    .def("restKappaSmoothness",        &DPP::restKappaSmoothness)
    .def("weighted_energy",            &DPP::weighted_energy)
    .def("weighted_smoothness",        &DPP::weighted_smoothness)
    .def("weighted_length",            &DPP::weighted_length)
    ;

}

PYBIND11_MODULE(elastic_rods, m) {
    m.doc() = "Elastic Rods Codebase";

    py::module::import("MeshFEM");
    py::module::import("mesh");
    py::module::import("sparse_matrices");

    py::module detail_module = m.def_submodule("detail");

    ////////////////////////////////////////////////////////////////////////////////
    // ElasticRods and nested classes
    ////////////////////////////////////////////////////////////////////////////////
    auto elastic_rod = py::class_<ElasticRod>(m, "ElasticRod");

    py::enum_<ElasticRod::EnergyType>(m, "EnergyType")
        .value("Full",    ElasticRod::EnergyType::Full   )
        .value("Bend",    ElasticRod::EnergyType::Bend   )
        .value("Twist",   ElasticRod::EnergyType::Twist  )
        .value("Stretch", ElasticRod::EnergyType::Stretch)
        ;

    py::enum_<SurfaceAttractedLinkage::SurfaceAttractionEnergyType>(m, "SurfaceAttractionEnergyType")
        .value("Full",       SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Full      )
        .value("Attraction", SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Attraction)
        .value("Elastic",    SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Elastic   )
        ;

    py::enum_<ElasticRod::BendingEnergyType>(m, "BendingEnergyType")
        .value("Bergou2010", ElasticRod::BendingEnergyType::Bergou2010)
        .value("Bergou2008", ElasticRod::BendingEnergyType::Bergou2008)
        ;

    py::enum_<InterleavingType>(m, "InterleavingType")
        .value("xshell",        InterleavingType::xshell)
        .value("weaving",       InterleavingType::weaving)
        .value("noOffset",      InterleavingType::noOffset)
        .value("triaxialWeave", InterleavingType::triaxialWeave)
        ;

    py::class_<GradientStencilMaskCustom>(m, "GradientStencilMaskCustom")
        .def(py::init<>())
        .def_readwrite("edgeStencilMask", &GradientStencilMaskCustom::edgeStencilMask)
        .def_readwrite("vtxStencilMask",  &GradientStencilMaskCustom::vtxStencilMask)
        ;

    py::class_<HessianComputationMask>(m, "HessianComputationMask")
        .def(py::init<>())
        .def_readwrite("dof_in",              &HessianComputationMask::dof_in)
        .def_readwrite("dof_out",             &HessianComputationMask::dof_out)
        .def_readwrite("designParameter_in",  &HessianComputationMask::designParameter_in)
        .def_readwrite("designParameter_out", &HessianComputationMask::designParameter_out)
        .def_readwrite("skipBRods",           &HessianComputationMask::skipBRods)
        ;

    py::class_<DesignParameterConfig>(m, "DesignParameterConfig")
        .def(py::init<>())
        .def_readonly("restLen", &DesignParameterConfig::restLen)
        .def_readonly("restKappa", &DesignParameterConfig::restKappa)
        .def(py::pickle([](const DesignParameterConfig &dpc) { return py::make_tuple(dpc.restLen, dpc.restKappa); },
                        [](const py::tuple &t) {
                        if (t.size() != 2) throw std::runtime_error("Invalid DesignParameterConfig!");
                            DesignParameterConfig dpc; 
                            dpc.restLen = t[0].cast<bool>();
                            dpc.restKappa = t[1].cast<bool>();
                            return dpc;
                        }));

    elastic_rod
        .def(py::init<std::vector<Point3D>>())
        .def("__repr__", [](const ElasticRod &e) { return "Elastic rod with " + std::to_string(e.numVertices()) + " points and " + std::to_string(e.numEdges()) + " edges"; })
        .def("setDeformedConfiguration", py::overload_cast<const std::vector<Point3D> &, const std::vector<Real> &>(&ElasticRod::setDeformedConfiguration))
        .def("setDeformedConfiguration", py::overload_cast<const ElasticRod::DeformedState &>(&ElasticRod::setDeformedConfiguration))
        .def("deformedPoints", &ElasticRod::deformedPoints)
        .def("restDirectors",  &ElasticRod::restDirectors)
        .def("thetas",         &ElasticRod::thetas)
        .def("setMaterial",    py::overload_cast<const             RodMaterial  &>(&ElasticRod::setMaterial))
        .def("setMaterial",    py::overload_cast<const std::vector<RodMaterial> &>(&ElasticRod::setMaterial))
        .def("setLinearlyInterpolatedMaterial", &ElasticRod::setLinearlyInterpolatedMaterial, py::arg("startMat"), py::arg("endMat"))
        .def("material", py::overload_cast<size_t>(&ElasticRod::material, py::const_), py::return_value_policy::reference)
        .def("set_design_parameter_config", &ElasticRod::setDesignParameterConfig, py::arg("use_restLen"), py::arg("use_restKappa"))
        .def("get_design_parameter_config", &ElasticRod::getDesignParameterConfig)
        .def("setRestLengths", &ElasticRod::setRestLengths, py::arg("val"))
        .def("setRestKappas", &ElasticRod::setRestKappas, py::arg("val"))
        .def("numRestKappaVars", &ElasticRod::numRestKappaVars)
        .def("setRestKappaVars", &ElasticRod::setRestKappaVars, py::arg("params"))
        .def("getRestKappaVars", &ElasticRod::getRestKappaVars)
        .def("restKappas", py::overload_cast<>(&ElasticRod::restKappas, py::const_))
        .def("restPoints", &ElasticRod::restPoints)
        .def("deformedMaterialFramesD1D2", &ElasticRod::deformedMaterialFramesD1D2)

        // Outputs mesh with normals
        .def("visualizationGeometry",             &getVisualizationGeometry<ElasticRod>, py::arg("averagedMaterialFrames") = true, py::arg("averagedCrossSections") = true)
        .def("visualizationGeometryHeightColors", &getVisualizationGeometryCSHeightField<ElasticRod>, "Get a per-visualization-vertex field representing height above the centerline")

        .def("rawVisualizationGeometry", [](ElasticRod &r, const bool averagedMaterialFrames, const bool averagedCrossSections) {
                std::vector<MeshIO::IOVertex > vertices;
                std::vector<MeshIO::IOElement> quads;
                r.visualizationGeometry(vertices, quads, averagedMaterialFrames, averagedCrossSections);
                const size_t nv = vertices.size(),
                             ne = quads.size();
                Eigen::MatrixX3d V(nv, 3);
                Eigen::MatrixX4i F(ne, 4);

                for (size_t i = 0; i < nv; ++i) V.row(i) = vertices[i].point;
                for (size_t i = 0; i < ne; ++i) {
                    const auto &q = quads[i];
                    if (q.size() != 4) throw std::runtime_error("Expected quads");
                    F.row(i) << q[0], q[1], q[2], q[3];
                }

                return std::make_pair(V, F);
            }, py::arg("averagedMaterialFrames") = false, py::arg("averagedCrossSections") = true)
        .def("saveVisualizationGeometry", &ElasticRod::saveVisualizationGeometry, py::arg("path"), py::arg("averagedMaterialFrames") = false, py::arg("averagedCrossSections") = false)
        .def("writeDebugData", &ElasticRod::writeDebugData)

        .def("deformedConfiguration", py::overload_cast<>(&ElasticRod::deformedConfiguration, py::const_), py::return_value_policy::reference)
        .def("updateSourceFrame", &ElasticRod::updateSourceFrame)

        .def("numEdges",    &ElasticRod::numEdges)
        .def("numVertices", &ElasticRod::numVertices)

        .def("numDoF",  &ElasticRod::numDoF)
        .def("getDoFs", &ElasticRod::getDoFs)
        .def("setDoFs", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &>(&ElasticRod::setDoFs), py::arg("values"))

        .def("posOffset",     &ElasticRod::posOffset)
        .def("thetaOffset",   &ElasticRod::thetaOffset)
        // TODO (Samara)
        .def("restLenOffset", &ElasticRod::restLenOffset)
        .def("designParameterOffset", &ElasticRod::designParameterOffset)

        .def("numExtendedDoF",  &ElasticRod::numExtendedDoF)
        .def("getExtendedDoFs", &ElasticRod::getExtendedDoFs)
        .def("setExtendedDoFs", &ElasticRod::setExtendedDoFs)
        .def("lengthVars"     , &ElasticRod::lengthVars, py::arg("variableRestLen") = false)

        .def("totalRestLength",           &ElasticRod::totalRestLength)
        .def("restLengths",               &ElasticRod::restLengths)
        // Determine the deformed position at curve parameter 0.5
        .def_property_readonly("midpointPosition", [](const ElasticRod &e) -> Point3D {
                size_t ne = e.numEdges();
                // Midpoint is in the middle of an edge for odd numbers of edges,
                // at a vertex for even numbers of edges.
                if (ne % 2) return 0.5 * (e.deformedPoint(ne / 2) + e.deformedPoint(ne / 2 + 1));
                else        return e.deformedPoint(ne / 2);
            })
        // Determine the deformed material frame vector d2 at curve parameter 0.5
        .def_property_readonly("midpointD2", [](const ElasticRod &e) -> Vector3D {
                size_t ne = e.numEdges();
                // Midpoint is in the middle of an edge for odd numbers of edges,
                // at a vertex for even numbers of edges.
                if (ne % 2) return e.deformedMaterialFrameD2(ne / 2);
                else        return 0.5 * (e.deformedMaterialFrameD2(ne / 2 - 1) + e.deformedMaterialFrameD2(ne / 2));
            })

        .def_property("bendingEnergyType", [](const ElasticRod &e) { return e.bendingEnergyType(); },
                                           [](ElasticRod &e, ElasticRod::BendingEnergyType type) { e.setBendingEnergyType(type); })
        .def("energyStretch", &ElasticRod::energyStretch, "Compute stretching energy")
        .def("energyBend",    &ElasticRod::energyBend   , "Compute bending    energy")
        .def("energyTwist",   &ElasticRod::energyTwist  , "Compute twisting   energy")
        .def("energyStretchPerEdge",   &ElasticRod::energyStretchPerEdge, "Compute stretching energy per edge")
        .def("energyBendPerVertex",    &ElasticRod::energyBendPerVertex   , "Compute bending    energy per vertex")
        .def("energyTwistPerVertex",   &ElasticRod::energyTwistPerVertex  , "Compute twisting   energy per vertex")
        .def("sqrtBendingEnergies",    &ElasticRod::sqrtBendingEnergies  , "Compute the square root of the bending energy per vertex")
        .def("energy",        py::overload_cast<ElasticRod::EnergyType>(&ElasticRod::energy, py::const_), "Compute elastic energy", py::arg("energyType") = ElasticRod::EnergyType::Full)

        .def("gradEnergyStretch", &ElasticRod::gradEnergyStretch<GradientStencilMaskCustom>, "Compute stretching energy gradient"                                                                                        , py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())
        .def("gradEnergyBend",    &ElasticRod::gradEnergyBend   <GradientStencilMaskCustom>, "Compute bending    energy gradient", py::arg("updatedSource") = false                                                      , py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())
        .def("gradEnergyTwist",   &ElasticRod::gradEnergyTwist  <GradientStencilMaskCustom>, "Compute twisting   energy gradient", py::arg("updatedSource") = false                                                      , py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())
        .def("gradient",          &ElasticRod::gradient         <GradientStencilMaskCustom>, "Compute elastic    energy gradient", py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("stencilMask") = GradientStencilMaskCustom())

        .def("hessianNNZ",             &ElasticRod::hessianNNZ,             "Tight upper bound for nonzeros in the Hessian.", py::arg("variableDesignParameters") = false)
        .def("hessianSparsityPattern", &ElasticRod::hessianSparsityPattern, "Compressed column matrix containing all potential nonzero Hessian entries", py::arg("variableDesignParameters") = false, py::arg("val") = 0.0)

        .def("hessian",           [](const ElasticRod &e, ElasticRod::EnergyType eType, bool variableDesignParameters) { return e.hessian(eType, variableDesignParameters); }, "Compute elastic energy Hessian", py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false)

        .def("applyHessian", &ElasticRod::applyHessian, "Elastic energy Hessian-vector product formulas.", py::arg("v"), py::arg("variableDesignParameters") = false, py::arg("mask") = HessianComputationMask())

        .def("massMatrix",        py::overload_cast<>(&ElasticRod::massMatrix, py::const_))
        .def("lumpedMassMatrix",  &ElasticRod::lumpedMassMatrix)

        .def("characteristicLength", &ElasticRod::characteristicLength)
        .def("approxLinfVelocity",   &ElasticRod::approxLinfVelocity)

        .def("bendingStiffnesses",  py::overload_cast<>(&ElasticRod::bendingStiffnesses,  py::const_), py::return_value_policy::reference)
        .def("twistingStiffnesses", py::overload_cast<>(&ElasticRod::twistingStiffnesses, py::const_), py::return_value_policy::reference)

        .def("stretchingStresses",      &ElasticRod::     stretchingStresses)
        .def("bendingStresses",         &ElasticRod::        bendingStresses)
        .def("minBendingStresses",      &ElasticRod::     minBendingStresses)
        .def("maxBendingStresses",      &ElasticRod::     maxBendingStresses)
        .def("twistingStresses",        &ElasticRod::       twistingStresses)
        .def("maxStresses",             &ElasticRod::            maxStresses, py::arg("stressType"))
        .def("maxVonMisesStresses",     &ElasticRod::    maxVonMisesStresses)
        .def("surfaceStressLpNorm",     &ElasticRod::    surfaceStressLpNorm, py::arg("stressType"), py::arg("p"),                               py::arg("takeRoot") = true)
        .def("gradSurfaceStressLpNorm", &ElasticRod::gradSurfaceStressLpNorm, py::arg("stressType"), py::arg("p"), py::arg("updateSourceFrame"), py::arg("takeRoot") = true)

        .def("edgeMaterials",    &ElasticRod::edgeMaterials)

        .def("visualizationField", [](const ElasticRod &r, const Eigen::VectorXd  &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
        .def("visualizationField", [](const ElasticRod &r, const Eigen::MatrixX3d &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))

        .def(py::pickle([](const ElasticRod &r) { return py::make_tuple(r.restPoints(), r.restDirectors(), r.restKappas(), r.restTwists(), r.restLengths(),
                                                         r.edgeMaterials(),
                                                         r.bendingStiffnesses(),
                                                         r.twistingStiffnesses(),
                                                         r.stretchingStiffnesses(),
                                                         r.bendingEnergyType(),
                                                         r.deformedConfiguration(),
                                                         r.densities(),
                                                         r.initialMinRestLength()); },
                        [](const py::tuple &t) {
                        if ((t.size() < 11) || (t.size() > 13)) throw std::runtime_error("Invalid state!");
                            ElasticRod r              (t[ 0].cast<std::vector<Point3D>              >());
                            r.setRestDirectors        (t[ 1].cast<std::vector<ElasticRod::Directors>>());
                            r.setRestKappas           (t[ 2].cast<ElasticRod::StdVectorVector2D     >());
                            r.setRestTwists           (t[ 3].cast<std::vector<Real>                 >());
                            r.setRestLengths          (t[ 4].cast<std::vector<Real>                 >());

                            // Support old pickling format where only a RodMaterial was written instead of a vector of rod materials.
                            try         { r.setMaterial(t[ 5].cast<std::vector<RodMaterial>>()); }
                            catch (...) { r.setMaterial(t[ 5].cast<            RodMaterial >()); }

                            r.setBendingStiffnesses   (t[ 6].cast<std::vector<RodMaterial::BendingStiffness>>());
                            r.setTwistingStiffnesses  (t[ 7].cast<std::vector<Real>                         >());
                            r.setStretchingStiffnesses(t[ 8].cast<std::vector<Real>                         >());
                            r.setBendingEnergyType    (t[ 9].cast<ElasticRod::BendingEnergyType             >());
                            r.setDeformedConfiguration(t[10].cast<ElasticRod::DeformedState                 >());

                            // Support old pickling format where densities were absent.
                            if (t.size() > 11)
                                r.setDensities(t[11].cast<std::vector<Real>>());

                            // Support old pickling format where densities were absent.
                            if (t.size() > 12)
                                r.setInitialMinRestLen(t[12].cast<Real>());

                            return r;
                        }))

        .def("fromGHState", [](const size_t startJoint, const size_t endJoint, const std::vector<Pt3_T<Real>> &pts, const std::vector<Real> &dirCoords, const std::vector<Real> &restKappas,
                               const std::vector<Real> &restTwists, const std::vector<Real> &restLengths, std::vector<RodMaterial> materials, std::vector<RodMaterial::BendingStiffness> bendingStiffness,  
                               const std::vector<Real> &twistingStiffness, const std::vector<Real> &stretchingStiffness, const ElasticRod::BendingEnergyType &energyType,
                               const ElasticRod::DeformedState &deformedState, const std::vector<Real> &densities, Real initialMinRestLen) 
              {                        
                    size_t idxStart = RodLinkage::NONE;
                    if(startJoint>-1) idxStart = startJoint;
                    size_t idxEnd = RodLinkage::NONE;
                    if(endJoint>-1) idxEnd = endJoint;
         
                    ElasticRod&& r(pts);

                    // Rest Directors
                    size_t count = dirCoords.size()/6;
                    std::vector<ElasticRod::Directors> dir;
                    dir.reserve(count);
                    for(size_t i=0; i<count; i++){
                        dir.emplace_back(Eigen::Vector3d(dirCoords[i*6], dirCoords[i*6+1],dirCoords[i*6+2]),
                                         Eigen::Vector3d(dirCoords[i*6+3], dirCoords[i*6+4],dirCoords[i*6+5]));
                    }
                    r.setRestDirectors(dir); 

                    // Rest Kappas
                    count = restKappas.size()/2;
                    CrossSection::AlignedPointCollection kData;
                    kData.reserve(count);
                    for(size_t i=0; i<count; i++){
                        kData.emplace_back(restKappas[i*2], restKappas[i*2+1]);
                    }
                    r.setRestKappas(ElasticRod::StdVectorVector2D(kData));

                    // Materials
                    if(materials.size()==1) r.setMaterial(materials[0]);
                    else r.setMaterial(materials);

                    // Bending stiffness
                    r.setBendingStiffnesses(bendingStiffness);
                    // Rest Twists
                    r.setRestTwists(restTwists);
                    // Rest Lengths
                    r.setRestLengths(restLengths);
                    // Twisting stiffness
                    r.setTwistingStiffnesses(twistingStiffness);
                    // Stretching stiffness
                    r.setStretchingStiffnesses(stretchingStiffness);
                    // Energy type
                    r.setBendingEnergyType(energyType);  
                    // Deformed state
                    r.setDeformedConfiguration(deformedState);
                    // Densities
                    r.setDensities(densities);
                    // Initial min rest length
                    r.setInitialMinRestLen(initialMinRestLen); 

                    return RodLinkage::RodSegment(idxStart, idxEnd, std::move(r));
              })
        ;

    // Note: the following bindings do not get used because PyBind thinks ElasticRod::Gradient is
    // just an Eigen::VectorXd. Also, they produce errors on Intel compilers.
    // py::class_<ElasticRod::Gradient>(elastic_rod, "Gradient")
    //     .def("__repr__", [](const ElasticRod::Gradient &g) { return "Elastic rod gradient with l2 norm " + to_string_with_precision(g.norm()); })
    //     .def_property_readonly("values", [](const ElasticRod::Gradient &g) { return Eigen::VectorXd(g); })
    //     .def("gradPos",   [](const ElasticRod::Gradient &g, size_t i) { return g.gradPos(i); })
    //     .def("gradTheta", [](const ElasticRod::Gradient &g, size_t j) { return g.gradTheta(j); })
    //     ;

    py::class_<ElasticRod::DeformedState>(elastic_rod, "DeformedState")
        .def("__repr__", [](const ElasticRod::DeformedState &) { return "Deformed state of an elastic rod (ElasticRod::DeformedState)."; })
        .def_readwrite("referenceDirectors", &ElasticRod::DeformedState::referenceDirectors)
        .def_readwrite("referenceTwist",     &ElasticRod::DeformedState::referenceTwist)
        .def_readwrite("tangent",            &ElasticRod::DeformedState::tangent)
        .def_readwrite("materialFrame",      &ElasticRod::DeformedState::materialFrame)
        .def_readwrite("kb",                 &ElasticRod::DeformedState::kb)
        .def_readwrite("kappa",              &ElasticRod::DeformedState::kappa)
        .def_readwrite("per_corner_kappa",   &ElasticRod::DeformedState::per_corner_kappa)
        .def_readwrite("len",                &ElasticRod::DeformedState::len)

        .def_readwrite("sourceTangent"           , &ElasticRod::DeformedState::sourceTangent)
        .def_readwrite("sourceReferenceDirectors", &ElasticRod::DeformedState::sourceReferenceDirectors)
        .def_readwrite("sourceMaterialFrame"     , &ElasticRod::DeformedState::sourceMaterialFrame)
        .def_readwrite("sourceReferenceTwist"    , &ElasticRod::DeformedState::sourceReferenceTwist)

        .def("updateSourceFrame", &ElasticRod::DeformedState::updateSourceFrame)

        .def("setReferenceTwist", &ElasticRod::DeformedState::setReferenceTwist)

        .def(py::pickle([](const ElasticRod::DeformedState &dc) { return py::make_tuple(dc.points(), dc.thetas(), dc.sourceTangent, dc.sourceReferenceDirectors, dc.sourceTheta, dc.sourceReferenceTwist); },
                        [](const py::tuple &t) {
                        // sourceReferenceTwist is optional for backwards compatibility
                        if (t.size() != 5 && t.size() != 6) throw std::runtime_error("Invalid state!");
                            ElasticRod::DeformedState dc;
                            const auto &pts             = t[0].cast<std::vector<Point3D              >>();
                            const auto &thetas          = t[1].cast<std::vector<Real                 >>();
                            dc.sourceTangent            = t[2].cast<std::vector<Vector3D             >>();
                            dc.sourceReferenceDirectors = t[3].cast<std::vector<ElasticRod::Directors>>();
                            dc.sourceTheta              = t[4].cast<std::vector<Real                 >>();
                            if (t.size() > 5)
                                dc.sourceReferenceTwist = t[5].cast<std::vector<Real                 >>();
                            else dc.sourceReferenceTwist.assign(thetas.size(), 0);

                            dc.update(pts, thetas);
                            return dc;
                        }))

        .def("fromGHState", [](const std::vector<Real> &ptCoords, const std::vector<Real> &thetas, const std::vector<Real> &srcTangentsCoords, const std::vector<Real> &refDirectorsCoords, 
                               const std::vector<Real> &srcThetas, const std::vector<Real> &srcRefTwist) 
            {
                // Points
                std::vector<Point3D> pts;
                size_t count = ptCoords.size()/3;
                for(size_t i=0; i<count; i++) pts.emplace_back(ptCoords[i*3],ptCoords[i*3+1],ptCoords[i*3+2]);

                // Tangents
                std::vector<Vector3D> tgt;
                count = srcTangentsCoords.size()/3;
                for(size_t i=0; i<count; i++) tgt.emplace_back(srcTangentsCoords[i*3],srcTangentsCoords[i*3+1],srcTangentsCoords[i*3+2]);

                // Directors
                std::vector<ElasticRod::Directors> directors;
                count = refDirectorsCoords.size()/6;
                for(size_t i=0; i<count; i++){
                    directors.emplace_back(Eigen::Vector3d(refDirectorsCoords[i*6],refDirectorsCoords[i*6+1],refDirectorsCoords[i*6+2]), Eigen::Vector3d(refDirectorsCoords[i*6+3],refDirectorsCoords[i*6+4],refDirectorsCoords[i*6+5]));
                }

                ElasticRod::DeformedState dc;
                dc.sourceTangent = tgt;
                dc.sourceReferenceDirectors = directors;
                dc.sourceTheta = srcThetas;
                dc.sourceReferenceTwist = srcRefTwist;

                dc.update(pts, thetas);
                return dc;
            })
        ;

    py::class_<ElasticRod::Directors>(elastic_rod, "Directors")
        .def("__repr__", [](const ElasticRod::Directors &dirs) { return "{ d1: [" + to_string_with_precision(dirs.d1.transpose()) + "], d2: [" + to_string_with_precision(dirs.d2.transpose()) + "] }"; })
        .def_readwrite("d1", &ElasticRod::Directors::d1)
        .def_readwrite("d2", &ElasticRod::Directors::d2)
        .def(py::pickle([](const ElasticRod::Directors &d) { return py::make_tuple(d.d1, d.d2); },
                        [](const py::tuple &t) {
                        if (t.size() != 2) throw std::runtime_error("Invalid state!");
                        return ElasticRod::Directors(
                                t[0].cast<Vector3D>(),
                                t[1].cast<Vector3D>());
                        }))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // CrossSection
    ////////////////////////////////////////////////////////////////////////////////
    py::class_<CrossSection>(m, "CrossSection")
        .def_static("construct", &CrossSection::construct, py::arg("type"), py::arg("E"), py::arg("nu"), py::arg("params"))
        .def_static("fromContour", [](const std::string &path, Real E, Real nu, Real scale) -> std::unique_ptr<CrossSection> {
                auto result = std::make_unique<CrossSections::Custom>(path, scale);
                result->E = E, result->nu = nu;
                return result;
            }, py::arg("path"), py::arg("E"), py::arg("nu"), py::arg("scale") = 1.0)

        .def("boundary",  &CrossSection::boundary)
        .def("interior",  [](const CrossSection &cs, Real triArea) {
                std::vector<MeshIO::IOVertex > vertices;
                std::vector<MeshIO::IOElement> elements;
                std::tie(vertices, elements) = cs.interior(triArea);
                return std::make_shared<CrossSectionMesh::Base>(elements, vertices); // must match the container type of MeshFEM's bindings or we silently get a memory bug!
            }, py::arg("triArea") = 0.001)

        .def("numParams", &CrossSection::numParams)
        .def("setParams", &CrossSection::setParams, py::arg("p"))
        .def("params",    &CrossSection::params)

        .def("holePts",   &CrossSection::holePts)

        .def("copy", &CrossSection::copy)
        .def_static("lerp", &CrossSection::lerp, py::arg("cs_a"), py::arg("cs_b"), py::arg("alpha"))

        .def_readwrite("E", &CrossSection::E)
        .def_readwrite("nu", &CrossSection::nu)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // CrossSectionStressAnalysis
    ////////////////////////////////////////////////////////////////////////////////
    using CSA = CrossSectionStressAnalysis;
    py::class_<CSA, std::shared_ptr<CSA>> pyCSA(m, "CrossSectionStressAnalysis");
    py::enum_<CSA::StressType>(pyCSA, "StressType")
        .value("VonMises",     CSA::StressType::VonMises)
        .value("MaxMag",       CSA::StressType::MaxMag)
        .value("MaxPrincipal", CSA::StressType::MaxPrincipal)
        .value("MinPrincipal", CSA::StressType::MinPrincipal)
        .value("ZStress",      CSA::StressType::ZStress)
        ;

    pyCSA
        .def("maxStress", &CSA::maxStress<double>, py::arg("type"), py::arg("tau"), py::arg("curvatureNormal"), py::arg("stretching_strain"))
        .def_readonly("boundaryV",            &CSA::boundaryV)
        .def_readonly("boundaryE",            &CSA::boundaryE)
        .def_readonly("unitTwistShearStrain", &CSA::unitTwistShearStrain)
        .def_static("stressMeasure", [](CSA::StressType type, const Eigen::Vector2d &shearStress, Real sigma_zz, bool squared) {
                    if (squared) return CSA::stressMeasure< true>(type, shearStress, sigma_zz);
                    else         return CSA::stressMeasure<false>(type, shearStress, sigma_zz);
                }, py::arg("type"), py::arg("shearStress"), py::arg("sigma_zz"), py::arg("squared"))
        .def_static("gradStressMeasure", [](CSA::StressType type, const Eigen::Vector2d &shearStress, Real sigma_zz, bool squared) {
                    Eigen::Vector2d grad_shearStress;
                    Real grad_sigma_zz;
                    if (squared) { CSA::gradStressMeasure< true>(type, shearStress, sigma_zz, grad_shearStress, grad_sigma_zz); return std::make_pair(grad_shearStress, grad_sigma_zz); }
                    else         { CSA::gradStressMeasure<false>(type, shearStress, sigma_zz, grad_shearStress, grad_sigma_zz); return std::make_pair(grad_shearStress, grad_sigma_zz); }
                }, py::arg("type"), py::arg("shearStress"), py::arg("sigma_zz"), py::arg("squared"))
        .def(py::pickle([](const CSA &csa) {
                return std::make_tuple(csa.boundaryV, csa.boundaryE,
                                       csa.unitTwistShearStrain, csa.youngModulus,
                                       csa.shearModulus);
            },
            [](const std::tuple<CrossSection::AlignedPointCollection, CrossSection::EdgeCollection,
                                Eigen::MatrixX2d, Real, Real> &t) {
                return std::make_shared<CSA>(std::get<0>(t), std::get<1>(t),
                                             std::get<2>(t), std::get<3>(t),
                                             std::get<4>(t));
            }))
        ;
    // Stress visualization binding must come after StressType is bound...
    elastic_rod.def("stressVisualization", [](const ElasticRod &e, bool amf, bool acs, CrossSectionStressAnalysis::StressType t) { return getVisualizationWithStress(e, amf, acs, t); }, py::arg("averagedMaterialFrames") = true, py::arg("averagedCrossSections") = true, py::arg("stressType") = CrossSectionStressAnalysis::StressType::VonMises)
        ;


    ////////////////////////////////////////////////////////////////////////////////
    // RodMaterial
    ////////////////////////////////////////////////////////////////////////////////
    py::enum_<RodMaterial::StiffAxis>(m, "StiffAxis")
        .value("D1", RodMaterial::StiffAxis::D1)
        .value("D2", RodMaterial::StiffAxis::D2)
        ;

    py::class_<RodMaterial>(m, "RodMaterial")
        .def(py::init<const std::string &, RodMaterial::StiffAxis, bool>(),
                py::arg("cross_section_path.json"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def(py::init<const std::string &, Real, Real, const std::vector<Real> &, RodMaterial::StiffAxis, bool>(),
                py::arg("type"), py::arg("E"), py::arg("nu"),py::arg("params"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def(py::init<const CrossSection &, RodMaterial::StiffAxis, bool>(), py::arg("cs"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def(py::init<>())
        .def("set", py::overload_cast<const std::string &, Real, Real, const std::vector<Real> &, RodMaterial::StiffAxis, bool>(&RodMaterial::set),
                py::arg("type"), py::arg("E"), py::arg("nu"),py::arg("params"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false)
        .def("set", py::overload_cast<const CrossSection &, RodMaterial::StiffAxis, bool, const std::string &>(&RodMaterial::set), 
                py::arg("cs"), py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false, py::arg("debug_psi_path") = std::string())
        .def("setEllipse", &RodMaterial::setEllipse, "Set elliptical cross section")
        .def("setContour", &RodMaterial::setContour, "Set using a custom profile whose boundary is read from a line mesh file",
                py::arg("E"), py::arg("nu"), py::arg("path"), py::arg("scale") = 1.0, py::arg("stiffAxis") = RodMaterial::StiffAxis::D1, py::arg("keepCrossSectionMesh") = false, py::arg("debug_psi_path") = std::string(), py::arg("triArea") = 0.001, py::arg("simplifyVisualizationMesh") = 0)
        .def_readwrite("area",                      &RodMaterial::area)
        .def_readwrite("stretchingStiffness",       &RodMaterial::stretchingStiffness)
        .def_readwrite("twistingStiffness",         &RodMaterial::twistingStiffness)
        .def_readwrite("bendingStiffness",          &RodMaterial::bendingStiffness)
        .def_readwrite("momentOfInertia",           &RodMaterial::momentOfInertia)
        .def_readwrite("torsionStressCoefficient",  &RodMaterial::torsionStressCoefficient)
        .def_readwrite("youngModulus",              &RodMaterial::youngModulus)
        .def_readwrite("shearModulus",              &RodMaterial::shearModulus)
        .def_readwrite("crossSectionHeight",        &RodMaterial::crossSectionHeight)
        .def_readwrite("crossSectionBoundaryPts",   &RodMaterial::crossSectionBoundaryPts,   py::return_value_policy::reference)
        .def_readwrite("crossSectionBoundaryEdges", &RodMaterial::crossSectionBoundaryEdges, py::return_value_policy::reference)
        .def("hasCrossSection",                     &RodMaterial::hasCrossSection)
        .def("hasCrossSectionMesh",                 &RodMaterial::hasCrossSectionMesh)
        .def("meshCrossSection",                    &RodMaterial::meshCrossSection, py::arg("triArea") = 0.001)
        .def("crossSection",                        &RodMaterial::crossSection,              py::return_value_policy::reference)
        .def("releaseCrossSectionMesh",             &RodMaterial::releaseCrossSectionMesh)
        .def_property_readonly("crossSectionMesh",  [](const RodMaterial &rmat) { return std::shared_ptr<CrossSectionMesh::Base>(rmat.crossSectionMeshPtr()); })
        .def("bendingStresses", &RodMaterial::bendingStresses, py::arg("curvatureNormal"))
        .def("copy", [](const RodMaterial &mat) { return std::make_unique<RodMaterial>(mat); })
        .def("stressAnalysis", [](const RodMaterial &mat) { mat.stressAnalysis(); return mat.stressAnalysisPtr(); })
        // Convenience accessors for individual bending stiffness/moment of inertia components
        .def_property("B11", [](const RodMaterial &m          ) { return m.bendingStiffness.lambda_1;       },
                             [](      RodMaterial &m, Real val) {        m.bendingStiffness.lambda_1 = val; })
        .def_property("B22", [](const RodMaterial &m          ) { return m.bendingStiffness.lambda_2;       },
                             [](      RodMaterial &m, Real val) {        m.bendingStiffness.lambda_2 = val; })
        .def_property("I11", [](const RodMaterial &m          ) { return m.momentOfInertia. lambda_1;       },
                             [](      RodMaterial &m, Real val) {        m.momentOfInertia. lambda_1 = val; })
        .def_property("I22", [](const RodMaterial &m          ) { return m.momentOfInertia. lambda_2;       },
                             [](      RodMaterial &m, Real val) {        m.momentOfInertia. lambda_2 = val; })
        .def(py::pickle([](const RodMaterial &mat) {
                    return py::make_tuple(mat.area, mat.stretchingStiffness, mat.twistingStiffness,
                                          mat.bendingStiffness, mat.momentOfInertia,
                                          mat.torsionStressCoefficient, mat.youngModulus, mat.shearModulus,
                                          mat.crossSectionHeight,
                                          mat.crossSectionBoundaryPts,
                                          mat.crossSectionBoundaryEdges,
                                          mat.stressAnalysisPtr());
                },
                [](const py::tuple &t) {
                    if (t.size() < 11 || t.size() > 12) throw std::runtime_error("Invalid state!");
                    RodMaterial mat;
                    mat.area                      = t[0 ].cast<Real>();
                    mat.stretchingStiffness       = t[1 ].cast<Real>();
                    mat.twistingStiffness         = t[2 ].cast<Real>();
                    mat.bendingStiffness          = t[3 ].cast<RodMaterial::DiagonalizedTensor>();
                    mat.momentOfInertia           = t[4 ].cast<RodMaterial::DiagonalizedTensor>();
                    mat.torsionStressCoefficient  = t[5 ].cast<Real>();
                    mat.youngModulus              = t[6 ].cast<Real>();
                    mat.shearModulus              = t[7 ].cast<Real>();
                    mat.crossSectionHeight        = t[8 ].cast<Real>();
                    mat.crossSectionBoundaryPts   = t[9 ].cast<CrossSection::AlignedPointCollection>();
                    mat.crossSectionBoundaryEdges = t[10].cast<std::vector<std::pair<size_t, size_t>>>();

                    if (t.size() < 12) return mat;

                    mat.setStressAnalysisPtr(t[11].cast<std::shared_ptr<CSA>>());

                    return mat;
                }))
        .def("fromGHState", [](Real area, Real stretchingStifness, Real twistingStiffness, const std::vector<Real> &bendingStiffness, const std::vector<Real> &momentOfInertia,
                               Real torsionStressCoefficient, Real youngModulus, Real shearModulus, Real crossSectionHeight, const std::vector<Real> &ptCoords, const std::vector<size_t> &edges) 
            {
                RodMaterial mat;
                mat.area                        = area;
                mat.stretchingStiffness         = stretchingStifness;
                mat.twistingStiffness           = twistingStiffness;
                mat.torsionStressCoefficient    = torsionStressCoefficient;
                mat.youngModulus                = youngModulus;
                mat.shearModulus                = shearModulus;
                mat.crossSectionHeight          = crossSectionHeight;

                RodMaterial::DiagonalizedTensor bStiffness;
                bStiffness.lambda_1 = bendingStiffness[0];
                bStiffness.lambda_2 = bendingStiffness[1];
                mat.bendingStiffness = bStiffness;

                RodMaterial::DiagonalizedTensor mInertia;
                mInertia.lambda_1 = momentOfInertia[0];
                mInertia.lambda_2 = momentOfInertia[1];
                mat.momentOfInertia = mInertia;

                CrossSection::AlignedPointCollection pts;
                size_t count = ptCoords.size()/2;
                for(size_t i=0; i<count; i++) pts.emplace_back(ptCoords[i*2], ptCoords[i*2+1]);
                mat.crossSectionBoundaryPts = pts;

                std::vector<std::pair<size_t, size_t>> edgePairs;
                count = edges.size()/2;
                for(size_t i=0; i<count; i++) edgePairs.emplace_back(edges[i*2], edges[i*2+1]);
                mat.crossSectionBoundaryEdges = edgePairs;
                
                return mat;
            })
        ;
    py::class_<RodMaterial::DiagonalizedTensor>(m, "DiagonalizedTensor")
        .def_readwrite("lambda_1", &RodMaterial::DiagonalizedTensor::lambda_1)
        .def_readwrite("lambda_2", &RodMaterial::DiagonalizedTensor::lambda_2)
        .def("trace", &RodMaterial::DiagonalizedTensor::trace)
        .def(py::pickle([](const RodMaterial::DiagonalizedTensor &d) { return py::make_tuple(d.lambda_1, d.lambda_2); },
                        [](const py::tuple &t) {
                            if (t.size() != 2) throw std::runtime_error("Invalid state!");
                            RodMaterial::DiagonalizedTensor result;
                            result.lambda_1 = t[0].cast<Real>();
                            result.lambda_2 = t[1].cast<Real>();
                            return result;
                        }))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // RectangularBoxCollection
    ////////////////////////////////////////////////////////////////////////////////
    auto rectangular_box_collection = py::class_<RectangularBoxCollection>(m, "RectangularBoxCollection")
        .def(py::init<std::vector<RectangularBoxCollection::Corners>>(), py::arg("box_corners"))
        .def(py::init<const std::string>(), py::arg("path"))
        .def("contains", &RectangularBoxCollection::contains, py::arg("p"))
        .def("visualizationGeometry", &getVisualizationGeometry<RectangularBoxCollection>)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // PeriodicRod
    ////////////////////////////////////////////////////////////////////////////////
    auto periodic_rod = py::class_<PeriodicRod>(m, "PeriodicRod")
        .def(py::init<std::vector<Point3D>, bool>(), py::arg("pts"), py::arg("zeroRestCurvature"))
        .def("setMaterial",            &PeriodicRod::setMaterial, py::arg("material"))
        .def("numDoF",                 &PeriodicRod::numDoF)
        .def("setDoFs",                &PeriodicRod::setDoFs,  py::arg("dofs"))
        .def("getDoFs",                &PeriodicRod::getDoFs)
        .def("energy",                 &PeriodicRod::energy,   py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("gradient",               &PeriodicRod::gradient, py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("hessianSparsityPattern", &PeriodicRod::hessianSparsityPattern, py::arg("val") = 0.0)
        .def("hessian",                [](const PeriodicRod &r, PeriodicRod::EnergyType etype) {
                PeriodicRod::CSCMat H;
                r.hessian(H, etype);
                return H; },  py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("thetaOffset",  &PeriodicRod::thetaOffset)
        .def_readonly("rod", &PeriodicRod::rod, py::return_value_policy::reference)
        .def_property("twist", &PeriodicRod::twist, &PeriodicRod::setTwist, "Twist discontinuity passing from last edge back to (overlapping) first")
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // RodLinkage
    ////////////////////////////////////////////////////////////////////////////////
    auto rod_linkage = py::class_<RodLinkage>(m, "RodLinkage")
        .def(py::init<const Eigen::MatrixX3d &, const Eigen::MatrixX2i &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>>(), py::arg("points"), py::arg("edges"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>())
        .def(py::init<const std::string &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>>(), py::arg("path"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>())
        .def(py::init<const RodLinkage &>(), "Copy constructor", py::arg("rod"))

        .def("set", (void (RodLinkage::*)(const Eigen::MatrixX3d &, const Eigen::MatrixX2i &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>))(&RodLinkage::set), py::arg("points"), py::arg("edges"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>()) // py::overload_cast fails
        .def("set", (void (RodLinkage::*)(const std::string &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>))(&RodLinkage::set), py::arg("path"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>()) // py::overload_cast fails
        .def("set_interleaving_type", &RodLinkage::set_interleaving_type, py::arg("new_type"), "Configure the rods' interleaving type; input parameter should be one xshell, weaving, and noOffset.")
        .def("setBendingEnergyType", &RodLinkage::setBendingEnergyType, py::arg("betype"), "Configure the rods' bending energy type.")

        .def("energyStretch", &RodLinkage::energyStretch, "Compute stretching energy")
        .def("energyBend",    &RodLinkage::energyBend   , "Compute bending    energy")
        .def("energyTwist",   &RodLinkage::energyTwist  , "Compute twisting   energy")
        .def("energy",        py::overload_cast<ElasticRod::EnergyType>(&RodLinkage::energy, py::const_), "Compute elastic energy", py::arg("energyType") = ElasticRod::EnergyType::Full)

        .def("updateSourceFrame", &RodLinkage::updateSourceFrame, "Use the current reference frame as the source for parallel transport")
        .def("updateRotationParametrizations", &RodLinkage::updateRotationParametrizations, "Update the joint rotation variables to represent infinitesimal rotations around the current frame")
        .def_readwrite("disableRotationParametrizationUpdates", &RodLinkage::disableRotationParametrizationUpdates)

        .def("rivetForces", &RodLinkage::rivetForces, "Compute the forces exerted by the A rods on the system variables.", py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("needTorque") = true)
        .def("rivetNetForceAndTorques", &RodLinkage::rivetNetForceAndTorques, "Compute the forces/torques exerted by the A rods on the center of each joint.", py::arg("energyType") = ElasticRod::EnergyType::Full)

        .def("gradient", &RodLinkage::gradient, "Elastic energy gradient", py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("hessian",  py::overload_cast<ElasticRod::EnergyType, bool>(&RodLinkage::hessian, py::const_), "Elastic energy  hessian", py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false)
        .def("applyHessian", &RodLinkage::applyHessian, "Elastic energy Hessian-vector product formulas.", py::arg("v"), py::arg("variableDesignParameters") = false, py::arg("mask") = HessianComputationMask())

        .def("gradientPerSegmentRestlen", &RodLinkage::gradientPerSegmentRestlen, "Elastic energy gradient for per segment rest length", py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("hessianPerSegmentRestlen",  py::overload_cast<ElasticRod::EnergyType>(&RodLinkage::hessianPerSegmentRestlen, py::const_), "Elastic energy  hessian for per segment rest length", py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("applyHessianPerSegmentRestlen", &RodLinkage::applyHessianPerSegmentRestlen, "Elastic energy Hessian-vector product formulas for per segment rest length.", py::arg("v"), py::arg("mask") = HessianComputationMask())

        .def("designParameterSolve_energy", &RodLinkage::designParameterSolve_energy, "Potential energy used for the design parameter solve (neglects surface-attraction term)")
        .def("grad_design_parameters", &RodLinkage::grad_design_parameters, "Elastic gradient with respect to the design parameters only.", py::arg("updatedSource") = false)
        .def("setDesignParameters", &RodLinkage::setDesignParameters, "Set the design parameter dof in the linkage.", py::arg("p"))
        .def("totalRestLength",   &RodLinkage::totalRestLength)
        .def("massMatrix",        py::overload_cast<bool, bool>(&RodLinkage::massMatrix, py::const_), py::arg("updatedSource") = false, py::arg("useLumped") = false)

        .def("characteristicLength", &RodLinkage::characteristicLength)
        .def("approxLinfVelocity",   &RodLinkage::approxLinfVelocity)

        .def("hessianNNZ",             &RodLinkage::hessianNNZ,             "Tight upper bound for nonzeros in the Hessian.",                            py::arg("variableDesignParameters") = false)
        .def("hessianSparsityPattern", &RodLinkage::hessianSparsityPattern, "Compressed column matrix containing all potential nonzero Hessian entries", py::arg("variableDesignParameters") = false, py::arg("val") = 0.0)
        .def("designParameterSolve_hessianSparsityPattern", &RodLinkage::designParameterSolve_hessianSparsityPattern, "Hessian sparsity pattern for design parameter solve with smoothing terms.")        
        .def("segment", py::overload_cast<size_t>(&RodLinkage::segment), py::return_value_policy::reference)
        .def("_set_segment", &RodLinkage::set_segment, "Set particular rod segment. Use with caution! Need to update segment to edge map.", py::arg("new_seg"), py::arg("i"))
        .def("joint",   py::overload_cast<size_t>(&RodLinkage::joint),   py::return_value_policy::reference)

        .def("segments", [](const RodLinkage &l) { return py::make_iterator(l.segments().cbegin(), l.segments().cend()); })
        .def("joints",   [](const RodLinkage &l) { return py::make_iterator(l.joints  ().cbegin(), l.joints  ().cend()); })

        .def("traceRods",   &RodLinkage::traceRods)
        .def("rodStresses", &RodLinkage::rodStresses)
        .def("florinVisualizationGeometry", [](const RodLinkage &l) {
                std::vector<std::vector<size_t>> polylinesA, polylinesB;
                std::vector<Eigen::Vector3d> points, normals;
                std::vector<double> stresses;
                l.florinVisualizationGeometry(polylinesA, polylinesB, points, normals, stresses);
                return py::make_tuple(polylinesA, polylinesB, points, normals, stresses);
            })

        .def("getDoFs",   &RodLinkage::getDoFs)
        .def("setDoFs", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool, bool>(&RodLinkage::setDoFs), py::arg("values"), py::arg("spatialCoherence") = false, py::arg("initializeOffset") = false)

        .def("getExtendedDoFs", &RodLinkage::getExtendedDoFs)
        .def("setExtendedDoFs", &RodLinkage::setExtendedDoFs, py::arg("values"), py::arg("spatialCoherence") = false)

        .def("getExtendedDoFsPSRL", &RodLinkage::getExtendedDoFsPSRL)
        .def("setExtendedDoFsPSRL", &RodLinkage::setExtendedDoFsPSRL, py::arg("values"), py::arg("spatialCoherence") = false)
        .def("getPerSegmentRestLength", &RodLinkage::getPerSegmentRestLength)

        .def("setPerSegmentRestLength", &RodLinkage::setPerSegmentRestLength, py::arg("values"))

        .def("getDesignParameters", &RodLinkage::getDesignParameters)
        .def("setDesignParameters", &RodLinkage::setDesignParameters, py::arg("p"))
        .def("swapJointAngleDefinitions", &RodLinkage::swapJointAngleDefinitions)
        .def_property("averageJointAngle", [](const RodLinkage &l)                   { return l.getAverageJointAngle(); },
                                           [](      RodLinkage &l, const Real alpha) { l.setAverageJointAngle(alpha);   })

        .def("set_design_parameter_config", &RodLinkage::setDesignParameterConfig, py::arg("use_restLen"), py::arg("use_restKappa"), py::arg("update_designParams_cache") = true)
        .def("get_design_parameter_config", &RodLinkage::getDesignParameterConfig)

        .def("setMaterial",               &RodLinkage::setMaterial, py::arg("material"))
        .def("setJointMaterials",         &RodLinkage::setJointMaterials, py::arg("jointMaterials"))
        .def("homogenousMaterial",        &RodLinkage::homogenousMaterial)

        .def("stiffenRegions",            &RodLinkage::stiffenRegions)
        .def("saveVisualizationGeometry", &RodLinkage::saveVisualizationGeometry, py::arg("path"), py::arg("averagedMaterialFrames") = false, py::arg("averagedCrossSections") = false)
        .def("saveStressVisualization",   &RodLinkage::saveStressVisualization)
        .def("writeRodDebugData",         py::overload_cast<const std::string &, const size_t>(&RodLinkage::writeRodDebugData, py::const_), py::arg("path"), py::arg("singleRod") = size_t(RodLinkage::NONE))
        .def("writeLinkageDebugData",     &RodLinkage::writeLinkageDebugData)
        .def("writeTriangulation",        &RodLinkage::writeTriangulation)

        // Outputs mesh with normals
        .def("visualizationGeometry", &getVisualizationGeometry<RodLinkage>, py::arg("averagedMaterialFrames") = true, py::arg("averagedCrossSections") = true)
        .def("visualizationGeometryHeightColors", &getVisualizationGeometryCSHeightField<RodLinkage>, "Get a per-visualization-vertex field representing height above the centerline")

        .def("hasCrossSection",     &RodLinkage::    hasCrossSection)
        .def("hasCrossSectionMesh", &RodLinkage::hasCrossSectionMesh)
        .def("meshCrossSection",    &RodLinkage::   meshCrossSection, py::arg("triArea")=0.001)
        .def("maxVonMisesStresses", &RodLinkage::maxVonMisesStresses)
        .def("sqrtBendingEnergies", &RodLinkage::sqrtBendingEnergies)
        .def("stretchingStresses",  &RodLinkage:: stretchingStresses)
        .def("stretchingEnergies",  &RodLinkage:: stretchingEnergies)
        .def("maxBendingStresses",  &RodLinkage:: maxBendingStresses)
        .def("minBendingStresses",  &RodLinkage:: minBendingStresses)
        .def("twistingStresses",    &RodLinkage::   twistingStresses)
        .def("twistingEnergies",    &RodLinkage::   twistingEnergies)

        .def("visualizationField", [](const RodLinkage &r, const std::vector<Eigen::VectorXd>  &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))
        .def("visualizationField", [](const RodLinkage &r, const std::vector<Eigen::MatrixX3d> &f) { return getVisualizationField(r, f); }, "Convert a per-vertex or per-edge field into a per-visualization-geometry field (called internally by MeshFEM visualization)", py::arg("perEntityField"))

        .def("numDoF",                  &RodLinkage::numDoF)
        .def("numSegments",             &RodLinkage::numSegments)
        .def("numJoints",               &RodLinkage::numJoints)
        .def("jointPositionDoFIndices", &RodLinkage::jointPositionDoFIndices)
        .def("jointAngleDoFIndices",    &RodLinkage::jointAngleDoFIndices)
        .def("jointDoFIndices",         &RodLinkage::jointDoFIndices)
        .def("designParameterSolveFixedVars",        &RodLinkage::designParameterSolveFixedVars)
        .def("lengthVars",              &RodLinkage::lengthVars, py::arg("variableRestLen") = false)

        .def("restLengthLaplacianEnergy", &RodLinkage::restLengthLaplacianEnergy)
        .def("getRestLengths",            &RodLinkage::getRestLengths)
        .def("minRestLength",             &RodLinkage::minRestLength)
        .def("averageRestLengths",        &RodLinkage::averageRestLengths)

        .def("numExtendedDoF", &RodLinkage::numExtendedDoF)
        .def("numExtendedDoFPSRL", &RodLinkage::numExtendedDoFPSRL)
        .def("restLenOffset",  &RodLinkage::restLenOffset)

        .def("numRestKappaVars", &RodLinkage::numRestKappaVars)
        .def("getRestKappaVars", &RodLinkage::getRestKappaVars)
        .def("setRestKappaVars", &RodLinkage::setRestKappaVars, py::arg("params"), py::arg("offset") = 0)
        .def("averageAbsRestKappaVars", &RodLinkage::averageAbsRestKappaVars)

        .def("numRestLengths", &RodLinkage::numRestLengths)
        .def("numFreeRestLengths", &RodLinkage::numFreeRestLengths)
        .def("numJointRestLengths", &RodLinkage::numJointRestLengths)
        .def("numCenterlinePos", &RodLinkage::numCenterlinePos)
        .def("centerLinePositions", &RodLinkage::centerLinePositions)
        .def("centralJoint",   &RodLinkage::centralJoint)
        .def("jointPositions", &RodLinkage::jointPositions)
        .def("deformedPoints", &RodLinkage::deformedPoints)

        .def("dofOffsetForJoint",            &RodLinkage::dofOffsetForJoint,            py::arg("index"))
        .def("dofOffsetForSegment",          &RodLinkage::dofOffsetForSegment,          py::arg("index"))
        .def("dofOffsetForCenterlinePos",    &RodLinkage::dofOffsetForCenterlinePos,    py::arg("index"))
        .def("restLenDofOffsetForJoint",     &RodLinkage::restLenDofOffsetForJoint,     py::arg("index"))
        .def("restLenDofOffsetForSegment",   &RodLinkage::restLenDofOffsetForSegment,   py::arg("index"))
        .def("restKappaDofOffsetForSegment", &RodLinkage::restKappaDofOffsetForSegment, py::arg("index"))

        .def("getTerminalEdgeSensitivity", [](RodLinkage &l, size_t si, int which, bool updatedSource, bool evalHessian) {
                    if ((which < 0) || (which > 1)) throw std::runtime_error("`which` must be 0 (start) or 1 (end)");
                    return l.getTerminalEdgeSensitivity(si, static_cast<RodLinkage::TerminalEdge>(which), updatedSource, evalHessian);
                }, py::arg("si"), py::arg("which"), py::arg("updatedSource"), py::arg("evalHessian"))

        .def("segmentRestLenToEdgeRestLenMapTranspose", &RodLinkage::segmentRestLenToEdgeRestLenMapTranspose)
        .def("constructSegmentRestLenToEdgeRestLenMapTranspose", &RodLinkage::constructSegmentRestLenToEdgeRestLenMapTranspose, py::arg("segmentRestLenGuess"))
        .def(py::pickle([](const RodLinkage &l) { return py::make_tuple(l.joints(), l.segments(), l.homogenousMaterial(), l.initialMinRestLength(), l.segmentRestLenToEdgeRestLenMapTranspose(), l.getPerSegmentRestLength(), l.getDesignParameterConfig()); },
                        [](const py::tuple &t) {
                            if (t.size() != 7) throw std::runtime_error("Invalid RodLinkage state!");
                            return std::make_unique<RodLinkage>(t[0].cast<std::vector<RodLinkage::Joint>>(),
                                                                t[1].cast<std::vector<RodLinkage::RodSegment>>(),
                                                                t[2].cast<RodMaterial>(),
                                                                t[3].cast<Real>(),
                                                                t[4].cast<SuiteSparseMatrix>(),
                                                                t[5].cast<Eigen::VectorXd>(),
                                                                t[6].cast<DesignParameterConfig>());

                        }))

        .def("fromGHState", [](const std::vector<RodLinkage::Joint> &joints, const std::vector<RodLinkage::RodSegment> &segments,
             const RodMaterial &homogMat, Real initMinRL, const std::vector<Real> &Ax, const std::vector<size_t> &Ai, const std::vector<size_t> &Ap, 
             const size_t M, const size_t N, const size_t NZ, Eigen::VectorXd &perSegmentRestLength, const bool use_restLength, const bool use_restKappa) 
              { 
                    SuiteSparseMatrix segmentRestLenToEdgeRestLenMapTranspose(M, N);
                    segmentRestLenToEdgeRestLenMapTranspose.nz = NZ;
                    auto &AAi = segmentRestLenToEdgeRestLenMapTranspose.Ai;
                    auto &AAx = segmentRestLenToEdgeRestLenMapTranspose.Ax;
                    auto &AAp = segmentRestLenToEdgeRestLenMapTranspose.Ap;

                    AAi.reserve(NZ);
                    AAx.reserve(NZ);
                    for(size_t i=0; i<NZ; i++){
                        AAi.push_back(Ai[i]);
                        AAx.push_back(Ax[i]);
                    }
                    AAp.reserve(N + 1);
                    for(size_t i=0; i<=N; i++) AAp.push_back(Ap[i]);

                    DesignParameterConfig dpc;
                    dpc.restLen = use_restLength;
                    dpc.restKappa = use_restKappa;

                    auto l = std::make_unique<RodLinkage>(joints,segments,homogMat,initMinRL,segmentRestLenToEdgeRestLenMapTranspose, perSegmentRestLength, dpc);
                    l->setMaterial(homogMat);
                    return l;
              })
        ;

    auto py_joint = py::class_<RodLinkage::Joint>(rod_linkage, "Joint");

    py::enum_<RodLinkage::Joint::Type>(py_joint, "Type")
        .value("PASSTHROUGH", RodLinkage::Joint::Type::PASSTHROUGH)
        .value("A_OVER_B",    RodLinkage::Joint::Type::A_OVER_B)
        .value("B_OVER_A",    RodLinkage::Joint::Type::B_OVER_A)
        ;

    py_joint
        .def("valence",         &RodLinkage::Joint::valence)
        .def_property("position", [](const RodLinkage::Joint &j) { return j.pos  (); }, [](RodLinkage::Joint &j, const Vector3D &v) { j.set_pos  (v); })
        .def_property("omega",    [](const RodLinkage::Joint &j) { return j.omega(); }, [](RodLinkage::Joint &j, const Vector3D &v) { j.set_omega(v); })
        .def_property("alpha",    [](const RodLinkage::Joint &j) { return j.alpha(); }, [](RodLinkage::Joint &j,            Real a) { j.set_alpha(a); })
        .def_property("len_A",    [](const RodLinkage::Joint &j) { return j.len_A(); }, [](RodLinkage::Joint &j,            Real l) { j.set_len_A(l); })
        .def_property("len_B",    [](const RodLinkage::Joint &j) { return j.len_B(); }, [](RodLinkage::Joint &j,            Real l) { j.set_len_B(l); })
        .def_property_readonly("normal",        [](const RodLinkage::Joint &j) { return j.normal(); })
        .def_property_readonly("e_A",           [](const RodLinkage::Joint &j) { return j.e_A(); })
        .def_property_readonly("e_B",           [](const RodLinkage::Joint &j) { return j.e_B(); })
        .def_property_readonly("source_t_A",    [](const RodLinkage::Joint &j) { return j.source_t_A(); })
        .def_property_readonly("source_t_B",    [](const RodLinkage::Joint &j) { return j.source_t_B(); })
        .def_property_readonly("source_normal", [](const RodLinkage::Joint &j) { return j.source_normal(); })
        .def_property_readonly("segments_A",    [](const RodLinkage::Joint &j) { return j.segmentsA(); })
        .def_property_readonly("segments_B",    [](const RodLinkage::Joint &j) { return j.segmentsB(); })

        .def_property_readonly("numSegmentsA", [](const RodLinkage::Joint &j) { return j.numSegmentsA(); })
        .def_property_readonly("numSegmentsB", [](const RodLinkage::Joint &j) { return j.numSegmentsB(); })
        .def_property_readonly("isStartA",     [](const RodLinkage::Joint &j) { return j.isStartA(); })
        .def_property_readonly("isStartB",     [](const RodLinkage::Joint &j) { return j.isStartB(); })
        .def("connectingSegment", &RodLinkage::Joint::connectingSegment, py::arg("ji"))
        .def("terminalEdgeNormalSign", &RodLinkage::Joint::terminalEdgeNormalSign, py::arg("segmentIdx"))
        .def("set_terminalEdgeNormalSign", &RodLinkage::Joint::set_terminalEdgeNormalSign, py::arg("segmentIdx"), py::arg("sign"))
        .def("continuationSegment", &RodLinkage::Joint::continuationSegment, py::arg("segmentIdx"))
        .def("neighbors", [](const RodLinkage::Joint &j) {
                    std::vector<size_t> result;
                    j.visitNeighbors([&result](size_t ji, size_t, size_t) { result.push_back(ji); });
                    return result;
            })

        .def_readwrite("type", &RodLinkage::Joint::type) // Read only for now--currently the user would need to manually trigger an update with setDoFs(getDoFs()).

        .def(py::pickle([](const RodLinkage::Joint &joint) { return to_pytuple(joint.getState()); },
                        [](const py::tuple &t) {
                            return from_pytuple<RodLinkage::Joint::SerializedState>(t);
                        }))

        .def("fromGHState", [](const Pt3_T<Real> &pos, const Vec3_T<Real> omega, const Real alpha, const Real lenA, const Real lenB, const Real signB, 
              const Vec3_T<Real> &t, const Vec3_T<Real> &norm, const std::array<int, 2> &segmentsA, const std::array<int, 2> &segmentsB, 
              const std::array<bool  , 2> &isStartA, const std::array<bool  , 2> &isStartB, size_t &type, const std::array<int, 4> &normalSigns) 
              { 
                    std::array<size_t, 2> sA{{RodLinkage::NONE, RodLinkage::NONE}}, sB{{RodLinkage::NONE, RodLinkage::NONE}};
                    for(int i=0;i<2;i++)
                    {
                        if(segmentsA[i]>-1) sA[i] = segmentsA[i];
                        if(segmentsB[i]>-1) sB[i] = segmentsB[i];
                    }

                    const RodLinkage::Joint::SerializedState state = std::make_tuple(pos, omega, alpha, lenA, lenB, signB, t, norm, sA, sB, isStartA, isStartB, RodLinkage::Joint::Type(type), normalSigns);
                    return RodLinkage::Joint(state);
              })
        ;

    py::class_<RodLinkage::RodSegment>(rod_linkage, "RodSegment")
        .def(py::init<const Pt3_T<Real> &, const Pt3_T<Real> &, size_t>(), py::arg("startPt"), py::arg("endPt"), py::arg("nsubdiv"))
        .def("hasStartJoint", &RodLinkage::RodSegment::hasStartJoint)
        .def("hasEndJoint",   &RodLinkage::RodSegment::hasEndJoint)
        .def_readonly("rod",        &RodLinkage::RodSegment::rod, py::return_value_policy::reference)
        .def_readwrite("startJoint", &RodLinkage::RodSegment::startJoint)
        .def_readwrite("endJoint",   &RodLinkage::RodSegment::endJoint)
        .def("setMinimalTwistThetas", &RodLinkage::RodSegment::setMinimalTwistThetas, py::arg("verbose") = false)
        .def(py::pickle([](const RodLinkage::RodSegment &s) { return py::make_tuple(s.startJoint, s.endJoint, s.rod); },
                        [](const py::tuple &t) {
                            if (t.size() != 3) throw std::runtime_error("Invalid RodLinkage::RodSegment state!");
                            return RodLinkage::RodSegment(t[0].cast<size_t>(), t[1].cast<size_t>(), t[2].cast<ElasticRod>());
                        }))

        .def("fromGHState", [](const int startJoint, const int endJoint, const std::vector<Pt3_T<Real>> &pts, const std::vector<Real> &dirCoords, const std::vector<Real> &restKappas,
                               const std::vector<Real> &restTwists, const std::vector<Real> &restLengths, std::vector<RodMaterial> materials, std::vector<Real> bendingStiffness1, std::vector<Real> bendingStiffness2, 
                               const std::vector<Real> &twistingStiffness, const std::vector<Real> &stretchingStiffness, const ElasticRod::BendingEnergyType &energyType,
                               const ElasticRod::DeformedState &deformedState, const std::vector<Real> &densities, Real initialMinRestLen) 
              {                        
                    size_t idxStart = RodLinkage::NONE;
                    if(startJoint!=-1) idxStart = startJoint;
                    size_t idxEnd = RodLinkage::NONE;
                    if(endJoint!=-1) idxEnd = endJoint;
         
                    ElasticRod&& r(pts);

                    // Rest Directors
                    size_t count = dirCoords.size()/6;
                    std::vector<ElasticRod::Directors> dir;
                    dir.reserve(count);
                    for(size_t i=0; i<count; i++){
                        dir.emplace_back(Eigen::Vector3d(dirCoords[i*6], dirCoords[i*6+1],dirCoords[i*6+2]),
                                         Eigen::Vector3d(dirCoords[i*6+3], dirCoords[i*6+4],dirCoords[i*6+5]));
                    }
                    r.setRestDirectors(dir); 

                    // Rest Kappas
                    count = restKappas.size()/2;
                    CrossSection::AlignedPointCollection kData;
                    kData.reserve(count);
                    for(size_t i=0; i<count; i++){
                        kData.emplace_back(restKappas[i*2], restKappas[i*2+1]);
                    }
                    r.setRestKappas(ElasticRod::StdVectorVector2D(kData));

                    // Materials
                    if(materials.size()==1) r.setMaterial(materials[0]);
                    else r.setMaterial(materials);

                    // Bending stiffness
                    std::vector<RodMaterial::BendingStiffness> bStiffness;
                    count = bendingStiffness1.size();
                    for(size_t i=0; i<count; i++){
                        RodMaterial::BendingStiffness b;
                        b.lambda_1 = bendingStiffness1[i];
                        b.lambda_2 = bendingStiffness2[i];
                        bStiffness.push_back(b);
                    }
                    r.setBendingStiffnesses(bStiffness);
                    // Rest Twists
                    r.setRestTwists(restTwists);
                    // Rest Lengths
                    r.setRestLengths(restLengths);
                    // Twisting stiffness
                    r.setTwistingStiffnesses(twistingStiffness);
                    // Stretching stiffness
                    r.setStretchingStiffnesses(stretchingStiffness);
                    // Energy type
                    r.setBendingEnergyType(energyType);  
                    // Deformed state
                    r.setDeformedConfiguration(deformedState);
                    // Densities
                    r.setDensities(densities);
                    // Initial min rest length
                    r.setInitialMinRestLen(initialMinRestLen); 

                    return RodLinkage::RodSegment(idxStart, idxEnd, std::move(r));
              })
        ;

    ////////////////////////////////////////////////////////////////////////////
    // LinkageTerminalEdgeSensitivity
    ////////////////////////////////////////////////////////////////////////////
    using LTES = LinkageTerminalEdgeSensitivity<Real>;
    py::class_<LTES>(detail_module, "LinkageTerminalEdgeSensitivity")
        .def_readonly("j",                    &LTES::j)
        .def_readonly("is_A",                 &LTES::is_A)
        .def_readonly("s_jA",                 &LTES::s_jA)
        .def_readonly("s_jB",                 &LTES::s_jB)
        .def_readonly("normal_sign",          &LTES::normal_sign)
        .def_readonly("crossingNormalOffset", &LTES::crossingNormalOffset)
        .def_readonly("jacobian",             &LTES::jacobian)
        .def_readonly("hessian",              &LTES::hessian)
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // SurfaceAttractedLinkage
    ////////////////////////////////////////////////////////////////////////////////
    auto surface_attracted_linkage = py::class_<SurfaceAttractedLinkage, RodLinkage>(m, "SurfaceAttractedLinkage")
        .def(py::init<const std::string &, const bool &, const Eigen::MatrixX3d &, const Eigen::MatrixX2i &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>>(), py::arg("surface_path"), py::arg("useCenterline"), py::arg("points"), py::arg("edges"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>())
        .def(py::init<const Eigen::MatrixX3d &, const Eigen::MatrixX3i &, const bool &, const Eigen::MatrixX3d &, const Eigen::MatrixX2i &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>>(), py::arg("target_vertices"), py::arg("target_faces"), py::arg("useCenterline"), py::arg("points"), py::arg("edges"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>())
        .def(py::init<const std::string &, const bool &, const RodLinkage &>(), "Copy constructor with rod linkage and path to target", py::arg("surface_path"), py::arg("useCenterline"), py::arg("rod"))
        .def(py::init<const std::string &, const bool &, const SurfaceAttractedLinkage &>(), "Copy constructor with surface attracted linkage and path to target surface", py::arg("surface_path"), py::arg("useCenterline"), py::arg("surface_attracted_linkage"))
        .def(py::init<const Eigen::MatrixX3d &, const Eigen::MatrixX3i &, const bool &, const RodLinkage &>(), "Copy constructor with rod linkage", py::arg("target_vertices"), py::arg("target_faces"), py::arg("useCenterline"), py::arg("rod"))
        .def(py::init<const Eigen::MatrixX3d &, const Eigen::MatrixX3i &, const bool &, const SurfaceAttractedLinkage &>(), "Copy constructor with surface attracted linkage and surface", py::arg("target_vertices"), py::arg("target_faces"), py::arg("useCenterline"), py::arg("surface_attracted_linkage"))
        .def(py::init<const SurfaceAttractedLinkage &>(), "Copy constructor with surface attracted linkage only", py::arg("surface_attracted_linkage"))
        .def(py::init<const std::string &, const bool &, const std::string &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>>(), py::arg("surface_path"), py::arg("useCenterline"), py::arg("linkage_path"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>())
        .def("set", (void (SurfaceAttractedLinkage::*)(const std::string &, const bool &, const std::string &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>))(&SurfaceAttractedLinkage::set), py::arg("surface_path"), py::arg("useCenterline"), py::arg("linkage_path"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>()) // py::overload_cast fails
        .def("energy", &SurfaceAttractedLinkage::energy, "Compute potential energy (elastic + surface attraction)", py::arg("energyType") = SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Full)
        .def("gradient", &SurfaceAttractedLinkage::gradient, "Potential energy gradient", py::arg("updatedSource") = false, py::arg("energyType") = SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Full, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("gradientBend", &SurfaceAttractedLinkage::gradientBend, "Potential Bending Energy Gradient", py::arg("updatedSource") = false, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("gradientTwist", &SurfaceAttractedLinkage::gradientTwist, "Potential Twisting Energy Gradient", py::arg("updatedSource") = false, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("gradientStretch", &SurfaceAttractedLinkage::gradientStretch, "Potential Stretching Energy Gradient", py::arg("updatedSource") = false, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("hessian",  py::overload_cast<SurfaceAttractedLinkage::SurfaceAttractionEnergyType, bool>(&SurfaceAttractedLinkage::hessian, py::const_), "Potential energy  hessian", py::arg("energyType") = SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Full, py::arg("variableDesignParameters") = false)
        .def("applyHessian", &SurfaceAttractedLinkage::applyHessian, "Potential energy Hessian-vector product formulas.", py::arg("v"), py::arg("variableDesignParameters") = false, py::arg("mask") = HessianComputationMask(), py::arg("energyType") = SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Full)
        .def("gradientPerSegmentRestlen", &SurfaceAttractedLinkage::gradientPerSegmentRestlen, "Potential energy gradient for per segment rest length", py::arg("updatedSource") = false, py::arg("energyType") = SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Full)
        .def("hessianPerSegmentRestlen",  py::overload_cast<SurfaceAttractedLinkage::SurfaceAttractionEnergyType>(&SurfaceAttractedLinkage::hessianPerSegmentRestlen, py::const_), "Potential energy  hessian for per segment rest length", py::arg("energyType") = SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Full)
        .def("applyHessianPerSegmentRestlen", &SurfaceAttractedLinkage::applyHessianPerSegmentRestlen, "Potential energy Hessian-vector product formulas for per segment rest length.", py::arg("v"), py::arg("mask") = HessianComputationMask(), py::arg("energyType") = SurfaceAttractedLinkage::SurfaceAttractionEnergyType::Full)
        .def_readwrite("attraction_weight", &SurfaceAttractedLinkage::attraction_weight, "Trade off between surface attraction and elastic energy")
        .def("set_attraction_tgt_joint_weight", &SurfaceAttractedLinkage::set_attraction_tgt_joint_weight, "Trade off between fitting target joint position and fitting the surface inside the potential energy of the linkage.", py::arg("weight"))
        .def("get_attraction_tgt_joint_weight", &SurfaceAttractedLinkage::get_attraction_tgt_joint_weight)
        .def("setDoFs", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool>(&SurfaceAttractedLinkage::setDoFs), py::arg("values"), py::arg("spatialCoherence") = false)
        .def("setExtendedDoFs", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool>(&SurfaceAttractedLinkage::setExtendedDoFs), py::arg("values"), py::arg("spatialCoherence") = false)
        .def("setExtendedDoFsPSRL", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool>(&SurfaceAttractedLinkage::setExtendedDoFsPSRL), py::arg("values"), py::arg("spatialCoherence") = false)
        .def("get_linkage_closest_point", &SurfaceAttractedLinkage::get_linkage_closest_point, "The list of closest point to the linkage.")
        .def("set_use_centerline", &SurfaceAttractedLinkage::set_use_centerline, "Choose to use centerline position or joint position in target surface fitter.", py::arg("useCenterline"))
        .def("get_holdClosestPointsFixed", &SurfaceAttractedLinkage::get_holdClosestPointsFixed, "Choose to update the closest point in the target surface fitting term during each equilibrium iteration.")
        .def("set_holdClosestPointsFixed", &SurfaceAttractedLinkage::set_holdClosestPointsFixed, "Choose to update the closest point in the target surface fitting term during each equilibrium iteration.", py::arg("set_holdClosestPointsFixed"))
        .def("get_squared_distance_to_target_surface", &SurfaceAttractedLinkage::get_squared_distance_to_target_surface, "Given a list of input points, compute the distance from those points to the target surface. Currently used for visualization.", py::arg("query_point_list"))
        .def("get_closest_point_for_visualization", &SurfaceAttractedLinkage::get_closest_point_for_visualization, "Given a list of input points, compute the list of closest points. Currently used for visualization.", py::arg("query_point_list"))
        .def("get_closest_point_normal", &SurfaceAttractedLinkage::get_closest_point_normal, "Given a list of input points, compute the normal of closest points (linearly interpotated from vertex normal)", py::arg("query_point_list"))
        .def("get_l0", &SurfaceAttractedLinkage::get_l0)
        .def("get_E0", &SurfaceAttractedLinkage::get_E0)
        .def("get_target_surface_fitter", &SurfaceAttractedLinkage::get_target_surface_fitter)
        .def("set_target_surface_fitter", py::overload_cast<const Eigen::MatrixX3d &, const Eigen::MatrixX3i &, const bool &>(&SurfaceAttractedLinkage::set_target_surface_fitter), py::arg("V"), py::arg("F"), py::arg("useCenterline"))
        .def("getTargetJointsPosition",   &SurfaceAttractedLinkage::getTargetJointsPosition)
        .def("setTargetJointsPosition",   &SurfaceAttractedLinkage::setTargetJointsPosition, py::arg("jointPosition"))
        .def("setTargetSurface",          &SurfaceAttractedLinkage::setTargetSurface,        py::arg("vertices"), py::arg("faces"))
        .def("getTargetSurfaceVertices",  &SurfaceAttractedLinkage::getTargetSurfaceVertices)
        .def("getTargetSurfaceFaces",     &SurfaceAttractedLinkage::getTargetSurfaceFaces)
        .def("getTargetSurfaceNormals",   &SurfaceAttractedLinkage::getTargetSurfaceNormals)
        .def("scaleJointWeights",         &SurfaceAttractedLinkage::scaleJointWeights, py::arg("jointPosWeight"), py::arg("valence2Multiplier") = 1.0, py::arg("additional_feature_pts") = std::vector<size_t>())
        .def("constructSegmentRestLenToEdgeRestLenMapTranspose", &SurfaceAttractedLinkage::constructSegmentRestLenToEdgeRestLenMapTranspose, py::arg("segmentRestLenGuess"))
        .def(py::pickle([](const SurfaceAttractedLinkage &l) { return py::make_tuple(l.get_surface_path(), l.get_use_centerline(), l.joints(), l.segments(), l.homogenousMaterial(), l.initialMinRestLength(), l.segmentRestLenToEdgeRestLenMapTranspose(), l.getPerSegmentRestLength(), l.getDesignParameterConfig()); },
                        [](const py::tuple &t) {
                            if (t.size() != 9) throw std::runtime_error("Invalid SurfaceAttractedLinkage state!");
                            return std::make_unique<SurfaceAttractedLinkage>(t[0].cast<std::string>(),
                                                                t[1].cast<bool>(),
                                                                t[2].cast<std::vector<RodLinkage::Joint>>(),
                                                                t[3].cast<std::vector<RodLinkage::RodSegment>>(),
                                                                t[4].cast<RodMaterial>(),
                                                                t[5].cast<Real>(),
                                                                t[6].cast<SuiteSparseMatrix>(),
                                                                t[7].cast<Eigen::VectorXd>(),
                                                                t[8].cast<DesignParameterConfig>());

                        }))
        ;
    ////////////////////////////////////////////////////////////////////////////////
    // Equilibrium solver
    ////////////////////////////////////////////////////////////////////////////////
    m.attr("TARGET_ANGLE_NONE") = py::float_(TARGET_ANGLE_NONE);

    m.def("compute_equilibrium",
          [](RodLinkage &linkage, Real targetAverageAngle, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              try {
                  std::cout << "Done creating SAL" << std::endl;
                  auto &sl = dynamic_cast<SurfaceAttractedLinkage &>(linkage);
                  return compute_equilibrium(sl, targetAverageAngle, options, fixedVars, cb);
              }
              catch (...) {
                  std::cout << "Failed creating SAL" << std::endl;
                  return compute_equilibrium(linkage, targetAverageAngle, options, fixedVars, cb);
              }
          },
          py::arg("linkage"),
          py::arg("targetAverageAngle") = TARGET_ANGLE_NONE,
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("callback") = nullptr
    );
    m.def("compute_equilibrium",
          [](RodLinkage &linkage, const Eigen::VectorXd &externalForces, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              try {
                  auto &sl = dynamic_cast<SurfaceAttractedLinkage &>(linkage);
                  return compute_equilibrium(sl, externalForces, options, fixedVars, cb);
              }
              catch (...) {
                  return compute_equilibrium(linkage, externalForces, options, fixedVars, cb);
              }
          },
          py::arg("linkage"),
          py::arg("externalForces"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("callback") = nullptr
    );
    m.def("compute_equilibrium",
          [](ElasticRod &rod, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              return compute_equilibrium(rod, options, fixedVars, cb);
          },
          py::arg("rod"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("callback") = nullptr
    );
    m.def("compute_equilibrium",
          [](PeriodicRod &rod, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              return compute_equilibrium(rod, options, fixedVars, cb);
          },
          py::arg("periodicRod"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("callback") = nullptr
    );
    m.def("get_equilibrium_optimizer",
          [](RodLinkage &linkage, Real targetAverageAngle, const std::vector<size_t> &fixedVars) { return get_equilibrium_optimizer(linkage, targetAverageAngle, fixedVars); },
          py::arg("linkage"),
          py::arg("targetAverageAngle") = TARGET_ANGLE_NONE,
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("get_equilibrium_optimizer",
          [](ElasticRod &rod, const std::vector<size_t> &fixedVars) { return get_equilibrium_optimizer(rod, fixedVars); },
          py::arg("rod"),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("restlen_solve",
          [](RodLinkage &linkage, const NewtonOptimizerOptions &opts, const std::vector<size_t> &fixedVars) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return restlen_solve(linkage, opts, fixedVars);
          },
          py::arg("linkage"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("restlen_solve",
          [](ElasticRod &rod, const NewtonOptimizerOptions &opts, const std::vector<size_t> &fixedVars) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return restlen_solve(rod, opts, fixedVars);
          },
          py::arg("rod"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("restlen_problem",
          [](RodLinkage &linkage, const std::vector<size_t> &fixedVars) -> std::unique_ptr<NewtonProblem> {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return restlen_problem(linkage, fixedVars);
          },
          py::arg("linkage"),
          py::arg("fixedVars") = std::vector<size_t>()
    );
    m.def("designParameter_solve",
          [](RodLinkage &linkage, const NewtonOptimizerOptions &opts, const Real regularization_weight, const Real smoothing_weight, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb, Real E0, Real l0) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              try {
                  auto &sl = dynamic_cast<SurfaceAttractedLinkage &>(linkage);
                  return designParameter_solve(sl, opts, regularization_weight, smoothing_weight, fixedVars, cb, E0, l0);
              }
              catch (...) {
                  return designParameter_solve(linkage, opts, regularization_weight, smoothing_weight, fixedVars, cb, E0, l0);
              }
          },
          py::arg("linkage"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("regularization_weight") = 0.1,
          py::arg("smoothing_weight") = 1,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("callback") = nullptr,
          py::arg("E0") = -1,
          py::arg("l0") = -1
    );
    m.def("designParameter_solve",
          [](ElasticRod &rod, const NewtonOptimizerOptions &opts, const Real regularization_weight, const Real smoothing_weight, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb, Real E0, Real l0) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              return designParameter_solve(rod, opts, regularization_weight, smoothing_weight, fixedVars, cb, E0, l0);
          },
          py::arg("rod"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("regularization_weight") = 0.1,
          py::arg("smoothing_weight") = 1,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("callback") = nullptr,
          py::arg("E0") = -1,
          py::arg("l0") = -1
    );
    // Note: SurfaceAttractedLinkage behaves identically to RodLinkage anyway for design parameter solve
    m.def("get_designParameter_optimizer", [](RodLinkage &l, const Real rw, const Real sw, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb, Real E0, Real l0) {
            return get_designParameter_optimizer(l, rw, sw, fixedVars, callbackWrapper(pcb), E0, l0);
         },
          py::arg("linkage"), py::arg("regularization_weight") = 0.1, py::arg("smoothing_weight") = 1,
          py::arg("fixedVars") = std::vector<size_t>(), py::arg("callback") = nullptr,
          py::arg("E0") = -1,
          py::arg("l0") = -1
    );
    m.def("designParameter_problem",
          [](RodLinkage &linkage, const Real regularization_weight, const Real smoothing_weight, const std::vector<size_t> &fixedVars, Real E0, Real l0) -> std::unique_ptr<NewtonProblem> {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return designParameter_problem(linkage, regularization_weight, smoothing_weight, fixedVars, E0, l0);
          },
          py::arg("linkage"),
          py::arg("regularization_weight") = 0.1,
          py::arg("smoothing_weight") = 1,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("E0") = -1,
          py::arg("l0") = -1
    );
    m.def("equilibrium_problem",
          [](RodLinkage &linkage, Real targetAverageAngle, const std::vector<size_t> &fixedVars) -> std::unique_ptr<NewtonProblem> {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              return equilibrium_problem(linkage, targetAverageAngle, fixedVars);
          },
          py::arg("linkage"),
          py::arg("targetAverageAngle") = TARGET_ANGLE_NONE,
          py::arg("fixedVars") = std::vector<size_t>()
    );

#if HAS_KNITRO
    m.def("compute_equilibrium_knitro",
          [](RodLinkage &linkage, size_t niter, int /* verbose */, const std::vector<size_t> &fixedVars, Real gradTol) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              knitro_compute_equilibrium(linkage, niter, fixedVars, gradTol);
          },
          py::arg("linkage"),
          py::arg("niter") = 100,
          py::arg("verbose") = 0,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("gradTol") = 2e-8
    );
    m.def("compute_equilibrium_knitro",
          [](ElasticRod &rod, size_t niter, int /* verbose */, const std::vector<size_t> &fixedVars, Real gradTol) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              knitro_compute_equilibrium(rod, niter, fixedVars, gradTol);
          },
          py::arg("rod"),
          py::arg("niter") = 100,
          py::arg("verbose") = 0,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("gradTol") = 2e-8
    );
    m.def("restlen_solve_knitro",
          [](RodLinkage &linkage, Real laplacianRegWeight, size_t niter, const std::vector<size_t> &fixedVars, Real gradTol) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              knitro_restlen_solve(linkage, laplacianRegWeight, niter, fixedVars, gradTol);
          },
          py::arg("linkage"),
          py::arg("laplacianRegWeight") = 1.0,
          py::arg("niter") = 100,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("gradTol") = 2e-8
    );
    m.def("restlen_solve_knitro",
          [](ElasticRod &rod, Real laplacianRegWeight, size_t niter, const std::vector<size_t> &fixedVars, Real gradTol) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              knitro_restlen_solve(rod, laplacianRegWeight, niter, fixedVars, gradTol);
          },
          py::arg("rod"),
          py::arg("laplacianRegWeight") = 1.0,
          py::arg("niter") = 100,
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("gradTol") = 2e-8
    );
#endif // HAS_KNITRO

    ////////////////////////////////////////////////////////////////////////////////
    // Design Parameter Solve
    ////////////////////////////////////////////////////////////////////////////////
    bindDesignParameterProblem<             ElasticRod>(detail_module, "ElasticRod");
    bindDesignParameterProblem<             RodLinkage>(detail_module, "RodLinkage");
    bindDesignParameterProblem<SurfaceAttractedLinkage>(detail_module, "SurfaceAttractedLinkage");
    m.def("DesignParameterProblem", [](RodLinkage &linkage) {
          try {
            auto &sl = dynamic_cast<SurfaceAttractedLinkage &>(linkage);
            return py::cast(new DesignParameterProblem<SurfaceAttractedLinkage>(sl), py::return_value_policy::take_ownership);
          }
          catch (...) {
            return py::cast(new DesignParameterProblem<RodLinkage>(linkage), py::return_value_policy::take_ownership);
          }
    }, py::arg("linkage"));
    m.def("DesignParameterProblem", [](ElasticRod &rod) { return std::make_unique<DesignParameterProblem<ElasticRod>>(rod); }, py::arg("rod"));

    ////////////////////////////////////////////////////////////////////////////////
    // Analysis
    ////////////////////////////////////////////////////////////////////////////////
    m.def("linkage_deformation_analysis", &linkage_deformation_analysis, py::arg("rest_linkge"), py::arg("defo_linkge"), py::arg("path"));

    py::class_<DeploymentEnergyIncrement>(m, "DeploymentEnergyIncrement")
        .def_readonly("linearTerm",    &DeploymentEnergyIncrement::linearTerm)
        .def_readonly("quadraticTerm", &DeploymentEnergyIncrement::quadraticTerm)
        .def("__call__",               &DeploymentEnergyIncrement::operator())
        ;

    py::class_<DeploymentPathAnalysis>(m, "DeploymentPathAnalysis")
        .def(py::init<NewtonOptimizer &>(), py::arg("opt"))
        .def(py::init<RodLinkage &, const std::vector<size_t> &>(), py::arg("linkage"), py::arg("fixedVars"))
        .def_readonly("deploymentStep",            &DeploymentPathAnalysis::deploymentStep)
        .def_readonly("secondBestDeploymentStep",  &DeploymentPathAnalysis::secondBestDeploymentStep)
        .def_readonly("relativeStiffnessGap",      &DeploymentPathAnalysis::relativeStiffnessGap)
        .def_readonly("bestEnergyIncrement",       &DeploymentPathAnalysis::bestEnergyIncrement)
        .def_readonly("secondBestEnergyIncrement", &DeploymentPathAnalysis::secondBestEnergyIncrement)
        ;

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

    ////////////////////////////////////////////////////////////////////////////
    // Free-standing output functions
    ////////////////////////////////////////////////////////////////////////////
    m.def("rescale_weaving", [](const std::string& model_path, const std::string& ref_model_path, Real target_bb_side) {
        std::vector<MeshIO::IOVertex > mod_verts;
        std::vector<MeshIO::IOElement> mod_eles;
        MeshIO::load(model_path, mod_verts, mod_eles);

        std::vector<MeshIO::IOVertex > ref_verts;
        std::vector<MeshIO::IOElement> ref_eles;
        MeshIO::load(ref_model_path, ref_verts, ref_eles);

        // Get axis aligned bounding box
        Point3D corner1 = Point3D(std::numeric_limits<double>::max(),std::numeric_limits<double>::max(),std::numeric_limits<double>::max());
        Point3D corner2 = Point3D(std::numeric_limits<double>::lowest(),std::numeric_limits<double>::lowest(),std::numeric_limits<double>::lowest());
        for (auto& v : mod_verts) {
            for (size_t coord = 0; coord < 3; coord++) {
                if (v.point[coord] < corner1[coord]) corner1[coord] = v.point[coord];
                if (v.point[coord] > corner2[coord]) corner2[coord] = v.point[coord];
            }
        }

        // Compute scaling
        Real longest_side = 0;
        for (size_t coord = 0; coord < 3; coord++) { if (longest_side < corner2[coord] - corner1[coord]) longest_side = corner2[coord] - corner1[coord];}
        Real factor = target_bb_side / longest_side;

        // Scale meshes and translate to origin
        Point3D geom_center = Point3D(0.,0.,0.);
        for (auto& v : mod_verts) {
            for (size_t coord = 0; coord < 3; coord++) v.point[coord] *= factor;
            geom_center += v.point;
        }
        geom_center /= (Real)mod_verts.size();
        for (auto& v : mod_verts) {
            v.point -= geom_center;
        }
        for (auto& v : ref_verts) {
            for (size_t coord = 0; coord < 3; coord++) v.point[coord] *= factor;
            v.point -= geom_center;
        }

        // Store meshes with new names
        std::string raw_path = model_path.substr(0, model_path.find_last_of(".")); 
        std::string extension = model_path.substr(model_path.find_last_of("."), model_path.length() - raw_path.length());
        MeshIO::save(raw_path + "_rescaled" + extension, mod_verts, mod_eles);
        raw_path = ref_model_path.substr(0, ref_model_path.find_last_of(".")); 
        extension = ref_model_path.substr(ref_model_path.find_last_of("."), ref_model_path.length() - raw_path.length());
        MeshIO::save(raw_path + "_rescaled" + extension, ref_verts, ref_eles);
    },
    py::arg("model_path"), py::arg("ref_model_path"), py::arg("target_bb_side"));

    m.def("save_mesh", [](const std::string &path, Eigen::MatrixX3d &V, Eigen::MatrixXi &F) {
        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> elements;
        const size_t nv = V.rows();
        vertices.reserve(nv);
        for (size_t i = 0; i < nv; ++i)
           vertices.emplace_back(V.row(i).transpose().eval());
        const size_t ne = F.rows();
        const size_t nc = F.cols();
        elements.reserve(ne);
        for (size_t i = 0; i < ne; ++i) {
            elements.emplace_back(nc);
            for (size_t c = 0; c < nc; ++c)
                elements.back()[c] = F(i, c);
        }

        MeshIO::MeshType type;
        if (nc == 3) { type = MeshIO::MeshType::MESH_TRI; }
        else if (nc == 4) { type = MeshIO::MeshType::MESH_QUAD; }
        else {throw std::runtime_error("unsupported element type"); }

        MeshIO::save(path, vertices, elements, MeshIO::Format::FMT_GUESS, type);
    });
}
