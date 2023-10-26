#include "../src/AverageAngleLinkage.hh"
#include "../ext/elastic_rods/RodLinkage.hh"
#include "../src/AverageAngleSurfaceAttractedLinkage.hh"
#include "../ext/elastic_rods/SurfaceAttractedLinkage.hh"

#include "../ext/elastic_rods/python_bindings/visualization.hh"
#include "../ext/elastic_rods/python_bindings/linkage_optimization.hh" // Target surface fitting
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/../../python_bindings/BindingUtils.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
namespace py = pybind11;

// Hack around a limitation of pybind11 where we cannot specify argument passing policies and
// pybind11 tries to make a copy if the passed instance is not already registered:
//      https://github.com/pybind/pybind11/issues/1200
// We therefore make our Python callback interface use a raw pointer to forbid this copy (which
// causes an error since NewtonProblem is not copyable).
using PyCallbackFunction = std::function<void(NewtonProblem *, size_t)>;
CallbackFunction callbackWrapper(const PyCallbackFunction &pcb) {
    return [pcb](NewtonProblem &p, size_t i) -> void { if (pcb) pcb(&p, i); };
}

PYBIND11_MODULE(average_angle_linkages, m) {
    m.doc() = "AverageAngleLinkage Codebase";

    py::module::import("MeshFEM");
    py::module::import("sparse_matrices");
    py::module::import("ElasticRods");

    py::module detail_module = m.def_submodule("detail");

    ////////////////////////////////////////////////////////////////////////////////
    // AverageAngleLinkage
    ////////////////////////////////////////////////////////////////////////////////

    using PyAAL = py::class_<AverageAngleLinkage, RodLinkage>;
    auto average_angle_linkage = PyAAL(m, "AverageAngleLinkage");

    average_angle_linkage
        .def(py::init<const std::string &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>>(), 
            py::arg("linkage_path"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>())
        .def(py::init<const Eigen::MatrixX3d &, const Eigen::MatrixX2i &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>, std::vector<size_t>>(), 
            py::arg("points"), py::arg("edges"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>(), py::arg("free_joint_angles") = std::vector<size_t>())
        .def(py::init<const AverageAngleLinkage &, std::vector<size_t>>(), "Copy constructor", py::arg("linkage"), py::arg("free_joint_angles"))
        .def(py::init<const AverageAngleLinkage &>(), "Copy constructor", py::arg("linkage"))
        .def(py::init<const RodLinkage &, std::vector<size_t>>(), "Copy constructor", py::arg("linkage"), py::arg("free_joint_angles") = std::vector<size_t>())
        .def("getAverageAngleIndex", &AverageAngleLinkage::getAverageAngleIndex)
        .def("getFreeAngles", &AverageAngleLinkage::getFreeAngles)
        .def("getActuatedAngles", &AverageAngleLinkage::getActuatedAngles)
        .def("getAverageActuatedJointsAngle", &AverageAngleLinkage::getAverageActuatedJointsAngle)
        .def("applyTransformation", &AverageAngleLinkage::applyTransformation, py::arg("AAV"))
        .def("applyTransformationTranspose", &AverageAngleLinkage::applyTransformationTranspose, py::arg("JAV"))
        .def("applyTransformationDoFSize", &AverageAngleLinkage::applyTransformationDoFSize, py::arg("vec"))
        .def("applyTransformationTransposeDoFSize", &AverageAngleLinkage::applyTransformationTransposeDoFSize, py::arg("vec"))
        .def("getAverageAngleToJointAngleMapTranspose", &AverageAngleLinkage::getAverageAngleToJointAngleMapTranspose)
        .def("getAverageAngleToJointAngleMapTranspose_AllJointVars", &AverageAngleLinkage::getAverageAngleToJointAngleMapTranspose_AllJointVars)
        .def("getDoFs", &AverageAngleLinkage::getDoFs)
        .def("setDoFs",  py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool>(&AverageAngleLinkage::setDoFs), py::arg("values"), py::arg("spatialCoherence") = false)
        .def("getExtendedDoFs", &AverageAngleLinkage::getExtendedDoFs)
        .def("setExtendedDoFs", &AverageAngleLinkage::setExtendedDoFs, py::arg("dof"), py::arg("spatialCoherence") = false)
        .def("getExtendedDoFsPSRL", &AverageAngleLinkage::getExtendedDoFsPSRL)
        .def("setExtendedDoFsPSRL", &AverageAngleLinkage::setExtendedDoFsPSRL, py::arg("dof"), py::arg("spatialCoherence") = false)
        .def("energy",        py::overload_cast<ElasticRod::EnergyType>(&AverageAngleLinkage::energy, py::const_), "Compute elastic energy", py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("gradient", &AverageAngleLinkage::gradient, "Elastic energy gradient", py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("gradientJA", &AverageAngleLinkage::gradientJA, "Elastic energy gradient, using joint angle variables (do not use for optimizing AA variables)", py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("gradientPerSegmentRestlen", &AverageAngleLinkage::gradientPerSegmentRestlen, py::arg("updatedSource") = false, py::arg("energyType") = ElasticRod::EnergyType::Full)
        .def("hessianNNZ",             &AverageAngleLinkage::hessianNNZ,             "Tight upper bound for nonzeros in the Hessian.",                            py::arg("variableDesignParameters") = false)
        .def("hessianSparsityPattern", &AverageAngleLinkage::hessianSparsityPattern, "Compressed column matrix containing all potential nonzero Hessian entries", py::arg("variableDesignParameters") = false, py::arg("val") = 0.0)
        .def("hessian",  py::overload_cast<ElasticRod::EnergyType, bool>(&AverageAngleLinkage::hessian, py::const_), "Elastic energy  hessian", py::arg("energyType") = ElasticRod::EnergyType::Full, py::arg("variableDesignParameters") = false)
        .def("applyHessian", &AverageAngleLinkage::applyHessian, "Elastic energy Hessian-vector product formulas.", py::arg("v"), py::arg("variableDesignParameters") = false, py::arg("mask") = HessianComputationMask())
        .def("applyHessianPerSegmentRestlen", &AverageAngleLinkage::applyHessianPerSegmentRestlen, "Elastic energy Hessian-vector product formulas for per segment rest length.", py::arg("v"), py::arg("mask") = HessianComputationMask())
        .def("massMatrix",        py::overload_cast<bool, bool>(&AverageAngleLinkage::massMatrix, py::const_), py::arg("updatedSource") = false, py::arg("useLumped") = false)
        .def(py::pickle([](const AverageAngleLinkage &l) { return py::make_tuple(l.joints(), l.segments(), l.homogenousMaterial(), l.initialMinRestLength(), l.segmentRestLenToEdgeRestLenMapTranspose(), l.getPerSegmentRestLength(), l.getDesignParameterConfig(), l.getFreeAngles()); },
                        [](const py::tuple &t) {
                            // For backward compatibility
                            if (t.size() == 7){
                                return std::make_unique<AverageAngleLinkage>(t[0].cast<std::vector<RodLinkage::Joint>>(),
                                                                            t[1].cast<std::vector<RodLinkage::RodSegment>>(),
                                                                            t[2].cast<RodMaterial>(),
                                                                            t[3].cast<Real>(),
                                                                            t[4].cast<SuiteSparseMatrix>(),
                                                                            t[5].cast<Eigen::VectorXd>(),
                                                                            t[6].cast<DesignParameterConfig>());
                            } else if (t.size() == 8){
                                return std::make_unique<AverageAngleLinkage>(t[0].cast<std::vector<RodLinkage::Joint>>(),
                                                                            t[1].cast<std::vector<RodLinkage::RodSegment>>(),
                                                                            t[2].cast<RodMaterial>(),
                                                                            t[3].cast<Real>(),
                                                                            t[4].cast<SuiteSparseMatrix>(),
                                                                            t[5].cast<Eigen::VectorXd>(),
                                                                            t[6].cast<DesignParameterConfig>(),
                                                                            t[7].cast<std::vector<size_t>>());
                            } else {
                                throw std::runtime_error("Invalid AAL state!");
                            }

                        }))
        ;

    ////////////
    // Surface Attracted Linkage for Average Angle Linkage. Code duplication with RodLinkage. 
    using AASAL = AverageAngleSurfaceAttractedLinkage;

    py::enum_<AASAL::SurfaceAttractionEnergyType>(m, "SurfaceAttractionEnergyType")
        .value("Full",       AASAL::SurfaceAttractionEnergyType::Full      )
        .value("Attraction", AASAL::SurfaceAttractionEnergyType::Attraction)
        .value("Elastic",    AASAL::SurfaceAttractionEnergyType::Elastic   )
        ;


    ////////////////////////////////////////////////////////////////////////////////
    // AASAL
    ////////////////////////////////////////////////////////////////////////////////
    auto average_angle_surface_attracted_linkage = py::class_<AASAL, AverageAngleLinkage>(m, "AverageAngleSurfaceAttractedLinkage")
        .def(py::init<const std::string &, const bool &, const Eigen::MatrixX3d &, const Eigen::MatrixX2i &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>, std::vector<size_t>>(), py::arg("surface_path"), py::arg("useCenterline"), py::arg("points"), py::arg("edges"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>(), py::arg("free_joint_angles") = std::vector<size_t>())
        .def(py::init<const Eigen::MatrixX3d &, const Eigen::MatrixX3i &, const bool &, const Eigen::MatrixX3d &, const Eigen::MatrixX2i &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>, std::vector<size_t>>(), py::arg("target_vertices"), py::arg("target_faces"), py::arg("useCenterline"), py::arg("points"), py::arg("edges"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>(), py::arg("free_joint_angles") = std::vector<size_t>())
        .def(py::init<const std::string &, const bool &, const AverageAngleLinkage &, std::vector<size_t>>(), "Copy constructor with rod linkage and path to target", py::arg("surface_path"), py::arg("useCenterline"), py::arg("rod"), py::arg("free_joint_angles") = std::vector<size_t>())
        .def(py::init<const std::string &, const bool &, const AASAL &, std::vector<size_t>>(), "Copy constructor with surface attracted linkage and path to target surface", py::arg("surface_path"), py::arg("useCenterline"), py::arg("surface_attracted_linkage"), py::arg("free_joint_angles") = std::vector<size_t>())
        .def(py::init<const std::string &, const bool &, const SurfaceAttractedLinkage &, std::vector<size_t>>(), "Copy constructor with surface attracted linkage and path to target surface", py::arg("surface_path"), py::arg("useCenterline"), py::arg("surface_attracted_linkage"), py::arg("free_joint_angles") = std::vector<size_t>())
        .def(py::init<const Eigen::MatrixX3d &, const Eigen::MatrixX3i &, const bool &, const AverageAngleLinkage &, std::vector<size_t>>(), "Copy constructor with rod linkage", py::arg("target_vertices"), py::arg("target_faces"), py::arg("useCenterline"), py::arg("rod"), py::arg("free_joint_angles") = std::vector<size_t>())
        .def(py::init<const Eigen::MatrixX3d &, const Eigen::MatrixX3i &, const bool &, const AASAL &, std::vector<size_t>>(), "Copy constructor with surface attracted linkage and surface", py::arg("target_vertices"), py::arg("target_faces"), py::arg("useCenterline"), py::arg("surface_attracted_linkage"), py::arg("free_joint_angles") = std::vector<size_t>())
        .def(py::init<const AASAL &, std::vector<size_t>>(), "Copy constructor with surface attracted linkage only", py::arg("surface_attracted_linkage"), py::arg("free_joint_angles"))
        .def(py::init<const AASAL &>(), "Copy constructor with surface attracted linkage only", py::arg("surface_attracted_linkage"))
        .def(py::init<const std::string &, const bool &, const std::string &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>, std::vector<size_t>>(), py::arg("surface_path"), py::arg("useCenterline"), py::arg("linkage_path"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>(), py::arg("free_joint_angles") = std::vector<size_t>())
        .def("set", (void (AASAL::*)(const std::string &, const bool &, const std::string &, size_t, bool, InterleavingType, std::vector<std::function<Pt3_T<Real>(Real, bool)>>, std::vector<Eigen::Vector3d>, std::vector<size_t>))(&AASAL::set), py::arg("surface_path"), py::arg("useCenterline"), py::arg("linkage_path"), py::arg("subdivision") = 10, py::arg("initConsistentAngle") = true, py::arg("rod_interleaving_type") = InterleavingType::noOffset, py::arg("edge_callbacks") = std::vector<std::function<Pt3_T<Real>(Real, bool)>>(), py::arg("input_joint_normals") = std::vector<Eigen::Vector3d>(), py::arg("free_joint_angles") = std::vector<size_t>()) // py::overload_cast fails
        .def("energy", &AASAL::energy, "Compute potential energy (elastic + surface attraction)", py::arg("surface_eType") = AASAL::SurfaceAttractionEnergyType::Full)
        .def("gradient", &AASAL::gradient, "Potential energy gradient", py::arg("updatedSource") = false, py::arg("surface_eType") = AASAL::SurfaceAttractionEnergyType::Full, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("gradientJA", &AASAL::gradientJA, "Potential energy gradient, using joint angle variables (do not use for optimizing AA variables)", py::arg("updatedSource") = false, py::arg("surface_eType") = AASAL::SurfaceAttractionEnergyType::Full, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("gradientBend", &AASAL::gradientBend, "Potential Bending Energy Gradient", py::arg("updatedSource") = false, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("gradientTwist", &AASAL::gradientTwist, "Potential Twisting Energy Gradient", py::arg("updatedSource") = false, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("gradientStretch", &AASAL::gradientStretch, "Potential Stretching Energy Gradient", py::arg("updatedSource") = false, py::arg("variableDesignParameters") = false, py::arg("designParameterOnly") = false, py::arg("skipBRods") = false)
        .def("hessian",  py::overload_cast<AASAL::SurfaceAttractionEnergyType, bool>(&AASAL::hessian, py::const_), "Potential energy  hessian", py::arg("surface_eType") = AASAL::SurfaceAttractionEnergyType::Full, py::arg("variableDesignParameters") = false)
        .def("applyHessian", &AASAL::applyHessian, "Potential energy Hessian-vector product formulas.", py::arg("v"), py::arg("variableDesignParameters") = false, py::arg("mask") = HessianComputationMask(), py::arg("surface_eType") = AASAL::SurfaceAttractionEnergyType::Full)
        .def("gradientPerSegmentRestlen", &AASAL::gradientPerSegmentRestlen, "Potential energy gradient for per segment rest length", py::arg("updatedSource") = false, py::arg("surface_eType") = AASAL::SurfaceAttractionEnergyType::Full)
        .def("hessianPerSegmentRestlen",  py::overload_cast<AASAL::SurfaceAttractionEnergyType>(&AASAL::hessianPerSegmentRestlen, py::const_), "Potential energy  hessian for per segment rest length", py::arg("surface_eType") = AASAL::SurfaceAttractionEnergyType::Full)
        .def("applyHessianPerSegmentRestlen", &AASAL::applyHessianPerSegmentRestlen, "Potential energy Hessian-vector product formulas for per segment rest length.", py::arg("v"), py::arg("mask") = HessianComputationMask(), py::arg("surface_eType") = AASAL::SurfaceAttractionEnergyType::Full)
        .def_readwrite("attraction_weight", &AASAL::attraction_weight, "Trade off between surface attraction and elastic energy")
        .def("set_attraction_tgt_joint_weight", &AASAL::set_attraction_tgt_joint_weight, "Trade off between fitting target joint position and fitting the surface inside the potential energy of the linkage.", py::arg("weight"))
        .def("get_attraction_tgt_joint_weight", &AASAL::get_attraction_tgt_joint_weight)
        .def("setDoFs", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool>(&AASAL::setDoFs), py::arg("values"), py::arg("spatialCoherence") = false)
        .def("setExtendedDoFs", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool>(&AASAL::setExtendedDoFs), py::arg("values"), py::arg("spatialCoherence") = false)
        .def("setExtendedDoFsPSRL", py::overload_cast<const Eigen::Ref<const Eigen::VectorXd> &, bool>(&AASAL::setExtendedDoFsPSRL), py::arg("values"), py::arg("spatialCoherence") = false)
        .def("get_linkage_closest_point", &AASAL::get_linkage_closest_point, "The list of closest point to the linkage.")
        .def("set_use_centerline", &AASAL::set_use_centerline, "Choose to use centerline position or joint position in target surface fitter.", py::arg("useCenterline"))
        .def("get_holdClosestPointsFixed", &AASAL::get_holdClosestPointsFixed, "Choose to update the closest point in the target surface fitting term during each equilibrium iteration.")
        .def("set_holdClosestPointsFixed", &AASAL::set_holdClosestPointsFixed, "Choose to update the closest point in the target surface fitting term during each equilibrium iteration.", py::arg("set_holdClosestPointsFixed"))
        .def("get_squared_distance_to_target_surface", &AASAL::get_squared_distance_to_target_surface, "Given a list of input points, compute the distance from those points to the target surface. Currently used for visualization.", py::arg("query_point_list"))
        .def("get_closest_point_for_visualization", &AASAL::get_closest_point_for_visualization, "Given a list of input points, compute the list of closest points. Currently used for visualization.", py::arg("query_point_list"))
        .def("get_closest_point_normal", &AASAL::get_closest_point_normal, "Given a list of input points, compute the normal of closest points (linearly interpotated from vertex normal)", py::arg("query_point_list"))
        .def("get_l0", &AASAL::get_l0)
        .def("get_E0", &AASAL::get_E0)
        .def("get_target_surface_fitter", &AASAL::get_target_surface_fitter)
        .def("set_target_surface_fitter", py::overload_cast<const Eigen::MatrixX3d &, const Eigen::MatrixX3i &, const bool &>(&AASAL::set_target_surface_fitter), py::arg("V"), py::arg("F"), py::arg("useCenterline"))
        .def("getTargetJointsPosition",   &AASAL::getTargetJointsPosition)
        .def("setTargetJointsPosition",   &AASAL::setTargetJointsPosition, py::arg("jointPosition"))
        .def("setTargetSurface",          &AASAL::setTargetSurface,        py::arg("vertices"), py::arg("faces"))
        .def("getTargetSurfaceVertices",  &AASAL::getTargetSurfaceVertices)
        .def("getTargetSurfaceFaces",     &AASAL::getTargetSurfaceFaces)
        .def("getTargetSurfaceNormals",   &AASAL::getTargetSurfaceNormals)
        .def("saveAttractionSurface",     &AASAL::saveAttractionSurface, py::arg("path"))
        .def("reflectAttractionSurface",  &AASAL::reflectAttractionSurface, py::arg("jointID"))
        .def("scaleJointWeights",         &AASAL::scaleJointWeights, py::arg("jointPosWeight"), py::arg("valence2Multiplier") = 1.0, py::arg("additional_feature_pts") = std::vector<size_t>())
        .def("constructSegmentRestLenToEdgeRestLenMapTranspose", &AASAL::constructSegmentRestLenToEdgeRestLenMapTranspose, py::arg("segmentRestLenGuess"))
        .def(py::pickle([](const AASAL &l) { return py::make_tuple(l.getTargetSurfaceVertices(), l.getTargetSurfaceFaces(), l.get_use_centerline(), l.joints(), l.segments(), l.homogenousMaterial(), l.initialMinRestLength(), l.segmentRestLenToEdgeRestLenMapTranspose(), l.getPerSegmentRestLength(), l.getDesignParameterConfig(), l.getFreeAngles()); },
                        [](const py::tuple &t) {
                            // For backward compatibility
                            if (t.size() == 10){
                                return std::make_unique<AASAL>(t[0].cast<Eigen::MatrixXd>(),
                                                            t[1].cast<Eigen::MatrixXi>(),
                                                            t[2].cast<bool>(),
                                                            t[3].cast<std::vector<AverageAngleLinkage::Joint>>(),
                                                            t[4].cast<std::vector<AverageAngleLinkage::RodSegment>>(),
                                                            t[5].cast<RodMaterial>(),
                                                            t[6].cast<Real>(),
                                                            t[7].cast<SuiteSparseMatrix>(),
                                                            t[8].cast<Eigen::VectorXd>(),
                                                            t[9].cast<DesignParameterConfig>());
                            } else if (t.size() == 11){
                                return std::make_unique<AASAL>(t[0].cast<Eigen::MatrixXd>(),
                                                            t[1].cast<Eigen::MatrixXi>(),
                                                            t[2].cast<bool>(),
                                                            t[3].cast<std::vector<AverageAngleLinkage::Joint>>(),
                                                            t[4].cast<std::vector<AverageAngleLinkage::RodSegment>>(),
                                                            t[5].cast<RodMaterial>(),
                                                            t[6].cast<Real>(),
                                                            t[7].cast<SuiteSparseMatrix>(),
                                                            t[8].cast<Eigen::VectorXd>(),
                                                            t[9].cast<DesignParameterConfig>(),
                                                            t[10].cast<std::vector<size_t>>());
                            } else {
                                throw std::runtime_error("Invalid AAL state!");
                            }
                        }))
        ;

    ////////////////////////////////////////////////////////////////////////////////
    // Equilibrium solver
    ////////////////////////////////////////////////////////////////////////////////
    m.attr("TARGET_ANGLE_NONE") = py::float_(TARGET_ANGLE_NONE);

    m.def("compute_equilibrium",
          [](AverageAngleLinkage &linkage, Real targetAverageAngle, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              try {
                  auto &aasl = dynamic_cast<AASAL &>(linkage);
                  return compute_equilibrium(aasl, targetAverageAngle, options, fixedVars, cb);
                  std::cout << "Done creating AASAL" << std::endl;
              }
              catch (...) {
                  std::cout << "Failed creating AASAL" << std::endl;
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
          [](AverageAngleLinkage &linkage, const Eigen::VectorXd &externalForces, const NewtonOptimizerOptions &options, const std::vector<size_t> &fixedVars, const PyCallbackFunction &pcb) {
              py::scoped_ostream_redirect stream1(std::cout, py::module::import("sys").attr("stdout"));
              py::scoped_ostream_redirect stream2(std::cerr, py::module::import("sys").attr("stderr"));
              auto cb = callbackWrapper(pcb);
              try {
                  auto &aasl = dynamic_cast<AASAL &>(linkage);
                  std::cout << "Done creating AASAL" << std::endl;
                  return compute_equilibrium(aasl, externalForces, options, fixedVars, cb);
              }
              catch (...) {
                  std::cout << "Failed creating AASAL" << std::endl;
                  return compute_equilibrium(linkage, externalForces, options, fixedVars, cb);
              }
          },
          py::arg("linkage"),
          py::arg("externalForces"),
          py::arg("options") = NewtonOptimizerOptions(),
          py::arg("fixedVars") = std::vector<size_t>(),
          py::arg("callback") = nullptr
    );
    m.def("get_equilibrium_optimizer",
          [](AverageAngleLinkage &linkage, Real targetAverageAngle, const std::vector<size_t> &fixedVars) { return get_equilibrium_optimizer(linkage, targetAverageAngle, fixedVars); },
          py::arg("linkage"),
          py::arg("targetAverageAngle") = TARGET_ANGLE_NONE,
          py::arg("fixedVars") = std::vector<size_t>()
    );
}