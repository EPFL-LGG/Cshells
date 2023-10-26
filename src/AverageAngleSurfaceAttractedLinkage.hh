#ifndef AVERAGEANGLESURFACEATTRACTEDLINKAGE_HH
#define AVERAGEANGLESURFACEATTRACTEDLINKAGE_HH

#include "AverageAngleLinkage.hh"
#include "../ext/elastic_rods/SurfaceAttractedLinkage.hh"
#include "RectangularBox.hh"
#include <rotation_optimization.hh>
#include <array>
#include <tuple>

#include <MeshFEM/Geometry.hh>
#include "compute_equilibrium.hh"
#include "LOMinAngleConstraint.hh"
#include "TargetSurfaceFitter.hh"


// Templated to support automatic differentiation types.
template<typename Real_>
struct AverageAngleSurfaceAttractedLinkage_T;

using AverageAngleSurfaceAttractedLinkage = AverageAngleSurfaceAttractedLinkage_T<Real>;

template<typename Real_>
struct AverageAngleSurfaceAttractedLinkage_T : public AverageAngleLinkage_T<Real_> {
    using Base = AverageAngleLinkage_T<Real_>;
    static constexpr size_t defaultSubdivision   = Base::defaultSubdivision;
    static constexpr bool defaultConsistentAngle = Base::defaultConsistentAngle;
    using TMatrix     = typename Base::TMatrix;
    using CSCMat      = typename Base::CSCMat;
    using VecX        = typename Base::VecX;
    using Vec3        = typename Base::Vec3;
    using Mat3        = Mat3_T<Real>;
    using EnergyType  = typename Base::EnergyType;
    using Linkage_dPC = typename Base::Linkage_dPC;

    enum class SurfaceAttractionEnergyType { Full, Attraction, Elastic };

	// Need to override constructors to add the surface model path.
    // Construct empty linkage, to be initialized later by calling set.
    AverageAngleSurfaceAttractedLinkage_T() : AverageAngleLinkage_T<Real_>() { }

    // Forward all constructor arguments to set(...)
    template<typename... Args>
    AverageAngleSurfaceAttractedLinkage_T(Args&&... args) : AverageAngleLinkage_T<Real_>() {
        set(std::forward<Args>(args)...);
    }

    AverageAngleSurfaceAttractedLinkage_T(const AverageAngleSurfaceAttractedLinkage_T &linkage) : AverageAngleLinkage_T<Real_>() { 
        set(linkage, linkage.getFreeAngles()); 
    } // The above forwarding constructor confuses pybind11
    AverageAngleSurfaceAttractedLinkage_T(const AverageAngleSurfaceAttractedLinkage_T &linkage, std::vector<size_t> freeAngles) : AverageAngleLinkage_T<Real_>() { 
        set(linkage, freeAngles); 
    }
    AverageAngleSurfaceAttractedLinkage_T(const SurfaceAttractedLinkage_T<Real_> &linkage, std::vector<size_t> freeAngles = {}) : AverageAngleLinkage_T<Real_>() { 
        set(linkage, freeAngles); 
    }

    // // Read the rod linkage from a line graph file.
    // void read(const std::string &surface_path, const std::string &linkage_path, size_t subdivision = defaultSubdivision, bool initConsistentAngle = defaultConsistentAngle) {
    //     Base::read(linkage_path, subdivision, initConsistentAngle);
    //     m_surface_path = surface_path;
    //     set_target_surface_fitter(surface_path);
    // }

    // Wrapper for all of RodLinkage's set methods that takes a surface_path as a first argument.
    void set(const std::string &surface_path, const bool &useCenterline, const std::string &linkage_path,
             size_t subdivision = defaultSubdivision, 
             bool initConsistentAngle = defaultConsistentAngle, 
             InterleavingType rod_interleaving_type = InterleavingType::xshell, 
             std::vector<std::function<Pt3_T<Real_>(Real_, bool)>> edge_callbacks = {}, 
             std::vector<Vec3> input_joint_normals = {},
             std::vector<size_t> freeAngles = {}) {
        Base::set(linkage_path, subdivision, initConsistentAngle, rod_interleaving_type, edge_callbacks, input_joint_normals);
        Base::setFreeAngles(freeAngles);
        m_surface_path = surface_path;
        set_target_surface_fitter(surface_path, useCenterline);
        m_l0 = BBox<Point3D>(Base::deformedPoints()).dimensions().norm();
        m_E0 = Base::homogenousMaterial().youngModulus * Base::homogenousMaterial().area * Base::getPerSegmentRestLength().sum();
    }

    void set(const std::string &surface_path, const bool useCenterline, std::vector<MeshIO::IOVertex > vertices, // copy edited inside
             std::vector<MeshIO::IOElement> edges,    // copy edited inside
             size_t subdivision = defaultSubdivision,
             bool initConsistentAngle = defaultConsistentAngle, 
             InterleavingType rod_interleaving_type = InterleavingType::xshell, 
             std::vector<std::function<Pt3_T<Real_>(Real_, bool)>> edge_callbacks = {}, 
             std::vector<Vec3> input_joint_normals = {},
             std::vector<size_t> freeAngles = {}) {
        Base::set(vertices, edges, subdivision, initConsistentAngle, rod_interleaving_type, edge_callbacks, input_joint_normals);
        Base::setFreeAngles(freeAngles);
        m_surface_path = surface_path;
        set_target_surface_fitter(surface_path, useCenterline);
        m_l0 = BBox<Point3D>(Base::deformedPoints()).dimensions().norm();
        m_E0 = Base::homogenousMaterial().youngModulus * Base::homogenousMaterial().area * Base::getPerSegmentRestLength().sum();
    }

    void set(const std::string &surface_path, const bool useCenterline, const Eigen::MatrixX3d &vertices,
             const Eigen::MatrixX2i &edges,
             size_t subdivision = defaultSubdivision, 
             bool initConsistentAngle = defaultConsistentAngle, 
             InterleavingType rod_interleaving_type = InterleavingType::xshell, 
             std::vector<std::function<Pt3_T<Real_>(Real_, bool)>> edge_callbacks = {}, 
             std::vector<Vec3> input_joint_normals = {},
             std::vector<size_t> freeAngles = {}) {
        Base::set(vertices, edges, subdivision, initConsistentAngle, rod_interleaving_type, edge_callbacks, input_joint_normals);
        Base::setFreeAngles(freeAngles);
        m_surface_path = surface_path;
        set_target_surface_fitter(surface_path, useCenterline);
        m_l0 = BBox<Point3D>(Base::deformedPoints()).dimensions().norm();
        m_E0 = Base::homogenousMaterial().youngModulus * Base::homogenousMaterial().area * Base::getPerSegmentRestLength().sum();
    }

    template<template<typename> class Object>
    void set(const std::string &surface_path, const bool useCenterline, 
             Object<Real> linkage,
             std::vector<size_t> freeAngles = {}) {
        Base::set(linkage.joints(), linkage.segments(), linkage.homogenousMaterial(), linkage.initialMinRestLength(), linkage.segmentRestLenToEdgeRestLenMapTranspose(), linkage.getPerSegmentRestLength(), linkage.getDesignParameterConfig());
        Base::setFreeAngles(freeAngles);
        m_surface_path = surface_path;
        set_target_surface_fitter(surface_path, useCenterline);
        m_l0 = BBox<Point3D>(Base::deformedPoints()).dimensions().norm();
        m_E0 = Base::homogenousMaterial().youngModulus * Base::homogenousMaterial().area * Base::getPerSegmentRestLength().sum();
    }

    // Wrapper for all of RodLinkage's set methods that take a mesh as first argument

    void set(const Eigen::MatrixX3d &targetV, const Eigen::MatrixX3i &targetF, const bool useCenterline, 
             const Eigen::MatrixX3d &vertices, const Eigen::MatrixX2i &edges,
             size_t subdivision = defaultSubdivision,
             bool initConsistentAngle = defaultConsistentAngle, 
             InterleavingType rod_interleaving_type = InterleavingType::xshell, 
             std::vector<std::function<Pt3_T<Real_>(Real_, bool)>> edge_callbacks = {}, 
             std::vector<Vec3> input_joint_normals = {},
             std::vector<size_t> freeAngles = {}) {
        Base::set(vertices, edges, subdivision, initConsistentAngle, rod_interleaving_type, edge_callbacks, input_joint_normals);
        Base::setFreeAngles(freeAngles);
        m_surface_path = "";
        set_target_surface_fitter(targetV, targetF, useCenterline);
        m_l0 = BBox<Point3D>(Base::deformedPoints()).dimensions().norm();
        m_E0 = Base::homogenousMaterial().youngModulus * Base::homogenousMaterial().area * Base::getPerSegmentRestLength().sum();
    }

    //Takes a linkage as input and adds a surface
    template<template<typename> class Object>
    void set(const Eigen::MatrixX3d &targetV, const Eigen::MatrixX3i &targetF, const bool useCenterline, 
             Object<Real> linkage, std::vector<size_t> freeAngles = {}) {
        Base::set(linkage.joints(), linkage.segments(), linkage.homogenousMaterial(), linkage.initialMinRestLength(), linkage.segmentRestLenToEdgeRestLenMapTranspose(), linkage.getPerSegmentRestLength(), linkage.getDesignParameterConfig());
        Base::setFreeAngles(freeAngles);
        m_surface_path = "";
        set_target_surface_fitter(targetV, targetF, useCenterline);
        m_l0 = BBox<Point3D>(Base::deformedPoints()).dimensions().norm();
        m_E0 = Base::homogenousMaterial().youngModulus * Base::homogenousMaterial().area * Base::getPerSegmentRestLength().sum();
    }

    // Initialize by copying from another AverageAngleSurfaceAttractedLinkage_T
    template<typename Real2_>
    void set(const AverageAngleSurfaceAttractedLinkage_T<Real2_> &linkage, std::vector<size_t> freeAngles) {
        Base::set(linkage);
        Base::setFreeAngles(freeAngles);
        m_surface_path = linkage.get_surface_path();
        attraction_weight = linkage.attraction_weight;
        m_attraction_tgt_joint_weight = linkage.get_attraction_tgt_joint_weight();
        target_surface_fitter = linkage.get_target_surface_fitter();

        m_l0 = linkage.get_l0();
        m_E0 = linkage.get_E0();
    }

    template<typename Real2_>
    void set(const AverageAngleSurfaceAttractedLinkage_T<Real2_> &linkage) {
        Base::set(linkage);
        Base::setFreeAngles(linkage.getFreeAngles());
        m_surface_path = linkage.get_surface_path();
        attraction_weight = linkage.attraction_weight;
        m_attraction_tgt_joint_weight = linkage.get_attraction_tgt_joint_weight();
        target_surface_fitter = linkage.get_target_surface_fitter();

        m_l0 = linkage.get_l0();
        m_E0 = linkage.get_E0();
    }

    // For serialization with surface path or attraction mesh directly
    template<typename Real2_>
    void set(const std::string &surface_path, const bool useCenterline, 
             const std::vector<typename AverageAngleLinkage_T<Real2_>::Joint> &joints, const std::vector<typename AverageAngleLinkage_T<Real2_>::RodSegment> &segments,
             const RodMaterial &homogMat, Real2_ initMinRL, const SuiteSparseMatrix &segmentRestLenToEdgeRestLenMapTranspose,
             const VecX_T<Real2_> &perSegmentRestLen, const Linkage_dPC &designParameter_config, std::vector<size_t> freeAngles = {}) {
        Base::set(joints, segments, homogMat, initMinRL, segmentRestLenToEdgeRestLenMapTranspose, perSegmentRestLen, designParameter_config);
        Base::setFreeAngles(freeAngles);
        m_surface_path = surface_path;
        set_target_surface_fitter(surface_path, useCenterline);
        m_l0 = BBox<Point3D>(Base::deformedPoints()).dimensions().norm();
        m_E0 = Base::homogenousMaterial().youngModulus * Base::homogenousMaterial().area * Base::getPerSegmentRestLength().sum();
    }

    template<typename Real2_>
    void set(const Eigen::MatrixX3d &targetV, const Eigen::MatrixX3i &targetF, const bool useCenterline, 
             const std::vector<typename AverageAngleLinkage_T<Real2_>::Joint> &joints, const std::vector<typename AverageAngleLinkage_T<Real2_>::RodSegment> &segments,
             const RodMaterial &homogMat, Real2_ initMinRL, const SuiteSparseMatrix &segmentRestLenToEdgeRestLenMapTranspose,
             const VecX_T<Real2_> &perSegmentRestLen, const Linkage_dPC &designParameter_config, std::vector<size_t> freeAngles = {}) {
        Base::set(joints, segments, homogMat, initMinRL, segmentRestLenToEdgeRestLenMapTranspose, perSegmentRestLen, designParameter_config);
        Base::setFreeAngles(freeAngles);
        m_surface_path = "";
        set_target_surface_fitter(targetV, targetF, useCenterline);
        m_l0 = BBox<Point3D>(Base::deformedPoints()).dimensions().norm();
        m_E0 = Base::homogenousMaterial().youngModulus * Base::homogenousMaterial().area * Base::getPerSegmentRestLength().sum();
    }

    // Avoid accidentally copying linkages around for performance reasons;
    // explicitly use RodLinkage::set instead.
    // If we choose to offer this operator in the future, it should be
    // implemented as a call to set (the joint linkage pointers must be updated)
    AverageAngleSurfaceAttractedLinkage_T &operator=(const AverageAngleSurfaceAttractedLinkage_T &b) = delete;

    void set_target_surface_fitter(const std::string &surface_path, const bool &useCenterline) {
	    // Initialize the target_surface_fitter so we can penalize the linkage's rigid transformation w.r.t the target surface during compute equilibrium.
        target_surface_fitter.setUseCenterline(*this, useCenterline, m_attraction_tgt_joint_weight);
        // Unless the user specifies otherwise, use the current deployed linkage joint positions as the target
        target_surface_fitter.joint_pos_tgt = Base::jointPositions();
        target_surface_fitter.loadTargetSurface(*this, surface_path);
        target_surface_fitter.holdClosestPointsFixed = true;
        // Trade off between fitting to the individual joint targets and the target surface.
        target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(*this, 0.01);
    }

    void set_target_surface_fitter(const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F, const bool &useCenterline) {
	    // Initialize the target_surface_fitter so we can penalize the linkage's rigid transformation w.r.t the target surface during compute equilibrium.
        target_surface_fitter.setUseCenterline(*this, useCenterline, m_attraction_tgt_joint_weight);
        // Unless the user specifies otherwise, use the current deployed linkage joint positions as the target
        target_surface_fitter.joint_pos_tgt = Base::jointPositions();
        target_surface_fitter.setTargetSurface(*this, V, F);
        target_surface_fitter.holdClosestPointsFixed = true;
        // Trade off between fitting to the individual joint targets and the target surface.
        target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(*this, 0.01);
    }

    void setTargetSurface(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F){ 
        bool hold = target_surface_fitter.holdClosestPointsFixed;
        // Force recomputing closest points
        target_surface_fitter.holdClosestPointsFixed = false;
        target_surface_fitter.setTargetSurface(*this, V, F); 
        target_surface_fitter.updateClosestPoints(*this);
        target_surface_fitter.holdClosestPointsFixed = hold;
    }
    Eigen::MatrixXd getTargetSurfaceVertices() const { return target_surface_fitter.getTargetSurfaceVertices(); }
    Eigen::MatrixXi getTargetSurfaceFaces()    const { return target_surface_fitter.getTargetSurfaceFaces(); }
    Eigen::MatrixXd getTargetSurfaceNormals()  const { return target_surface_fitter.getTargetSurfaceNormals(); }

    // Need to call update closest point at each iteration of the compute equilibrium.
    void setDoFs(const Eigen::Ref<const VecX> &dofs, bool spatialCoherence = false) {
        // std::cout<<"complaint happens before this! 1"<<std::endl;
        Base::setDoFs(dofs, spatialCoherence);
        // std::cout<<"complaint happens before this! 2"<<std::endl;

        target_surface_fitter.updateClosestPoints(*this);
    }

    void setExtendedDoFs(const Eigen::Ref<const VecX> &dofs, bool spatialCoherence = false) {
        Base::setExtendedDoFs(dofs, spatialCoherence);
        target_surface_fitter.updateClosestPoints(*this);
    }

    void setExtendedDoFsPSRL(const Eigen::Ref<const VecX> &dofs, bool spatialCoherence = false) {
        Base::setExtendedDoFsPSRL(dofs, spatialCoherence);
        target_surface_fitter.updateClosestPoints(*this);
    }
    // Need to override the energy, gradient, and hessian function to include the closeness terms.

	// Elastic energy stored in the linkage
    Real_ energy(SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full) const {
        Real result = 0.0;
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Elastic))
            result += Base::energy();
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Attraction))
            result += (attraction_weight * m_E0 / (m_l0 * m_l0)) * target_surface_fitter.objective(*this);
        return result;
    }

    void add_surface_attraction_gradient(VecX &result) const {
        target_surface_fitter.accumulateGradient(*this, result, attraction_weight * m_E0 / (m_l0 * m_l0));
    }

    // Gradient wrt average angle variables
    VecX gradient(bool updatedSource = false, SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full, bool variableDesignParameters = false, bool designParameterOnly = false, const bool skipBRods = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".gradient");
        VecX gradient(variableDesignParameters ? Base::numExtendedDoF() : Base::numDoF());
        gradient.setZero();
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Elastic))
            gradient += Base::gradient(updatedSource, EnergyType::Full, variableDesignParameters, designParameterOnly, skipBRods);
        if (designParameterOnly)
            return gradient;
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Attraction)) {
            add_surface_attraction_gradient(gradient);
        }
        return gradient;
    }

    // Gradient wrt joint angle variables
    VecX gradientJA(bool updatedSource = false, SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full, bool variableDesignParameters = false, bool designParameterOnly = false, const bool skipBRods = false) const {
        VecX gradient(variableDesignParameters ? Base::numExtendedDoF() : Base::numDoF());
        gradient.setZero();
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Elastic))
            gradient += Base::gradientJA(updatedSource, EnergyType::Full, variableDesignParameters, designParameterOnly, skipBRods);
        if (designParameterOnly)
            return gradient;
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Attraction)) {
            add_surface_attraction_gradient(gradient);
        }
        return gradient;
    }

    VecX gradientBend(bool updatedSource = false, bool variableDesignParameters = false, bool designParameterOnly = false, const bool skipBRods = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".gradientBend");
        return Base::gradient(updatedSource, EnergyType::Bend, variableDesignParameters, designParameterOnly, skipBRods);
    }
    VecX gradientTwist(bool updatedSource = false, bool variableDesignParameters = false, bool designParameterOnly = false, const bool skipBRods = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".gradientTwist");
        return Base::gradient(updatedSource, EnergyType::Twist, variableDesignParameters, designParameterOnly, skipBRods);
    }
    VecX gradientStretch(bool updatedSource = false, bool variableDesignParameters = false, bool designParameterOnly = false, const bool skipBRods = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".gradientStretch");
        return Base::gradient(updatedSource, EnergyType::Stretch, variableDesignParameters, designParameterOnly, skipBRods);
    }
    
    VecX gradientPerSegmentRestlen(bool updatedSource = false, SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full) const {
        VecX gradient(Base::numExtendedDoFPSRL());
        gradient.setZero();
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Elastic))
            gradient += Base::gradientPerSegmentRestlen(updatedSource, EnergyType::Full);
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Attraction))
            add_surface_attraction_gradient(gradient);
        return gradient;
    }


    // Gradient of elastic energy with respect to the design parameters
    VecX grad_design_parameters(bool updatedSource = false) const {
        auto gPerEdgeRestLen = gradient(updatedSource, SurfaceAttractionEnergyType::Full, true, /* only compute design parameter components (but the vector is still full length, the DoF part is just zeros) */ true);
        VecX result(Base::numDesignParams());
        result.setZero();
        if (Base::m_linkage_dPC.restKappa) result.head(Base::numRestKappaVars()) = gPerEdgeRestLen.segment(Base::designParameterOffset(), Base::numRestKappaVars());
        if (Base::m_linkage_dPC.restLen) Base::m_segmentRestLenToEdgeRestLenMapTranspose.applyRaw(gPerEdgeRestLen.tail(Base::numRestLengths()).data(), result.tail(Base::numSegments()).data(), /* no transpose */ false);
        return result;
    }

    // The number of non-zeros in the Hessian's sparsity pattern (a tight
    // upper bound for the number of non-zeros for any configuration).
    size_t hessianNNZ(bool variableDesignParameters = false) const {
        return Base::hessianNNZ(variableDesignParameters);
    }

    void add_surface_attraction_hessian(CSCMat &H) const {
        size_t num_projection_pos = target_surface_fitter.getUseCenterline() ? Base::numCenterlinePos() : Base::numJoints();
        Real weight = (attraction_weight * m_E0 / (m_l0 * m_l0));
        for (size_t i = 0; i < num_projection_pos; ++i) {
            const size_t varOffset = target_surface_fitter.getUseCenterline() ? Base::dofOffsetForCenterlinePos(i) : Base::dofOffsetForJoint(i);
            H.addDiagBlock(varOffset, weight * target_surface_fitter.pt_project_hess(i));
        }

        for (size_t ji = 0; ji < Base::numJoints(); ++ji) {
            const size_t varOffset = Base::dofOffsetForJoint(ji);
            for (size_t c = 0; c < 3; ++c)
                H.addDiagEntry(varOffset + c, weight * target_surface_fitter.W_diag_joint_pos[3 * ji + c]);
        }
    }

    // Accumulate the Hessian into the sparse matrix "H," which must already be initialized
    // with the sparsity pattern.
    void hessian(CSCMat &H, SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full, const bool variableDesignParameters = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".hessian");
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Elastic))
            Base::hessian(H, EnergyType::Full, variableDesignParameters);
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Attraction))
            add_surface_attraction_hessian(H);
    }

    // Hessian of the linkage's elastic energy and surface attraction energy with respect to all degrees of freedom.
    TMatrix hessian(SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full, const bool variableDesignParameters = false) const {
        auto H = Base::hessianSparsityPattern(variableDesignParameters);
        hessian(H, surface_eType, variableDesignParameters);
        return H.getTripletMatrix();
    }

    void hessianPerSegmentRestlen(CSCMat &H, SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full) const {
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Elastic))
            Base::hessianPerSegmentRestlen(H, EnergyType::Full);
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Attraction))
            add_surface_attraction_hessian(H);
    }

    TMatrix hessianPerSegmentRestlen(SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full) const {
        auto H = Base::hessianPerSegmentRestlenSparsityPattern();
        hessianPerSegmentRestlen(H, surface_eType);
        return H.getTripletMatrix();
    }

    // TODO: reimplement using TargetSurfaceFitter::applyHessian...
    void apply_surface_attraction_hessian(VecX &result, const VecX &v, const HessianComputationMask &mask) const {
        if (!(mask.dof_in && mask.dof_out)) return; // This term is dof-dof only...

        size_t num_projection_pos = target_surface_fitter.getUseCenterline() ? Base::numCenterlinePos() : Base::numJoints();
        for (size_t i = 0; i < num_projection_pos; ++i) {
            const Mat3 &pt_project_hess_var = (attraction_weight / (m_l0 * m_l0)) * target_surface_fitter.pt_project_hess(i);
            const size_t varOffset = target_surface_fitter.getUseCenterline() ? Base::dofOffsetForCenterlinePos(i) : Base::dofOffsetForJoint(i);
            for (size_t comp_a = 0; comp_a < 3; ++comp_a)
                for (size_t comp_b = comp_a; comp_b < 3; ++comp_b) {
                    auto sensitivity = pt_project_hess_var(comp_a, comp_b);
                    result[varOffset + comp_a] += v[varOffset + comp_b] * sensitivity;
                }
        }

        for (size_t i = 0; i < Base::numJoints(); ++i) {
            const Mat3 &pt_tgt_hess_var = (attraction_weight / (m_l0 * m_l0)) * target_surface_fitter.pt_tgt_hess(i);
            const size_t varOffset = Base::dofOffsetForJoint(i);
            for (size_t comp_a = 0; comp_a < 3; ++comp_a) {
                for (size_t comp_b = comp_a; comp_b < 3; ++comp_b) {
                    auto sensitivity = pt_tgt_hess_var(comp_a, comp_b);
                    result[varOffset + comp_a] += v[varOffset + comp_b] * sensitivity;
                }
            }
        }
    }

    VecX applyHessian(const VecX &v, bool variableDesignParameters = false, const HessianComputationMask &mask = HessianComputationMask(), SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full) const {
        VecX result(v.size());
        result.setZero();
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Elastic))
            result += Base::applyHessian(v, variableDesignParameters, mask);
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Attraction))
            apply_surface_attraction_hessian(result, v, mask);
        return result;

    }

    VecX applyHessianPerSegmentRestlen(const VecX &v, const HessianComputationMask &mask = HessianComputationMask(), SurfaceAttractionEnergyType surface_eType = SurfaceAttractionEnergyType::Full) const {
        VecX result(v.size());
        result.setZero();
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Elastic))
            result += Base::applyHessianPerSegmentRestlen(v, mask);
        if ((surface_eType == SurfaceAttractionEnergyType::Full) || (surface_eType == SurfaceAttractionEnergyType::Attraction))
            apply_surface_attraction_hessian(result, v, mask);
        return result;
    }

    std::string get_surface_path() const { return m_surface_path; }
    const TargetSurfaceFitter &get_target_surface_fitter() const { return target_surface_fitter; }
    void set_attraction_tgt_joint_weight(Real weight) {
        m_attraction_tgt_joint_weight = weight;
        target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(*this, m_attraction_tgt_joint_weight);
    }
    Real get_attraction_tgt_joint_weight() const { return m_attraction_tgt_joint_weight; }

    const Eigen::VectorXd getTargetJointsPosition() { return target_surface_fitter.joint_pos_tgt; }
    void setTargetJointsPosition(Eigen::VectorXd input_target_joint_pos) { target_surface_fitter.setTargetJointsPositions(input_target_joint_pos); }

    Real attraction_weight = 0.0001;
    Real get_l0() const { return m_l0; }
    Real get_E0() const { return m_E0; }
    void set_use_centerline(bool useCenterline) {
        target_surface_fitter.setUseCenterline(*this, useCenterline, m_attraction_tgt_joint_weight);
        target_surface_fitter.forceUpdateClosestPoints(*this);
    }
    bool get_use_centerline() const { return target_surface_fitter.getUseCenterline(); }

    bool get_holdClosestPointsFixed() {
        return target_surface_fitter.holdClosestPointsFixed;
    }
    void set_holdClosestPointsFixed(bool holdClosestPointsFixed) {
        target_surface_fitter.holdClosestPointsFixed = holdClosestPointsFixed;
    }

    void scaleJointWeights(Real jointPosWeight, Real featureMultiplier = 1.0, const std::vector<size_t> &additional_feature_pts = std::vector<size_t>()) {
        m_attraction_tgt_joint_weight = jointPosWeight;
        target_surface_fitter.scaleJointWeights(*this, m_attraction_tgt_joint_weight, featureMultiplier, additional_feature_pts);
    }

    void saveAttractionSurface(const std::string &path) {
        target_surface_fitter.saveTargetSurface(path);
    }

    void reflectAttractionSurface(size_t ji) { target_surface_fitter.reflect(*this, ji) ; }
    Eigen::VectorXd get_linkage_closest_point() const { return target_surface_fitter.linkage_closest_surf_pts; }

    std::vector<Real> get_squared_distance_to_target_surface(Eigen::VectorXd query_point_list) const {return target_surface_fitter.get_squared_distance_to_target_surface(query_point_list); };
    Eigen::VectorXd get_closest_point_for_visualization(Eigen::VectorXd query_point_list) const {return target_surface_fitter.get_closest_point_for_visualization(query_point_list); };

    Eigen::VectorXd get_closest_point_normal(Eigen::VectorXd query_point_list) {return target_surface_fitter.get_closest_point_normal(query_point_list); };

    std::string mangledName() const { return "AverageAngleSurfaceAttractedLinkage<" + autodiffOrNotString<Real_>() + ">"; }

    void constructSegmentRestLenToEdgeRestLenMapTranspose(const VecX_T<Real_> &segmentRestLenGuess) {
        Base::constructSegmentRestLenToEdgeRestLenMapTranspose(segmentRestLenGuess);
        target_surface_fitter.forceUpdateClosestPoints(*this);
        target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(*this, m_attraction_tgt_joint_weight);
    }
    virtual ~AverageAngleSurfaceAttractedLinkage_T() { }
protected:
    TargetSurfaceFitter target_surface_fitter;
    std::string m_surface_path;
    Real m_l0 = 1, m_E0  = 1;
    Real m_attraction_tgt_joint_weight = 0.001;

};

#endif /* end of include guard: AVERAGEANGLESURFACEATTRACTEDLINKAGE_HH */
