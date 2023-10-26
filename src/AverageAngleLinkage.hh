#ifndef AVERAGEANGLELINKAGE_HH
#define AVERAGEANGLELINKAGE_HH

#include "RodLinkage.hh"
#include "RectangularBox.hh"
#include <rotation_optimization.hh>
#include <array>
#include <tuple>
#include<algorithm>

#include <MeshFEM/Geometry.hh>
#include "compute_equilibrium.hh"
#include "LOMinAngleConstraint.hh"
#include "TargetSurfaceFitter.hh"


// Templated to support automatic differentiation types.
template<typename Real_>
struct AverageAngleLinkage_T;

using AverageAngleLinkage = AverageAngleLinkage_T<Real>;

// A derived class of RodLinkage where the average angle of the linkage is directly exposed as a simulation variable. 
// The main purpose of this class is to handle this change of variables. 
// To access any other information such as joints and segments, please refer back to the base class. 
// Change of variable matrix:
// alpha_0    = |  1  0  0  ...  0  1 | |a_tilde_0|
// alpha_1    = | -1  1  0  ...  0  1 | |a_tilde_1|
// alpha_2    = |  0 -1  1  ...  0  1 | |a_tilde_2|
// alpha_3    = |  0  0  0  ...  0  1 | |a_tilde_3|
// ...        = | ...     ...     ... | |   ...   |
// alpha_n-1  = |  0  0  0  ... -1  1 | | *a_bar* |

// alpha_i are the original angle variables; a_tildes are the new variables.
template<typename Real_>
struct AverageAngleLinkage_T : public RodLinkage_T<Real_> {
    using Base = RodLinkage_T<Real_>;
    static constexpr size_t defaultSubdivision   = Base::defaultSubdivision;
    static constexpr bool defaultConsistentAngle = Base::defaultConsistentAngle;
    using TMatrix     = typename Base::TMatrix;
    using CSCMat      = typename Base::CSCMat;
    using VecX        = typename Base::VecX;
    using Vec3        = typename Base::Vec3;
    using Mat3        = Mat3_T<Real>;
    using EnergyType  = typename Base::EnergyType;
    using Linkage_dPC = typename Base::Linkage_dPC;
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();

    // TODO: no attraction for AAL?
    enum class AverageAngleLinkageEnergyType { Full, Attraction, Elastic };

    // Construct empty linkage, to be initialized later by calling set.
    AverageAngleLinkage_T() : RodLinkage_T<Real_>() { }

    // Forward all constructor arguments to set(...)
    template<typename... Args>
    AverageAngleLinkage_T(Args&&... args) : RodLinkage_T<Real_>() {
        set(std::forward<Args>(args)...);
    }

    AverageAngleLinkage_T(const AverageAngleLinkage_T &linkage) : RodLinkage_T<Real_>() { 
        set(linkage, linkage.getFreeAngles()); 
    }

    AverageAngleLinkage_T(const AverageAngleLinkage_T &linkage, std::vector<size_t> freeAngles) : RodLinkage_T<Real_>() { 
        set(linkage, freeAngles); 
    } // The above forwarding constructor confuses pybind11

    AverageAngleLinkage_T(const RodLinkage_T<Real_> &linkage, std::vector<size_t> freeAngles = {}) : RodLinkage_T<Real_>() { 
        Base::set(linkage); 
        setFreeAngles(freeAngles);
    }

    template<typename Real2_>
    void set(const AverageAngleLinkage_T<Real2_> &linkage, std::vector<size_t> freeAngles) { 
        Base::set(linkage); 
        setFreeAngles(freeAngles);
    }

    template<typename Real2_>
    void set(const AverageAngleLinkage_T<Real2_> &linkage) { 
        Base::set(linkage); 
        setFreeAngles(linkage.getFreeAngles());
    }

    template<typename Real2_>
    void set(const std::vector<typename RodLinkage_T<Real2_>::Joint> &joints, const std::vector<typename RodLinkage_T<Real2_>::RodSegment> &segments,
             const RodMaterial &homogMat, Real2_ initMinRL, const SuiteSparseMatrix &segmentRestLenToEdgeRestLenMapTranspose,
             const VecX_T<Real2_> &perSegmentRestLen, const Linkage_dPC &designParameter_config,
             std::vector<size_t> freeAngles = {}){
        Base::set(joints, segments, homogMat, initMinRL, segmentRestLenToEdgeRestLenMapTranspose, perSegmentRestLen, designParameter_config);
        setFreeAngles(freeAngles);
    }

    void set(const std::string linkage_path,
             size_t subdivision = defaultSubdivision, 
             bool consistentAngle = defaultConsistentAngle,
             InterleavingType rod_interleaving_type = InterleavingType::noOffset,
             std::vector<std::function<Pt3_T<Real_>(Real_, bool)>> edge_callbacks = {}, 
             std::vector<Vec3> input_joint_normals = {},
             std::vector<size_t> freeAngles = {}){
        Base::set(linkage_path, subdivision, consistentAngle, rod_interleaving_type, edge_callbacks, input_joint_normals);
        setFreeAngles(freeAngles);
    }

    void set(std::vector<MeshIO::IOVertex > &vertices,
             std::vector<MeshIO::IOElement> &edges,   
             size_t subdivision = defaultSubdivision, 
             bool consistentAngle = defaultConsistentAngle,
             InterleavingType rod_interleaving_type = InterleavingType::noOffset,
             std::vector<std::function<Pt3_T<Real_>(Real_, bool)>> edge_callbacks = {}, 
             std::vector<Vec3> input_joint_normals = {},
             std::vector<size_t> freeAngles = {}){
        Base::set(vertices, edges, subdivision, consistentAngle, rod_interleaving_type, edge_callbacks, input_joint_normals);
        setFreeAngles(freeAngles);
    }

    void set(const Eigen::MatrixX3d &vertices,
             const Eigen::MatrixX2i &edges,   
             size_t subdivision = defaultSubdivision, 
             bool consistentAngle = defaultConsistentAngle,
             InterleavingType rod_interleaving_type = InterleavingType::noOffset,
             std::vector<std::function<Pt3_T<Real_>(Real_, bool)>> edge_callbacks = {}, 
             std::vector<Vec3> input_joint_normals = {},
             std::vector<size_t> freeAngles = {}){
        Base::set(vertices, edges, subdivision, consistentAngle, rod_interleaving_type, edge_callbacks, input_joint_normals);
        setFreeAngles(freeAngles);
    }

    void setFreeAngles(std::vector<size_t> freeAngles){
        m_freeAngles = freeAngles;
        m_buildAngleIndexForDoFIndex();
        m_constructActuatedAngles();
        m_constructAverageAngleToJointAngleMapTranspose();
        m_constructAverageAngleToJointAngleMapTranspose_AllJointVars();
    }

    // Avoid accidentally copying linkages around for performance reasons;
    // explicitly use RodLinkage::set instead.
    // If we choose to offer this operator in the future, it should be
    // implemented as a call to set (the joint linkage pointers must be updated)
    AverageAngleLinkage_T &operator=(const AverageAngleLinkage_T &b) = delete;


    // m_AverageAngleToJointAngle
    const VecX applyTransformation(const VecX &AAV) const {
        return m_averageAngleToJointAngleMapTranspose.apply(AAV, /* transpose */ true);
    }
    // m_JointAngleToAverageAngle
    const VecX applyTransformationTranspose(const VecX &JAV) const {
        return m_averageAngleToJointAngleMapTranspose.apply(JAV, /* transpose */ false);
    }
    // The DoFs vector for AverageAngleLinkage is very similar to RodLinkage 
    // except the variables at the location for the angle variables in the 
    // joints are now the new set of variables. 
    const VecX applyTransformationDoFSize(const Eigen::Ref<const VecX> &dofs) const {
        // Convert AA variables to joint angle variables. 
        std::vector<size_t> angleDoFIndices = Base::jointAngleDoFIndices();
        VecX AAVars(angleDoFIndices.size());
        for (size_t i = 0; i < angleDoFIndices.size(); ++i) {
            AAVars(i) = dofs(angleDoFIndices[i]);
        }
        VecX JAVars = applyTransformation(AAVars);
        VecX base_dofs = dofs;

        for (size_t i = 0; i < angleDoFIndices.size(); ++i) {
            base_dofs(angleDoFIndices[i]) = JAVars(i);
        }
        return base_dofs;
    }

    const VecX applyTransformationTransposeDoFSize(const Eigen::Ref<const VecX> &vec) const {
        // Convert joint angle variables to average angle variables. 
        std::vector<size_t> angleDoFIndices = Base::jointAngleDoFIndices();
        VecX JAVars(angleDoFIndices.size());
        for (size_t i = 0; i < angleDoFIndices.size(); ++i) {
            JAVars(i) = vec(angleDoFIndices[i]);
        }
        VecX AAVars = applyTransformationTranspose(JAVars);
        VecX derived_vec = vec;

        for (size_t i = 0; i < angleDoFIndices.size(); ++i) {
            derived_vec(angleDoFIndices[i]) = AAVars(i);
        }
        return derived_vec;
    }

    // Outputs AA DoFs
    VecX getDoFs() const {
        VecX base_dofs = Base::getDoFs();
        VecX result(base_dofs.size());
        result.setZero();
        std::vector<size_t> angleDoFIndices = Base::jointAngleDoFIndices();
        VecX JAVars(angleDoFIndices.size());
        Real_ A_bar = 0.0;
        for (size_t i = 0; i < angleDoFIndices.size(); ++i) {
            JAVars(i) = base_dofs(angleDoFIndices[i]);
            if (isActuatedAngle(i)){
                A_bar = A_bar + JAVars(i);
            }
        }
        VecX AAVars(Base::numJoints());
        AAVars.setZero();
        A_bar = A_bar / Real_ (m_actuatedAngles.size());
        
        size_t prevActAngleIndex = NONE;
        for (size_t i = 0; i < Base::numJoints(); ++i) {
            if (i == m_firstActuatedAngle) {
                AAVars(i) = JAVars(i) - A_bar;
                prevActAngleIndex = i;
            } else if (i == m_lastActuatedAngle) {
                AAVars(i) = A_bar;
            } else if (isActuatedAngle(i)){
                AAVars(i) = JAVars(i) + AAVars(prevActAngleIndex) - A_bar;
                prevActAngleIndex = i;
            } else {
                AAVars(i) = JAVars(i);
            }
        }
        VecX derived_dofs = base_dofs;
        for (size_t i = 0; i < angleDoFIndices.size(); ++i) {
            derived_dofs(angleDoFIndices[i]) = AAVars(i);
        }
        return derived_dofs;
    }

    // Takes AA DoFs
    void setDoFs(const Eigen::Ref<const VecX> &dofs, bool spatialCoherence = false) {
        VecX base_dofs = applyTransformationDoFSize(dofs);
        Base::setDoFs(base_dofs, spatialCoherence);
    }

    VecX getExtendedDoFs() const {
        VecX base_edofs = Base::getExtendedDoFs();
        base_edofs.head(Base::numDoF()) = getDoFs();
        return base_edofs;
    }

    void setExtendedDoFs(const Eigen::Ref<const VecX> &dofs, bool spatialCoherence = false) {
        VecX base_dofs = applyTransformationDoFSize(dofs);
        Base::setExtendedDoFs(base_dofs, spatialCoherence);
    }

    VecX getExtendedDoFsPSRL() const {
        VecX base_edofs = Base::getExtendedDoFsPSRL();
        base_edofs.head(Base::numDoF()) = getDoFs();
        return base_edofs;
    }

    void setExtendedDoFsPSRL(const Eigen::Ref<const VecX> &dofs, bool spatialCoherence = false) {
        VecX base_dofs = applyTransformationDoFSize(dofs);
        Base::setExtendedDoFsPSRL(base_dofs, spatialCoherence);
    }
    // Need to override the energy, gradient, and hessian function to include the closeness terms.

	// Elastic energy stored in the linkage
    Real_ energy(EnergyType type = EnergyType::Full) const {
        return Base::energy(type);
    }

    // Gradient wrt average angle variables
    VecX gradient(bool updatedSource = false, EnergyType eType = EnergyType::Full, bool variableDesignParameters = false, bool designParameterOnly = false, const bool skipBRods = false) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".gradient");
        return applyTransformationTransposeDoFSize(Base::gradient(updatedSource, eType, variableDesignParameters, designParameterOnly, skipBRods));
    }

    // Gradient wrt joint angle variables
    VecX gradientJA(bool updatedSource = false, EnergyType eType = EnergyType::Full, bool variableDesignParameters = false, bool designParameterOnly = false, const bool skipBRods = false) const {
        return Base::gradient(updatedSource, eType, variableDesignParameters, designParameterOnly, skipBRods);
    }

    VecX gradientPerSegmentRestlen(bool updatedSource = false, EnergyType eType = EnergyType::Full) const {
        return applyTransformationTransposeDoFSize(Base::gradientPerSegmentRestlen(updatedSource, eType));
    }

    // The number of non-zeros in the Hessian's sparsity pattern (a tight
    // upper bound for the number of non-zeros for any configuration).
    size_t hessianNNZ(bool variableDesignParameters = false) const;

    CSCMat hessianSparsityPattern(bool variableDesignParameters = false, Real_ val = 0.0) const;

    auto hessian(EnergyType eType = EnergyType::Full, bool variableDesignParameters = false) const -> TMatrix {
        auto H = hessianSparsityPattern(variableDesignParameters);
        hessian(H, eType, variableDesignParameters);
        return H.getTripletMatrix();
    }

    // Accumulate the Hessian into the sparse matrix "H," which must already be initialized
    // with the sparsity pattern.
    void hessian(CSCMat &H, EnergyType eType = EnergyType::Full, const bool variableDesignParameters = false) const;

    VecX applyHessian(const VecX &v, bool variableDesignParameters = false, const HessianComputationMask &mask = HessianComputationMask()) const {
        return applyTransformationTransposeDoFSize(Base::applyHessian(applyTransformationDoFSize(v), variableDesignParameters, mask));
    }

    VecX applyHessianPerSegmentRestlen(const VecX &v, const HessianComputationMask &mask = HessianComputationMask()) const {
        return applyTransformationTransposeDoFSize(Base::applyHessianPerSegmentRestlen(applyTransformationDoFSize(v), mask));
    }

    // useLumped: whether to use the rods' diagonal lumped mass matrices.
    // The linkage's mass matrix will be non-diagonal in either case because the joint
    // parameters control multiple rod centerline point/theta variables.
    void massMatrix(CSCMat &M, bool updatedSource = false, bool useLumped = false) const;
    TMatrix massMatrix(bool updatedSource = false, bool useLumped = false) const {
        auto M = hessianSparsityPattern();
        massMatrix(M, updatedSource, useLumped);
        return M.getTripletMatrix();
    }
    // Probably not useful: this matrix is usually not positive definite
    VecX lumpedMassMatrix(bool updatedSource = false) const {
        return Base::lumpedMassMatrix(updatedSource);
    }

    std::string mangledName() const { return "AverageAngleLinkage<" + autodiffOrNotString<Real_>() + ">"; }

    void constructSegmentRestLenToEdgeRestLenMapTranspose(const VecX_T<Real_> &segmentRestLenGuess) {
        Base::constructSegmentRestLenToEdgeRestLenMapTranspose(segmentRestLenGuess);
    }

    SuiteSparseMatrix getAverageAngleToJointAngleMapTranspose() { return m_averageAngleToJointAngleMapTranspose; }
    SuiteSparseMatrix getAverageAngleToJointAngleMapTranspose_AllJointVars() { return m_averageAngleToJointAngleMapTranspose_AllJointVars; }

    bool isAngleVar(size_t di) const { return getAngleIndexFromDoFIndex(di) != NONE; }
    bool isFreeAngleVar(size_t di) const { return isAngleVar(di) && isFreeAngle(getAngleIndexFromDoFIndex(di)); }
    bool isActuatedAngleVar(size_t di) const { return isAngleVar(di) && !isFreeAngle(getAngleIndexFromDoFIndex(di)); }
    bool isFreeAngle(size_t ai) const { return std::count(m_freeAngles.begin(), m_freeAngles.end(), ai); }
    bool isActuatedAngle(size_t ai) const { return !isFreeAngle(ai); }

    size_t getAverageAngleIndex() const { return m_averageAngleIndex; }
    std::vector<size_t> getFreeAngles() const { return m_freeAngles; }
    std::vector<size_t> getActuatedAngles() const { return m_actuatedAngles; }
    // Compute the average over all joints of the joint opening angle.
    Real_ getAverageActuatedJointsAngle() const { 
        VecX base_dofs = Base::getDoFs();
        std::vector<size_t> angleDoFIndices = Base::jointAngleDoFIndices();
        Real_ A_bar = 0.0;
        for (size_t i = 0; i < angleDoFIndices.size(); ++i) {
            if (isActuatedAngle(i)){
                A_bar = A_bar + base_dofs(angleDoFIndices[i]);
            }
        }
        A_bar = A_bar / Real_ (m_actuatedAngles.size());
        return A_bar;
    }

    size_t getAngleIndexFromDoFIndex(size_t di) const { 
        if (di > Base::numDoF()) throw std::runtime_error("The angle index is out of range (DoF to angle)!");
        return m_angleIndexFromDoFIndex.at(di); 
    }

    size_t getDoFIndexFromAngleIndex(size_t ai) const { 
        if (ai > Base::numJoints()) throw std::runtime_error("The angle index is out of range (angle to DoF)!");
        return Base::jointAngleDoFIndices().at(ai); 
    }
    virtual ~AverageAngleLinkage_T() { }
protected:
    Real m_l0 = 1.0, m_E0  = 1.0;
    SuiteSparseMatrix m_averageAngleToJointAngleMapTranspose;
    SuiteSparseMatrix m_averageAngleToJointAngleMapTranspose_AllJointVars;
    std::vector<size_t> m_angleIndexFromDoFIndex;
    std::vector<size_t> m_actuatedAngles;
    std::vector<size_t> m_freeAngles;
    size_t m_firstActuatedAngle;
    size_t m_lastActuatedAngle;
    size_t m_averageAngleIndex;
    
    ////////////////////////////////////////////////////////////////////////////
    // Cache for hessian sparsity patterns
    ////////////////////////////////////////////////////////////////////////////
    mutable std::unique_ptr<CSCMat> m_cachedHessianSparsity, m_cachedHessianVarRLSparsity;
    void m_constructAverageAngleToJointAngleMapTranspose();
    void m_constructAverageAngleToJointAngleMapTranspose_AllJointVars();
    void m_constructActuatedAngles();

    void m_buildAngleIndexForDoFIndex();
};

#endif /* end of include guard: AVERAGEANGLELINKAGE_HH */
