////////////////////////////////////////////////////////////////////////////////
// AverageAngleCShellOptimization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Evaluates objective, constraints, and gradients for solving the following
//  optimal design problem for rod linkages:
//      min_p J(p)
//      s.t. c(p) = 0
//      J(p) =      gamma  / E_0 E(x_2D(p), p) +
//             (1 - gamma) / E_0 E(x_3D(p, alpha_t), p) +
//             beta / (2 l_0^2) ||x_3D(p, alpha_t) - x_tgt||_W^2
//      c(p) = || S_z x_2D(p) ||^2,
//
//      where x_2D is the equilibrium configuration of the closed linkage,
//            x_3D is the equilibrium configuration of the opened linkage,
//            x_tgt are the user-specified target positions for each joint
//            gamma, beta, W are weights
//            S_z selects the z component of each joint
//  See writeup/LinkageOptimization.pdf for more information.
*/
////////////////////////////////////////////////////////////////////////////////
#ifndef AVERAGEANGLECSHELLOPTIMIZATION_HH
#define AVERAGEANGLECSHELLOPTIMIZATION_HH

#include <MeshFEM/Geometry.hh>
#include "compute_equilibrium.hh"
#include "LOMinAngleConstraint.hh"
#include "TargetSurfaceFitter.hh"
#include "LinkageOptimization.hh"

template<template<typename> class Object>
struct AverageAngleCShellOptimization : public LinkageOptimization<Object>{
    using LO = LinkageOptimization<Object>;
    using LO::m_linesearch_base;
    using LO::m_base;
    using LO::target_surface_fitter;
    using LO::objective;
    using LO::invalidateAdjointState;
    using LO::m_numParams;
    using LO::m_E0;
    using LO::beta;
    using LO::m_l0;
    using LO::m_rigidMotionFixedVars;
    using LO::m_rm_constrained_joint;
    using LO::m_equilibrium_options;
    using LO::m_equilibriumSolveSuccessful;
    using LO::m_adjointStateIsCurrent;
    using LO::m_autodiffLinkagesAreCurrent;
    using LO::prediction_order;
    using LO::numParams;
    using LO::m_rl0;
    using LO::m_rk0;
    using LO::params;
    using LO::getRestLengthMinimizationWeight;
#if HAS_KNITRO
    using LO::optimize;
#endif
    using LO::apply_hess_J;
    using LO::J;
    // allowFlatActuation: whether we allow the application of average-angle actuation to enforce the minimum angle constraint at the beginning of optimization.
    AverageAngleCShellOptimization(Object<Real> &flat, Object<Real> &deployed, const NewtonOptimizerOptions &eopts = NewtonOptimizerOptions(), std::unique_ptr<LOMinAngleConstraint<Object>> &&minAngleConstraint = std::unique_ptr<LOMinAngleConstraint<Object>>(), int pinJoint = -1, 
                                  bool allowFlatActuation = true, bool optimizeTargetAngle = true, bool fixDeployedVars = true, const std::vector<size_t> &additionalFixedFlatVars = std::vector<size_t>(), const std::vector<size_t> &additionalFixedDeployedVars = std::vector<size_t>());

    // It is easier to pybind this constructor since there is no need to bind LOMinAngleConstraint.
    AverageAngleCShellOptimization(Object<Real> &flat, Object<Real> &deployed, const NewtonOptimizerOptions &eopts = NewtonOptimizerOptions(), Real minAngleConstraint = 0 , int pinJoint = -1, 
                                  bool allowFlatActuation = true, bool optimizeTargetAngle = true, bool fixDeployedVars = true, const std::vector<size_t>& additionalFixedFlatVars = std::vector<size_t>(), const std::vector<size_t> &additionalFixedDeployedVars = std::vector<size_t>()) 
        : AverageAngleCShellOptimization(flat, deployed, eopts, std::make_unique<LOMinAngleConstraint<Object>>(minAngleConstraint), pinJoint, allowFlatActuation, optimizeTargetAngle, fixDeployedVars, additionalFixedFlatVars, additionalFixedDeployedVars) {}

    // Evaluate at a new set of parameters and commit this change to the flat/deployed linkages (which
    // are used as a starting point for solving the line search equilibrium)
    void newPt(const Eigen::VectorXd &params) override {
        const size_t nfp = numFullParams();
        std::runtime_error mismatch("Dimension mismatch for the design parameters in newPt");
        if (params.size() != int(nfp))     throw mismatch;

        // if (m_optimizeTargetAngle){
        //     m_Linesearch_alpha_tgt = params[np];
        //     m_deployed_optimizer->get_problem().setLEQConstraintRHS(m_Linesearch_alpha_tgt);
        // }
        m_updateAdjointState(params); // update the adjoint state as well as the equilibrium, since we'll probably be needing gradients at this point.
        commitLinesearchLinkage();
    }

    Eigen::VectorXd gradp_J_target() { return gradp_J_target(params()); }
    Real c()         { return c(params()); }
    size_t numFullParams() const override { return m_numFullParams; }

    Real J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) override {
        const size_t nfp = numFullParams();
        std::runtime_error mismatch("Dimension mismatch for the design parameters in objective eval (from cshell optim)");
        if (params.size() != int(nfp))     throw mismatch;
        m_updateEquilibria(params);
        if (!m_equilibriumSolveSuccessful) return std::numeric_limits<Real>::max();

        objective.printReport();
        Real val = objective.value(opt_eType);
        return val;
    }

    Real J_target(const Eigen::Ref<const Eigen::VectorXd> &params) {
        m_updateEquilibria(params);
        return target_surface_fitter.objective(m_linesearch_deployed);
    }

    Real c(const Eigen::Ref<const Eigen::VectorXd> &params) override {
        m_updateEquilibria(params);
        return m_apply_S_z(m_linesearch_base.getDoFs()).squaredNorm();
    }

    Real angle_constraint(const Eigen::Ref<const Eigen::VectorXd> &params) override {
        if (!m_minAngleConstraint) throw std::runtime_error("No minimum angle constraint is applied.");
        m_updateEquilibria(params);
        return m_minAngleConstraint->eval(m_linesearch_base);
    }

    Eigen::VectorXd gradp_J               (const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) override ;
    Eigen::VectorXd gradp_J_target        (const Eigen::Ref<const Eigen::VectorXd> &params);
    Eigen::VectorXd gradp_c               (const Eigen::Ref<const Eigen::VectorXd> &params) override ;
    Eigen::VectorXd gradp_angle_constraint(const Eigen::Ref<const Eigen::VectorXd> &params) override ;
    
    // Jacobian vector product
    Eigen::VectorXd pushforward(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p);

    // Hessian matvec: H delta_p
    Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, Real coeff_c, Real coeff_angle_constraint, OptEnergyType opt_eType = OptEnergyType::Full) override ;
    
    // To handle the case when the target angle is also optimized
    Eigen::VectorXd getFullDesignParameters() override {
        const size_t np = numParams();
        if (!m_optimizeTargetAngle){
            return m_deployed.getDesignParameters();
        } else {
            Eigen::VectorXd params(np + 1);
            params.head(np) = m_deployed.getDesignParameters();
            params[np]      = m_alpha_tgt;
            return params;
        }
    }

    // To handle the case when the target angle is also optimized
    Eigen::VectorXd getLinesearchDesignParameters(){
        const size_t np = numParams();
        if (!m_optimizeTargetAngle){
            return m_linesearch_deployed.getDesignParameters();
        } else {
            Eigen::VectorXd params(np + 1);
            params.head(np) = m_linesearch_deployed.getDesignParameters();
            params[np]      = m_Linesearch_alpha_tgt;
            return params;
        }
    }

    // Access adjoint state for debugging
    Real       get_w_lambda() const { return m_w_lambda; }
    Eigen::VectorXd get_w_x() const { return m_w_x; }
    Eigen::VectorXd get_y()   const { return m_y; }
    Eigen::VectorXd get_s_x() const { return m_s_x; }
    Eigen::VectorXd get_delta_x3d() const { return m_delta_x3d; }
    Eigen::VectorXd get_delta_x2d() const { return m_delta_x2d; }
    Eigen::VectorXd get_delta_w_x() const { return m_delta_w_x; }
    Real       get_delta_w_lambda() const { return m_delta_w_lambda; }
    Eigen::VectorXd get_delta_s_x() const { return m_delta_s_x; }
    Eigen::VectorXd get_delta_y  () const { return m_delta_y; }

    Eigen::VectorXd get_delta_delta_x3d() const { return m_delta_delta_x3d; }
    Eigen::VectorXd get_delta_delta_x2d() const { return m_delta_delta_x2d; }
    Eigen::VectorXd get_second_order_x3d() const { return m_second_order_x3d; }
    Eigen::VectorXd get_second_order_x2d() const { return m_second_order_x2d; }

    // The base linkage from the parent class is the flat linkage.
    Object<Real> &getBaseLinkage()               { return m_base; }
    Object<Real> &getLinesearchBaseLinkage()     { return m_linesearch_base; }
    Object<Real> &getDeployedLinkage()           { return m_deployed; }
    Object<Real> &getLinesearchDeployedLinkage() { return m_linesearch_deployed; }

    void setLinkageInterleavingType(InterleavingType new_type) override {
        m_linesearch_base.  set_interleaving_type(new_type);
        m_base.             set_interleaving_type(new_type);
        m_diff_linkage_flat.set_interleaving_type(new_type);

        // Apply the new joint configuration to the rod segment terminals.
        m_linesearch_base.  setDoFs(  m_linesearch_base.getDoFs(), true /* set spatially coherent thetas */);
        m_base.             setDoFs(             m_base.getDoFs(), true /* set spatially coherent thetas */);
        m_diff_linkage_flat.setDoFs(m_diff_linkage_flat.getDoFs(), true /* set spatially coherent thetas */);

        m_linesearch_deployed.  set_interleaving_type(new_type);
        m_deployed.             set_interleaving_type(new_type);
        m_diff_linkage_deployed.set_interleaving_type(new_type);

        // Apply the new joint configuration to the rod segment terminals.
        m_linesearch_deployed.  setDoFs(  m_linesearch_deployed.getDoFs(), true /* set spatially coherent thetas */);
        m_deployed.             setDoFs(             m_deployed.getDoFs(), true /* set spatially coherent thetas */);
        m_diff_linkage_deployed.setDoFs(m_diff_linkage_deployed.getDoFs(), true /* set spatially coherent thetas */);
    }

    void setAverageAngleCShellOptimization(Object<Real> &flat, Object<Real> &deployed) {
        m_numParams     = flat.numDesignParams();
        m_numFullParams = flat.numDesignParams() + int(m_optimizeTargetAngle);
        m_E0 = deployed.energy();  
        m_linesearch_base.set(flat);
        m_linesearch_deployed.set(deployed);

        m_forceEquilibriumUpdate();
        commitLinesearchLinkage();
    }

    // Change the deployed linkage's opening angle by "alpha", resolving for equilibrium.
    // Side effect: commits the linesearch linkage (like calling newPt)
    void setTargetAngle(Real alpha) {
        m_Linesearch_alpha_tgt = alpha;
        const size_t idxAlphaBarDeployed = m_deployed.getAverageAngleIndex();
        Eigen::VectorXd curr_x3d         = m_deployed.getDoFs();
        curr_x3d[idxAlphaBarDeployed]    = alpha;
        m_deployed.setDoFs(curr_x3d);
        // m_deployed_optimizer->get_problem().setLEQConstraintRHS(alpha);

        m_linesearch_base    .set(m_base    );
        m_linesearch_deployed.set(m_deployed);

        m_forceEquilibriumUpdate();
        commitLinesearchLinkage();
    }

    // This one does not commit the linkage
    void setTargetAngleLinesearch(Real alpha) {
        m_Linesearch_alpha_tgt = alpha;
        const size_t idxAlphaBarDeployed = m_linesearch_deployed.getAverageAngleIndex();
        Eigen::VectorXd curr_x3d         = m_linesearch_deployed.getDoFs();
        curr_x3d[idxAlphaBarDeployed]    = alpha;
        m_linesearch_deployed.setDoFs(curr_x3d);
        // m_deployed_optimizer->get_problem().setLEQConstraintRHS(alpha);

        m_forceEquilibriumUpdate();
    }

    void setAllowFlatActuation(bool allow) {
        m_allowFlatActuation = allow;
        m_updateMinAngleConstraintActuation();
        m_forceEquilibriumUpdate();
        commitLinesearchLinkage();
    }

    void commitLinesearchLinkage() override {
        m_alpha_tgt = m_Linesearch_alpha_tgt;

        m_base    .set(m_linesearch_base);
        m_deployed.set(m_linesearch_deployed);
        // Stash the current factorizations to be reused at each step of the linesearch
        // to predict the equilibrium at the new design parameters.
        getFlatOptimizer()    .solver.stashFactorization();
        getDeployedOptimizer().solver.stashFactorization();
    }

    Real getTargetAngle() const { return m_alpha_tgt; }

    void setOptimizeTargetAngle(bool optimizeTargetAngle) {
        m_optimizeTargetAngle = optimizeTargetAngle;
        m_numFullParams       = m_numParams + int(m_optimizeTargetAngle);
        try{
            if (m_optimizeTargetAngle) { objective.get("ElasticEnergyDeployed").setUseEnvelopeTheorem(false); }
            else                       { objective.get("ElasticEnergyDeployed").setUseEnvelopeTheorem(true);  }
        } catch (...){
            std::cout << "ElasticEnergyDeployed does not belong to the objective." << std::endl;
        }
    }

    bool getOptimizeTargetAngle() const override { return m_optimizeTargetAngle; }
    bool getFixDeployedVars() const { return m_fixDeployedVars; }
    const std::vector<size_t> getFixedDeployedVars() { return m_deployed_optimizer->get_problem().fixedVars(); }
    const std::vector<size_t> getFixedFlatVars()     { return m_flat_optimizer->get_problem().fixedVars(); }

    // Construct/update a target surface for the surface fitting term by
    // inferring a smooth surface from the current joint positions.
    // Also update the closest point projections.
    void scaleJointWeights(Real jointPosWeight, Real featureMultiplier = 1.0, const std::vector<size_t> &additional_feature_pts = std::vector<size_t>()) {
        target_surface_fitter.scaleJointWeights(m_linesearch_base, jointPosWeight, featureMultiplier, additional_feature_pts); 
        objective.update(); 
        invalidateAdjointState(); 
    }

    void scaleFeatureJointWeights(Real jointPosWeight, Real featureMultiplier = 1.0, const std::vector<size_t> &feature_pts = std::vector<size_t>()) {
        target_surface_fitter.scaleFeatureJointWeights(m_linesearch_base, jointPosWeight, featureMultiplier, feature_pts); 
        objective.update(); 
        invalidateAdjointState(); 
    }

    void constructTargetSurface(size_t loop_subdivisions = 0, size_t num_extension_layers = 1, Eigen::Vector3d scale_factors = Eigen::Vector3d(1, 1, 1)) {
        target_surface_fitter.constructTargetSurface(m_linesearch_deployed, loop_subdivisions, num_extension_layers, scale_factors);
        invalidateAdjointState();
    }

    void loadTargetSurface(const std::string &path) {
        target_surface_fitter.loadTargetSurface(m_linesearch_deployed, path);
        invalidateAdjointState();
    }

    void setTargetSurface(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F){
        target_surface_fitter.setTargetSurface(m_linesearch_deployed, V, F);
        invalidateAdjointState();
    }

    void saveTargetSurface(const std::string &path) {
        target_surface_fitter.saveTargetSurface(path);
    }

    const TargetSurfaceFitter &get_target_surface_fitter() const { return target_surface_fitter; }
    void setHoldClosestPointsFixed(bool hold) {target_surface_fitter.setHoldClosestPointsFixed(hold, m_linesearch_deployed); }
    const Eigen::VectorXd getTargetJointsPosition() { return target_surface_fitter.joint_pos_tgt; }
    void setTargetJointsPosition(Eigen::VectorXd input_target_joint_pos) { target_surface_fitter.setTargetJointsPositions(input_target_joint_pos); objective.update(); invalidateAdjointState(); }
    void reflectTargetSurface(size_t ji) { target_surface_fitter.reflect(m_deployed, ji) ; }

    void setEquilibriumOptions(const NewtonOptimizerOptions &eopts) override {
        getDeployedOptimizer().options = eopts;
        getFlatOptimizer    ().options = eopts;
    }

    NewtonOptimizerOptions getEquilibriumOptions() const override {
        return m_flat_optimizer->options;
    }

    NewtonOptimizerOptions getDeploymentOptions() const {
        return m_deployed_optimizer->options;
    }

    NewtonOptimizer &getDeployedOptimizer() { return *m_deployed_optimizer; }
    NewtonOptimizer &getFlatOptimizer() {
        if (m_minAngleConstraint && m_allowFlatActuation && m_minAngleConstraint->inWorkingSet) {
            if (!m_flat_optimizer_actuated) throw std::runtime_error("Actuated flat linkage solver doesn't exist.");
            return *m_flat_optimizer_actuated;
        }
        return *m_flat_optimizer;
    }

    const LOMinAngleConstraint<Object> &getMinAngleConstraint() const override {
        if (m_minAngleConstraint) return *m_minAngleConstraint;
        throw std::runtime_error("No min angle constraint has been applied.");
    }

    LOMinAngleConstraint<Object> &getMinAngleConstraint() override {
        if (m_minAngleConstraint) return *m_minAngleConstraint;
        throw std::runtime_error("No min angle constraint has been applied.");
    }

    Real getEpsMinAngleConstraint() {
        if (m_minAngleConstraint) return m_minAngleConstraint->eps;
        throw std::runtime_error("No min angle constraint has been applied.");
    }

    // Write the full, dense Hessians of J and angle_constraint to a file.
    void dumpHessians(const std::string &hess_J_path, const std::string &hess_ac_path, Real fd_eps = 1e-5);


    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    Real gamma = 0.9;
    void setGamma(Real val) override { 
        gamma = val; 
        objective.get("ElasticEnergyFlat").setWeight(val / m_E0); 
        objective.get("ElasticEnergyDeployed").setWeight((1. - val) / m_E0); 
        invalidateAdjointState();
    }
    Real getGamma() const override { 
        return objective.get("ElasticEnergyFlat").getWeight() * m_E0; 
    }
    void set_E0(Real E0)  override { 
        gamma = getGamma(); 
        m_E0 = E0; 
        objective.get("ElasticEnergyFlat").setWeight(gamma / E0); 
        objective.get("ElasticEnergyDeployed").setWeight((1. - gamma) / E0); 
        invalidateAdjointState();
    }


private:
    void m_forceEquilibriumUpdate() override;
    // Return whether "params" are actually new...
    bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &params) override;

    // Update the closest point projections for each joint to the target surface.
    void m_updateClosestPoints() override { target_surface_fitter.updateClosestPoints(m_linesearch_deployed); }

    // Check if the minimum angle constraint is active and if so, change the closed
    // configuration's actuation angle to satisfy the constraint.
    void m_updateMinAngleConstraintActuation();

    // Update the adjoint state vectors "w" and "y"
    bool m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params, const OptEnergyType opt_eType=OptEnergyType::Full) override;

    // Extract the z coordinates of the joints
    Eigen::VectorXd m_apply_S_z(const Eigen::Ref<const Eigen::VectorXd> &x) {
        Eigen::VectorXd result(m_base.numJoints());
        for (size_t ji = 0; ji < m_base.numJoints(); ++ji)
            result[ji] = x[m_base.dofOffsetForJoint(ji) + 2];
        return result;
    }

    // Take a vector of per-joint z coordinates and place them in the
    // appropriate locations of the state vector.
    Eigen::VectorXd m_apply_S_z_transpose(const Eigen::Ref<const Eigen::VectorXd> &zcoords) {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(m_base.numDoF());
        for (size_t ji = 0; ji < m_base.numJoints(); ++ji)
            result[m_base.dofOffsetForJoint(ji) + 2] = zcoords[ji];
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Private member variables
    ////////////////////////////////////////////////////////////////////////////
    Eigen::VectorXd m_w_x, m_y, m_s_x; // adjoint state vectors
    Real m_w_lambda, m_delta_w_lambda; // last component of the adjoint state vector associated to the deployed linkage
    Eigen::VectorXd m_delta_w_x, m_delta_x2d, m_delta_x3d, m_delta_s_x, m_delta_y; // variations of adjoint/forward state from the last call to apply_hess (for debugging)
    Eigen::VectorXd m_delta_delta_x2d, m_delta_delta_x3d;   // second variations of forward state from last call to m_updateEquilibrium (for debugging)
    Eigen::VectorXd m_second_order_x3d, m_second_order_x2d; // second-order predictions of the linkage's equilibrium (for debugging)
    Eigen::VectorXd m_d3E_w;
    Eigen::VectorXd m_w_rhs, m_delta_w_rhs;
    // Real m_w_lambda, m_delta_w_lambda;
    Real m_alpha_tgt = 0.0;
    Real m_Linesearch_alpha_tgt = 0.0;
    Object<Real> &m_deployed; // m_flat is defined in the base class as m_base.
    Object<Real> m_linesearch_deployed; // m_linesearch_flat is defined in the base class as m_linesearch_base.
    std::unique_ptr<NewtonOptimizer> m_flat_optimizer, m_deployed_optimizer;

    std::unique_ptr<LOMinAngleConstraint<Object>> m_minAngleConstraint;
    std::unique_ptr<NewtonOptimizer> m_flat_optimizer_actuated;

    Object<ADReal> m_diff_linkage_flat, m_diff_linkage_deployed;

    size_t m_numFullParams;            // Total number of parameters, numParams only gives the rest quantities
    bool m_fixDeployedVars     = true; // whether we decide to fix the deployed vars as well as the flat vars
    bool m_optimizeTargetAngle = true; // whether we optimize for the opening angle as well
    bool m_allowFlatActuation  = false; // whether we allow the application of average-angle actuation to enforce the minimum angle constraint at the beginning of optimization.

};

#include "AverageAngleCShellOptimization.inl"
#endif /* end of include guard: AVERAGEANGLECSHELLOPTIMIZATION_HH */
