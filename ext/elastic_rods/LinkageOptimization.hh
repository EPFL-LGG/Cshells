////////////////////////////////////////////////////////////////////////////////
// LinkageOptimization.hh
////////////////////////////////////////////////////////////////////////////////
#ifndef LINKAGEOPTIMIZATION_HH
#define LINKAGEOPTIMIZATION_HH

#include "RodLinkage.hh"
#include <MeshFEM/Geometry.hh>
#include "compute_equilibrium.hh"
#include "LOMinAngleConstraint.hh"
#include "TargetSurfaceFitter.hh"
#include "RegularizationTerms.hh"
#include "DesignOptimizationTerms.hh"

enum class OptAlgorithm    : int { NEWTON_CG=0, BFGS=1 };
enum class OptEnergyType { Full, ElasticBase, ElasticDeployed, Target, Regularization, Smoothing };
enum class PredictionOrder : int { Zero = 0, One = 1, Two = 2};

template<template<typename> class Object>
struct LinkageOptimization {
    LinkageOptimization(Object<Real> &baseLinkage, const NewtonOptimizerOptions &eopts, Real E0, Real l0, Real rl0, Real rk0)
    : m_equilibrium_options(eopts), m_numParams(baseLinkage.numDesignParams()), m_E0(E0), m_l0(l0), m_rl0(rl0), m_rk0(rk0), m_base(baseLinkage), m_linesearch_base(baseLinkage) {}

    // Evaluate at a new set of parameters and commit this change to the flat/deployed linkages (which
    // are used as a starting point for solving the line search equilibrium)
    virtual void newPt(const Eigen::VectorXd &params) {
        std::cout << "newPt at dist " << (m_base.getDesignParameters() - params).norm() << std::endl;
        m_updateAdjointState(params); // update the adjoint state as well as the equilibrium, since we'll probably be needing gradients at this point.
        commitLinesearchLinkage();
    }

    size_t numParams() const { return m_numParams; }
    virtual size_t numFullParams() const { return m_numParams; } // To be overidden in case we have more parameters
    size_t numRestKappaVars() const {return m_linesearch_base.numRestKappaVars(); }
    size_t numRestLen() const { return m_linesearch_base.numSegments(); }
    virtual const Eigen::VectorXd &params() const { return m_base.getDesignParameters(); }

    // Objective function definition.
    Real J()        { return J(params()); }
    // Target fitting objective definition.
    Real J_target() { return J_target(params()); }
    // Gradient of the objective over the design parameters.
    virtual Eigen::VectorXd gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) = 0;
    Eigen::VectorXd gradp_J()        { return gradp_J(params()); }

    virtual Real J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType = OptEnergyType::Full) {
        std::cout << "eval at dist (from linkage optim) " << (m_base.getDesignParameters() - params).norm() << std::endl;
        std::cout << "eval at linesearch dist (from linkage optim) " << (m_linesearch_base.getDesignParameters() - params).norm() << std::endl;
        m_updateEquilibria(params);
        if (!m_equilibriumSolveSuccessful) return std::numeric_limits<Real>::max();

        objective.printReport();
        return objective.value(opt_eType);
    }

    Real J_target(const Eigen::Ref<const Eigen::VectorXd> &params) {
        m_updateEquilibria(params);
        return objective.value("TargetFitting");
    }

    Real defaultLengthBound() const { return 0.125 * m_base.getPerSegmentRestLength().minCoeff(); }

    Real J_regularization() { return objective.value("RestLengthMinimization"); }
    Real J_smoothing()      { return objective.value("RestCurvatureSmoothing"); }

    Real get_l0() const  { return m_l0;  }
    Real get_rl0() const { return m_rl0; }
    Real get_rk0() const { return m_rk0; }
    Real get_E0() const  { return m_E0;  }

    virtual void set_l0(Real l0)   { 
        beta = getBeta(); 
        m_l0 = l0; 
        objective.get("TargetFitting").setWeight(beta / (m_l0 * m_l0)); 
        invalidateAdjointState();
    }
    virtual void set_rl0(Real rl0)   { 
        Real rlWeight = getRestLengthMinimizationWeight(); 
        m_rl0 = rl0; 
        objective.get("RestLengthMinimization").setWeight(rlWeight / m_rl0); 
        invalidateAdjointState();
    }
    virtual void set_rk0(Real rk0)   { 
        Real rkWeight = getRestKappaSmoothingWeight(); 
        m_rk0 = rk0; 
        objective.get("RestCurvatureSmoothing").setWeight(rkWeight / (m_rk0 * m_rk0)); 
        invalidateAdjointState();
    }
    virtual void set_E0(Real E0)   { 
        m_E0 = E0; 
        invalidateAdjointState();
    }

    // Hessian matvec: H delta_p
    Eigen::VectorXd apply_hess_J               (const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, OptEnergyType opt_eType = OptEnergyType::Full) { return apply_hess(params, delta_p, 1.0, 0.0, 0.0, opt_eType); }
    Eigen::VectorXd apply_hess_c               (const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, OptEnergyType opt_eType = OptEnergyType::Full) { return apply_hess(params, delta_p, 0.0, 1.0, 0.0, opt_eType); }
    Eigen::VectorXd apply_hess_angle_constraint(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, OptEnergyType opt_eType = OptEnergyType::Full) { return apply_hess(params, delta_p, 0.0, 0.0, 1.0, opt_eType); }

    virtual Eigen::VectorXd apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p, Real coeff_J, Real coeff_c = 0.0, Real coeff_angle_constraint = 0.0, OptEnergyType opt_eType = OptEnergyType::Full) = 0;

    Object<Real> &getLinesearchBaseLinkage()     { return m_linesearch_base; }

    virtual void setLinkageInterleavingType(InterleavingType new_type) = 0;

    virtual Eigen::VectorXd getFullDesignParameters(){ return m_linesearch_base.getDesignParameters(); }
    virtual bool getOptimizeTargetAngle() const { return false; }

    bool use_restKappa() { return m_linesearch_base.getDesignParameterConfig().restKappa; }
    bool use_restLen()   { return m_linesearch_base.getDesignParameterConfig().restLen; }

    // Get the index of the joint whose orientation is constrained to pin
    // down the linkage's rigid motion.
    size_t getRigidMotionConstrainedJoint() const { return m_rm_constrained_joint; }

    virtual void commitLinesearchLinkage() = 0;

    virtual void setEquilibriumOptions(const NewtonOptimizerOptions &eopts) = 0;

    virtual NewtonOptimizerOptions getEquilibriumOptions() const = 0;

    // When the fitting weights change the adjoint state must be recompouted.
    // Let the user manually inform us of this change.
    void invalidateAdjointState() { m_adjointStateIsCurrent = false; }

    // For python bindings
    const Eigen::MatrixXd getTargetSurfaceVertices(){ return target_surface_fitter.getTargetSurfaceVertices(); }
    const Eigen::MatrixXi getTargetSurfaceFaces()   { return target_surface_fitter.getTargetSurfaceFaces(); }
    const Eigen::MatrixXd getTargetSurfaceNormals() { return target_surface_fitter.getTargetSurfaceNormals(); }

    // int Optimize(OptAlgorithm alg, 
    //         bool applyAngleConstraint, bool applyFlatnessConstraint, 
    //         size_t num_steps, Real trust_region_scale, Real optimality_tol, 
    //         std::function<void()> &update_viewer, double minRestLen);
    // // Write the full, dense Hessians of J and angle_constraint to a file.
    // void dumpHessians(const std::string &hess_J_path, const std::string &hess_ac_path, Real fd_eps = 1e-5);

    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    virtual void setGamma(Real val) = 0;
    void setBeta(Real val)                         { beta = val; objective.get("TargetFitting").setWeight(val / (m_l0 * m_l0)); }
    void setRestLengthMinimizationWeight(Real val) { objective.get("RestLengthMinimization").setWeight(val / m_rl0); }
    void setRestKappaSmoothingWeight(Real val)     { objective.get("RestCurvatureSmoothing").setWeight(val / (m_rk0 * m_rk0)); }
    
    virtual Real getGamma() const = 0;
    Real getBeta()                         const { return objective.get("TargetFitting").getWeight() * m_l0 * m_l0; }
    Real getRestLengthMinimizationWeight() const { return objective.get("RestLengthMinimization").getWeight() * m_rl0; }
    Real getRestKappaSmoothingWeight()     const { return objective.get("RestCurvatureSmoothing").getWeight() * m_rk0 * m_rk0; }

    Real restKappaSmoothness() const { return objective.value("RestCurvatureSmoothing"); }


    using DOO  = DesignOptimizationObjective<Object, OptEnergyType>;
    using TRec = typename DOO::TermRecord;
    DOO objective;
    TargetSurfaceFitter target_surface_fitter;
    // Configure how the equilibrium at a perturbed set of parameters is predicted (using 0th, 1st, or 2nd order Taylor expansion)
    PredictionOrder prediction_order = PredictionOrder::Two;

    virtual const LOMinAngleConstraint<Object> &getMinAngleConstraint() const {
        throw std::runtime_error("Min angle constraint is not implemented.");
    }
    virtual LOMinAngleConstraint<Object> &getMinAngleConstraint() {
        throw std::runtime_error("Min angle constraint is not implemented.");
    }
    virtual Real angle_constraint(const Eigen::Ref<const Eigen::VectorXd> & /*params */) {
        throw std::runtime_error("Minimum angle constraint is not implemented.");
    }
    virtual Real c(const Eigen::Ref<const Eigen::VectorXd> & /*params*/) {
        throw std::runtime_error("Flatness constraint is not implemented.");
    }
    virtual Eigen::VectorXd gradp_c (const Eigen::Ref<const Eigen::VectorXd> &/*params*/) {
        throw std::runtime_error("Flatness constraint is not implemented.");
    }
    virtual Eigen::VectorXd gradp_angle_constraint(const Eigen::Ref<const Eigen::VectorXd> &/*params*/) {
        throw std::runtime_error("Minimum angle constraint is not implemented.");
    }

    int optimize(OptAlgorithm alg,  
            size_t num_steps, Real trust_region_scale, Real optimality_tol, 
            std::function<void()> &update_viewer,
            double minRestLen = -1, 
            bool applyAngleConstraint = false, bool applyFlatnessConstraint = false);

    virtual ~LinkageOptimization() = default;

protected:
    virtual void m_forceEquilibriumUpdate() = 0;
    // Return whether "params" are actually new...
    virtual bool m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &params) = 0;

    // Update the closest point projections for each joint to the target surface.
    virtual void m_updateClosestPoints() = 0;

    // Update the adjoint state vectors "w" and "y"
    virtual bool m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params, const OptEnergyType opt_eType=OptEnergyType::Full) = 0;

    
    ////////////////////////////////////////////////////////////////////////////
    // Private member variables
    ////////////////////////////////////////////////////////////////////////////
    NewtonOptimizerOptions m_equilibrium_options;
    // Real m_w_lambda, m_delta_w_lambda;
    std::vector<size_t> m_rigidMotionFixedVars;
    size_t m_numParams;
    size_t m_rm_constrained_joint; // index of joint whose rigid motion is constrained.
    Real m_E0 = 1.0, m_l0 = 1.0, m_rl0 = 1.0, m_rk0 = 1.0;
    Real beta = 1.0;
    Real m_alpha_tgt = 0.0;
    Object<Real> &m_base;
    Object<Real> m_linesearch_base;

    bool m_adjointStateIsCurrent = false, m_autodiffLinkagesAreCurrent = false;
    bool m_equilibriumSolveSuccessful = false;

};

#include "LinkageOptimization.inl"
#endif /* end of include guard: LINKAGEOPTIMIZATION_HH */
