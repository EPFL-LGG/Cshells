#include "XShellOptimization.hh"

template<template<typename> class Object>
XShellOptimization<Object>::XShellOptimization(Object<Real> &flat, Object<Real> &deployed, const NewtonOptimizerOptions &eopts, std::unique_ptr<LOMinAngleConstraint<Object>> &&minAngleConstraint, int pinJoint, bool allowFlatActuation, bool optimizeTargetAngle, bool fixDeployedVars)
    : LinkageOptimization<Object>(flat, eopts, deployed.energy(), BBox<Point3D>(deployed.deformedPoints()).dimensions().norm(), deployed.totalRestLength(), deployed.averageAbsRestKappaVars() < 1.0e-10 ? 1.0 : deployed.averageAbsRestKappaVars()), 
      m_deployed(deployed), m_linesearch_deployed(deployed),
      m_minAngleConstraint(std::move(minAngleConstraint)),
      m_fixDeployedVars(fixDeployedVars),
      m_allowFlatActuation(allowFlatActuation)
{
    std::runtime_error mismatch("Linkage mismatch");
    if (m_numParams != deployed.numDesignParams())                                    throw mismatch;
    if ((deployed.getDesignParameters() - flat.getDesignParameters()).norm() > 1e-16) throw mismatch;
    m_alpha_tgt = deployed.getAverageJointAngle();
    m_Linesearch_alpha_tgt = deployed.getAverageJointAngle();
    m_numFullParams        = flat.numDesignParams() + int(m_optimizeTargetAngle);

    // Create the objective terms
    using OET = OptEnergyType;
    using EEO = ElasticEnergyObjective<Object>;
    using TSF = TargetFittingDOOT<Object>;
    using RLM = RegularizationTermDOOWrapper<Object, RestLengthMinimization>;
    using RCS = RegularizationTermDOOWrapper<Object, RestCurvatureSmoothing>;
    auto &tsf = target_surface_fitter;
    objective.add("ElasticEnergyFlat",      OET::ElasticBase,      std::make_shared<EEO>(m_linesearch_base),          gamma / m_E0);
    objective.add("ElasticEnergyDeployed",  OET::ElasticDeployed,  std::make_shared<EEO>(m_linesearch_deployed),      (1.0 - gamma) / m_E0);
    objective.add("TargetFitting",          OET::Target,           std::make_shared<TSF>(m_linesearch_deployed, tsf), beta / (m_l0 * m_l0));
    objective.add("RestLengthMinimization", OET::Regularization,   std::make_shared<RLM>(m_linesearch_deployed),      1.0 / m_rl0);
    objective.add("RestCurvatureSmoothing", OET::Smoothing,        std::make_shared<RCS>(m_linesearch_deployed),      10.0 / (m_rk0 * m_rk0));

    // Cannot use enveloppe theorem if we choose to optimize alpha
    setOptimizeTargetAngle(optimizeTargetAngle);

    // Unless the user specifies otherwise, use the current deployed linkage joint positions as the target
    target_surface_fitter.joint_pos_tgt = deployed.jointPositions();
    constructTargetSurface(2, 1);
    // Set to true for testing the derivatives
    target_surface_fitter.holdClosestPointsFixed = false;

    // Trade off between fitting to the individual joint targets and the target surface.
    target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(deployed, 0.01);

    // Constrain the position and orientation of the centermost joint to prevent global rigid motion.
    if (pinJoint != -1) {
        m_rm_constrained_joint = pinJoint;
        if (m_rm_constrained_joint >= flat.numJoints()) throw std::runtime_error("Manually specified pinJoint is out of bounds");
    }
    else {
        m_rm_constrained_joint = flat.centralJoint();
    }
    const size_t jdo = flat.dofOffsetForJoint(m_rm_constrained_joint);
    for (size_t i = 0; i < 6; ++i) m_rigidMotionFixedVars.push_back(jdo + i);

    m_flat_optimizer     = get_equilibrium_optimizer(m_linesearch_base,     TARGET_ANGLE_NONE,      m_rigidMotionFixedVars);
    if (m_fixDeployedVars) {
        m_deployed_optimizer = get_equilibrium_optimizer(m_linesearch_deployed, m_Linesearch_alpha_tgt, m_rigidMotionFixedVars);
    } else {
        std::vector<size_t> emptyFixedVars;
        m_deployed_optimizer = get_equilibrium_optimizer(m_linesearch_deployed, m_Linesearch_alpha_tgt, emptyFixedVars);
    }
    
    m_flat_optimizer    ->options = m_equilibrium_options;
    m_deployed_optimizer->options = m_equilibrium_options;

    // Ensure we start at an equilibrium (using the passed equilibrium solver options)
    m_forceEquilibriumUpdate();
    m_updateMinAngleConstraintActuation();

    commitLinesearchLinkage();
}

template<template<typename> class Object>
void XShellOptimization<Object>::m_forceEquilibriumUpdate() {
    // Update the flat/deployed equilibria
    m_equilibriumSolveSuccessful = true;
    try {
        if (m_equilibrium_options.verbose)
            std::cout << "Flat equilibrium solve" << std::endl;
        auto cr_flat = getFlatOptimizer().optimize();
        // A backtracking failure will happen if the gradient tolerance is set too low
        // and generally does not indicate a complete failure/bad estimate of the equilibrium.
        // We therefore accept such equilibria with a warning.
        // (We would prefer to reject saddle points, but the surface-attracted structures
        //  appear to //  sometimes have backtracking failures in saddle points
        //  close to reasonably stable equilibria...)
        bool acceptable_failed_flat_equilibrium = cr_flat.backtracking_failure; 
        if (!cr_flat.success && !acceptable_failed_flat_equilibrium) throw std::runtime_error("Flat equilibrium solve did not converge");
        if (acceptable_failed_flat_equilibrium) { std::cout << "WARNING: Flat equillibrium solve backtracking failure." << std::endl; }

        if (m_equilibrium_options.verbose)
            std::cout << "Deployed equilibrium solve" << std::endl;
        auto cr_deploy = getDeployedOptimizer().optimize();
        bool acceptable_failed_deployed_equilibrium = cr_deploy.backtracking_failure; 
        if (!cr_deploy.success && !acceptable_failed_deployed_equilibrium) throw std::runtime_error("Deployed equilibrium solve did not converge");
        if (acceptable_failed_deployed_equilibrium) { std::cout << "WARNING: Deployed equillibrium solve backtracking failure." << std::endl; }
    }
    catch (const std::runtime_error &e) {
        std::cout << "Equilibrium solve failed: " << e.what() << std::endl;
        m_equilibriumSolveSuccessful = false;
        return; // subsequent update_factorizations will fail if we caught a Tau runaway...
    }

    // We will be evaluating the Hessian/using the simplified gradient expressions:
    m_linesearch_base    .updateSourceFrame();
    m_linesearch_base    .updateRotationParametrizations();
    m_linesearch_deployed.updateSourceFrame();
    m_linesearch_deployed.updateRotationParametrizations();

    // Use the final equilibria's Hessians for sensitivity analysis, not the second-to-last iterates'
    try {
        getFlatOptimizer()    .update_factorizations();
        getDeployedOptimizer().update_factorizations();
    }
    catch (const std::runtime_error &e) {
        std::cout << "Equilibrium solve failed: " << e.what() << std::endl;
        m_equilibriumSolveSuccessful = false;
        return;
    }

    // The cached adjoint state is invalidated whenever the equilibrium is updated...
    m_adjointStateIsCurrent      = false;
    m_autodiffLinkagesAreCurrent = false;

    objective.update();
}

template<template<typename> class Object>
bool XShellOptimization<Object>::m_updateEquilibria(const Eigen::Ref<const Eigen::VectorXd> &newParams) {
    
    const size_t np = numParams(), nfp = numFullParams();
    std::runtime_error mismatch("Dimension mismatch for the design parameters in update equilibria");
    if (newParams.size() != int(nfp))     throw mismatch;
    
    // The linesearch linkage is already up to date
    if ((getLinesearchDesignParameters() - newParams).norm() < 1e-14) { return false; } 

    m_linesearch_deployed.set(m_deployed);
    m_linesearch_base.    set(m_base);

    const Eigen::VectorXd currParams = getFullDesignParameters();
    Eigen::VectorXd delta_p = newParams - currParams;

    if (delta_p.norm() == 0) { // returning to linesearch start; no prediction/Hessian factorization necessary
        std::cout << "Nothing has changed compared to last commit" << std::endl;
        m_Linesearch_alpha_tgt = m_alpha_tgt; 
        m_deployed_optimizer->get_problem().setLEQConstraintRHS(m_alpha_tgt); 
        m_forceEquilibriumUpdate();
        return true;
        // The following caching is not working for some reason...
        // TODO: debug
        // auto &opt_2D = getFlatOptimizer();
        // auto &opt_3D = getDeployedOptimizer();

        // if (!(opt_2D.solver.hasStashedFactorization() && opt_3D.solver.hasStashedFactorization()))
        //     throw std::runtime_error("Factorization was not stashed... was commitLinesearchLinkage() called?");

        // // Copy the stashed factorization into the current one (preserving the stash)
        // opt_2D.solver.swapStashedFactorization();
        // opt_3D.solver.swapStashedFactorization();
        // opt_2D.solver.stashFactorization();
        // opt_3D.solver.stashFactorization();

        // // The cached adjoint state is invalidated whenever the equilibrium is updated...
        // m_adjointStateIsCurrent      = false;
        // m_autodiffLinkagesAreCurrent = false;

        // m_updateClosestPoints();
        // 
        // return true;
    }

    // Apply the new design parameters and measure the energy with the 0^th order prediction
    // (i.e. at the current equilibrium).
    // We will only replace this equilibrium if the higher order predictions achieve a lower energy.
    if (m_optimizeTargetAngle) { 
        m_Linesearch_alpha_tgt = newParams[np]; 
        m_deployed_optimizer->get_problem().setLEQConstraintRHS(newParams[np]);
    }
    m_linesearch_deployed.setDesignParameters(newParams.head(np));
    m_linesearch_base    .setDesignParameters(newParams.head(np));
    Real bestEnergy3d = m_linesearch_deployed.energy(),
         bestEnergy2d = m_linesearch_base    .energy();

    Eigen::VectorXd curr_x3d = m_deployed.getDoFs(),
                    curr_x2d = m_base    .getDoFs();
    Eigen::VectorXd best_x3d = curr_x3d,
                    best_x2d = curr_x2d;

    if (prediction_order > PredictionOrder::Zero) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Predict equilibrium");
        // Return to using the Hessian for the last committed linkage
        // (i.e. for the equilibrium stored in m_flat and m_deployed).
        auto &opt_2D = getFlatOptimizer();
        auto &opt_3D = getDeployedOptimizer();
        if (!(opt_2D.solver.hasStashedFactorization() && opt_3D.solver.hasStashedFactorization()))
            throw std::runtime_error("Factorization was not stashed... was commitLinesearchLinkage() called?");
        opt_2D.solver.swapStashedFactorization();
        opt_3D.solver.swapStashedFactorization();

        {
            // Solve for equilibrium perturbation corresponding to delta_p:
            //      [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
            //      [a^T  0][delta lambda]   [        0        ]
            //                               \_________________/
            //                                        b
            // In case alpha_t is also optimized, this transforms into
            //       [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
            //       [a^T  0][delta lambda]   [  delta_alpha_t  ]
            //                                \_________________/
            //                                         b
            
            const size_t np = numParams(), nd = m_base.numDoF();
            VecX_T<Real> neg_deltap_padded(nd + np);
            neg_deltap_padded.setZero();
            neg_deltap_padded.tail(np) = -delta_p.head(np);

            // Computing -d2E/dxdp delta_p can skip the *-x and designParameter-* blocks
            HessianComputationMask mask_dxdp;
            mask_dxdp.dof_in      = false;
            mask_dxdp.designParameter_out = false;

            Eigen::VectorXd b_reduced_3D = opt_3D.removeFixedEntries(m_deployed.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd));
            if (m_optimizeTargetAngle){
                m_delta_x3d = opt_3D.extractFullSolution(opt_3D.kkt_solver(opt_3D.solver, b_reduced_3D, delta_p[np]));
            } else {
                m_delta_x3d = opt_3D.extractFullSolution(opt_3D.kkt_solver(opt_3D.solver, b_reduced_3D));
            }
            Eigen::VectorXd b_reduced = opt_2D.removeFixedEntries(m_base.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd));
            if (opt_2D.get_problem().hasLEQConstraint()) m_delta_x2d = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, b_reduced));
            else                                         m_delta_x2d = opt_2D.extractFullSolution(opt_2D.solver.solve(b_reduced));

            // Evaluate the energy at the 1st order-predicted equilibrium
            {
                auto first_order_x3d = (curr_x3d + m_delta_x3d).eval(),
                     first_order_x2d = (curr_x2d + m_delta_x2d).eval();
                m_linesearch_deployed.setDoFs(first_order_x3d);
                m_linesearch_base    .setDoFs(first_order_x2d);
                Real energy1stOrder3d = m_linesearch_deployed.energy(),
                     energy1stOrder2d = m_linesearch_base    .energy();
                if (energy1stOrder3d < bestEnergy3d) { std::cout << " used first order prediction, energy reduction " << bestEnergy3d - energy1stOrder3d << std::endl; bestEnergy3d = energy1stOrder3d; best_x3d = first_order_x3d; } else { m_linesearch_deployed.setDoFs(best_x3d); }
                if (energy1stOrder2d < bestEnergy2d) { std::cout << " used first order prediction, energy reduction " << bestEnergy2d - energy1stOrder2d << std::endl; bestEnergy2d = energy1stOrder2d; best_x2d = first_order_x2d; } else { m_linesearch_base    .setDoFs(best_x2d); }
            }
            
            if (prediction_order > PredictionOrder::One) {
                // TODO: also stash autodiff linkages for committed linkages?
                // Solve for perturbation of equilibrium perturbation corresponding to delta_p:
                //      [H_3D a][delta_p^T d2x/dp^2 delta_p] = -[d3E/dx3 delta_x + d3E/dx2dp delta_p 0][delta_x     ] + [-(d3E/dxdpdx delta_x + d3E/dxdpdp delta_p) delta_p] = -[d3E/dx3 delta_x + d3E/dx2dp delta_p    d3E/dxdpdx delta_x + d3E/dxdpdp delta_p    0][delta_x     ]
                //      [a^T  0][delta_p^T d2l/dp^2 delta_p]    [0                                   0][delta_lambda] + [                       0                          ]    [                0                                         0                       0][delta_p     ]
                //                                                                                                                                                                                                                                                   [delta_lambda]
                // Same system when optimizing for alpha too
                m_diff_linkage_deployed.set(m_deployed);
                m_diff_linkage_flat    .set(m_base);

                Eigen::VectorXd neg_d3E_delta_x3d, neg_d3E_delta_x2d;
                {
                    // inject design parameter perturbation.
                    VecX_T<ADReal> ad_p = currParams;
                    for (size_t i = 0; i < np; ++i) ad_p[i].derivatives()[0] = delta_p[i];
                    m_diff_linkage_deployed.setDesignParameters(ad_p);
                    m_diff_linkage_flat    .setDesignParameters(ad_p);

                    // inject equilibrium perturbation
                    VecX_T<ADReal> ad_x_3d = curr_x3d;
                    VecX_T<ADReal> ad_x_2d = curr_x2d;
                    for (int i = 0; i < ad_x_3d.size(); ++i) ad_x_3d[i].derivatives()[0] = m_delta_x3d[i];
                    for (int i = 0; i < ad_x_2d.size(); ++i) ad_x_2d[i].derivatives()[0] = m_delta_x2d[i];
                    m_diff_linkage_deployed.setDoFs(ad_x_3d);
                    m_diff_linkage_flat    .setDoFs(ad_x_2d);

                    VecX_T<Real> delta_edof_3d(nd + np);
                    VecX_T<Real> delta_edof_2d(nd + np);
                    delta_edof_3d.head(nd) = m_delta_x3d;
                    delta_edof_2d.head(nd) = m_delta_x2d;
                    delta_edof_3d.tail(np) = delta_p.head(np);
                    delta_edof_2d.tail(np) = delta_p.head(np);

                    neg_d3E_delta_x3d = -extractDirectionalDerivative(m_diff_linkage_deployed.applyHessianPerSegmentRestlen(delta_edof_3d)).head(nd);
                    neg_d3E_delta_x2d = -extractDirectionalDerivative(m_diff_linkage_flat    .applyHessianPerSegmentRestlen(delta_edof_2d)).head(nd);
                }

                m_delta_delta_x3d = opt_3D.extractFullSolution(opt_3D.kkt_solver(opt_3D.solver, opt_3D.removeFixedEntries(neg_d3E_delta_x3d)));
                b_reduced = opt_2D.removeFixedEntries(neg_d3E_delta_x2d);
                if (opt_2D.get_problem().hasLEQConstraint()) m_delta_delta_x2d = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, b_reduced));
                else                                         m_delta_delta_x2d = opt_2D.extractFullSolution(opt_2D.solver.solve(b_reduced));

                // Evaluate the energy at the 2nd order-predicted equilibrium, roll back to previous best if energy is higher.
                {
                    m_second_order_x3d = (curr_x3d + m_delta_x3d + 0.5 * m_delta_delta_x3d).eval(),
                    m_second_order_x2d = (curr_x2d + m_delta_x2d + 0.5 * m_delta_delta_x2d).eval();
                    m_linesearch_deployed.setDoFs(m_second_order_x3d);
                    m_linesearch_base    .setDoFs(m_second_order_x2d);
                    Real energy2ndOrder3d = m_linesearch_deployed.energy(),
                         energy2ndOrder2d = m_linesearch_base    .energy();
                    if (energy2ndOrder3d < bestEnergy3d) { std::cout << " used second order prediction, energy reduction " << bestEnergy3d - energy2ndOrder3d << std::endl; bestEnergy3d = energy2ndOrder3d; best_x3d = m_second_order_x3d;} else { m_linesearch_deployed.setDoFs(best_x3d); }
                    if (energy2ndOrder2d < bestEnergy2d) { std::cout << " used second order prediction, energy reduction " << bestEnergy2d - energy2ndOrder2d << std::endl; bestEnergy2d = energy2ndOrder2d; best_x2d = m_second_order_x2d;} else { m_linesearch_base    .setDoFs(best_x2d); }
                }
            }
        }

        // Return to using the primary factorization, storing the committed
        // linkages' factorizations back in the stash for later use.
        opt_2D.solver.swapStashedFactorization();
        opt_3D.solver.swapStashedFactorization();
    }

    m_forceEquilibriumUpdate();

    return true;
}

template<template<typename> class Object>
void XShellOptimization<Object>::m_updateMinAngleConstraintActuation() {
    if (!m_minAngleConstraint || !m_allowFlatActuation) return;

    getFlatOptimizer().optimize(); // We need to update the flat equilibrium to determine if the minimum angle constraint is in the working set

    // Add/remove the minimum angle constraint to the working set.
    if (m_minAngleConstraint->shouldRelease(m_linesearch_base, getFlatOptimizer())) {
        m_minAngleConstraint->inWorkingSet = false;
    }
    else if (m_minAngleConstraint->violated(m_linesearch_base)) {
        m_minAngleConstraint->inWorkingSet = true;
        Real alpha_bar_0 = m_linesearch_base.getAverageJointAngle();
        m_minAngleConstraint->actuationAngle = alpha_bar_0;
        // Construct the actuated flat equilibrium solver if it hasn't been.
        if (!m_flat_optimizer_actuated) {
            m_flat_optimizer_actuated = get_equilibrium_optimizer(m_linesearch_base, alpha_bar_0, m_rigidMotionFixedVars);
            m_flat_optimizer_actuated->options = m_equilibrium_options;
            m_flat_optimizer_actuated->optimize();
            m_linesearch_base.updateSourceFrame();
            m_linesearch_base.updateRotationParametrizations();
            m_flat_optimizer_actuated->update_factorizations();
        }
    }

    // If the minimum angle is in the working set, solve for the actuation angle such
    // that the bound is satisifed.
    m_minAngleConstraint->enforce(m_linesearch_base, getFlatOptimizer());
}


// Update the adjoint state vectors "w", "y", and "s"
template<template<typename> class Object>
bool XShellOptimization<Object>::m_updateAdjointState(const Eigen::Ref<const Eigen::VectorXd> &params, const OptEnergyType opt_eType) {
    const size_t nd = m_linesearch_deployed.numDoF(), nfp = numFullParams();
    std::runtime_error mismatch("Dimension mismatch for the design parameters in update adjoint state");
    if (params.size() != int(nfp))     throw mismatch;
    
    m_updateEquilibria(params);
    if (m_adjointStateIsCurrent) return false;

    // Solve the adjoint problems needed to efficiently evaluate the gradient.
    const auto &prob3D = getDeployedOptimizer().get_problem();

    // Note: if the Hessian modification failed (tau runaway), the adjoint state
    // solves will fail. To keep the solver from giving up entirely, we simply
    // set the adjoint state to 0 in these cases. Presumably this only happens
    // at bad iterates that will be discarded anyway.
    try {
        // Adjoint solve for the flatness constraint on the closed linkage:
        // H_2D y = 2 S_z^T S_z x_2D      or      [H_2D a][y_x     ] = [2 S_z^T S_z x_2D]
        //                                        [a^T  0][y_lambda]   [         0      ]
        // Depending on whether the closed linkage is actuated.
        {
            auto &opt = getFlatOptimizer();
            Eigen::VectorXd b_reduced = opt.removeFixedEntries(m_apply_S_z_transpose(2 * m_apply_S_z(m_linesearch_base.getDoFs())));
            if (opt.get_problem().hasLEQConstraint()) m_y = opt.extractFullSolution(opt.kkt_solver(opt.solver, b_reduced));
            else                                      m_y = opt.extractFullSolution(opt.solver.solve(b_reduced));
        }

        // Adjoint solve for the target fitting objective on the deployed linkage
        {
            if (!prob3D.hasLEQConstraint()) throw std::runtime_error("The deployed linkage must have a linear equality constraint applied!");
            auto &opt_3D = getDeployedOptimizer();
            Eigen::VectorXd grad_x;
            grad_x.setZero(nd);
            for (const auto &t : objective.terms) {
                if (t.term->getWeight() == 0.0) continue;
                if ((opt_eType == OptEnergyType::Full) || (opt_eType == t.type)) {
                    if (t.type == OptEnergyType::ElasticBase){
                        continue;
                    } else {
                        grad_x += t.term->grad_x();
                    }
                }
            }
            // Expose lambda in the KKT solver
            Eigen::VectorXd Hinv_b_reduced_3D;
            // Eigen::VectorXd b_reduced_3D = opt_3D.removeFixedEntries(target_surface_fitter.gradient(m_linesearch_deployed));
            Eigen::VectorXd b_reduced_3D = opt_3D.removeFixedEntries(grad_x);
            // Compute Hinv_b_reduced_3D
            opt_3D.solver.solve(b_reduced_3D, Hinv_b_reduced_3D);
            Real w_lambda = opt_3D.kkt_solver.lambda(Hinv_b_reduced_3D);
            // Eigen::VectorXd full_solution_3D = opt_3D.kkt_solver.solve(Hinv_b_reduced_3D);
            Eigen::VectorXd full_solution_3D = Hinv_b_reduced_3D - w_lambda * opt_3D.kkt_solver.Hinv_a;
            m_w_x = opt_3D.extractFullSolution(full_solution_3D);
            // m_w_x = target_surface_fitter.adjoint_solve(m_linesearch_deployed, getDeployedOptimizer());
            m_w_lambda = w_lambda;
        }

        // Adjoint solve for the minimum opening angle constraint
        // [H_2D a][s_x     ] = [d alpha_min / d_x]
        // [a^T  0][s_lambda]   [        0        ]
        if (m_minAngleConstraint) {
            auto &opt = getFlatOptimizer();
            Eigen::VectorXd Hinv_b_reduced;
            opt.solver.solve(opt.removeFixedEntries(m_minAngleConstraint->grad(m_linesearch_base)), Hinv_b_reduced);
            if (opt.get_problem().hasLEQConstraint()) m_s_x = opt.extractFullSolution(opt.kkt_solver.solve(Hinv_b_reduced));
            else                                      m_s_x = opt.extractFullSolution(Hinv_b_reduced);
        }
    }
    catch (...) {
        std::cout << "WARNING: Adjoint state solve failed" << std::endl;
        m_y.setZero();
        m_w_x.setZero();
        m_s_x.setZero();
        m_w_lambda = 0.;
    }

    m_adjointStateIsCurrent = true;

    return true;
}

// template<template<typename> class Object>
// Eigen::VectorXd XShellOptimization<Object>::gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType /* opt_eType */) {
//     m_updateAdjointState(params);
//     Eigen::VectorXd gradient = 
//         (      gamma  / m_E0) * m_linesearch_base    .grad_design_parameters(true)
//         + ((1.0 - gamma) / m_E0) * m_linesearch_deployed.grad_design_parameters(true)
//         + (beta / (m_l0 * m_l0)) * gradp_J_target(params);
    
//     bool use_restLen = m_linesearch_deployed.getDesignParameterConfig().restLen;
//     bool use_restKappa = m_linesearch_deployed.getDesignParameterConfig().restKappa;
//     if (use_restLen && use_restKappa) {
//         for (size_t rli = 0; rli < m_linesearch_deployed.numSegments(); ++rli) gradient[m_linesearch_deployed.numRestKappaVars() + rli] += (1.0 / m_rl0) * getRestLengthMinimizationWeight();
//     }
//     return gradient;
// }

template<template<typename> class Object>
Eigen::VectorXd XShellOptimization<Object>::gradp_J(const Eigen::Ref<const Eigen::VectorXd> &params, OptEnergyType opt_eType) {
    const size_t np = numParams(), nd = m_linesearch_deployed.numDoF(), nfp = numFullParams();
    std::runtime_error mismatch("Dimension mismatch for the design parameters in gradp_J");
    if (params.size() != int(nfp))     throw mismatch;

    // Then force equilibria and update adjoint state
    m_updateAdjointState(params, opt_eType);
    
    Eigen::VectorXd gradp_tot;
    gradp_tot.setZero(nfp);

    // Compute terms for the adjoint
    HessianComputationMask mask;
    mask.dof_out = false;
    mask.designParameter_in = false;

    Eigen::VectorXd w_padded(nd + np);
    w_padded.head(nd) = m_w_x;
    w_padded.tail(np).setZero();

    Eigen::VectorXd gradp_adj = - m_linesearch_deployed.applyHessianPerSegmentRestlen(w_padded, mask).tail(np);

    // Collect all the gradients
    gradp_tot.head(np) = objective.grad_p(opt_eType) + gradp_adj;

    // Add gradient with respect to the target angle
    if (m_optimizeTargetAngle) { gradp_tot[np] = m_w_lambda; }

    return gradp_tot;
}

template<template<typename> class Object>
Eigen::VectorXd XShellOptimization<Object>::gradp_J_target(const Eigen::Ref<const Eigen::VectorXd> &params) {
    const size_t np = numParams(), nd = m_linesearch_deployed.numDoF(), nfp = numFullParams();
    std::runtime_error mismatch("Dimension mismatch for the design parameters in gradp_J_target");
    if (params.size() != int(nfp))     throw mismatch;

    m_updateAdjointState(params, OptEnergyType::Target);

    Eigen::VectorXd gradp_tot;
    gradp_tot.setZero(nfp);

    HessianComputationMask mask;
    mask.dof_out = false;
    mask.designParameter_in = false;

    Eigen::VectorXd w_padded(nd + np);
    w_padded.head(nd) = m_w_x;
    w_padded.tail(np).setZero();
    return -m_linesearch_deployed.applyHessianPerSegmentRestlen(w_padded, mask).tail(np);
}

template<template<typename> class Object>
Eigen::VectorXd XShellOptimization<Object>::gradp_c(const Eigen::Ref<const Eigen::VectorXd> &params) {
    const size_t np = numParams(), nd = m_linesearch_base.numDoF(), nfp = numFullParams();
    std::runtime_error mismatch("Dimension mismatch for the design parameters in gradp_c");
    if (params.size() != int(nfp))     throw mismatch;
    m_updateAdjointState(params);

    Eigen::VectorXd gradp_tot;
    if (m_optimizeTargetAngle) gradp_tot.setZero(np + 1);
    else                       gradp_tot.setZero(np    );

    HessianComputationMask mask;
    mask.dof_out = false;
    mask.designParameter_in = false;

    Eigen::VectorXd y_padded(nd + np);
    y_padded.head(m_y.size()) = m_y;
    y_padded.tail(numParams()).setZero();

    gradp_tot.head(np) = - m_linesearch_base.applyHessianPerSegmentRestlen(y_padded, mask).tail(np);
    return gradp_tot;
}

template<template<typename> class Object>
Eigen::VectorXd XShellOptimization<Object>::gradp_angle_constraint(const Eigen::Ref<const Eigen::VectorXd> &params) {
    if (!m_minAngleConstraint) throw std::runtime_error("No minimum angle constraint is applied.");
    const size_t np = numParams(), nd = m_linesearch_base.numDoF(), nfp = numFullParams();
    std::runtime_error mismatch("Dimension mismatch for the design parameters in gradp_c");
    if (params.size() != int(nfp))     throw mismatch;
    m_updateAdjointState(params);

    Eigen::VectorXd gradp_tot;
    gradp_tot.setZero(nfp);

    HessianComputationMask mask;
    mask.dof_out = false;
    mask.designParameter_in = false;

    Eigen::VectorXd s_padded(nd + np);
    s_padded.head(m_s_x.size()) = m_s_x;
    s_padded.tail(numParams()).setZero();
    gradp_tot.head(np) = - m_linesearch_base.applyHessianPerSegmentRestlen(s_padded, mask).tail(np);
    return gradp_tot;
}

template<template<typename> class Object>
Eigen::VectorXd XShellOptimization<Object>::pushforward(const Eigen::Ref<const Eigen::VectorXd> &params, const Eigen::Ref<const Eigen::VectorXd> &delta_p) {

    Eigen::VectorXd delta_x2d,
                    delta_x3d;
    const size_t np = numParams(), nd = m_linesearch_base.numDoF();

    m_updateAdjointState(params);
    std::cout << "Done updating adjoint in pushforward" << std::endl;

    // Use current factorization
    auto &opt_2D = getFlatOptimizer();
    auto &opt_3D = getDeployedOptimizer();

    {
        // Solve for equilibrium perturbation corresponding to delta_p:
        //      [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
        //      [a^T  0][delta lambda]   [        0        ]
        //                               \_________________/
        //                                        b
        // In case alpha_t is also optimized, this transforms into
        //      [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
        //      [a^T  0][delta lambda]   [  delta_alpha_t  ]
        //                               \_________________/
        //                                        b
        
        VecX_T<Real> neg_deltap_padded(nd + np);
        neg_deltap_padded.setZero();
        neg_deltap_padded.tail(np) = -delta_p.head(np);

        // Computing -d2E/dxdp delta_p can skip the *-x and designParameter-* blocks
        HessianComputationMask mask_dxdp;
        mask_dxdp.dof_in      = false;
        mask_dxdp.designParameter_out = false;

        Eigen::VectorXd b_reduced_3D = opt_3D.removeFixedEntries(m_linesearch_deployed.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd));
        if (m_optimizeTargetAngle){
            delta_x3d = opt_3D.extractFullSolution(opt_3D.kkt_solver(opt_3D.solver, b_reduced_3D, delta_p[np]));
        } else {
            delta_x3d = opt_3D.extractFullSolution(opt_3D.kkt_solver(opt_3D.solver, b_reduced_3D));
        }
        Eigen::VectorXd b_reduced_2D = opt_2D.removeFixedEntries(m_linesearch_base.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd));
        if (opt_2D.get_problem().hasLEQConstraint()) delta_x2d = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, b_reduced_2D));
        else                                         delta_x2d = opt_2D.extractFullSolution(opt_2D.solver.solve(b_reduced_2D));
    }

    Eigen::VectorXd delta_x(2 * nd);
    delta_x.head(nd) = delta_x2d;
    delta_x.tail(nd) = delta_x3d;

    return delta_x;
}

template<template<typename> class Object>
Eigen::VectorXd XShellOptimization<Object>::apply_hess(const Eigen::Ref<const Eigen::VectorXd> &params,
                                                const Eigen::Ref<const Eigen::VectorXd> &delta_p,
                                                Real coeff_J, Real coeff_c, Real coeff_angle_constraint, OptEnergyType opt_eType) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("apply_hess_J");
    BENCHMARK_START_TIMER_SECTION("Preamble");
    const size_t np = numParams(), nd = m_linesearch_base.numDoF(), nfp = numFullParams();
    if (params.size()  != int(nfp))     throw std::runtime_error("Incorrect parameter vector size");
    if (delta_p.size() != int(nfp))     throw std::runtime_error("Incorrect delta parameter vector size");

    m_updateAdjointState(params, opt_eType);

    if (!m_autodiffLinkagesAreCurrent) {
        BENCHMARK_SCOPED_TIMER_SECTION timer2("Update autodiff linkages");
        m_diff_linkage_deployed.set(m_linesearch_deployed);
        m_diff_linkage_flat    .set(m_linesearch_base);
        m_autodiffLinkagesAreCurrent = true;
    }

    auto &opt_3D  = getDeployedOptimizer();
    const auto &prob3D = opt_3D.get_problem();
    if (!prob3D.hasLEQConstraint()) throw std::runtime_error("The deployed linkage must have a linear equality constraint applied!");
    auto &H_3D = opt_3D.solver;

    BENCHMARK_STOP_TIMER_SECTION("Preamble");

    VecX_T<Real> neg_deltap_padded(nd + np);
    neg_deltap_padded.head(nd).setZero();
    neg_deltap_padded.tail(np) = - delta_p.head(np);

    // Computing -d2E/dxdp delta_p can skip the *-x and designParameter-* blocks
    HessianComputationMask mask_dxdp, mask_dxpdx;
    mask_dxdp.dof_in      = false;
    mask_dxdp.designParameter_out = false;
    mask_dxpdx.designParameter_in = false;

    // VecX_T<Real> delta_dJ_dx3dp;

    // Note: if the Hessian modification failed (tau runaway), the delta forward/adjoint state
    // solves will fail. To keep the solver from giving up entirely, we simply
    // set the failed quantities to 0 in these cases. Presumably this only happens
    // at bad iterates that will be discarded anyway.
    // Solve for closed state perturbation
    try {
        // H_2D delta_x = [-d2E/dxdp delta_p]   or   [H_2D a][delta_x     ] = [-d2E/dxdp delta_p]
        //                \_________________/        [a^T  0][delta_lambda]   [        0        ]
        //                         b                                          \_________________/
        //                                                                             b
        // Depending on whether the closed linkage is actuated.
        BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta x2d");
        auto &opt_2D = getFlatOptimizer();

        Eigen::VectorXd b_reduced = opt_2D.removeFixedEntries(m_linesearch_base.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd));
        if (opt_2D.get_problem().hasLEQConstraint()) m_delta_x2d = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, b_reduced));
        else                                         m_delta_x2d = opt_2D.extractFullSolution(opt_2D.solver.solve(b_reduced));

    }
    catch (...) { 
        m_delta_x2d.setZero(); 
    }

    VecX_T<Real> d3E_s, d3E_y;
    VecX_T<Real> delta_grad_xp;
    delta_grad_xp.setZero(nd + np);
    try {
        // Solve for deployed state perturbation
        // [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
        // [a^T  0][delta lambda]   [        0        ]
        //                          \_________________/
        //                                   b
        //
        // In case alpha_t is also optimized, this transforms into
        // [H_3D a][delta x     ] = [-d2E/dxdp delta_p]
        // [a^T  0][delta lambda]   [  delta_alpha_t  ]
        //                          \_________________/
        //                                   b
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta x3d");
            VecX_T<Real> b = m_linesearch_deployed.applyHessianPerSegmentRestlen(neg_deltap_padded, mask_dxdp).head(nd);
            if (m_optimizeTargetAngle){
                m_delta_x3d = opt_3D.extractFullSolution(opt_3D.kkt_solver(H_3D, opt_3D.removeFixedEntries(b), delta_p[np]));
            } else {
                m_delta_x3d = opt_3D.extractFullSolution(opt_3D.kkt_solver(H_3D, opt_3D.removeFixedEntries(b)));
            }
        }

        VecX_T<Real> delta_x2dp(nd + np);
        delta_x2dp.head(nd) = m_delta_x2d;
        delta_x2dp.tail(np) = delta_p.head(np);

        VecX_T<Real> delta_x3dp(nd + np);
        delta_x3dp.head(nd) = m_delta_x3d;
        delta_x3dp.tail(np) = delta_p.head(np);

        // Solve for deployed adjoint state perturbation
        BENCHMARK_START_TIMER_SECTION("getDoFs and inject state");
        bool need_2d_autodiff = (coeff_c != 0.0) || (coeff_angle_constraint != 0.0);

        auto ad_xp_3d = m_diff_linkage_deployed.getExtendedDoFsPSRL();
        auto ad_xp_2d = m_diff_linkage_flat    .getExtendedDoFsPSRL();

        auto inject_delta_state_3d = [&](VecX_T<Real> delta) {
            for (int i = 0; i < ad_xp_3d.size(); ++i) ad_xp_3d[i].derivatives()[0] = delta[i];
            m_diff_linkage_deployed.setExtendedDoFsPSRL(ad_xp_3d);
        };

        auto inject_delta_state_2d = [&](VecX_T<Real> delta) {
            for (int i = 0; i < ad_xp_2d.size(); ++i) ad_xp_2d[i].derivatives()[0] = delta[i];
            m_diff_linkage_flat.setExtendedDoFsPSRL(ad_xp_2d);
        };
        inject_delta_state_3d(delta_x3dp);
        if (need_2d_autodiff)
            inject_delta_state_2d(delta_x2dp);
        BENCHMARK_STOP_TIMER_SECTION("getDoFs and inject state");

        // [H_3D a][delta w_x     ] = [W delta_x + W_surf (I - dp_dx) delta_x ] - [d3E/dx dx dx delta_x + d3E/dx dx dp delta_p] w
        // [a^T  0][delta w_lambda]   [               0                       ]   [                     0                     ]
        //                            \________________________________________________________________________________________/
        //                                                                      b
        
        if (coeff_J != 0.0) {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta w x");
            BENCHMARK_START_TIMER_SECTION("Hw");
            VecX_T<ADReal> w_padded(nd + np);
            w_padded.head(nd) = m_w_x;
            w_padded.tail(np).setZero();
            // Note: we need the "p" rows of d3E_w for evaluating the full Hessian matvec expressions below...
            m_d3E_w = extractDirectionalDerivative(m_diff_linkage_deployed.applyHessianPerSegmentRestlen(w_padded, mask_dxpdx));
            BENCHMARK_STOP_TIMER_SECTION("Hw");

            BENCHMARK_START_TIMER_SECTION("KKT_solve");
            // Sensitivity of adjoint solve for the target fitting objective on the deployed linkage
            if (!prob3D.hasLEQConstraint()) throw std::runtime_error("The deployed linkage must have a linear equality constraint applied!");
            auto &opt_3D = getDeployedOptimizer();
            
            for (const auto &t : objective.terms) {
                if (t.term->getWeight() == 0.0) continue;
                if ((opt_eType == OptEnergyType::Full) || (opt_eType == t.type)) {
                    if (t.type == OptEnergyType::ElasticBase){
                        continue;
                    } else {
                        delta_grad_xp += t.term->delta_grad(delta_x3dp, m_diff_linkage_deployed);
                    }
                }
            }
            // Expose delta_w_lambda in the KKT solver
            Eigen::VectorXd Hinv_b_reduced_3D;
            Eigen::VectorXd b_reduced_3D = opt_3D.removeFixedEntries((delta_grad_xp.head(nd) - m_d3E_w.head(nd)).eval());
            // Compute Hinv_b_reduced_3D
            opt_3D.solver.solve(b_reduced_3D, Hinv_b_reduced_3D);
            Real delta_w_lambda = opt_3D.kkt_solver.lambda(Hinv_b_reduced_3D);
            // Eigen::VectorXd full_solution_3D = opt_3D.kkt_solver.solve(Hinv_b_reduced_3D);
            Eigen::VectorXd full_solution_3D = Hinv_b_reduced_3D - delta_w_lambda * opt_3D.kkt_solver.Hinv_a;
            m_delta_w_x = opt_3D.extractFullSolution(full_solution_3D);
            m_delta_w_lambda = delta_w_lambda;
            // m_delta_w_x = opt_3D.extractFullSolution(opt_3D.kkt_solver(opt_3D.solver, opt_3D.removeFixedEntries(b)));
            // m_delta_w_x = target_surface_fitter.delta_adjoint_solve(m_linesearch_deployed, opt_3D, m_delta_x3d, m_d3E_w);
            BENCHMARK_STOP_TIMER_SECTION("KKT_solve");
        }

        // [H_2D a][delta s_x     ] = [d^2 alpha_min /d_x^2 delta_x] - [d3E/dx dx dx delta_x + d3E/dx dx dp delta_p] s
        // [a^T  0][delta s_lambda]   [               0            ]   [                     0                     ]
        //                            \______________________________________________________________________________/
        //                                                               b
        if (m_minAngleConstraint && (coeff_angle_constraint != 0.0)) {
            auto &opt_2D = getFlatOptimizer();

            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta s x");
            BENCHMARK_START_TIMER_SECTION("Hs");
            VecX_T<ADReal> s_padded(nd + np);
            s_padded.head(nd) = m_s_x;
            s_padded.tail(np).setZero();
            // Note: we need the "p" rows of d3E_s for evaluating the full angle constraint Hessian matvec expression below...
            d3E_s = extractDirectionalDerivative(m_diff_linkage_flat.applyHessianPerSegmentRestlen(s_padded, mask_dxpdx));
            BENCHMARK_STOP_TIMER_SECTION("Hs");

            BENCHMARK_START_TIMER_SECTION("KKT_solve");
            auto b = (m_minAngleConstraint->delta_grad(m_linesearch_base, m_delta_x2d) - d3E_s.head(nd)).eval();
            if (opt_2D.get_problem().hasLEQConstraint()) m_delta_s_x = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, opt_2D.removeFixedEntries(b)));
            else                                         m_delta_s_x = opt_2D.extractFullSolution(opt_2D.solver.solve(opt_2D.removeFixedEntries(b)));
            BENCHMARK_STOP_TIMER_SECTION("KKT_solve");
        }

        // H_2D delta y = 2 S_z^T S_z delta x_2D - delta H_2D y      or      [H_2D a][delta y_x     ] = [2 S_z^T S_z delta x_2D] - [delta H_2D] y
        //                                                                   [a^T  0][delta y_lambda]   [         0            ]   [          ]
        // depending on whether the closed linkage is actuated,
        // where delta H_2D = d3E/dx dx dx delta_x + d3E/dx dx dp delta_p.
        if (coeff_c != 0.0) {
            auto &opt_2D = getFlatOptimizer();
            BENCHMARK_SCOPED_TIMER_SECTION timer2("solve delta y");
            BENCHMARK_START_TIMER_SECTION("Hy");
            VecX_T<ADReal> y_padded(nd + np);
            y_padded.head(nd) = m_y;
            y_padded.tail(np).setZero();
            // Note: we need the "p" rows of d3E_y for evaluating the full Hessian matvec expressions below...
            d3E_y = extractDirectionalDerivative(m_diff_linkage_flat.applyHessianPerSegmentRestlen(y_padded, mask_dxpdx));

            BENCHMARK_STOP_TIMER_SECTION("Hy");

            BENCHMARK_START_TIMER_SECTION("KKT_solve");
            auto b = (m_apply_S_z_transpose(2 * m_apply_S_z(m_delta_x2d)) - d3E_y.head(nd)).eval();
            if (opt_2D.get_problem().hasLEQConstraint()) m_delta_y = opt_2D.extractFullSolution(opt_2D.kkt_solver(opt_2D.solver, opt_2D.removeFixedEntries(b)));
            else                                         m_delta_y = opt_2D.extractFullSolution(opt_2D.solver.solve(opt_2D.removeFixedEntries(b)));
            BENCHMARK_STOP_TIMER_SECTION("KKT_solve");
        }
    }
    catch (...) {
        m_delta_x3d      = VecX_T<Real>::Zero(nd     );
        m_delta_w_x      = VecX_T<Real>::Zero(nd     );
        m_delta_w_lambda = 0.;
        m_delta_s_x      = VecX_T<Real>::Zero(nd     );
        m_delta_y        = VecX_T<Real>::Zero(nd     );
        m_d3E_w          = VecX_T<Real>::Zero(nd + np);
        d3E_s            = VecX_T<Real>::Zero(nd + np);

        delta_grad_xp.setZero(nd + np);
    }



    VecX_T<Real> result;
    result.setZero(nfp);

    // Accumulate the J hessian matvec
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer3("evaluate hessian matvec");

        if (coeff_J != 0.0) {

            if (objective.terms.empty()) throw std::runtime_error("no terms present");
            VecX_T<Real> delta_edofs_2d(nd + np);
            delta_edofs_2d.head(nd) = m_delta_x2d;
            delta_edofs_2d.tail(np) = delta_p.head(np);

            VecX_T<Real> delta_edofs_adj3d(nd + np);
            delta_edofs_adj3d.head(nd) = m_delta_w_x;
            delta_edofs_adj3d.tail(np).setZero();

            HessianComputationMask mask;
            mask.dof_out = false;
            mask.designParameter_in = false;
            
            // First accumulate the delta grad we previously computed
            result.head(np) += delta_grad_xp.tail(np);

            // Then add the remaining terms 
            
            result.head(np) += - m_linesearch_deployed.applyHessianPerSegmentRestlen(delta_edofs_adj3d, mask).tail(np);
            result.head(np) += - m_d3E_w.tail(np);

            for (const auto &t : objective.terms) {
                if (t.term->getWeight() == 0.0) continue;
                if ((opt_eType == OptEnergyType::Full) || (opt_eType == t.type)) {
                    if (t.type == OptEnergyType::ElasticBase){
                        result.head(np) += t.term->delta_grad(delta_edofs_2d, m_diff_linkage_flat).tail(np);
                    }
                }
            }
            // Handle the last component
            if (m_optimizeTargetAngle) { result[np] = m_delta_w_lambda; }
            result *= coeff_J;
        }
        if (coeff_c != 0) {
            HessianComputationMask mask;
            mask.dof_out = false;
            VecX_T<Real> delta_edofs(nd + np);
            delta_edofs.head(nd) = -m_delta_y;
            delta_edofs.tail(np).setZero();
            result.head(np) += coeff_c * (m_linesearch_base.applyHessianPerSegmentRestlen(delta_edofs, mask).tail(np) - d3E_y.tail(np));
        }
        if (coeff_angle_constraint != 0) {
            HessianComputationMask mask;
            mask.dof_out = false;
            VecX_T<Real> delta_edofs(nd + np);
            delta_edofs.head(nd) = -m_delta_s_x;
            delta_edofs.tail(np).setZero();
            result.head(np) += coeff_angle_constraint * (m_linesearch_base.applyHessianPerSegmentRestlen(delta_edofs, mask).tail(np) - d3E_s.tail(np));
        }
    }

    return result;
}

template<template<typename> class Object>
void XShellOptimization<Object>::dumpHessians(const std::string &hess_J_path, const std::string &hess_ac_path, Real fd_eps) {
    // Eigen::MatrixXd hess_J (m_numParams, m_numParams),
    //                 hess_ac(m_numParams, m_numParams);

    auto curr_params = getFullDesignParameters();
    auto grad_J = gradp_J(curr_params);

    // std::cout << "Evaluating full Hessians" << std::endl;
    // for (size_t p = 0; p < m_numParams; ++p) {
    //     hess_J .col(p) = apply_hess_J (curr_params, Eigen::VectorXd::Unit(m_numParams, p));
    //     hess_ac.col(p) = apply_hess_angle_constraint(curr_params, Eigen::VectorXd::Unit(m_numParams, p));
    // }

    // std::ofstream hess_J_file(hess_J_path);
    // hess_J_file.precision(19);
    // hess_J_file << hess_J << std::endl;

    // std::ofstream hess_ac_file(hess_ac_path);
    // hess_ac_file.precision(19);
    // hess_ac_file << hess_ac << std::endl;

    size_t nperturbs = 3;
    Eigen::VectorXd relerror_fd_diff_grad_p_J(nperturbs),
                    relerror_delta_Hw(nperturbs),
                    relerror_delta_w(nperturbs),
                    relerror_delta_w_rhs(nperturbs),
                    relerror_delta_x(nperturbs),
                    relerror_delta_J(nperturbs);
    Eigen::VectorXd matvec_relerror_fd_diff_grad_p_J(nperturbs);
    Eigen::VectorXd grad_J_relerror(nperturbs);

    auto w = m_w_x;
    auto H = m_linesearch_deployed.hessian();
    auto Hw = H.apply(w);
    // Real w_lambda = m_w_lambda;

    for (size_t i = 0; i < nperturbs; ++i) {
        Eigen::VectorXd perturb = Eigen::VectorXd::Random(m_numParams + int(m_optimizeTargetAngle));

        apply_hess_J(curr_params, perturb);
        auto delta_w = m_delta_w_x;
        auto H_delta_w = m_linesearch_deployed.applyHessian(delta_w);

        Real Jplus = J(curr_params + fd_eps * perturb);
        auto gradp_J_plus = gradp_J(curr_params + fd_eps * perturb);
        auto w_plus = m_w_x;
        auto x_plus = m_linesearch_deployed.getDoFs();
        auto Hw_plus = m_linesearch_deployed.applyHessian(w);
        auto H_plus_w_plus = m_linesearch_deployed.applyHessian(w_plus);
        auto w_rhs_plus = m_w_rhs;
        auto H_plus = m_linesearch_deployed.hessian();
        // Real w_lambda_plus = m_w_lambda;

        {
            Eigen::VectorXd v = Eigen::VectorXd::Random(m_linesearch_deployed.numDoF());
            auto my_H = m_linesearch_deployed.hessianSparsityPattern(false);
            m_linesearch_deployed.hessian(my_H);
            auto matvec_one = my_H.apply(v);
            auto matvec_two = m_linesearch_deployed.applyHessian(v);
            std::cout << "matvec error: " << (matvec_one - matvec_two).norm() / matvec_one.norm() << std::endl;

            v = w_plus;
            matvec_one = my_H.apply(v);
            matvec_two = m_linesearch_deployed.applyHessian(v);
            auto matvec_three = m_linesearch_deployed.applyHessian(v);
            std::cout << "w_plus matec error: " << (matvec_one - matvec_two).norm() / matvec_one.norm() << std::endl;

            std::cout << "w_plus.norm(): " << w_plus.norm() << std::endl;
            std::cout << "H w_plus.norm() 1: " << matvec_one.norm() << std::endl;
            std::cout << "H w_plus.norm() 2: " << matvec_two.norm() << std::endl;
            std::cout << "H w_plus.norm() 3: " << matvec_three.norm() << std::endl;

            std::ofstream out_file_w("w_plus.txt");
            out_file_w.precision(16);
            out_file_w << w_plus << std::endl;

            std::ofstream out_file("matvec_one.txt");
            out_file.precision(16);
            out_file << matvec_one << std::endl;

            std::ofstream out_file2("matvec_two.txt");
            out_file2.precision(16);
            out_file2 << matvec_two << std::endl;

            std::ofstream out_file3("matvec_three.txt");
            out_file3.precision(16);
            out_file3 << matvec_two << std::endl;
        }

        Real Jminus = J(curr_params - fd_eps * perturb);
        auto gradp_J_minus = gradp_J(curr_params - fd_eps * perturb);
        auto w_minus = m_w_x;
        auto x_minus = m_linesearch_deployed.getDoFs();
        auto Hw_minus = m_linesearch_deployed.applyHessian(w);
        auto H_minus_w_minus = m_linesearch_deployed.applyHessian(w_minus);
        auto w_rhs_minus = m_w_rhs;
        // Real w_lambda_minus = m_w_lambda;

        Real fd_J = (Jplus - Jminus) / (2 * fd_eps);
        relerror_delta_J[i] = std::abs((grad_J.dot(perturb) - fd_J) / fd_J);

        Eigen::VectorXd fd_diff_grad_p_J = (gradp_J_plus - gradp_J_minus) / (2 * fd_eps);
        Eigen::VectorXd fd_delta_w = (w_plus - w_minus) / (2 * fd_eps);
        Eigen::VectorXd fd_delta_x = (x_plus - x_minus) / (2 * fd_eps);
        Eigen::VectorXd fd_delta_Hw = (Hw_plus - Hw_minus) / (2 * fd_eps);
        // Eigen::VectorXd fd_delta_w_rhs = (w_rhs_plus - w_rhs_minus) / (2 * fd_eps);
        // Real fd_delta_lambda = (w_lambda_plus - w_lambda_minus) / (2 * fd_eps);

#if 0
        auto fd_H_delta_w = H.apply(fd_delta_w);
        Eigen::VectorXd soln_error = ((Hw + fd_eps * fd_delta_Hw + fd_eps * fd_H_delta_w) - w_rhs_plus) + opt.extractFullSolution(opt.kkt_solver.a * (w_lambda + fd_eps * fd_delta_lambda));
        std::cout << "||Hw + delta_H w + H delta w + a (lambda + delta lambda) - b||: " << opt.removeFixedEntries(soln_error).norm() << std::endl;
        std::cout << soln_error.head(8).transpose() << std::endl;
        std::cout << soln_error.segment<8>(m_linesearch_deployed.dofOffsetForJoint(0)).transpose() << std::endl;

        Eigen::VectorXd soln_error2 = (H_plus.apply(w_plus) + w_lambda_plus * opt.extractFullSolution(opt.kkt_solver.a) - w_rhs_plus);
        std::cout << "||Hw_plus + a lambda_plus - b_plus||: " << opt.removeFixedEntries(soln_error2).norm() << std::endl;

        Eigen::VectorXd soln_error3 = ((Hw + fd_eps * fd_delta_Hw + fd_eps * H.apply(delta_w)) - w_rhs_plus) + opt.extractFullSolution(opt.kkt_solver.a * (w_lambda + fd_eps * fd_delta_lambda));
        std::cout << "||Hw + delta_H w + H delta w + a (lambda + delta lambda) - b||: " << opt.removeFixedEntries(soln_error3).norm() << std::endl;

        Eigen::VectorXd soln_error4 = H_plus_w_plus - w_rhs_plus + opt.extractFullSolution(opt.kkt_solver.a * w_lambda_plus);
        std::cout << "||H_plus w_plus + a (lambda + delta lambda) - b||: " << opt.removeFixedEntries(soln_error4).norm() << std::endl;
#endif

        // Eigen::VectorXd fd_diff_grad_p_ac = (gradp_angle_constraint(curr_params + fd_eps * perturb) - gradp_angle_constraint(curr_params - fd_eps * perturb)) / (2 * fd_eps);
        // relerror_fd_diff_grad_p_ac[i] = (hess_ac * perturb - fd_diff_grad_p_ac).norm() / fd_diff_grad_p_ac.norm();

        // relerror_fd_diff_grad_p_J[i] = (hess_J  * perturb - fd_diff_grad_p_J ).norm() / fd_diff_grad_p_J .norm();

        matvec_relerror_fd_diff_grad_p_J[i] = (apply_hess_J(curr_params, perturb) - fd_diff_grad_p_J ).norm() / fd_diff_grad_p_J .norm();
        relerror_delta_x[i] = (m_delta_x3d - fd_delta_x).norm() / fd_delta_x.norm();
        relerror_delta_w[i] = (m_delta_w_x - fd_delta_w).norm() / fd_delta_w.norm();
        relerror_delta_Hw[i] = (m_d3E_w.head(w.size()) - fd_delta_Hw).norm() / fd_delta_Hw.norm();
        // relerror_delta_w_rhs[i] = (m_delta_w_rhs - fd_delta_w_rhs).norm() / fd_delta_w_rhs.norm();

        // // Eigen::VectorXd semi_analytic_rhs = fd_delta_w_rhs - fd_delta_Hw;
        // opt.update_factorizations();
        // Eigen::VectorXd semi_analytic_delta_w = opt.extractFullSolution(opt.kkt_solver(opt.solver, opt.removeFixedEntries(semi_analytic_rhs)));
        // relerror_semi_analytic_delta_w[i] = (semi_analytic_delta_w - fd_delta_w).norm() / fd_delta_w.norm();

        // {
        //     std::cout << "semi_analytic_delta_w: " << semi_analytic_delta_w.head(5).transpose() << std::endl;
        //     std::cout << "delta_w: " << m_delta_w_x.head(5).transpose() << std::endl;
        //     std::cout << "fd_delta_w: " << fd_delta_w.head(5).transpose() << std::endl;
        //     std::cout << std::endl;

        //     int idx;
        //     Real err = (semi_analytic_delta_w - fd_delta_w).cwiseAbs().maxCoeff(&idx);
        //     std::cout << "greatest abs error " << err << " at entry " << idx << ": "
        //               <<  semi_analytic_delta_w[idx] << " vs " << fd_delta_w[idx] << std::endl;
        //     std::cout <<  semi_analytic_delta_w.segment(idx - 5, 10).transpose() << std::endl;
        //     std::cout << m_delta_w_x.segment(idx - 5, 10).transpose() << std::endl;
        //     std::cout <<  fd_delta_w.segment(idx - 5, 10).transpose() << std::endl;
        //     std::cout << std::endl;
        // }
    }


    std::cout << "Wrote " << hess_J_path << ", " << hess_ac_path << std::endl;
    std::cout << "Rel errors in delta        J: " << relerror_delta_J .transpose() << std::endl;
    // std::cout << "Rel errors in hessian-vec  J: " << relerror_fd_diff_grad_p_J .transpose() << std::endl;
    std::cout << "Rel errors in matvec hessian-vec  J: " << matvec_relerror_fd_diff_grad_p_J .transpose() << std::endl;
    std::cout << "Rel errors in delta x: " << relerror_delta_x.transpose() << std::endl;
    std::cout << "Rel errors in delta w: " << relerror_delta_w.transpose() << std::endl;
    std::cout << "Rel errors in delta Hw: " << relerror_delta_Hw.transpose() << std::endl;
    // std::cout << "Rel errors in delta w rhs: " << relerror_delta_w_rhs.transpose() << std::endl;
}
