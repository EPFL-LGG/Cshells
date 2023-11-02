#if HAS_KNITRO
#include "knitro.hh"

template<template<typename> class Object>
struct OptKnitroProblem : public KnitroProblem<OptKnitroProblem<Object>> {
    using Base = KnitroProblem<OptKnitroProblem<Object>>;

    OptKnitroProblem(LinkageOptimization<Object> &lopt, bool applyAngleConstraint, bool applyFlatnessConstraint, double minRestLen)
        : Base(lopt.numFullParams(), /* num constraints */ int(applyAngleConstraint) + int(applyFlatnessConstraint)), m_lopt(lopt),
          m_applyAngleConstraint(applyAngleConstraint),
          m_applyFlatnessConstraint(applyFlatnessConstraint),
          m_optimizeTargetAngle(lopt.getOptimizeTargetAngle())
    {
        std::cout << "Constructed optimization problem with applyFlatnessConstraint = " << applyFlatnessConstraint << std::endl;
        this->setObjType(KPREFIX(OBJTYPE_GENERAL));
        this->setObjType(KPREFIX(OBJGOAL_MINIMIZE));

        // Set the bounds for the design parameters:
        //     Rest length parameters are between epsilon and infinity
        //     Rest curvature parameters are between -2*pi and 2*pi
        //     Target average opening angle are between -2*pi and 2*pi
        if (minRestLen < 0) minRestLen = m_lopt.defaultLengthBound();
#if KNITRO_LEGACY
        if (m_applyAngleConstraint) {
            this->setConTypes (angleConstraintIdx(), KPREFIX(CONTYPE_GENERAL));
            // Constraint smin(alpha) >= pi/1024; note that lopt.LOMinAngleConstraint::eval() returns smin(alpha) - eps.
            this->setConLoBnds(angleConstraintIdx(), M_PI / 1024 - lopt.getMinAngleConstraint().eps);
            this->setConUpBnds(angleConstraintIdx(), KPREFIX(INFBOUND));
        }

        if (m_applyFlatnessConstraint) {
            this->setConTypes (flatnessConstraintIdx(), KPREFIX(CONTYPE_GENERAL));
            this->setConLoBnds(flatnessConstraintIdx(), 0.0);
            this->setConUpBnds(flatnessConstraintIdx(), 0.0);
        }
#else
        throw std::runtime_error("TODO: port nonlinear constraints to the new Knitro API");
#endif
        
        if (minRestLen < 0) minRestLen = m_lopt.defaultLengthBound();
        std::vector<double> restLenLoBounds(m_lopt.numRestLen(), minRestLen);
        std::vector<double> restLenUpBounds(m_lopt.numRestLen(), KPREFIX(INFBOUND));

        std::vector<double> restKappaLoBounds(m_lopt.numRestKappaVars(), -2 * M_PI);
        std::vector<double> restKappaUpBounds(m_lopt.numRestKappaVars(),  2 * M_PI);

        std::vector<double> alphaLoBounds(1, -2 * M_PI);
        std::vector<double> alphaUpBounds(1,  2 * M_PI);

        std::vector<double> loBounds;
        std::vector<double> upBounds;
        if (lopt.use_restKappa()) {
            loBounds.insert(loBounds.end(), restKappaLoBounds.begin(), restKappaLoBounds.end());
            upBounds.insert(upBounds.end(), restKappaUpBounds.begin(), restKappaUpBounds.end());
        }

        if (lopt.use_restLen()) {
            loBounds.insert(loBounds.end(), restLenLoBounds.begin(), restLenLoBounds.end());
            upBounds.insert(upBounds.end(), restLenUpBounds.begin(), restLenUpBounds.end());
        }

        if (m_optimizeTargetAngle) {
            loBounds.insert(loBounds.end(), alphaLoBounds.begin(), alphaLoBounds.end());
            upBounds.insert(upBounds.end(), alphaUpBounds.begin(), alphaUpBounds.end());
        }

        this->setVarLoBnds(loBounds);
        this->setVarUpBnds(upBounds);
    }

    size_t numConstraints() const { return m_applyAngleConstraint + m_applyFlatnessConstraint; }
    size_t angleConstraintIdx()    const { return 0; }
    size_t flatnessConstraintIdx() const { return size_t(m_applyAngleConstraint); }

    double evalFC(const double *x,
                        double *cval,
                        double *objGrad,
                        double *jac) {
        const size_t np = m_lopt.numParams(), nfp = m_lopt.numFullParams();

        auto params = Eigen::Map<const Eigen::VectorXd>(x, nfp);
        Real val = m_lopt.J(params);

        auto g = m_lopt.gradp_J(params);
        Eigen::Map<Eigen::VectorXd>(objGrad, nfp) = g;

        if (m_applyAngleConstraint || m_applyFlatnessConstraint) {
	        // Using KTRProblem class, by default the Jacobian is assumed to be dense and stored row-wise.
	        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> constraintJacobian(jac, numConstraints(), nfp);

	        if (m_applyAngleConstraint) {
	            cval[angleConstraintIdx()] = m_lopt.angle_constraint(params);
	            constraintJacobian.row(angleConstraintIdx()) = m_lopt.gradp_angle_constraint(params).transpose();
	        }

	        if (m_applyFlatnessConstraint) {
	            cval[flatnessConstraintIdx()] = m_lopt.c(params);
	            constraintJacobian.row(flatnessConstraintIdx()) = m_lopt.gradp_c(params).transpose();
	        }
		}

        return val;
    }

    int evalGA(const double * /* x */, double * /* objGrad */, double * /* jac */) {
        // Tell Knitro that gradient is evaluated by evaluateFC
        return KPREFIX(RC_EVALFCGA);
    }

    // Note: "lambda" contains a Lagrange multiplier for each constraint and each variable.
    // The first numConstraints entries give each constraint's multiplier in order, and the remaining
    // numVars entries give each the multiplier for the variable's active simple bound constraints (if any).
    int evalHessVec(const double *x, double sigma, const double *lambda,
                    const double *vec, double *hessVec) {
        const size_t np = m_lopt.numParams(), nfp = m_lopt.numFullParams();
        const size_t nc = numConstraints();
        if ((nc == 0) && (sigma == 0.0)) throw std::runtime_error("Knitro requested empty Hessian!");

        auto params  = Eigen::Map<const Eigen::VectorXd>(x, nfp);
        auto delta_p = Eigen::Map<const Eigen::VectorXd>(vec, nfp);

        // Apply Hessian of sigma * J + lambda[0] * angle_constraint if angle constraint is active, J otherwise
        auto result = m_lopt.apply_hess(params, delta_p, sigma,
                                        m_applyFlatnessConstraint ? lambda[flatnessConstraintIdx()] : 0.0,
                                        m_applyAngleConstraint    ? lambda[angleConstraintIdx()]    : 0.0);
        Eigen::Map<Eigen::VectorXd>(hessVec, nfp) = result;
        return 0; // indicate success
    }

private:
    LinkageOptimization<Object> &m_lopt;
    bool m_applyAngleConstraint, m_applyFlatnessConstraint, m_optimizeTargetAngle;
};


template<template<typename> class Object>
struct OptKnitroNewPtCallback : public NewPtCallbackBase {
    OptKnitroNewPtCallback(LinkageOptimization<Object> &lopt, std::function<void()> &update_viewer)
        : NewPtCallbackBase(), m_update_viewer(update_viewer), m_lopt(lopt) { }

    virtual int operator()(const double *x) override {
        std::cout << "Starting evaluating new point." << std::endl;
        const size_t nfp = m_lopt.numFullParams();
        m_lopt.newPt(Eigen::Map<const Eigen::VectorXd>(x, nfp));
        std::cout << "Done evaluating new point." << std::endl;
        m_update_viewer();
        std::cout << "Done updating viewer." << std::endl;
        return 0;
    }
private:
    std::function<void()> m_update_viewer;
    LinkageOptimization<Object> &m_lopt;
};

void configureKnitroSolver(KnitroSolver &solver, int num_steps, Real trust_region_scale, Real optimality_tol) {
    solver.useNewptCallback();
    solver.setParam(KPREFIX(PARAM_HONORBNDS), KPREFIX(HONORBNDS_ALWAYS)); // always respect bounds during optimization
    solver.setParam(KPREFIX(PARAM_MAXIT), num_steps);
    solver.setParam(KPREFIX(PARAM_PRESOLVE), KPREFIX(PRESOLVE_NONE));
    solver.setParam(KPREFIX(PARAM_DELTA), trust_region_scale);
    // solver.setParam(KPREFIX(PARAM_DERIVCHECK), KPREFIX(DERIVCHECK_ALL));
    // solver.setParam(KPREFIX(PARAM_DERIVCHECK_TYPE), KPREFIX(DERIVCHECK_CENTRAL));
    // solver.setParam(KPREFIX(PARAM_ALGORITHM), KPREFIX(ALG_BAR_DIRECT));   // interior point with exact Hessian
    solver.setParam(KPREFIX(PARAM_PAR_NUMTHREADS), 12);
    solver.setParam(KPREFIX(PARAM_HESSIAN_NO_F), KPREFIX(HESSIAN_NO_F_ALLOW)); // allow Knitro to call our hessvec with sigma = 0
    // solver.setParam(KPREFIX(PARAM_LINSOLVER), KPREFIX(LINSOLVER_MKLPARDISO));
    solver.setParam(KPREFIX(PARAM_ALGORITHM), KPREFIX(ALG_ACT_CG));
    solver.setParam(KPREFIX(PARAM_ACT_QPALG), KPREFIX(ACT_QPALG_ACT_CG)); // default ended up choosing KPREFIX(ACT_QPALG_BAR_DIRECT)
    // solver.setParam(KPREFIX(PARAM_CG_MAXIT), 25);
    // solver.setParam(KPREFIX(PARAM_CG_MAXIT), int(lopt.numParams())); // TODO: decide on this.
    // solver.setParam(KPREFIX(PARAM_BAR_FEASIBLE), KPREFIX(BAR_FEASIBLE_NO));

    solver.setParam(KPREFIX(PARAM_OPTTOL), optimality_tol);
    solver.setParam(KPREFIX(PARAM_OUTLEV), KPREFIX(OUTLEV_ALL));
    solver.setParam(KPREFIX(PARAM_DEBUG), 0);
}

template<template<typename> class Object>
int LinkageOptimization<Object>::optimize(OptAlgorithm alg, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, std::function<void()> &update_viewer, double minRestLen, bool
              applyAngleConstraint, bool applyFlatnessConstraint) {
    OptKnitroProblem<Object> problem(*this, applyAngleConstraint, applyFlatnessConstraint, minRestLen);

    const size_t np = numParams(), nfp = numFullParams();
    std::vector<Real> x_init(nfp);
    Eigen::Map<Eigen::VectorXd>(x_init.data(), x_init.size()) = getFullDesignParameters();
    problem.setXInitial(x_init);

    OptKnitroNewPtCallback<Object> callback(*this, update_viewer);
    problem.setNewPointCallback(&callback);
    // Create a solver - optional arguments:
    // exact first and second derivatives; no KPREFIX(GRADOPT_)* or KPREFIX(HESSOPT_)* parameter is needed.
    int hessopt = 0;
    if (alg == OptAlgorithm::NEWTON_CG) hessopt = 5;  // exact Hessian-vector products
    else if (alg == OptAlgorithm::BFGS) hessopt = 2;  // BFGS approximation
    else throw std::runtime_error("Unknown algorithm");

    KnitroSolver solver(&problem, /* exact gradients */ 1, hessopt);
    configureKnitroSolver(solver, int(num_steps), trust_region_scale, optimality_tol);

    int solveStatus = 1;
    try {
        BENCHMARK_RESET();
        std::cout << "Starting..." << std::endl;
        int solveStatus = solver.solve();
        BENCHMARK_REPORT_NO_MESSAGES();

        if (solveStatus != 0) {
            std::cout << std::endl;
            std::cout << "KNITRO failed to solve the problem, final status = ";
            std::cout << solveStatus << std::endl;
        }
    }
    catch (KnitroException &e) {
        problem.setNewPointCallback(nullptr);
        printKnitroException(e);
        throw e;
    }
    problem.setNewPointCallback(nullptr);

    return solveStatus;
}

#endif
