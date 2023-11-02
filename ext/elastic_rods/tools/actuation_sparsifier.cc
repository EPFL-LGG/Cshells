#include <iostream>
#include <fstream>
#include "../ActuationSparsifier.hh"
#include "../RodLinkage.hh"
#include "../restlen_solve.hh"
#include "../compute_equilibrium.hh"
#include "../open_linkage.hh"
#include "../knitro.hh"

struct ActuationSparsifyingKnitroProblem : public KnitroProblem<ActuationSparsifyingKnitroProblem> {
    using Base = KnitroProblem<ActuationSparsifyingKnitroProblem>;

    ActuationSparsifyingKnitroProblem(ActuationSparsifier &sparsifier)
        : Base(sparsifier.numVars(), /* num constraints */ 0), m_sparsifier(sparsifier)
    {
        setObjType(KPREFIX(OBJTYPE_GENERAL));
        setObjType(KPREFIX(OBJGOAL_MINIMIZE));

        double torque_ub = KPREFIX(INFBOUND); // TODO: set torque upper bound
        std::vector<double> loBounds;
        std::vector<double> upBounds;

        loBounds.assign(m_sparsifier.numVars(), 0.0);
        upBounds.assign(m_sparsifier.numVars(), torque_ub);

        if (m_sparsifier.regularization() == ActuationSparsifier::Regularization::L0) {
            const int nt = m_sparsifier.numTorques();
            std::vector<int> indexList1(nt), indexList2(nt);

            const int torque_offset = 0;
            const int xi_offset = m_sparsifier.numForceVars();

            for (int ti = 0; ti < nt; ++ti) {
                indexList1[ti] = torque_offset + ti;
                indexList2[ti] = xi_offset     + ti;
            }

#if KNITRO_LEGACY
            setComplementarity(indexList1, indexList2);
#else
            setCompConstraintsParts(knitro::KNSparseMatrixStructure(indexList1, indexList2));
#endif
        }

        setVarLoBnds(loBounds);
        setVarUpBnds(upBounds);
    }

    double evalFC(const double *x,
                        double * /* cval */,
                        double *objGrad,
                        double * /* jac */) {
        const size_t nv = m_sparsifier.numVars();
        auto vars = Eigen::Map<const Eigen::VectorXd>(x, nv);
        Eigen::Map<Eigen::VectorXd>(objGrad, nv) = m_sparsifier.grad(vars);
        return m_sparsifier.eval(vars);
    }

    int evalGA(const double * /* x */, double * /* objGrad */, double * /* jac */) {
        // Tell Knitro that gradient is evaluated by evaluateFC
        return KPREFIX(RC_EVALFCGA);
    }

    // Note: "lambda" contains a Lagrange multiplier for each constraint and each variable.
    // The first numConstraints entries give each constraint's multiplier in order, and the remaining
    // numVars entries give each the multiplier for the variable's active simple bound constraints (if any).
    int evalHessVec(const double *x, double sigma, const double * /* lambda */,
                    const double *vec, double *hessVec) {
        const size_t nv = m_sparsifier.numVars();
        const size_t nc = 0;
        if ((nc == 0) && (sigma == 0.0)) throw std::runtime_error("Knitro requested empty Hessian!");

        auto vars       = Eigen::Map<const Eigen::VectorXd>(x, nv);
        auto delta_vars = Eigen::Map<const Eigen::VectorXd>(vec, nv);

        // Apply Hessian of sigma * J + lambda[0] * angle_constraint if angle constraint is active, J otherwise
        auto result = m_sparsifier.apply_hess(vars, delta_vars);
        Eigen::Map<Eigen::VectorXd>(hessVec, nv) = result;
        return 0; // indicate success
    }

private:
    ActuationSparsifier &m_sparsifier;
};

struct ActuationSparsifyingKnitroNewPtCallback : public NewPtCallbackBase {
    ActuationSparsifyingKnitroNewPtCallback(ActuationSparsifier &sp) : m_sparsifier(sp) { }

    virtual int operator()(const double *x) override {
        m_sparsifier.newPt(Eigen::Map<const Eigen::VectorXd>(x, m_sparsifier.numVars()));
        return 0;
    }

private:
    ActuationSparsifier &m_sparsifier;
};

void sparsify(RodLinkage &linkage, NewtonOptimizer &opt,
              Real eta, Real p) {
    ActuationSparsifier::Regularization reg = (p == 1.0) ? ActuationSparsifier::Regularization::L1
                                                         : ActuationSparsifier::Regularization::LP;
    ActuationSparsifier sparsifier(linkage, opt, reg);
    sparsifier.eta = eta;
    sparsifier.p = p;

    ////////////////////////////////////////////////////////////////////////////
    // Run optimization
    ////////////////////////////////////////////////////////////////////////////
    int num_steps = 5000;
    ActuationSparsifyingKnitroProblem problem(sparsifier);
    std::vector<Real> x_init(sparsifier.numVars());
    Eigen::Map<Eigen::VectorXd>(x_init.data(), x_init.size()) = sparsifier.initialVarsFromElasticForces(linkage.gradient());
    problem.setXInitial(x_init);

    ActuationSparsifyingKnitroNewPtCallback callback(sparsifier);
    problem.setNewPointCallback(&callback);
    // Create a solver - optional arguments:
    // exact first and second derivatives; no KPREFIX(GRADOPT_)* or KPREFIX(HESSOPT_)* parameter is needed.
    KnitroSolver solver(&problem, /* exact gradients */ 1, /* exact Hessian-vector products */ 5);
    // KnitroSolver solver(&problem, /* exact gradients */ 1, /* BFGS */ 2);
    solver.useNewptCallback();
    solver.setParam(KPREFIX(PARAM_HONORBNDS), KPREFIX(HONORBNDS_ALWAYS)); // always respect bounds during optimization
    solver.setParam(KPREFIX(PARAM_MAXIT), num_steps);
    solver.setParam(KPREFIX(PARAM_PRESOLVE), KPREFIX(PRESOLVE_NONE));
    solver.setParam(KPREFIX(PARAM_PAR_NUMTHREADS), 12);
    solver.setParam(KPREFIX(PARAM_HESSIAN_NO_F), KPREFIX(HESSIAN_NO_F_ALLOW)); // allow Knitro to call our hessvec with sigma = 0
    solver.setParam(KPREFIX(PARAM_ALGORITHM), KPREFIX(ALG_ACT_CG));
    solver.setParam(KPREFIX(PARAM_ACT_QPALG), KPREFIX(ACT_QPALG_ACT_CG)); // default ended up choosing KPREFIX(ACT_QPALG_BAR_DIRECT)

    solver.setParam(KPREFIX(PARAM_OPTTOL), 1e-6);
    solver.setParam(KPREFIX(PARAM_OUTLEV), KPREFIX(OUTLEV_ALL));

    try {
        BENCHMARK_RESET();
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

    {
        std::ofstream outFile("torques.txt");
        outFile << sparsifier.extractTorques(linkage.gradient());
        std::ofstream gradFile("gradient.txt");
        gradFile << linkage.gradient();
    }
}

int main(int argc, const char *argv[]) {
    if (argc != 7) {
        std::cout << "Usage: actuation_sparsifier flat_linkage.obj cross_section.json initial_deployment_angle rest_lengths.txt eta p" << std::endl;
        exit(-1);
    }

    const std::string linkage_path(argv[1]),
                cross_section_path(argv[2]);

    Real deployedActuationAngle = std::stod(argv[3]);

    RodLinkage linkage(linkage_path);

    RodMaterial mat;
    if (cross_section_path.substr(cross_section_path.size() - 4) == "json") {
        mat.set(*CrossSection::load(cross_section_path), RodMaterial::StiffAxis::D1, false);
    }
    else {
        mat.setContour(20000, 0.3, cross_section_path, 1.0, RodMaterial::StiffAxis::D1);
    }

    linkage.setMaterial(mat);
    // std::cout << "stretching stiffness: " << mat.stretchingStiffness << std::endl;
    // std::cout << "twisting   stiffness: " << mat.twistingStiffness << std::endl;
    // std::cout << "bending    stiffness: " << mat.bendingStiffness.lambda_1 << '\t' << mat.bendingStiffness.lambda_2 << std::endl;

    // If the user passes in the deployed linkage instead of the flat linkage,
    // we likely should pick the obtuse angle to define the opening angles instead of the acute
    // angles automatically picked by the RodLinkage constructor.
    if (std::abs(deployedActuationAngle - (M_PI - linkage.getAverageJointAngle())) < 0.01) {
        linkage.swapJointAngleDefinitions();
    }

    // std::cout << "Post-load energy: " << linkage.energy() << std::endl;
    // linkage.writeRodDebugData("debug.msh");
    // linkage.writeLinkageDebugData("ldebug.msh");

    if (argc >= 5) {
        const std::string rl_path(argv[4]);
        std::ifstream rl_file(rl_path);
        if (!rl_file.is_open()) throw std::runtime_error("Failed to open input file '" + rl_path + "'");
        Real rl = 0.0;
        std::vector<Real> rlens;
        while (rl_file >> rl)
            rlens.push_back(rl);
        if (rlens.size() != linkage.numSegments()) throw std::runtime_error("Read incorrect number of rest lengths");
        linkage.setPerSegmentRestLength(Eigen::Map<const Eigen::VectorXd>(rlens.data(), rlens.size())); // These rest lengths should actually place the flat linkage in equilibrium if the input is valid...
    }
    else {
        std::cout << "Solving for rest lengths" << std::endl;
        restlen_solve(linkage);
    }

    // std::cout << "Post-rl energy: " << linkage.energy() << std::endl;

    Real eta = std::stod(argv[5]);
    Real   p = std::stod(argv[6]);

    ////////////////////////////////////////////////////////////////////////////
    // Computed deployed equilibrium under full actuation
    ////////////////////////////////////////////////////////////////////////////
    // Constrain global rigid motion by fixing the position, orientation of the centermost joint
    const size_t jdo = linkage.dofOffsetForJoint(linkage.centralJoint());
    std::vector<size_t> rigidMotionFixedVars = { jdo + 0, jdo + 1, jdo + 2, jdo + 3, jdo + 4, jdo + 5 };

    // Compute undeployed equilibrium (preserving initial average actuation angle).
    NewtonOptimizerOptions eopts;
    eopts.beta = 1e-8;
    eopts.gradTol = 1e-7;
    eopts.verbose = 10;
    eopts.niter = 50;

    open_linkage(linkage, deployedActuationAngle, eopts, linkage.centralJoint());

    // {
    //     std::ofstream gradFile("gradient_post_open.txt");
    //     gradFile << linkage.gradient();
    //     std::cout << "energy: " << linkage.energy() << std::endl;
    //     std::cout << "average opening angle: " << linkage.getAverageJointAngle() << std::endl;
    // }

    ////////////////////////////////////////////////////////////////////////////
    // Check that deformation doesn't change when we switch to using explicit
    // torques to hold the linkage open.
    ////////////////////////////////////////////////////////////////////////////
    std::cout << "Simulating the applied forces" << std::endl;
    auto custom_force_actuation_equilibrium = get_equilibrium_optimizer(linkage, TARGET_ANGLE_NONE, rigidMotionFixedVars);
    custom_force_actuation_equilibrium->options = eopts;
    auto &externalForces = dynamic_cast<EquilibriumProblem<RodLinkage>&>(custom_force_actuation_equilibrium->get_problem()).external_forces;
    const size_t nj = linkage.numJoints();
    {
        auto elasticForces = linkage.gradient();
        externalForces.setZero(elasticForces.size());
        for (size_t ji = 0; ji < nj; ++ji) {
            const size_t alpha_idx = linkage.dofOffsetForJoint(ji) + 6;
            externalForces[alpha_idx] = elasticForces[alpha_idx];
        }
    }
    std::cout << "Running external torque simulation" << std::endl;
    custom_force_actuation_equilibrium->optimize();
    linkage.saveVisualizationGeometry("pre_sparsification.msh");

    sparsify(linkage, *custom_force_actuation_equilibrium, eta, p);

    linkage.saveVisualizationGeometry("post_sparsification.msh");

    // BENCHMARK_REPORT_NO_MESSAGES();

    return 0;
}
