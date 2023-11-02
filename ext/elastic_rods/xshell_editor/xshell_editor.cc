#include <igl/opengl/glfw/Viewer.h>
#include <GLFW/glfw3.h>
#include "../RodLinkage.hh"
#include "../LinkageOptimization.hh"
#include "../XShellOptimization.hh"
#include "../design_parameter_solve.hh"
#include "../infer_target_surface.hh"
#include "../open_linkage.hh"
#include <MeshFEM/Geometry.hh>
#include <MeshFEM/unused.hh>

#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/file_dialog_open.h>
#include <imgui/imgui.h>

#include "../knitro.hh"

#include <string>
#include <iostream>
#include <map>
#include <thread>

using Viewer     = igl::opengl::glfw::Viewer;
using ViewerCore = igl::opengl::ViewerCore;
   
using Linkage_dPC = DesignParameterConfig;
using LO = LinkageOptimization<RodLinkage_T>;


// Global flag used to interrupt optimization; setting this to true will
// cause Knitro to quit when the current iteration completes.
bool optimization_cancelled = false;
bool optimization_running = false;
bool needs_redraw = false;
bool requestHessianDump = false;

size_t numHessDump = 0;

double fd_eps = 1e-3;

void dumpHessians(XShellOptimization<RodLinkage_T> &lopt) {
    lopt.dumpHessians("hess_J_"  + std::to_string(numHessDump) + ".txt",
                      "hess_ac_" + std::to_string(numHessDump) + ".txt",
                      fd_eps);
    ++numHessDump;
}

struct CachedStats {
    Real deployed_avg_angle, closed_avg_angle, closed_min_angle, closed_smin_angle, J, J_target, gradp_J_norm, gradp_J_target_norm, E_deployed, E_flat, E_deployed_rod_max,
         flatness;

    void update(XShellOptimization<RodLinkage_T> &lopt) {
        deployed_avg_angle  = lopt.getTargetAngle();
        closed_avg_angle    = lopt.getLinesearchFlatLinkage().getAverageJointAngle();
        closed_min_angle    = lopt.getLinesearchFlatLinkage().getMinJointAngle();
        closed_smin_angle   = lopt.angle_constraint(lopt.params()) + lopt.getMinAngleConstraint().eps;
        J                   = lopt.J();
        J_target            = lopt.LO::J_target();
        gradp_J_norm        = lopt.LO::gradp_J().norm();
        gradp_J_target_norm = lopt.gradp_J_target().norm();
        E_deployed          = lopt.getLinesearchDeployedLinkage().energy();
        E_flat              = lopt.getLinesearchFlatLinkage().energy();
        E_deployed_rod_max  = lopt.getLinesearchDeployedLinkage().maxRodEnergy();
        flatness            = lopt.c();
    }
};

CachedStats stats;

struct LOptKnitroNewPtCallback : public NewPtCallbackBase {
    LOptKnitroNewPtCallback(XShellOptimization<RodLinkage_T> &lopt) : m_lopt(lopt) { }

    virtual int operator()(const double *x) override {
        const size_t np = m_lopt.numParams();
        m_lopt.newPt(Eigen::Map<const Eigen::VectorXd>(x, np));
        if (requestHessianDump) { dumpHessians(m_lopt); }
        needs_redraw = true;
        stats.update(m_lopt);
        glfwPostEmptyEvent(); // Run another iteration of the event loop so the viewer redraws.
        return (optimization_cancelled) ? KPREFIX(RC_USER_TERMINATION) : 0;
    }

private:
    XShellOptimization<RodLinkage_T> &m_lopt;
};

void optimize(OptAlgorithm alg, XShellOptimization<RodLinkage_T> &lopt, bool
              applyAngleConstraint, bool applyFlatnessConstraint, size_t num_steps,
              Real trust_region_scale, Real optimality_tol, double minRestLen) {
    OptKnitroProblem<RodLinkage_T> problem(lopt, applyAngleConstraint, applyFlatnessConstraint, minRestLen);

    std::vector<Real> x_init(lopt.numParams());
    Eigen::Map<Eigen::VectorXd>(x_init.data(), x_init.size()) = lopt.getLinesearchFlatLinkage().getDesignParameters();
    problem.setXInitial(x_init);

    optimization_cancelled = false;
    optimization_running = true;
    LOptKnitroNewPtCallback callback(lopt);
    problem.setNewPointCallback(&callback);
    // Create a solver - optional arguments:
    // exact first and second derivatives; no KPREFIX(GRADOPT_)* or KPREFIX(HESSOPT_)* parameter is needed.
    int hessopt = 0;
    if (alg == OptAlgorithm::NEWTON_CG) hessopt = 5;  // exact Hessian-vector products
    else if (alg == OptAlgorithm::BFGS) hessopt = 2;  // BFGS approximation
    else throw std::runtime_error("Unknown algorithm");

    KnitroSolver solver(&problem, /* exact gradients */ 1, hessopt);
    solver.useNewptCallback();
    solver.setParam(KPREFIX(PARAM_HONORBNDS), KPREFIX(HONORBNDS_ALWAYS)); // always respect bounds during optimization
    solver.setParam(KPREFIX(PARAM_MAXIT), int(num_steps));
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
        optimization_running = false;
        throw e;
    }
    problem.setNewPointCallback(nullptr);

    optimization_running = false;
}

void getLinkageMesh(const RodLinkage &l, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    static std::vector<MeshIO::IOVertex > vertices;
    static std::vector<MeshIO::IOElement> elements;
    vertices.clear();
    elements.clear();
    l.visualizationGeometry(vertices, elements, true);
    meshio_to_igl(vertices, elements, V, F);
    // std::cout << "getLinkageMesh" << std::endl;
    // static size_t i = 0;
    // l.saveVisualizationGeometry("visMesh_" + std::to_string(i++) + ".msh");
}

int main(int argc, char * argv[])
{
    if ((argc != 3) && (argc != 5)) {
        std::cerr << "Usage: xshell_editor linkage.obj cross_section.json [stiffen_factor stiffen_boxes.rbc]" << std::endl;
        exit(-1);
    }

    const size_t width = 1280;
    const size_t height = 800;

    // Step size for gradient descent
    size_t num_steps = 2000;
    OptAlgorithm opt_algorithm = OptAlgorithm::NEWTON_CG;
    double trust_region_scale = 1.0;
    PredictionOrder prediction_order = PredictionOrder::Two;

    // Whether to apply the minimum angle constraint for the flat linkage during optimization.
    bool applyAngleConstraint = true;
    bool applyFlatnessConstraint = false;
    bool allowFlatActuation = true;
    double jointPosWeight = 0.01,              // controls trade-off between fitting to target joint positions and fitting to target surface
           jointPosValence2Multiplier = 10.0; // controls whether we try harder to fit valence 2 vertices to their target positions
    bool showTargetSurface = false;
    bool skipTargetSurfaceConstruction = false;
    size_t loop_subdivisions = 0;

    double scale_x = 1.0;
    double scale_y = 1.0;
    double scale_z = 1.0;


    bool setting_deployment_angle = false;
    double target_deployment_angle = 0.0;

    double designOptimizationTol = 1e-2;

    bool useRestKappa = true;

    // Construct the flat and deployed linkage
    const std::string linkage_path(argv[1]);
    const std::string cross_section_path(argv[2]);

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> edges;
    MeshIO::load(linkage_path, vertices, edges);
    BBox<Point3D> bb(vertices);
    auto dim = bb.dimensions();
    // Orient the longer axis along x; assume z is already the shortest.
    const bool rot_90 = (dim[1] > dim[0]);
    if (rot_90) {
        for (auto &v : vertices)
            v.point = Point3D(-v[1], v[0], v[2]);
    }

    // RodLinkage flatLinkage(vertices, edges, 20);
    RodLinkage flatLinkage(vertices, edges, 20, true, InterleavingType::xshell);
    // TODO: make a custom cross-section that does the same thing as RodMaterial::setContour
    {
        auto ext = cross_section_path.substr(cross_section_path.size() - 4);
        if ((ext == ".obj") || ext == ".msh") {
            RodMaterial mat;
            mat.setContour(20000, 0.3, cross_section_path, 1.0, RodMaterial::StiffAxis::D1,
                    false, "", 0.001, 10);
            flatLinkage.setMaterial(mat);
        }
        else flatLinkage.setMaterial(RodMaterial(*CrossSection::load(cross_section_path)));
    }

    // Stiffen the rectangular regions passed on the command line (if any).
    // This is to simulate the addition of stiff sleeves to connect rods.
    if (argc == 5) {
        std::cout << "Stiffening regions" << std::endl;
        double factor = std::stod(argv[3]);
        RectangularBoxCollection boxes(argv[4]);
        if (rot_90) { // we may have reoriented the linkage above!
            for (auto &b : boxes.boxes) {
                for (size_t c = 0; c < 8; ++c) {
                    Point3D p = b.corner(c);
                    b.corner(c) << -p[1], p[0], p[2];
                }
            }
        }
        flatLinkage.stiffenRegions(boxes, factor);
        // Verify that the stiffening regions worked by outputting the bending
        // and twisting stiffnesses over the rod vertices.
        flatLinkage.writeRodDebugData("stiffen_debug.msh");
        std::cout << "Wrote 'stiffen_debug.msh'" << std::endl;
    }

    std::cout << "Initial design parameter solve:" << std::endl;
    {
        NewtonOptimizerOptions designParameter_eopts;
        // designParameter_eopts.useIdentityMetric = true;
        designParameter_eopts.niter = 10000;
        designParameter_eopts.verbose = 10;
        flatLinkage.setDesignParameterConfig(true, true);
        designParameter_solve(flatLinkage, designParameter_eopts);
        flatLinkage.setDesignParameterConfig(true, true);
    }

    NewtonOptimizerOptions eopts;
    eopts.gradTol = 1e-6;
    eopts.verbose = 10;
    eopts.niter = 5;
    eopts.beta = 1e-8;
    // eopts.beta = 1e-4;
    // eopts.beta = 1e-2;
    // eopts.beta = 1;

    size_t pinJoint = flatLinkage.centralJoint();
    {
        const size_t jdo = flatLinkage.dofOffsetForJoint(pinJoint);
        std::vector<size_t> rigidMotionPinVars;
        for (size_t i = 0; i < 6; ++i) rigidMotionPinVars.push_back(jdo + i);
        compute_equilibrium(flatLinkage, flatLinkage.getAverageJointAngle(), eopts, rigidMotionPinVars);
    }

    // eopts.niter = 25;
    eopts.niter = 5;

    RodLinkage deployedLinkage(flatLinkage);

    // Construct the linkage optimizer, running the initial equilibrium solve.
    // XShellOptimization lopt(flatLinkage, deployedLinkage, eopts);
    // XShellOptimization lopt(flatLinkage, deployedLinkage, eopts, std::make_unique<LOMinAngleConstraint>(M_PI / 48));
    XShellOptimization<RodLinkage_T> lopt(flatLinkage, deployedLinkage, eopts, M_PI / 128, pinJoint, allowFlatActuation);

    BENCHMARK_REPORT_NO_MESSAGES();

    stats.update(lopt);

    // Create viewer with three empty mesh slots.
    Viewer viewer;
    viewer.append_mesh();
    viewer.append_mesh();
    const int flat_mesh_id = 0, deployed_mesh_id = 1, target_surface_mesh_id = 2;

    // Scratch space for updateLinkageMeshes
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    auto updateLinkageMeshes = [&](){
        getLinkageMesh(flatLinkage, V, F);
        viewer.data(    flat_mesh_id).set_mesh(V, F);
        getLinkageMesh(deployedLinkage, V, F);
        viewer.data(deployed_mesh_id).set_mesh(V, F);
        for (size_t id = 0; id < 2; ++id) {
            viewer.data(id).face_based = true;
            viewer.data(id).show_lines = false;
            viewer.data(id).set_colors(Eigen::RowVector3d(0.65, 0.65, 0.65));
        }
    };

    auto updateTargetMesh = [&]() {
        viewer.data(target_surface_mesh_id).clear();
        viewer.data(target_surface_mesh_id).set_mesh(lopt.target_surface_fitter.getV(), lopt.target_surface_fitter.getF());
        viewer.data(target_surface_mesh_id).face_based = true;
        viewer.data(target_surface_mesh_id).show_lines = false;
        viewer.data(target_surface_mesh_id).set_colors(Eigen::RowVector3d(0.95, 0.95, 0.95));
    };

    updateLinkageMeshes();
    updateTargetMesh();

    int top_view, bot_view;
    viewer.callback_init = [&](Viewer &)
    {
        glfwSetWindowTitle(viewer.window, "Linkage Editor");
        top_view = viewer.core_list[0].id;
        bot_view = viewer.append_core(Eigen::Vector4f(0, 0, width, height)); // will be resized by callback_post_resize
        viewer.core(top_view).background_color << 0.95, 0.95, 0.95, 1.0;
        viewer.core(bot_view).background_color << 0.92, 0.92, 0.92, 1.0;

        viewer.core(top_view).rotation_type = ViewerCore::ROTATION_TYPE_TRACKBALL;
        viewer.core(bot_view).rotation_type = ViewerCore::ROTATION_TYPE_TRACKBALL;

        viewer.core(top_view).camera_dnear = viewer.core(bot_view).camera_dnear = 0.005;
        viewer.core(top_view).camera_dfar  = viewer.core(bot_view).camera_dfar  = 50;
        // Show deployed linkage above, flat linkage below (by hiding)
        viewer.data(    flat_mesh_id).set_visible(false, top_view);
        viewer.data(deployed_mesh_id).set_visible(false, bot_view);

        // Initially the target surface mesh is invisible (in both views)
        viewer.data(target_surface_mesh_id).set_visible(false, top_view);
        viewer.data(target_surface_mesh_id).set_visible(false, bot_view);

        return false; // also init the plugins
    };

    // Update meshes before redrawing if the optimizer has updated the linkage
    // state.
    viewer.callback_pre_draw = [&](Viewer &/* v */) {
        if (needs_redraw) {
            updateLinkageMeshes();
            needs_redraw = false;
        }
        return false;
    };

    std::thread optimization_thread;

    viewer.callback_key_pressed = [&](Viewer &, unsigned int key, int /* mod */)
    {
        if ((key == GLFW_KEY_EQUAL) || (key == GLFW_KEY_MINUS)) {
            if (viewer.data(deployed_mesh_id).dirty) return false;
            if (optimization_running) return false; // Opening/closing disabled while optimization is running.
            Real newAngle = lopt.getTargetAngle() + 0.02 * ((key == GLFW_KEY_EQUAL) ? 1 : -1);

            eopts.niter = 30;
            lopt.setEquilibriumOptions(eopts);
            lopt.setTargetAngle(newAngle);
            eopts.niter = 10;
            // eopts.useIdentityMetric = true;
            lopt.setEquilibriumOptions(eopts);

            updateLinkageMeshes();
            stats.update(lopt);
            return true;
        }

        if ((key == 'g') || (key == 'G')) {
            if (!optimization_running) {
                optimization_running = true;
                if (optimization_thread.joinable())
                    optimization_thread.join(); // shouldn't actually happen...
                optimization_thread = std::thread(optimize, opt_algorithm, std::ref(lopt), applyAngleConstraint, applyFlatnessConstraint, num_steps, trust_region_scale, designOptimizationTol, -1);
            }
            return true;
        }

        if ((key == 'h') || (key == 'H')) {
            if (!optimization_running) dumpHessians(lopt);
            else                       requestHessianDump = true;
            return true;
        }

        if ((key == 'c') || (key == 'C')) {
            optimization_cancelled = true;
            return true;
        }

        // if ((key == 'f') || (key == 'f')) {
		// 	auto params = lopt.getLinesearchFlatLinkage().getDesignParameters();
		// 	auto grad = lopt.gradp_J(params);
        //     Real eps = 1e-4;
        //     params[0] += eps;
        //     Real Jplus = lopt.J(params);
        //     params[0] -= 2 * eps;
        //     Real Jminus = lopt.J(params);
        //     std::cout.precision(19);
        //     std::cout << "fd diff J" << (Jplus - Jminus) / (2 * eps) << std::endl;
        //     std::cout << "grad    J" << grad[0] << std::endl;

        //     return true;
        // }

        if ((key == 't') || (key == 'T')) {
            std::vector<MeshIO::IOVertex > vertices;
            std::vector<MeshIO::IOElement> elements;
            infer_target_surface(lopt.getLinesearchDeployedLinkage(), vertices, elements, 2);
            MeshIO::save("triangulation.msh", vertices, elements);
            return true;
        }

        return false;
    };

    viewer.callback_post_resize = [&](Viewer &v, int w, int h) {
        v.core(top_view).viewport = Eigen::Vector4f(0, h / 3, w, h - (h / 3));
        v.core(bot_view).viewport = Eigen::Vector4f(0, 0, w, h / 3);
        return true;
    };

    // Initialize the split views' viewports
    viewer.callback_post_resize(viewer, width, height);

    ////////////////////////////////////////////////////////////////////////////
    // IMGui UI
    ////////////////////////////////////////////////////////////////////////////
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_window = [&]() { };

    // Draw additional windows
    menu.callback_draw_custom_window = [&]()
    {
        std::string title = optimization_running ? "Optimizer - Running" : "Optimizer";
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(10, 10),    ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 600), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Optimizer", nullptr,         ImGuiWindowFlags_NoSavedSettings);
        if (optimization_running) ImGui::StyleColorsLight();
        else                      ImGui::StyleColorsDark();

        ImGui::PushItemWidth(-80);

        if (optimization_running) {
            if (ImGui::Button("Request Stop", ImVec2(-1,0)))
                optimization_cancelled = true;
        }
        else {
            // Optimization settings are disabled when the optimization is running.
            double beta_min = 0;
            bool updated = false;
            if (ImGui::DragScalar("beta",  ImGuiDataType_Double, &lopt.beta,  10, &beta_min, 0, "%.4f")) {
                lopt.LO::setBeta(lopt.beta);
                updated = true;
            }
            double gamma_min = 0, gamma_max = 1, gamma = 0.1;
            if (ImGui::DragScalar("gamma", ImGuiDataType_Double, &gamma, 0.1, &gamma_min, &gamma_max, "%.4f")) {
                lopt.setGamma(gamma);
                updated = true;
            }

            double rlrw_min = 0, rlrw_max = 100, rl_regularization_weight = 0.1;
            if (ImGui::DragScalar("rl_regularization_weight", ImGuiDataType_Double, &rl_regularization_weight, 0.1, &rlrw_min, &rlrw_max, "%.4f")) {
                lopt.LO::setRestLengthMinimizationWeight(rl_regularization_weight);
                updated = true;
            }

            updated |= ImGui::InputScalar("Smooth min s",  ImGuiDataType_Double, &lopt.getMinAngleConstraint().s, 0, 0, "%.7f");
            if (ImGui::InputScalar("joint pos weight", ImGuiDataType_Double, &jointPosWeight, 0, 0, "%.7f")) {
                lopt.target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(deployedLinkage, jointPosWeight, jointPosValence2Multiplier);
                lopt.invalidateAdjointState();
                updated = true;
            }
            if (ImGui::InputScalar("valence 2 multiplier", ImGuiDataType_Double, &jointPosValence2Multiplier, 0, 0, "%.7f")) {
                lopt.target_surface_fitter.setTargetJointPosVsTargetSurfaceTradeoff(deployedLinkage, jointPosWeight, jointPosValence2Multiplier);
                lopt.invalidateAdjointState();
                updated = true;
            }

            if (ImGui::InputScalar("Equilibrium gradTol",  ImGuiDataType_Double, &eopts.gradTol, 0, 0, "%0.6e")) {
                lopt.setEquilibriumOptions(eopts);
            }
            ImGui::InputScalar("fd_eps",  ImGuiDataType_Double, &fd_eps, 0, 0, "%.3e");
            ImGui::InputScalar("numsteps",  ImGuiDataType_U64, &num_steps, 0, 0, "%i");
            ImGui::InputScalar("trust_region_scale",  ImGuiDataType_Double, &trust_region_scale, 0, 0, "%.7f");
            ImGui::Combo("Optimization Algorithm", (int *)(&opt_algorithm), "Newton CG\0BFGS\0\0");
            if (ImGui::Combo("Prediction Order", (int *)(&prediction_order), "Constant\0Linear\0Quadratic\0\0")) {
                lopt.prediction_order = prediction_order;
            }
            ImGui::InputScalar("Design optimization tol",  ImGuiDataType_Double, &designOptimizationTol, 0, 0, "%0.6e");

            ImGui::Checkbox("applyAngleConstraint", &applyAngleConstraint);
            ImGui::Checkbox("applyFlatnessConstraint", &applyFlatnessConstraint);
            if (ImGui::Checkbox("allowFlatActuation", &allowFlatActuation)) {
                lopt.setAllowFlatActuation(allowFlatActuation);
                updateLinkageMeshes();
                updated = true;
            }
            ImGui::InputScalar("# Loop Subdivisions",  ImGuiDataType_U64, &loop_subdivisions, 0, 0, "%i");
            ImGui::InputScalar("x scale factor",  ImGuiDataType_Double, &scale_x, 0, 0, "%.3e");
            ImGui::InputScalar("y scale factor",  ImGuiDataType_Double, &scale_y, 0, 0, "%.3e");
            ImGui::InputScalar("z scale factor",  ImGuiDataType_Double, &scale_z, 0, 0, "%.3e");
            ImGui::Checkbox("skip target surface construction", &skipTargetSurfaceConstruction);
            if (ImGui::Button("Set joint pos as target", ImVec2(-1,0))) {
                lopt.target_surface_fitter.joint_pos_tgt = deployedLinkage.jointPositions();
                if (!skipTargetSurfaceConstruction) {
                    lopt.constructTargetSurface(loop_subdivisions, 2, Eigen::Vector3d(scale_x, scale_y, scale_z));
                    updateTargetMesh();
                }
                updated = true;
            }
            if (ImGui::Button("Load Target Surface", ImVec2(-1,0))) {
                auto path = igl::file_dialog_open();
                if (!path.empty()) {
                    try {
                        lopt.target_surface_fitter.loadTargetSurface(deployedLinkage, path);
                        updateTargetMesh();
                        updated = true;
                    }
                    catch (std::exception & e) {
                        std::cout<<"Could not open target surface file!"<<std::endl;
                    }
                }
            }
            if (ImGui::Button("Reflect target surface", ImVec2(-1,0))) {
                lopt.target_surface_fitter.reflect(deployedLinkage, lopt.getRigidMotionConstrainedJoint());
                updateTargetMesh();
                updated = true;
            }
            if (ImGui::Button("Set deployment angle", ImVec2(-1,0))) {
                setting_deployment_angle = true;
            }

            if (updated) stats.update(lopt);
        }

        if (ImGui::Checkbox("Show target surface", &showTargetSurface)) {
            viewer.data(target_surface_mesh_id).set_visible(showTargetSurface, top_view);
            viewer.data(target_surface_mesh_id).dirty = true;
        }

        if (ImGui::Checkbox("Use rest kappa", &useRestKappa)) {
            bool deployedUseRestLen = deployedLinkage.getDesignParameterConfig().restLen;
            deployedLinkage.setDesignParameterConfig(deployedUseRestLen, useRestKappa);
            bool flatUseRestLen = flatLinkage.getDesignParameterConfig().restLen;
            flatLinkage.setDesignParameterConfig(flatUseRestLen, useRestKappa);
            lopt.setXShellOptimization(flatLinkage, deployedLinkage);

        }   

        ImGui::Text("Deployed avg angle: %f", stats.deployed_avg_angle);
        ImGui::Text("Closed avg angle: %f",   stats.closed_avg_angle);
        ImGui::Text("J: %f",                  stats.J);
        ImGui::Text("J_fit: %f",              stats.J_target);
        ImGui::Text("||grad J||: %f",         stats.gradp_J_norm);
        ImGui::Text("||grad J_fit||: %f",     stats.gradp_J_target_norm);
        ImGui::Text("E deployed: %e",         stats.E_deployed);
        ImGui::Text("E flat: %e",             stats.E_flat);
        ImGui::Text("Max_rod E deployed: %e", stats.E_deployed_rod_max);
        ImGui::Text("Closed min angle: %f",   stats.closed_min_angle);
        ImGui::Text("Smooth min angle: %f",   stats.closed_smin_angle);
        ImGui::Text("Flatness: %f",           stats.flatness);

        if (ImGui::Button("Save linkage data", ImVec2(-1,0))) {
            deployedLinkage.writeLinkageDebugData("deployed_opt.msh");
            flatLinkage.writeLinkageDebugData("flat_opt.msh");
            std::ofstream params_out("design_parameters.txt");
            if (!params_out.is_open()) throw std::runtime_error(std::string("Couldn't open output file ") + "design_parameters.txt");
            params_out.precision(19);
            params_out << lopt.params() << std::endl;
        }

        ImGui::PopItemWidth();

        ImGui::End();

        if (setting_deployment_angle) {
            // Define next window position + size
            ImGui::SetNextWindowPos(ImVec2(220, 10),   ImGuiSetCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
            ImGui::Begin("Deployment", nullptr,        ImGuiWindowFlags_NoSavedSettings);

            ImGui::InputScalar("Angle",  ImGuiDataType_Double, &target_deployment_angle, 0, 0, "%.7f");
            if (ImGui::Button("Apply", ImVec2(-1,0))) {
                setting_deployment_angle = false;
                open_linkage(deployedLinkage, target_deployment_angle, eopts, lopt.getRigidMotionConstrainedJoint());
                lopt.getLinesearchDeployedLinkage().set(deployedLinkage);
                lopt.setTargetAngle(target_deployment_angle);
                updateLinkageMeshes();
                stats.update(lopt);
            }
            if (ImGui::Button("Cancel", ImVec2(-1,0))) {
                setting_deployment_angle = false;
            }

            ImGui::End();
        }
    };

    viewer.launch(true, false, width, height);

    // Let the optimization finish its current iteration before exiting.
    optimization_cancelled = true;
    if (optimization_thread.joinable())
        optimization_thread.join();

    return EXIT_SUCCESS;
}
