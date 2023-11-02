////////////////////////////////////////////////////////////////////////////////
// TargetSurfaceFitter.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implementation of a target surface to which points are fit using the
//  distance to their closest point projections.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  01/02/2019 11:51:11
////////////////////////////////////////////////////////////////////////////////
#ifndef TARGETSURFACEFITTER_HH
#define TARGETSURFACEFITTER_HH

#include "RodLinkage.hh"
#include "compute_equilibrium.hh"

// Forward declare mesh data structure for holding the target surface (to avoid bringing in MeshFEM::TriMesh when unnecessary)
struct TargetSurfaceMesh;
// Forward declare AABB data structure
struct TargetSurfaceAABB;

struct TargetSurfaceFitter {
    using VecX   = VecX_T<Real>;
    using Mat3   = Mat3_T<Real>;

    TargetSurfaceFitter(); // Needed because target_surface is a smart pointer to an incomplete type.
    TargetSurfaceFitter(const TargetSurfaceFitter &tsf) { *this = tsf; }

    // Update the closest points regardless of `holdClosestPointsFixed`
    template<typename Real_>
    void forceUpdateClosestPoints(const RodLinkage_T<Real_> &linkage);

    void loadTargetSurface(const RodLinkage &linkage, const std::string &path);
    void saveTargetSurface(const std::string &path);

    TargetSurfaceFitter &operator=(const TargetSurfaceFitter &tsf) {
        // Copy *all* public variables
        W_diag_joint_pos              = tsf.W_diag_joint_pos;
        Wsurf_diag_linkage_sample_pos = tsf.Wsurf_diag_linkage_sample_pos;
        joint_pos_tgt                 = tsf.joint_pos_tgt;

        target_surface                        = tsf.target_surface;
        linkage_closest_surf_pts              = tsf.linkage_closest_surf_pts;
        linkage_closest_surf_pt_sensitivities = tsf.linkage_closest_surf_pt_sensitivities;
        linkage_closest_surf_tris             = tsf.linkage_closest_surf_tris;
        holdClosestPointsFixed                = tsf.holdClosestPointsFixed;

        // Copy *all* private variables
        m_tgt_surf_aabb_tree = tsf.m_tgt_surf_aabb_tree;
        m_tgt_surf_V         = tsf.m_tgt_surf_V;
        m_tgt_surf_F         = tsf.m_tgt_surf_F;
        m_tgt_surf_N         = tsf.m_tgt_surf_N;
        m_tgt_surf_VN        = tsf.m_tgt_surf_VN;
        m_useCenterline      = tsf.m_useCenterline;

        return *this;
    }

    // Reflect the surface across the plane defined by joint "ji"'s position and normal.
    // This is useful in case the surface buckled in the opposite direction from the
    // loaded target surface.
    void reflect(const RodLinkage &linkage, size_t ji) {
        Point3D p = linkage.joint(ji).pos();
        Vector3D n = linkage.joint(ji).normal();
        for (int i = 0; i < m_tgt_surf_V.rows(); ++i) {
            Vector3D v = m_tgt_surf_V.row(i).transpose() - p;
            v -= (2 * n.dot(v)) * n;
            m_tgt_surf_V.row(i) = (p + v).transpose();
        }
        // Also reverse the orientation of each triangle (to flip the normals for proper visualization lighting)
        for (int i = 0; i < m_tgt_surf_F.rows(); ++i)
            std::swap(m_tgt_surf_F(i, 0), m_tgt_surf_F(i, 1));

        setTargetSurface(linkage, m_tgt_surf_V, m_tgt_surf_F);

        // Update the target joints positions too
        const size_t nj = linkage.numJoints();
        for (size_t ji = 0; ji < nj; ++ji) {
            Vector3D v = joint_pos_tgt.segment<3>(3 * ji) - p;
            v -= (2 * n.dot(v)) * n;
            joint_pos_tgt.segment<3>(3 * ji) = (p + v).transpose();
        }
    }

    const Eigen::MatrixXd &getV() const { return m_tgt_surf_V; }
    const Eigen::MatrixXi &getF() const { return m_tgt_surf_F; }
    const Eigen::MatrixXd &getN() const { return m_tgt_surf_N; }

    Real objective(const RodLinkage &linkage) const {
        VecX jpos = linkage.jointPositions();
        VecX projectionQueries = m_useCenterline ? linkage.centerLinePositions() : jpos;

        Eigen::VectorXd jointPosDiff = jpos - joint_pos_tgt;
        Eigen::VectorXd surfLinkageSamplePosDiff = projectionQueries - linkage_closest_surf_pts;
        return 0.5 * (jointPosDiff.dot(W_diag_joint_pos.cwiseProduct(jointPosDiff)) +
                      surfLinkageSamplePosDiff.dot(Wsurf_diag_linkage_sample_pos.cwiseProduct(surfLinkageSamplePosDiff)));
    }

    // Gradient with respect to the query points
    // Note: for the closest point projection term, this gradient expression assumes all components of a query point's
    // weight vector are equal; otherwise the dP/dx expression does not vanish
    // and we need more derivatives of the closest point projection (envelope theorem no longer applies)!
    VecX gradient(const RodLinkage &linkage) const {
        VecX jpos = linkage.jointPositions();
        VecX projectionQueries = m_useCenterline ? linkage.centerLinePositions() : jpos;
        return m_apply_W(linkage, jpos - joint_pos_tgt) + m_apply_Wsurf(linkage, projectionQueries - linkage_closest_surf_pts);
    }

    // More efficient implementation of `gradient` that avoids multiple memory allocations.
    void accumulateGradient(const RodLinkage &linkage, VecX &result, Real weight) const {
        if (!m_useCenterline) {
            const size_t nj = linkage.numJoints();
            for (size_t ji = 0; ji < nj; ++ji) {
                const auto &pos = linkage.joint(ji).pos();
                result.segment<3>(linkage.dofOffsetForJoint(ji)) += weight * (             W_diag_joint_pos.segment<3>(3 * ji).cwiseProduct(pos -            joint_pos_tgt.segment<3>(3 * ji))
                                                                            + Wsurf_diag_linkage_sample_pos.segment<3>(3 * ji).cwiseProduct(pos - linkage_closest_surf_pts.segment<3>(3 * ji)));
            }
        }
        else {
            const size_t nj = linkage.numJoints();
            for (size_t ji = 0; ji < nj; ++ji)
                result.segment<3>(linkage.dofOffsetForJoint(ji)) += weight * (W_diag_joint_pos.segment<3>(3 * ji).cwiseProduct(linkage.joint(ji).pos() - joint_pos_tgt.segment<3>(3 * ji)));

            size_t cpi = 0; // global centerline point index
            for (size_t si = 0; si < linkage.numSegments(); ++si) {
                const auto &s = linkage.segment(si);
                const auto &r = s.rod;
                const size_t end = r.numVertices() - 2 * s.hasEndJoint();
                for (size_t i = 2 * s.hasStartJoint(); i < end; ++i) {
                    result.segment<3>(linkage.dofOffsetForCenterlinePos(cpi)) +=
                            weight * (Wsurf_diag_linkage_sample_pos.segment<3>(3 * cpi).cwiseProduct(r.deformedPoint(i) - linkage_closest_surf_pts.segment<3>(3 * cpi)));
                    ++cpi;
                }
            }
        }
    }

    // Hessian with respect to query point "vi"
    Mat3 pt_project_hess(size_t vi) const {
        if (holdClosestPointsFixed) return Wsurf_diag_linkage_sample_pos.segment<3>(vi * 3).asDiagonal();
        // Note: we must assume Wsurf_diag_linkage_sample_pos.segment<3>(vi * 3) is a multiple of [1, 1, 1], otherwise
        // the true Hessian will require second derivatives of the closest point projection (envelope theorem no longer applies).
        return Wsurf_diag_linkage_sample_pos[vi * 3] * (Mat3::Identity() - linkage_closest_surf_pt_sensitivities[vi]);
    }

    Mat3 pt_tgt_hess(size_t vi) const { return W_diag_joint_pos.segment<3>(vi * 3).asDiagonal(); }

    template<class RL_>
    typename RL_::VecX applyHessian(const RL_ &linkage, Eigen::Ref<const typename RL_::VecX> delta_xp) const {
        using VXd = typename RL_::VecX;
        VXd result = VXd::Zero(linkage.numExtendedDoFPSRL());

        // Hessian of the surface projection term
        const size_t nsp = numSamplePoints(linkage);
        for (size_t spi = 0; spi < nsp; ++spi) {
            const size_t spo = dofOffsetForSamplePoint(linkage, spi);
            result.template segment<3>(spo) += pt_project_hess(spi) * delta_xp.template segment<3>(spo);
        }

        // Hessian of the target joint position term.
        //      0.5 * ||x - x^tgt||^2_W --> W delta_x
        const size_t nj = linkage.numJoints();
        for (size_t ji = 0; ji < nj; ++ji) {
            const size_t jdo = linkage.dofOffsetForJoint(ji);
            result.template segment<3>(jdo) += W_diag_joint_pos.segment<3>(ji * 3).asDiagonal() * delta_xp.template segment<3>(jdo);
        }

        return result;
    }

    // Adjoint solve for the target fitting objective on the deployed linkage
    //      [H_3D a][w_x     ] = [b]   or    H_3D w_x = b
    //      [a^T  0][w_lambda]   [0]
    // where b = W * (x_3D - x_tgt) + Wsurf * (x_3D - p(x_3D))
    // depending on whether average angle actuation is applied.
    Eigen::VectorXd adjoint_solve(const RodLinkage &linkage, NewtonOptimizer &opt) const {
        Eigen::VectorXd b = gradient(linkage);
        if (opt.get_problem().hasLEQConstraint())
            return opt.extractFullSolution(opt.kkt_solver(opt.solver, opt.removeFixedEntries(b)));
        return opt.extractFullSolution(opt.solver.solve(opt.removeFixedEntries(b)));
    }

    // Solve for the change in adjoint state induced by a perturbation of the equilibrium state delta_x (and possibly the structure's design parameters p):
    //                                                                                                d3E_w
    //                                                                              _____________________________________________
    //                                                                             /                                             `.
    //      [H_3D a][delta w_x     ] = [W delta_x + W_surf (I - dp_dx) delta_x ] - [d3E/dx dx dx delta_x + d3E/dx dx dp delta_p] w
    //      [a^T  0][delta w_lambda]   [               0                       ]   [                     0                     ]
    //                                 \_________________________________________________________________________________________/
    //                                                                           b
    // Note that this equation is for when an average angle actuation is applied. If not, then the last row/column of the system is removed.
    Eigen::VectorXd delta_adjoint_solve(const RodLinkage &linkage, NewtonOptimizer &opt, const Eigen::VectorXd &delta_x, const Eigen::VectorXd &d3E_w) const {
        Eigen::VectorXd target_surf_term(Wsurf_diag_linkage_sample_pos.size());
        const size_t nsp = numSamplePoints(linkage);
        for (size_t spi = 0; spi < nsp; ++spi) {
            Vector3D dx = delta_x.segment<3>(dofOffsetForSamplePoint(linkage, spi));
            if (holdClosestPointsFixed)
                target_surf_term.segment<3>(3 * spi) = dx;
            else
                target_surf_term.segment<3>(3 * spi) = dx - linkage_closest_surf_pt_sensitivities[spi] * dx;
        }

        auto b = (m_unpackJointPositions(linkage, W_diag_joint_pos).cwiseProduct(delta_x)
                + m_apply_Wsurf(linkage, target_surf_term)
                - d3E_w.head(delta_x.size())).eval();

        if (opt.get_problem().hasLEQConstraint())
            return opt.extractFullSolution(opt.kkt_solver(opt.solver, opt.removeFixedEntries(b)));
        return opt.extractFullSolution(opt.solver.solve(opt.removeFixedEntries(b)));
    }

    void constructTargetSurface(const RodLinkage &linkage, size_t loop_subdivisions = 0, size_t num_extension_layers = 1, Eigen::Vector3d scale_factors = Eigen::Vector3d(1, 1, 1));
    void setTargetSurface(const RodLinkage &linkage, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
    template<typename Real_>
    void updateClosestPoints(const RodLinkage_T<Real_> &linkage) {
        if (!holdClosestPointsFixed)
            forceUpdateClosestPoints(linkage);
    }

    // Mostly for python bindings
    Eigen::MatrixXd getTargetSurfaceVertices() const { return m_tgt_surf_V; }
    Eigen::MatrixXi getTargetSurfaceFaces()    const { return m_tgt_surf_F; }
    Eigen::MatrixXd getTargetSurfaceNormals()  const { return m_tgt_surf_N; }

    template<typename Real_>
    void setHoldClosestPointsFixed(bool hold, const RodLinkage_T<Real_> &linkage) {holdClosestPointsFixed = hold; updateClosestPoints(linkage) ;}

    // Reset the target fitting weights to penalize all joints' deviations
    // equally and control the trade-off between fitting to the individual,
    // fixed joint targets and fitting to the target surface.
    // If valence2Multiplier > 1 we attempt to fit the valence 2 joints more closely
    // to their target positions than the rest of the joints.
    template<typename Real_>
    void setTargetJointPosVsTargetSurfaceTradeoff(const RodLinkage_T<Real_> &linkage, Real jointPosWeight, Real valence2Multiplier = 1.0) {
        // Given the valence 2 vertices a valence2Multiplier times higher weight for fitting to their target positions.
        // (But leave the target surface fitting weights uniform).
        const size_t nj = linkage.numJoints();
        size_t numValence2 = 0;
        for (const auto &j : linkage.joints())
            numValence2 += (j.valence() == 2);

        size_t numNonValence2 = nj - numValence2;
        Real nonValence2Weight = 1.0 / (3.0 * (numValence2 * valence2Multiplier + numNonValence2));
        size_t numJointPosComponents = 3 * nj;
        W_diag_joint_pos.resize(numJointPosComponents);
        for (size_t ji = 0; ji < nj; ++ji)
            W_diag_joint_pos.segment<3>(3 * ji).setConstant(jointPosWeight * nonValence2Weight * ((linkage.joint(ji).valence() == 2) ? valence2Multiplier : 1.0));

        size_t nsc = 3 * numSamplePoints(linkage);
        Wsurf_diag_linkage_sample_pos = Eigen::VectorXd::Constant(nsc, (1.0 - jointPosWeight) / nsc);
    }

    template<typename Real_>
    void scaleJointWeights(const RodLinkage_T<Real_> &linkage, Real jointPosWeight, Real featureMultiplier = 1.0, const std::vector<size_t> &additional_feature_pts = std::vector<size_t>()) {
        // Given the valence 2 vertices a valence2Multiplier times higher weight for fitting to their target positions.
        // (But leave the target surface fitting weights uniform).
        // TODO: should have a safeguard for joints in additional_feature_pts and with valence 2
        const size_t nj = linkage.numJoints();
        size_t numValence2 = 0;
        for (const auto &j : linkage.joints())
            numValence2 += (j.valence() == 2);

        size_t numFeatures = numValence2 + additional_feature_pts.size();
        size_t numNonFeatures = nj - numFeatures;
        Real nonFeatureWeight = 1.0 / (3.0 * (numFeatures * featureMultiplier + numNonFeatures));
        size_t numJointPosComponents = 3 * nj;
        W_diag_joint_pos.resize(numJointPosComponents);
        for (size_t ji = 0; ji < nj; ++ji){
            W_diag_joint_pos.segment<3>(3 * ji).setConstant(jointPosWeight * nonFeatureWeight * (((linkage.joint(ji).valence() == 2) || (std::find(additional_feature_pts.begin(), additional_feature_pts.end(), ji) != additional_feature_pts.end())) ? featureMultiplier : 1.0));
        }

        size_t nsc = 3 * numSamplePoints(linkage);
        Wsurf_diag_linkage_sample_pos = Eigen::VectorXd::Constant(nsc, (1.0 - jointPosWeight) / nsc);

        forceUpdateClosestPoints(linkage);
    }

    template<typename Real_>
    void scaleFeatureJointWeights(const RodLinkage_T<Real_> &linkage, Real jointPosWeight, Real featureMultiplier = 1.0, const std::vector<size_t> &feature_pts = std::vector<size_t>()) {
        // Same as before but ignores valence 2 joints
        // (Leaves the target surface fitting weights uniform).
        const size_t nj = linkage.numJoints();
        
        // First the target joints positions weights
        size_t numFeatures = feature_pts.size();
        size_t numNonFeatures = nj - numFeatures;
        Real nonFeatureWeight = 1.0 / (3.0 * (numFeatures * featureMultiplier + numNonFeatures));
        size_t numJointPosComponents = 3 * nj;
        W_diag_joint_pos.resize(numJointPosComponents);
        for (size_t ji = 0; ji < nj; ++ji){
            W_diag_joint_pos.segment<3>(3 * ji).setConstant(
                jointPosWeight * nonFeatureWeight * 
                ((std::find(feature_pts.begin(), feature_pts.end(), ji) != feature_pts.end()) ? featureMultiplier : 1.0)
            );
        }

        // Then the target surface fitting weights
        size_t nsc = 3 * numSamplePoints(linkage);
        Wsurf_diag_linkage_sample_pos = Eigen::VectorXd::Constant(nsc, (1.0 - jointPosWeight) / nsc);

        forceUpdateClosestPoints(linkage);
    }

    std::vector<Real> get_squared_distance_to_target_surface(Eigen::VectorXd query_point_list) const;
    Eigen::VectorXd get_closest_point_for_visualization(Eigen::VectorXd query_point_list) const;
    Eigen::VectorXd get_closest_point_normal(Eigen::VectorXd query_point_list);

    bool getUseCenterline() const { return m_useCenterline; }

    template<typename Real_>
    void  setUseCenterline(const RodLinkage_T<Real_> &linkage, bool useCenterline, double jointPosWeight, double jointPosValence2Multiplier = 1.0) {
        if (m_useCenterline != useCenterline) {
            m_useCenterline = useCenterline;
            // We are changing the number of sample points; must re-initialize the weights array or
            // we risk crashing due to out-of-bounds access.
            setTargetJointPosVsTargetSurfaceTradeoff(linkage, jointPosWeight, jointPosValence2Multiplier);
            forceUpdateClosestPoints(linkage);
        }
    }

    void setTargetJointsPositions(Eigen::VectorXd newJointPosTgt) {
        joint_pos_tgt =  newJointPosTgt;
    }

    template<typename Real_>
    size_t numSamplePoints(const RodLinkage_T<Real_> &linkage) const { return m_useCenterline ? linkage.numCenterlinePos() : linkage.numJoints(); }
    template<typename Real_>
    size_t dofOffsetForSamplePoint(const RodLinkage_T<Real_> &linkage, size_t spi) const { return m_useCenterline ? linkage.dofOffsetForCenterlinePos(spi)
                                                                                                                  : linkage.dofOffsetForJoint(spi); }

    ~TargetSurfaceFitter(); // Needed because target_surface is a smart pointer to an incomplete type.

private:
    // Apply the joint position weight matrix W to a compressed state vector that
    // contains only variables corresponding to joint positions.
    // Returns an uncompressed vector with an entry for each state variable.
    Eigen::VectorXd m_apply_W    (const RodLinkage &linkage, const Eigen::Ref<const Eigen::VectorXd> &x_joint_pos) const { return m_unpackJointPositions(linkage, W_diag_joint_pos.cwiseProduct(x_joint_pos)); }
    Eigen::VectorXd m_apply_Wsurf(const RodLinkage &linkage, const Eigen::Ref<const Eigen::VectorXd> &x_sample_point_pos) const {
        auto weighted_sample_point_pos = Wsurf_diag_linkage_sample_pos.cwiseProduct(x_sample_point_pos);
        return m_useCenterline ?
                m_unpackCenterlinePositions(linkage, weighted_sample_point_pos) :
                m_unpackJointPositions(linkage, weighted_sample_point_pos);
    }
    // Extract a full state vector from a compressed version that only holds
    // variables corresponding to joint positions.
    Eigen::VectorXd m_unpackJointPositions(const RodLinkage &linkage, const Eigen::Ref<const Eigen::VectorXd> &x_joint_pos) const {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(linkage.numDoF());

        for (size_t ji = 0; ji < linkage.numJoints(); ++ji)
            result.segment<3>(linkage.dofOffsetForJoint(ji)) = x_joint_pos.segment<3>(3 * ji);
        return result;
    }

    // Extract a full state vector from a compressed version that only holds
    // variables corresponding to centerline positions.
    Eigen::VectorXd m_unpackCenterlinePositions(const RodLinkage &linkage, const Eigen::Ref<const Eigen::VectorXd> &x_centerline_pos) const {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(linkage.numDoF());
        for (size_t cli = 0; cli < linkage.numCenterlinePos(); ++cli)
            result.segment<3>(linkage.dofOffsetForCenterlinePos(cli)) = x_centerline_pos.segment<3>(3 * cli);
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Private memeber variables
    ////////////////////////////////////////////////////////////////////////////
    // Target surface to which the deployed joints are fit.
    std::shared_ptr<TargetSurfaceAABB> m_tgt_surf_aabb_tree;
    Eigen::MatrixXd m_tgt_surf_V, m_tgt_surf_N, m_tgt_surf_VN;
    Eigen::MatrixXi m_tgt_surf_F;

    bool m_useCenterline = false;
public:
    ////////////////////////////////////////////////////////////////////////////
    // Public member variables
    ////////////////////////////////////////////////////////////////////////////
    // Fitting weights
    Eigen::VectorXd W_diag_joint_pos,       // compressed version of W from the writeup: only include weights corresponding to joint position variables.
                    Wsurf_diag_linkage_sample_pos;   // Similar to above, the weights for fitting each joint to its closest point on the surface.
                                            // WARNING: if this is changed from zero to a nonzero value, the joint_closest_surf_pts will not be updated
                                            // until the next equilibrium solve.
    Eigen::VectorXd joint_pos_tgt; // compressed, flattened version of x_tgt from the writeup: only include the joint position variables

    std::shared_ptr<TargetSurfaceMesh> target_surface;                // halfedge structure storing the target surface.
    Eigen::VectorXd linkage_closest_surf_pts;                           // compressed, flattened version of p(x)  from the writeup: only include the joint position variables
    std::vector<Eigen::Matrix3d> linkage_closest_surf_pt_sensitivities; // dp_dx(x) from writeup (sensitivity of closest point projection)
    std::vector<int> linkage_closest_surf_tris;                         // for debugging: index of the closest triangle to each joint.
    bool holdClosestPointsFixed = false;
};

#endif /* end of include guard: TARGETSURFACEFITTER_HH */
