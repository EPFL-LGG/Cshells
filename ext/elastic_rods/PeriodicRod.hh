////////////////////////////////////////////////////////////////////////////////
// PeriodicRod.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  This class represents a closed loop formed by gluing together the ends of
//  an elastic rod. To ensure stretching, bending, and twisting elastic
//  energies are calculated properly, the first and last ends of the rod are
//  constrained to overlap, and the stretching stiffness of this rod is halved
//  to avoid double-counting. An additional twist variable is introduced to allow a
//  twist discontinuity at the joint. This variable specifies the offset in frame
//  angle theta between the two overlapping edges (measured from the last edge
//  to the first). By fixing this angle to a constant, nonzero twist can be
//  maintained in the rod.
//
//  The ends are glued together with the following simple equality constraints on
//  the deformation variables:
//      x_{nv - 2}     = x_0
//      x_{nv - 1}     = x_1
//      theta_{ne - 1} = theta_0 - twist
//
//  We implement these constraints efficiently with a change of deformation
//  variables from "unreduced" variables
//      [x_0, ..., x_{nv - 1}, theta_0, ..., theta_{ne - 1}]
//  to "reduced" variables
//      [x_0, ..., x_{nv - 3}, theta_0, ..., theta_{ne - 2}, twist]
//
//  In matrix form, this linear change of variables looks like:
//                  [I_6 0             0 0           0][ x_0 \\ x_1              ]
//                  [0   I_{3(nv - 2)} 0 0           0][ x_2 .. x_{nv - 3}       ]
//  unreducedVars = [I_6 0             0 0           0][ theta_0                 ]
//                  [0   0             1 0           0][ theta_1..theta_{ne - 2} ]
//                  [0   0             0 I_{ne - 1}  0][ twist                   ]
//                  [0   0             1 0          -1] 
//                  \_______________ J ______________/ \______ reducedVars _____/
//  where `J` is the sparse Jacobian matrix of the change of variables consisting of
//  an arrangement of Identity blocks.
//
//  The periodic rod's elastic energy gradient is obtained by applying J^T to
//  the underlying rod's gradient.
//  The periodic rod's elastic energy Hessian is obtained from the underlying
//  rod's Hessian H as:
//      H_reduced = J^T H J
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/07/2021 13:42:56
////////////////////////////////////////////////////////////////////////////////
#ifndef PERIODICROD_HH
#define PERIODICROD_HH
#include "ElasticRod.hh"

// Templated to support automatic differentiation types.
template<typename Real_>
struct PeriodicRod_T;

using PeriodicRod = PeriodicRod_T<Real>;

template<typename Real_>
struct PeriodicRod_T {
    using Rod    = ElasticRod_T<Real_>;
    using Pt3    = Pt3_T<Real_>;
    using Vec2   = Vec2_T<Real_>;
    using VecX   = VecX_T<Real_>;
    using CSCMat = CSCMatrix<SuiteSparse_long, Real_>;
    using StdVectorVector2D = std::vector<Vec2, Eigen::aligned_allocator<Vec2>>; // Work around alignment issues.
    using EnergyType = typename Rod::EnergyType;

    PeriodicRod_T(const std::vector<Pt3> &points, bool zeroRestCurvature = false)
        : rod(points)
    {
        const size_t nv = rod.numVertices();
        if (((points[0] - points[nv - 2]).norm() > 1e-12) ||
            ((points[1] - points[nv - 1]).norm() > 1e-12)) throw std::runtime_error("First and last edge must overlap!");

        // Overwrite final edge's reference frame with the 
        auto restDirectors = rod.restDirectors();
        restDirectors.back() = restDirectors.front();
        rod.setRestDirectors(restDirectors);

        // Recompute rest curvature's decomposition in the updated frame (though it could simply be rotated...)
        // or reset it to zero if requested.
        StdVectorVector2D restKappa(nv, Vec2::Zero());
        if (!zeroRestCurvature) {
            for (size_t i = 1; i < nv - 1; ++i) {
                auto kb = curvatureBinormal(rod.restEdgeVector(i - 1).normalized(), rod.restEdgeVector(i).normalized());
                restKappa[i] = Vec2(0.5 * kb.dot(restDirectors[i - 1].d2 + restDirectors[i].d2),
                                   -0.5 * kb.dot(restDirectors[i - 1].d1 + restDirectors[i].d1));
            }
        }

        rod.setRestKappas(restKappa);
        rod.deformedConfiguration().initialize(rod);
    }

    // Converting constructor from another floating point type (e.g., double to autodiff)
    template<typename Real2>
    PeriodicRod_T(const PeriodicRod_T<Real2> &pr) : rod(pr.rod) { }

    // Set a homogeneous material for the rod
    void setMaterial(const RodMaterial &mat) {
        rod.setMaterial(mat);
        // Avoid double-counting stiffness/mass for the overlapping edge.
        rod.density(0) = 0.5;
        rod.density(rod.numEdges() - 1) = 0.5;
    }

    size_t numDoF()      const { return rod.numDoF() - 6; } // we remove the last two endpoint position variables.
    size_t thetaOffset() const { return 3 * (rod.numVertices() - 2); }

    VecX getDoFs() const {
        VecX result(numDoF());
        VecX unreducedDoFs = rod.getDoFs();
        result.head   (3 * (rod.numVertices() - 2))                     = unreducedDoFs.head   (3 * (rod.numVertices() - 2));
        result.segment(3 * (rod.numVertices() - 2), rod.numEdges() - 1) = unreducedDoFs.segment(3 * rod.numVertices(), rod.numEdges() - 1);
        result[result.size() - 1] = m_twist;
        return result;
    }

    VecX applyJacobian(const Eigen::Ref<const VecX> &dofs) const {
        if (size_t(dofs.size()) != numDoF()) throw std::runtime_error("DoF vector has incorrect length.");
        VecX unreducedDoFs(rod.numDoF());
        unreducedDoFs.head   (3 * (rod.numVertices() - 2))               = dofs.head   (3 * (rod.numVertices() - 2));
        unreducedDoFs.template segment<6>(3 * (rod.numVertices() - 2))   = dofs.template head<6>();
        unreducedDoFs.segment(3 * rod.numVertices(), rod.numEdges() - 1) = dofs.segment(3 * (rod.numVertices() - 2), rod.numEdges() - 1);
        unreducedDoFs[unreducedDoFs.size() - 1] = unreducedDoFs[rod.thetaOffset()] - m_twist;
        return unreducedDoFs;
    }

    void setDoFs(const Eigen::Ref<const VecX> &dofs) {
        m_twist = dofs[dofs.size() - 1];
        rod.setDoFs(applyJacobian(dofs));
    }

    Real_ twist()    const { return m_twist; }
    void setTwist(Real_ t) { m_twist = t; setDoFs(getDoFs()); }

    Real_ energy(EnergyType etype = EnergyType::Full) const { return rod.energy(etype); }
    Real_ energyStretch() const { return rod.energyStretch(); }
    Real_ energyBend()    const { return rod.energyBend(); }
    Real_ energyTwist()   const { return rod.energyTwist(); }
    VecX gradient(bool updatedSource = false, EnergyType etype = EnergyType::Full) const {
        auto unreducedGradient = rod.gradient(updatedSource, etype, /* variableDesignParameters = */ false, /* designParameterOnly = */ false);
        VecX result(numDoF());

        // Apply the transposed Jacobian
        // First two column blocks of J^T
        result.head   (3 * (rod.numVertices() - 2))                     = unreducedGradient.head   (3 * (rod.numVertices() - 2));
        result.segment(3 * (rod.numVertices() - 2), rod.numEdges() - 1) = unreducedGradient.segment(3 * rod.numVertices(), rod.numEdges() - 1);

        // Third column block of J^T
        result.template segment<6>(0) += unreducedGradient.template segment<6>(3 * (rod.numVertices() - 2));

        // Column blocks 4 and 5 of J^T
        Real_ gradLastTwist = unreducedGradient[unreducedGradient.size() - 1];

        // Column block 6 of J^T
        result[3 * (rod.numVertices() - 2)] +=  gradLastTwist;
        result[result.size() - 1]            = -gradLastTwist;

        return result;
    }

    // Hout = J^T H J. This has the effect of rewriting the
    // row/column indices for all triplets, except for the last row/column of
    // H, whose indices get duplicated into a +/- copy.
    template<class SPMat>
    void reduceHessian(const CSCMat &H, SPMat &Hout) const {
        const size_t nv = rod.numVertices();
        const size_t reducedPosVars = 3 * (nv - 2);
        const size_t unreducedPosVars = 3 * nv;
        const size_t unreducedVars = rod.numDoF();
        const size_t firstReducedTheta = reducedPosVars;

        // rewrite unreduced index i to its (first) corresponding reduced index
        auto reducedVarIdx = [&](size_t i) -> size_t { 
            if (i < reducedPosVars)    return i;
            if (i < unreducedPosVars)  return i - reducedPosVars; // first 6 displacement variables
            if (i < unreducedVars - 1) return i - unreducedPosVars + firstReducedTheta;
            return firstReducedTheta;
        };

        const size_t lastTheta = unreducedVars - 1;
        const size_t  twistVar = numDoF() - 1;

        auto emitNZ = [&](size_t i, size_t j, Real_ v) {
            if (i > j) return; // omit entries in the lower triangle
            Hout.addNZ(i, j, v);
        };

        for (const auto &t : H) {
            // Note: triplet `t` is in the upper triangle of H; we want to generate
            // the upper triangle of Hout = J^T H J.
            int ri = reducedVarIdx(t.i), rj = reducedVarIdx(t.j);
            emitNZ(ri, rj, t.v);
            if (t.i != t.j) {
                emitNZ(rj, ri, t.v);
                // Generate the extra triplets produced by the dependency of the
                // unreduced theta variable on m_twist.
                if (t.i == lastTheta) { emitNZ(twistVar, rj, -t.v); emitNZ(rj, twistVar, -t.v); }
                if (t.j == lastTheta) { emitNZ(twistVar, ri, -t.v); emitNZ(ri, twistVar, -t.v); }
            }
            else if (t.i == lastTheta) {
                // Generate the extra diagonal entry produced by the dependency of the
                // unreduced theta variable on m_twist.
                emitNZ(ri, twistVar, -t.v);
                emitNZ(twistVar, twistVar, t.v);
            }
        }
    }

    // Optimizers like Knitro and Ipopt need to know all Hessian entries that
    // could ever possibly be nonzero throughout the course of optimization.
    // The current Hessian may be missing some of these entries.
    // Knowing the fixed sparsity pattern also allows us to more efficiently construct the Hessian.
    CSCMat hessianSparsityPattern(Real_ val = 0.0) const {
        if (m_cachedHessianSparsityPattern.m == 0) {
            auto Hsp = rod.hessianSparsityPattern(false, 1.0);

            TripletMatrix<Triplet<Real_>> Htrip(numDoF(), numDoF());
            Htrip.symmetry_mode = TripletMatrix<Triplet<Real_>>::SymmetryMode::UPPER_TRIANGLE;
            Htrip.reserve(Hsp.nnz());
            reduceHessian(Hsp, Htrip);
            m_cachedHessianSparsityPattern = CSCMat(Htrip);
        }

        m_cachedHessianSparsityPattern.fill(val);
        return m_cachedHessianSparsityPattern;
    }

    void hessian(CSCMat &H, EnergyType etype = EnergyType::Full) const {
        CSCMat H_unreduced = m_getCachedUnreducedHessianSparsityPattern();
        rod.hessian(H_unreduced, etype, /* variableDesignParameters = */ false);

        H = hessianSparsityPattern(0.0);
        reduceHessian(H_unreduced, H);
    }

    CSCMat hessian(EnergyType etype = EnergyType::Full) const {
        CSCMat H;
        hessian(H, etype);
        return H;
    }

    // Note: the "lumped mass matrix" is not perfectly diagonal due to the "twist" variable's
    // coupling with the first theta variable.
    void massMatrix(CSCMat &M, bool updatedSource = false, bool useLumped = false) const {
        CSCMat M_unreduced = m_getCachedUnreducedHessianSparsityPattern();
        rod.massMatrix(M_unreduced, updatedSource, useLumped);

        M = hessianSparsityPattern(0.0);
        reduceHessian(M_unreduced, M);
    }

    CSCMat massMatrix() const {
        CSCMat M;
        massMatrix(M);
        return M;
    }

    // Additional methods required by compute_equilibrium
    void updateSourceFrame() { return rod.updateSourceFrame(); }
    void updateRotationParametrizations() { return rod.updateRotationParametrizations(); }
    Real_ characteristicLength() const { return rod.characteristicLength(); }
    Real_ initialMinRestLength() const { return rod.initialMinRestLength(); }
    std::vector<size_t> lengthVars(bool variableRestLen) const { return rod.lengthVars(variableRestLen); }
    Real_ approxLinfVelocity(const VecX &paramVelocity) const {
        return rod.approxLinfVelocity(applyJacobian(paramVelocity));
    }
    void writeDebugData(const std::string &path) const { rod.writeDebugData(path); }
    void saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames = false, const bool averagedCrossSections = false) const { rod.saveVisualizationGeometry(path, averagedMaterialFrames, averagedCrossSections); }

    Rod rod;
private:
    Real_ m_twist = 0.0;

    CSCMat &m_getCachedUnreducedHessianSparsityPattern() const {
        if (m_cachedUnreducedHessianSparsityPattern.m == 0)
            m_cachedUnreducedHessianSparsityPattern = rod.hessianSparsityPattern(0.0);
        return m_cachedUnreducedHessianSparsityPattern;
    }

    mutable CSCMat m_cachedHessianSparsityPattern,
                   m_cachedUnreducedHessianSparsityPattern;
};

#endif /* end of include guard: PERIODICROD_HH */
