#include "AverageAngleLinkage.hh"

template<typename Real_>
void AverageAngleLinkage_T<Real_>::m_constructAverageAngleToJointAngleMapTranspose() {
    // std::cout<<"Samara construct change of variable matrix!"<<std::endl;
    // Each joint has a single angle variable.
    const SuiteSparse_long m = Base::numJoints();
    const SuiteSparse_long n = Base::numJoints();
    const SuiteSparse_long nja = m_actuatedAngles.size();
    SuiteSparseMatrix result(m, n);
    result.nz = m + 2 * (nja - 1);
    // Now we fill out the transpose of the map one column (each joint angle) at a time:
    // # Average Angle Vars [            ]
    //                       # All Angles

    auto &Ax = result.Ax;
    auto &Ai = result.Ai;
    auto &Ap = result.Ap;

    Ax.reserve(result.nz);
    Ai.reserve(result.nz);
    Ap.push_back(0);
    size_t prevActAngleIndex = NONE;
    for (size_t ai = 0; ai < (size_t)n; ++ai) {
        if (isActuatedAngle(ai) && prevActAngleIndex != NONE) {
            Ai.push_back(prevActAngleIndex);
            Ax.push_back(-1);
            prevActAngleIndex = ai;
        }
            Ai.push_back(ai  );
            Ax.push_back( 1);
        if (isActuatedAngle(ai) && ai < m_lastActuatedAngle) {
            Ai.push_back(m_lastActuatedAngle);
            Ax.push_back( 1);
            prevActAngleIndex = ai;
        }
        Ap.push_back(Ai.size()); // col end.
    }

    assert(Ai.size() == size_t(result.nz));
    assert(Ap.size() == size_t(n+1      ));
    m_averageAngleToJointAngleMapTranspose = std::move(result);
}

template<typename Real_>
void AverageAngleLinkage_T<Real_>::m_constructAverageAngleToJointAngleMapTranspose_AllJointVars() {
    // Each joint has a single angle variable.
    size_t jointOffset = Base::dofOffsetForJoint(0);
    size_t nj = Base::numJoints();
    const SuiteSparse_long m = Base::dofOffsetForJoint(nj - 1) + Base::joint(nj - 1).numDoF() - jointOffset; // total number of joints DoFs
    const SuiteSparse_long n = m;
    const SuiteSparse_long nja = m_actuatedAngles.size();
    SuiteSparseMatrix result(m, n);
    result.nz = m + 2 * (nja - 1);
    // Now we fill out the transpose of the map one column (each joint angle) at a time:
    // # Average Angle Vars [            ]
    //                       # All Angles

    auto &Ax = result.Ax;
    auto &Ai = result.Ai;
    auto &Ap = result.Ap;

    Ax.reserve(result.nz);
    Ai.reserve(result.nz);
    Ap.push_back(0);
    size_t prevActAngleIndex = NONE;
    for (size_t ji = 0; ji < (size_t)n; ++ji) {
        if (isActuatedAngleVar(ji + jointOffset) && prevActAngleIndex != NONE) {
            size_t ai = getAngleIndexFromDoFIndex(ji + jointOffset);
            Ai.push_back(getDoFIndexFromAngleIndex(prevActAngleIndex) - jointOffset);
            Ax.push_back(-1);
            prevActAngleIndex = ai;
        }
        Ai.push_back(ji  );
        Ax.push_back( 1);
        if (isActuatedAngleVar(ji + jointOffset)) {
            size_t ai = getAngleIndexFromDoFIndex(ji + jointOffset);
            if (ai < m_lastActuatedAngle) {
                Ai.push_back(getDoFIndexFromAngleIndex(m_lastActuatedAngle) - jointOffset);
                Ax.push_back( 1);
                prevActAngleIndex = ai;
            }
        }
        Ap.push_back(Ai.size()); // col end.
    }

    assert(Ai.size() == size_t(result.nz));
    assert(Ap.size() == size_t(n+1      ));
    m_averageAngleToJointAngleMapTranspose_AllJointVars = std::move(result);
}

template<typename Real_>
void AverageAngleLinkage_T<Real_>::m_constructActuatedAngles(){
    if (m_freeAngles.size() == Base::numJoints()) throw std::runtime_error("Too many joints are left free.");
    m_actuatedAngles.clear();
    m_firstActuatedAngle = Base::numJoints();
    m_lastActuatedAngle = 0;
    for (size_t ji = 0; ji < Base::numJoints(); ji++){
        if (isActuatedAngle(ji)){
            m_actuatedAngles.push_back(ji);
            if (ji != NONE && ji < m_firstActuatedAngle) { m_firstActuatedAngle = ji; }
            if (ji != NONE && ji > m_lastActuatedAngle) { m_lastActuatedAngle = ji; }
        }
    }
    m_averageAngleIndex = getDoFIndexFromAngleIndex(m_lastActuatedAngle);
}


template<typename Real_>
void AverageAngleLinkage_T<Real_>::m_buildAngleIndexForDoFIndex() {
    m_angleIndexFromDoFIndex.resize(Base::numDoF());
    for (size_t i = 0; i < Base::numDoF(); ++i) m_angleIndexFromDoFIndex[i] = NONE;
    for (size_t ji = 0; ji < Base::numJoints(); ++ji) m_angleIndexFromDoFIndex[Base::dofOffsetForJoint(ji) + 6] = ji;
}

template<typename Real_>
size_t AverageAngleLinkage_T<Real_>::hessianNNZ(bool variableDesignParameters) const {
    size_t result = Base::hessianNNZ();
    // for (size_t ji = 0; ji < Base::numJoints() - 1; ++ji) {
    for (size_t ji = 0; ji < m_actuatedAngles.size() - 1; ++ji) {
        auto & j = Base::joint(m_actuatedAngles[ji]);
        auto & next_j = Base::joint(m_actuatedAngles[ji+1]);
        // The last average angle variable interacts with every other joint variables 
        // as well as the incident segments end edge variables that those joints control. 
        // The contribution of the last joint's average angle variable's interaction with its own variables are already counted. 
        result += j.numDoF() + j.valence() * 2 * (3 + 1); // per valence: 2 overlapping edges with 3 (position) + 1 (frame angle) DoFs
        // Every joint's angle variable interacts with the variables of the next joints as well as the incident segments end edge variables that those joints control. 
        result += next_j.numDoF() + next_j.valence() * 2 * (3 + 1);

        if (variableDesignParameters) {            
            throw std::runtime_error("variableDesignParameters is not implemented in hessianNNZ of AverageAngleLinkage!");
        }
    }
    // The contribution of the interaction between the last two angle variables is overcounted. 
    result -= 1;

    return result;
}

template<typename Real_>
auto AverageAngleLinkage_T<Real_>::hessianSparsityPattern(bool variableDesignParameters, Real_ val) const -> CSCMat {
    if (m_cachedHessianSparsity) return *m_cachedHessianSparsity;
    TMatrix result(Base::numDoF(), Base::numDoF()); 
    result.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;

    if (variableDesignParameters) { throw std::runtime_error("AAL hsp is not implemented for design parameters yet!"); }
    auto baseHSP = Base::hessianSparsityPattern(variableDesignParameters, 0.0);
    for (const auto t : baseHSP) { result.addNZ(t.i, t.j, 1.0); }

    // Add new entries representing the coupling between the transformed angle variables. 
    // for (size_t ji = 0; ji < Base::numJoints() - 1; ++ji) {
    for (size_t ji = 0; ji < m_actuatedAngles.size() - 1; ++ji) {
        const auto &j = Base::joint(m_actuatedAngles[ji]);
        const size_t jo = Base::dofOffsetForJoint(m_actuatedAngles[ji]);
        const auto &next_j = Base::joint(m_actuatedAngles[ji + 1]);
        const size_t next_jo = Base::dofOffsetForJoint(m_actuatedAngles[ji + 1]);

        // Add coupling between the current joint angle variable and the next joint's all variables + its segments vars. 
        for (size_t k = 0; k < next_j.numDoF(); ++k) result.addNZ(jo + 6, next_jo + k, 1.0);
        auto addIdx = [&](const size_t idx) { result.addNZ(idx, jo + 6, 1.0); };
        next_j.visitInfluencedSegmentVars(6, addIdx);

        // Add coupling between the average angle variable (stored in the last joint) and the current joint's all variables + its segments vars. 
        for (size_t k = 0; k < j.numDoF(); ++k) result.addNZ(jo + k, m_averageAngleIndex, 1.0);

        auto addSecondIdx = [&](const size_t idx) { result.addNZ(idx, m_averageAngleIndex, 1.0); };
        j.visitInfluencedSegmentVars(6, addSecondIdx);

    }

    if (variableDesignParameters) { 
        throw std::runtime_error("AAL hsp is not implemented for design parameters yet!");
        // m_cachedHessianVarRLSparsity = std::make_unique<CSCMat>(result); 
        // m_cachedHessianVarRLSparsity ->fill(val);
        // return *m_cachedHessianVarRLSparsity;
    } else { 
        m_cachedHessianSparsity      = std::make_unique<CSCMat>(result); 
        m_cachedHessianSparsity      ->fill(val);
        if (size_t(m_cachedHessianSparsity->nz) != hessianNNZ()) throw std::runtime_error("Incorrect NNZ prediction: " + std::to_string(m_cachedHessianSparsity->nz) + " vs " + std::to_string(hessianNNZ()));

        return *m_cachedHessianSparsity;
    }

    return *m_cachedHessianSparsity;
}



template<typename Real_>
void AverageAngleLinkage_T<Real_>::hessian(CSCMat &H, EnergyType eType, const bool variableDesignParameters) const {
    assert(H.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".hessian");
    if (variableDesignParameters) throw std::runtime_error("AvergaeAngleLinkage hessian is not implemented for variableDesignParameters!");
    // Compute Hessian using per-edge rest lengths
    auto baseH = Base::hessianSparsityPattern(variableDesignParameters);
    Base::hessian(baseH, eType, variableDesignParameters);
    const SuiteSparseMatrix &A = m_averageAngleToJointAngleMapTranspose_AllJointVars;

    std::vector<size_t> angleDoFIndices = Base::jointAngleDoFIndices();
    const size_t nj = Base::numJoints();
    const size_t jointOffset = Base::dofOffsetForJoint(0);
    const size_t njv = Base::dofOffsetForJoint(nj - 1) + Base::joint(nj - 1).numDoF() - jointOffset;

    // Copy the rods hessian over. 
    H.addWithSubSparsity(baseH, /* scale */ 1.0, /*  idx offset */ 0, /* block start */ 0, /* block end */ jointOffset);

    size_t hint = 0;
    for (size_t ji = 0; ji < njv; ++ji) {
        const size_t j = ji + jointOffset;

        // Loop over each output column "l" generated by angle variable "j"
        const size_t lend = A.Ap[ji + 1];
        for (size_t idx = A.Ap[ji]; idx < lend; ++idx) {
            const size_t l = A.Ai[idx] + jointOffset;
            const Real_ colMultiplier = A.Ax[idx];

            // Create entries for each input Hessian entry
            const size_t input_end = baseH.Ap[j + 1];
            for (size_t idx_in = baseH.Ap[j]; idx_in < input_end; ++idx_in) {
                const Real_ colVal = colMultiplier * baseH.Ax[idx_in];
                const size_t i = baseH.Ai[idx_in];
                if (i < jointOffset) { // left transformation is in the identity block
                    hint = H.addNZ(i, l, colVal, hint);
                }
                else {
                    // Loop over each output entry
                    const size_t jj = i - jointOffset;
                    size_t kprev = 0;
                    size_t kprev_idx = 0;
                    const size_t outrow_end = A.Ap[jj + 1];
                    for (size_t outrow_idx = A.Ap[jj]; outrow_idx < outrow_end; ++outrow_idx) {
                        const size_t k = A.Ai[outrow_idx] + jointOffset;
                        const Real_ val = A.Ax[outrow_idx] * colVal;
                        if (k <= l) {
                            // Accumulate entries from input's upper triangle
                            if (k == kprev) { H.addNZ(kprev_idx, val); }
                            else     { hint = H.addNZ(k, l, val, hint);
                                       kprev = k, kprev_idx = hint - 1; }
                        }
                        if ((i != j) && (l <= k)) H.addNZ(l, k, val); // accumulate entries from input's (strict) lower triangle
                    }
                }
            }
        }
    }

}

template<typename Real_>
void AverageAngleLinkage_T<Real_>::massMatrix(CSCMat &M, bool updatedSource, bool useLumped) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".massMatrix");
    assert(M.symmetry_mode == CSCMat::SymmetryMode::UPPER_TRIANGLE);

    // Compute Hessian using per-edge rest lengths
    auto baseM = Base::hessianSparsityPattern();
    Base::massMatrix(baseM, updatedSource, useLumped);
    const SuiteSparseMatrix &A = m_averageAngleToJointAngleMapTranspose_AllJointVars;

    std::vector<size_t> angleDoFIndices = Base::jointAngleDoFIndices();
    const size_t nj = Base::numJoints();
    const size_t jointOffset = Base::dofOffsetForJoint(0);
    const size_t njv = Base::dofOffsetForJoint(nj - 1) + Base::joint(nj - 1).numDoF() - jointOffset;

    // Copy the RodLinkage massMatrix over. 
    M.addWithSubSparsity(baseM, /* scale */ 1.0, /*  idx offset */ 0, /* block start */ 0, /* block end */ jointOffset);

    size_t hint = 0;
    for (size_t ji = 0; ji < njv; ++ji) {
        const size_t j = ji + jointOffset;

        // Loop over each output column "l" generated by angle variable "j"
        const size_t lend = A.Ap[ji + 1];
        for (size_t idx = A.Ap[ji]; idx < lend; ++idx) {
            const size_t l = A.Ai[idx] + jointOffset;
            const Real_ colMultiplier = A.Ax[idx];

            // Create entries for each input Hessian entry
            const size_t input_end = baseM.Ap[j + 1];
            for (size_t idx_in = baseM.Ap[j]; idx_in < input_end; ++idx_in) {
                const Real_ colVal = colMultiplier * baseM.Ax[idx_in];
                const size_t i = baseM.Ai[idx_in];
                if (i < jointOffset) { // left transformation is in the identity block
                    hint = M.addNZ(i, l, colVal, hint);
                }
                else {
                    // Loop over each output entry
                    const size_t jj = i - jointOffset;
                    size_t kprev = 0;
                    size_t kprev_idx = 0;
                    const size_t outrow_end = A.Ap[jj + 1];
                    for (size_t outrow_idx = A.Ap[jj]; outrow_idx < outrow_end; ++outrow_idx) {
                        const size_t k = A.Ai[outrow_idx] + jointOffset;
                        const Real_ val = A.Ax[outrow_idx] * colVal;
                        if (k <= l) {
                            // Accumulate entries from input's upper triangle
                            if (k == kprev) { M.addNZ(kprev_idx, val); }
                            else     { hint = M.addNZ(k, l, val, hint);
                                       kprev = k, kprev_idx = hint - 1; }
                        }
                        if ((i != j) && (l <= k)) M.addNZ(l, k, val); // accumulate entries from input's (strict) lower triangle
                    }
                }
            }
        }
    }

}


////////////////////////////////////////////////////////////////////////////////
// Explicit instantiation for ordinary double type and autodiff type.
////////////////////////////////////////////////////////////////////////////////
template struct AverageAngleLinkage_T<Real>;
template struct AverageAngleLinkage_T<ADReal>;
