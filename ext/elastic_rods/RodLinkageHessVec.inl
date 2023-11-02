////////////////////////////////////////////////////////////////////////////////
// RodLinkageHessVec.inl
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Elastic energy Hessian-vector product formulas for RodLinkage.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  12/09/2018 15:31:49
////////////////////////////////////////////////////////////////////////////////
#ifndef RODLINKAGEHESSVEC_INL
#define RODLINKAGEHESSVEC_INL

template<typename Real_>
struct RodHessianApplierData {
    VecX_T<Real_> v_local, Hv_local, Hv;
    bool constructed = false;
};

#if MESHFEM_WITH_TBB
template<typename Real_>
using RHALocalData = tbb::enumerable_thread_specific<RodHessianApplierData<Real_>>;

template<typename F, typename Real_>
struct RodHessianApplier {
    RodHessianApplier(F &f, const size_t nvars, RHALocalData<Real_> &locals) : m_f(f), m_nvars(nvars), m_locals(locals) { }

    void operator()(const tbb::blocked_range<size_t> &r) const {
        RodHessianApplierData<Real_> &data = m_locals.local();
        if (!data.constructed) { data.Hv.setZero(m_nvars); data.constructed = true; }
        for (size_t si = r.begin(); si < r.end(); ++si) { m_f(si, data); }
    }
private:
    F &m_f;
    size_t m_nvars;
    RHALocalData<Real_> &m_locals;
};

template<typename F, typename Real_>
RodHessianApplier<F, Real_> make_rod_hessian_applier(F &f, size_t nvars, RHALocalData<Real_> &locals) {
    return RodHessianApplier<F, Real_>(f, nvars, locals);
}
#endif

template<typename Real_>
auto RodLinkage_T<Real_>::applyHessian(const VecX &v, bool variableDesignParameters, const HessianComputationMask &mask) const -> VecX {
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".applyHessian");
    const size_t ndof = variableDesignParameters ? numExtendedDoF() : numDoF();
    if (size_t(v.size()) != ndof) throw std::runtime_error("Input vector size mismatch");

    // Our Hessian can only be evaluated after the source configuration has
    // been updated; use the more efficient gradient formulas.
    const bool updatedSource = true;
    {
        const bool hessianNeeded = mask.dof_in && mask.dof_out; // joint parametrization Hessian only needed for dof-dof part
        if (hessianNeeded) m_sensitivityCache.update(*this, updatedSource, v); // directional derivative only
        else               m_sensitivityCache.update(*this, updatedSource, false); // In all cases, we need at least the Jacobian
    }

    // Note: `mask.skipBRods` is a hack to compute derivatives of rivet forces;
    // it effectively detaches the joints from their local B segments (but
    // doesn't actually skip the whole "B segment" since a given segment could
    // be labeled "A" at one joint and "B" at the other.
    auto applyPerSegmentHessian = [&](const size_t si, RodHessianApplierData<Real_> &data) {
        VecX & v_local = data.v_local;
        VecX &Hv_local = data.Hv_local;
        VecX &Hv       = data.Hv;

        const auto &s = m_segments[si];
        const auto &r = s.rod;

        std::array<const LinkageTerminalEdgeSensitivity<Real_> *, 2> jointSensitivity{{ nullptr, nullptr }};
        std::array<size_t, 2> segmentJointDofOffset;
        for (size_t i = 0; i < 2; ++i) {
            size_t ji = s.joint(i);
            if (ji == NONE) continue;
            jointSensitivity[i] = &m_sensitivityCache.lookup(si, static_cast<TerminalEdge>(i));
            segmentJointDofOffset[i] = m_dofOffsetForJoint[ji];
        }

        const size_t ndof_local = variableDesignParameters ? s.rod.numExtendedDoF() : s.rod.numDoF();

        // Apply dv_dr (compute the perturbation of the rod variables).
        v_local.resize(ndof_local);
        if (mask.dof_in) {
            // Copy over the interior/free-end vertex and theta perturbations.
            const size_t free_vtx_components = 3 * s.numFreeVertices(),
                         local_theta_offset = s.rod.thetaOffset();
            v_local.segment((3 * 2) * s.hasStartJoint(),         free_vtx_components) = v.segment(m_dofOffsetForSegment[si], free_vtx_components);
            v_local.segment(local_theta_offset + s.hasStartJoint(), s.numFreeEdges()) = v.segment(m_dofOffsetForSegment[si] + free_vtx_components, s.numFreeEdges());

            // Compute the perturbations of the constrained vertex/theta variables.
            for (size_t lji = 0; lji < 2; ++lji) {
                size_t ji = s.joint(lji);
                if (ji == NONE) continue;
                const auto &js = *jointSensitivity[lji];
                const size_t jo = m_dofOffsetForJoint[ji];
                if (mask.skipBRods && !js.is_A) {
                    v_local.template segment<3>(3 * (js.j + 1)).setZero();
                    v_local.template segment<3>(3 * (js.j    )).setZero();
                    v_local[local_theta_offset + js.j] = 0.0;
                    continue;
                }

                Eigen::Matrix<Real_, 7, 1> delta_e_theta_n = js.jacobian * v.template segment<6>(jo + 3);
                v_local.template segment<3>(3 * (js.j + 1)) = v.template segment<3>(jo) + (js.s_jX * 0.5) * delta_e_theta_n.template segment<3>(0) + js.crossingNormalOffset * delta_e_theta_n.template segment<3>(4);
                v_local.template segment<3>(3 * (js.j    )) = v.template segment<3>(jo) - (js.s_jX * 0.5) * delta_e_theta_n.template segment<3>(0) + js.crossingNormalOffset * delta_e_theta_n.template segment<3>(4);
                v_local[local_theta_offset + js.j] = delta_e_theta_n[3];
            }
        }
        else { v_local.head(s.rod.numDoF()).setZero(); }
        if (variableDesignParameters) {
            if (mask.designParameter_in) {
                const size_t local_dp_offset = s.rod.designParameterOffset();
                if (m_linkage_dPC.restLen) {
                    // Copy over the interior/free-end edge rest length perturbations.
                    v_local.segment(local_dp_offset + s.hasStartJoint(), s.numFreeEdges())
                        = v.segment(m_restLenDofOffsetForSegment[si], s.numFreeEdges());

                    // Copy constrained terminal edges' rest length perturbations from their controlling joint.
                    for (size_t lji = 0; lji < 2; ++lji) {
                        size_t ji = s.joint(lji);
                        if (ji == NONE) continue;
                        const auto &js = *jointSensitivity[lji];
                        Real_ val = v[m_designParameterDoFOffsetForJoint[ji] + (js.is_A ? 0 : 1)];
                        if (mask.skipBRods && !js.is_A) val = 0.0;
                        v_local[local_dp_offset + js.j] = val;
                    }
                }

                if (m_linkage_dPC.restKappa) {
                    // Copy over all rest kappa variable perturbations.
                    v_local.segment(local_dp_offset + s.rod.numEdges() * m_linkage_dPC.restLen, s.rod.numRestKappaVars())
                        = v.segment(m_restKappaDofOffsetForSegment[si], s.rod.numRestKappaVars());
                }
            }
            else { v_local.tail(s.rod.numDesignParameters()).setZero(); }
        }

        // Apply rod Hessian
        Hv_local.setZero(ndof_local);
        r.applyHessEnergy(v_local, Hv_local, variableDesignParameters, mask);

        // Apply dv_dr transpose (accumulate contribution to output gradient)
        if (mask.dof_out) {
            // Copy over the interior/free-end vertex and theta delta grad components
            const size_t free_vtx_components = 3 * s.numFreeVertices(),
                         local_theta_offset = s.rod.thetaOffset();
            Hv.segment(m_dofOffsetForSegment[si],                    free_vtx_components) = Hv_local.segment((3 * 2) * s.hasStartJoint(), free_vtx_components);
            Hv.segment(m_dofOffsetForSegment[si] + free_vtx_components, s.numFreeEdges()) = Hv_local.segment(local_theta_offset + s.hasStartJoint(), s.numFreeEdges());

            // Compute the perturbations of the constrained vertex/theta variables.
            for (size_t lji = 0; lji < 2; ++lji) {
                size_t ji = s.joint(lji);
                if (ji == NONE) continue;
                const auto &js = *jointSensitivity[lji];
                const size_t jo = m_dofOffsetForJoint[ji];
                if (mask.skipBRods && !js.is_A) continue;

                Eigen::Matrix<Real_, 7, 1> delta_grad_e_theta;
                delta_grad_e_theta.template segment<3>(0) = (0.5 * js.s_jX) * (Hv_local.template segment<3>(3 * (js.j + 1)) - Hv_local.template segment<3>(3 * js.j));
                delta_grad_e_theta[3] = Hv_local[local_theta_offset + js.j];
                delta_grad_e_theta.template segment<3>(4) = js.crossingNormalOffset * (Hv_local.template segment<3>(3 * (js.j + 1)) + Hv_local.template segment<3>(3 * js.j));

                Hv.template segment<3>(jo    ) += Hv_local.template segment<3>(3 * (js.j + 1)) + Hv_local.template segment<3>(3 * js.j); // Joint position identity block
                Hv.template segment<6>(jo + 3) += js.jacobian.transpose() * delta_grad_e_theta;                                          // Joint orientation/angle/length Jacobian block
            }
        }
        if (variableDesignParameters && mask.designParameter_out) {
            const size_t local_dp_offset = s.rod.designParameterOffset();
            if (m_linkage_dPC.restLen) {
                // Copy over the interior/free-end edge rest length delta grad components.
                Hv.segment(m_restLenDofOffsetForSegment[si], s.numFreeEdges()) = Hv_local.segment(local_dp_offset + s.hasStartJoint(), s.numFreeEdges());

                // Accumulate constrained terminal edges' rest length delta grad components to their controlling joint.
                for (size_t lji = 0; lji < 2; ++lji) {
                    size_t ji = s.joint(lji);
                    if (ji == NONE) continue;
                    const auto &js = *jointSensitivity[lji];
                    if (mask.skipBRods && !js.is_A) continue;
                    Hv[m_designParameterDoFOffsetForJoint[ji] + (js.is_A ? 0 : 1)] += Hv_local[local_dp_offset + js.j];
                }
            }

            if (m_linkage_dPC.restKappa) {
                Hv.segment(m_restKappaDofOffsetForSegment[si], s.rod.numRestKappaVars()) = Hv_local.segment(local_dp_offset + s.rod.numEdges() * m_linkage_dPC.restLen, s.rod.numRestKappaVars());
            }
        }

        // Compute joint Hessian term.
        if (mask.dof_in && mask.dof_out) {
            // typename ElasticRod_T<Real_>::Gradient sg(r);
            // sg.setZero();
            // Note: we only need the gradient with respect to the terminal
            // degrees of freedom, so we can ignore many of the energy contributions.
            const auto sg = r.template gradient<GradientStencilMaskTerminalsOnly>(updatedSource); // we never need the variable rest length gradient since the mapping from global to local rest lengths is linear

            // Accumulate contribution of the Hessian of e^j and theta^j wrt the joint parameters.
            //      dE/var^j (d^2 var^j / djoint_var_k djoint_var_l)
            for (size_t ji = 0; ji < 2; ++ji) {
                if (jointSensitivity[ji] == nullptr) continue;
                const auto &js = *jointSensitivity[ji];
                if (mask.skipBRods && !js.is_A) continue;
                const size_t o = segmentJointDofOffset[ji] + 3; // DoF index for first component of omega
                Eigen::Matrix<Real_, 7, 1> dE_djointvar;
                dE_djointvar.template segment<3>(0) = (0.5 * js.s_jX) * (sg.gradPos(js.j + 1) - sg.gradPos(js.j));
                dE_djointvar[3]                     = sg.gradTheta(js.j);
                dE_djointvar.template segment<3>(4) = js.crossingNormalOffset * (sg.gradPos(js.j + 1) + sg.gradPos(js.j));
                Hv.template segment<6>(o).noalias() += js.delta_jacobian.transpose() * dE_djointvar;
            }
        }
    };

    VecX result(VecX::Zero(v.size()));
#if MESHFEM_WITH_TBB
    RHALocalData<Real_> rhaLocalData;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numSegments()), make_rod_hessian_applier(applyPerSegmentHessian, v.size(), rhaLocalData));

    for (const auto &data : rhaLocalData)
        result += data.Hv;
#else
    RodHessianApplierData<Real_> data;
    data.Hv.setZero(result.size());
    for (size_t si = 0; si < numSegments(); ++si)
        applyPerSegmentHessian(si, data);
    result = data.Hv;
#endif

    if ((mask.dof_in && mask.dof_out) && (m_angleBoundEnforcement == AngleBoundEnforcement::Penalty)) {
        visitAngleBounds([&](size_t ji, Real_ lower, Real_ upper) {
                size_t var = m_dofOffsetForJoint[ji] + 6;
                result[var] += m_constraintBarrier.d2eval(joint(ji).alpha(), lower, upper) * v[var];
            });
    }

    return result;
}

template<typename Real_>
auto RodLinkage_T<Real_>::applyHessianPerSegmentRestlen(const VecX &v, const HessianComputationMask &mask) const -> VecX {
    BENCHMARK_SCOPED_TIMER_SECTION timer(mangledName() + ".applyHessianPSRL");
    const size_t ndof = numExtendedDoFPSRL();
    if (size_t(v.size()) != ndof) throw std::runtime_error("Input vector size mismatch");

    VecX vPerEdge(numExtendedDoF());
    size_t unchanged_length = numDoF() + numRestKappaVars() * m_linkage_dPC.restKappa;
    vPerEdge.head(unchanged_length) = v.head(unchanged_length);
    if (mask.designParameter_in && m_linkage_dPC.restLen) m_segmentRestLenToEdgeRestLenMapTranspose.applyRaw(v.tail(numSegments()).data(), vPerEdge.tail(numRestLengths()).data(), /* transpose */ true);
    else if (m_linkage_dPC.restLen)                       vPerEdge.tail(numRestLengths()).setZero();
    auto HvPerEdge = applyHessian(vPerEdge, true, mask);

    VecX result(v.size());
    result.head(unchanged_length) = HvPerEdge.head(unchanged_length);
    if (mask.designParameter_out && m_linkage_dPC.restLen) m_segmentRestLenToEdgeRestLenMapTranspose.applyRaw(HvPerEdge.tail(numRestLengths()).data(), result.tail(numSegments()).data(), /* no transpose */ false);
    else if (m_linkage_dPC.restLen)                        result.tail(numSegments()).setZero();

    return result;
}

#endif /* end of include guard: RODLINKAGEHESSVEC_INL */
