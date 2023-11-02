////////////////////////////////////////////////////////////////////////////////
// RegularizationTerms.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Regularization terms for the DesignParameterSolve/weaving optimization:
//     - Laplacian regularization to smooth the rest curvatures
//     - Total rest length of the structure
*///////////////////////////////////////////////////////////////////////////////
#ifndef REGULARIZATIONTERMS_HH
#define REGULARIZATIONTERMS_HH

#include "ElasticRod.hh"
#include "RodLinkage.hh"
#include "SurfaceAttractedLinkage.hh"

template<typename Object>
struct RegularizationTerm {
    static_assert(std::is_same<typename Object::RealType, Real>::value,
                  "Regularization terms currently assume plain `double` number type.");
    using CSCMat = CSCMatrix<SuiteSparse_long, Real>;
    using VecX   = Eigen::VectorXd;

    RegularizationTerm(const Object &obj) : m_object(obj) { }

    Real unweightedEnergy() const {
        return m_unweightedEnergy();
    }

    // Weighted energy
    Real energy() const {
        if (!enabled()) return 0.0;
        return weight * m_unweightedEnergy();
    }

    // Add weighted energy's gradient to g
    void accumulateGradient(Eigen::Ref<VecX> g) const {
        if (enabled()) m_accumulateGradient(g);
    }

    // Add weighted energy's Hessian to H
    void accumulateHessian(CSCMat &H) const {
        if (enabled()) m_accumulateHessian(H);
    }

    // Apply weighted energy's Hessian to `v`, accumulating to `out`
    void applyHessian(Eigen::Ref<const VecX> v, Eigen::Ref<VecX> out) const {
        if (enabled()) m_applyHessian(v, out);
    }

    virtual void injectHessianSparsityPattern(CSCMat &H, Real val = 0.0) const = 0;

    virtual bool enabled() const = 0;

    const Object &getObject() const { return m_object; }

    virtual ~RegularizationTerm() { }

    Real weight = 1.0;
protected:
    // Actual implementations provided by subclasses.
    virtual Real m_unweightedEnergy() const = 0;
    virtual void m_accumulateGradient(Eigen::Ref<VecX> g) const = 0;
    virtual void m_accumulateHessian(CSCMat &H) const = 0;
    virtual void m_applyHessian(Eigen::Ref<const VecX> v, Eigen::Ref<VecX> out) const = 0;

    const Object &m_object;
};

////////////////////////////////////////////////////////////////////////////////
// Rest curvature smoothing
////////////////////////////////////////////////////////////////////////////////

// LaplacianStencilElasticRod1D is duplicate for LaplacianStencil1D<ElasticRod> needed
// to create the RodStencil in LaplacianStencil1D in the general case
struct LaplacianStencilElasticRod1D {
    using Vec2 = Eigen::Vector2d;
    // Call `visitor(endPtKappaVars, endPtKappas)` once for each interior
    // edge in the rod.
    template<class F>
    static void visit(const ElasticRod &r, const F &visitor, const size_t rk_offset) {
        for (size_t j = 1; j < r.numEdges() - 1; ++j) {
            visitor(std::array<size_t, 2>{{rk_offset + j - 1, rk_offset + j}},
                    Vec2{r.restKappas()[j][0], r.restKappas()[j + 1][0]});
        }
    }

    template<class F>
    static void visit(const ElasticRod &r, const F &visitor) {
        visit(r, visitor, r.restKappaOffset());
    }
};

template<typename Object>
struct LaplacianStencil1D {
    using RodStencil = LaplacianStencilElasticRod1D;
    using Vec2 = Eigen::Vector2d;
    // Call `visitor(endPtKappaVars, endPtKappas)` once for each interior
    // edge in the rod segments and once for each pair of
    // overlapping joint edges in the linkage.
    template<class F>
    static void visit(const Object &l, const F &visitor) {
        // Visit the interior edges
        for (size_t si = 0; si < l.numSegments(); ++si) {
            RodStencil::visit(l.segment(si).rod, visitor,
                              l.restKappaDofOffsetForSegment(si));
        }

        // Visit the overlapping joint edges
        for (const auto &j : l.joints()) {
            for (size_t abo = 0; abo < 2; ++abo) {
                if (j.numSegments(abo) != 2) continue;
                std::array<size_t, 2> endptKappaVars;
                Vec2 endptKappas;
                for (size_t endpt = 0; endpt < 2; ++endpt) {
                    size_t si = j.segments(abo)[endpt];
                    const auto &r = l.segment(si).rod;
                    size_t sourceVtx = j.isStart(abo)[endpt] ? 1 : r.numVertices() - 2;
                    endptKappaVars[endpt] = l.restKappaDofOffsetForSegment(si) + (sourceVtx - 1);
                    endptKappas   [endpt] = r.restKappas()[sourceVtx][0];
                }
                visitor(endptKappaVars, endptKappas);
            }
        }
    }
};

template<>
struct LaplacianStencil1D<ElasticRod> {
    using Vec2 = Eigen::Vector2d;
    // Call `visitor(endPtKappaVars, endPtKappas)` once for each interior
    // edge in the rod.
    template<class F>
    static void visit(const ElasticRod &r, const F &visitor, const size_t rk_offset) {
        for (size_t j = 1; j < r.numEdges() - 1; ++j) {
            visitor(std::array<size_t, 2>{{rk_offset + j - 1, rk_offset + j}},
                    Vec2{r.restKappas()[j][0], r.restKappas()[j + 1][0]});
        }
    }

    template<class F>
    static void visit(const ElasticRod &r, const F &visitor) {
        visit(r, visitor, r.restKappaOffset());
    }
};

template<typename Object>
struct RestCurvatureSmoothing : public RegularizationTerm<Object> {
    using Base    = RegularizationTerm<Object>;
    using Stencil = LaplacianStencil1D<Object>;
    using CSCMat  = typename Base::CSCMat;
    using VecX    = typename Base::VecX;
    using Vec2    = typename Object::Vec2;

    using Base::Base;

    virtual bool enabled() const override {
        return m_object.getDesignParameterConfig().restKappa;
    }

    virtual void injectHessianSparsityPattern(CSCMat &H, Real val = 0.0) const override {
        if (enabled()) {
            TripletMatrix<> newEntries(H.m, H.n);
            m_accumulateHessianImpl(newEntries, 1.0);
            H.addWithDistinctSparsityPattern(newEntries);
            H.fill(val);
        }
    }

    using Base::weight;
private:
    using Base::m_object;
    virtual Real m_unweightedEnergy() const override {
        Real result = 0.0;
        Stencil::visit(m_object, [&result](const std::array<size_t, 2> &/* endptKappaVars */,
                                           const Vec2 &endptKappas) {
                Real diff = endptKappas[1] - endptKappas[0];
                result += 0.5 * diff * diff;
            });
        return result;
    }

    virtual void m_accumulateGradient(Eigen::Ref<VecX> g) const override {
        Stencil::visit(m_object, [&](const std::array<size_t, 2> &endptKappaVars,
                                     const Vec2 &endptKappas) {
                Real weightedDiff = weight * (endptKappas[1] - endptKappas[0]);
                g[endptKappaVars[0]] -= weightedDiff;
                g[endptKappaVars[1]] += weightedDiff;
            });
    }

    // Templated to work with triplet matrix too for sparsity pattern construction
    template<class SpMat>
    void m_accumulateHessianImpl(SpMat &H, Real customWeight) const {
        Stencil::visit(m_object, [&](std::array<size_t, 2> endptKappaVars,
                                     const Vec2 &/* endptKappas */) {
                std::sort(endptKappaVars.begin(), endptKappaVars.end());
                H.addNZ(endptKappaVars[0], endptKappaVars[0],  customWeight);
                H.addNZ(endptKappaVars[0], endptKappaVars[1], -customWeight); // Note: endptKappaVars are not
                H.addNZ(endptKappaVars[1], endptKappaVars[1],  customWeight); //       necessarily contiguous!
            });
    }

    virtual void m_accumulateHessian(CSCMat &H) const override { m_accumulateHessianImpl(H, weight); }

    virtual void m_applyHessian(Eigen::Ref<const VecX> v, Eigen::Ref<VecX> out) const override {
        VecX result = VecX::Zero(v.rows());
        Stencil::visit(m_object, [&](const std::array<size_t, 2> &endptKappaVars,
                                     const Vec2 &/* endptKappas */) {
            out[endptKappaVars[0]] += weight * v[endptKappaVars[0]];
            out[endptKappaVars[1]] -= weight * v[endptKappaVars[0]];
            out[endptKappaVars[0]] -= weight * v[endptKappaVars[1]];
            out[endptKappaVars[1]] += weight * v[endptKappaVars[1]];
        });
    }
};

////////////////////////////////////////////////////////////////////////////////
// Rest length minimization
////////////////////////////////////////////////////////////////////////////////

inline size_t restLenNumDoFGetter(const ElasticRod &r) { return r.numRestLengths(); }
inline size_t restLenNumDoFGetter(const RodLinkage &l) { return l.numSegments(); } // assumes per-segment rest lengths!

template<typename Object>
struct RestLengthMinimization : public RegularizationTerm<Object> {
    using Base    = RegularizationTerm<Object>;
    using CSCMat  = typename Base::CSCMat;
    using VecX    = typename Base::VecX;
    using Vec2    = typename Object::Vec2;

    using Base::Base;

    virtual bool enabled() const override {
        return m_object.getDesignParameterConfig().restLen;
    }

    // Objective is linear; Hessian is zero.
    virtual void injectHessianSparsityPattern(CSCMat &/* H */, Real /* val */ = 0.0) const override { }

    using Base::weight;
private:
    using Base::m_object;
    virtual Real m_unweightedEnergy() const override { 
        VecX restLengths = m_object.getDesignParameters().tail(m_object.numSegments());
        return restLengths.sum(); 
    }

    virtual void m_accumulateGradient(Eigen::Ref<VecX> g) const override {
        const size_t nrl = restLenNumDoFGetter(m_object);
        const size_t rlo = m_object.restLenOffset();
        g.segment(rlo, nrl) += VecX::Constant(nrl, weight);
    }

    // Objective is linear; Hessian is zero.
    virtual void m_accumulateHessian(CSCMat &/* H */)                                       const override { }
    virtual void m_applyHessian(Eigen::Ref<const VecX> /* v */, Eigen::Ref<VecX> /* out */) const override { }
};

#endif /* end of include guard: REGULARIZATIONTERMS_HH */
