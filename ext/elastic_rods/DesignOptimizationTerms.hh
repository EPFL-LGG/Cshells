////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Terms for the design optimization, where the objective and constraints are
//  expressed in terms of the equilbrium x^*(p) (which is considered to be a
//  function of the design parameters p).
//
//  Objective terms are of the form w_i * J_i(x, p), where w_i is term's weights
//  Constraint terms are of the form c_i(x, p).
//
//  Then the full objective is the the form J(p) = sum_i w_i J_i(x^*(p)), p.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  04/26/2020 17:57:37
////////////////////////////////////////////////////////////////////////////////
#ifndef DESIGNOPTIMIZATIONTERMS_HH
#define DESIGNOPTIMIZATIONTERMS_HH

#include <MeshFEM/AutomaticDifferentiation.hh>
#include <memory>
#include <algorithm>

#include "ElasticRod.hh"

// Traits class to be specialized for each object type template (RodLinkage_T,
// SurfaceAttractedLinkage_T, UmbrellaMesh_T, etc.) in order to implement the
// following standardized inteface.
//  "Simulation variables:" the variables exposed to the equilibrium optimizer
//  "Design variables:" the variables exposed to the design optimizer
//  "Augmented variables:" concatenation of simulation variables and design variables (in that order)
// The default/example implementation provided here is meant for compability
// with the RodLinkage_T/SurfaceAttractedLinkage_T classes.
template<template<typename> class Object_T>
struct DesignOptimizationObjectTraits {
    template<typename T> static size_t     numSimVars      (const Object_T<T> &obj) { return obj.numDoF(); }
    template<typename T> static size_t     numDesignVars   (const Object_T<T> &obj) { return obj.numDesignParams(); }
    template<typename T> static size_t     numAugmentedVars(const Object_T<T> &obj) { return numSimVars(obj) + numDesignVars(obj); }
    template<typename T> static auto       getAugmentedVars(const Object_T<T> &obj) { return obj.getExtendedDoFsPSRL(); }
    template<typename T> static void       setAugmentedVars(      Object_T<T> &obj, const VecX_T<T> &v) { obj.setExtendedDoFsPSRL(v); }
    // WARNING: in the SurfaceAttractedLinkage_T case, the folowing also includes the target fitting term with a small weight.
    // This should, however, be OK, since it corresponds to slightly increasing the weight on the fitting term in the design objective.
    template<typename T> static auto          elasticEnergy(const Object_T<T> &obj) { return obj.energy(); }
    template<typename T> static auto      gradElasticEnergy(const Object_T<T> &obj) { return obj.gradientPerSegmentRestlen(/* updated source */ true); }
    template<typename T> static auto applyHessElasticEnergy(const Object_T<T> &obj, Eigen::Ref<const VecX_T<T>> delta_xp, const HessianComputationMask &mask = HessianComputationMask()) { return obj.applyHessianPerSegmentRestlen(delta_xp, mask); }
};

template<>
struct DesignOptimizationObjectTraits<ElasticRod_T> {
    template<typename T> static size_t     numSimVars      (const ElasticRod_T<T> &obj) { return obj.numDoF(); }
    template<typename T> static size_t     numDesignVars   (const ElasticRod_T<T> &obj) { return obj.numDesignParameters(); }
    template<typename T> static size_t     numAugmentedVars(const ElasticRod_T<T> &obj) { return numSimVars(obj) + numDesignVars(obj); }
    template<typename T> static auto       getAugmentedVars(const ElasticRod_T<T> &obj) { return obj.getExtendedDoFs(); }
    template<typename T> static void       setAugmentedVars(      ElasticRod_T<T> &obj, const VecX_T<T> &v) { obj.setExtendedDoFs(v); }
    template<typename T> static auto          elasticEnergy(const ElasticRod_T<T> &obj) { return obj.energy(); }
    template<typename T> static auto      gradElasticEnergy(const ElasticRod_T<T> &obj) { return obj.gradient(/* updated source */ true, ElasticRod_T<T>::EnergyType::Full, true); }
    template<typename T> static auto applyHessElasticEnergy(const ElasticRod_T<T> &obj, Eigen::Ref<const VecX_T<T>> delta_xp, const HessianComputationMask &mask = HessianComputationMask()) { return obj.applyHessian(delta_xp, mask); }
};

template<template<typename> class Object_T>
struct DesignOptimizationTerm {
    using   Object = Object_T<Real>;
    using ADObject = Object_T<ADReal>;
    using  OTraits = DesignOptimizationObjectTraits<Object_T>;
    using      VXd = Eigen::VectorXd;

    DesignOptimizationTerm(const Object &obj) : m_obj(obj) { }

    size_t numVars()       const { return OTraits::numAugmentedVars(m_obj); }
    size_t numSimVars()    const { return OTraits::numSimVars(m_obj); }
    size_t numDesignVars() const { return OTraits::numDesignVars(m_obj); }

    // So that we can use it in the linkage optimization (only relevant for a few terms)
    bool    useEnvelopeTheorem() const { return m_useEnvelopeTheorem; }
    void setUseEnvelopeTheorem(bool b) {        m_useEnvelopeTheorem = b; this->update(); }

    Real unweightedValue() const { return m_value(); }
    Real value() const { return m_weight() * m_value(); }

    // partial value / partial xp
    VXd grad() const { return m_weight() * cachedGrad(); }

    // partial value / partial x
    VXd grad_x() const { return m_weight() * cachedGrad().head(numSimVars()); }

    // partial value / partial p
    VXd grad_p() const { return m_weight() * cachedGrad().tail(numDesignVars()); }

    // (d^2 J / d{x, p} d{x, p}) [ delta_x, delta_p ]
    // Note: autodiffObject should have [delta_x, delta_p] already injected.
    VXd delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &autodiffObject) const { return m_weight() * m_delta_grad(delta_xp, autodiffObject); }

    // Reset the cache of the partial derivatives wrt x and p
    void update() { m_update(); m_cachedGrad.resize(0); }

    const VXd cachedGrad() const {
        if (size_t(m_cachedGrad.rows()) != numVars())
            m_cachedGrad = m_grad();
        return m_cachedGrad;
    }

    virtual ~DesignOptimizationTerm() { }

    ////////////////////////////////////////////////////////////////////////////
    // For validation/debugging
    ////////////////////////////////////////////////////////////////////////////
    const Object &object() const { return m_obj; }
    VXd computeGrad() const { return m_weight() * m_grad(); }

    VXd computeDeltaGrad(Eigen::Ref<const VXd> delta_xp) const {
        if (size_t(delta_xp.size()) != OTraits::numAugmentedVars(m_obj)) throw std::runtime_error("Size mismatch");
        ADObject diff_obj(m_obj);
        VecX_T<ADReal> ad_dofs = OTraits::getAugmentedVars(m_obj);
        const size_t nv = numVars();
        for (size_t i = 0; i < nv; ++i) ad_dofs[i].derivatives()[0] = delta_xp[i];
        OTraits::setAugmentedVars(diff_obj, ad_dofs);
        return m_weight() * m_delta_grad(delta_xp, diff_obj);
    }

    Real getWeight() const { return m_weight(); }
    virtual void setWeight(Real) { throw std::runtime_error("Weight cannot be changed."); }

protected:
    const Object &m_obj;
    mutable VXd m_cachedGrad;
    bool m_useEnvelopeTheorem = true; // accelerate one term by assuming dE/dx is identically zero; arguably could introduce some error if equillibrium problem is not solved exactly.

    virtual Real m_weight() const = 0; // Always 1.0 for constraints, custom weight for objective terms.

    // Methods that must be implemented by each term...
    virtual Real  m_value() const = 0;

    // [partial value/partial x, partial value/ partial p]
    virtual VXd  m_grad() const = 0;
    // delta [partial value/partial x, partial value/ partial p]
    virtual VXd m_delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &autodiffObject) const = 0;

    // (Optional) update cached quantities shared between energy/gradient calculations.
    virtual void m_update() { }
};

template<template<typename> class Object_T>
struct DesignOptimizationConstraint : public DesignOptimizationTerm<Object_T> {
protected:
    virtual Real m_weight() const override { return 1.0; } // constraints are unweighted.
};

template<template<typename> class Object_T>
struct DesignOptimizationObjectiveTerm : public DesignOptimizationTerm<Object_T> {
    using DOT  = DesignOptimizationTerm<Object_T>;
    using DOT::DOT;
    Real weight = 1.0;

    virtual void setWeight(Real w) override { weight = w; }
protected:
    virtual Real m_weight() const override { return weight; }
};

template<template<typename> class Object_T>
struct ElasticEnergyObjective : public DesignOptimizationObjectiveTerm<Object_T> {
    using DOT      = DesignOptimizationTerm<Object_T>;
    using DOOT     = DesignOptimizationObjectiveTerm<Object_T>;
    using ADObject = typename DOT::ADObject;
    using VXd      = Eigen::VectorXd;
    using EType    = typename DOT::Object::EnergyType;

    using DOOT::DOOT;

    using OTraits = typename DOT::OTraits;

    bool    useEnvelopeTheorem() const { return m_useEnvelopeTheorem; }
    void setUseEnvelopeTheorem(bool b) {        m_useEnvelopeTheorem = b; this->update(); }

protected:
    // Whether the envelope theorem applies, in which case we can assume dE/dx
    // is identically zero for a slight speed-up.
    // This cannot be done when the potential energy used for the equilibrium solve
    // differs from the `elasticEnergy` computed here.
    bool m_useEnvelopeTheorem = false;
    EType m_energyType = EType::Full;

    ////////////////////////////////////////////////////////////////////////////
    // Implementation of DesignOptimizationTerm interface
    ////////////////////////////////////////////////////////////////////////////
    virtual Real m_value() const override { return OTraits::elasticEnergy(this->m_obj); }
    virtual VXd m_grad()   const override {
        if (m_useEnvelopeTheorem) {
            VXd  result(this->numVars());
            result.tail(this->numDesignVars()) = this->m_obj.grad_design_parameters(/* updated source */ true);
            result.head(this->numSimVars()).setZero();
            return result;
        }
        return OTraits::gradElasticEnergy(this->m_obj);
    }

    virtual VXd m_delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &) const override {
        if (this->m_useEnvelopeTheorem) {
            HessianComputationMask mask;
            mask.dof_out = false;
            VXd result = OTraits::applyHessElasticEnergy(this->m_obj, delta_xp, mask);
            result.head(this->numSimVars()).setZero(); // maybe not needed...
            return result;
        }
        return OTraits::applyHessElasticEnergy(this->m_obj, delta_xp);
    }
};

#include "RodLinkage.hh"

template<template<typename> class Object_T>
struct LpStressDOOT : public DesignOptimizationObjectiveTerm<Object_T> {
    using DOT      = DesignOptimizationTerm<Object_T>;
    using DOOT     = DesignOptimizationObjectiveTerm<Object_T>;
    using Object   = typename DOT::Object;
    using ADObject = typename DOT::ADObject;
    using VXd      = Eigen::VectorXd;

    using DOOT::DOOT;

    double p = 7;
    CrossSectionStressAnalysis::StressType stressType = CrossSectionStressAnalysis::StressType::VonMises;

protected:
    using DOT::m_obj;
    bool m_useEnvelopeTheorem = true; // accelerate this term by assuming dE/dx is identically zero; arguably could introduce some error if equillibrium problem is not solved exactly.

    ////////////////////////////////////////////////////////////////////////////
    // Implementation of DesignOptimizationTerm interface
    ////////////////////////////////////////////////////////////////////////////
    virtual Real m_value() const override { return m_obj.surfaceStressLpNorm(stressType, p); }

    template<typename Real_>
    VecX_T<Real_> m_grad_impl(const Object_T<Real_> &obj) const {
        // Note that the objective terms/gradients are always evaluated after
        // the source frame is updated (by NewtonOptimizer)! We must explicitly
        // signal this to get the "correct" (asymmetric!) Hessian-vector product via autodiff.
        // (In other words, we do *not* want gradient contributions from the finite-transport
        //  terms that are killed off by the source frame update.)
        const bool updatedSource = true;
        return obj.gradSurfaceStressLpNorm(stressType, p, updatedSource);
    }
    virtual VXd m_grad() const override { return m_grad_impl(m_obj); }

    virtual VXd m_delta_grad(Eigen::Ref<const VXd> /* delta_xp */, const ADObject &autodiffObject) const override {
        return extractDirectionalDerivative(m_grad_impl(autodiffObject));
    }
};

// Wrapper for the target-fitting objective.
template<template<typename> class Object_T>
struct TargetFittingDOOT : public DesignOptimizationObjectiveTerm<Object_T> {
    using DOT      = DesignOptimizationTerm<Object_T>;
    using DOOT     = DesignOptimizationObjectiveTerm<Object_T>;
    using Object   = typename DOT::Object;
    using ADObject = typename DOT::ADObject;
    using VXd      = Eigen::VectorXd;

    TargetFittingDOOT(const Object &obj, TargetSurfaceFitter &tsf)
        : DOOT(obj), m_tsf(tsf) { }

    using DOOT::weight;
protected:
    using DOT::m_obj;
    TargetSurfaceFitter &m_tsf;

    ////////////////////////////////////////////////////////////////////////////
    // Implementation of DesignOptimizationTerm interface
    ////////////////////////////////////////////////////////////////////////////
    virtual void m_update() override { m_tsf.updateClosestPoints(m_obj); }

    virtual Real m_value() const override { return m_tsf.objective(m_obj); }

    virtual VXd m_grad() const override {
        VXd result = VXd::Zero(this->numVars());
        result.head(this->numSimVars()) = m_tsf.gradient(m_obj);
        return result;
    }

    virtual VXd m_delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &/* ado */) const override {
        return m_tsf.applyHessian(m_obj, delta_xp);
    }
};

// Wrapper for RegularizationTerms, which were originally implemented with an
// interface more convenient for the DesignParameterSolve.
template<template<typename> class Object_T, template<class> class RTerm_>
struct RegularizationTermDOOWrapper : public DesignOptimizationTerm<Object_T> {
    using DOT      = DesignOptimizationTerm<Object_T>;
    using Object   = typename DOT::Object;
    using ADObject = typename DOT::ADObject;
    using RTerm    = RTerm_<Object>;
    using VXd      = Eigen::VectorXd;

    RegularizationTermDOOWrapper(const Object &obj)
        : DOT(obj), m_term(std::make_shared<RTerm>(obj)),
          weight(m_term->weight) { }

    RegularizationTermDOOWrapper(const std::shared_ptr<RTerm> &term)
        : DOT(term->getObject()), m_term(term), weight(term->weight) { }

    virtual void setWeight(Real w) override { weight = w; }

protected:
    using DOT::m_obj;
    std::shared_ptr<RTerm> m_term;
public:
    Real &weight; // must be after m_term for proper initialization order.
protected:

    virtual Real m_weight() const override { return weight; }

    ////////////////////////////////////////////////////////////////////////////
    // Implementation of DesignOptimizationTerm interface
    ////////////////////////////////////////////////////////////////////////////
    virtual Real m_value() const override { return m_term->unweightedEnergy(); }

    virtual VXd m_grad() const override {
        Real oldWeight = weight;
        weight = 1.0;

        VXd result = VXd::Zero(this->numVars());
        m_term->accumulateGradient(result);

        weight = oldWeight;
        return result;
    }

    virtual VXd m_delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &/* autodiffObject */) const override {
        Real oldWeight = weight;
        weight = 1.0;

        VXd result = VXd::Zero(this->numVars());
        m_term->applyHessian(delta_xp, result);

        weight = oldWeight;
        return result;
    }
};

// CRTP base class for implementing adjoint state.
template<template<typename> class Object_T, class OptimizationTerm, typename EType>
struct AdjointState {
    using Derived = OptimizationTerm;

    using DOT = DesignOptimizationTerm<Object_T>;
    using VXd = Eigen::VectorXd;
    using   Object = typename DOT::  Object;
    using ADObject = typename DOT::ADObject;

    const OptimizationTerm &derived() const { return *static_cast<const OptimizationTerm *>(this); }
          OptimizationTerm &derived()       { return *static_cast<      OptimizationTerm *>(this); }

    // Adjoint solve for the optimization term:
    //      [H_3D a][w_x     ] = [dJ/dx]   or    H_3D w_x = dJ/dx
    //      [a^T  0][w_lambda]   [  0  ]
    // depending on whether average angle actuation is applied.
    void updateAdjointState(NewtonOptimizer &opt, EType etype = EType::Full) {
        m_adjointState.resize(opt.get_problem().numVars()); // ensure correct size in case adjoint solve fails...
        if (opt.get_problem().hasLEQConstraint()) m_adjointState = opt.extractFullSolution(opt.kkt_solver(opt.solver, opt.removeFixedEntries(derived().grad_x(etype))));
        else                                      m_adjointState = opt.extractFullSolution(          opt.solver.solve(opt.removeFixedEntries(derived().grad_x(etype))));
    }

    // Solve for the adjoint state in the case that bound constraints are active.
    // This should have 0s in the components corresponding to the constrained variables.
    void updateAdjointState(NewtonOptimizer &opt, const WorkingSet &ws, EType etype = EType::Full) {
        updateAdjointState(opt, etype);
        ws.getFreeComponentInPlace(m_adjointState);
    }

    const VXd &adjointState() const { return m_adjointState; }
    void clearAdjointState() { m_adjointState.setZero(); }

protected:
    VXd m_adjointState, m_deltaAdjointState;
};

// Collect objective terms depending on a single object/equilibrium.
// (All of these collectively will be given a single adjoint state).
template<template<typename> class Object_T, class EType>
struct DesignOptimizationObjective : public AdjointState<Object_T, DesignOptimizationObjective<Object_T, EType>, EType> {
    using DOT      = DesignOptimizationTerm<Object_T>;
    using ADObject = typename DOT::ADObject;
    using VXd      = Eigen::VectorXd;

    using TermPtr = std::shared_ptr<DOT>;
    struct TermRecord {
        TermRecord(const std::string &n, EType t, TermPtr tp) : name(n), type(t), term(tp) { }
        std::string name;
        EType type;
        TermPtr term;
    };

    void update() {
        for (TermRecord &t : terms)
            t.term->update();
    }

    Real value(EType type = EType::Full) const {
        Real val = 0.0;
        for (const TermRecord &t : terms) {
            if (t.term->getWeight() == 0.0) continue;
            if ((type == EType::Full) || (type == t.type))
                val += t.term->value();
        }
        return val;
    }

    Real operator()(EType type = EType::Full) const { return value(type); }

    // Note: individual terms return their *unweighted* values.
    Real value(const std::string &name) const {
        auto it = m_find(name);
        if (it == terms.end()) throw std::runtime_error("No term with name " + name);
        return it->term->unweightedValue();
    }

    VXd grad(EType type = EType::Full) const {
        if (terms.empty()) throw std::runtime_error("no terms present");
        VXd result = VXd::Zero(terms[0].term->numVars());
        for (const TermRecord &t : terms) {
            if (t.term->getWeight() == 0.0) continue;
            if ((type == EType::Full) || (type == t.type))
                result += t.term->grad();
        }
        return result;
    }

    VXd grad_x(EType type = EType::Full) const {
        return grad(type).head(terms.at(0).term->numSimVars());
    }

    VXd grad_p(EType type = EType::Full) const {
        return grad(type).tail(terms.at(0).term->object().numDesignParams());
    }

    VXd computeGrad(EType type = EType::Full) const {
        if (terms.empty()) throw std::runtime_error("no terms present");
        VXd result = VXd::Zero(terms[0].term->numVars());
        for (const TermRecord &t : terms) {
            if ((type == EType::Full) || (type == t.type))
                result += t.term->computeGrad();
        }
        return result;
    }

    // (d^2 J / d{x, p} d{x, p}) [ delta_x, delta_p ]
    // Note: autodiffObject should have [delta_x, delta_p] already injected.
    VXd delta_grad(Eigen::Ref<const VXd> delta_xp, const ADObject &autodiffObject, EType type = EType::Full) const {
        if (terms.empty()) throw std::runtime_error("no terms present");
        VXd result = VXd::Zero(terms.front().term->numVars());
        for (const TermRecord &t : terms) {
            if (t.term->getWeight() == 0.0) continue;
            if ((type == EType::Full) || (type == t.type))
                result += t.term->delta_grad(delta_xp, autodiffObject);
        }
        return result;
    }

    VXd computeDeltaGrad(Eigen::Ref<const VXd> delta_xp, EType type = EType::Full) const {
        if (terms.empty()) throw std::runtime_error("no terms present");
        VXd result = VXd::Zero(terms.front().term->numVars());
        for (const TermRecord &t : terms) {
            if ((type == EType::Full) || (type == t.type))
                result += t.term->computeDeltaGrad(delta_xp);
        }
        return result;
    }

    std::map<std::string, Real> values() const {
        std::map<std::string, Real> result;
        for (const TermRecord &t : terms)
            result[t.name] = t.term->unweightedValue();
        return result;
    }

    std::map<std::string, Real> weightedValues() const {
        std::map<std::string, Real> result;
        for (const TermRecord &t : terms)
            result[t.name] = t.term->value();
        return result;
    }

    void printReport() const {
        std::cout << "value " << value();
        for (const TermRecord &t : terms)
            std::cout << " " << t.name << t.term->unweightedValue();
        std::cout << std::endl;
    }

    void add(const std::string &name, EType etype, TermPtr term) {
        if (m_find(name) != terms.end()) throw std::runtime_error("Term " + name + " already exists");
        terms.emplace_back(name, etype, term);
    }

    void add(const std::string &name, EType etype, TermPtr term, Real weight) {
        add(name, etype, term);
        terms.back().term->setWeight(weight);
    }

    void remove(const std::string &name) {
        auto it = m_find(name);
        if (it == terms.end()) throw std::runtime_error("Term " + name + " doesn't exist");
        terms.erase(it);
    }

    DOT &get(const std::string &name) {
        auto it = m_find(name);
        if (it == terms.end()) throw std::runtime_error("Term " + name + " doesn't exist");
        return *(it->term);
    }

    const DOT &get(const std::string &name) const {
        auto it = m_find(name);
        if (it == terms.end()) throw std::runtime_error("Term " + name + " doesn't exist");
        return *(it->term);
    }

    std::vector<TermRecord> terms;

private:
    decltype(terms.cbegin()) m_find(const std::string &name) const {
        return std::find_if(terms.begin(), terms.end(),
                            [&](const TermRecord &t) { return t.name == name; });
    }
};

#endif /* end of include guard: DESIGNOPTIMIZATIONTERMS_HH */
