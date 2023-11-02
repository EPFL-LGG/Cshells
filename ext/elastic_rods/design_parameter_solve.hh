#ifndef DESIGNPARAMETER_SOLVE_HH
#define DESIGNPARAMETER_SOLVE_HH

#include <vector>
#include <cmath>
#include <memory>
#include <map>

#include <MeshFEM/newton_optimizer/newton_optimizer.hh>
#include "compute_equilibrium.hh"

#include "RegularizationTerms.hh"

// Solve an equilibrium problem augemented with rest length variables.
template<typename Object>
struct DesignParameterProblem : public NewtonProblem {
    // E0, l0 are the elastic energy/length of the original design.
    // These can be specified manually in case the DesignParameterProblem is being
    // constructed from a design that has already been modified.
    DesignParameterProblem(Object &obj, Real E0 = -1, Real l0 = -1)
        : object(obj), m_characteristicLength(obj.characteristicLength())
    {
        m_E0 = (E0 > 0) ? E0 : obj.designParameterSolve_energy();
        m_l0 = (l0 > 0) ? l0 : obj.totalRestLength();
        // Make sure length variables aren't shrunk down to zero/inverted
        // when the sign of the length variables flips, the corresponding tangent vector will turn exactly 180 degrees, which is singularity for parallel transport. 
        if (obj.getDesignParameterConfig().restLen) {
            const Real initMinRestLen = obj.initialMinRestLength(); 
            auto lengthVars = obj.designParameterSolve_lengthVars();
            m_boundConstraints.reserve(lengthVars.size());
            for (size_t var : lengthVars)
                m_boundConstraints.emplace_back(var, 0.01 * initMinRestLen, BoundConstraint::Type::LOWER);
        }

        setFixedVars(obj.designParameterSolveFixedVars());

        m_regularizationTerms[ "smoothing"] = std::make_unique<RestCurvatureSmoothing<Object>>(object);
        m_regularizationTerms["restlenMin"] = std::make_unique<RestLengthMinimization<Object>>(object);

        m_hessianSparsity = obj.designParameterSolve_hessianSparsityPattern();
        for (const auto &reg : m_regularizationTerms)
            reg.second->injectHessianSparsityPattern(m_hessianSparsity);
    }

    virtual void setVars(const Eigen::VectorXd &vars) override { object.designParameterSolve_setDoF(vars); }
    virtual const Eigen::VectorXd getVars() const override { return object.designParameterSolve_getDoF(); }
    virtual size_t numVars() const override { return object.designParameterSolve_numDoF(); }

    virtual Real energy() const override {
        Real result = gamma / m_E0 * object.designParameterSolve_energy();
        for (const auto &reg : m_regularizationTerms)
            result += reg.second->energy();
        return result;
    }

    virtual Eigen::VectorXd gradient(bool freshIterate = false) const override {
        Eigen::VectorXd g = gamma / m_E0 * object.designParameterSolve_gradient(freshIterate, ElasticRod::EnergyType::Full);
        for (const auto &reg : m_regularizationTerms)
            reg.second->accumulateGradient(g);
        return g;
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { return m_hessianSparsity; }

    virtual void writeIterateFiles(size_t it) const override { if (writeIterates) { ::writeIterateFiles(object, it); } }

    virtual void writeDebugFiles(const std::string &errorName) const override {
        auto H = object.hessian();
        H.rowColRemoval(fixedVars());
        H.reflectUpperTriangle();
        H.dumpBinary("debug_" + errorName + "_hessian.mat");
        objectSpecificDebugFiles(object, errorName);
        object.saveVisualizationGeometry("debug_" + errorName + "_geometry.msh");
    }

    void set_smoothing_weight     (const Real weight) { m_regularizationTerms.at( "smoothing")->weight = weight       ; m_clearCache(); }
    void set_regularization_weight(const Real weight) { m_regularizationTerms.at("restlenMin")->weight = weight / m_l0; m_clearCache(); }
    void set_gamma (const Real new_gamma)             { gamma = new_gamma; m_clearCache(); }

    Real restKappaSmoothness() const { return m_regularizationTerms.at("smoothing")->unweightedEnergy(); }
    Real weighted_energy() const { return gamma / m_E0 * object.designParameterSolve_energy(); }
    Real weighted_smoothness() const { return m_regularizationTerms.at("smoothing")->energy(); }
    Real weighted_length() const { return m_regularizationTerms.at("restlenMin")->energy(); }

    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

    Real elasticEnergyWeight() const { return gamma / m_E0; }

    Real E0() const { return m_E0; }
    Real l0() const { return m_l0; }

private:
    virtual void m_iterationCallback(size_t i) override {
        object.updateSourceFrame(); object.updateRotationParametrizations();
        if (m_customCallback) m_customCallback(*this, i);
    }

    virtual void m_evalHessian(SuiteSparseMatrix &result, bool /* projectionMask */) const override {
        result.setZero();
        object.designParameterSolve_hessian(result, ElasticRod::EnergyType::Full);
        result.scale(gamma / m_E0);
        for (const auto &reg : m_regularizationTerms)
            reg.second->accumulateHessian(result);
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        result.setZero();
        object.massMatrix(result);
        const size_t dpo = object.designParameterOffset(), ndp = object.designParameterSolve_numDesignParameters();
        for (size_t j = 0; j < ndp; ++j) {
            result.addNZ(result.findDiagEntry(dpo + j), m_characteristicLength);
            // TODO: figure out a more sensible mass to use for rest length variables.
            // Initial mass of each segment?
        }
    }

    Object &object;
    mutable SuiteSparseMatrix m_hessianSparsity;
    Real m_characteristicLength = 1.0;
    Real m_E0 = 1.0;
    Real m_l0 = 1.0;
    Real gamma = 1;

    std::map<std::string, std::unique_ptr<RegularizationTerm<Object>>> m_regularizationTerms;

    CallbackFunction m_customCallback;
};

template<typename Object>
std::unique_ptr<DesignParameterProblem<Object>> designParameter_problem(Object &obj, const Real regularization_weight, const Real smoothing_weight, const std::vector<size_t> &fixedVars = std::vector<size_t>(), Real E0 = -1, Real l0 = -1) {
    auto problem = std::make_unique<DesignParameterProblem<Object>>(obj, E0, l0);
    problem->set_regularization_weight(regularization_weight);
    problem->set_smoothing_weight(smoothing_weight);
    // Also fix the variables specified by the user.
    problem->addFixedVariables(fixedVars);
    return problem;
}

template<typename Object>
std::unique_ptr<NewtonOptimizer> get_designParameter_optimizer(Object &obj, const Real regularization_weight, const Real smoothing_weight, const std::vector<size_t> &fixedVars = std::vector<size_t>(), CallbackFunction customCallback = nullptr, Real E0 = -1, Real l0 = -1) {
    auto problem = designParameter_problem(obj, regularization_weight, smoothing_weight, fixedVars, E0, l0);
    problem->setCustomIterationCallback(customCallback);
    return std::make_unique<NewtonOptimizer>(std::move(problem));
}

// Rest length solve with custom optimizer options.
template<typename Object>
ConvergenceReport designParameter_solve(Object &obj, const NewtonOptimizerOptions &opts, const Real regularization_weight = 0.1, const Real smoothing_weight = 1, const std::vector<size_t> &fixedVars = std::vector<size_t>(), CallbackFunction customCallback = nullptr, Real E0 = -1, Real l0 = -1) {
    auto opt = get_designParameter_optimizer(obj, regularization_weight, smoothing_weight, fixedVars, customCallback, E0, l0);
    opt->options = opts;
    return opt->optimize();
}

// Default options for rest length solve: use the identity metric.
template<typename Object>
ConvergenceReport designParameter_solve(Object &obj, const Real regularization_weight = 0.1, const Real smoothing_weight = 1, const std::vector<size_t> &fixedVars = std::vector<size_t>(), CallbackFunction customCallback = nullptr, Real E0 = -1, Real l0 = -1) {
    NewtonOptimizerOptions opts;
    opts.useIdentityMetric = true;
    return designParameter_solve(obj, opts, regularization_weight, smoothing_weight, fixedVars, customCallback, E0, l0);
}

#endif /* end of include guard: DESIGNPARAMETER_SOLVE_HH */
