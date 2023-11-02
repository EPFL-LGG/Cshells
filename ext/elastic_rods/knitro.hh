#ifndef KNITRO_HH
#define KNITRO_HH
#include <iostream>

struct NewPtCallbackBase {
    virtual int operator()(const double *x) = 0;
    virtual ~NewPtCallbackBase() { }
};

#if KNITRO_LEGACY
#include <KTRSolver.h>
#include <KTRProblem.h>

struct KnitroLegacyNewPtCallback : public knitro::KTRNewptCallback {
    KnitroLegacyNewPtCallback(NewPtCallbackBase &ncb) : m_ncb(ncb) { }

    virtual int CallbackFunction(const std::vector<double> &x, const std::vector<double>& /* lambda */, double /* obj */,
            const std::vector<double>& /* c */, const std::vector<double>& /* objGrad */,
            const std::vector<double>& /* jac */, knitro::KTRISolver * /* solver */) override {
        return m_ncb(x.data());
    }

private:
    NewPtCallbackBase &m_ncb;
};

template<class Derived>
struct KnitroProblem : public knitro::KTRProblem {
    using Base = knitro::KTRProblem;

    KnitroProblem(int numParams, int numConstraints)
        : Base(numParams, numConstraints)
    {
    }

    double evaluateFC(const std::vector<double> &x,
                            std::vector<double> &cval,
                            std::vector<double> &objGrad,
                            std::vector<double> &jac) {
        return derived().evalFC(x.data(), cval.data(), objGrad.data(), jac.data());
    }

    int evaluateGA(const std::vector<double> &x, std::vector<double> &objGrad, std::vector<double> &jac) {
        return derived().evalGA(x.data(), objGrad.data(), jac.data());
    }

    // Note: "lambda" contains a Lagrange multiplier for each constraint and each variable.
    // The first numConstraints entries give each constraint's multiplier in order, and the remaining
    // numVars entries give each the multiplier for the variable's active simple bound constraints (if any).
    int evaluateHessianVector(const std::vector<double> &x, double sigma, const std::vector<double> &lambda,
                              std::vector<double> &vec) {
        return derived().evalHessVec(x.data(), sigma, lambda.data(), vec.data(), vec.data());
        // return derived().evalHessVec(x, sigma, lambda, vec);
    }

    void setNewPointCallback(NewPtCallbackBase *newPt) {
        m_callbackNewPT = std::make_unique<KnitroLegacyNewPtCallback>(*newPt);
        Base::setNewPointCallback(m_callbackNewPT.get());
    }

          Derived &derived()       { return *static_cast<Derived *>(this); }
    const Derived &derived() const { return *static_cast<Derived *>(this); }

    std::unique_ptr<KnitroLegacyNewPtCallback> m_callbackNewPT;
};


#define KPREFIXBARE(x) KTR ## x
#define KPREFIX(x)     KTR ## _ ## x
using KnitroException = knitro::KPREFIXBARE(Exception);
using KnitroSolver    = knitro::KPREFIXBARE(Solver);

inline void printKnitroException(const KnitroException &e) {
    e.printMessage();
}

#else

#include <KNSolver.h>
#include <KNProblem.h>
#include <memory>

#define KPREFIXBARE(x) KN ## x
#define KPREFIX(x)     KN ## _ ## x

template<class Derived>
struct KnitroProblem : public knitro::KNProblem {
    using Base = knitro::KNProblem;

    KnitroProblem(int numParams, int numConstraints)
        : Base(numParams, numConstraints)
    {
        Base:: setObjEvalCallback(&callbackEvalFCGA);
        // Base::setGradEvalCallback(&callbackEvalGA);
        Base::setHessEvalCallback(&callbackEvalH );
        // What a bizarre API... Configure the userParams passed to *all* of the
        // callbacks set above (not just the ObjEvalCallback).
        Base::getObjEvalCallback().setParams(this);
    }

    void setNewPointCallback(NewPtCallbackBase *newPt) {
        if (newPt == nullptr) {
            m_callbackNewPT.reset();
            Base::setNewPointCallback(nullptr);
            return;
        }
        m_callbackNewPT = std::make_unique<knitro::KNUserCallback>(&callbackNewPoint, newPt);
        Base::setNewPointCallback(*m_callbackNewPT);
    }

    static int callbackEvalFCGA(KN_context_ptr             /* kc */,
                              CB_context_ptr             /* cb */,
                              KN_eval_request_ptr const  evalRequest,
                              KN_eval_result_ptr  const  evalResult,
                              void              * const  userParams) {
        assert(evalResult->objGrad != nullptr && "Gradient not requested...");
        assert(evalResult->jac     != nullptr && "Constraint Jacobian not requested...");
        double &result = *(evalResult->obj);
        result = ((Derived *)(userParams))->evalFC(evalRequest->x, evalResult->c,
                                                   evalResult->objGrad, evalResult->jac);
        return 0;
    }

    static int callbackEvalGA(KN_context_ptr             /* kc */,
                              CB_context_ptr             /* cb */,
                              KN_eval_request_ptr const  evalRequest,
                              KN_eval_result_ptr  const  evalResult,
                              void              * const  userParams) {
        assert(evalResult->objGrad != nullptr && "Gradient not requested...");
        assert(evalResult->jac     != nullptr && "Constraint Jacobian not requested...");
        return ((Derived *)(userParams))->evalGA(evalRequest->x,
                                                evalResult->objGrad, evalResult->jac);
    }

    static int callbackEvalH(KN_context_ptr             /* kc */,
                             CB_context_ptr             /* cb */,
                             KN_eval_request_ptr const  evalRequest,
                             KN_eval_result_ptr  const  evalResult,
                             void              * const  userParams) {
        if (evalRequest->type != KN_RC_EVALHV) throw std::runtime_error("Unimplemented Hessian evaluation type");
        return ((Derived *)(userParams))->evalHessVec(evalRequest->x, *(evalRequest->sigma), evalRequest->lambda,
                                                     evalRequest->vec, evalResult->hessVec);
    }

    static int callbackNewPoint(KN_context_ptr        /* kc */,
                                const double * const  x,
                                const double * const  /* lambda */,
                                void   *        raw_newpt) {
        return (*(NewPtCallbackBase *)(raw_newpt))(x);
    }

private:
    std::unique_ptr<knitro::KNUserCallback> m_callbackNewPT;
};

struct KnitroSolver : public knitro::KNSolver {
    using Base = knitro::KNSolver;
    KnitroSolver(knitro::KNProblem *problem, int gradopt, int hessopt)
        : Base(problem, gradopt, hessopt)
    {
        // The C++ wrappers in Knitro 12 no longer call `initProblem` for us
        // despite the comment describing `initProblem` in `KNSolver.h`: 
        //  "Called by the KNSolver constructor to register the problem properties with the KNITRO solver."
        Base::initProblem();

        Base::setParam(KN_PARAM_EVAL_FCGA, KN_EVAL_FCGA_YES);
    }
    void useNewptCallback() { } // no longer needed...
};

using KnitroException = knitro::KNExceptionClass;

inline void printKnitroException(const KnitroException &e) {
    std::cout << e.what() << std::endl;
}

#define KN_INFBOUND KN_INFINITY

#endif

#endif /* end of include guard: KNITRO_HH */
