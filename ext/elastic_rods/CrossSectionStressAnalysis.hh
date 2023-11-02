////////////////////////////////////////////////////////////////////////////////
// CrossSectionStressAnalysis.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Collects the quantities needed for computing stress integrals over the
//  cross-section boundary. We choose to integrate over the boundary rather
//  than the interior since we expect stress measures to be maximized at
//  the boundary.
//  One can show the von Mises stress in the cross-section due to a combination
//  of stretching, bending, and twisting is in fact maximized at the boundary
//  (it is the sum of two subharmonic functions). It is not obvious whether
//  the maximum principal stress is also maximized at the boundary, but it
//  should be in at least most cases.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  01/05/2022 16:13:59
////////////////////////////////////////////////////////////////////////////////
#ifndef CROSSSECTIONSTRESSANALYSIS_HH
#define CROSSSECTIONSTRESSANALYSIS_HH

#include <MeshFEM/GaussQuadrature.hh>
#include "CrossSection.hh"
#include <MeshFEM/AutomaticDifferentiation.hh>

struct CrossSectionStressAnalysis {
    static constexpr size_t QUADRATURE_DEGREE = 3;
    enum class StressType { VonMises, MaxPrincipal, MinPrincipal, ZStress, MaxMag };

    template<class FEMMesh_>
    CrossSectionStressAnalysis(const FEMMesh_ &mesh, const Eigen::MatrixXd &unitTwistShearStrainVolVtx, Real E, Real G)
        : youngModulus(E), shearModulus(G) {
        boundaryV.resize(mesh.numBoundaryVertices());
        boundaryE.resize(mesh.numBoundaryElements());
        unitTwistShearStrain.resize(mesh.numBoundaryVertices(), 2);
        for (const auto bv : mesh.boundaryVertices()) {
            unitTwistShearStrain.row(bv.index()) = unitTwistShearStrainVolVtx.row(bv.volumeVertex().index());
            boundaryV[bv.index()] = bv.volumeVertex().node()->p;
        }
        for (const auto be : mesh.boundaryEdges())
            boundaryE[be.index()] = std::make_pair(be.vertex(0).index(), be.vertex(1).index());
    }

    // Direct construction (for serialization)
    CrossSectionStressAnalysis(const CrossSection::AlignedPointCollection &bV, const CrossSection::EdgeCollection &bE,
                               const Eigen::MatrixX2d &utSS, Real E, Real G)
        : boundaryV(bV), boundaryE(bE), unitTwistShearStrain(utSS), youngModulus(E), shearModulus(G) { }

    // For Lp norm objectives we can avoid the sqrt and the associated derivative singularities
    template<bool Squared = false, typename Real_>
    static Real_ stressMeasure(StressType type, const Vec2_T<Real_> &shearStress, Real_ sigma_zz) {
        if (type == StressType::VonMises) return Squared ? Real_(3 * shearStress.squaredNorm() + sigma_zz * sigma_zz) : Real_(sqrt(3 * shearStress.squaredNorm() + sigma_zz * sigma_zz));
        if (type == StressType::ZStress)  return Squared ? sigma_zz * sigma_zz : sigma_zz;

        Real_ radical = sqrt(shearStress.squaredNorm() + 0.25 * sigma_zz * sigma_zz);
        if (type == StressType::MaxMag)
            type = sigma_zz > 0 ? StressType::MaxPrincipal : StressType::MinPrincipal;

        if (type == StressType::MinPrincipal) { radical *= -1; type = StressType::MaxPrincipal; }
        if (type == StressType::MaxPrincipal) { Real_ lmax = 0.5 * sigma_zz + radical; return Squared ? lmax * lmax : lmax; }

        throw std::runtime_error("Unknown StressType: " + std::to_string(int(type)));
    }

    // Derivative of `stressMeasure` with respect to the `shearStress` and `sigma_zz` arguments.
    template<bool Squared = false, typename Real_>
    static void gradStressMeasure(StressType type, const Vec2_T<Real_> &shearStress, Real_ sigma_zz,
                                  Vec2_T<Real_> &grad_shearStress, Real_ &grad_sigma_zz) {
        if (type == StressType::VonMises) {
            grad_shearStress = 6 * shearStress;
            grad_sigma_zz    = 2 * sigma_zz;
            if (!Squared) {
                Real_ scale_factor = 0.5 / sqrt(3 * shearStress.squaredNorm() + sigma_zz * sigma_zz);
                grad_shearStress *= scale_factor;
                grad_sigma_zz    *= scale_factor;
            }
            return;
        }
        if (type == StressType::ZStress) {
            grad_shearStress.setZero();
            grad_sigma_zz = Squared ? 2 * sigma_zz : Real_(1);
            return;
        }

        if (type == StressType::MaxMag)
            type = sigma_zz > 0 ? StressType::MaxPrincipal : StressType::MinPrincipal;

        double epsilon = 1e-10; // mitigate blow-ups in derivative formulas
        Real_ radical = sqrt(shearStress.squaredNorm() + 0.25 * sigma_zz * sigma_zz + epsilon);
        if (type == StressType::MinPrincipal) { radical *= -1.0; type = StressType::MaxPrincipal; }

        if (type == StressType::MaxPrincipal) {
            grad_shearStress = 1 / radical * shearStress;
            grad_sigma_zz    = 0.5 + (0.25 / radical) * sigma_zz;
            if (Squared) {
                Real_ unsquared = 0.5 * sigma_zz + radical;
                grad_shearStress *= 2 * unsquared;
                grad_sigma_zz    *= 2 * unsquared;
            }
            return;
        }

        throw std::runtime_error("Unknown StressType: " + std::to_string(int(type)));
    }

    // Integrate a function of a scalar stress measure `sigma` over the boundary curve.
    //      int_dOmega stressIntegrand(sigma) dl        if `SquaredMeasure` is false
    //      int_dOmega stressIntegrand(sigma^2) dl      if `SquaredMeasure` is true
    template<bool SquaredMeasure = false, class Integrand, typename Real_>
    Real_ stressIntegral(StressType type, Real_ tau, const Vec2_T<Real_> &curvatureNormal, Real_ eps_s, const Integrand &stressIntegrand) const {
        const size_t ne = boundaryE.size();
        Real_ result = 0;
        for (size_t ei = 0; ei < ne; ++ei) {
            int v0, v1;
            std::tie(v0, v1) = boundaryE[ei];
            result += Quadrature</* K = */ 1, QUADRATURE_DEGREE>::integrate(
                    [&](const EvalPt<1> &p) {
                        Vec2_T<Real_> shearStress = p[0] * unitTwistShearStrain.row(v0) // (computation broken up to work around Eigen autodiff limitations :()
                                                  + p[1] * unitTwistShearStrain.row(v1);
                        shearStress *= shearModulus * tau;
                        Real_ sigma_zz = youngModulus * (eps_s - curvatureNormal.dot(p[0] * boundaryV[v0] + p[1] * boundaryV[v1]));
                        return stressIntegrand(stressMeasure<SquaredMeasure>(type, shearStress, sigma_zz));
                    }, (boundaryV[v1] - boundaryV[v0]).norm());
        }
        return result;
    }

    // Derivative of `stressIntegral` with respect to the strain quantities `tau`, curvatureNormal` and `eps_s`.
    template<bool SquaredMeasure = false, class Integrand, typename Real_>
    void gradStressIntegral(StressType type, Real_ tau, const Vec2_T<Real_> &curvatureNormal, Real_ eps_s,
                            const Integrand &stressIntegrandPrime,
                            Real_ &grad_tau, Vec2_T<Real_> &grad_curvatureNormal, Real_ &grad_eps_s) const {
        const size_t ne = boundaryE.size();
        Eigen::Matrix<Real_, 4, 1> grad_result;
        grad_result.setZero();
        for (size_t ei = 0; ei < ne; ++ei) {
            int v0, v1;
            std::tie(v0, v1) = boundaryE[ei];
            grad_result += Quadrature</* K = */ 1, QUADRATURE_DEGREE>::integrate(
                    [&](const EvalPt<1> &p) {
                        Eigen::Matrix<Real_, 4, 1> gradIntegrand;
                        Vec2_T<Real_> shearStress = p[0] * unitTwistShearStrain.row(v0)
                                                  + p[1] * unitTwistShearStrain.row(v1);
                        shearStress *= shearModulus * tau;
                        Real_ sigma_zz = youngModulus * (eps_s - curvatureNormal.dot(p[0] * boundaryV[v0] + p[1] * boundaryV[v1]));
                        Real_ dI = stressIntegrandPrime(stressMeasure<SquaredMeasure>(type, shearStress, sigma_zz));
                        Vec2_T<Real_> grad_shearStress;
                        Real_ grad_sigma_zz;
                        gradStressMeasure<SquaredMeasure>(type, shearStress, sigma_zz, grad_shearStress, grad_sigma_zz);
                        // grad_tau
                        gradIntegrand[0] = dI * shearModulus * (p[0] * unitTwistShearStrain.row(v0)
                                                              + p[1] * unitTwistShearStrain.row(v1)).dot(grad_shearStress);
                        // grad_curvatureNormal (computation broken up to work around Eigen autodiff limitations :()
                        auto gcn = gradIntegrand.template segment<2>(1);
                        gcn = (p[0] * boundaryV[v0] + p[1] * boundaryV[v1]);
                        gcn *= dI * (-grad_sigma_zz * youngModulus);
                        // grad_eps_s
                        gradIntegrand[3] = dI * grad_sigma_zz * youngModulus;
                        return gradIntegrand;
                    }, (boundaryV[v1] - boundaryV[v0]).norm());
        }
        grad_tau             = grad_result[0];
        grad_curvatureNormal = grad_result.template segment<2>(1);
        grad_eps_s           = grad_result[3];
    }

    template<typename Real_>
    Real_ maxStress(StressType type, Real_ tau, const Vec2_T<Real_> &curvatureNormal, Real_ eps_s) const {
        using namespace std; // work around compilation issue with std::abs + autodiff
        Real_ result = 0;
        for (size_t i = 0; i < boundaryV.size(); ++i) {
            Vec2_T<Real_> shearStress = unitTwistShearStrain.row(i);
            shearStress *= (tau * shearModulus);
            Real_ sigma_zz = youngModulus * (eps_s - curvatureNormal.dot(boundaryV[i]));
            result = std::max<Real_>(result, /* not std::abs because of autodiff! */ abs(stressMeasure<false>(type, shearStress, sigma_zz)));
        }

        return result;
    }

    Eigen::VectorXd boundaryVertexStresses(StressType type, Real tau, const Eigen::Vector2d &curvatureNormal, Real eps_s) const {
        Eigen::VectorXd result(boundaryV.size());
        for (size_t i = 0; i < boundaryV.size(); ++i)
            result[i] = stressMeasure<false>(type, (tau * shearModulus * unitTwistShearStrain.row(i)).transpose().eval(), youngModulus * (eps_s - curvatureNormal.dot(boundaryV[i])));
        return result;
    }

    CrossSection::AlignedPointCollection boundaryV;
    CrossSection::EdgeCollection boundaryE;
    Eigen::MatrixX2d unitTwistShearStrain;
    Real youngModulus, shearModulus;
};

#endif /* end of include guard: CROSSSECTIONSTRESSANALYSIS_HH */
