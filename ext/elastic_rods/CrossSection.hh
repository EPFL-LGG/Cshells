////////////////////////////////////////////////////////////////////////////////
// CrossSection.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Abstract base class representing the interface to the cross-section
//  geometry.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  06/22/2018 11:14:23
////////////////////////////////////////////////////////////////////////////////
#ifndef CROSSSECTION_HH
#define CROSSSECTION_HH
#include <vector>
#include <stdexcept>
#include <memory>
#include <MeshFEM/Types.hh>
#include <MeshFEM/MeshIO.hh>

struct CrossSection {
    using Edge = std::pair<size_t, size_t>;
    using EdgeCollection = std::vector<Edge>;
    using AlignedPointCollection = std::vector<Point2D,
                                               Eigen::aligned_allocator<Point2D>>; // Work around alignment issues.
    using BRep = std::pair<AlignedPointCollection, EdgeCollection>;
    using VRep = std::pair<std::vector<MeshIO::IOVertex>, std::vector<MeshIO::IOElement>>;

    static std::unique_ptr<CrossSection> load(const std::string &path);
    static std::unique_ptr<CrossSection> construct(std::string type, Real E, Real nu, const std::vector<Real> &params);

    CrossSection() : E(0), nu(0) { }
    CrossSection(const CrossSection &b) : E(b.E), nu(b.nu) { }

    ////////////////////////////////////////////////////////////////////////////
    /*! Get the cross-section's boundary.
    //  @param[in]  highRes    whether this is the low-res
    //                         visualization geometry or the high-res
    //                         geometry used for calculating moduli.
    *///////////////////////////////////////////////////////////////////////////
    virtual BRep boundary(bool highRes = false) const = 0;

    ////////////////////////////////////////////////////////////////////////////
    /*! Mesh the interior of the cross-section with triangles (to be used for
    //  calculating moduli).
    //  @param[in]  triArea     triangulation size (relative to the
    //                          cross-section's bounding box.
    *///////////////////////////////////////////////////////////////////////////
    VRep interior(Real triArea = 0.001) const;

    virtual size_t numParams() const = 0;
    virtual void setParams(const std::vector<Real> &p) = 0;
    virtual std::vector<Real> params() const = 0;

    virtual std::unique_ptr<CrossSection> copy() const = 0;

    const static AlignedPointCollection EMPTY;
    virtual const AlignedPointCollection &holePts() const { return EMPTY; }

    // Linearly interpolate between two cross-sections.
    static std::unique_ptr<CrossSection> lerp(const CrossSection &a, const CrossSection &b, Real alpha) {
        auto lerpParams = a.params();
        auto bParams = b.params();
        const size_t np = lerpParams.size();
        if (np != b.numParams()) throw std::runtime_error("Parameter size mismatch");

        for (size_t i = 0; i < np; ++i)
            lerpParams[i] = lerpParams[i] * (1.0 - alpha) + bParams[i] * alpha;

        auto result = a.copy();
        result->setParams(lerpParams);
        result->E  = a.E  * (1.0 - alpha) + b.E  * alpha;
        result->nu = a.nu * (1.0 - alpha) + b.nu * alpha;

        return result;
    }

    virtual ~CrossSection() { }

    Real E = 0, nu = 0;
};

// Cross-section with a small number of separately-stored parameters
struct ParametricCrossSection : public CrossSection {
    ParametricCrossSection() { }

    ParametricCrossSection(const std::vector<Real> &p)
        : m_params(p) { m_validateParams(); }

    ParametricCrossSection(const ParametricCrossSection &b)
        : CrossSection(b), m_params(b.m_params) { }

    virtual void setParams(const std::vector<Real> &p) override {
        if (p.size() != numParams()) throw std::runtime_error("Trying to set incorrect number of parameters");
        m_params = p;
    }

    virtual std::vector<Real> params() const override { return m_params; }

protected:
    std::vector<Real> m_params;
    void m_validateParams() const { if (m_params.size() != numParams()) throw std::runtime_error("Incorrect number of parameters"); }
};

#endif /* end of include guard: CROSSSECTION_HH */
