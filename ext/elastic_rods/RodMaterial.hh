////////////////////////////////////////////////////////////////////////////////
// RodMaterial.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Determines the stretching, twisting, and bending stiffness for an elastic
//  rod. This is a function of the elastic material and the geometry of the rod
//  cross section.
//  It is assumed that the cross section's principal axes of the moment of
//  inertia tensor are aligned with the material axes so that the bending
//  stiffness quadratic form is represented by a diagonal matrix.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  05/31/2018 14:32:19
////////////////////////////////////////////////////////////////////////////////
#ifndef RODMATERIAL_HH
#define RODMATERIAL_HH

#include <cmath>
#include <vector>
#include <string>
#include <MeshFEM/Types.hh>
#include <MeshFEM/Geometry.hh>

#include "CrossSection.hh"
#include "CrossSectionStressAnalysis.hh"

#include <Eigen/StdVector> // Work around alignment issues with std::vector

// Forward declare CrossSectionMesh to avoid bringing in FEMMesh when unnecessary
class CrossSectionMesh;

struct RodMaterial {
    struct DiagonalizedTensor {
        Real lambda_1, lambda_2;
        Real get(size_t i) const {
            if (i == 0) return lambda_1;
            if (i == 1) return lambda_2;
            throw std::runtime_error("Invalid index: " + std::to_string(i));
        }
        Real trace() const { return lambda_1 + lambda_2; }

        DiagonalizedTensor operator+(const DiagonalizedTensor &b) const { return DiagonalizedTensor{lambda_1 + b.lambda_1, lambda_2 + b.lambda_2}; }
        DiagonalizedTensor operator*(Real s)                      const { return DiagonalizedTensor{lambda_1 * s         , lambda_2 * s         }; }
        DiagonalizedTensor &operator=(Real val) { lambda_1 = lambda_2 = val; return *this; }
    };

    using BendingStiffness = DiagonalizedTensor;
    enum class StiffAxis { D1, D2 };

    // Constructors must be implemented in .cc because of forward-declared CrossSectionMesh.
    RodMaterial();
    RodMaterial(const std::string &cross_section_path, StiffAxis stiffAxis = StiffAxis::D1, bool keepCrossSectionMesh = false, const std::string &debug_psi_path = std::string()) {
        set(*CrossSection::load(cross_section_path), stiffAxis, keepCrossSectionMesh, debug_psi_path);
    }

    RodMaterial(const std::string &type, Real E, Real nu, const std::vector<Real> &params, StiffAxis stiffAxis = StiffAxis::D1, bool keepCrossSectionMesh = false);
    RodMaterial(const CrossSection &cs, StiffAxis stiffAxis = StiffAxis::D1, bool keepCrossSectionMesh = false);

    // The primary method for specifying a rod's material properties is via a
    // cross-section object.
    void set(const CrossSection &cs, StiffAxis stiffAxis = StiffAxis::D1, bool keepCrossSectionMesh = false, const std::string &debug_psi_path = std::string());

    // Set the rod material properties using a cross-section description.
    void set(const std::string &type, Real E, Real nu, const std::vector<Real> &params, StiffAxis stiffAxis = StiffAxis::D1, bool keepCrossSectionMesh = false) {
        set(*CrossSection::construct(type, E, nu, params), stiffAxis, keepCrossSectionMesh);
    }

    // Use a spatially constant isotropic elastic material (E, nu) filling an
    // elliptical cross section.
    // (Assumed to be oriented along rest reference frame: a is width along d1 axis)
    // Note, there is an error in the twisting stiffness formula from
    // Bergou2010; the correct formula is:
    //      G (a^3 b^3 M_PI) / (a^2 + b^2) = G (a^2 b^2 * area) / (a^2 + b^2).
    // The formula in Bergou2010,
    //      0.25 * G * area * (a^2 + b^2),
    // is correct only for a circular cross section a = b = r:
    //      G r^4 area / (2 r^2) = 0.5 G r^2 area = 0.25 G area 2 r^2).
    void setEllipse(Real E, Real nu, Real a, Real b) {
        area = M_PI * a * b;
        Real G = E / (2 * (1 + nu));
        stretchingStiffness = E * area;
        twistingStiffness   = G * area * a * a * b * b / (a * a + b * b);
        bendingStiffness    = {0.25 * E * area * a * a, 0.25 * E * area * b * b};
        momentOfInertia     = {0.25 *     area * b * b, 0.25 *     area * a * a};

        const size_t nsubdiv = 20;
        crossSectionBoundaryPts.clear(), crossSectionBoundaryPts.reserve(nsubdiv);
        crossSectionBoundaryEdges.clear(), crossSectionBoundaryEdges.reserve(nsubdiv);
        for (size_t k = 0; k < nsubdiv; ++k) {
            // Generate points in ccw order in the d1-d2 plane.
            Real phi = (2.0 * M_PI * k) / nsubdiv;
            crossSectionBoundaryPts.emplace_back(a * cos(phi), b * sin(phi));
            crossSectionBoundaryEdges.push_back({k, (k + 1) % nsubdiv});
        }

        youngModulus = E;
        shearModulus = G;
        crossSectionHeight = std::min(a, b);
        // TODO: torsionStressCoefficient...
    }

    // Use a spatially constant isotropic elastic material (E, nu) filling
    // a given cross-section geometry.
    void setMesh(Real E, Real nu, const std::string &path, StiffAxis stiffAxis, bool keepCrossSectionMesh);

    // Use a spatially constant isotropic elastic material (E, nu) filling
    // a cross-section whose boundary is specified by a line mesh.
    // If the cross-section has holes, these must be specified by
    // a single point inside each hole. These points are represented by
    // dangling vertices in the line mesh.
    void setContour(Real E, Real nu, const std::string &path, Real scale, StiffAxis stiffAxis,
                    bool keepCrossSectionMesh = false, const std::string &debug_psi_path = std::string(),
                    Real triArea = 0.001, size_t simplifyVisualizationMesh = 0);

    void meshCrossSection(Real triArea = 0.001) {
        if (!m_crossSection) throw std::runtime_error("No cached cross-section in this material instance, cannot mesh the cross section.");
        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> elements;
        std::tie(vertices, elements) = m_crossSection->interior(triArea);

        Eigen::Matrix2d R;
        Point2D cm;
        std::tie(cm, R) = m_computeStiffnesses(m_crossSection->E, m_crossSection->nu, vertices, elements, m_stiffAxis, true, "");
        // m_crossSectionMesh = std::make_shared<CrossSectionMesh>(elements, vertices);
    }

    bool hasCrossSection()     const { return m_crossSection!=nullptr; }
    bool hasCrossSectionMesh() const { return m_crossSectionMesh!=nullptr; }
    const CrossSectionMesh &crossSectionMesh() const {
        if (!m_crossSectionMesh.get()) throw std::runtime_error("No cross-section mesh.");
        return *m_crossSectionMesh;
    }
    CrossSectionMesh &crossSectionMesh() {
        if (!m_crossSectionMesh.get()) throw std::runtime_error("No cross-section mesh.");
        return *m_crossSectionMesh;
    }
    std::shared_ptr<CrossSectionMesh> crossSectionMeshPtr() const { return m_crossSectionMesh; }
    void releaseCrossSectionMesh() { m_crossSectionMesh.reset(); }

    // Compute the maximum tensile and compressive z stresses arising from bending
    // the material as discussed in docs/structural_analysis.tex.
    // Return [max, min]
    Vector2D bendingStresses(const Vector2D &curvatureNormal) const {
        Real sigma_max = 0.0,
             sigma_min = 0.0;
        for (const Point2D &p : crossSectionBoundaryPts) {
            Real sigma = -curvatureNormal.dot(p);
            sigma_max = std::max(sigma_max, sigma);
            sigma_min = std::min(sigma_min, sigma);
        }

        return youngModulus * Vector2D(sigma_max, sigma_min);
    }

    const CrossSection &crossSection() const {
        if (!m_crossSection) throw std::runtime_error("No cached cross-section in this material instance.");
        return *m_crossSection;
    }
    StiffAxis getStiffAxis() const { return m_stiffAxis; }

    // Destructor must be implemented in .cc because of forward-declared CrossSectionMesh.
    ~RodMaterial();

    ////////////////////////////////////////////////////////////////////////////
    // Material properties (derived from cross section)
    // NOTE: when adding to these, be sure to update all `set` methods and
    // the Python binding's pickling functions.
    ////////////////////////////////////////////////////////////////////////////
    Real area = 1;
    Real stretchingStiffness = 1, twistingStiffness = 1;
    BendingStiffness bendingStiffness{1, 1};
    DiagonalizedTensor momentOfInertia{1, 1};

    // S^j from doc/structural_analysis.tex; units: MPa * mm
    Real torsionStressCoefficient = std::nan("");
    Real youngModulus = 1, shearModulus = 1; // needed, e.g., for stress analysis

    // Used for offsetting rods to be in contact at joints.
    Real crossSectionHeight = 0.0;

    // Polygonal representation of the rod cross-section used for visualization.
    // These are the points in the d1, d2 coordinate system.
    CrossSection::AlignedPointCollection crossSectionBoundaryPts;
    CrossSection::EdgeCollection         crossSectionBoundaryEdges;

    const CrossSectionStressAnalysis &stressAnalysis() const;

    // For serialization
    std::shared_ptr<CrossSectionStressAnalysis> stressAnalysisPtr() const { return m_crossSectionStressAnalysis; }
    void setStressAnalysisPtr(std::shared_ptr<CrossSectionStressAnalysis> ptr) { m_crossSectionStressAnalysis = ptr; }

private:
    // Returns the transformation rotating the cross-section geometry's major
    // principal axis to the +x axis.
    // Also updates the cross-section bounding box.
    std::pair<Point2D, Eigen::Matrix2d>
    m_computeStiffnesses(Real E, Real nu,
                         std::vector<MeshIO::IOVertex> vertices, // copy is modified...
                         const std::vector<MeshIO::IOElement> &elements,
                         StiffAxis stiffAxis = StiffAxis::D1,
                         bool keepCrossSectionMesh = false,
                         const std::string &debug_psi_path = std::string());


    // Cached cross-section and stiffAxis information needed to interpolate materials
    // by interpolating cross-sections.
    std::shared_ptr<CrossSection> m_crossSection;
    StiffAxis m_stiffAxis = StiffAxis::D2;

    // Optional mesh of the cross-section's interior (in case an integration needs to be performed over the cross-section)
    // Shared so that we easily copy the material.
    std::shared_ptr<CrossSectionMesh> m_crossSectionMesh;
    // Optional quantities needed for performing stress analysis. These involve
    // a fair bit of storage for complicated centerlines, so we use a shared
    // pointer so that they can be shared across copies.
    mutable std::shared_ptr<CrossSectionStressAnalysis> m_crossSectionStressAnalysis;
};

#endif /* end of include guard: RODMATERIAL_HH */
