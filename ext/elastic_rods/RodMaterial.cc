#include "RodMaterial.hh"

#include <MeshFEM/MeshIO.hh>

#include <vector>
#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Fields.hh>
#include <MeshFEM/Triangulate.h>
#include <MeshFEM/FieldPostProcessing.hh>
#include <MeshFEM/filters/merge_duplicate_vertices.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/Utilities/EdgeSoupAdaptor.hh>
#include "3rdparty/visvalingam_simplify/src/visvalingam_algorithm.h"

#include "CrossSectionMesh.hh"
#include <MeshFEM/Laplacian.hh>

// Constructors/destructor
// Note: keepCrossSectionMesh is currently only needed for our finite
// difference tests of the mass matrix (which require computing integrals over
// the cross-section); all other integrals are evaluated within this class when
// the material is configured.
RodMaterial::RodMaterial() { }
RodMaterial::RodMaterial(const std::string &type, Real E, Real nu, const std::vector<Real> &params, StiffAxis stiffAxis, bool keepCrossSectionMesh) { set(type, E, nu, params, stiffAxis, keepCrossSectionMesh); }
RodMaterial::RodMaterial(const CrossSection &cs, StiffAxis stiffAxis, bool keepCrossSectionMesh) { set(cs, stiffAxis, keepCrossSectionMesh); }
RodMaterial::~RodMaterial() { }

void RodMaterial::set(const CrossSection &cs, StiffAxis stiffAxis, bool keepCrossSectionMesh, const std::string &debug_psi_path) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    std::tie(vertices, elements) = cs.interior(0.001);

    Eigen::Matrix2d R;
    Point2D cm;
    std::tie(cm, R) = m_computeStiffnesses(cs.E, cs.nu, vertices, elements, stiffAxis, keepCrossSectionMesh, debug_psi_path);
    std::tie(crossSectionBoundaryPts, crossSectionBoundaryEdges) = cs.boundary();
    for (auto &v : crossSectionBoundaryPts) v = (R * (v - cm));
    m_crossSection = cs.copy();
    // MeshIO::save("cross_section_boundary.msh", EdgeSoup<CrossSection::AlignedPointCollection, CrossSection::EdgeCollection>(crossSectionBoundaryPts, crossSectionBoundaryEdges));
}

void RodMaterial::setMesh(Real E, Real nu, const std::string &path, StiffAxis stiffAxis, bool keepCrossSectionMesh) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> tris;
    MeshIO::load(path, vertices, tris, MeshIO::FMT_GUESS, MeshIO::MESH_TRI);

    m_computeStiffnesses(E, nu, vertices, tris, stiffAxis, true); // keep the cross-section mesh around for extracting the boundary...

    ////////////////////////////////////////////////////////////////////////////
    // Extract the cross-section boundary.
    const auto &mesh = crossSectionMesh();
    crossSectionBoundaryPts.clear();
    crossSectionBoundaryPts.reserve(mesh.numBoundaryVertices());
    for (const auto bv : mesh.boundaryVertices())
        crossSectionBoundaryPts.emplace_back(bv.volumeVertex().node()->p);
    for (const auto be : mesh.boundaryEdges())
        crossSectionBoundaryEdges.push_back({be.tail().index(),
                                             be. tip().index()});

    if (!keepCrossSectionMesh) m_crossSectionMesh.reset();
}

void RodMaterial::setContour(Real E, Real nu, const std::string &path, Real scale, StiffAxis stiffAxis,
                             bool keepCrossSectionMesh, const std::string &debug_psi_path, Real triArea,
                             size_t simplifyVisualizationMesh /* = 0 */) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> lines;
    MeshIO::load(path, vertices, lines, MeshIO::FMT_GUESS, MeshIO::MESH_LINE);

    // Apply scaling to the cross-section vertices
    for (auto &v : vertices) v.point *= scale;

    // Detect dangling vertices; these are interpreted as hole points.
    std::vector<bool> dangling(vertices.size(), true);
    for (const auto &e : lines)
        for (size_t vi : e) dangling.at(vi) = false;

    std::vector<Point2D> holePts;
    for (size_t i = 0; i < vertices.size(); ++i)
        if (dangling[i]) holePts.emplace_back(truncateFrom3D<Point2D>(vertices[i].point));
    if (holePts.size())
        std::cout << "Read " << holePts.size() << " hole points (dangling vertices)" << std::endl;

    // Remove the dangling vertices from the line mesh.
    size_t curr = 0;
    std::vector<size_t> vertexRenumber(vertices.size(), std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (!dangling[i]) {
            vertices[curr] = vertices[i];
            vertexRenumber[i] = curr++;
        }
    }
    for (auto &e : lines)
        for (size_t &vi : e)
            vi = vertexRenumber.at(vi);
    vertices.resize(curr);

    // Merge duplicate vertices in the contour
    BBox<Point2D> bb(vertices);
    merge_duplicate_vertices(vertices, lines, vertices, lines, 1e-8 * bb.dimensions().norm());
    for (const auto &e : lines)
        if (e[0] == e[1]) throw std::runtime_error("Merging duplicates collapsed an edge");

    // Strip z dimension, storing the boundary
    crossSectionBoundaryPts.clear();
    crossSectionBoundaryPts.reserve(vertices.size());
    for (const auto &v : vertices)
        crossSectionBoundaryPts.emplace_back(truncateFrom3D<Point2D>(v));
    crossSectionBoundaryEdges.clear();
    for (const auto &e : lines) {
        if (e.size() != 2) throw std::runtime_error("Non line element found in the contour mesh");
        crossSectionBoundaryEdges.push_back({e[0], e[1]});
    }

    // Triangulate the contour
    std::vector<MeshIO::IOVertex > triangulatedVertices;
    std::vector<MeshIO::IOElement> triangles;
    triangulatePSLG(crossSectionBoundaryPts, crossSectionBoundaryEdges, holePts,
                    triangulatedVertices, triangles, triArea * bb.volume(), "Q");

    Eigen::Matrix2d R;
    Point2D cm;
    std::tie(cm, R) = m_computeStiffnesses(E, nu, triangulatedVertices, triangles, stiffAxis, keepCrossSectionMesh, debug_psi_path);
    for (auto &v : crossSectionBoundaryPts) v = (R * (v - cm));

    if (simplifyVisualizationMesh) {
        const size_t npts = crossSectionBoundaryPts.size();
        // inefficient...
        std::vector<std::vector<size_t>> adj(npts);
        for (const auto &e : crossSectionBoundaryEdges) {
            adj.at(e.first).push_back(e.second);
            adj.at(e.second).push_back(e.first);
        }

        for (const auto &a : adj)
            if (a.size() != 2) throw std::runtime_error("All contours must be closed and simple; valence other than 2 detected");

        // Extract loops to simplify.
        std::vector<bool> visited(npts, false);
        std::vector<visvalingam_simplify::Linestring> contours;
        for (size_t i = 0; i < npts; ++i) {
            if (visited[i]) continue;
            contours.emplace_back();
            auto &contour = contours.back();

            auto addPt = [&](const Point2D &p) { contour.emplace_back(p[0], p[1]); };

            size_t u = i, prev = i;
            while (!visited[u]) {
                addPt(crossSectionBoundaryPts[u]);
                visited[u] = true;
                size_t next = (adj[u][0] == prev) ? adj[u][1] : adj[u][0]; // step to the next point around the curve
                prev = u;
                u = next;
            }
            if (u != i) {
                std::cout << "i: " << i << std::endl;
                std::cout << "u: " << u << std::endl;
                throw std::runtime_error("All contours must be closed and simple; traversal error"); // we should have gotten back to where we started
            }
            // Note: simplifier doesn't operate on closed curves, but it at least won't
            // delete the first or last point. This means one edge in the profile will never
            // collapse/disappear.
        }

        // Simplify each loop
        std::vector<visvalingam_simplify::Linestring> simplifiedContours;
        for (const auto &contour : contours) {
            simplifiedContours.emplace_back();
            visvalingam_simplify::Visvalingam_Algorithm simplifier(contour);
            simplifier.simplify(simplifyVisualizationMesh, simplifiedContours.back());
        }

        // Convert back to edge soup
        crossSectionBoundaryPts.clear();
        crossSectionBoundaryEdges.clear();
        EdgeSoupFromClosedPolygonCollection<std::vector<visvalingam_simplify::Linestring>> esoup(simplifiedContours);
        for (const auto &pt : esoup.points()) crossSectionBoundaryPts.emplace_back(pt.X, pt.Y);
        for (const auto &e  : esoup.edges())  crossSectionBoundaryEdges.push_back(e);
    }
}

// Twisting stiffness and stress quantities are computed first by solving for
// the out-of-plane cross-section displacement field psi putting the rod in
// static equilibrium under an applied twist. The static equilibrium conditions
// amount to a Laplace equation with Neumann conditions:
//      - Laplacian psi = 0           in  Omega
//        d psi / dn    = (y, -x) * n on dOmega
// For more details, see Landau and Lifshitz.
// The twisting stiffness is computed then as:
//      G int_Omega || grad psi + (-y, x)^T ||^2 dx,
// while the nonzero components of the strain induced by unit twist are just:
//      [ eps_zx, eps_yz]^T = grad psi + (-y, x)^T
// We compute and return this shear strain 2-vector field as a piecewise linear
// (per-vetex) field.
Eigen::MatrixXd torsionAnalysis(const CrossSectionMesh &mesh, const std::string &debug_psi_path = "") {
    auto L = Laplacian::construct(mesh);

    constexpr size_t Deg = 2;
    constexpr size_t K   = 2;

    // Compute load vector for the Neuman condition:
    //      d psi / dn  = (y, -x) * n := g.n
    // l_i = int_be (phi_i g.n) dA
    // where g = (y, -x)^T
    std::vector<Real> rhs(mesh.numNodes());
    for (const auto be : mesh.boundaryElements()) {
        Interpolant<Vector2D, K - 1, 1> g;
        for (const auto v : be.vertices()) {
            const auto &p = v.node().volumeNode()->p;
            g[v.localIndex()] = Vector2D(p[1], -p[0]);
        }

        // Accumulate boundary element's contribution to each boundary node's load.
        for (const auto bn : be.nodes()) {
            Interpolant<Real, K - 1, Deg> phi;
            phi = 0;
            phi[bn.localIndex()] = 1.0;

            rhs[bn.volumeNode().index()] +=
                Quadrature<K - 1, Deg + 1>::integrate(
                    [&] (const EvalPt<K - 1> &p) {
                        return phi(p) * g(p).dot(be->normal());
                }, be->volume());
        }
    }

    SPSDSystem<Real> system(L);
    // Fix a single value so that the solution is unique.
    // We use only the psi's gradient, so this value/node can be arbitrary.
    system.fixVariables(std::vector<size_t>(1, 0), std::vector<Real>(1, 0.0));
    auto psi = system.solve(rhs);

    if (!debug_psi_path.empty()) {
        MSHFieldWriter writer(debug_psi_path, mesh);
        VectorField<Real, 3> defo(psi.size());
        defo.clear();
        for (size_t i = 0; i < psi.size(); ++i)
            defo(i)[2] = psi[i];
        writer.addField("psi", ScalarField<Real>(psi), DomainType::PER_NODE);
        writer.addField("u", defo, DomainType::PER_NODE);
    }

    // Note that `sigma_vec` is a piecewise linear but discontinuous interpolant
    // field. We postprocess it into a continuous piecewise linear (per-vertex) field.
    // This should improve accuracy slightly and simplifies subsequent usage.
    return vertexAveragedFieldEigen(mesh, [&](size_t ei, size_t evalLocalVtxIndex) {
            const auto e = mesh.element(ei);
            const auto &p = e.vertex(evalLocalVtxIndex).node()->p;
            Eigen::Vector2d result(-p[1], p[0]);
            for (const auto n : e.nodes())
                result += psi[n.index()] * e->gradPhi(n.localIndex())[evalLocalVtxIndex];
            return result;
        });
}

// Returns the transformation rotating the cross-section geometry's major
// principal axis to the +x axis.
std::pair<Point2D, Eigen::Matrix2d>
RodMaterial::m_computeStiffnesses(Real E, Real nu,
                                  std::vector<MeshIO::IOVertex> vertices,
                                  const std::vector<MeshIO::IOElement> &elements,
                                  StiffAxis stiffAxis,
                                  bool keepCrossSectionMesh,
                                  const std::string &debug_psi_path) {
    const Real G = E / (2 * (1 + nu));

    constexpr size_t K = 2;

    m_crossSectionMesh = std::make_shared<CrossSectionMesh>(elements, vertices);
    auto &mesh = crossSectionMesh();
    auto cm = mesh.centerOfMass();

    ////////////////////////////////////////////////////////////////////////////
    // Translate the center of mass to the origin (the neutral surface passes
    // through the center of mass, so the bending stiffness matrix is determined
    // by the moment of inertia tensor computed around the center of mass).
    {
        auto cm3d = padTo3D(cm);
        for (auto &v : vertices) v.point -= cm3d;
        mesh.setNodePositions(vertices);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Stretching stiffness is just E times the cross-sectional area.
    area = mesh.volume();
    stretchingStiffness = E * area;

    ////////////////////////////////////////////////////////////////////////////
    // Bending stiffness is E times the cross-section's moment of inertia
    auto computeInertiaTensor = [&]() {
        SymmetricMatrixValue<Real, 2> I;
        for (const auto e : mesh.elements()) {
            // Build linear interpolant that evaluates the coordinate functions.
            Interpolant<Vector2D, K, 1> pos;
            for (const auto v : e.vertices())
                pos[v.localIndex()] = v.node()->p;
            // I = int [  y^2  -xy ] dA
            //         [ -xy    x^2]
            I += Quadrature<K, 2>::integrate(
                    [&pos] (const EvalPt<K> &p) {
                        return SymmetricMatrixValue<Real, 2>(Vector3D(
                         pos(p)[1] * pos(p)[1],
                         pos(p)[0] * pos(p)[0],
                        -pos(p)[0] * pos(p)[1]));
                    }, e->volume()
                );
        }
        return I;
    };

    auto I = computeInertiaTensor();
    // std::cout << I << std::endl;

    // Rotate the cross-section so that its principal axes align with our
    // coordinate system (diagonalizing the moment of inertia tensor)
    Vector2D lambda;
    Eigen::Matrix2d Q;
    std::tie(lambda, Q) = I.eigenDecomposition();
    if (stiffAxis == StiffAxis::D2) {
        // Orient the stiff direction along the d2
        // (joint normal) direction.
        Q.col(0).swap(Q.col(1));
        std::swap(lambda[0], lambda[1]);
    }
    if (Q.determinant() < 0) Q.col(0) *= -1;
    {
        // Rotate the major principal axis (first eigenvector) to the +x axis.
        for (auto &v : vertices) v.point = padTo3D((Q.transpose() * truncateFrom3D<Point2D>(v.point)).eval());
        mesh.setNodePositions(vertices);
    }

    // std::cout << "Updated I:" << std::endl << computeInertiaTensor() << std::endl;

    momentOfInertia = DiagonalizedTensor{lambda[0], lambda[1]};

    // The bending stiffness tensor "E I" derived in Landau and Lifshitz
    // defines a bilinear form acting on the **curvature binormal** (the
    // rotation axis for the cross-sections). However, our bending energy
    // quadratic form acts on the **curvature normal**, so the principal
    // moments must be swapped.
    bendingStiffness = BendingStiffness{E * lambda[1], E * lambda[0]};

    youngModulus = E;
    shearModulus = G;

    // "Torsion stress coefficient" S^j from doc/structural_analysis.tex is the
    // maximum of G ||sigma_vec|| over the cross-section.
    Eigen::MatrixXd unitTwistShearStrain = torsionAnalysis(mesh, debug_psi_path);
    torsionStressCoefficient = G * std::sqrt(unitTwistShearStrain.rowwise().squaredNorm().maxCoeff());
    twistingStiffness = 0;
    for (const auto e : mesh.elements()) {
        twistingStiffness += Quadrature<K, 2>::integrate(
                [&](const EvalPt<K> &p) {
                    return (p[0] * unitTwistShearStrain.row(e.vertex(0).index())
                          + p[1] * unitTwistShearStrain.row(e.vertex(1).index())
                          + p[2] * unitTwistShearStrain.row(e.vertex(2).index())).squaredNorm();
                }, e->volume());
    }
    twistingStiffness *= G;

    // Note: m_crossSectionMesh has been rotated so that the d2 frame vector is aligned with the y axis.
    crossSectionHeight = m_crossSectionMesh->boundingBox().dimensions()[1];

    // Needed to implement interpolation feature.
    m_stiffAxis = stiffAxis;

    if (!keepCrossSectionMesh) m_crossSectionMesh.reset();

    return std::make_pair(cm, Q.transpose());
}

const CrossSectionStressAnalysis &RodMaterial::stressAnalysis() const {
    if (m_crossSectionStressAnalysis) {
        m_crossSectionStressAnalysis->youngModulus = youngModulus;
        m_crossSectionStressAnalysis->shearModulus = shearModulus;
        return *m_crossSectionStressAnalysis;
    }
    if (!m_crossSectionMesh) throw std::runtime_error("A cross-section mesh is needed to construct a stress analysis object!");
    m_crossSectionStressAnalysis = std::make_shared<CrossSectionStressAnalysis>(crossSectionMesh(), torsionAnalysis(crossSectionMesh()), youngModulus, shearModulus);
    return *m_crossSectionStressAnalysis;
}
