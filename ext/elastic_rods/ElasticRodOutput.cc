#include <MeshFEM/MSHFieldWriter.hh>
#include "ElasticRod.hh"
#include <memory>
#include <functional>

using VField = VectorField<Real, 3>;
using SField = ScalarField<Real>;

template<typename Real_>
void ElasticRod_T<Real_>::writeDebugData(const std::string &path) const {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;

    const size_t nv = numVertices(), ne = numEdges();

    for (auto &p : deformedPoints())
        vertices.emplace_back(stripAutoDiff(p));
    for (size_t i = 0; i < ne; ++i) 
        elements.emplace_back(i, i + 1);

    MSHFieldWriter writer(path, vertices, elements);

    const auto &dc = deformedConfiguration();
    VField referenceD1(ne), materialD1(ne);
    VField restD1(ne);

    for (size_t j = 0; j < ne; ++j) {
        restD1(j)      = stripAutoDiff(m_restDirectors[j].d1);
        referenceD1(j) = stripAutoDiff(dc.referenceDirectors[j].d1);
        materialD1(j)  = stripAutoDiff(dc.materialFrame[j].d1);
    }

    SField referenceTwist(nv);
    VField curvatureBinormal(nv);
    for (size_t i = 0; i < nv; ++i) {
        referenceTwist[i] = stripAutoDiff(dc.referenceTwist[i]);
        curvatureBinormal(i) = stripAutoDiff(dc.kb[i]);
    }

    SField restLen(ne), len(ne);
    for (size_t j = 0; j < ne; ++j) {
        restLen[j] = stripAutoDiff(m_restLen[j]);
        len[j] = stripAutoDiff(dc.len[j]);
    }

    writer.addField("rest len",       restLen,           DomainType::PER_ELEMENT);
    writer.addField("len",            len,               DomainType::PER_ELEMENT);
    writer.addField("rest d1",        restD1,            DomainType::PER_ELEMENT);
    writer.addField("reference d1",   referenceD1,       DomainType::PER_ELEMENT);
    writer.addField("material d1",    materialD1,        DomainType::PER_ELEMENT);
    writer.addField("referenceTwist", referenceTwist,    DomainType::PER_NODE);
    writer.addField("kb",             curvatureBinormal, DomainType::PER_NODE);

    auto gradPToVField = [nv](const Gradient &g) {
        VField result(nv);
        for (size_t i = 0; i < nv; ++i)
            result(i) = stripAutoDiff(g.gradPos(i).eval());
        return result;
    };

    writer.addField("grad stretch energy", gradPToVField(gradEnergyStretch()), DomainType::PER_NODE);
    writer.addField("grad bend energy",    gradPToVField(gradEnergyBend()),    DomainType::PER_NODE);
    writer.addField("grad twist energy",   gradPToVField(gradEnergyTwist()),   DomainType::PER_NODE);
    writer.addField("grad energy",         gradPToVField(gradEnergy()),        DomainType::PER_NODE);
}

// Visualization data "colored" by a height scalar field that will help us visualize the top and
// bottom surfaces of the rods
template<typename Real_>
void ElasticRod_T<Real_>::coloredVisualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                                                       std::vector<MeshIO::IOElement> &quads,
                                                       const bool averagedMaterialFrames,
                                                       const bool averagedCrossSections,
                                                       Eigen::VectorXd *height,
                                                       Eigen::VectorXd *stress,
                                                       CrossSectionStressAnalysis::StressType type) const {
    const size_t ne = numEdges();
    const auto &dc = deformedConfiguration();
    using Pts = typename CrossSection::AlignedPointCollection;

    const bool outputtingHeights   = (height != nullptr);
    const bool stressVisualization = (stress != nullptr);
    std::vector<double> heightStl, stressStl;
    if (outputtingHeights)   heightStl.assign(height->data(), height->data() + height->size());
    if (stressVisualization) stressStl.assign(stress->data(), stress->data() + stress->size());

    std::function<const Pts *(size_t)> ptsGetter;
    if (stressVisualization) ptsGetter = [this](size_t i) -> const Pts * { return &(material(i).stressAnalysis().boundaryV); };
    else                     ptsGetter = [this](size_t i) -> const Pts * { return &(material(i).crossSectionBoundaryPts); };

    // Construct a generalized cylinder for each edge in the rod.
    for (size_t j = 0; j < ne; ++j) {
        // We need separate cross-section points for each endpoint in case we are averaging the edges' cross-sections for
        // vertex-continuous visualization geometry.
        const Pts *crossSectionPts_a = ptsGetter(j),
                  *crossSectionPts_b = ptsGetter(j);

        auto &crossSectionEdges = stressVisualization ? material(j).stressAnalysis().boundaryE
                                                      : material(j).crossSectionBoundaryEdges;

        Vec3 d1_a = dc.materialFrame[j].d1,
             d2_a = dc.materialFrame[j].d2;
        Vec3 d1_b = d1_a,
             d2_b = d2_a;

        if (averagedMaterialFrames) {
            if (j >      0) { d1_a += dc.materialFrame[j - 1].d1; d2_a += dc.materialFrame[j - 1].d2; d1_a *= 0.5; d2_a *= 0.5; }
            if (j < ne - 1) { d1_b += dc.materialFrame[j + 1].d1; d2_b += dc.materialFrame[j + 1].d2; d1_b *= 0.5; d2_b *= 0.5; }
        }

        // If averaging cross-sections, replace the begin/end cross sections with the average with the previous/next edge
        // (if one exists)
        std::unique_ptr<Pts> csp_a_uptr, csp_b_uptr;
        if (averagedCrossSections) {
            auto average = [](const Pts &pts_a, const Pts &pts_b) {
                if (pts_a.size() != pts_b.size()) throw std::runtime_error("Cross-section contour size mismatch in interpolation");
                Pts result;
                result.reserve(pts_a.size());
                for (size_t i = 0; i < pts_a.size(); ++i)
                    result.push_back(0.5 * (pts_a[i] + pts_b[i]));
                return result;
            };

            if (j > 0) {
                csp_a_uptr = std::make_unique<Pts>(average(*crossSectionPts_a, *ptsGetter(j - 1)));
                crossSectionPts_a = csp_a_uptr.get();
            }

            if (j < ne - 1) {
                csp_b_uptr = std::make_unique<Pts>(average(*crossSectionPts_b, *ptsGetter(j + 1)));
                crossSectionPts_b = csp_b_uptr.get();
            }
        }

        size_t offset = vertices.size();

        // First, create copies of the cross section points for both cylinder end caps
        auto &cpa = *crossSectionPts_a;
        auto &cpb = *crossSectionPts_b;
        for (size_t k = 0; k < cpa.size(); ++k) {
            // Vec3 bdryVec = dc.materialFrame[j].d1 * crossSectionPts[k][0] + dc.materialFrame[j].d2 * crossSectionPts[k][1];
            vertices.emplace_back(stripAutoDiff((dc.point(j    ) + d1_a * cpa[k][0] + d2_a * cpa[k][1]).eval()));
            vertices.emplace_back(stripAutoDiff((dc.point(j + 1) + d1_b * cpb[k][0] + d2_b * cpb[k][1]).eval()));
            if (outputtingHeights) { heightStl.push_back(cpa[k][1]); heightStl.push_back(cpb[k][1]); }
        }
        if (stressVisualization) {
            Eigen::VectorXd stresses_a, stresses_b;
            for (size_t i = j; i <= j + 1; ++i) {
                visitVertexStrains(i, [&](size_t corner, Real_ tau, const Vec2_T<Real_> &curvatureNormal, Real_ eps_s) {
                    Eigen::VectorXd &dst = (i == j) ? stresses_a : stresses_b;
                    if (corner == j) dst = material(j).stressAnalysis().boundaryVertexStresses(type, stripAutoDiff(tau), stripAutoDiff(curvatureNormal), stripAutoDiff(eps_s));
                });
            }
            if (size_t(stresses_a.size()) != cpa.size()) throw std::logic_error("stresses_a incorrect size for j = " + std::to_string(j));
            if (size_t(stresses_b.size()) != cpa.size()) throw std::logic_error("stresses_b incorrect size for j = " + std::to_string(j));
            for (size_t k = 0; k < cpa.size(); ++k) {
                stressStl.push_back(stresses_a[k]);
                stressStl.push_back(stresses_b[k]);
            }
        }

        for (const auto &ce : crossSectionEdges) {
            // The cross-section edges are oriented ccw in the d1-d2 plane,
            // (looking along the rod's minus tangent vector).
            quads.emplace_back(offset + 2 * ce.first  + 1,
                               offset + 2 * ce.first  + 0,
                               offset + 2 * ce.second + 0,
                               offset + 2 * ce.second + 1);
        }
    }

    if (outputtingHeights)   *height = Eigen::Map<Eigen::VectorXd>(heightStl.data(), heightStl.size());
    if (stressVisualization) *stress = Eigen::Map<Eigen::VectorXd>(stressStl.data(), stressStl.size());
}

// Append this rod's data to existing geometry/scalar field.
template<typename Real_>
void ElasticRod_T<Real_>::stressVisualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                                                      std::vector<MeshIO::IOElement> &quads,
                                                      Eigen::VectorXd &sqrtBendingEnergy,
                                                      Eigen::VectorXd &stretchingStress,
                                                      Eigen::VectorXd &maxBendingStress,
                                                      Eigen::VectorXd &minBendingStress,
                                                      Eigen::VectorXd &twistingStress) const {
    size_t rod_output_offset = vertices.size();
    visualizationGeometry(vertices, quads);
    sqrtBendingEnergy.conservativeResize(vertices.size());
    stretchingStress .conservativeResize(vertices.size());
    maxBendingStress .conservativeResize(vertices.size());
    minBendingStress .conservativeResize(vertices.size());
    twistingStress   .conservativeResize(vertices.size());

    auto              bendingEnergy = energyBendPerVertex();
    auto stretchingStressCenterline = stretchingStresses();
    auto    bendingStressCenterline =    bendingStresses();
    auto   twistingStressCenterline =   twistingStresses();

    const size_t ne = numEdges();
    size_t edge_offset = 0;
    for (size_t j = 0; j < ne; ++j) {
        const size_t numCrossSectionPts = material(j).crossSectionBoundaryPts.size();
        for (size_t i = 0; i < numCrossSectionPts; ++i) {
            for (size_t adj_vtx = 0; adj_vtx < 2; ++adj_vtx) {
                int outIdx = rod_output_offset + edge_offset + 2 * i + adj_vtx;
                sqrtBendingEnergy(outIdx) = std::sqrt(stripAutoDiff(bendingEnergy[j + adj_vtx]));
                 stretchingStress(outIdx) = stretchingStressCenterline[j];
                 maxBendingStress(outIdx) =    bendingStressCenterline(j + adj_vtx, 0);
                 minBendingStress(outIdx) =    bendingStressCenterline(j + adj_vtx, 1);
                   twistingStress(outIdx) =   twistingStressCenterline[j + adj_vtx];
            }
        }
        edge_offset += 2 * numCrossSectionPts;
    }
}

template<typename Real_>
void ElasticRod_T<Real_>::saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames, const bool averagedCrossSections) const {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> quads;
    visualizationGeometry(vertices, quads, averagedMaterialFrames, averagedCrossSections);
    MeshIO::save(path, vertices, quads, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);
}

////////////////////////////////////////////////////////////////////////////////
// Explicit instantiation for ordinary double type and autodiff type.
////////////////////////////////////////////////////////////////////////////////
template struct ElasticRod_T<double>;
template struct ElasticRod_T<ADReal>;
