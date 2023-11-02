////////////////////////////////////////////////////////////////////////////////
// Custom.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A custom cross-section represented by a boundary contour.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  03/27/2020 15:32:30
////////////////////////////////////////////////////////////////////////////////
#ifndef CUSTOM_CROSSSECTION_HH
#define CUSTOM_CROSSSECTION_HH
#include <string>
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/filters/merge_duplicate_vertices.hh>

namespace CrossSections {

struct Custom : public ::CrossSection {
    Custom(const std::string &path, Real scale = 1.0) {
        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> lines;
        MeshIO::load(path, vertices, lines, MeshIO::FMT_GUESS, MeshIO::MESH_LINE);

        // Apply scaling to the cross-section vertices
        for (auto &v : vertices) v.point *= scale;

        // Detect dangling vertices; these are interpreted as hole points.
        std::vector<bool> dangling(vertices.size(), true);
        for (const auto &e : lines)
            for (size_t vi : e) dangling.at(vi) = false;

        for (size_t i = 0; i < vertices.size(); ++i)
            if (dangling[i]) m_holePts.emplace_back(truncateFrom3D<Point2D>(vertices[i].point));

        if (m_holePts.size())
            std::cout << "Read " << m_holePts.size() << " hole points (dangling vertices)" << std::endl;

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
        auto &crossSectionBoundaryPts = m_brep.first;
        auto &crossSectionBoundaryEdges = m_brep.second;

        crossSectionBoundaryPts.clear();
        crossSectionBoundaryPts.reserve(vertices.size());
        for (const auto &v : vertices)
            crossSectionBoundaryPts.emplace_back(truncateFrom3D<Point2D>(v));
        crossSectionBoundaryEdges.clear();
        for (const auto &e : lines) {
            if (e.size() != 2) throw std::runtime_error("Non line element found in the contour mesh");
            crossSectionBoundaryEdges.push_back({e[0], e[1]});
        }
    }

    Custom(const Custom &b) : CrossSection(b), m_brep(b.m_brep), m_holePts(b.m_holePts) { }

    virtual size_t numParams() const override { return 2 * bdryPts().size(); }

    virtual BRep boundary(bool /* hiRes */) const override {
        return m_brep;
    }

    virtual void setParams(const std::vector<Real> &p) override {
        if (p.size() != numParams()) throw std::runtime_error("Incorrect number of parameters");
        size_t i = 0;
        for (auto &pt : m_brep.first) {
            pt[0] = p[i++];
            pt[1] = p[i++];
        }
        assert(i == p.size());
    }

    virtual std::vector<Real> params() const override {
        std::vector<Real> result(numParams());
        size_t i = 0;
        for (const auto &pt : m_brep.first) {
            result[i++] = pt[0];
            result[i++] = pt[1];
        }
        assert(i == result.size());
        return result;
    }

    const AlignedPointCollection &bdryPts  () const { return m_brep.first; }
    const EdgeCollection         &bdryEdges() const { return m_brep.second; }

    const AlignedPointCollection &holePts  () const override { return m_holePts; }

    virtual std::unique_ptr<CrossSection> copy() const override { return std::make_unique<Custom>(*this); }

protected:
    BRep m_brep;
    AlignedPointCollection m_holePts;
};

}

#endif
