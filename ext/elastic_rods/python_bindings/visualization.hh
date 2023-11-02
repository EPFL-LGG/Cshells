#ifndef VISUALIZATION_HH
#define VISUALIZATION_HH

using VisualizationGeometry = std::tuple<Eigen::Matrix<float,    Eigen::Dynamic, 3>,  // Pts
                                         Eigen::Matrix<uint32_t, Eigen::Dynamic, 3>,  // Tris
                                         Eigen::Matrix<float,    Eigen::Dynamic, 3>>; // Normals

// Convert MeshIO quad mesh to a triangulated mesh with normals.
inline
VisualizationGeometry visualizationGeometry(const std::vector<MeshIO::IOVertex > &vertices,
                                            const std::vector<MeshIO::IOElement> &elements) {
    const size_t nq = elements.size(); // Elements are planar quads
    const size_t nt = 2 * nq;          // that we triangulate.
    const size_t nv = 4 * nq;          // We duplicate the quad's corners (for per-face normals)

    VisualizationGeometry result;
    auto &pts     = std::get<0>(result);
    auto &tris    = std::get<1>(result);
    auto &normals = std::get<2>(result);
    pts.resize(nv, 3);
    tris.resize(nt, 3);
    normals.resize(nv, 3);

    for (size_t i = 0; i < nq; ++i) {
        const auto &quad = elements[i];
        if (quad.size() != 4) throw std::runtime_error("Expected quads");

        pts.row(4 * i + 0) = vertices[quad[0]].point.cast<float>();
        pts.row(4 * i + 1) = vertices[quad[1]].point.cast<float>();
        pts.row(4 * i + 2) = vertices[quad[2]].point.cast<float>();
        pts.row(4 * i + 3) = vertices[quad[3]].point.cast<float>();

        Eigen::Vector3f n = (pts.row(4 * i + 1) - pts.row(4 * i + 0)).cross(pts.row(4 * i + 3) - pts.row(4 * i + 0)).normalized();
        normals.row(4 * i + 0) = n;
        normals.row(4 * i + 1) = n;
        normals.row(4 * i + 2) = n;
        normals.row(4 * i + 3) = n;

        tris.row(2 * i + 0) << 4 * i + 0, 4 * i + 1, 4 * i + 2;
        tris.row(2 * i + 1) << 4 * i + 0, 4 * i + 2, 4 * i + 3;
    }

    return result;
}

template<class Object>
VisualizationGeometry getVisualizationGeometry(const Object &obj, const bool averagedMaterialFrames = true, const bool averagedCrossSections = true) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    obj.visualizationGeometry(vertices, elements, averagedMaterialFrames, averagedCrossSections);
    return visualizationGeometry(vertices, elements);

}

// Replicate per-quad/quad-corner data to per-triangle/triangle-corner data
template<class FieldType>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
quadToTriField(const std::vector<MeshIO::IOVertex > &quadVertices,
               const std::vector<MeshIO::IOElement> &quadElements,
               const FieldType &quadVisField) {
    const size_t fs = quadVisField.rows();
    const size_t nq = quadElements.size();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> result;
    if (fs == quadVertices.size()) {
        result.resize(4 * nq, quadVisField.cols());
        for (size_t i = 0; i < nq; ++i) {
            const auto &q = quadElements[i];
            result.row(4 * i + 0) = quadVisField.row(q[0]);
            result.row(4 * i + 1) = quadVisField.row(q[1]);
            result.row(4 * i + 2) = quadVisField.row(q[2]);
            result.row(4 * i + 3) = quadVisField.row(q[3]);
        }
        return result;
    }

    if (fs == nq) {
        result.resize(2 * nq, quadVisField.cols());
        for (size_t i = 0; i < nq; ++i) {
            result.row(2 * i + 0) = quadVisField.row(i);
            result.row(2 * i + 1) = quadVisField.row(i);
        }
        return result;
    }

    throw std::runtime_error("Unexpected field size " + std::to_string(fs));
}

template<class Object, class FieldType>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
getVisualizationField(const Object &obj, const FieldType &field) {
    auto quadVisField = obj.visualizationField(field);

    // TODO: avoid rebuilding the geometry...
    std::vector<MeshIO::IOVertex > quadVertices;
    std::vector<MeshIO::IOElement> quadElements;
    obj.visualizationGeometry(quadVertices, quadElements);

    return quadToTriField(quadVertices, quadElements, quadVisField);
}

template<class Object>
Eigen::VectorXd getVisualizationGeometryCSHeightField(const Object &obj) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> quads;

    Eigen::VectorXd quadHeightField;
    obj.coloredVisualizationGeometry(vertices, quads, false, false, &quadHeightField);
    return quadToTriField(vertices, quads, quadHeightField);
}

template<class Object>
auto getVisualizationWithStress(const Object &obj, bool averagedMaterialFrames, bool averagedCrossSections, CrossSectionStressAnalysis::StressType type) {
    std::tuple<VisualizationGeometry, Eigen::VectorXd> result;

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> quads;
    Eigen::VectorXd stress;
    obj.coloredVisualizationGeometry(vertices, quads, averagedMaterialFrames, averagedCrossSections, /* height = */ nullptr, &stress, type);
    return std::make_tuple(visualizationGeometry(vertices, quads), quadToTriField(vertices, quads, stress));
}

#endif /* end of include guard: VISUALIZATION_HH */
