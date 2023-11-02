#include "TargetSurfaceFitter.hh"
#include "infer_target_surface.hh"
#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/Parallelism.hh>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_simplex_squared_distance.h>
#include <igl/AABB.h>
#include <fstream>

#include "TargetSurfaceFitterMesh.hh"

struct TargetSurfaceAABB : public igl::AABB<Eigen::MatrixXd, 3> {
    using Base = igl::AABB<Eigen::MatrixXd, 3>;
    using Base::Base;
};

void TargetSurfaceFitter::constructTargetSurface(const RodLinkage &linkage, size_t loop_subdivisions, size_t num_extension_layers, Eigen::Vector3d scale_factors) {
    try {
        infer_target_surface(linkage, m_tgt_surf_V, m_tgt_surf_F, /* smoothing iterations */ loop_subdivisions, /* num extension layers */ num_extension_layers, /* manipulate the target surface by scaling */scale_factors);
    }
    catch (std::exception &e) {
        std::cerr << "ERROR: failed to infer target surface: " << e.what() << std::endl;
        std::cerr << "You must load a new target surface or set the joint position weight to 1.0" << std::endl;

        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> quads;
        linkage.visualizationGeometry(vertices, quads);
        BBox<Eigen::Vector3d> bb(vertices);
        Eigen::Vector3d minC = bb.minCorner;
        Eigen::Vector3d maxC = bb.maxCorner;

        // We need *some* target surface with the same general position/scale
        // as the linkage or the libigl-based viewer will not work properly.
        m_tgt_surf_V.resize(3, 3);
        m_tgt_surf_F.resize(1, 3);
        m_tgt_surf_V << minC[0], minC[1], minC[2],
                        maxC[0], minC[1], minC[2],
                        maxC[0], maxC[1], maxC[2];
        m_tgt_surf_F << 0, 1, 2;
    }
    setTargetSurface(linkage, m_tgt_surf_V, m_tgt_surf_F);
}

void TargetSurfaceFitter::setTargetSurface(const RodLinkage &linkage, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    m_tgt_surf_V = V;
    m_tgt_surf_F = F;
    igl::per_face_normals(m_tgt_surf_V, m_tgt_surf_F, m_tgt_surf_N);

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    igl_to_meshio(m_tgt_surf_V, m_tgt_surf_F, vertices, elements);
    target_surface = std::make_unique<TargetSurfaceMesh>(elements, vertices.size());

    m_tgt_surf_aabb_tree = std::make_unique<TargetSurfaceAABB>();
    m_tgt_surf_aabb_tree->init(m_tgt_surf_V, m_tgt_surf_F);

    updateClosestPoints(linkage);

    static size_t i = 0;
    igl_to_meshio(m_tgt_surf_V, m_tgt_surf_F, vertices, elements);
    // MeshIO::save("target_surface_" + std::to_string(i) + ".msh", vertices, elements);
    // linkage.writeLinkageDebugData("linkage_" + std::to_string(i) + ".msh");
    ++i;
}

void TargetSurfaceFitter::loadTargetSurface(const RodLinkage &linkage, const std::string &path) {
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    MeshIO::load(path, vertices, elements);
    std::cout << "Loaded " << vertices.size() << " vertices and " << elements.size() << " triangles" << std::endl;
    meshio_to_igl(vertices, elements, m_tgt_surf_V, m_tgt_surf_F);
    setTargetSurface(linkage, m_tgt_surf_V, m_tgt_surf_F);
}

void TargetSurfaceFitter::saveTargetSurface(const std::string &path) {
    // Check if the vertex/faces matrices are none?
    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    igl_to_meshio(m_tgt_surf_V, m_tgt_surf_F, vertices, elements);
    std::cout << "Saved " << vertices.size() << " vertices and " << elements.size() << " triangles" << std::endl;
    MeshIO::save(path, vertices, elements);
}

template<typename Real_>
void TargetSurfaceFitter::forceUpdateClosestPoints(const RodLinkage_T<Real_> &linkage) {
    if (!m_tgt_surf_aabb_tree) return;
    const size_t numSamplePts = m_useCenterline ? linkage.numCenterlinePos() : linkage.numJoints();

    // If we have nonzero weights in the surface-fitting term,
    // or if the closest point array is uninitialized,
    // update each joint's closest surface point.
    if ((size_t(linkage_closest_surf_pts.size()) == 3 * numSamplePts) && (Wsurf_diag_linkage_sample_pos.norm() == 0.0)) return;

    BENCHMARK_SCOPED_TIMER_SECTION timer("Update closest points");
    linkage_closest_surf_pts.resize(3 * numSamplePts);
    linkage_closest_surf_pt_sensitivities.resize(numSamplePts);
    linkage_closest_surf_tris.resize(numSamplePts);

    int numInterior = 0, numBdryEdge = 0, numBdryVtx = 0;

    using Range = tbb::blocked_range<size_t>;
    tbb::parallel_for(Range(0, numSamplePts), [&](const Range &b) {
        for (size_t pt_i = b.begin(); pt_i < b.end(); ++pt_i) {
            int closest_idx;
            // Could be parallelized (libigl does this internally for multi-point queries)
            Eigen::RowVector3d p, query;
            if (m_useCenterline) {
                query = stripAutoDiff(linkage.centerLinePositions()).template segment<3>(pt_i * 3).transpose();
            } else {
                const auto &j = linkage.joint(pt_i);
                query = stripAutoDiff(j.pos()).transpose();
            }
            Real sqdist = m_tgt_surf_aabb_tree->squared_distance(m_tgt_surf_V, m_tgt_surf_F, query, closest_idx, p);
            linkage_closest_surf_pts.segment<3>(3 * pt_i) = p.transpose();
            linkage_closest_surf_tris[pt_i] = closest_idx;

            // Compute the sensitivity of the closest point projection with respect to the query point (dp_dx).
            // There are three cases depending on whether the closest point lies in the target surface's
            // interior, on one of its boundary edges, or on a boundary vertex.
            Eigen::RowVector3d barycoords;
            igl::point_simplex_squared_distance<3>(query,
                                                   m_tgt_surf_V, m_tgt_surf_F, closest_idx,
                                                   sqdist, p, barycoords);

            std::array<int, 3> boundaryNonzeroLoc;
            int numNonzero = 0, numBoundaryNonzero = 0;
            for (int i = 0; i < 3; ++i) {
                if (barycoords[i] == 0.0) continue;
                ++numNonzero;
                // It is extremely unlikely a vertex will be closest to a point/edge if this is not a stable association.
                // Therefore we assume even for smoothish surfaces that points are constrained to lie on their closest
                // simplex.
                // Hack away the old boundry-snapping-only behavior: treat all non-boundary edges/vertices as active too...
                // TODO: decide on this!
                // if (target_surface->vertex(m_tgt_surf_F(closest_idx, i)).isBoundary())
                    boundaryNonzeroLoc[numBoundaryNonzero++] = i;
            }
            assert(numNonzero >= 1);

            if ((numNonzero == 3) || (numNonzero != numBoundaryNonzero)) {
                // If the closest point lies in the interior, the sensitivity is (I - n n^T) (the query point perturbation is projected onto the tangent plane).
                linkage_closest_surf_pt_sensitivities[pt_i] = Eigen::Matrix3d::Identity() - m_tgt_surf_N.row(closest_idx).transpose() * m_tgt_surf_N.row(closest_idx);
                ++numInterior;
            }
            else if ((numNonzero == 2) && (numBoundaryNonzero == 2)) {
                // If the closest point lies on a boundary edge, we assume it can only slide along this edge (i.e., the constraint is active)
                // (The edge orientation doesn't matter.)
                Eigen::RowVector3d e = m_tgt_surf_V.row(m_tgt_surf_F(closest_idx, boundaryNonzeroLoc[0])) -
                                       m_tgt_surf_V.row(m_tgt_surf_F(closest_idx, boundaryNonzeroLoc[1]));
                e.normalize();
                linkage_closest_surf_pt_sensitivities[pt_i] = e.transpose() * e;
                ++numBdryEdge;
            }
            else if ((numNonzero == 1) && (numBoundaryNonzero == 1)) {
                // If the closest point coincides with a boundary vertex, we assume it is "stuck" there (i.e., the constraint is active)
                linkage_closest_surf_pt_sensitivities[pt_i].setZero();
                ++numBdryVtx;
            }
            else {
                assert(false);
            }
        }
    });
}


// Visualization functions
std::vector<Real> TargetSurfaceFitter::get_squared_distance_to_target_surface(Eigen::VectorXd query_point_list) const {
    std::vector<Real> output(query_point_list.size() / 3);
    for (size_t pt_i = 0; pt_i < size_t(query_point_list.size()/3); ++pt_i) {
        int closest_idx;
        // Could be parallelized (libigl does this internally for multi-point queries)
        Eigen::RowVector3d p, query;
        query = query_point_list.segment<3>(pt_i * 3).transpose();

        Real sqdist = m_tgt_surf_aabb_tree->squared_distance(m_tgt_surf_V, m_tgt_surf_F, query, closest_idx, p);
        output[pt_i] = sqdist;
    }
    return output;
}

Eigen::VectorXd TargetSurfaceFitter::get_closest_point_for_visualization(Eigen::VectorXd query_point_list) const {
    Eigen::VectorXd output(query_point_list.size());
    for (size_t pt_i = 0; pt_i < size_t(query_point_list.size()/3); ++pt_i) {
        int closest_idx;
        // Could be parallelized (libigl does this internally for multi-point queries)
        Eigen::RowVector3d p, query;
        query = query_point_list.segment<3>(pt_i * 3).transpose();
        m_tgt_surf_aabb_tree->squared_distance(m_tgt_surf_V, m_tgt_surf_F, query, closest_idx, p);
        output.segment<3>(3 * pt_i) = p.transpose();
    }
    return output;
}

Eigen::VectorXd TargetSurfaceFitter::get_closest_point_normal(Eigen::VectorXd query_point_list) {
    Eigen::VectorXd output(query_point_list.size());
    igl::per_vertex_normals(m_tgt_surf_V, m_tgt_surf_F, m_tgt_surf_VN);

    for (size_t pt_i = 0; pt_i < size_t(query_point_list.size()/3); ++pt_i) {
        int closest_idx;
        // Could be parallelized (libigl does this internally for multi-point queries)
        Eigen::RowVector3d p, query, barycoords;

        query = query_point_list.segment<3>(pt_i * 3).transpose();
        Real sqdist = m_tgt_surf_aabb_tree->squared_distance(m_tgt_surf_V, m_tgt_surf_F, query, closest_idx, p);
        igl::point_simplex_squared_distance<3>(query,
                                               m_tgt_surf_V, m_tgt_surf_F, closest_idx,
                                               sqdist, p, barycoords);
        Eigen::Vector3d interpolated_normal(0, 0, 0);
        for (int i = 0; i < 3; ++i) interpolated_normal += barycoords[i] * m_tgt_surf_VN.row(m_tgt_surf_F(closest_idx, i));
        interpolated_normal.normalized();
        output.segment<3>(3 * pt_i) = interpolated_normal;
    }
    return output;
}

TargetSurfaceFitter:: TargetSurfaceFitter() = default;
TargetSurfaceFitter::~TargetSurfaceFitter() = default;

template void TargetSurfaceFitter::forceUpdateClosestPoints<Real>(const RodLinkage_T<Real> &linkage); // explicit instantiation.
template void TargetSurfaceFitter::forceUpdateClosestPoints<ADReal>(const RodLinkage_T<ADReal> &linkage); // explicit instantiation.
