#ifndef TARGETSURFACEFITTERMESH_HH
#define TARGETSURFACEFITTERMESH_HH

#include <MeshFEM/TriMesh.hh>

struct TargetSurfaceMesh : public TriMesh<> {
    using TriMesh<>::TriMesh;
};

#endif /* end of include guard: TARGETSURFACEFITTERMESH_HH */
