import numpy as np
import pythreejs
import ipywidgets
import ipywidgets.embed
import MeshFEM
import tri_mesh_viewer
import elastic_rods
import reflection

# Render a linkage or a single elastic rod.
class LinkageViewer(tri_mesh_viewer.TriMeshViewer):
    def __init__(self, rodObject, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None, transparent=False, wireframe=False):
        if isinstance(rodObject, elastic_rods.PeriodicRod):
            rodObject = rodObject.rod
        super().__init__(rodObject, width=width, height=height, textureMap=textureMap, scalarField=scalarField, vectorField=vectorField, superView=superView, transparent=transparent, wireframe=wireframe)

    @property
    def averagedMaterialFrames(self):
        return self._averagedMaterialFrames

    @averagedMaterialFrames.setter
    def averagedMaterialFrames(self, value):
        self._averagedMaterialFrames = value
        self.update(scalarField=self.scalarField, vectorField=self.vectorField)

    @property
    def averagedCrossSections(self):
        return self._averagedCrossSections

    @averagedCrossSections.setter
    def averagedCrossSections(self, value):
        self._averagedCrossSections = value
        self.update(scalarField=self.scalarField, vectorField=self.vectorField)

    def getVisualizationGeometry(self):
        # Hack to support visualization of triangle meshes too
        if not reflection.hasArg(self.mesh.visualizationGeometry, 'averagedMaterialFrames'):
            return self.mesh.visualizationGeometry()

        # Note: getVisualizationGeometry is called by TriMeshViewer's constructor,
        # so we can initialize our member variables here. (So we don't need to
        # implement a __init__ method that forwards arguments and breaks tab completion).
        if not hasattr(self, '_averagedMaterialFrames'):
            self._averagedMaterialFrames = False
            self._averagedCrossSections  = False
            # redraw flicker isn't usually a problem for the linkage
            # deployment--especially since the index buffer isn't changing--and
            # we prefer to enable smooth interaction during deployment
            self.avoidRedrawFlicker = False

        return self.mesh.visualizationGeometry(self._averagedMaterialFrames, self._averagedCrossSections)

class CenterlineMesh:
    def __init__(self, rodOrLinkage):
        self.rodOrLinkage = rodOrLinkage

    def visualizationGeometry(self):
        rods = []
        rodOrLinkage = self.rodOrLinkage
        if isinstance(rodOrLinkage, elastic_rods.ElasticRod):
            rods = [rodOrLinkage]
        elif isinstance(rodOrLinkage, elastic_rods.RodLinkage):
            rods = [s.rod for s in rodOrLinkage.segments()]
        else: raise Exception('Unsupported object type')

        V = np.empty((0, 3), dtype=np.float32)
        E = np.empty((0, 2), dtype=np. uint32)
        N = np.empty((0, 3), dtype=np.float32)
        for r in rods:
            idxOffset = V.shape[0]
            V = np.row_stack([V, np.array(r.deformedPoints(), dtype=np.float32)])
            E = np.row_stack([E, idxOffset + np.column_stack([np.arange(r.numVertices() - 1, dtype=np.uint32), np.arange(1, r.numVertices(), dtype=np.uint32)])])
            # Use the d2 directory averaged onto the vertices as the per-vertex normal
            padded = np.pad([np.array(f.d2, dtype=np.float32) for f in r.deformedConfiguration().materialFrame], [(1, 1), (0, 0)], mode='edge') # duplicate the first and last edge frames
            N = np.row_stack([N, 0.5 * (padded[:-1] + padded[1:])])
        return V, E, N

    # No decoding needed for per-entity fields on raw meshes.
    def visualizationField(self, data):
        return data

class CenterlineViewer(tri_mesh_viewer.LineMeshViewer):
    def __init__(self, rodOrLinkage, width=512, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None):
        super().__init__(CenterlineMesh(rodOrLinkage), width=width, height=height, scalarField=scalarField, vectorField=vectorField, superView=superView)

from enum import Enum
class LinkageViewerWithSurface(LinkageViewer):
    class ViewType(Enum):
        LINKAGE = 1
        SURFACE = 2

    class ViewOption():
        def __init__(self, transparent = False, color = 'lightgray'):
            self.transparent = transparent
            self.color = color

    def __init__(self, straight_linkage, target_surf, wireframeSurf=False, *args, **kwargs):
        if (isinstance(target_surf, str)):
            import mesh
            target_surf = mesh.Mesh(target_surf)

        super().__init__(straight_linkage, *args, **kwargs)
        self.surfView = tri_mesh_viewer.TriMeshViewer(target_surf, superView=self, wireframe=wireframeSurf)

        self.averagedMaterialFrames = True
        self.averagedCrossSections  = True

        self.viewOptions = {self.ViewType.LINKAGE: self.ViewOption(False, 'green'),
                            self.ViewType.SURFACE: self.ViewOption(True, 'gray')}
        self.applyViewOptions()
        if not self.scalarField is None:
            self.showScalarField(self.scalarField)

    def applyViewOptions(self):
        if (not hasattr(self, 'viewOptions')): return # Triggered from __init__ as side effect of update

        for k, v in self.viewOptions.items():
            if (v.color is None or v.transparent is None):
                if (v.transparent is not None or v.color is not None):
                    print('WARNING: either both or neither of {color, transparency} must both None')
                continue
            if v.transparent: self.subview(k).makeTransparent(v.color)
            else:             self.subview(k).makeOpaque     (v.color)

    def subview(self, viewType):
        if (viewType == self.ViewType.LINKAGE): return self.linkageView()
        if (viewType == self.ViewType.SURFACE): return self.surfaceView()
        else: raise Exception('unknown view type')

    def offscreenRenderer(self, width = None, height = None):
        orender = super().offscreenRenderer(width, height)

        attr = self.surfaceView().meshes.children[0].geometry.attributes
        P = attr['position'].array
        N = attr['normal'].array
        F = attr['index'].array if 'index' in attr else None
        orender.addMesh(P, F, N, '#D3D3D3', makeDefault=False)
        orender.meshes[-1].alpha = 0.25
        orender.meshes[-1].modelMatrix(self.objects.position, self.objects.scale, self.objects.quaternion)

        return orender

    def linkageView(self):
        return self

    def surfaceView(self):
        return self.surfView

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.applyViewOptions() # This is rather hacky and will introduce flickering if we use non-default colors or transparency
        if not self.scalarField is None:
            self.showScalarField(self.scalarField)

    def updateTargetSurf(self, vertices, faces, wireframeSurf=False):
        import mesh
        target_surf   = mesh.Mesh(*(vertices, faces))
        self.surfView = tri_mesh_viewer.TriMeshViewer(target_surf, superView=self, wireframe=wireframeSurf)
        self.applyViewOptions()

    def updateTargetSurfFromPath(self, pathToTargetSurf, wireframeSurf=False):
        import mesh
        target_surf   = mesh.Mesh(pathToTargetSurf)
        self.surfView = tri_mesh_viewer.TriMeshViewer(target_surf, superView=self, wireframe=wireframeSurf)
        self.applyViewOptions()

    def showScalarField(self, sfield):
        super().update(scalarField=sfield)
        # calling `self.applyViewOptions()` overwrites the scalar field!

################################################################################
# Convenience functions for constructing viewers
################################################################################
def getColoredRodOrientationViewer(linkage, bottomColor=[0.5, 0.5, 0.5], topColor=[1.0, 0.0, 0.0], **kwargs):
    heights = linkage.visualizationGeometryHeightColors()
    colors = np.take(np.array([bottomColor, topColor]), heights < heights.mean(), axis=0)
    return LinkageViewer(linkage, scalarField=colors, **kwargs)

def getGraphOverTargetViewer(graph_path, surf_path, offset_dist = 5e-3):
    """
    Visualize the linkage graph superimposed over the target surface. We offset the
    graph nodes by `offset_dist` in the surface's normal direction so that the
    graph isn't obscured by the target surface.
    """
    import mesh, linkage_optimization
    P, E = mesh.load_raw(graph_path)
    l = elastic_rods.RodLinkage(P, E, initConsistentAngle=False)
    tsf = linkage_optimization.TargetSurfaceFitter()
    tsf.loadTargetSurface(l, surf_path)
    P += offset_dist * tsf.N[tsf.linkage_closest_surf_tris]
    surfview = tri_mesh_viewer.TriMeshViewer(mesh.Mesh(surf_path))
    return tri_mesh_viewer.LineMeshViewer((P, E), superView=surfview)
