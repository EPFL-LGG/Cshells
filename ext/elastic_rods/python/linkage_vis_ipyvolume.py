import ipyvolume as ipv
import numpy as np
import uuid

class LinkageViewer:
    def __init__(self, linkage, material):
        self.linkage = linkage
        self.material = material
        self.translate = None
        self.bbSize = None
        self.key = uuid.uuid4()
        self.update(False)
    
    def update(self, keep, l = None, mat = None):
        if (l != None):   self.linkage = l
        if (mat != None): self.material = mat
        (pts, tris, normals) = self.linkage.visualizationGeometry(self.material)
        if (self.translate is None):
            self.translate = -np.mean(pts, axis=0)
        pts += self.translate
        fig = ipv.figure(key=self.key, width=512, height=512)
        # ipv.style.use({'background-color': 'orange'})
        # ipv.style.axes_off()

        m = ipv.Mesh(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                     triangles=tris.astype(dtype=np.uint32), color='#DDEEFF')
        if (keep):
            for m2 in fig.meshes:
                m2.color='#111111'
            fig.meshes = fig.meshes + [m]
        else:
            fig.meshes = [m]

        if (self.bbSize is None):
            self.bbSize = np.max(np.abs(pts))
        ipv.xyzlim(-self.bbSize, self.bbSize)
            
    def show(self):
        ipv.figure(key=self.key)
        ipv.show()
