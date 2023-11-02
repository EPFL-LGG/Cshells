try:    import _offscreen_renderer
except: raise Exception('Could not load compiled module; is OffscreenRenderer missing a dependency?')

from _offscreen_renderer import *
import os
import numpy as np
import scipy.spatial
import video_writer

SHADER_DIR = os.path.dirname(__file__) + '/../shaders'

class ShaderLibrary:
    """
    Load GLSL shaders from files, with caching.
    """
    def __init__(self, ctx):
        self.shaders = {}
        self.ctx = ctx
    def load(self, vtxFile, fragFile, geoFile = ""):
        files = [vtxFile, fragFile]
        if geoFile != "": files.append(geoFile)
        files = tuple(files)
        if files not in self.shaders:
            self.shaders[files] = Shader(self.ctx, *[open(f).read() for f in files])
        return self.shaders[files]

class OpenGLContext(_offscreen_renderer.OpenGLContext):
    def array(self, unpremultiply = True):
        self.finish() # Copy image to internal buffer
        # Reshape flat buffer into (height, width, components) 3D array
        # and flip vertically (since OpenGL's image coordinate system is vertically flipped
        # with respect to PIL/png/etc.)
        buf = self.unpremultipliedBuffer() if unpremultiply else self.buffer()
        return buf.reshape((self.height, self.width, 4))[::-1,:,:]

    def image(self, unpremultiply = True):
        from PIL import Image
        return Image.fromarray(self.array(unpremultiply))

    def save(self, path, unpremultiply = True):
        ext = os.path.splitext(path)[-1][1:].lower()
        if (ext not in ['png', 'ppm']): raise Exception('Output file extension not supported')
        self.finish() # Copy image to internal buffer
        if ext == 'png': self.writePNG(path, unpremultiply)
        if ext == 'ppm': self.writePPM(path, unpremultiply)

    def shaderLibrary(self):
        if not hasattr(self, '_shaderLib'):
            self._shaderLib = ShaderLibrary(self)
        return self._shaderLib

def hexColorToFloat(c):
    """
    Pythreejs likes using hex color strings like #FFFFFF; convert
    them to OpenGL-compatible float arrays.
    """
    if not isinstance(c, str): return c
    if len(c) != 7: raise Exception('Invalid hex color')
    return np.array([int(c[i:i+2], 16) / 255 for i in range(1, len(c), 2)])

def decodeColor(c):
    """
    Allow specifying colors by (r, g, b) triplets, hex codes, or X11 names.
    (decodes into an (r, g, b) triplet)
    """
    import colors
    if isinstance(c, str):
        if c[0] == '#' and len(c) == 7:
            c = hexColorToFloat(c)
        elif c in colors.x11_colors:
            c = hexColorToFloat(colors.x11_colors[c])
        else: raise Exception('Unrecognized color: ' + c)
    return c

def normalize(v):
    return v / np.linalg.norm(v)

def lookAtMatrix(position, target, up):
    viewDir = normalize(np.array(target) - np.array(position))
    right   = normalize(np.cross(viewDir, up))
    upPerp  = normalize(np.cross(right, viewDir))
    matView = np.identity(4)
    matView[0, 0:3] = right
    matView[1, 0:3] = upPerp
    matView[2, 0:3] = -viewDir
    matView[0:3, 3] = -matView[0:3, 0:3] @ position
    matView[3, 0:4] = [0, 0, 0, 1]
    return matView

class Mesh:
    def __init__(self, ctx, V, F, N, color):
        self.ctx = ctx

        self.alpha = 1.0 # Global opacity of the mesh
        self._meshColorOpaque = True # to be determined from the user-passed color
        self._sorted = False

        self.setWireframe(0.0)
        self.matModel = np.identity(4)
        self.shininess = 20.0

        # The triangle index array in active use to replicate vertex data to
        # per-corner data.
        self._activeReplicationIndices = None

        self.vao = None

        self.setMesh(V, F, N, color)

    def _validateSizes(self, V, F, N, color):
        """ Side effect: update constColor """
        ccolor = isinstance(color, str) or len(np.ravel(color)) in [3, 4] # `str` case probably corresponds to RGB hex code '#FFFFFF'
        self.constColor = ccolor
        nv = self.numVertices
        if len(V) != nv:                      raise Exception(f'Unexpected vertex array size {len(V)} vs {nv} (use setMesh if changing connectivity)')
        if len(N) != nv:                      raise Exception(f'Unexpected normal array size {len(N)} vs {nv} (must be per-vertex)')
        if not ccolor and (len(color) != nv): raise Exception(f'Unexpected `color` size {len(N)} vs {nv} (must be per-vertex or constant)')
        if F is not None and F.max() >= nv:   raise Exception('Corner index out of bounds')

    def setMesh(self, V, F, N, color):
        """
        Initialize/update the mesh data and connectivity.
        `F` can be `None` to disable indexed face set representation
        (i.e., to use glDrawArrays instead of glDrawElements)
        """
        self.ctx.makeCurrent()

        self.numVertices = len(V)
        self._validateSizes(V, F, N, color)
        self.F = F
        self._activeReplicationIndices = None

        if self.vao is None: # TODO: figure out why eliminating this check leads to a memory leak
            # (It's definitely wasteful to keep deleting/recreating our VAO, but it shouldn't leak!)
            self.vao = VertexArrayObject(self.ctx)

        if self.F is not None:
            self.vao.setIndexBuffer(self.F)

        self.updateMeshData(V, N, color)

    def isOpaque(self): return (self.alpha == 1.0) and self._meshColorOpaque
    def needsDepthSort(self):
        return not (self._sorted or self.isOpaque())

    def _depthSort(self, matView):
        """
        If the mesh has translucency, sort the triangles from back to front.
        To avoid shuffling vertex attribute data when the viewpoint changes, we
        always use an index buffer when depth sorting.
        """
        if not self.needsDepthSort(): return

        if self.F is None:
            self.F = np.arange(self.numVertices)

        # Sort the triangles in increasing order by their barycenters'
        # z coordinates in eye space. Note, we discard the irrelevant
        # translation part of `matModelView` to avoid working with
        # homogeneous coordinates.
        order = np.argsort((self.V[self.F].reshape((-1, 3, 3)).mean(axis=1) @ (matView @ self.matModel)[0:3, 0:3].T)[:, 2])
        self.F = self.F.reshape((-1, 3))[order]

        self.ctx.makeCurrent()
        self.vao.setIndexBuffer(self.F)

    def updateMeshData(self, V, N, color = None):
        """
        Update the mesh's data without changing its connectivity.
        """
        self.ctx.makeCurrent()

        if (color is None): color = self.color
        self._validateSizes(V, None, N, color) # updates `self.constColor`

        color = decodeColor(color)

        # Replicate the data to per-corner if in replication mode.
        if self._activeReplicationIndices is not None:
            V = V[self._activeReplicationIndices]
            N = N[self._activeReplicationIndices]
            if not self.constColor: color = color[self._activeReplicationIndices]

        # Keep track of the vertex data in case it we need to switch to replicating per
        # incident triangle (e.g., for wireframe rendering)
        self.V = V
        self.N = N
        self.color = color

        self.vao.setAttribute(0, V)
        self.vao.setAttribute(1, N)

        if self.constColor:
            self.vao.setConstantAttribute(2, color)
            self._meshColorOpaque = (len(color) == 3) or (color[3] == 1.0)
        else:
            self.vao.setAttribute(2, color)
            self._meshColorOpaque = (color.shape[1] == 3) or (color[:, 3].min() == 1.0)

    def setColor(self, color):
        """
        Update the mesh color without changing its geometry.
        """
        self.ctx.makeCurrent()
        self._validateSizes(self.V, None, self.N, color) # updates `self.constColor`

        # Replicate the data to per-corner if in replication mode.
        if self._activeReplicationIndices is not None:
            if not self.constColor: color = color[self._activeReplicationIndices]

        color = decodeColor(color)
        self.color = color

        if self.constColor:
            self.vao.setConstantAttribute(2, color)
            self._meshColorOpaque = (len(color) == 3) or (color[3] == 1.0)
            print('set const color: ', color)
        else:
            self.vao.setAttribute(2, color)
            self._meshColorOpaque = (color.shape[1] == 3) or (color[:, 3].min() == 1.0)

    def modelMatrix(self, position, scale, quaternion):
        self.matModel[0:3, 0:3] = scale * scipy.spatial.transform.Rotation.from_quat(quaternion).as_matrix()
        self.matModel[0:3,   3] = position
        self.matModel[  3, 0:4] = [0, 0, 0, 1]

        self._sorted = False # Changing the mesh orientation invalidates its depth sort

    def hasPerCornerVtxData(self):
        return (self.F is None) or (len(self.F.ravel()) == self.numVertices)

    def setWireframe(self, lineWidth = 1.0, color = [0.0, 0.0, 0.0, 1.0]):
        self.lineWidth = lineWidth
        self.lineColor = color

    def replicatePerCorner(self):
        self.ctx.makeCurrent()
        # Note: this operation preserves depth sorting!
        # (And it is more efficient to perform depth sorting first...)
        if self.hasPerCornerVtxData(): return

        # Switch to replication mode
        self._activeReplicationIndices = self.F.ravel()
        self.updateMeshData(self.V, self.N, self.color)

        # Disable the index buffers
        self.F = None
        self.vao.unsetIndexBuffer()

    def render(self, shader, matView):
        self.ctx.makeCurrent()

        # Sort triangles if needed
        self._depthSort(matView)

        # Our wireframe rendering technique needs distinct copies of vertices
        # for each incident triangle.
        if self.lineWidth != 0: self.replicatePerCorner()

        modelViewMatrix = matView @ self.matModel
        shader.setUniform('modelViewMatrix',   modelViewMatrix)
        shader.setUniform('normalMatrix',      np.linalg.inv(modelViewMatrix[0:3, 0:3]).T)

        shader.setUniform('shininess',         self.shininess)
        shader.setUniform('alpha',             self.alpha)

        shader.setUniform('lineWidth',         self.lineWidth)
        shader.setUniform('wireframeColor',    self.lineColor)

        # Any constant color configured is not part of the VAO state and must be set again to ensure it hasn't been overwritten
        if self.constColor: self.vao.setConstantAttribute(2, self.color)
        self.vao.draw(shader)

class MeshRenderer:
    def __init__(self, width, height):
        self.ctx = OpenGLContext(width, height)
        self.shader = self.ctx.shaderLibrary().load(SHADER_DIR + '/phong_with_wireframe.vert',
                                                    SHADER_DIR + '/phong_with_wireframe.frag')
        self.meshes = []

        self.matView       = np.identity(4)
        self.matProjection = np.identity(4)

        white = np.ones(3)
        self.lightEyePos       = [0, 0, 5]
        self. diffuseIntensity = 0.6 * white
        self. ambientIntensity = 0.5 * white
        self.specularIntensity = 1.0 * white

        self.transparentBackground = True

    def resize(self, width, height):
        self.ctx.resize(width, height)

    def setMesh(self, V, F, N, color, which = 0):
        """
        Initialize/update the mesh data and connectivity.
        `F` can be `None` to disable indexed face set representation
        (i.e., to use glDrawArrays instead of glDrawElements)
        """
        if len(self.meshes) == 0: self.meshes = [Mesh(self.ctx, V, F, N, color)]
        else: self.meshes[which].setMesh(V, F, N, color)

    def addMesh(self, V, F, N, color, makeDefault = True):
        """
        Add a mesh to the scene. Arguments are the same as `setMesh`.
        By default, the new mesh becomes the active default one (index 0).
        """
        self.meshes.insert(0 if makeDefault else len(self.meshes), Mesh(self.ctx, V, F, N, color))

    def removeMesh(self, which):
        self.self.meshes.remove(self.meshes[which])

    def modelMatrix(self, position, scale, quaternion, which = 0):
        self.meshes[which].modelMatrix(position, scale, quaternion)

    def updateMeshData(self, V, N, color = None, which = 0):
        """
        Update the mesh's data without changing its connectivity.
        """
        self.meshes[which].updateMeshData(V, N, color)

    def setViewMatrix(self, mat):
        self.matView = mat
        self._sorted = False # Changing the viewpoint invalidates the depth sort

    def lookAt(self, position, target, up):
        self.cam_position = position
        self.cam_target   = target
        self.cam_up       = up
        self.setViewMatrix(lookAtMatrix(position, target, up))

    def orbitedLookAt(self, position, target, up, axis, angle):
        """
        View looking at the target from a camera position rotated by `angle` around the up vector.
        """
        v = np.array(position) - np.array(target)
        R = scipy.spatial.transform.Rotation.from_rotvec(angle * np.array(axis) / np.linalg.norm(axis))
        self.lookAt(target + R.apply(v), target, R.apply(up))

    def perspective(self, fovy, aspect, near, far):
        f = 1.0 / np.tan(0.5 * np.deg2rad(fovy))
        self.matProjection = np.zeros((4, 4))
        self.matProjection[0, 0] = f / aspect
        self.matProjection[1, 1] = f
        self.matProjection[2, 2] = (near + far) / (near - far)
        self.matProjection[2, 3] = 2.0 * (near * far) / (near - far)
        self.matProjection[3, 2] = -1.0

    def getCameraParams(self):
        """
        Emulate with MeshFEM's TriMeshViewer.getCameraParams. As an unfortunate
        consequence, the `up` and `target` vectors are swapped wrt the
        arguments of  `lookAt`
        """
        return (self.cam_position, self.cam_up, self.cam_target)

    def setCameraParams(self, params):
        """
        Emulate with MeshFEM's TriMeshViewer.setCameraParams. As an unfortunate
        consequence, the `up` and `target` vectors are swapped wrt the
        arguments of  `lookAt`
        """
        self.lookAt(params[0], params[2], params[1])

    def setWireframe(self, lineWidth = 1.0, color = [0.0, 0.0, 0.0, 1.0], which = 0):
        self.meshes[which].setWireframe(lineWidth, color)

    def render(self, clear=True, clearColor = None):
        self.ctx.makeCurrent()

        if clear:
            if clearColor is None: clearColor = np.zeros(4) if self.transparentBackground else np.ones(4)
            self.ctx.clear(clearColor)

        self.ctx. enable(GLenum.GL_DEPTH_TEST)
        self.ctx.disable(GLenum.GL_CULL_FACE)

        # Set blending mode to obtain a correct (but alpha-premultiplied)
        # RGBA image when rendering atop a cleared ([0, 0, 0, 0]) buffer.
        # In particular, we need to be careful with how the alpha components
        # are blended.
        self.ctx.enable(GLenum.GL_BLEND)
        self.ctx.blendFunc(GLenum.GL_SRC_ALPHA, GLenum.GL_ONE_MINUS_SRC_ALPHA,
                           GLenum.GL_ONE,       GLenum.GL_ONE_MINUS_SRC_ALPHA)

        self.shader.use()
        # Set mesh-independent attributes
        self.shader.setUniform('projectionMatrix',  self.matProjection)

        self.shader.setUniform('lightEyePos',       self.lightEyePos)
        self.shader.setUniform('diffuseIntensity',  self.diffuseIntensity)
        self.shader.setUniform('ambientIntensity',  self.ambientIntensity)
        self.shader.setUniform('specularIntensity', self.specularIntensity)

        # Render the opaque meshes first
        # This will result in a perfely rendered scene with N opaque objects and 1 transparent object.
        # Proper ordering of triangles of multiple transparent objects is not implemented.
        transparencySortedMeshes = sorted(self.meshes, key=lambda m: not m.isOpaque())

        for mesh in transparencySortedMeshes:
            mesh.render(self.shader, self.matView)

    def array(self      ): return self.ctx.array(     unpremultiply=self.transparentBackground)
    def image(self      ): return self.ctx.image(     unpremultiply=self.transparentBackground)
    def  save(self, path): return self.ctx.save(path, unpremultiply=self.transparentBackground)

    def renderAnimation(self, outPath, nframes, frameCallback, *videoWriterArgs, **videoWriterKWargs):
        """
        Write an animation out as a video/image sequence, where each frame is set up by `frameCallback`
        """
        vw = video_writer.MeshRendererVideoWriter(outPath, self, *videoWriterArgs, **videoWriterKWargs)
        for frame in range(nframes):
            frameCallback(self, frame)
            vw.writeFrame()

    def orbitAnimation(self, outPath, nframes, axis=None, *videoWriterArgs, **videoWriterKWargs):
        """
        Render an animation of the camera making a full orbit around the up axis centered at its target.
        """
        pos, tgt, up = np.array(self.cam_position), np.array(self.cam_target), np.array(self.cam_up)
        if axis is None: axis = up.copy()
        def c(r, i): r.orbitedLookAt(pos, tgt, up, axis, 2 * np.pi / nframes * i)
        self.renderAnimation(outPath, nframes, c, *videoWriterArgs, **videoWriterKWargs)
