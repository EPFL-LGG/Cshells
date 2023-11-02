import numpy as np
from numpy.linalg import norm
import sys

import elastic_rods
from MeshFEM import sparse_matrices
from linkage_vis import LinkageViewer
from elastic_rods import EnergyType

import scipy.optimize
from scipy.sparse import csc_matrix

l = elastic_rods.RodLinkage('../examples/sphere_open.obj', 50)
mat = elastic_rods.RodMaterial('rectangle', 20000, 0.3, [0.008, 0.0008])
l.setMaterial(mat)
l.joint(135).setConstrained(True)

matSoft = mat
matSoft.stretchingStiffness = 1e-4;
matSoft.twistingStiffness=1e-5;
l.setMaterial(matSoft)
jposVars = l.jointPositionDoFIndices()

def iterateCallback(xk):
    l.setDoFs(xk)
    print(l.energy())
    l.updateSourceFrame()

def fun(x):
    l.setDoFs(x)
    print(l.energy())
    return l.energy()
    
def grad(x):
    l.setDoFs(x)
    return np.array(l.gradient(False))

def hessian(x):
    print('hessian eval')
    l.setDoFs(x)
    h = l.hessian()
    h.reflectUpperTriangle()
    return csc_matrix(h.compressedColumn())

def hessian_prod(x, v):
    l.setDoFs(x)
    return l.hessian().apply(v)

scipy.optimize.minimize(fun, l.getDoFs(), method='trust-ncg', jac=grad, hessp=hessian_prod, callback=iterateCallback, options={'disp': True})
