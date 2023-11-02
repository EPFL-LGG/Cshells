import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer

linkage_bindings = ct.cdll.LoadLibrary('libgrasshopper_bindings.dylib')

rod_linkage_grasshopper_interface = linkage_bindings.rod_linkage_grasshopper_interface
rod_linkage_grasshopper_interface.restype = None
rod_linkage_grasshopper_interface.argtypes = [ct.c_size_t,
                                              ct.c_size_t,
                                              ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                              ndpointer(ct.c_size_t, flags="C_CONTIGUOUS"),
                                              ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                              ndpointer(ct.c_double, flags="C_CONTIGUOUS")]

pts = np.array([
    -0.039153, -0.157135, 0.000000,
    -0.039153,  0.157135, 0.000000,
     0.000130,  0.000000, 0.000000,
     0.039414, -0.157135, 0.000000,
     0.039414,  0.157135, 0.000000
], dtype=np.double)

edges = np.array([
    0, 2,
    2, 1,
    3, 2,
    2, 4
], dtype=np.uint64)

outPtsClosed   = pts.copy()
outPtsDeployed = pts.copy()

linkage_bindings.rod_linkage_grasshopper_interface(int(len(pts) / 3), int(len(edges) / 2), pts, edges, outPtsClosed, outPtsDeployed)
