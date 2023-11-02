import elastic_rods, pickle, scipy, linkage_vis, numpy as np, time
from MeshFEM import sparse_matrices
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from bending_validation import suppress_stdout

elastic_rods.set_max_num_tbb_threads(4)

linkage = elastic_rods.RodLinkage('../examples/nonuniform_linkage.obj', 10)
mat = elastic_rods.RodMaterial('+', 2000, 0.3, [0.02, 0.02, 0.002, 0.002])
# mat = elastic_rods.RodMaterial('ellipse', 20000, 0.3, [0.02, 0.002])
linkage.setMaterial(mat)

elastic_rods.benchmark_reset()
elastic_rods.restlen_solve       (linkage, niter=1000, verbose=True, useIdentityMetric = True)
#elastic_rods.restlen_solve_knitro(linkage, niter=40, laplacianRegWeight = 1e-8)
elastic_rods.benchmark_report()

linkage.joint(64).setConstrained(True)

linkage.saveVisualizationGeometry('closed.msh')

import open_linkage

elastic_rods.benchmark_reset()

# It seems, with random perturbations of 1e-3, knitro randomly switches between succeeding and taking 3-4 seconds, sometimes down to 1.6s
# or failing and taking 0.6-0.7s

equilibriumSolver = lambda l, nit, verbose: elastic_rods.compute_equilibrium(l, nit, verbose, useIdentityMetric=True, beta = 1e-8, useNegativeCurvatureDirection = True)
#finalEquilibriumSolver = lambda l, nit, verbose: elastic_rods.compute_equilibrium(l, nit, verbose, useIdentityMetric=False, beta = 1e-9, useNegativeCurvatureDirection = True)

equilibriumSolver(linkage, 40, True)
open_linkage.open_linkage(linkage, 64, np.pi/4, 10, None, zPerturbationEpsilon=0, equilibriumSolver=equilibriumSolver)
#open_linkage.open_linkage(linkage, 64, np.pi/4, 10, None, zPerturbationEpsilon=1e-3, equilibriumSolver=elastic_rods.compute_equilibrium_knitro)
elastic_rods.benchmark_report()

print(norm(linkage.gradient()))

#pickle.dump(linkage, open('rlo.pkl', 'wb'))
#linkage.saveVisualizationGeometry('rlo.msh')
linkage.saveVisualizationGeometry('opened.msh')
