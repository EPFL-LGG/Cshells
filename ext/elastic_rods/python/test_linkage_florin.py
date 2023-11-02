import elastic_rods, sparse_matrices, pickle, scipy, linkage_vis, numpy as np, time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from bending_validation import suppress_stdout

elastic_rods.set_max_num_tbb_threads(4)

#linkage = elastic_rods.RodLinkage('../examples/florin/20181008_103824_meshID_95a3e4ba-e509-432a-9442-738b53a06248.obj', 5)
#linkage = elastic_rods.RodLinkage('../examples/florin/jgrids/20181022_112222_meshID_31244361-74cc-4d73-b5b2-c429d83fb6db.obj', 5)
#linkage = elastic_rods.RodLinkage('../examples/florin/jgrids/20181022_112227_meshID_e44b3753-0555-4f9e-9f65-0aae883bea63.obj', 5)
#linkage = elastic_rods.RodLinkage('../examples/florin/jgrids/20181022_112230_meshID_73265cae-cb87-4a2e-a36b-38036f90b033.obj', 10)
driver = 113
linkage = elastic_rods.RodLinkage('../examples/florin/jgrids/20181022_112232_meshID_664b2f13-096d-4cb7-8562-75150733a3cc.obj', 10) # flat equilibrium fail
driver = 29

#mat = elastic_rods.RodMaterial('+', 2000, 0.3, [2.0, 2.0, 0.2, 0.2])
#linkage.setMaterial(mat)

mat = elastic_rods.RodMaterial('+', 2000, 0.3, [1.0, 5.0, 1.0, 0.5], stiffAxis=elastic_rods.StiffAxis.D2)
linkage.setMaterial(mat)



elastic_rods.benchmark_reset()
elastic_rods.restlen_solve       (linkage, niter=1000, verbose=True, useIdentityMetric = True)
#elastic_rods.restlen_solve_knitro(linkage, niter=40, laplacianRegWeight = 1e-8)
elastic_rods.benchmark_report()

linkage.joint(driver).setConstrained(True)

# It seems, with random perturbations of 1e-3, knitro randomly switches between succeeding and taking 3-4 seconds, sometimes down to 1.6s
# or failing and taking 0.6-0.7s

optimizer = elastic_rods.get_equilibrium_optimizer(linkage)
equilibriumSolver      = lambda l, nit, verbose: optimizer.optimize(nit, verbose, useIdentityMetric=False, beta = 1e-6, useNegativeCurvatureDirection = True, gradTol=1e-4)
finalEquilibriumSolver = lambda l, nit, verbose: optimizer.optimize(nit, verbose, useIdentityMetric=False, beta = 1e-6, useNegativeCurvatureDirection = True)

equilibriumSolver(linkage, 40, True)
linkage.saveVisualizationGeometry('closed.msh')

import open_linkage

elastic_rods.benchmark_reset()

open_linkage.open_linkage(linkage, driver, np.pi/3, 30, None, zPerturbationEpsilon=0, equilibriumSolver=equilibriumSolver)
#open_linkage.open_linkage(linkage, driver, np.pi / 3, 40, None, zPerturbationEpsilon=0, equilibriumSolver=elastic_rods.compute_equilibrium_knitro)

newton_its = equilibriumSolver(linkage, 50, True)
elastic_rods.benchmark_report()

print(norm(linkage.gradient()))

#pickle.dump(linkage, open('rlo.pkl', 'wb'))
#linkage.saveVisualizationGeometry('rlo.msh')
linkage.saveVisualizationGeometry('opened.msh')

pickle.dump(linkage, open('opened.pkl', 'wb'))

#def fd_hessian_test(linkage, stepSize, etype=EnergyType.Full, direction=None, variableRestLen=False):
