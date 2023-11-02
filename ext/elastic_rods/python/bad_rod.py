import elastic_rods, pickle

r = pickle.load(open('bad_rod.pkl', 'rb'))
r.setMaterial(elastic_rods.RodMaterial('ellipse', 200, 0.3, [0.01, 0.005]))

fixedVars = [0, 1, 2, 77, 150, 151, 152]

elastic_rods.benchmark_reset()
elastic_rods.compute_equilibrium(r, verbose=True, fixedVars=fixedVars, niter=100, useIdentityMetric = False, beta = 1e-4, useNegativeCurvatureDirection = True, gradTol=2e-8)
#elastic_rods.compute_equilibrium_knitro(r, verbose=True, fixedVars=fixedVars, niter=100, gradTol=1e-8)
elastic_rods.benchmark_report()
