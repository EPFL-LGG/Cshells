from bending_validation import suppress_stdout as so
from elastic_rods import ElasticRod
from elastic_rods import compute_equilibrium
import json
from OptimalSLEquilibrium import ComputeEnergyGradients, ComputeDeployedLinkageStability
import py_newton_optimizer
import torch

newtonOptimizerOptions = py_newton_optimizer.NewtonOptimizerOptions()
newtonOptimizerOptions.gradTol = 1.0e-7
newtonOptimizerOptions.verbose = 1
newtonOptimizerOptions.beta = 1.0e-8
newtonOptimizerOptions.niter = 50
newtonOptimizerOptions.verboseNonPosDef = False

def SaveStraightLinkages(filepath, jointPosFlat, jointPosDep, curves, curvesFamily, rodEdges, jointsQuads, jointsTris, rodMaterial):
    '''
    Args:
        filepath: the path where to save the json
        jointPosFlat: the joints positions in the flat state, np array of shape (nJ, 3)
        jointPosDep: the joints positions in the deployed state, np array of shape (nJ, 3)
        curves: list of lists that contain the joints indices for each curve
        curvesFamily: the curve family for each curve
        rodEdges: torch tensor of shape (nE, 2) containing the joint indices for each edge
        jointsQuads: list of list of 4 elements containing the quads ordered in a clockwise fashion
        jointsTris: list of lists of 3 indices containing triangles of a triangulation
        rodMaterial: the rod material used
    '''
    
    if not rodMaterial.hasCrossSectionMesh():
        rodMaterial.meshCrossSection(0.001)
    
    rodsList = []
    curvesFrameX = []
    curvesFrameY = []
    curvesFrameZ = []
    energyBend = []
    energyStretch = []
    energyTwist = []
    vmStress = []
    for i, crv in enumerate(curves):
        jointsFlatCrv = jointPosFlat[crv, :]
        jointsDepCrv = jointPosDep[crv, :]

        # Create a rod and deform
        rodTmp = ElasticRod(list(jointsFlatCrv))
        rodTmp.setMaterial(rodMaterial)
        dofs = rodTmp.getDoFs()
        dofs[:3 * jointsDepCrv.shape[0]] = jointsDepCrv.reshape(-1,)
        rodTmp.setDoFs(dofs)
        with so(): compute_equilibrium(rodTmp, options=newtonOptimizerOptions, fixedVars=list(range(3 * rodTmp.numVertices())))
        rodsList.append(rodTmp)
        
        frameX = [list(rodTmp.deformedPoints()[i+1] - rodTmp.deformedPoints()[i]) for i in range(rodTmp.numEdges())]
        frameY = [list(d1) for d1 in rodTmp.deformedMaterialFramesD1D2()[:rodTmp.numEdges()]]
        frameZ = [list(d2) for d2 in rodTmp.deformedMaterialFramesD1D2()[rodTmp.numEdges():]]
        
        curvesFrameX.append(frameX)
        curvesFrameY.append(frameY)
        curvesFrameZ.append(frameZ)
        
        energyBend.append(rodTmp.energyBendPerVertex().tolist())
        energyStretch.append(rodTmp.energyStretchPerEdge().tolist())
        energyTwist.append(rodTmp.energyTwistPerVertex().tolist())
        vmStress.append(rodTmp.maxVonMisesStresses().tolist())
        
    gradEnergy = ComputeEnergyGradients(jointPosFlat, jointPosDep, curves, rodMaterial=rodMaterial, cachedThetas=None)[0]
    _, resSL, torqueSL = ComputeDeployedLinkageStability(torch.tensor(jointPosDep), gradEnergy, jointsQuads)

    jsonLinkage = {
        "jointsFlat": jointPosFlat.reshape(-1, 3).tolist(),
        "jointsDeployed": jointPosDep.reshape(-1, 3).tolist(),
        "curves": curves,
        "curvesFamily": curvesFamily,
        "rodEdges": rodEdges.tolist(),
        "jointsQuads": jointsQuads,
        "jointsTris": jointsTris,
        "crossSection": [rodMaterial.crossSection().params()],
        "youngModulus": rodMaterial.youngModulus,
        "poissonRatio": rodMaterial.crossSection().nu,
        "frameX": curvesFrameX,
        "frameY": curvesFrameY,
        "frameZ": curvesFrameZ,
        "energyBendPerVertex": energyBend,
        "energyStretchPerEdge": energyStretch,
        "energyTwistPerVertex": energyTwist,
        "maxVonMisesStresses": vmStress,
        'jointForceResiduals': resSL.tolist(),
        'torque': torqueSL.item(),
    }
    
    with open(filepath, "w") as f:
        json.dump(jsonLinkage, f) 