import numpy as np
import torch
from torch.autograd.functional import jacobian

import elastic_rods
import time
from CShell import GetEdgesFromCurves
from OptimalSLEquilibrium import ComputeDeployedLinkageStabilityFull, ComputeGradientDeployedLinkageStabilityFull, ComputeJointAngles, ComputeEnergyGradients, ComputeDeployedLinkageStability

try:
    KNITRO_FOUND = True
    from knitro.numpy import *
except Exception as e:
    KNITRO_FOUND = False
    print("Knitro may not have been found: {}.".format(e))

torch_dtype = torch.float64
torch.set_default_dtype(torch_dtype)

########################################################################
##  GENERAL UTILITY FUNCTIONS
########################################################################

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

def ComputeJointsToCurves(nJ, curvesFamily, curves):
    '''
    Args:
        nJ           : number of joints in the linkage
        curvesFamily : the curve family for each curve
        curves       : list of lists that contain the joints indices for each curve

    Returns:
        jointsToCurves : list of nJ dictionnaries giving the curve index for each family
    '''
    jointsToCurves = [{} for _ in range(nJ)]

    for idxCurve, (fam, crv) in enumerate(zip(curvesFamily, curves)):
        strFam = "A" if fam==0 else "B"
        for idxJoint in crv:
            jointsToCurves[idxJoint][strFam] = idxCurve
    return jointsToCurves

def ExtractCurvesPerFamily(curvesFamily, curves):
    '''
    Args:
        curvesFamily : the curve family for each curve
        curves       : list of lists that contain the joints indices for each curve

    Returns:
        curvesA : same a curves, simply contains family A
        curvesB : same a curves, simply contains family B
    '''
    curvesA = [curves[i] for i in range(len(curves)) if curvesFamily[i]==0 ]
    curvesB = [curves[i]  for i in range(len(curves)) if curvesFamily[i]==1]
    return curvesA, curvesB

def ComputeEdgesPerJointPerFamily(nJ, curvesA, curvesB):
    '''
    Args:
        nJ      : number of joints in the linkage
        curvesA : list of lists that contain the joints indices for each curve of family A
        curvesB : list of lists that contain the joints indices for each curve of family B

    Returns:
        idxEdgesA : np array containing the edges for each joint along curves of family A
        idxEdgesB : np array containing the edges for each joint along curves of family B
    '''
    
    idxEdgesA = np.zeros(shape=(nJ, 2)).astype(np.int32)
    idxEdgesB = np.zeros(shape=(nJ, 2)).astype(np.int32)

    for crvA in curvesA:
        for i in range(len(crvA)):
            if i == len(crvA)-1:
                idxEdgesA[crvA[i], 0] = crvA[i-1]
                idxEdgesA[crvA[i], 1] = crvA[i]
            else:
                idxEdgesA[crvA[i], 0] = crvA[i]
                idxEdgesA[crvA[i], 1] = crvA[i+1]
                
    for crvB in curvesB:
        for i in range(len(crvB)):
            if i == len(crvB)-1:
                idxEdgesB[crvB[i], 0] = crvB[i-1]
                idxEdgesB[crvB[i], 1] = crvB[i]
            else:
                idxEdgesB[crvB[i], 0] = crvB[i]
                idxEdgesB[crvB[i], 1] = crvB[i+1]

    return idxEdgesA, idxEdgesB

def ComputeFlatJointSignedAngles(joints, idxEdgesA, idxEdgesB):
    ''' We assume the z component of the joints to equal 0!
    Args:
        joints    : torch tensor containing the joints positions (nJ, 3)
        idxEdgesA : np array containing the edges for each joint along curves of family A
        idxEdgesB : np array containing the edges for each joint along curves of family B

    Returns:
        angles : torch tensor containing the opening angle at each joint (nJ,)
    '''
    edgesA = joints[idxEdgesA[:, 1], :] - joints[idxEdgesA[:, 0], :]
    edgesB = joints[idxEdgesB[:, 1], :] - joints[idxEdgesB[:, 0], :]

    sinJoints = torch.cross(edgesA, edgesB)[:, 2]
    cosJoints = torch.einsum('ij, ij -> i', edgesA, edgesB)
    
    angles = torch.atan2(sinJoints, cosJoints)
    
    return angles

def ComputeJointAnglesFromQuads(joints, quads, flat=False):
    '''
    Args:
        joints    : torch tensor containing the joints positions (nJ, 3)
        quads : list of list of 4 elements containing the quads ordered in a clockwise fashion
        flat  : whether the joints are assumed to have 0 

    Returns:
        angles : torch tensor containing the opening angle at each joint (nQuads, 4)
    
    Notes:
        This is how vertices are labelled within a quad
                    3 +----+ 2
                      |    |
                    0 +----+ 1
    '''
    
    quadsTorch = torch.tensor(quads)
    angles = torch.zeros(size=(quadsTorch.shape[0], 4))
    
    anglesFun = ComputeFlatJointSignedAngles if flat else ComputeJointAngles
    
    # Not the most efficient, some edges can be reused in ComputeJointAngles
    angles[:, 0] = anglesFun(
        joints, 
        torch.stack([quadsTorch[:, 0], quadsTorch[:, 1]], axis=1), 
        torch.stack([quadsTorch[:, 0], quadsTorch[:, 3]], axis=1),
    )
    
    angles[:, 1] = anglesFun(
        joints, 
        torch.stack([quadsTorch[:, 1], quadsTorch[:, 2]], axis=1), 
        torch.stack([quadsTorch[:, 1], quadsTorch[:, 0]], axis=1),
    )
    
    angles[:, 2] = anglesFun(
        joints, 
        torch.stack([quadsTorch[:, 2], quadsTorch[:, 3]], axis=1), 
        torch.stack([quadsTorch[:, 2], quadsTorch[:, 1]], axis=1),
    )
    
    angles[:, 3] = anglesFun(
        joints, 
        torch.stack([quadsTorch[:, 3], quadsTorch[:, 0]], axis=1), 
        torch.stack([quadsTorch[:, 3], quadsTorch[:, 2]], axis=1),
    )
    
    return angles

def ComputeQuads(nJ, jointsToCurves, curves):
    '''
    Args:
        nJ : number of joints in the linkage
        jointsToCurves : list of nJ dictionnaries giving the curve index for each family
        curves : list of lists that contain the joints indices for each curve
    
    Returns:
        jointsQuads : list of list of 4 elements containing the quads ordered in a clockwise fashion
    '''

    jointsQuads = []
    seq   = [("A", 1), ("B", 1), ("A", -1), ("B", -1)]

    for j in range(nJ):
        currJoint  = j
        jointsQuad = []
        addQuad    = True
        for (fam, inc) in seq:
            # Test if we can find a curve of the correct family passing by the current joint
            if fam in jointsToCurves[currJoint]:
                crvId = jointsToCurves[currJoint][fam]
                crv   = curves[crvId]
                posJointInCrv = crv.index(currJoint)

                isFirst = posJointInCrv == 0
                isLast  = posJointInCrv == len(crv) - 1
                # Test if we're at the end of the curve and need to go one step further (same for beginning of the curve)
                if (isFirst and inc==-1) or (isLast and inc==1):
                    addQuad = False
                    break
                else:
                    currJoint = crv[posJointInCrv+inc]
                    jointsQuad.append(currJoint)
            else:
                addQuad = False

        if addQuad:
            jointsQuads.append(jointsQuad)
    
    return jointsQuads

def ComputeQuadsTriangulation(quads, transpose=False):
    '''
    Args:
        quads : list of list of 4 elements containing the quads ordered in a clockwise fashion
    
    Returns:
        triangulation: list of list of 3 elements 
    
    Note:
        Sketch of the 2 triangulations
    Triangulation 1             Triangulation 2 (transpose)
     4 ----- 3                   4 ----- 3
       |\ 2|                       |4 /|
       | \ |                       | / |
       |1 \|                       |/ 3|
     1 ----- 2                   1 ----- 2
    '''
    
    tris1 = []
    tris2 = []
    
    if not transpose:
        for quad in quads:
            tris1.append([quad[3], quad[0], quad[1]])
            tris2.append([quad[1], quad[2], quad[3]])
    else:
        for quad in quads:
            tris1.append([quad[0], quad[1], quad[2]])
            tris2.append([quad[2], quad[3], quad[0]])
    
    return tris1 + tris2
        

def QuadsTriangulationsSignedAreas(quads):
    '''
    Args:
        quads : torch tensor of shape (nQuads, 4, 2) containing the joints positions for each quad

    Returns:
        signedAreas : torch tensor of shape (nQuads, 2, 2) containing the signed areas for each triangle for each triangulation

    Note:
        Sketch of the areas calculated
    Triangulation 1             Triangulation 2
     4 ----- 3                   4 ----- 3
       |\ 2|                       |4 /|
       | \ |                       | / |
       |1 \|                       |/ 3|
     1 ----- 2                   1 ----- 2
    '''

    edge1Tri1 = quads[:, 1, :] - quads[:, 0, :]
    edge2Tri1 = quads[:, 3, :] - quads[:, 0, :]
    edge3Tri1 = quads[:, 1, :] - quads[:, 2, :]
    edge4Tri1 = quads[:, 3, :] - quads[:, 2, :]

    # edge1Tri1 x edge2Tri1 / 2
    area11 = (edge1Tri1[..., 0] * edge2Tri1[..., 1] - edge1Tri1[..., 1] * edge2Tri1[..., 0]) / 2
    # edge4Tri1 x edge3Tri1 / 2
    area12 = (edge4Tri1[..., 0] * edge3Tri1[..., 1] - edge4Tri1[..., 1] * edge3Tri1[..., 0]) / 2

    edge1Tri2 = quads[:, 0, :] - quads[:, 1, :]
    edge2Tri2 = quads[:, 2, :] - quads[:, 1, :]
    edge3Tri2 = quads[:, 0, :] - quads[:, 3, :]
    edge4Tri2 = quads[:, 2, :] - quads[:, 3, :]

    # edge2Tri2 x edge1Tri2 / 2
    area21 = (edge2Tri2[..., 0] * edge1Tri2[..., 1] - edge2Tri2[..., 1] * edge1Tri2[..., 0]) / 2
    # edge3Tri2 x edge4Tri2 / 2
    area22 = (edge3Tri2[..., 0] * edge4Tri2[..., 1] - edge3Tri2[..., 1] * edge4Tri2[..., 0]) / 2

    signedAreas = torch.zeros(size=(quads.shape[0], 2, 2))
    signedAreas[..., 0, 0] = area11
    signedAreas[..., 0, 1] = area12
    signedAreas[..., 1, 0] = area21
    signedAreas[..., 1, 1] = area22

    return signedAreas

########################################################################
##  FUNCTIONS RELATED TO PLANARIZATION
########################################################################

def ObjectivePlanarization(xFlatTmp, jointsDepTmp, rodEdges, bndIndices, 
                         idxEdgesA, idxEdgesB, jointsTris,
                         jointsFlatInit, jointsDepInit, 
                         weightLengths=1.0, weightAngles=0.0, weightMeanAngle=0.0, 
                         weightBnd=0.0, weightFlatReg=1.0e-3,
                         bndOnly=True):
    '''
    Args:
        xFlatTmp            : the torch tensor of shape (nVarsFlat,) that contains the variables related to joints in their flat state to be optimized
        jointsDepTmp        : the torch tensor of shape (nJ, 3) that contains the joints positions in their deployed state
        rodEdges            : torch tensor of shape (nE, 2) containing the joint indices for each edge
        bndIndices          : torch tensor of shape (nB,) containing the boundary joint indices
        idxEdgesA           : np array containing the edges for each joint along curves of family A
        idxEdgesB           : np array containing the edges for each joint along curves of family B
        jointsTris          : list of lists of 3 indices containing triangles of a triangulation
        targetCurvesLengths : the curves lengths to match in the planar state, np array of shape (nCurves,)
        jointsFlatInit      : the initial joints positions in the flat state, np array of shape (nJ, 2)
        jointsDepInit       : the initial joints positions in the deployed state, np array of shape (nJ, 3)
        weightLengths       : the weight in front of the length preservation term
        weightAngles        : regularization weight on the angles uniformity term
        weightMeanAngle     : regularization weight for the mean angle increment value (should be away from zero)
        weightBnd           : regularization weight in front of a terms that keeps the boundary joints positions close to their original spots
        weightFlatReg       : a weight that enforces an arbitrary registration to the flat joints
        bndOnly             : whether we only softly pin teh boundary or all joints
    
    Returns:
        loss : the objective to be minimized
    '''
    
    jointsFlatTmp = xFlatTmp.reshape(-1, 2)
    
    distFlat = torch.linalg.norm(jointsFlatTmp[rodEdges[:, 0]] - jointsFlatTmp[rodEdges[:, 1]], dim=1)
    distDep  = torch.linalg.norm(jointsDepTmp[rodEdges[:, 0]] - jointsDepTmp[rodEdges[:, 1]], dim=1)
    lossDists = torch.mean((distFlat - distDep) ** 2) / 4
    
    jointsFlat3D        = torch.zeros(size=(jointsFlatTmp.shape[0], 3))
    jointsFlat3D[:, :2] = jointsFlatTmp
    anglesFlat  = ComputeJointAngles(jointsFlat3D, idxEdgesA, idxEdgesB)
    anglesDep   = ComputeJointAngles(jointsDepTmp, idxEdgesA, idxEdgesB)
    angleDeltas = anglesDep - anglesFlat
    lossAngles = torch.var(angleDeltas) / 2
    lossMeanAngles = torch.exp(-torch.abs(torch.mean(angleDeltas)) / 0.5)
    
    if bndOnly:
        lossBnd = torch.mean(torch.sum((jointsDepTmp[bndIndices] - torch.tensor(jointsDepInit[bndIndices])) ** 2, dim=1)) / 2
    else:
        lossBnd = torch.mean(torch.sum((jointsDepTmp - torch.tensor(jointsDepInit)) ** 2, dim=1)) / 2
    
    lossFlatReg = torch.mean(torch.sum((jointsFlatTmp - torch.tensor(jointsFlatInit)[:, :2]) ** 2, axis=1)) / 2

    loss = (
        weightLengths * lossDists 
        + weightAngles * lossAngles 
        + weightMeanAngle * lossMeanAngles
        + weightBnd * lossBnd 
        + weightFlatReg * lossFlatReg
    )
    
    return loss

def ConstraintsPlanarization(xFlatTmp, jointsDepTmp, curves, jointsFlatInit, jointsQuads, quadsSignsInit, areaConstraintLSE=False):
    '''
    Args:
        xFlatTmp           : the torch tensor of shape (nVarsFlat,) that contains the variables related to joints in their flat state to be optimized
        jointsDepTmp       : the torch tensor of shape (nJ, 2) that contains the joints positions in their deployed state
        curves             : list of lists that contain the joints indices for each curve
        jointsFlatInit     : the initial joints positions, np array of shape (nJ, 2)
        jointsQuads        : list of list of 4 elements containing the quads ordered in a clockwise fashion
        quadsSignsInit     : torch tensor of shape (nQuads, 2, 2) containing the sign of the signed areas for each triangle for each triangulation
        areaConstraintLSE  : whether to use LSE to aggregate area constraints
    
    Returns:
        constraints : the constraints of the problem optimization (nQuads,)
    '''
    
    constraints = torch.zeros(size=((1 + 4) * len(jointsQuads),))

    jointsFlatTmp = xFlatTmp.reshape(-1, 2)
    quadsTmp = QuadsTriangulationsSignedAreas(jointsFlatTmp[jointsQuads, :]) * quadsSignsInit
    
    areasMaxMin = torch.maximum(torch.minimum(quadsTmp[:, 0, 0], quadsTmp[:, 0, 1]), 
                                torch.minimum(quadsTmp[:, 1, 0], quadsTmp[:, 1, 1]))
    constraints[:len(jointsQuads)] = areasMaxMin
    
    jointsFlat3D        = torch.zeros(size=(jointsFlatTmp.shape[0], 3))
    jointsFlat3D[:, :2] = jointsFlatTmp
    anglesFlat  = ComputeJointAnglesFromQuads(jointsFlat3D, jointsQuads, flat=True).reshape(-1,)
    constraints[len(jointsQuads):(1+4) * len(jointsQuads)] = anglesFlat
    
    return constraints

def MetricsPlanarization(xFlatTmp, jointsDepTmp, rodEdges, jointsFlatInit, jointsDepInit, 
                       idxEdgesA, idxEdgesB, jointsTris, bndIndices,
                       curves, jointsQuads, rodMaterial,
                       ):
    '''
    Args:
        xFlatTmp            : the torch tensor of shape (nVarsFlat,) that contains the variables related to joints in their flat state to be optimized
        jointsDepTmp        : the torch tensor of shape (nJ, 3) that contains the joints positions in their deployed state
        rodEdges            : torch tensor of shape (nE, 2) containing the joint indices for each edge
        jointsFlatInit      : the initial joints positions in the flat state, np array of shape (nJ, 2)
        jointsDepInit       : the initial joints positions in the deployed state, np array of shape (nJ, 3)
        idxEdgesA           : np array containing the edges for each joint along curves of family A
        idxEdgesB           : np array containing the edges for each joint along curves of family B
        jointsTris          : list of lists of 3 indices containing triangles of a triangulation
        bndIndices          : torch tensor of shape (nB,) containing the boundary joint indices
        curves              : list of lists that contain the joints indices for each curve
        jointsQuads         : list of list of 4 elements containing the quads ordered in a clockwise fashion
        rodMaterial         : the rod material to use
    
    Returns:
        metrics : a dictionnary filled with relevant metrics
    '''
    jointsFlatTmp = xFlatTmp.reshape(-1, 2)
    
    distFlat = torch.linalg.norm(jointsFlatTmp[rodEdges[:, 0]] - jointsFlatTmp[rodEdges[:, 1]], dim=1)
    distDep  = torch.linalg.norm(jointsDepTmp[rodEdges[:, 0]]  - jointsDepTmp[rodEdges[:, 1]] , dim=1)
    
    jointsFlat3D        = torch.zeros(size=(jointsFlatTmp.shape[0], 3), dtype=torch_dtype)
    jointsFlat3D[:, :2] = jointsFlatTmp
    anglesFlat  = ComputeJointAngles(jointsFlat3D, idxEdgesA, idxEdgesB)
    anglesDep   = ComputeJointAngles(jointsDepTmp, idxEdgesA, idxEdgesB)
    angleDeltas = anglesDep - anglesFlat
    
    gradEnergy, rodsList = ComputeEnergyGradients(jointsFlat3D, jointsDepTmp, curves, rodMaterial=rodMaterial, cachedThetas=None)
    objEqSL, resSL, torqueSL = ComputeDeployedLinkageStability(jointsDepTmp, gradEnergy, jointsQuads)
    rodsEnergies = np.array([rod.energy() for rod in rodsList])

    metrics = {
        'Distance Discrepancy': ToNumpy(torch.abs(distFlat - distDep)),
        'Angle Deviation to Mean': ToNumpy(torch.abs(angleDeltas - torch.mean(angleDeltas))),
        'Angle Increments Mean': ToNumpy(torch.mean(angleDeltas)),
        'Boundary Fitting': ToNumpy(torch.linalg.norm(jointsDepTmp[bndIndices] - torch.tensor(jointsDepInit[bndIndices]), dim=1)),
        'Equilibrium Gap (SL)':  objEqSL.item() * np.ones(shape=(rodsEnergies.shape[0],)),
        'Rods Energies (SL)':  rodsEnergies,
        'Force Residuals (SL)': np.linalg.norm(resSL, axis=-1),
        'Torque (SL)': torqueSL * np.ones(shape=(rodsEnergies.shape[0],)),
    }

    return metrics
    

class PlanarizationOptimizer():
    '''
    Attributes:
        curves               : list of lists that contain the joints indices for each curve
        curvesFamily         : the curve family for each curve
        targetSurface        : a surface provided by geomdl
        jointsFlatInit       : the initial joints positions in their flat state, np array of shape (nJ, 2)
        jointsDepInit        : the initial joints positions in their deployed state, np array of shape (nJ, 2)
        configPath           : a string giving the absolute path to the .opt file used for the optimization
        pinnedVarsDepIdx     : the list of pinned degrees of freedom in the deployed state (uv-coordinates)
        rodMaterial          : the rod material to use
        weightLengths        : the weight in front of the lengths preservation term
        weightAngles         : regularization weight on the angles uniformity term
        weightMeanAngle      : regularization weight for the mean angle increment value (should be away from zero)
        weightEqSL           : regularization weight in front of the force balance term (model as an ensemble of elastic rods)
        weightBnd            : regularization weight in front of a terms that keeps the boundary joints positions close to their original spots
        weightFlatReg        : a weight that enforces an arbitrary registration to the flat joints
        factorAreaConstraint : the multiplicative factor that determines how far we want to be from the self-intersection frontier
        factorLenConstraint  : the multiplicative factor that tells how much we allow curves length in the deployed and flat state to differ
        areaConstraintLSE    : whether to use LSE to aggregate area constraints
        bndOnly              : whether we only softly pin the boundary or all joints
    '''

    def __init__(self, curves, curvesFamily, targetSurface, jointsFlatInit, jointsDepInit, configPath, 
                 pinnedVarsDepIdx=None, rodMaterial=None,
                 weightLengths=1.0, weightAngles=0.0, weightMeanAngle=0.0, weightEqSL=0.0,
                 weightBnd=0.0, weightFlatReg=1.0e-5, factorAreaConstraint=0.0, factorLenConstraint=0.0,
                 areaConstraintLSE=False, bndOnly=True):
        self.curves               = curves
        self.curvesFamily         = curvesFamily
        self.targetSurface        = targetSurface
        self.jointsFlatInit       = jointsFlatInit
        self.jointsDepInit        = jointsDepInit
        self.weightLengths        = weightLengths
        self.weightAngles         = weightAngles
        self.weightMeanAngle      = weightMeanAngle
        self.weightEqSL           = weightEqSL
        self.weightBnd            = weightBnd
        self.weightFlatReg        = weightFlatReg
        self.factorAreaConstraint = factorAreaConstraint
        self.factorLenConstraint  = factorLenConstraint
        self.configPath           = configPath
        self.areaConstraintLSE    = areaConstraintLSE
        self.bndOnly              = bndOnly
        
        self.objVals              = []
        self.metrics              = {}
        self.times                = []
        self.previousTime         = 0.0
        
        self.nJ             = jointsFlatInit.shape[0]
        self.jointsToCurves = ComputeJointsToCurves(self.nJ, self.curvesFamily, self.curves)
        self.jointsQuads    = ComputeQuads(self.nJ, self.jointsToCurves, self.curves)
        self.jointsTris     = ComputeQuadsTriangulation(self.jointsQuads, transpose=False)
        self.quadsInit      = QuadsTriangulationsSignedAreas(torch.tensor(self.jointsFlatInit)[self.jointsQuads, :])
        self.quadsSignsInit = torch.sign(self.quadsInit)
        
        rodEdges, rodEdgeToCurve = GetEdgesFromCurves(self.curves)
        self.rodEdges            = rodEdges
        self.rodEdgeToCurve      = rodEdgeToCurve
        self.bndIndices          = np.unique([crv[0] for crv in self.curves] + [crv[-1] for crv in self.curves])
        
        curvesA, curvesB = ExtractCurvesPerFamily(self.curvesFamily, self.curves)
        self.curvesA = curvesA
        self.curvesB = curvesB
        idxEdgesA, idxEdgesB = ComputeEdgesPerJointPerFamily(self.nJ, self.curvesA, self.curvesB)
        self.idxEdgesA = idxEdgesA
        self.idxEdgesB = idxEdgesB
        
        if pinnedVarsDepIdx is None: pinnedVarsDepIdx = []
        for varIdx in pinnedVarsDepIdx:
            assert varIdx <= 2 * self.nJ
        self.pinnedVarsDepIdx = np.unique(pinnedVarsDepIdx)
        self.pinnedVarsIdx = self.pinnedVarsDepIdx + 2 * self.nJ
        self.pinnedVarsDep = None # will be filled with the initial guess
        self.unpinnedVarsDepIdx = np.array([i for i in range(2 * self.nJ) if not i in self.pinnedVarsDepIdx])
        self.unpinnedVarsIdx = np.array([i for i in range(4 * self.nJ) if not (i - 2 * self.nJ) in self.pinnedVarsDepIdx])
        
        # self.nVarsFlat    = 2 * self.nJ - 3 # if we pin 3 dofs for rigid motion
        self.nVarsFlat    = 2 * self.nJ
        self.nVarsDep     = self.unpinnedVarsDepIdx.shape[0]
        self.nConstraints = len(self.jointsQuads) + 4 * len(self.jointsQuads)
        
        depLengthCurvesInit = np.array([sum([np.linalg.norm(self.jointsDepInit[crv[i+1]] - self.jointsDepInit[crv[i]]) for i in range(len(crv)-1)]) for crv in self.curves])
        self.scaleLengthA = max(depLengthCurvesInit[[fam == 0 for fam in self.curvesFamily]])
        self.scaleLengthB = max(depLengthCurvesInit[[fam == 1 for fam in self.curvesFamily]])
        # Gives the expected area of a triangle in a triangulation of the quads formed by the grid
        self.areaThresh   = self.factorAreaConstraint * (self.scaleLengthA * self.scaleLengthB) / (2 * self.quadsSignsInit.shape[0])
        self.lengthThresh = self.factorLenConstraint * min(self.scaleLengthA, self.scaleLengthB)
        
        self.cachedThetas     = None # for the SL equilibrium term
        
        # Rescale each weight using the normalization factors
        if rodMaterial is None: 
            self.rodMaterial = elastic_rods.RodMaterial('rectangle', 2000.0, 0.3, [0.40, 0.15], keepCrossSectionMesh=True)
        else:
            self.rodMaterial = rodMaterial
        self.weightLengths     = self.weightLengths / (min(self.scaleLengthA, self.scaleLengthB) ** 2)
        # self.weightLengths     = self.weightLengths / (min(self.scaleLengthA, self.scaleLengthB) ** 4)
        self.weightAngles      = self.weightAngles / 1.0
        self.weightMeanAngle   = self.weightMeanAngle / 1.0
        self.weightEqSL        = self.weightEqSL / (self.rodMaterial.youngModulus * self.rodMaterial.area) ** 2
        self.weightBnd         = self.weightBnd / (min(self.scaleLengthA, self.scaleLengthB) ** 2)
        self.weightFlatReg     = self.weightFlatReg / (min(self.scaleLengthA, self.scaleLengthB) ** 2)

    def callbackEvalF (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro. The different terms in the
        optimization are hardcoded for simplicity, since some terms might use different 
        representations of the cshell
        '''
        if evalRequest.type != KN_RC_EVALFC:
            print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
            return -1
        currVals = evalRequest.x
        
        currValsFlat   = torch.tensor(currVals)[:self.nVarsFlat]
        currValsDep    = np.zeros(shape=(2 * self.nJ,))
        if self.pinnedVarsDepIdx.shape[0] > 0: currValsDep[self.pinnedVarsDepIdx] = self.pinnedVarsDep
        currValsDep[self.unpinnedVarsDepIdx] = np.array(currVals)[self.nVarsFlat:len(currVals)]
        currValsDep = currValsDep.reshape(-1, 2)
        currJointsDep  = torch.tensor(self.targetSurface.evaluate_list(currValsDep.tolist().copy()))
        objSS = ObjectivePlanarization(currValsFlat, currJointsDep, self.rodEdges, self.bndIndices, 
                                     self.idxEdgesA, self.idxEdgesB,
                                     self.jointsTris,
                                     self.jointsFlatInit, self.jointsDepInit, 
                                     weightLengths=self.weightLengths, weightAngles=self.weightAngles, 
                                     weightMeanAngle=self.weightMeanAngle, 
                                     weightBnd=self.weightBnd, weightFlatReg=self.weightFlatReg,
                                     bndOnly=self.bndOnly).item()

        currJointsFlatDet = torch.zeros(size=(self.nJ, 3))
        currJointsFlatDet[:, :2] = currValsFlat.detach().reshape(-1, 2)
        currJointsDepDet = currJointsDep.detach()
        objEqSL = 0.0
        if self.weightEqSL > 1.0e-12:
            objEqSL = ComputeDeployedLinkageStabilityFull(
                currJointsFlatDet, currJointsDepDet, self.curves, self.jointsQuads, rodMaterial=self.rodMaterial, cachedThetas=self.cachedThetas,
            )[0]
        
        evalResult.obj = objSS + self.weightEqSL * objEqSL
        
        constraints  = ConstraintsPlanarization(currValsFlat, currJointsDep, self.curves, self.jointsFlatInit, self.jointsQuads, self.quadsSignsInit)
        evalResult.c = ToNumpy(constraints)

        return 0

    def callbackEvalG (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro
        '''
        if evalRequest.type != KN_RC_EVALGA:
            print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1
        currVals = evalRequest.x
        
        currValsFlat  = torch.tensor(currVals)[:self.nVarsFlat]
        currValsDep    = np.zeros(shape=(2 * self.nJ,))
        if self.pinnedVarsDepIdx.shape[0] > 0: currValsDep[self.pinnedVarsDepIdx] = self.pinnedVarsDep
        currValsDep[self.unpinnedVarsDepIdx] = np.array(currVals)[self.nVarsFlat:len(currVals)]
        currValsDep = currValsDep.reshape(-1, 2)
        
        currJointsDep = []
        dS_du         = []
        dS_dv         = []
        for uv in currValsDep.tolist().copy():
            derivTmp = self.targetSurface.derivatives(uv[0], uv[1], order=1)
            currJointsDep.append(derivTmp[0][0])
            dS_du.append(derivTmp[1][0])
            dS_dv.append(derivTmp[0][1])

        currJointsDep = torch.tensor(currJointsDep)
        dS_du         = np.array(dS_du)
        dS_dv         = np.array(dS_dv)
        
        currValsFlat.requires_grad  = True
        currJointsDep.requires_grad = True
        
        objTmp = ObjectivePlanarization(currValsFlat, currJointsDep, self.rodEdges, self.bndIndices, 
                                      self.idxEdgesA, self.idxEdgesB,
                                      self.jointsTris,
                                      self.jointsFlatInit, self.jointsDepInit, 
                                      weightLengths=self.weightLengths, weightAngles=self.weightAngles, 
                                      weightMeanAngle=self.weightMeanAngle, 
                                      weightBnd=self.weightBnd, weightFlatReg=self.weightFlatReg,
                                      bndOnly=self.bndOnly)
        objTmp.backward(torch.ones_like(objTmp))
        
        currJointsFlatDet = torch.zeros(size=(self.nJ, 3))
        currJointsFlatDet[:, :2] = currValsFlat.detach().reshape(-1, 2)
        currJointsDepDet = currJointsDep.detach()
        
        gradStabFlatSL, gradStabDepSL = np.zeros(shape=(self.nJ, 3)), np.zeros(shape=(self.nJ, 3))
        
        if self.weightEqSL > 1.0e-12:
            gradStabFlatSL, gradStabDepSL = ComputeGradientDeployedLinkageStabilityFull(
                currJointsFlatDet, currJointsDepDet, self.curves, self.jointsQuads, rodMaterial=self.rodMaterial, cachedThetas=self.cachedThetas,
            )

        if np.any(np.isnan(ToNumpy(currValsFlat.grad))):
            print('Nan in currValsGrad')
            print(currValsFlat.grad.reshape(-1, 2))
            assert 0
        if np.any(np.isnan(gradStabFlatSL[:, :2])):
            print('Nan in gradStabFlatSL')
            print(gradStabFlatSL[:, :2])
            assert 0
        if np.any(np.isnan(gradStabDepSL)):
            print('Nan in gradStabDepSL')
            print(gradStabDepSL)
            assert 0

        # Adds gradient of the stability criterion wrt flat joints positions
        gradObjValsFlat  = ToNumpy(currValsFlat.grad) + self.weightEqSL * gradStabFlatSL[:, :2].reshape(-1,)
        
        # Backprop through surface evaluation
        gradObjJointsDep     = ToNumpy(currJointsDep.grad) + self.weightEqSL * gradStabDepSL
        gradObjValsDep_u     = np.sum(gradObjJointsDep * dS_du, axis=1)
        gradObjValsDep_v     = np.sum(gradObjJointsDep * dS_dv, axis=1)
        gradObjValsDep       = np.zeros(shape=(2 * self.nJ,))
        gradObjValsDep[ ::2] = gradObjValsDep_u
        gradObjValsDep[1::2] = gradObjValsDep_v
        gradObjValsDep       = gradObjValsDep[self.unpinnedVarsDepIdx]

        evalResult.objGrad = np.concatenate([gradObjValsFlat, gradObjValsDep], axis=0)
        
        currValsFlat    = currValsFlat.detach().clone()
        currJointsDep   = currJointsDep.detach().clone()
        constraintsFun  = lambda x, y: ConstraintsPlanarization(x, y, self.curves, self.jointsFlatInit, self.jointsQuads, self.quadsSignsInit)
        constraintsJacs = jacobian(constraintsFun, (currValsFlat, currJointsDep))
        
        # Backprop through surface evaluation
        constraintsJacFlat = ToNumpy(constraintsJacs[0])
        constraintsJacJointsDep = ToNumpy(constraintsJacs[1])
        constraintsJacValsDep_u = np.sum(constraintsJacJointsDep * np.expand_dims(dS_du, axis=0), axis=2)
        constraintsJacValsDep_v = np.sum(constraintsJacJointsDep * np.expand_dims(dS_dv, axis=0), axis=2)
        constraintsJacDep          = np.zeros(shape=(self.nConstraints, 2 * self.nJ))
        constraintsJacDep[:,  ::2] = constraintsJacValsDep_u
        constraintsJacDep[:, 1::2] = constraintsJacValsDep_v
        constraintsJacDep          = constraintsJacDep[:, self.unpinnedVarsDepIdx]

        evalResult.jac = np.concatenate([constraintsJacFlat, constraintsJacDep], axis=1).reshape(-1,)

        return 0

    def newPtCallback(self, kc, x, lbda, userParams):
        
        self.times.append(time.time() - self.previousTime)
        
        currVals = x
        
        currValsFlat  = torch.tensor(currVals)[:self.nVarsFlat]
        currValsDep    = np.zeros(shape=(2 * self.nJ,))
        if self.pinnedVarsDepIdx.shape[0] > 0: currValsDep[self.pinnedVarsDepIdx] = self.pinnedVarsDep
        currValsDep[self.unpinnedVarsDepIdx] = np.array(currVals)[self.nVarsFlat:len(currVals)]
        currValsDep = currValsDep.reshape(-1, 2)
        currJointsDep = torch.tensor(self.targetSurface.evaluate_list(currValsDep.tolist().copy()))
        
        # Update the cached cshell
        currJointsFlat = torch.zeros(size=(self.nJ, 3))
        currJointsFlat[:, :2] = currValsFlat.reshape(-1, 2)

        self.objVals.append(ObjectivePlanarization(currValsFlat, currJointsDep, self.rodEdges, self.bndIndices, 
                                                 self.idxEdgesA, self.idxEdgesB,
                                                 self.jointsTris,
                                                 self.jointsFlatInit, self.jointsDepInit, 
                                                 weightLengths=self.weightLengths, weightAngles=self.weightAngles, 
                                                 weightMeanAngle=self.weightMeanAngle, 
                                                 weightBnd=self.weightBnd, weightFlatReg=self.weightFlatReg,
                                                 bndOnly=self.bndOnly).item())
        
        metricsTmp = MetricsPlanarization(currValsFlat, currJointsDep, self.rodEdges, self.jointsFlatInit, self.jointsDepInit, 
                                        self.idxEdgesA, self.idxEdgesB, self.jointsTris, self.bndIndices,
                                        self.curves, self.jointsQuads, self.rodMaterial)
        for key in self.metrics.keys():
            self.metrics[key] = np.concatenate([self.metrics[key], metricsTmp[key].reshape(1, -1)], axis=0)
        
        constraints  = ToNumpy(ConstraintsPlanarization(currValsFlat, currJointsDep, self.curves, self.jointsFlatInit, self.jointsQuads, self.quadsSignsInit))
        quadsFeas = np.all(constraints[:len(self.jointsQuads)] >= self.areaThresh)
        angleLowFeas = np.all(constraints[len(self.jointsQuads):] >= 0.0)
        angleHighFeas = np.all(constraints[len(self.jointsQuads):] <= np.pi)
        hasNaN = np.any(np.isnan(currVals))
        betterObj = self.objVals[-1] < self.objVals[-2]
        if quadsFeas and angleLowFeas and angleHighFeas and (not hasNaN) and betterObj:
            self.lastFeasible = currVals
            
        self.previousTime = time.time()

        return 0

    def Optimize(self, xInit, numSteps=None, trustRegionScale=1.0, 
                 optTol=1.0e-3, ftol=1.0e-6, xtol=1.0e-6, ftol_iters=2, xtol_iters=2, honorbounds=1,
                 verbosity='iter'):
        '''
        Args:
            xInit            : initial optimization variables, np array of shape (nVars,)
            numSteps         : number of steps. If None, knitro figures it out itself
            trustRegionScale : the initial trust region radius, larger means larger steps at the expense of quality
            optTol           : the relative stopping tolerance for the KKT error
            ftol             : the decrease threshold that tells if a step is significant in terms of objective reduction
            xtol             : the tolerance on the relative change of the optimization variables
            ftol_iters       : the number of consecutive unsignificant steps in terms of objective required to stop the optimization
            xtol_iters       : the number of consecutive unsignificant steps in terms of variable change required to stop the optimization 
            honorbounds      : whether the constraints should always be respected or not
            verbosity        : verbosity of the knitro solver: int from 0 to 5

        Returns:
            jointsFlatOpt : the optimal joints positions in the flact configuration for stage 1
        '''

        try:
            kc = KN_new()
        except:
            print("Failed to find a valid license.")

        self.lastFeasible = xInit

        rng = np.random.default_rng(1123)
        orderOfMagFlat = np.mean(np.abs(xInit[:self.nVarsFlat]))
        orderOfMagDep  = np.mean(np.abs(xInit[self.nVarsFlat:]))
        perturbFlat = 1.0e-2 * orderOfMagFlat * rng.normal(0.0, 1.0, size=(2*self.nJ,))
        perturbDep = 1.0e-3 * orderOfMagDep * rng.normal(0.0, 1.0, size=(2*self.nJ,))
        if self.pinnedVarsDepIdx.shape[0] > 0: perturbDep[self.pinnedVarsDepIdx] = 0.0
        perturbDir = np.logical_or(
            (0.0 > xInit[self.nVarsFlat:] + perturbDep), 
            (xInit[self.nVarsFlat:] + perturbDep > 1.0)
        )
        perturbDep[perturbDir] *= -1.0
        xInit = xInit + np.concatenate([perturbFlat, perturbDep], axis=0)
        
        valsFlatInit  = torch.tensor(xInit[:self.nVarsFlat])
        valsDepInit   = xInit[self.nVarsFlat:].reshape(-1, 2)
        if self.pinnedVarsDepIdx.shape[0] > 0: self.pinnedVarsDep = valsDepInit.reshape(-1,)[self.pinnedVarsDepIdx]
        jointsDepInit = torch.tensor(self.targetSurface.evaluate_list(valsDepInit.tolist().copy()))
        jointsFlatInit = torch.zeros(size=(self.nJ, 3))
        jointsFlatInit[:, :2] = valsFlatInit.reshape(-1, 2)

        self.objVals  = [ObjectivePlanarization(valsFlatInit, jointsDepInit, self.rodEdges, self.bndIndices, 
                                              self.idxEdgesA, self.idxEdgesB,
                                              self.jointsTris,
                                              self.jointsFlatInit, self.jointsDepInit, 
                                              weightLengths=self.weightLengths, weightAngles=self.weightAngles, 
                                              weightMeanAngle=self.weightMeanAngle, 
                                              weightBnd=self.weightBnd, weightFlatReg=self.weightFlatReg,
                                              bndOnly=self.bndOnly).item()]
        
        metricsTmp =  MetricsPlanarization(valsFlatInit, jointsDepInit, self.rodEdges, self.jointsFlatInit, self.jointsDepInit, 
                                         self.idxEdgesA, self.idxEdgesB, self.jointsTris, self.bndIndices,
                                         self.curves, self.jointsQuads, self.rodMaterial)
        for key in metricsTmp:
            self.metrics[key] = metricsTmp[key].reshape(1, -1)
            
        KN_load_param_file(kc, self.configPath)
        
        KN_set_double_param(kc, "delta",          trustRegionScale)
        KN_set_double_param(kc, "opttol",         optTol)
        KN_set_double_param(kc, "ftol",           ftol)
        KN_set_double_param(kc, "xtol",           xtol)
        KN_set_int_param(kc,    "ftol_iters",     ftol_iters)
        KN_set_int_param(kc,    "xtol_iters",     xtol_iters)
        KN_set_int_param(kc,    "par_numthreads", 12)
        KN_set_int_param(kc,    "honorbnds",      honorbounds) # 1: always enforce feasibility
        KN_set_int_param(kc,    "presolve",       0)  # 0: no presolve
        KN_set_int_param(kc,    "derivcheck",     0)  # watch out for the initial state when checking derivatives
        KN_set_int_param(kc,    "outlev",         verbosity)

        # Specify the dimensionality of the input vector
        nVars = xInit.reshape(-1,).shape[0] - self.pinnedVarsDepIdx.shape[0]
        varsIndices = KN_add_vars(kc, nVars)
        xInitVals = np.zeros(shape=(nVars,))
        xInitVals = xInit[self.unpinnedVarsIdx]

        # Set the initial guess to be the current degrees of freedom
        KN_set_var_primal_init_values(kc, xInitVals=xInitVals)
        
        # Set lower and upper bounds on the surface parameters
        idxSP = list(range(self.nVarsFlat, self.nVarsFlat+self.nVarsDep))
        lbSP  = len(idxSP) * [0.0]
        ubSP  = len(idxSP) * [1.0]
        KN_set_var_lobnds(kc, indexVars=idxSP, xLoBnds=lbSP)
        KN_set_var_upbnds(kc, indexVars=idxSP, xUpBnds=ubSP)
        
        # Set the constraints callbacks
        cIndices = KN_add_cons(kc, self.nConstraints)
        for i in range(self.nConstraints):
            if i <= len(self.jointsQuads):
                KN_set_con_lobnds(kc, cIndices[i], self.areaThresh)
            else:
                KN_set_con_lobnds(kc, cIndices[i], 0.0)
                KN_set_con_upbnds(kc, cIndices[i], np.pi)
            
        # Set the objective callbacks
        cb = KN_add_eval_callback(kc, evalObj=None, indexCons=None, funcCallback=self.callbackEvalF)
        KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE, jacIndexVars=KN_DENSE_ROWMAJOR, gradCallback=self.callbackEvalG)

        KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE) # 0 minimize, 1 maximize
        if not numSteps is None:
            KN_set_int_param(kc, "maxit", numSteps)

        # Add new point callback
        KN_set_newpt_callback(kc, self.newPtCallback, userParams=None)

        # Solve the problem
        self.times = [0.0]
        self.previousTime = time.time()
        nStatus = KN_solve(kc)

        # An example of obtaining solution information.
        nStatus, objSol, xOptFree, lambda_ = KN_get_solution(kc)
        if nStatus==0:
            print("The solution has converged.\nOptimal objective value: {:.2e}".format(objSol))
        else:
            print("The solution has not converged.\nThe status is {}".format(nStatus))

        # Delete the Knitro solver instance.
        KN_free(kc)
        
        xOpt = np.zeros_like(xInit)
        if self.pinnedVarsDepIdx.shape[0] > 0: xOpt[self.pinnedVarsIdx] = self.pinnedVarsDep
        xOpt[self.unpinnedVarsIdx] = xOptFree

        return xOpt
