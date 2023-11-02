import json
import numpy as np
import os
import pickle
import time
import torch
from torch.autograd.functional import jacobian

from CShell import GetEdgesFromCurves
from InitializationOptimizers import ToNumpy, ComputeJointsToCurves, ExtractCurvesPerFamily, ComputeQuads, QuadsTriangulationsSignedAreas, ComputeEdgesPerJointPerFamily, ComputeJointAnglesFromQuads, ComputeQuadsTriangulation
from OptimalSLEquilibrium import ComputeJointAngles
from StraightLinkageToJSON import SaveStraightLinkages
from VisUtils import PlotHist
from VisUtilsInitialization import PlotUV

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

def QuadsTriangulationsAreas3D(quads):
    '''
    Args:
        quads : torch tensor of shape (nQuads, 4, 3) containing the joints positions for each quad

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
    area11 = torch.linalg.norm(torch.cross(edge1Tri1, edge2Tri1), dim=-1) / 2
    # edge4Tri1 x edge3Tri1 / 2
    area12 = torch.linalg.norm(torch.cross(edge4Tri1, edge3Tri1), dim=-1) / 2

    edge1Tri2 = quads[:, 0, :] - quads[:, 1, :]
    edge2Tri2 = quads[:, 2, :] - quads[:, 1, :]
    edge3Tri2 = quads[:, 0, :] - quads[:, 3, :]
    edge4Tri2 = quads[:, 2, :] - quads[:, 3, :]

    # edge2Tri2 x edge1Tri2 / 2
    area21 = torch.linalg.norm(torch.cross(edge2Tri2, edge1Tri2), dim=-1) / 2
    # edge3Tri2 x edge4Tri2 / 2
    area22 = torch.linalg.norm(torch.cross(edge3Tri2, edge4Tri2), dim=-1) / 2

    signedAreas = torch.zeros(size=(quads.shape[0], 2, 2))
    signedAreas[..., 0, 0] = area11
    signedAreas[..., 0, 1] = area12
    signedAreas[..., 1, 0] = area21
    signedAreas[..., 1, 1] = area22

    return signedAreas

def ComputeJointsIncidentQuads(nJ, jointsQuads):
    '''Find the quads indices for each joint in the linkage
    
    Args:
        nJ: number of joints
        jointsQuads: list of list of 4 joint inidices
        
    Returns:
        jointsIncidentQuads: a dictionnary that gives the list of incident quads indices for each joint
    '''
    jointsIncidentQuads = {i:[] for i in range(nJ)}
    for i, quad in enumerate(jointsQuads):
        for jQuad in quad:
            jointsIncidentQuads[jQuad].append(i)
            
    return jointsIncidentQuads


def ComputeJointsIncidentEdges(nJ, rodEdges):
    '''Find the quads indices for each joint in the linkage
    
    Args:
        nJ: number of joints
        rodEdges: torch tensor of size (nEdges, 2)
        
    Returns:
        jointsIncidentEdges: a dictionnary that gives the list of incident edge indices for each joint
    '''
    jointsIncidentEdges = {i:[] for i in range(nJ)}

    for i, edge in enumerate(rodEdges):
        for jEdge in edge:
            jointsIncidentEdges[jEdge.item()].append(i)
            
    return jointsIncidentEdges

def ShearScaleLinkage(shearingFactor, jointsDep, uv, curves, curvesFamily):
    '''Shear the uv coordinates by a certain amount and scale them to match length of the target state
    
    Args:
        shearingFactor: the amout of shearing applied to the uv coordinates
        jointsDep: the torch tensor of shape (nJ, 3) that contains the joints positions in their deployed state
        uv: the torch tensor of shape (nJ, 2) that contains the uv parameters
        curves: list of lists that contain the joints indices for each curve
        curvesFamily: the curve family for each curve
        
    Returns:
        jointsFlat: the sheared joints in the flat layout (nJ, 2)
    '''
    
    # Compute the sheared version of the linkage
    depLengthCurves = np.array([sum([np.linalg.norm(jointsDep[crv[i+1]] - jointsDep[crv[i]]) for i in range(len(crv)-1)]) for crv in curves])
    scaleLengthA = np.mean(depLengthCurves[[fam == 0 for fam in curvesFamily]])
    scaleLengthB = np.mean(depLengthCurves[[fam == 1 for fam in curvesFamily]])

    shearingFactor = 1.5

    xRange = scaleLengthA
    yRange = scaleLengthB / np.sqrt(shearingFactor ** 2 + 1)

    initShearingTopo = np.array([[1.0, shearingFactor], [0.0, 1.0]])
    jointsFlat = uv.copy()
    jointsFlat[:, 0] *= xRange
    jointsFlat[:, 1] *= yRange
    jointsFlat = jointsFlat @ initShearingTopo.T
    jointsFlat -= np.mean(jointsFlat, axis=0, keepdims=True)
    
    return jointsFlat

########################################################################
##  FUNCTIONS FOR THE TARGET LINKAGE OPTIMIZATION
########################################################################


def ObjectiveTLOpt(jointsDepTmp, curves, jointsQuads, bndIndices, 
                   idxEdgesA, idxEdgesB,
                   surfArea, jointsDepInit, 
                   weightQuadsSpread=1.0, weightQuadsSpan=1.0, weightAnglesSpread=1.0,
                   weightCrvLength=1.0, weightBnd=1.0,
                   bndOnly=True):
    '''
    Args:
        jointsDepTmp        : the torch tensor of shape (nJ, 3) that contains the joints positions in their deployed state
        curves              : list of lists that contain the joints indices for each curve
        jointsQuads         : list of list of 4 elements containing the quads ordered in a clockwise fashion
        bndIndices          : torch tensor of shape (nB,) containing the boundary joint indices
        idxEdgesA           : np array containing the edges for each joint along curves of family A
        idxEdgesB           : np array containing the edges for each joint along curves of family B
        surfArea            : the total surface area
        jointsDepInit       : the initial joints positions in the deployed state, np array of shape (nJ, 3)
        weightQuadsSpread   : weight for the quads area similarity measure
        weightQuadsSpan     : weight for the quads span measure
        weightAnglesSpread  : weight for the average angle spread
        weightCrvLength     : weight that transforms polylines into geodesics
        weightBnd           : weight that keeps the boundary points at their locations
        bndOnly             : whether we only softly pin the boundary or all joints
    
    Returns:
        loss : the objective to be minimized
    '''
    
    quadsTmp = torch.mean(torch.sum(QuadsTriangulationsAreas3D(jointsDepTmp[jointsQuads, :]), dim=2), dim=1)
    lossQuadsSpread = torch.var(quadsTmp)
    lossQuadsSpan = (torch.nn.functional.relu(torch.sum(quadsTmp) - surfArea)) ** 2 # Do not penalize if too large!
    
    anglesDep   = ComputeJointAngles(jointsDepTmp, idxEdgesA, idxEdgesB)
    lossAnglesSpread = torch.var(anglesDep)
    
    lossCrvLengths = 0.0
    for crv in curves:
        # Leave the boundary point unaffected
        jCrv = torch.zeros(size=(len(crv), 3))
        jCrv[1:-1] = jointsDepTmp[crv[1:-1]]
        jCrv[0] = jointsDepTmp[crv[0]].detach()
        jCrv[-1] = jointsDepTmp[crv[-1]].detach()
        lossCrvLengths = lossCrvLengths + torch.sum((jCrv[1:, :] - jCrv[:-1, :]) ** 2)
    
    if bndOnly:
        lossBnd = torch.mean(torch.sum((jointsDepTmp[bndIndices] - torch.tensor(jointsDepInit[bndIndices])) ** 2, dim=1)) / 2
    else:
        lossBnd = torch.mean(torch.sum((jointsDepTmp - torch.tensor(jointsDepInit)) ** 2, dim=1)) / 2

    loss = (
        weightQuadsSpread * lossQuadsSpread 
        + weightQuadsSpan * lossQuadsSpan 
        + weightAnglesSpread * lossAnglesSpread
        + weightCrvLength * lossCrvLengths
        + weightBnd * lossBnd
    )
    
    return loss

def ConstraintsTLOpt(uvTmp, jointsQuads, quadsSignsInit):
    '''
    Args:
        uvTmp          : the torch tensor containing the current uv parameters (2 * nJ,)
        jointsQuads    : list of list of 4 elements containing the quads ordered in a clockwise fashion
        quadsSignsInit : torch tensor of shape (nQuads, 2, 2) containing the sign of the signed areas for each triangle for each triangulation
    
    Returns:
        constraints : the constraints of the problem optimization (nQuads,)
    '''
    
    constraints = torch.zeros(size=(len(jointsQuads),))

    uvReshapedTmp = uvTmp.reshape(-1, 2)
    quadsTmp = QuadsTriangulationsSignedAreas(uvReshapedTmp[jointsQuads, :]) * quadsSignsInit
    
    areasMaxMin = torch.maximum(torch.minimum(quadsTmp[:, 0, 0], quadsTmp[:, 0, 1]), 
                                torch.minimum(quadsTmp[:, 1, 0], quadsTmp[:, 1, 1]))
    constraints = areasMaxMin
    
    return constraints

def MetricsTLOpt(jointsDepTmp, curves, jointsQuads, bndIndices, 
                 idxEdgesA, idxEdgesB,
                 surfArea, jointsDepInit,):
    '''
    Args:
        jointsDepTmp        : the torch tensor of shape (nJ, 3) that contains the joints positions in their deployed state
        curves              : list of lists that contain the joints indices for each curve
        jointsQuads         : list of list of 4 elements containing the quads ordered in a clockwise fashion
        bndIndices          : torch tensor of shape (nB,) containing the boundary joint indices
        idxEdgesA           : np array containing the edges for each joint along curves of family A
        idxEdgesB           : np array containing the edges for each joint along curves of family B
        surfArea            : the total surface area
        jointsDepInit       : the initial joints positions in the deployed state, np array of shape (nJ, 3)
    
    Returns:
        metrics : a dictionnary filled with relevant metrics
    '''
    
    quadsTmp = torch.mean(torch.sum(QuadsTriangulationsAreas3D(jointsDepTmp[jointsQuads, :]), dim=2), dim=1)
    anglesDep = ComputeJointAngles(jointsDepTmp, idxEdgesA, idxEdgesB)

    metrics = {
        'Quadrilaterals Areas': ToNumpy(quadsTmp),
        'Rel. deviation to Total Area (%)': 100.0 * (torch.sum(quadsTmp) - surfArea) / surfArea,
        'Angle Deviation to Mean': ToNumpy(torch.abs(anglesDep - torch.mean(anglesDep))),
        'Angles Mean': ToNumpy(torch.mean(anglesDep)),
        'Boundary Fitting': ToNumpy(torch.linalg.norm(jointsDepTmp[bndIndices] - torch.tensor(jointsDepInit[bndIndices]), dim=1)),
    }

    return metrics
    

class TargetLinkageOptimizer():
    '''
    Attributes:
        curves               : list of lists that contain the joints indices for each curve
        curvesFamily         : the curve family for each curve
        targetSurface        : a surface provided by geomdl
        jointsDepInit        : the initial joints positions in their deployed state, np array of shape (nJ, 2)
        configPath           : a string giving the absolute path to the .opt file used for the optimization
        pinnedVarsIdx        : the list of pinned degrees of freedom in the deployed state (uv-coordinates)
        weightQuadsSpread    : weight for the quads area similarity measure
        weightQuadsSpan      : weight for the quads span measure
        weightAnglesSpread   : weight for the average angle spread
        weightCrvLength      : weight that transforms polylines into geodesics
        weightBnd            : weight that keeps the boundary points at their locations
        weightFlatReg        : a weight that enforces an arbitrary registration to the flat joints
        factorAreaConstraint : the multiplicative factor that determines how far we want to be from the self-intersection frontier
        forceComputeEqCS     : whether we want to report the EqCS criterion even when not needed in the loss (decreases efficiency)
        areaConstraintLSE    : whether to use LSE to aggregate area constraints
        bndOnly              : whether we only softly pin teh boundary or all joints
    '''

    def __init__(self, curves, curvesFamily, targetSurface, jointsDepInit, configPath, pinnedVarsIdx=None,
                 weightQuadsSpread=1.0, weightQuadsSpan=1.0, weightAnglesSpread=1.0, weightCrvLength=1.0,
                 weightBnd=1.0, factorAreaConstraint=0.0, bndOnly=True):
        self.curves               = curves
        self.curvesFamily         = curvesFamily
        self.targetSurface        = targetSurface
        self.jointsDepInit        = jointsDepInit
        self.weightQuadsSpread    = weightQuadsSpread
        self.weightQuadsSpan      = weightQuadsSpan
        self.weightAnglesSpread   = weightAnglesSpread
        self.weightCrvLength      = weightCrvLength
        self.weightBnd            = weightBnd
        self.factorAreaConstraint = factorAreaConstraint
        self.configPath           = configPath
        self.bndOnly              = bndOnly
        
        self.objVals              = []
        self.metrics              = {}
        self.times                = []
        self.previousTime         = 0.0
        
        self.nJ               = jointsDepInit.shape[0]
        self.jointsToCurves   = ComputeJointsToCurves(self.nJ, self.curvesFamily, self.curves)
        self.jointsQuads      = ComputeQuads(self.nJ, self.jointsToCurves, self.curves)
        self.quadsInit        = torch.mean(torch.sum(QuadsTriangulationsAreas3D(torch.tensor(jointsDepInit)[self.jointsQuads, :]), dim=2), dim=1)
        self.uvQuadsSignsInit = None # Filled later
        self.surfArea         = torch.sum(torch.abs(self.quadsInit))
        
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
        
        if pinnedVarsIdx is None: pinnedVarsIdx = []
        for varIdx in pinnedVarsIdx:
            assert varIdx <= 2 * self.nJ
        self.pinnedVarsIdx = np.unique(pinnedVarsIdx)
        self.unpinnedVarsIdx = np.array([i for i in range(2 * self.nJ) if not i in self.pinnedVarsIdx])
    
        self.nVars = self.unpinnedVarsIdx.shape[0]
        self.nConstraints = len(self.jointsQuads)
        
        depLengthCurvesInit = np.array([sum([np.linalg.norm(self.jointsDepInit[crv[i+1]] - self.jointsDepInit[crv[i]]) for i in range(len(crv)-1)]) for crv in self.curves])
        self.scaleLengthA = max(depLengthCurvesInit[[fam == 0 for fam in self.curvesFamily]])
        self.scaleLengthB = max(depLengthCurvesInit[[fam == 1 for fam in self.curvesFamily]])
        # Gives the expected area of a triangle in a triangulation of the quads formed by the grid
        self.areaThresh   = self.factorAreaConstraint / self.quadsInit.shape[0]
        
        # Rescale each weight using the normalization factors
        self.weightQuadsSpread    = self.weightQuadsSpread / (2.0 * torch.mean(self.quadsInit))
        self.weightQuadsSpan      = self.weightQuadsSpan / (2.0 * torch.sum(self.quadsInit))
        self.weightAnglesSpread   = self.weightAnglesSpread / 1.0
        self.weightCrvLength      = self.weightCrvLength / (2.0 * min(self.scaleLengthA, self.scaleLengthB) ** 2)
        self.weightBnd            = self.weightBnd / (2.0 * min(self.scaleLengthA, self.scaleLengthB) ** 2)

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
        
        currValsDep    = np.zeros(shape=(2 * self.nJ,))
        if self.pinnedVarsIdx.shape[0] > 0: currValsDep[self.pinnedVarsIdx] = self.pinnedVars
        currValsDep[self.unpinnedVarsIdx] = np.array(currVals)
        currValsDep = currValsDep.reshape(-1, 2)
        currJointsDep  = torch.tensor(self.targetSurface.evaluate_list(currValsDep.tolist().copy()))
        obj = ObjectiveTLOpt(currJointsDep, self.curves, self.jointsQuads, self.bndIndices, 
                             self.idxEdgesA, self.idxEdgesB,
                             self.surfArea, self.jointsDepInit, 
                             weightQuadsSpread=self.weightQuadsSpread, weightQuadsSpan=self.weightQuadsSpan, 
                             weightAnglesSpread=self.weightAnglesSpread, weightCrvLength=self.weightCrvLength, 
                             weightBnd=self.weightBnd, bndOnly=self.bndOnly).item()
        
        evalResult.obj = obj
        uvTmp = torch.tensor(currValsDep).reshape(-1,)
        constraints  = ConstraintsTLOpt(uvTmp, self.jointsQuads, self.uvQuadsSignsInit)
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
        
        currValsDep    = np.zeros(shape=(2 * self.nJ,))
        if self.pinnedVarsIdx.shape[0] > 0: currValsDep[self.pinnedVarsIdx] = self.pinnedVars
        currValsDep[self.unpinnedVarsIdx] = np.array(currVals)
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
        currJointsDep.requires_grad = True
        
        obj = ObjectiveTLOpt(currJointsDep, self.curves, self.jointsQuads, self.bndIndices, 
                             self.idxEdgesA, self.idxEdgesB,
                             self.surfArea, self.jointsDepInit, 
                             weightQuadsSpread=self.weightQuadsSpread, weightQuadsSpan=self.weightQuadsSpan, 
                             weightAnglesSpread=self.weightAnglesSpread, weightCrvLength=self.weightCrvLength, 
                             weightBnd=self.weightBnd, bndOnly=self.bndOnly)
        obj.backward(torch.ones_like(obj))
        
        # Backprop through surface evaluation
        gradObjJointsDep     = ToNumpy(currJointsDep.grad)
        gradObjValsDep_u     = np.sum(gradObjJointsDep * dS_du, axis=1)
        gradObjValsDep_v     = np.sum(gradObjJointsDep * dS_dv, axis=1)
        gradObjValsDep       = np.zeros(shape=(2 * self.nJ,))
        gradObjValsDep[ ::2] = gradObjValsDep_u
        gradObjValsDep[1::2] = gradObjValsDep_v
        gradObjValsDep       = gradObjValsDep[self.unpinnedVarsIdx]

        evalResult.objGrad = gradObjValsDep
        
        uvTmp = torch.tensor(currValsDep).reshape(-1,).detach().clone()
        constraintsFun  = lambda x: ConstraintsTLOpt(x, self.jointsQuads, self.uvQuadsSignsInit)
        constraintsJacs = jacobian(constraintsFun, uvTmp)
        evalResult.jac = ToNumpy(constraintsJacs).reshape(-1,)

        return 0

    def newPtCallback(self, kc, x, lbda, userParams):
        
        self.times.append(time.time() - self.previousTime)
        
        currVals = x

        currValsDep    = np.zeros(shape=(2 * self.nJ,))
        if self.pinnedVarsIdx.shape[0] > 0: currValsDep[self.pinnedVarsIdx] = self.pinnedVars
        currValsDep[self.unpinnedVarsIdx] = np.array(currVals)
        currValsDep = currValsDep.reshape(-1, 2)
        currJointsDep = torch.tensor(self.targetSurface.evaluate_list(currValsDep.tolist().copy()))

        self.objVals.append(ObjectiveTLOpt(currJointsDep, self.curves, self.jointsQuads, self.bndIndices, 
                                           self.idxEdgesA, self.idxEdgesB,
                                           self.surfArea, self.jointsDepInit, 
                                           weightQuadsSpread=self.weightQuadsSpread, weightQuadsSpan=self.weightQuadsSpan, 
                                           weightAnglesSpread=self.weightAnglesSpread, weightCrvLength=self.weightCrvLength, 
                                           weightBnd=self.weightBnd, bndOnly=self.bndOnly).item())
        
        metricsTmp = MetricsTLOpt(currJointsDep, self.curves, self.jointsQuads, self.bndIndices, 
                                  self.idxEdgesA, self.idxEdgesB,
                                  self.surfArea, self.jointsDepInit)
        for key in self.metrics.keys():
            self.metrics[key] = np.concatenate([self.metrics[key], metricsTmp[key].reshape(1, -1)], axis=0)
        
        uvTmp = torch.tensor(currValsDep).reshape(-1,)
        constraints  = ToNumpy(ConstraintsTLOpt(uvTmp, self.jointsQuads, self.uvQuadsSignsInit))
        quadsFeas = np.all(constraints[:len(self.jointsQuads)] >= self.areaThresh)
        hasNaN = np.any(np.isnan(currVals))
        betterObj = self.objVals[-1] < self.objVals[-2]
        if quadsFeas and (not hasNaN) and betterObj:
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
        orderOfMag  = np.mean(np.abs(xInit))
        perturbInit = 1.0e-3 * orderOfMag * rng.normal(0.0, 1.0, size=(2*self.nJ,))
        if self.pinnedVarsIdx.shape[0] > 0: perturbInit[self.pinnedVarsIdx] = 0.0
        perturbDir = np.logical_or(
            (0.0 > xInit + perturbInit), 
            (xInit + perturbInit > 1.0)
        )
        perturbInit[perturbDir] *= -1.0
        xInit = xInit + perturbInit
        
        valsDepInit   = xInit.reshape(-1, 2)
        if self.pinnedVarsIdx.shape[0] > 0: self.pinnedVars = valsDepInit.reshape(-1,)[self.pinnedVarsIdx]
        jointsDepInit = torch.tensor(self.targetSurface.evaluate_list(valsDepInit.tolist().copy()))
        uvQuadsInit = QuadsTriangulationsSignedAreas(torch.tensor(valsDepInit)[self.jointsQuads, :])
        self.uvQuadsSignsInit = torch.sign(uvQuadsInit)

        self.objVals  = [ObjectiveTLOpt(jointsDepInit, self.curves, self.jointsQuads, self.bndIndices, 
                                        self.idxEdgesA, self.idxEdgesB,
                                        self.surfArea, self.jointsDepInit, 
                                        weightQuadsSpread=self.weightQuadsSpread, weightQuadsSpan=self.weightQuadsSpan, 
                                        weightAnglesSpread=self.weightAnglesSpread, weightCrvLength=self.weightCrvLength, 
                                        weightBnd=self.weightBnd, bndOnly=self.bndOnly).item()]
        
        metricsTmp =  MetricsTLOpt(jointsDepInit, self.curves, self.jointsQuads, self.bndIndices, 
                                   self.idxEdgesA, self.idxEdgesB,
                                   self.surfArea, self.jointsDepInit)
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
        nVars = xInit.reshape(-1,).shape[0] - self.pinnedVarsIdx.shape[0]
        varsIndices = KN_add_vars(kc, nVars)

        # Set the initial guess to be the current degrees of freedom
        KN_set_var_primal_init_values(kc, xInitVals=xInit)
        
        # Set lower and upper bounds on the surface parameters
        idxSP = list(range(self.nVars))
        lbSP  = len(idxSP) * [0.0]
        ubSP  = len(idxSP) * [1.0]
        KN_set_var_lobnds(kc, indexVars=idxSP, xLoBnds=lbSP)
        KN_set_var_upbnds(kc, indexVars=idxSP, xUpBnds=ubSP)
        
        # Set the constraints callbacks
        cIndices = KN_add_cons(kc, self.nConstraints)
        for i in range(self.nConstraints):
            KN_set_con_lobnds(kc, cIndices[i], self.areaThresh)
            
        # Set the objective callbacks
        # cb = KN_add_eval_callback(kc, evalObj=True, indexCons=cIndices[0], funcCallback=self.callbackEvalF)
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
        if self.pinnedVarsIdx.shape[0] > 0: xOpt[self.pinnedVarsIdx] = self.pinnedVars
        xOpt[self.unpinnedVarsIdx] = xOptFree

        return xOpt

########################################################################
##  FUNCTIONS FOR THE CORNERS DELETION
########################################################################

def DeleteCorners(
        nJ, jointsQuads, rodEdges, 
        jointsFlat, jointsDep, uv,
        curves, curvesFamily,
        useLengthDiscrepancy=False
    ):
    '''Delete corners and update the linkage connectivity.
    
    Args:
        nJ: number of joints
        jointsQuads: list of list of 4 elements containing the quads ordered in a clockwise fashion
        jointsFlat: torch tensor of shape (nJ, 2) that contains the joints positions in their flat state
        jointsDep: the torch tensor of shape (nJ, 3) that contains the joints positions in their deployed state
        uv: the torch tensor of shape (nJ, 2) that contains the uv parameters
        curves: list of lists that contain the joints indices for each curve
        curvesFamily: the curve family for each curve
        useLengthDiscrepancy: whether we want to look at edge lengths discrepancies when removing joints
    
    Returns:
        dictOut: a dictionnary that contains the updated linkage connectivity
    '''
    
    jointsIncidentQuads = ComputeJointsIncidentQuads(nJ, jointsQuads)
    jointsIncidentEdges = ComputeJointsIncidentEdges(nJ, rodEdges)
    
    _, valence = torch.unique(rodEdges, return_counts=True)
    corners = torch.argwhere(valence == 2).reshape(-1,)
    cornersQuads = [jointsIncidentQuads[jId.item()][0] for jId in corners]
    
    # Compute outliers of the angles distribution
    anglesQuads = ComputeJointAnglesFromQuads(torch.tensor(jointsDep), jointsQuads, flat=False)
    averageQuadAngles = torch.mean(anglesQuads, dim=0)
    devQuadAngles = torch.mean(torch.abs(anglesQuads - averageQuadAngles.reshape(1, -1)), dim=1)
    firstQuartile, thirdQuartile = np.percentile(devQuadAngles, q=[25, 75])
    iqDevQuadAngles = thirdQuartile - firstQuartile
    ubDevQuadAngles = thirdQuartile + 1.5 * iqDevQuadAngles
    outliersAngles = torch.argwhere(devQuadAngles > ubDevQuadAngles).reshape(-1,)
    
    # Compute outliers of the lengths discrepancy distribution
    outliersLengths = []
    if useLengthDiscrepancy:
        reLengths = np.linalg.norm(jointsDep[rodEdges[:, 1], :] - jointsDep[rodEdges[:, 0], :], axis=1)
        incommingLengths = [[reLengths[eId.item()] for eId in edge] for edge in rodEdges]
        devIncommingLengths = [max(il) / min(il) for il in incommingLengths]
        firstQuartile, thirdQuartile = np.percentile(devIncommingLengths, q=[25, 75])
        iqDevIncommingLengths = thirdQuartile - firstQuartile
        ubDevIncommingLengths = thirdQuartile + 1.5 * iqDevIncommingLengths
        outliersLengths = np.argwhere(devIncommingLengths > ubDevIncommingLengths).reshape(-1,)
    
    # Delete joints
    deleteJoints = []

    for i in range(len(corners)):
        if cornersQuads[i] in outliersAngles.tolist()+outliersLengths:
            deleteJoints.append(corners[i].item())

    deleteJoints = np.array(deleteJoints, dtype=np.int32)
    keepJoints = np.array([jId for jId in range(nJ) if not jId in deleteJoints], dtype=np.int32)
    
    nJNew = nJ - deleteJoints.shape[0]
    curvesNew = [[int(jId - sum(jId > deleteJoints)) for jId in crv if not jId in deleteJoints] for crv in curves]
    curvesFamilyNew = curvesFamily
    rodEdgesNew, rodEdgeToCurveNew = GetEdgesFromCurves(curvesNew)
    jointsToCurvesNew = ComputeJointsToCurves(nJNew, curvesFamilyNew, curvesNew)
    jointsQuadsNew = ComputeQuads(nJNew, jointsToCurvesNew, curvesNew)

    curvesANew, curvesBNew = ExtractCurvesPerFamily(curvesFamilyNew, curvesNew)
    idxEdgesANew, idxEdgesBNew = ComputeEdgesPerJointPerFamily(nJNew, curvesANew, curvesBNew)
    jointsTrisNew = ComputeQuadsTriangulation(jointsQuadsNew, transpose=False)

    jointsFlatNew = jointsFlat[keepJoints]
    jointsDepNew = jointsDep[keepJoints]
    uvNew = uv[keepJoints]
    
    dictOut = {
        'nJ': nJNew,
        'curves': curvesNew,
        'curvesFamily': curvesFamilyNew,
        'rodEdges': rodEdgesNew,
        'rodEdgeToCurve': rodEdgeToCurveNew,
        'jointsToCurves': jointsToCurvesNew,
        'jointsQuads': jointsQuadsNew,
        'curvesA': curvesANew,
        'curvesB': curvesBNew,
        'idxEdgesA': idxEdgesANew,
        'idxEdgesB': idxEdgesBNew,
        'jointsTris': jointsTrisNew,
        'jointsFlat': jointsFlatNew,
        'jointsDep': jointsDepNew,
        'uv': uvNew,
    }
    
    return dictOut

########################################################################
##  Prune Linkage
########################################################################

def LinkagePruning(
        jointsFlat, jointsDep, uv, shearingFactor,
        nJ, curves, curvesFamily, targetSurface, 
        configPath, dictWeights, nSteps=10, verbosity=2,
        pathToSave=None, rodMaterial=None,
        jointMarkerParams=None, curveWidthParams=None,
    ):
    '''Repeats the target linkage optimization and corners pruning several times until no more corner is deleted, and instantiates the flat joints layout

    Args:
        jointsFlat: torch tensor of shape (nJ, 2) that contains the joints positions in their flat state
        jointsDep: the torch tensor of shape (nJ, 3) that contains the joints positions in their deployed state
        uv: the torch tensor of shape (nJ, 2) that contains the uv parameters
        shearingFactor: the shearing amount applied to the flat layout
        nJ: number of joints
        curves: list of lists that contain the joints indices for each curve
        curvesFamily: the curve family for each curve
        targetSurface: target surface used by the TargetLinkageOptimizer
        configPath: path to the knitro configuration file
        dictWeights: dictionnary containing all the weights needed by the TargetLinkageOptimizer
        nSteps: maximum number of steps taken before breaking
        verbosity: verbosity for the TargetLinkageOptimizer: int from 0 to 5
        pathToSave: the path where we save intermediate and final geometries (list containing the path to the save folder and the model's name)
        rodMaterial: the rod material to use in order to save the deployed models
        jointMarkerParams: plotting setting for PlotUV
        curveWidthParams: plotting setting for PlutUV
        
    Returns:
        jointsFlat, jointsDep, uv, nJ, curves, curvesFamily: updated versions of the inputs
    '''
    
    breakNextIt = False
    saveFiles = False
    listMetrics = []
    
    rodEdges, rodEdgeToCurve = GetEdgesFromCurves(curves)
    jointsToCurves = ComputeJointsToCurves(nJ, curvesFamily, curves)
    jointsQuads = ComputeQuads(nJ, jointsToCurves, curves)
    jointsTris = ComputeQuadsTriangulation(jointsQuads, transpose=False)
    curvesA, curvesB = ExtractCurvesPerFamily(curvesFamily, curves)
    idxEdgesA, idxEdgesB = ComputeEdgesPerJointPerFamily(nJ, curvesA, curvesB)
    
    if pathToSave is not None:
        assert len(pathToSave) == 2
        saveFiles = True
        fnSLPreOptim = os.path.join(pathToSave[0], "SL_{}_Pre_".format(pathToSave[1]) + "{}.json")
        fnSLPostOptim = os.path.join(pathToSave[0], "SL_{}_Post_".format(pathToSave[1]) + "{}.json")
        fnSLFinal = os.path.join(pathToSave[0], "SL_{}_FinalPruning.json".format(pathToSave[1]))
        fnADPreOptim = os.path.join(pathToSave[0], "angle_distrib_{}_pre_".format(pathToSave[1].lower()) + "{}.png")
        fnADPostOptim = os.path.join(pathToSave[0], "angle_distrib_{}_post_".format(pathToSave[1].lower()) + "{}.png")
        fnADFinal = os.path.join(pathToSave[0], "angle_distrib_{}_final_pruning.png".format(pathToSave[1]))
        angleBinRes = np.pi/11
        angleBins = np.linspace(0.0, np.pi, int(np.pi / angleBinRes) + 1)
        fnUVPreOptim = os.path.join(pathToSave[0], "diagram_uv_{}_pre_".format(pathToSave[1].lower()) + "{}.png")
        fnUVPostOptim = os.path.join(pathToSave[0], "diagram_uv_{}_post_".format(pathToSave[1].lower()) + "{}.png")
        fnUVFinal = os.path.join(pathToSave[0], "diagram_uv_{}_final_pruning.png".format(pathToSave[1].lower()))
        fnWeigths = os.path.join(pathToSave[0], "weights_pruning_{}.png".format(pathToSave[1].lower()))
        fnMetrics = os.path.join(pathToSave[0], "metrics_pruning_{}.png".format(pathToSave[1].lower()))
    
    for step in range(nSteps):
        
        if saveFiles:
            jointsFlat = ShearScaleLinkage(shearingFactor, jointsDep, uv, curves, curvesFamily)
            jointsFlatSave = np.zeros(shape=(jointsFlat.shape[0], 3))
            jointsFlatSave[:, :2] = jointsFlat
            SaveStraightLinkages(fnSLPreOptim.format(str(step).zfill(3)), jointsFlatSave, jointsDep, 
                                 curves, curvesFamily, rodEdges, jointsQuads, 
                                 jointsTris, rodMaterial)
            anglesDep = ToNumpy(ComputeJointAngles(torch.tensor(jointsDep), idxEdgesA, idxEdgesB))
            PlotHist(anglesDep, angleBins,
                    col1="tab:orange", showText=False, fn=fnADPreOptim.format(str(step).zfill(3)), xTicks=[0.0, np.pi])
            PlotUV(uv, curves, pathToSave=fnUVPreOptim.format(str(step).zfill(3)),
                   jointMarkerParams=jointMarkerParams, curveWidthParams=curveWidthParams)
        
        tlo = TargetLinkageOptimizer(
            curves, curvesFamily, targetSurface, jointsDep, configPath, pinnedVarsIdx=None,
            weightQuadsSpread=dictWeights['quadsSpread'], weightQuadsSpan=dictWeights['quadsSpan'], 
            weightAnglesSpread=dictWeights['anglesSpread'], weightCrvLength=dictWeights['crvLength'],
            weightBnd=dictWeights['bnd'], factorAreaConstraint=dictWeights['areaConstraint'], bndOnly=dictWeights['bndOnly'],
        )
        
        uv = tlo.Optimize(
            uv.reshape(-1,), numSteps=10000, trustRegionScale=1.0e-2,
            optTol=1.0e-5, ftol=1.0e-7, xtol=1.0e-7, 
            ftol_iters=5, xtol_iters=5, honorbounds=1,
            verbosity=verbosity,
        ).reshape(-1, 2)
        
        listMetrics.append(tlo.metrics)
        
        if breakNextIt:
            jointsDep = torch.tensor(targetSurface.evaluate_list(uv.tolist().copy()))
            break
        
        if saveFiles:
            jointsFlat = ShearScaleLinkage(shearingFactor, jointsDep, uv, curves, curvesFamily)
            jointsFlatSave = np.zeros(shape=(jointsFlat.shape[0], 3))
            jointsFlatSave[:, :2] = jointsFlat
            SaveStraightLinkages(fnSLPostOptim.format(str(step).zfill(3)), jointsFlatSave, jointsDep, 
                                 curves, curvesFamily, rodEdges, jointsQuads, 
                                 jointsTris, rodMaterial)
            anglesDep = ToNumpy(ComputeJointAngles(torch.tensor(jointsDep), idxEdgesA, idxEdgesB))
            PlotHist(anglesDep, angleBins,
                    col1="tab:orange", showText=False, fn=fnADPostOptim.format(str(step).zfill(3)), xTicks=[0.0, np.pi])
            PlotUV(uv, curves, pathToSave=fnUVPostOptim.format(str(step).zfill(3)),
                   jointMarkerParams=jointMarkerParams, curveWidthParams=curveWidthParams)
        
        dictOut = DeleteCorners(
            nJ, tlo.jointsQuads, tlo.rodEdges, 
            jointsFlat, jointsDep, uv,
            curves, curvesFamily,
            useLengthDiscrepancy=False
        )
        
        breakNextIt = (nJ == dictOut['nJ'])
        nJ = dictOut['nJ']
        curves = dictOut['curves']
        curvesFamily = dictOut['curvesFamily']
        rodEdges = dictOut['rodEdges']
        jointsQuads = dictOut['jointsQuads']
        curvesA, curvesB = dictOut['curvesA'], dictOut['curvesB']
        idxEdgesA, idxEdgesB = dictOut['idxEdgesA'], dictOut['idxEdgesB']
        jointsTris = dictOut['jointsTris']
        jointsFlat = dictOut['jointsFlat']
        jointsDep = dictOut['jointsDep']
        uv = dictOut['uv']
        
    # Compute the sheared version of the linkage
    jointsFlat = ShearScaleLinkage(shearingFactor, jointsDep, uv, curves, curvesFamily)
    jointsFlatSave = np.zeros(shape=(jointsFlat.shape[0], 3))
    jointsFlatSave[:, :2] = jointsFlat
    if saveFiles:
        SaveStraightLinkages(fnSLFinal, jointsFlatSave, jointsDep, 
                             curves, curvesFamily, rodEdges, jointsQuads, 
                             jointsTris, rodMaterial)
        anglesDep = ToNumpy(ComputeJointAngles(torch.tensor(jointsDep), idxEdgesA, idxEdgesB))
        PlotHist(anglesDep, angleBins,
                 col1="tab:orange", showText=False, fn=fnADFinal.format(str(step).zfill(3)), xTicks=[0.0, np.pi])
        PlotUV(uv, curves, pathToSave=fnUVFinal,
               jointMarkerParams=jointMarkerParams, curveWidthParams=curveWidthParams)
        with open(fnWeigths, 'w') as f:
            json.dump(dictWeights, f)
        with open(fnMetrics, 'wb') as f:
            pickle.dump(listMetrics, f)
    
    return jointsFlat, jointsDep, uv, nJ, curves, curvesFamily