import os
import sys as _sys

SCRIPT_PATH = os.path.abspath(os.getcwd())
split = SCRIPT_PATH.split("Cshells")
if len(split)<2:
    print("Please rename the repository 'Cshells'")
    raise ValueError
PATH_TO_CUBICSPLINES = split[0] + "Cshells/ext/torchcubicspline"
_sys.path.append(PATH_TO_CUBICSPLINES)

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import torch
from torchcubicspline import (natural_cubic_spline_coeffs, NaturalCubicSpline)
from torch.autograd.functional import jvp, jacobian
from torch.autograd import grad

from LaplacianSmoothing import ComputeLaplacianCP, LeastSquaresLaplacian, LeastSquaresLaplacianFullGradient, LeastSquaresLaplacianFullHVP
from Utils import MakeConstantSpeed
from VisUtils import CShellOptimizationCallback
from VisUtilsCurveLinkage import IndicesDiscretePositionsNoDuplicate

torch.set_default_dtype(torch.float64)

import MeshFEM
import ElasticRods
import average_angle_linkages
from bending_validation import suppress_stdout as so
import elastic_rods
from linkage_vis import LinkageViewer, LinkageViewerWithSurface
from open_average_angle_linkage import open_average_angle_linkage
import py_newton_optimizer
try:
    KNITRO_FOUND = True
    import cshell_optimization
    import linkage_optimization
except Exception as e:
    KNITRO_FOUND = False
    print("Knitro may not have been found: {}.".format(e))

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

class CShell:
    '''
    Attributes:
        curvesDoFs                : torch tensor of size (2*nJ + nIntCP,) this excludes alphaTar (note that this is different if symmetry is not None)
        fullCurvesDoF             : torch tensor of size (2*nJ + nIntCP,), which is equal to curvesDoF in case symmetry is None
        joints                    : torch tensor of size (nJ, 3)
        nJ                        : total number of joints
        rodEdges                  : list of edges between joints
        valence                   : torch tensor of size (nJ,) giving the valence of each joint
        rodEdgesToCurve           : list of curve indices that tell which curve each rod edge belongs to
        curves                    : list of list giving the joints through which each curve passes
        curvesFamily              : list containing whether the curve is labeled as A (0) or B (1)
        curvesPlotXLim            : the x limits for plotting the curve linkage
        curvesPlotYLim            : the y limits for plotting the curve linkage
        rodEdgesFamily            : list containing whether the rod edge is labeled as A (0) or B (1)
        jointsToCurves            : list of nJ dictionnaries giving the curve index for each family
        jointsQuads               : list of nQuads list of 4 elements containing the quads with arbitrary order
        quadsOrientation          : torch tensor of shape (nQuads,) containing the sign 
        nCPperRodEdge             : number of points to add between edges (list of list with nJcurve-1 elements)
        cachedQuantities          : some quantities to be reused for speeding up the equilibrium solve
        alphaTar                  : targeted average opening angle for the deployed linkage
        subdivision               : number of subdivision for each rod segment (including overlapping edges)
        symmetry                  : an element of class CurvesDoFReducer given by Symmetries.py
        mult                      : serves controlling the constant speed reparameterization of the splines
        curvesWithCP              : list of list giving the control points used for each curve
        segmentsWithCP            : list of nEdges lists containing the segments with the new CPs
        edgesWithCP               : torch tensor of shape (nEdges, 2) containing all the edges
        edgeWithCPToCurve         : list of nCurves 
        splines                   : list of nCurves cubic splines
        discList                  : list of tensors of shape (subdivision*(nJ-1) + 2, 3) containing the centerline position along rods
        discSList                 : list of nCurves tensors of shape (subdivision*(nJ-1) + 2,) containing the curve parameters of the discretized rods
        restLengths               : tensor containing the rest lengths of each edge, starting from the free edges to the joint edges
        restKappas                : tensor containing the rest curvatures at each vertex
        restQuantities            : tensor containing the rest quantities of the discretized linkage. First the rest length for the free edges
                                    then the rest lengths at the joints, and lastly the rest kappas.
        designParameters          : tensor containing the rest quantities of the discretized linkage. First the rest kappas,
                                    then the rest lengths for each segment
        fullRodSplines            : list containing one spline for each curve (scipy version)
        segmentSplines            : list containing one constant speed spline per rod segment
        synchronized              : tells whether the curve linkage has all its quantities synchronized (to avoid recomputing)
        freeAngles                : a list containing the joint indices that are left non actuated (by default, all of them are actuated during deployment)
        rodMaterial               : an elastic_rods.RodMaterial object to be passed
        pathSurf                  : gives the path to the target surface. If None, it is assumed that the target surface is inferred from the deployed linkage
        useSAL                    : whether or not to use Surface Attracted Linkage
        attractionWeight          : the weight in front of the attraction term used during deployment
        attractionMesh            : dictionnary containing an array of shape (nTargetV, 3) called "V" and an array of shape (nTargetF, 3) called "F" that corresponds to the attraction mesh,
                                    and potentially an flattened array of joints positions (3*nJ,) "targetJP".
        targetMesh                : dictionnary containing an array of shape (nTargetV, 3) called "V" and an array of shape (nTargetF, 3) called "F" that corresponds to the target mesh,
                                    and potentially an flattened array of joints positions (3*nJ,) "targetJP".
        linkagesGuess             : if provided, should be a dictionnary that both contains the "flat" and "deployed" guesses
        createViewers             : whether we create viewers or not
        newtonOptimizerOptions    : a newton optimizer options object
        flatLinkage               : an elastic rod object
        flatView                  : the linkage viewer associated to flatLinkage
        deployedLinkage           : the deployed version of flatLinkage
        deployedView              : the linkage viewer associated to deployedLinkage
        optimizeAlpha             : whether the target average opening angle is to be optimized or not
        cpRegWeight               : weight in front of the control points regularization (beta_cp / l0^2 in the writeup)
        featJoints                : list storing the joints for which the weights have been modified in the target surface fitting term
        linkageOptimizer          : a CShellOptimization object
        optimizationCallback      : the object that collects the objective value, time, and gradient magnitude on the parameters
        jacCurDoFToDP             : the Jacobian from curve DoF to design parameters
        jacDPToDepDoF             : the Jacobian from design parameters to deployed DoF
        jacFull                   : the Jacobian from curves DoF to deployed DoF
        jacFullPos2D              : the Jacobian from the design parameters to the positional DoF for 2D
        jacFullPos3D              : the Jacobian from the design parameters to the positional DoF for 3D
        jacFullJoints2D           : the Jacobian from the design parameters to the joints positions for 2D
        jacFullJoints3D           : the Jacobian from the design parameters to the joints positions for 3D
        numOpeningSteps           : number of opening steps for deploying the linkage
        maxNewtonIterIntermediate : maximum number of iterations per opening step
        
    Important note:
        In case symmetry is specified, the arguments nJ, curves, curvesFamily, nCPperRodEdge can be simply set to None
    and will be provided by symmetry
    '''

    def __init__(self, curvesDoF, nJ, curves, curvesFamily, nCPperRodEdge, alphaTar, mult, subdivision, 
                 symmetry=None, freeAngles=None,
                 rodMaterial=None, pathSurf=None, newtonOptimizerOptions=None, optimizeAlpha=False, 
                 useSAL=False, attractionWeight=1.0e-4, attractionJointPosWeight=0.1, attractionMesh=None, targetMesh=None,
                 dictWeights=None, linkagesGuess=None, createViewers=True,
                 numOpeningSteps=40, maxNewtonIterIntermediate=20, flatOnly=False):
        
        self.curvesDoF        = curvesDoF
        if not symmetry is None:
            self.fullCurvesDoF = symmetry.MapToFullCurvesDoF(curvesDoF)
        else:
            self.fullCurvesDoF = curvesDoF

        self.symmetry         = symmetry
        if not symmetry is None:
            self.nJ            = symmetry.nJ
            self.curves        = symmetry.curves
            self.curvesFamily  = symmetry.curvesFamily
            self.nCPperRodEdge = symmetry.nCPperRodEdge
        else:
            self.nJ            = nJ
            self.curves        = curves
            self.curvesFamily  = curvesFamily
            self.nCPperRodEdge = nCPperRodEdge

        self.joints           = torch.zeros(size=(self.nJ, 3))
        self.joints[:, :2]    = self.fullCurvesDoF[:2*self.nJ].reshape(-1, 2)
        self.rodEdges, self.rodEdgesToCurve = GetEdgesFromCurves(self.curves)
        self.curvesEnds       = np.unique([crv[0] for crv in self.curves] + [crv[-1] for crv in self.curves])
        self.rodEdgesFamily   = [self.curvesFamily[curveID] for curveID in self.rodEdgesToCurve]
        
        self.synchronized     = True
        self.cachedQuantities = {}
        self.numOpeningSteps  = numOpeningSteps
        self.maxNewtonIterIntermediate = maxNewtonIterIntermediate
        _, self.valence       = torch.unique(self.rodEdges, return_counts=True)
        self.curvesPlotXLim   = None
        self.curvesPlotYLim   = None

        self.flatOnly        = flatOnly
        # Some assertions on the dimension
        nIntCP = sum([a for subList in self.nCPperRodEdge for a in subList])
        if (2*self.nJ + nIntCP != self.fullCurvesDoF.shape[0]): 
            raise ValueError("Mismatch between full curvesDoF ({} elements), nCPperRodEdge ({} intermediate CP), and nJ ({} joints).".format(self.fullCurvesDoF.shape[0], nIntCP, self.nJ))
        if (len(self.curves) != len(self.nCPperRodEdge)): 
            raise ValueError("Length mismatch between curves ({} curves) and nCPperRodEdge ({} curves).".format(len(self.curves), len(self.nCPperRodEdge)))
        for idxCurve, (crv, nIntCP) in enumerate(zip(self.curves, self.nCPperRodEdge)):
            if len(crv) != len(nIntCP) + 1: 
                raise ValueError("The number of joints ({} joints) and the number of rod edges ({} rod edges) along curve {} do not match in nCPperRodEdge.".format(len(crv), len(nIntCP), idxCurve))

        # Compute the map from joints to curves
        self.jointsToCurve    = None
        self.jointsQuads      = None
        self.quadsOrientation = None
        self.ComputeJointsToCurves()
        self.ComputeQuads()
        self.ComputeQuadsOrientation()

        self.mult        = mult
        self.subdivision = subdivision

        # Update the connectivity when adding control points
        self.curvesWithCP      = None
        self.segmentsWithCP    = None
        self.edgesWithCP       = None
        self.edgeWithCPToCurve = None
        self.edgeWithCPFamily  = None
        self.InsertControlPointsInCurves()

        # Compute the control points
        self.controlPoints      = None
        self.controlPointsFixed = None
        self.FullCurvesDoFToControlPointsMap()
        self.ResetFixedCPPosition()

        # Compute the Laplacian on the control points
        self.lapCP = ComputeLaplacianCP(self.edgesWithCP, self.curvesWithCP, self.curvesFamily, disconnectEnd=True, fixedIdx=[])

        # Compute the splines and their discretization
        self.splines   = None
        self.discList  = None
        self.discSList = None
        self.ControlPointsToDiscretePositionsMap(addFreeEnds=True)

        # Update the rest quantities
        self.restLengths      = None
        self.restKappas       = None
        self.restQuantities   = None
        self.designParameters = None
        self.DiscretePositionsToDesignParametersMap()

        # Compute the splines for each segment
        self.fullRodSplines = None
        self.segmentSplines = None
        self.MakeSegmentSplines()

        # Instanciate the rod linkages
        self.alphaTar                 = torch.tensor(alphaTar)
        self.pathSurf                 = pathSurf
        self.useSAL                   = useSAL
        self.attractionWeight         = attractionWeight
        self.attractionJointPosWeight = np.clip(attractionJointPosWeight, 0.0, 1.0)
        self.attractionMesh           = attractionMesh
        self.targetMesh               = targetMesh

        if freeAngles is None: self.freeAngles = [] # All angles are actuated by default
        else: self.freeAngles = freeAngles
        if rodMaterial is None: self.rodMaterial = elastic_rods.RodMaterial('rectangle', 2000, 0.3, [0.40, 0.15], keepCrossSectionMesh=True)
        else: self.rodMaterial = rodMaterial
        self.createViewers   = createViewers
        self.flatLinkage     = None
        self.flatView        = None
        self.deployedLinkage = None
        self.deployedView    = None
        if newtonOptimizerOptions is None:
            self.newtonOptimizerOptions = py_newton_optimizer.NewtonOptimizerOptions()
            self.newtonOptimizerOptions.gradTol = 1e-7
            self.newtonOptimizerOptions.verbose = 1
            self.newtonOptimizerOptions.beta = 1e-8
            self.newtonOptimizerOptions.niter = 500
            self.newtonOptimizerOptions.verboseNonPosDef = False
        else:
            self.newtonOptimizerOptions = newtonOptimizerOptions

        if linkagesGuess and "targetJP" in linkagesGuess:
            nTargetJP = linkagesGuess["targetJP"].reshape(-1, 3).shape[0]
            if nTargetJP != self.nJ:
                raise ValueError("Mismatch between the number of target joints positions ({} joints) and the actual number of joints ({} joints).".format(nTargetJP, self.nJ))
        self.MakeLinkages(linkagesGuess=linkagesGuess, flatOnly=self.flatOnly)
        self.optimizeAlpha          = optimizeAlpha
        self.linkageOptimizer       = None
        self.optimizationCallback   = None
        self.cpRegWeight            = None
        self.featJoints             = [idx for idx in range(self.valence.shape[0]) if self.valence[idx] == 2]
        if dictWeights is None:
            # Some default values
            self.dictWeights = {
                "beta": 1.0e3,
                "gamma": 0.0,
                "smoothingWeight": 1.0,
                "rlRegWeight": 1.0,
                "contactForceWeight": 0.0,
                "cpRegWeight": 1.0
            }
        else:
            self.dictWeights = dictWeights
        
        # Store the Jacobians
        self.jacCurDoFToDP   = [None, False]
        self.jacDPToDepDoF   = [None, False]
        self.jacFull         = [None, False]
        self.jacFullJoints2D = [None, False]
        self.jacFullJoints3D = [None, False]
        self.jacFullPos2D    = [None, False]
        self.jacFullPos3D    = [None, False]
        
        if (not self.flatOnly):
            # Create the linkage optimizer
            if not KNITRO_FOUND:
                print("Instantiation will fail since knitro 10 has not been found.")
            self.MakeLinkageOptimizer(createDeployedViewer=self.createViewers)

    #############################################
    #############       UTILS       #############
    #############################################

    def UpdateCShell(self, curvesDoF, alpha, force=False, 
                     resetCPFixed=False,
                     skipRL=False, resetRL=False, 
                     commitLinesearchLinkage=False,
                     cacheDeployed=False, useCached=False):
        '''Runs the full pipeline.

        Args:
            curvesDoF               : the new joints' position and orthogonal offsets (if symmetry is None)
            alpha                   : new target opening angle
            force                   : whether we want to force recomputing all the quantities
            resetCPFixed            : reset the fixed control points for the laplacian on the control points
            skipRL                  : whether we just want to run the "curves" section (no rod linkage)
            resetRL                 : whether we create a new rod linkage/linkage optimizer or not
            commitLinesearchLinkage : whether we simply update the linesearch linkage or both base and linesearch
            cacheDeployed           : whether we want to reuse the current deployed state for later
            useCached               : whether we run the optimizations using the cached deployed linkage's DoFs
        '''

        if torch.linalg.norm(self.curvesDoF - curvesDoF) < 1.0e-16 and not force:
            if (abs(alpha - self.alphaTar) > 1.0e-16):
                self.jacDPToDepDoF[1]   = False
                self.jacFull[1]         = False
                self.jacFullJoints2D[1] = False
                self.jacFullJoints3D[1] = False
                self.jacFullPos2D[1]    = False
                self.jacFullPos3D[1]    = False
            if not torch.is_tensor(self.alphaTar):
                self.alphaTar = torch.tensor(alpha)
            else:
                self.alphaTar = alpha
            self.UpdateLinkageOptimizer(commitLinesearchLinkage=commitLinesearchLinkage)
        else:
            self.alphaTar  = alpha
            self.curvesDoF = curvesDoF
            self.FullCurvesDoFToControlPointsMap()
            self.ControlPointsToDiscretePositionsMap()
            self.DiscretePositionsToDesignParametersMap()
            if not skipRL:
                if resetRL:
                    self.MakeSegmentSplines()
                    self.MakeLinkages(useCached)
                    self.MakeLinkageOptimizer()
                else:
                    self.UpdateLinkageOptimizer(commitLinesearchLinkage=commitLinesearchLinkage)

            # The Jacobians should be recomputed
            self.jacCurDoFToDP[1]   = False
            self.jacDPToDepDoF[1]   = False
            self.jacFull[1]         = False
            self.jacFullJoints2D[1] = False
            self.jacFullJoints3D[1] = False
            self.jacFullPos2D[1]    = False
            self.jacFullPos3D[1]    = False

        if resetCPFixed:
            self.ResetFixedCPPosition()

        if cacheDeployed:
            try:
                self.CacheDeployedQuantities()
            except Exception as e:
                print("Quantities have not been cached: {}".format(e))

    def CleanGradients(self):
        self.curvesDoF = self.curvesDoF.detach()
        self.controlPoints = self.controlPoints.detach()
        self.discList = [disc.detach() for disc in self.discList]
        self.restQuantities = self.restQuantities.detach()
        self.designParameters = self.designParameters.detach()

    def CacheDeployedQuantities(self):
        self.cachedQuantities["curvesDoF"]        = self.GetFullDoF().detach()
        self.cachedQuantities["designParameters"] = self.linkageOptimizer.getFullDesignParameters()
        self.cachedQuantities["deployedDoF"]      = self.deployedLinkage.getDoFs()

    def ComputeJointsToCurves(self):
        '''
        Updated attributes:
            jointsToCurves : list of nJ dictionnaries giving the curve index for each family
        '''
        self.jointsToCurves = [{} for _ in range(self.nJ)]

        for idxCurve, (fam, crv) in enumerate(zip(self.curvesFamily, self.curves)):
            strFam = "A" if fam==0 else "B"
            for idxJoint in crv:
                self.jointsToCurves[idxJoint][strFam] = idxCurve

    def ComputeQuads(self):
        '''
        Updated attributes:
            jointsQuads : list of list of 4 elements containing the quads ordered in a clockwise fashion
        '''

        self.jointsQuads = []
        seq   = [("A", 1), ("B", 1), ("A", -1), ("B", -1)]

        for j in range(self.nJ):
            currJoint  = j
            jointsQuad = []
            addQuad    = True
            for (fam, inc) in seq:
                # Test if we can find a curve of the correct family passing by the current joint
                if fam in self.jointsToCurves[currJoint]:
                    crvId = self.jointsToCurves[currJoint][fam]
                    crv   = self.curves[crvId]
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
                self.jointsQuads.append(jointsQuad)

    def ComputeQuadsOrientation(self):
        '''
        Updated attributes:
            quadsOrientation : torch tensor of shape (nQuads,) containing the sign 
        '''
        joints = self.fullCurvesDoF[:2*self.nJ].reshape(-1, 2)
        assert len(self.jointsQuads[0]) == 4
        quadsExt = torch.zeros(size=(len(self.jointsQuads), 5, 2))
        quadsExt[:, :-1, :] = joints[self.jointsQuads, :]
        quadsExt[:, -1, :]  = quadsExt[:, 0, :] 

        det = torch.zeros(size=(quadsExt.shape[0],))
        for i in range(quadsExt.shape[1]-1):
            det += quadsExt[:, i, 0] * quadsExt[:, i+1, 1] - quadsExt[:, i, 1] * quadsExt[:, i+1, 0]

        self.quadsOrientation = torch.sign(det)

    def InsertControlPointsInCurves(self):
        '''
        Updated attributes:
            curvesWithCP      : list of list giving the joints through which each curve passes with the new CPs
            segmentsWithCP    : list of nEdges lists containing the segments with the new CPs
            edgesWithCP       : torch tensor of shape (nEdges, 2) containing all the edges
            edgeWithCPToCurve : list of nCurves 
        '''

        nAdd = sum([sum(addCP) for addCP in self.nCPperRodEdge])

        if nAdd == 0:
            edgesWithCP, edgeWithCPToCurve = GetEdgesFromCurves(self.curves)
            edgeWithCPFamily = [self.curvesFamily[curveID] for curveID in edgeWithCPToCurve]
            self.curvesWithCP      = self.curves
            self.segmentsWithCP    = edgesWithCP
            self.edgesWithCP       = edgesWithCP
            self.edgeWithCPToCurve = edgeWithCPToCurve
            self.edgeWithCPFamily  = edgeWithCPFamily
            return 0

        curvesWithCP   = []
        segmentsWithCP = []

        idTmp = 0
        # Collect the number of joints
        nCP   = torch.unique(torch.tensor([idx for crv in self.curves for idx in crv])).shape[0]
        for idCrv, (nCPforRodEdge, crv) in enumerate(zip(self.nCPperRodEdge.copy(), self.curves.copy())):
            addCP = torch.tensor(nCPforRodEdge)
            crvWithCP = crv.copy()
            for i in range(len(crv)-1):
                addedElts = torch.sum(addCP[:i]).item()
                segmentsWithCP.append([crvWithCP[i+addedElts]] + list(range(nCP, nCP+addCP[i])) + [crvWithCP[i+1+addedElts]])
                crvWithCP = crvWithCP[:i+1+addedElts] + list(range(nCP, nCP+addCP[i])) + crvWithCP[i+1+addedElts:]
                nCP += nCPforRodEdge[i]
            curvesWithCP.append(crvWithCP)
            idTmp += len(crv) - 1

        edgesWithCP, edgeWithCPToCurve = GetEdgesFromCurves(curvesWithCP)
        edgeWithCPFamily = [self.curvesFamily[curveID] for curveID in edgeWithCPToCurve]

        self.curvesWithCP      = curvesWithCP
        self.segmentsWithCP    = segmentsWithCP
        self.edgesWithCP       = edgesWithCP
        self.edgeWithCPToCurve = edgeWithCPToCurve
        self.edgeWithCPFamily  = edgeWithCPFamily

        return 1

    def ResetFixedCPPosition(self): self.controlPointsFixed = self.controlPoints.detach().clone()

    def MakeSegmentSplines(self):
        '''
        Updated attributes:
            fullRodSplines : list containing one spline for each curve (scipy version)
            segmentSplines : list containing one constant speed spline per rod segment
        '''
        
        ClampAlpha = lambda alpha: np.clip(alpha, 0.0, 1.0)

        def CurveFunCreator(spline, tStart, tEnd):
            def CurveFun(alpha, correctOrientation):
                if correctOrientation:
                    return spline(tStart + (tEnd - tStart) * ClampAlpha(alpha))
                else:
                    return spline(tStart + (tEnd - tStart) * ClampAlpha(1.-alpha))
                
            return CurveFun

        self.fullRodSplines = []
        self.segmentSplines = []

        for crv, addCP in zip(self.curvesWithCP, self.nCPperRodEdge):
            # First make constant speed
            cpTmp = self.controlPoints[crv].reshape(1, -1, 3)
            jointsIdx = [0] + torch.cumsum(torch.tensor(addCP)+1, dim=0).tolist()
            if torch.isnan(cpTmp).any():
                print("NaN in make splines: {}".format(torch.isnan(cpTmp).any()))
            _, newSKnots, newTs, newKnotsAtTs = MakeConstantSpeed(cpTmp, self.mult)
            newSJoints = ToNumpy(newSKnots[0, jointsIdx])
            self.fullRodSplines.append(CubicSpline(ToNumpy(newTs[0]), ToNumpy(newKnotsAtTs[0]), axis=0, bc_type='natural'))
            self.segmentSplines += [CurveFunCreator(self.fullRodSplines[-1], newSJoints[i], newSJoints[i+1]) for i in range(newSJoints.shape[0]-1)]
    
    def MakeLinkages(self, linkagesGuess=None, useCached=False, flatOnly=False):
        '''
        Args:
            linkagesGuess   : linkages to use
            useCached       : whether we want to reuse the cached degrees of freedom to potentially accelerate the deployed equilibrium solve

        Updated attributes:
            flatLinkage     : an elastic rod object
            flatView        : the linkage viewer associated to flatLinkage
            deployedLinkage : the deployed version of flatLinkage
            deployedView    : the linkage viewer associated to deployedLinkage
        '''
        if not linkagesGuess is None:
            try:
                self.flatLinkage     = linkagesGuess["flat"]
                if self.createViewers: self.flatView        = LinkageViewer(self.flatLinkage, width=768, height=480)
                self.deployedLinkage = linkagesGuess["deployed"]
                if self.createViewers: self.deployedView    = LinkageViewer(self.deployedLinkage, width=768, height=480)

                self.useSAL = isinstance(self.deployedLinkage, average_angle_linkages.AverageAngleSurfaceAttractedLinkage)

                driver = self.flatLinkage.centralJoint()
                self.flatLinkage.setMaterial(self.rodMaterial)

                jdo = self.flatLinkage.dofOffsetForJoint(driver)
                fixedVars = list(range(jdo, jdo + 6)) # fix rigid motion for a single joint
                idxAverageAngleDep = self.deployedLinkage.getAverageAngleIndex()
                fixedDepVars = []
                if not self.useSAL:
                    fixedDepVars = list(range(jdo, jdo + 6))
                else:
                    if self.attractionMesh is None:
                        self.attractionMesh = {
                            "V"       : self.deployedLinkage.getTargetSurfaceVertices(),
                            "F"       : self.deployedLinkage.getTargetSurfaceFaces(),
                            "targetJP": self.deployedLinkage.getTargetJointsPosition(),
                        }
                    if isinstance(self.flatLinkage, average_angle_linkages.AverageAngleLinkage):
                        self.flatLinkage = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(self.attractionMesh["V"], self.attractionMesh["F"], False, self.flatLinkage, free_joint_angles=self.freeAngles)
                        self.flatLinkage.attraction_weight = 0.
                fixedDepVars.append(idxAverageAngleDep)

                angleDeviation = abs(self.alphaTar.item() - self.deployedLinkage.getDoFs()[idxAverageAngleDep])
                if angleDeviation > 1.0e-10:
                    dof = self.deployedLinkage.getDoFs()
                    dof[idxAverageAngleDep] = self.alphaTar.item()
                    print("Target angle: ", self.alphaTar.item())
                    self.deployedLinkage.setDoFs(dof)
                    # raise ValueError("ERROR: the deployed average angle has a deviation of {:.2e} from the one specified to the Cshell constructor!".format(angleDeviation))
                
                maxFlat = np.max(abs(self.flatLinkage.getDesignParameters() - ToNumpy(self.designParameters)))
                if maxFlat > 1.0e-10:
                    self.oldDP = self.flatLinkage.getDesignParameters()
                    print(self.flatLinkage.getDesignParameters() - ToNumpy(self.designParameters))
                    print("WARNING: the design parameters differ for the flat linkage by a deviation of {:.2e}!".format(maxFlat))
                    self.flatLinkage.setDesignParameters(ToNumpy(self.designParameters))
                    # raise ValueError("WARNING: the design parameters differ for the flat linkage by a deviation of {:.2e}!".format(maxFlat))

                maxDep = np.max(abs(self.deployedLinkage.getDesignParameters() - ToNumpy(self.designParameters)))
                if maxDep > 1.0e-10:
                    self.deployedLinkage.setDesignParameters(ToNumpy(self.designParameters))
                    print("WARNING: the design parameters differ for the deployed linkage by a deviation of {:.2e}!".format(maxDep))
                    # raise ValueError("WARNING: the design parameters differ for the deployed linkage by a deviation of {:.2e}!".format(maxDep))

                # Compute the deployed equilibrium
                self.deployedLinkage.setMaterial(self.rodMaterial)
                self.deployedLinkage.attraction_weight = self.attractionWeight
                self.deployedLinkage.scaleJointWeights(jointPosWeight=self.attractionJointPosWeight)
                self.deployedLinkage.set_holdClosestPointsFixed(False)
                self.deployedLinkage.setTargetSurface(self.attractionMesh["V"], self.attractionMesh["F"])
                self.deployedLinkage.setTargetJointsPosition(self.attractionMesh["targetJP"].reshape(-1,))

                # elastic_rods.compute_equilibrium(self.flatLinkage, fixedVars=fixedVars)
                average_angle_linkages.compute_equilibrium(self.flatLinkage, elastic_rods.TARGET_ANGLE_NONE, options=self.newtonOptimizerOptions, fixedVars=fixedVars)
                average_angle_linkages.compute_equilibrium(self.deployedLinkage, elastic_rods.TARGET_ANGLE_NONE, options=self.newtonOptimizerOptions, fixedVars=fixedDepVars)
                return 1

            except Exception as e:
                print("Couldn't load the linkages: {}\nFalling back to classic initialization.".format(e))

        jointsPosition           = ToNumpy(self.joints)
        inputJointNormals        = np.ones_like(jointsPosition)
        inputJointNormals[:, :2] = 0.0

        # Create the flat linkage and open it
        flatRodLinkage = elastic_rods.RodLinkage(jointsPosition, self.rodEdges, 
                                                 edge_callbacks=self.segmentSplines, input_joint_normals=inputJointNormals,
                                                 rod_interleaving_type=elastic_rods.InterleavingType.xshell, subdivision=self.subdivision)
        self.flatLinkage = average_angle_linkages.AverageAngleLinkage(flatRodLinkage, free_joint_angles=self.freeAngles)
        self.flatLinkage.setDesignParameters(ToNumpy(self.designParameters))
        driver = self.flatLinkage.centralJoint()
        self.flatLinkage.setMaterial(self.rodMaterial)

        jdo = self.flatLinkage.dofOffsetForJoint(driver)
        fixedVars = list(range(jdo, jdo + 6)) # fix rigid motion for a single joint
        with so(): average_angle_linkages.compute_equilibrium(self.flatLinkage, options=self.newtonOptimizerOptions, fixedVars=fixedVars)

        if self.createViewers: self.flatView = LinkageViewer(self.flatLinkage, width=768, height=480)

        if (not flatOnly):
            # Create the deployed linkage and open it
            if not self.useSAL:
                self.deployedLinkage = average_angle_linkages.AverageAngleLinkage(self.flatLinkage, free_joint_angles=self.freeAngles)
            else:
                # Prioritize self.attractionMesh over self.pathSurf
                # The second argument is useCenterline
                if not self.attractionMesh is None:
                    self.flatLinkage     = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(self.attractionMesh["V"], self.attractionMesh["F"], False, self.flatLinkage, free_joint_angles=self.freeAngles)
                    self.deployedLinkage = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(self.attractionMesh["V"], self.attractionMesh["F"], False, self.flatLinkage, free_joint_angles=self.freeAngles)
                elif not self.pathSurf is None:
                    self.flatLinkage     = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(self.pathSurf, False, self.flatLinkage, free_joint_angles=self.freeAngles)
                    self.deployedLinkage = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(self.pathSurf, False, self.flatLinkage, free_joint_angles=self.freeAngles)
                else:
                    raise ValueError("The variable self.useSAL should be set to False if both self.attractionMesh and self.pathSurf are None.")
                self.flatLinkage.attraction_weight     = 0.0
                self.deployedLinkage.attraction_weight = self.attractionWeight # 1.0e-5, 1.0e3
                self.deployedLinkage.set_holdClosestPointsFixed(False)
                if "targetJP" in self.attractionMesh.keys():
                    assert self.attractionMesh["targetJP"].shape[0] == 3 * self.nJ
                    self.deployedLinkage.scaleJointWeights(jointPosWeight=self.attractionJointPosWeight, valence2Multiplier=1.0)
                    self.deployedLinkage.setTargetJointsPosition(self.attractionMesh["targetJP"])
                else:
                    self.deployedLinkage.scaleJointWeights(jointPosWeight=0.0)
                self.ComputeDeployedLinkageEquilibrium()
            if useCached:
                try:
                    self.deployedLinkage.setDoFs(self.cachedQuantities["deployedDoF"])
                except Exception as e:
                    print("Could not use cached quantities when creating the deployed linkage: {}".format(e))
        
            def equilibriumSolver(tgtAngle, l, opts, fv):
                opts.gradTol = 1.0e-5
                return average_angle_linkages.compute_equilibrium(l, tgtAngle, options=opts, fixedVars=fv)

            with so(): open_average_angle_linkage(self.deployedLinkage, driver, self.alphaTar.item() - self.deployedLinkage.averageJointAngle, 
                                                  self.numOpeningSteps, self.deployedView, equilibriumSolver=equilibriumSolver, 
                                                  maxNewtonIterationsIntermediate=self.maxNewtonIterIntermediate)
            if self.createViewers: self.deployedView = LinkageViewer(self.deployedLinkage, width=768, height=480)


    def MakeLinkageOptimizer(self, updateSurf=True, createDeployedViewer=True):
        '''
        Args:
            updateSurf : whether we want to update the target surface as well.
            createDeployedViewer : whether we update the deployed viewer 

        Updated attributes:
            linkageOptimizer : a CShellOptimization object
        '''

        if self.useSAL:
            self.linkageOptimizer = cshell_optimization.AverageAngleCShellOptimizationSAL(self.flatLinkage, self.deployedLinkage, self.newtonOptimizerOptions, 
                                                                                          minAngleConstraint=0., allowFlatActuation=False,
                                                                                          optimizeTargetAngle=self.optimizeAlpha, fixDeployedVars=not self.useSAL)
        else:
            self.linkageOptimizer = cshell_optimization.AverageAngleCShellOptimization(self.flatLinkage, self.deployedLinkage, self.newtonOptimizerOptions,
                                                                                       minAngleConstraint=0., allowFlatActuation=False,
                                                                                       optimizeTargetAngle=self.optimizeAlpha, fixDeployedVars=not self.useSAL)

        self.linkageOptimizer.setTargetAngle(self.alphaTar)

        if updateSurf:
            if not self.targetMesh is None:
                self.linkageOptimizer.setTargetSurface(self.targetMesh["V"], self.targetMesh["F"])
                if "targetJP" in self.targetMesh.keys():
                    self.linkageOptimizer.scaleJointWeights(jointPosWeight=0.1)
                    self.linkageOptimizer.setTargetJointsPosition(self.targetMesh["targetJP"])
                else:
                    self.linkageOptimizer.scaleJointWeights(jointPosWeight=0.)
            elif self.pathSurf is None:
                self.pathSurf = "inferred_surface.msh"               
                self.linkageOptimizer.saveTargetSurface(self.pathSurf)
                self.targetMesh = {
                    "V"       : self.linkageOptimizer.getTargetSurfaceVertices(),
                    "F"       : self.linkageOptimizer.getTargetSurfaceFaces(),
                    "targetJP": self.linkageOptimizer.getTargetJointsPosition(),
                }
        if not isinstance(self.deployedView, LinkageViewerWithSurface):
            if self.pathSurf is None:
                self.pathSurf = "inferred_surface.msh"               
            self.linkageOptimizer.saveTargetSurface(self.pathSurf)
            if createDeployedViewer: self.deployedView = LinkageViewerWithSurface(self.deployedLinkage, self.pathSurf, wireframeSurf=False, transparent=True, width=768, height=480)
        self.UpdateDeployedViewer()
        self.ApplyWeights()

    def ComputeDeployedLinkageEquilibrium(self):
        averageAngleIndex = self.deployedLinkage.getAverageAngleIndex()
        driver            = self.deployedLinkage.centralJoint()
        jdo               = self.deployedLinkage.dofOffsetForJoint(driver)
        if self.useSAL: fixedVars = [averageAngleIndex]
        else          : fixedVars = list(range(jdo, jdo + 6)) + [averageAngleIndex]
        with so(): 
            report = average_angle_linkages.compute_equilibrium(self.deployedLinkage, elastic_rods.TARGET_ANGLE_NONE, options=self.newtonOptimizerOptions, fixedVars=fixedVars)
        print("Deployed equilibrium success: {} (gradient magnitude: {:.3e}).".format(report.success, report.freeGradientNorm[-1]))

    def UpdateDeployedViewer(self, showAttraction=False):
        if self.deployedView:
            if self.targetMesh is None:
                if self.pathSurf is None:
                    self.pathSurf = "inferred_surface.msh"
                self.targetMesh = {
                    "V": self.linkageOptimizer.getTargetSurfaceVertices(),
                    "F": self.linkageOptimizer.getTargetSurfaceFaces()
                }

            if showAttraction and self.useSAL: self.deployedView.updateTargetSurf(self.attractionMesh["V"], self.attractionMesh["F"], wireframeSurf=False)
            else                             : self.deployedView.updateTargetSurf(self.targetMesh["V"], self.targetMesh["F"], wireframeSurf=False)
            self.deployedView.update(preserveExisting=False)
        else:
            print("First create the deployed viewer before calling UpdateDeployedViewer")

    def UpdateTargetSurfaceFromPath(self, path):
        self.pathSurf = path
        self.linkageOptimizer.loadTargetSurface(path)
        self.targetMesh = {
            "V": self.linkageOptimizer.getTargetSurfaceVertices(),
            "F": self.linkageOptimizer.getTargetSurfaceFaces()
        }
        pass

    def UpdateTargetSurface(self, vertices, faces, targetJoints=None):
        self.linkageOptimizer.setTargetSurface(vertices, faces)
        self.targetMesh = {
            "V": vertices,
            "F": faces
        }
        if not targetJoints is None:
            self.targetMesh["targetJP"] = targetJoints
            self.linkageOptimizer.scaleJointWeights(jointPosWeight=0.5)
            self.linkageOptimizer.setTargetJointsPosition(self.targetMesh["targetJP"])
        self.UpdateDeployedViewer()
        pass

    def UpdateAttractionSurface(self, vertices, faces, targetJoints=None, showAttraction=True):
        '''
        This method updates the attraction surface for the deployed linkage and re-computes the equilibrium.
        '''
        # isSAL = isinstance(self.deployedLinkage, elastic_rods.SurfaceAttractedLinkage)
        isSAL = isinstance(self.deployedLinkage, average_angle_linkages.AverageAngleSurfaceAttractedLinkage)
        if isSAL!=self.useSAL: 
            self.useSAL = isSAL
            if isSAL:
                print("The deployed linkage was not an SAL and was declared as one, updated self.useSAL accordingly.")
            else:
                print("The deployed linkage was an SAL and was not declared as one, updated self.useSAL accordingly.")
        
        self.attractionMesh = {
            "V": vertices,
            "F": faces 
        }

        if self.useSAL:
            self.deployedLinkage.setTargetSurface(vertices, faces)
        else:
            self.useSAL = True
            # self.deployedLinkage = elastic_rods.SurfaceAttractedLinkage(vertices, faces, False, self.deployedLinkage)
            self.deployedLinkage = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(vertices, faces, False, self.deployedLinkage)
            self.deployedLinkage.attraction_weight = self.attractionWeight
            self.deployedLinkage.set_holdClosestPointsFixed(False)
            self.deployedLinkage.scaleJointWeights(jointPosWeight=0.)
            self.deployedView.update(mesh=self.deployedLinkage)

        if not targetJoints is None:
            self.attractionMesh["targetJP"] = targetJoints
            self.deployedLinkage.scaleJointWeights(jointPosWeight=self.attractionJointPosWeight)
            self.deployedLinkage.setTargetJointsPosition(self.attractionMesh["targetJP"])
        
        self.ComputeDeployedLinkageEquilibrium()
        self.UpdateDeployedViewer(showAttraction=showAttraction)
        self.MakeLinkageOptimizer(updateSurf=True)
        pass
    
    def UpdateAttractionWeights(self, attractionWeight, attractionJointPosWeight, showAttraction=True):
        '''
        This method updates the attraction weights and updates the deployed state accordingly
        '''

        if not self.useSAL: 
            print("The linkages are not surface attracted, updating attraction weight won't affect the deployed state.")
            return
        
        self.attractionWeight = attractionWeight
        self.attractionJointPosWeight = attractionJointPosWeight

        self.deployedLinkage.attraction_weight = self.attractionWeight
        self.deployedLinkage.set_holdClosestPointsFixed(False)
        self.deployedLinkage.scaleJointWeights(jointPosWeight=self.attractionJointPosWeight)
        
        self.ComputeDeployedLinkageEquilibrium()
        self.UpdateDeployedViewer(showAttraction=showAttraction)
        self.MakeLinkageOptimizer(updateSurf=True)
        pass

    def UpdateLinkageOptimizer(self, commitLinesearchLinkage=False):
        '''
        Args:
            commitLinesearchLinkage : whether we simply update the linesearch linkage or both base and linesearch

        Updated attributes:
            linkageOptimizer : a CShellOptimization object
        '''
        
        if self.linkageOptimizer:
            newParams = ToNumpy(self.GetFullDP())
            if commitLinesearchLinkage:
                self.linkageOptimizer.newPt(newParams)
                if not self.optimizeAlpha:
                    try:
                        self.linkageOptimizer.setTargetAngle(self.alphaTar.item())
                    except Exception as e:
                        print("Did you make sure that self.alphaTar was a tensor with 1 component? Error: {}".format(e))
                        self.linkageOptimizer.setTargetAngle(self.alphaTar)
            else:
                # Evaluating the objective simply updates the linesearch linkages
                _ = self.linkageOptimizer.J(newParams)
        pass

    def GetNumDoF(self): return self.curvesDoF.shape[0] + self.optimizeAlpha

    def GetFullDoF(self):
        fullDoF = torch.zeros(size=(self.GetNumDoF(),))
        fullDoF[:self.curvesDoF.shape[0]] = self.curvesDoF.clone()
        if self.optimizeAlpha:
            fullDoF[-1] = self.alphaTar
        return fullDoF

    def GetNumDP(self): return self.designParameters.shape[0] + self.optimizeAlpha

    def GetFullDP(self):
        fullDP = torch.zeros(size=(self.GetNumDP(),))
        fullDP[:self.designParameters.shape[0]] = self.designParameters.clone()
        if self.optimizeAlpha:
            fullDP[-1] = self.alphaTar
        return fullDP

    def GetCShellParams(self):
        '''
        Returns a dictionnary that contains all the necessary information to re-create the C-Shell
        '''
        dictCShell = {
            "curvesDoF"       : self.curvesDoF,
            "fullCurvesDoF"   : self.fullCurvesDoF,
            "nJ"              : self.nJ,
            "curves"          : self.curves,
            "curvesFamily"    : self.curvesFamily,
            "nCPperRodEdge"   : self.nCPperRodEdge,
            "alphaTar"        : self.alphaTar,
            "mult"            : self.mult,
            "subdivision"     : self.subdivision,
            "deployedLinkage" : self.deployedLinkage,
            "flatLinkage"     : self.flatLinkage,
            "attractionMesh"  : self.attractionMesh,
            "targetMesh"      : self.targetMesh,
            "symmetry"        : self.symmetry,
            "initE0"          : self.linkageOptimizer.get_E0()
        }

        return dictCShell

    #############################################
    #############        MAPS       #############
    #############################################

    def FullCurvesDoFToControlPointsMap(self):
        '''
        Updated attributes:
            controlPoints : torch tensor of size (nCP, 3)
            joints        : torch tensor of size (nJ, 3)
        '''

        # First map to full DoF if needed
        if not self.symmetry is None:
            self.fullCurvesDoF = self.symmetry.MapToFullCurvesDoF(self.curvesDoF)
        else:
            self.fullCurvesDoF = self.curvesDoF

        nAdd = sum([sum(addCP) for addCP in self.nCPperRodEdge])
        if nAdd == 0:
            self.controlPoints = torch.zeros(size=(self.nJ, 3))
            self.controlPoints[:, :2] = self.fullCurvesDoF[:2*self.nJ].reshape(-1, 2)

        joints        = torch.zeros(size=(self.nJ, 3))
        joints[:, :2] = self.fullCurvesDoF[:2*self.nJ].reshape(-1, 2)
        self.joints   = joints.detach().clone()
        orthComp      = self.fullCurvesDoF[2*self.nJ:]

        startJoints = joints[self.rodEdges[:, 0]]
        endJoints   = joints[self.rodEdges[:, 1]]

        # Compute tangents
        allTangents = endJoints - startJoints
        idxTangents = []
        flatNCPperRodEdge = [n for addCP in self.nCPperRodEdge for n in addCP]
        nRepeats = [n for n in flatNCPperRodEdge if n!=0]
        for i, n in enumerate(flatNCPperRodEdge):
            idxTangents += n * [i]
        tangentDirs = allTangents[idxTangents] / torch.linalg.norm(allTangents[idxTangents], dim=-1, keepdims=True)

        # Compute orthogonal directions
        rot      = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        orthDirs = tangentDirs @ rot.T
        alphas = torch.cat([torch.linspace(0., 1., n+2)[1:-1] for n in nRepeats]).reshape(-1, 1)

        newCP = (1. - alphas) * startJoints[idxTangents] + alphas * endJoints[idxTangents] + orthComp.reshape(-1, 1) * orthDirs

        self.controlPoints = torch.cat((joints, newCP), dim=0)

    def ControlPointsToDiscretePositionsMap(self, addFreeEnds=True):
        '''        
        Updated attributes:
            splines        : list of nCurves cubic splines
            discList       : list of tensors of shape (subdivision*(nJ-1) + 2, 3) containing the centerline position along rods
            segmentsLength : tensor that collects all the segments length
        '''

        splinesList = []
        discList    = []
        discSList   = []

        idxSeg = 0
        for crv, addCP in zip(self.curvesWithCP, self.nCPperRodEdge):
            # First make constant speed
            controlPointsCurve = self.controlPoints[crv].reshape(1, -1, 3)
            # jointsIdx = [0] + torch.cumsum(torch.tensor(addCP)+1, dim=0).tolist()
            jointsIdx = [0]
            for nCP in addCP:
                jointsIdx.append(jointsIdx[-1] + nCP + 1)
            if torch.isnan(controlPointsCurve).any():
                print(self.fullCurvesDoF)
                print(controlPointsCurve)
                print("NaN in cp to dp map: {}".format(torch.isnan(controlPointsCurve).any()))
            spline, discS = ComputeElasticRodsDiscretization(controlPointsCurve, jointsIdx, self.subdivision, mult=self.mult, addFreeEnds=addFreeEnds)
            
            oldDisc = spline.evaluate(discS.reshape(1, -1)).reshape(-1, 3)
            newDisc = StraightenInnerJointEdges(oldDisc, discS, spline, self.subdivision)

            splinesList.append(spline)
            discList.append(newDisc)
            discSList.append(discS)
            # self.segmentsLength[idxSeg:idxSeg + segLength.shape[0]] = segLength
            # idxSeg += segLength.shape[0]

        self.discList  = discList
        self.discSList = discSList
        self.splines   = splinesList

    def DiscretePositionsToDesignParametersMap(self):
        '''
        Output:
            restLengths       : tensor containing the rest lengths of each edge, starting from the free edges to the joint edges
            restKappas        : tensor containing the rest curvatures at each vertex
            restQuantities    : tensor containing the rest quantities of the discretized linkage. First the rest length for the free edges
                                then the rest lengths at the joints, and lastly the rest kappas.
            designParameters  : tensor containing the rest quantities of the discretized linkage. First the rest kappas,
                                then the rest lengths for each segment
        '''

        # Rest lengths first
        restLengthsFreeList   = []
        restLengthsJointsAB   = torch.zeros(size=(self.nJ, 2))
        maskRestLengthsJoints = torch.zeros_like(restLengthsJointsAB)
        self.segmentsLength   = torch.zeros(size=(len(self.segmentsWithCP),))

        idxSeg = 0
        for crv, fam, disc in zip(self.curves, self.curvesFamily, self.discList):
            nJoints = len(crv)
            restLengthsFree, restLengthsJoint = ExtractRestLengths(disc, nJoints, self.subdivision)
            restLengthsFreeList  += [restLengthsFree]
            restLengthsJointsAB[crv, fam] = restLengthsJoint
            maskRestLengthsJoints[crv, fam] = 1
            # Comment this line out and uncomment the paragraph below if you want to estimate
            # the actual length of the curves
            # self.subdivision-2 is the number of free edges per rod segment
            # self.segmentsLength[idxSeg:idxSeg+len(crv)-1] = (self.subdivision - 1) * restLengthsFree[::self.subdivision-2]
            idxSeg += len(crv) - 1 # gives the number of segments along the curve

        restLengthsJoint = torch.masked_select(restLengthsJointsAB, maskRestLengthsJoints.to(torch.bool))
        self.restLengths = torch.cat(restLengthsFreeList + [restLengthsJoint], dim=0)

        # Rest kappas then
        restKappasList = []

        for crv, disc in zip(self.curves, self.discList):
            nJoints = len(crv)
            restKappasCurve  = ExtractRestKappasVars(disc, self.subdivision)
            restKappasList  += [restKappasCurve]
            
        self.restKappas = torch.cat(restKappasList, dim=0)

        self.restQuantities = torch.cat([self.restLengths, self.restKappas], dim=0)

        # Rest segments length, might be wrong since it should account for overlapping edges??
        # We might simply be able to use the rest lengths from self.restLengths
        
        nDisc = 20
        idxSeg = 0
        for crv, addCP in zip(self.curvesWithCP, self.nCPperRodEdge):
            knots      = self.controlPoints[crv]
            jointsIdx = [0]
            for nCP in addCP:
                jointsIdx.append(jointsIdx[-1] + nCP + 1)
            initTs     = torch.linspace(0., 1., knots.shape[0])
            initCoeffs = natural_cubic_spline_coeffs(initTs, knots)
            splines    = NaturalCubicSpline(initCoeffs)
            
            listTs     = [torch.linspace(initTs[jointsIdx[i]], initTs[jointsIdx[i+1]], nDisc) for i in range(len(jointsIdx)-1)]
            refinedTs  = torch.vstack(listTs)
            points     = splines.evaluate(refinedTs)
            lengths    = torch.sum(torch.linalg.norm(points[:, 1:, :] - points[:, :-1, :], dim=2), dim=1)

            self.segmentsLength[idxSeg:idxSeg+lengths.shape[0]] = lengths
            idxSeg += lengths.shape[0]

        self.designParameters = torch.cat([self.restKappas, self.segmentsLength], dim=0)


    #############################################
    ####   PUSHING/PULLING TANGENT VECTORS   ####
    #############################################

    def PullDesignParametersToCurvesDoF(self, dDesignParameters):
        '''
        Args:
            dDesignParameters : torch tensor containing the differential of the design parameters of the discretized linkage

        Returns:
            gradCurveDoFs : torch tensor containing the pullback of dDesignParameters
        '''

        gradCurvesDof = torch.zeros(size=(self.GetNumDoF(),))

        self.CleanGradients()

        # Go through operations
        self.curvesDoF.requires_grad = True
        self.FullCurvesDoFToControlPointsMap()
        self.ControlPointsToDiscretePositionsMap()
        self.DiscretePositionsToDesignParametersMap()
        self.designParameters.backward(dDesignParameters[:self.designParameters.shape[0]])

        gradCurvesDof[:self.curvesDoF.shape[0]] = self.curvesDoF.grad
        self.CleanGradients()

        if self.optimizeAlpha:
            gradCurvesDof[-1] = dDesignParameters[-1]

        return gradCurvesDof
    
    def VJPDesignParametersToCurvesDoF(self, jacDesignParameters):
        '''
        Args:
            jacDesignParameters : torch tensor containing the jacobian with respect to the design parameters of the discretized linkage (?, nDP)

        Returns:
            jacCurveDoFs : torch tensor containing the jacobian with respect to the curves DoFs (?, nCurvesDoF)
        '''

        jacCurveDoFs = torch.zeros(size=(jacDesignParameters.shape[0], self.GetNumDoF()))

        self.CleanGradients()
        currCurvesDoF = self.curvesDoF
        
        def MapCurvesDoFToDP(curvesDoF):
            self.curvesDoF = curvesDoF
            # Go through operations
            self.FullCurvesDoFToControlPointsMap()
            self.ControlPointsToDiscretePositionsMap()
            self.DiscretePositionsToDesignParametersMap()
            return self.designParameters
        
        vjpFun = torch.vmap(torch.func.vjp(MapCurvesDoFToDP, currCurvesDoF)[1], in_dims=(0,))
        jacCurveDoFs[:, :self.curvesDoF.shape[0]] = vjpFun(jacDesignParameters[:, :self.designParameters.shape[0]])[0]
        self.CleanGradients()

        if self.optimizeAlpha:
            jacCurveDoFs[:, -1] = jacDesignParameters[:, -1]

        return jacCurveDoFs

    def PushCurvesDoFToDesignParameters(self, dCurvesDoF):
        '''
        Args:
            dCurvesDoF : torch tensor of size (2*nJ + nIntCP + self.optimizeAlpha,) containing the modifications to be pushed
        
        Returns:
            dDesignParameters : torch tensor containing the pushforward of dCurveDoFs
        '''

        dDesignParameters = torch.zeros(size=(self.designParameters.shape[0] + self.optimizeAlpha,))

        def pipeline(dof):
            '''
            This also changes the whole Cshell object!
            '''
            self.curvesDoF = dof
            self.FullCurvesDoFToControlPointsMap()
            self.ControlPointsToDiscretePositionsMap()
            self.DiscretePositionsToDesignParametersMap()
            return self.designParameters

        _, dDesignParametersPartial = jvp(pipeline, self.curvesDoF, v=dCurvesDoF[:self.curvesDoF.shape[0]])

        dDesignParameters[:self.designParameters.shape[0]] = dDesignParametersPartial
        if self.optimizeAlpha:
            dDesignParameters[-1] = dCurvesDoF[-1]

        return dDesignParameters

    def PullControlPointsToCurvesDoF(self, dControlPoints):
        '''
        Args:
            dControlPoints : torch tensor of shape containing the differential of the control points
        
        Returns:
            gradCurveDoFs : torch tensor containing the pullback of dControlPoints
        '''

        gradCurvesDof = torch.zeros(size=(self.GetNumDoF(),))

        self.CleanGradients()

        # Go through operations
        self.curvesDoF.requires_grad = True
        self.FullCurvesDoFToControlPointsMap()
        self.controlPoints.backward(dControlPoints)

        gradCurvesDof[:self.curvesDoF.shape[0]] = self.curvesDoF.grad
        self.CleanGradients()

        return gradCurvesDof

    def PushCurvesDoFToControlPoints(self, dCurvesDoF):
        '''
        Args:
            dCurvesDoF : torch tensor of size (2*nJ + nIntCP + self.optimizeAlpha,) containing the modifications to be pushed
        
        Returns:
            dControlPoints : torch tensor of size (nCP, 3), one part of the pushforward of dCurveDoFs
        '''

        def pipeline(dof):
            '''
            This also changes the whole Cshell object!
            '''
            self.curvesDoF = dof
            self.FullCurvesDoFToControlPointsMap()
            return self.controlPoints

        _, dControlPoints = jvp(pipeline, self.curvesDoF, v=dCurvesDoF[:self.curvesDoF.shape[0]])

        return dControlPoints

    def PullRestQuantitiesToControlPoints(self, dDesignParameters):
        '''
        Args:
            dDesignParameters : torch tensor containing the differential of the design parameters of the discretized linkage
        
        Returns:
            gradControlPoints : torch tensor containing the pullback of dDesignParameters
        '''

        self.CleanGradients()
        
        # Go through operations
        self.controlPoints.requires_grad = True
        self.ControlPointsToDiscretePositionsMap()
        self.DiscretePositionsToDesignParametersMap()
        self.designParameters.backward(dDesignParameters[:self.designParameters.shape[0]])

        gradControlPoints = self.controlPoints.grad
        self.CleanGradients()

        return gradControlPoints

    def PushControlPointsToDesignParameters(self, dControlPoints):
        '''
        Args:
            dControlPoints  : torch tensor of shape containing the differential of the control points
        
        Returns:
            dDesignParameters : torch tensor containing the pushforward of dCurveDoFs
        '''

        dDesignParameters = torch.zeros(size=(self.designParameters.shape[0] + self.optimizeAlpha,))

        def pipeline(cp):
            self.controlPoints = cp
            self.ControlPointsToDiscretePositionsMap()
            self.DiscretePositionsToDesignParametersMap()
            return self.designParameters

        _, dDesignParametersPartial = jvp(pipeline, self.controlPoints, v=dControlPoints)

        dDesignParameters[:self.designParameters.shape[0]] = dDesignParametersPartial

        return dDesignParameters

    def PushCurvesDoFToDeployedDoF(self, dCurvesDoF):
        '''
        Args:
            dCurvesDoF : torch tensor of size (2*nJ + nIntCP,) containing the modifications to be pushed
        
        Returns:
            dDeployedDoF : torch tensor containing the pushforward of dCurveDoFs
        '''
        if not KNITRO_FOUND:
            print("Knitro has not been found.")
            raise ModuleNotFoundError

        dDesignParameters = ToNumpy(self.PushCurvesDoFToDesignParameters(dCurvesDoF))
        dDeployedDoF = self.linkageOptimizer.pushforward(self.linkageOptimizer.getFullDesignParameters(), dDesignParameters)

        return torch.tensor(dDeployedDoF)

    def MakeJacobianCurveDoFToDP(self, vectorize=True):
        '''
        Args:
            vectorize : whether we vectorize the jacobian computation (still experimental when code has beem written)

        Updated attributes:
            jacCurDoFToDP : the jacobian from curve DoF to design parameters
        '''

        jacCurDoFToDP = torch.zeros(size=(self.designParameters.shape[0] + self.optimizeAlpha, self.GetNumDoF()))

        def pipeline(dof):
            '''
            This also changes the whole Cshell object!
            '''
            self.curvesDoF = dof
            self.FullCurvesDoFToControlPointsMap()
            self.ControlPointsToDiscretePositionsMap()
            self.DiscretePositionsToDesignParametersMap()
            return self.designParameters

        jacCurDoFToDP[:self.designParameters.shape[0], :self.curvesDoF.shape[0]] = jacobian(pipeline, self.curvesDoF, vectorize=vectorize)
        jacCurDoFToDP[-1, -1] = 1.

        self.jacCurDoFToDP = [jacCurDoFToDP, True]

    def MakeJacobianDPToDoF(self):
        '''
        Updated attributes:
            jacDPToDoF : the jacobian from design parameters to all DoF (first nDoF rows for 2D)
        '''
        if not KNITRO_FOUND:
            print("Knitro has not been found.")
            raise ModuleNotFoundError

        if self.jacDPToDepDoF[1]: return

        jacDPToDoF = torch.zeros(size=(2 * self.deployedLinkage.numDoF(), self.linkageOptimizer.getFullDesignParameters().shape[0]))
        for i, ei in enumerate(np.eye(self.linkageOptimizer.getFullDesignParameters().shape[0])):
            jacDPToDoF[:, i] = torch.tensor(self.linkageOptimizer.pushforward(self.linkageOptimizer.getFullDesignParameters(), ei))

        self.jacDPToDepDoF = [jacDPToDoF, True]

    def MakeFullJacobian(self):
        '''
        Updated attributes:
            jacFull : the jacobian from design parameters to DoF (first nDoF rows for 2D)
        '''

        if not KNITRO_FOUND:
            print("Knitro has not been found.")
            raise ModuleNotFoundError

        if self.jacFull[1]: return

        if not self.jacCurDoFToDP[1]:
            self.MakeJacobianCurveDoFToDP()
        if not self.jacDPToDepDoF[1]:
            self.MakeJacobianDPToDoF()
        
        self.jacFull = [self.jacDPToDepDoF[0] @ self.jacCurDoFToDP[0], True]

    def MakeFullJacobianToPositions(self):
        '''
        Updated attributes:
            jacFullPos2D : the jacobian from the design parameters to the positional DoF for 2D
            jacFullPos3D : the jacobian from the design parameters to the positional DoF for 3D
        '''

        if not KNITRO_FOUND:
            print("Knitro has not been found.")
            raise ModuleNotFoundError

        if not self.jacFull[1]:
            self.MakeFullJacobian()

        if self.jacFullPos2D[1] and self.jacFullPos3D[1]: return

        idxDiscPosFlat = IndicesDiscretePositionsNoDuplicate(self.deployedLinkage, self.rodEdges, self.subdivision)

        jacFull2D = self.jacFull[0][:self.deployedLinkage.numDoF()]
        self.jacFullPos2D = [jacFull2D[idxDiscPosFlat, :], True]

        jacFull3D = self.jacFull[0][self.deployedLinkage.numDoF():]
        self.jacFullPos3D = [jacFull3D[idxDiscPosFlat, :], True]

    def MakeFullJacobianToJoints(self):
        '''
        Updated attributes:
            jacFullJoints2D : the jacobian from the design parameters to the joints positions for 2D
            jacFullJoints3D : the jacobian from the design parameters to the joints positions for 3D
        '''

        if not KNITRO_FOUND:
            print("Knitro has not been found.")
            raise ModuleNotFoundError

        if not self.jacFull[1]:
            self.MakeFullJacobian()

        if self.jacFullJoints2D[1] and self.jacFullJoints3D[1]: return

        idxJointPos= self.deployedLinkage.jointPositionDoFIndices()

        jacFull2D = self.jacFull[0][:self.deployedLinkage.numDoF()]
        self.jacFullJoints2D = [jacFull2D[idxJointPos, :], True]

        jacFull3D = self.jacFull[0][self.deployedLinkage.numDoF():]
        self.jacFullJoints3D = [jacFull3D[idxJointPos, :], True]

    
    #############################################
    ####   FUNCTIONS FOR THE OPTIMIZATION    ####
    #############################################

    def ComputeObjective(self):
        '''
        Returns:
            objective : return the full objective (a torch tensor)
        '''
        self.CleanGradients()
        objective = torch.tensor(self.linkageOptimizer.J(ToNumpy(self.GetFullDP())))
        if self.cpRegWeight != 0.:
            objective += self.cpRegWeight * LeastSquaresLaplacian(self.controlPoints, self.controlPointsFixed, self.lapCP)

        return objective

    def ComputeGradient(self):
        '''
        Returns:
            gradient : computes the full gradient with respect to the curve DoFs
        '''
        if not KNITRO_FOUND:
            print("Knitro has not been found.")
            raise ModuleNotFoundError

        gradObjDP = torch.tensor(self.linkageOptimizer.gradp_J(ToNumpy(self.GetFullDP())))

        if self.jacFull[1]:
            grad =  gradObjDP @ self.jacFull[0]
        else:
            grad = self.PullDesignParametersToCurvesDoF(gradObjDP)

        if self.cpRegWeight != 0.:
            gradCP = self.cpRegWeight * LeastSquaresLaplacianFullGradient(self.controlPoints, self.controlPointsFixed, self.lapCP)
            grad  += self.PullControlPointsToCurvesDoF(gradCP)

        return grad

    def ComputeHessianVectorProduct(self, dCurvesDoF, coeffJ=1., coeffC=0., coeffAC=0.):
        '''
        Args:
            dCurvesDoF : torch tensor containing the differential of the curves dof, and alpha!
            coeffJ     : coefficient in front of the objective (useful for knitro, see sigma)
            coeffC     : coefficient in front of the flatness constraint (useful for knitro, see corresponding lagrange multiplier lambda_)
            coeffAC    : coefficient in front of the min angle constraint (useful for knitro, see corresponding lagrange multiplier lambda_)

        Returns:
            HVP : torch tensor containing the Hessian Vector Product of the full pipeline (including elastic rods)
        '''
        self.CleanGradients()

        if not KNITRO_FOUND:
            print("Knitro has not been found.")
            raise ModuleNotFoundError

        innerHVP  = lambda dRQ: torch.tensor(self.linkageOptimizer.apply_hess(ToNumpy(self.GetFullDP()), dRQ,
                                                                              coeffJ, coeffC, coeffAC))
        gradObjRQ  = coeffJ  * torch.tensor(self.linkageOptimizer.gradp_J(ToNumpy(self.GetFullDP())))
        gradObjRQ += coeffC  * torch.tensor(self.linkageOptimizer.gradp_c(ToNumpy(self.GetFullDP())))
        gradObjRQ += coeffAC * torch.tensor(self.linkageOptimizer.gradp_angle_constraint(ToNumpy(self.GetFullDP())))

        # with torch.autograd.profiler.profile() as profFirst:
        dDesignParameters = self.PushCurvesDoFToDesignParameters(dCurvesDoF)
        firstTerm = self.PullDesignParametersToCurvesDoF(innerHVP(dDesignParameters))
        
        # with torch.autograd.profiler.profile() as profSecond:
        secondTerm = torch.zeros(size=(self.GetNumDoF(),))
        self.curvesDoF.requires_grad = True
        self.FullCurvesDoFToControlPointsMap()
        self.ControlPointsToDiscretePositionsMap()
        self.DiscretePositionsToDesignParametersMap()
        secondTerm[:self.curvesDoF.shape[0]] = grad(dCurvesDoF[:self.curvesDoF.shape[0]] @ grad(self.designParameters @ gradObjRQ[:self.designParameters.shape[0]],    
                                                                                                self.curvesDoF, create_graph=True)[0], 
                                                    self.curvesDoF)[0]

        HVP = firstTerm + secondTerm

        self.CleanGradients()

        if self.cpRegWeight != 0.0:
            innerHVPCP = lambda dCP: coeffJ * self.cpRegWeight * LeastSquaresLaplacianFullHVP(self.controlPoints, dCP.reshape(-1, 3), self.controlPointsFixed, self.lapCP)
            gradObjCP  = coeffJ * self.cpRegWeight * LeastSquaresLaplacianFullGradient(self.controlPoints, self.controlPointsFixed, self.lapCP)

            dCP         = self.PushCurvesDoFToControlPoints(dCurvesDoF)
            firstTermCP = self.PullControlPointsToCurvesDoF(innerHVPCP(dCP))

            secondTermCP = torch.zeros(size=(self.GetNumDoF(),))
            self.curvesDoF.requires_grad = True
            self.FullCurvesDoFToControlPointsMap()
            secondTermCP[:self.curvesDoF.shape[0]] = grad(dCurvesDoF[:self.curvesDoF.shape[0]] @ grad(torch.sum(self.controlPoints * gradObjCP),
                                                                                                    self.curvesDoF, create_graph=True)[0], 
                                                        self.curvesDoF)[0]

            HVP += firstTermCP + secondTermCP

            self.CleanGradients()

        return HVP

    def ApplyWeights(self):
        self.linkageOptimizer.beta                     = self.dictWeights["beta"]
        self.linkageOptimizer.gamma                    = self.dictWeights["gamma"]
        self.linkageOptimizer.smoothing_weight         = self.dictWeights["smoothingWeight"]
        self.linkageOptimizer.rl_regularization_weight = self.dictWeights["rlRegWeight"]
        self.linkageOptimizer.contact_force_weight     = self.dictWeights["contactForceWeight"]
        self.cpRegWeight                               = self.dictWeights["cpRegWeight"] / (self.linkageOptimizer.get_l0() ** 2)

        self.linkageOptimizer.invalidateAdjointState()

    def SetWeights(self, beta, gamma, smoothingWeight, rlRegWeight, cpRegWeight, contactForceWeight=0):
        self.dictWeights = {
            "beta": beta,
            "gamma": gamma,
            "smoothingWeight": smoothingWeight,
            "rlRegWeight": rlRegWeight,
            "contactForceWeight": contactForceWeight,
            "cpRegWeight" : cpRegWeight
        }
        self.ApplyWeights()
        
    def GetWeightDescription(self):
        descrWeights = "Weights used: "
        for i, key in enumerate(self.dictWeights):
            if i == len(self.dictWeights)-1:
                descrWeights += key + ": " + str(self.dictWeights[key]) + "."
            else:
                descrWeights += key + ": " + str(self.dictWeights[key]) + ", "
        return descrWeights

    def ScaleJointWeights(self, jointPosWeight, featureMultiplier=1., additionalFeaturePts=[]):
        '''
        Args:
            jointPosWeight       : weight assigned to each joint, in case they don't have valence 2 or are not listed in additionalFeaturePts
            featureMultiplier    : multiplicative factor applied to the joints that don't fall in the previous category
            additionalFeaturePts : list of integer giving the joint ID for which a factor featureMultiplier should be applied
        '''
        addFeatJoints = [idx for idx in additionalFeaturePts if self.valence[idx] != 2]
        self.featJoints = addFeatJoints + [idx for idx in range(self.valence.shape[0]) if self.valence[idx] == 2]
        self.linkageOptimizer.scaleJointWeights(jointPosWeight, featureMultiplier=featureMultiplier, additional_feature_pts=addFeatJoints)

    def OptimizeDesignParameters(self, nSteps=100, trustRegionScale=1.0, optTol=1e-2, useCG=True, 
                                 applyAngleConstraint=True, applyFlatnessConstraint=True,
                                 resetCB=True, turnOnCB=False):
        '''
        Args:
            nSteps                  : number of steps taken for the design optimization
            trustRegionScale        : radius of the trust region when taking the newton step (according to some defined metric, usually the mass matrix)
            optTol                  : minimum relative improvement expected  on the objective
            useCG                   : whether we use NewtonCG or L-BFGS
            applyAngleConstraint    : whether we set a lower bound on the opening angles
            applyFlatnessConstraint : whether we want to enforce flatness of the undeployed state
            resetCB                 : whether we first reset the whole callback or not
            turnOnCB                : whether we want to get the convergence plots or not, note that this makes the optimization more costly
        '''

        if useCG:
            alg = linkage_optimization.OptAlgorithm.NEWTON_CG
        else:
            alg = linkage_optimization.OptAlgorithm.BFGS

        # Define the callback
        if resetCB or (self.optimizationCallback is None):
            self.optimizationCallback = CShellOptimizationCallback(self, updateColor=True, full=False,
                                                                   applyAngleConstraint=applyAngleConstraint, 
                                                                   applyFlatnessConstraint=applyFlatnessConstraint)

        self.optimizationCallback.ReinitializeTime()
        self.optimizationCallback.ReinitializeVars()
        self.optimizationCallback.SetTurnOnCB(turnOnCB)

        if applyAngleConstraint:
            angleValInit = self.linkageOptimizer.angle_constraint(self.linkageOptimizer.getFullDesignParameters())
            print("Inital min angle constraint value: {:.2e} (should be >= 0)".format(angleValInit))

        if applyFlatnessConstraint:
            cValInit = self.linkageOptimizer.c(self.linkageOptimizer.getFullDesignParameters())
            print("Inital flatness constraint value: {:.2e} (should be identically = 0)".format(cValInit))

        self.linkageOptimizer.CShellOptimize(alg, nSteps, trustRegionScale, optTol,
                                             self.optimizationCallback,
                                             applyAngleConstraint=applyAngleConstraint, 
                                             applyFlatnessConstraint=applyFlatnessConstraint)
        pass

    def SetOptimizationCallback(self, cb): self.optimizationCallback = cb

    def GetGradientTerm(self, currType, full):
        '''
        Change the weight so that we only get one gradient at a time

        Args:
            currType  : the current objective term we want to evaluate the gradient of
            full      : whether we run the full optimization (on the curves DoF) or not (just the rest quantities)

        Returns:
            grad : the corresponding gradient, a torch tensor of shape (nCurvesDoF,)
        '''

        self.linkageOptimizer.beta = 0
        self.linkageOptimizer.gamma = 0
        self.linkageOptimizer.smoothing_weight = 0
        self.linkageOptimizer.rl_regularization_weight = 0
        self.linkageOptimizer.contact_force_weight = 0
        self.cpRegWeight = 0
        useGradp = False
        if currType == linkage_optimization.OptEnergyType.Full:
            useGradp = True
            self.ApplyWeights()
        elif currType == linkage_optimization.OptEnergyType.Target:
            useGradp = True
            self.linkageOptimizer.beta = self.dictWeights["beta"]
        elif currType == linkage_optimization.OptEnergyType.Smoothing:
            useGradp = True
            self.linkageOptimizer.smoothing_weight = self.dictWeights["smoothingWeight"]
        elif currType == linkage_optimization.OptEnergyType.Regularization:
            useGradp = True
            self.linkageOptimizer.rl_regularization_weight = self.dictWeights["rlRegWeight"]
        elif currType == linkage_optimization.OptEnergyType.ElasticBase:
            useGradp = True
            self.linkageOptimizer.gamma = self.dictWeights["gamma"]
        elif currType == linkage_optimization.OptEnergyType.ElasticDeployed:
            useGradp = True
            self.linkageOptimizer.gamma = self.dictWeights["gamma"]
        elif currType == linkage_optimization.OptEnergyType.ContactForce:
            useGradp = True
            self.linkageOptimizer.contact_force_weight = self.dictWeights["contactForceWeight"]

        if useGradp:
            self.linkageOptimizer.invalidateAdjointState()
            gradp = self.linkageOptimizer.gradp_J(self.linkageOptimizer.getFullDesignParameters(), currType)
            if full:
                grad = self.PullDesignParametersToCurvesDoF(torch.tensor(gradp))
            else:
                grad = gradp

        else:
            if currType == "LaplacianCP":
                self.cpRegWeight = self.dictWeights["cpRegWeight"]
                gradCP = self.cpRegWeight * LeastSquaresLaplacianFullGradient(self.controlPoints, self.controlPointsFixed, self.lapCP)
                grad   = self.PullControlPointsToCurvesDoF(gradCP)

        # Restore the weights
        self.ApplyWeights()

        return grad

    def GetDissimilarityWithGradients(self, dirsCurvesDoF):
        '''
        Args:
            dirsCurvesDoF : torch tensor of size (?, nCurvesDoF) giving directions to compare with gradients

        Returns:
            dissims     : dictionnary of tensors of shape (?,) containing the dissimilarity for each term
            listDissims : list of dictionnaries containing the dissimilarity for each directions, for each term in the objective
        '''
        dictTerms = {
            "TargetFitting": linkage_optimization.OptEnergyType.Target,
            "RestCurvatureSmoothing": linkage_optimization.OptEnergyType.Smoothing,
            "RestLengthMinimization": linkage_optimization.OptEnergyType.Regularization,
            "ElasticEnergyFlat": linkage_optimization.OptEnergyType.ElasticBase,
            "ElasticEnergyDeployed": linkage_optimization.OptEnergyType.ElasticDeployed,
            "LaplacianCP": "LaplacianCP"
        }

        dissims = {}
        dirsNormed = dirsCurvesDoF / (torch.linalg.norm(dirsCurvesDoF, dim=1, keepdims=True) + 1e-14)

        for key in dictTerms:
            grad = self.GetGradientTerm(dictTerms[key], True)
            gradNorm = torch.linalg.norm(grad)
            
            dissims[key] = - dirsNormed @ grad / (gradNorm + 1e-14)

        listDissims = [{key: dissims[key][i].item() for key in dissims}for i in range(dirsCurvesDoF.shape[0])]

        return dissims, listDissims

    def PlotCurveLinkage(self, resetLims=False):
        '''
        Args:
            resetLim : whether or not to plot limits
        '''

    
        color = ['#63B0CD' if family else '#F6AE2D' for family in self.curvesFamily]
        zorders = [1 if family else 0 for family in self.curvesFamily]

        gs = gridspec.GridSpec(nrows=1, ncols=1, width_ratios=[1], height_ratios=[1])
        fig = plt.figure(figsize=(10, 10))

        axTmp = plt.subplot(gs[0, 0])
        for i, disc in enumerate(self.discList):
            axTmp.plot(ToNumpy(disc)[:, 0], ToNumpy(disc)[:, 1], c=color[i], linewidth=8, zorder=zorders[i])
        axTmp.scatter(ToNumpy(self.joints)[:, 0], ToNumpy(self.joints)[:, 1], c='k', marker='.', s=50, zorder=2)
        # axTmp.set_title("Joints and Edges", fontsize=14)
        axTmp.axis('equal')
        if not self.curvesPlotXLim is None:
            self.curvesPlotXLim = axTmp.set_xlim()
        else:
            self.curvesPlotXLim = axTmp.get_xlim()

        if not self.curvesPlotYLim is None:
            self.curvesPlotYLim = axTmp.set_ylim()
        else:
            self.curvesPlotYLim = axTmp.get_ylim()

        if resetLims:
            self.curvesPlotXLim = axTmp.get_xlim()
            self.curvesPlotYLim = axTmp.get_ylim()
        axTmp.axis("off")
        # axTmp.axes.xaxis.set_visible(False)
        # axTmp.axes.yaxis.set_visible(False)
        plt.show()
        pass

###########################################################################
#############           FUNCTIONS FOR THE PIPELINE            #############
###########################################################################

def GetEdgesFromCurves(curves):
    '''
    Args:
        curves      : list of list giving the joints through which each curve passes

    Returns:
        edges       : torch tensor of shape (nEdges, 2) containing all the edges
        edgeToCurve : list of nCurves 
    '''

    edges = torch.tensor([[crv[i], crv[i+1]] for crv in curves for i in range(len(crv)-1)])
    edgeToCurve = [i for i, crv in enumerate(curves) for j in range(len(crv)-1)]

    return edges, edgeToCurve


def ComputeElasticRodsDiscretization(controlPoints, jointsIdx, subdivision, mult=10, addFreeEnds=False):
    '''
    Args:
        controlPoints : shape (nCP, 3)
        jointsIdx     : list of the joints in the tensor of control points
        subdivision   : number of discretized points per rod segments
        mult          : multiplier for making curves unit speed
        addFreeEnds   : boolean telling whether we add free ends of the rods

    Returns:
        newSpline      : new constant speed splines (nRods of them)
        discSPoints    : shape (subdivision*(nJoints-1) + 2*addFreeEnds) containing 
                       the curve parameters of the discretized points

    Note:
        This is not perfectly matching for segments which do not impose their length. The reason for that is 
        certainly the shared edge computation
    '''

    # First make constant speed
    nJoints   = len(jointsIdx)
    if torch.isnan(controlPoints).any():
        print(controlPoints)
        print("NaN in ER discretization: {}".format(torch.isnan(controlPoints).any()))
    newSpline, newSKnots, _, _ = MakeConstantSpeed(controlPoints, mult)
    sJoints   = newSKnots[0, jointsIdx].reshape(-1,)                # (nJoints,)
    lenCurves = sJoints[1:] - sJoints[:-1]                          # (nJoints-1,)
    minLen    = torch.minimum(lenCurves[1:], lenCurves[:-1])        # (nJoints-2,)

    ## Get the length of edges for each rod segment
    sharedLen = torch.zeros_like(lenCurves)                         # (nJoints-1,)
    sharedLen[:-1] += minLen / (2 * (subdivision-1))
    sharedLen[1:]  += minLen / (2 * (subdivision-1))
    remSub          = (subdivision - 2) * torch.ones_like(lenCurves)      # (nJoints-1,)
    remSub[0]       = subdivision - 1.5 # this is required as end segments should have half an edge near the end joint
    remSub[-1]      = subdivision - 1.5
    lenOfEdge       = (lenCurves - sharedLen) / remSub               # (nJoints-1,)

    ## It remains to add stuff at each joint
    allLenEdges = torch.zeros(size=(2 + (nJoints-1) + (nJoints-2),)) # Even indices: edge len at each rod segment. Odd indices: edge length at joints (shared)
    allLenEdges[0]      = lenOfEdge[0] / 2
    allLenEdges[1:-1:2] = lenOfEdge
    allLenEdges[2:-1:2] = minLen / (2 * (subdivision-1))
    allLenEdges[-1]     = lenOfEdge[-1] / 2

    ## Get the discretized lengths
    repeats         = torch.zeros(size=(allLenEdges.shape[0],)).to(dtype=torch.int32)
    repeats[0]      = 1
    repeats[1:-1:2] = remSub
    repeats[2:-1:2] = 2
    repeats[-1]     = 1
    lenPerEdge      = torch.repeat_interleave(allLenEdges, repeats, dim=0)                        # ((nJoints-1)*subdivision,)

    ## Compute the discretized points' position
    discLenS      = torch.cumsum(torch.cat((torch.zeros(size=(1,)), lenPerEdge), dim=0), dim=0)   # (1 + (nJoints-1)*subdivision,)
    discSPoints   = discLenS / discLenS[-1]
    
    if addFreeEnds:
        startS      = (discSPoints[0] - discSPoints[1]).reshape(-1,)
        endS        = (2*discSPoints[-1] - discSPoints[-2]).reshape(-1,)
        discSPoints = torch.cat((startS, discSPoints, endS), dim=0)

    return newSpline, discSPoints

def StraightenInnerJointEdges(disc, discS, spline, subdivision):
    '''
    Args:
        disc        : shape (subdivision*(nJoints-1) + 2*addFreeEnds, 3) containing
                      the discrete positions of the rod segments
        discS       : shape (subdivision*(nJoints-1) + 2*addFreeEnds,) containing 
                      the curve parameters of the discretized points
        spline      : the current spline that we discretize
        subdivision : number of discretized points per rod segments

    Returns:
        disc        : shape (subdivision*(nJoints-1) + 2*addFreeEnds, 3) containing
                      the new discrete positions of the rod segments
    '''
    innerJointsIdx = [i for i in range(2, disc.shape[0]-2) if i%subdivision==1]
    if len(innerJointsIdx) == 0:
        return disc

    tangentJoints = spline.derivative(discS[innerJointsIdx].reshape(1, -1)).reshape(-1, 3) # shape (#innerJoints, 3)
    
    neighPointIdx = torch.tensor([[i + j for j in [-1, 0, 1]] for i in innerJointsIdx]).to(torch.int64)
    neighPoints   = disc[neighPointIdx]
    lenAtJoints   = torch.sum(torch.linalg.norm(neighPoints[:, 1:, :] - neighPoints[:, :-1, :], dim=2), dim=1) # shape (#innerJoints,)
    
    tangentJointsScaled  = lenAtJoints.reshape(-1, 1) * tangentJoints / torch.linalg.norm(tangentJoints, dim=-1, keepdims=True)
    pointBeforeJointsIdx = [i - 1 for i in innerJointsIdx]
    pointAfterJointsIdx  = [i + 1 for i in innerJointsIdx]
    
    disc[pointBeforeJointsIdx] = disc[innerJointsIdx] - tangentJointsScaled / 2
    disc[pointAfterJointsIdx]  = disc[innerJointsIdx] + tangentJointsScaled / 2
    
    return disc

def ExtractRestLengths(discPoints, nJoints, subdivision):
    '''
    Args:
        discPoints  : torch tensor of shape (nDisc, 3) containing the discretized point along the rod
        nJoints     : number of joints along each rod (should be the same accross the nRods rods)
        subdivision : number of subdivisions for each rod segment

    Returns:
        restLengthsFree  : tensor of shape (nDisc-nJoints-1,) containing the length of each free edge
        restLengthsJoint : tensor of shape (nJoints,) containing the length of each edge at joints
    '''

    edges   = discPoints[1:, :] - discPoints[:-1, :]      # (nDisc-1, 3)
    lengths = torch.linalg.norm(edges, dim=-1)            # (nDisc-1,)

    # First we get the rest lengths for the free edges
    # There are 2 joint edges per rod segment, hence the -2
    numFreeEdges = subdivision - 2
    idxFree = torch.tensor([(2 + numFreeEdges)*j + 2 + i for j in range(nJoints-1) for i in range(numFreeEdges)])
    restLengthsFree = lengths[idxFree].reshape(-1,)

    # Then we get the rest lengths for the joint edges
    idxAll    = torch.arange(lengths.shape[0])
    idxFilter = ~torch.isin(idxAll, idxFree)
    idxJoint  = idxAll[idxFilter]
    restLengthsJointDoubled = lengths[idxJoint].reshape(-1,)
    
    # This sums two consecutive elements of a vector and returns a vector of half the size
    sumTwoConsecutive = torch.repeat_interleave(torch.eye(restLengthsJointDoubled.shape[0] // 2), 2, dim=1)
    restLengthsJoint  = sumTwoConsecutive @ restLengthsJointDoubled

    return restLengthsFree, restLengthsJoint

def ExtractRestKappasVars(discPoints, subdivision):
    '''
    Args:
        discPoints  : torch tensor of shape (nDisc, 3) containing the discretized point along the rod
        subdivision : number of subdivisions for each rod segment

    Returns:
        restKappas : tensor of shape (nDisc-2-nJoints,) containing the rest kappas at each internal rod segment vertex
    '''
    # Remove the joints first
    idxKeep = [i for i in range(discPoints.shape[0]) if (i-1)%subdivision!=0]
    discKeepPoints = discPoints[idxKeep]                                             # (nDisc-nJoints, 3)

    edges      = discKeepPoints[1:, :] - discKeepPoints[:-1, :]                      # (nDisc-1-nJoints, 3)
    unitEdges  = edges / torch.linalg.norm(edges, dim=-1, keepdims=True)             # (nDisc-1-nJoints, 3)
    dotEdges   = torch.einsum('jk, jk -> j', unitEdges[:-1, :], unitEdges[1:, :])    # (nDisc-2-nJoints,)
    crossEdges = torch.cross(unitEdges[:-1, :], unitEdges[1:, :], dim=-1)            # (nDisc-2-nJoints, 3)

    # First compute curvature binormal and extract the curvature
    binormals = 2.0 * crossEdges / (1.0 + dotEdges.unsqueeze(-1))                    # (nDisc-2-nJoints, 3)
    allKappas = binormals[..., -1]                                                   # (nDisc-2-nJoints,)
    
    return allKappas.reshape(-1,)

