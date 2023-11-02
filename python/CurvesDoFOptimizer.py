from knitro.numpy import *
import math
import numpy as np
import torch

from PlanarizationOptimizers import QuadsTriangulationsSignedAreas
from VisUtils import CShellOptimizationCallback

torch.set_default_dtype(torch.float64)
    
def ToNumpy(tensor):
    return tensor.cpu().detach().clone().numpy()

########################################################################
##  GENERAL UTILITY FUNCTIONS
########################################################################

def ConstraintsQuadsTriangulationsAreas(cshell):
    '''
    Args:
        cshell: the cshell object for which we would like to compute the constraints
    
    Returns:
        areasMaxMin: the maximum of the minimum
    '''
    
    if not cshell.symmetry is None:
        cshell.fullCurvesDoF = cshell.symmetry.MapToFullCurvesDoF(cshell.curvesDoF)
    else:
        cshell.fullCurvesDoF = cshell.curvesDoF

    nAdd = sum([sum(addCP) for addCP in cshell.nCPperRodEdge])
    if nAdd == 0:
        cshell.controlPoints = torch.zeros(size=(cshell.nJ, 3))
        cshell.controlPoints[:, :2] = cshell.fullCurvesDoF[:2*cshell.nJ].reshape(-1, 2)

    jointsFlatTmp        = torch.zeros(size=(cshell.nJ, 3))
    jointsFlatTmp[:, :2] = cshell.fullCurvesDoF[:2*cshell.nJ].reshape(-1, 2)
    quadsTmp = QuadsTriangulationsSignedAreas(jointsFlatTmp[cshell.jointsQuads, :]) * cshell.quadsOrientation.reshape(-1, 1, 1)
    areasMaxMin = torch.maximum(torch.minimum(quadsTmp[:, 0, 0], quadsTmp[:, 0, 1]), 
                                torch.minimum(quadsTmp[:, 1, 0], quadsTmp[:, 1, 1]))
    return areasMaxMin

def JacConstraintsQuadsTriangulationsAreasCurvesDoFs(cshell):
    '''
    Args:
        cshell: the cshell object for which we would like to compute the constraints
    
    Returns:
        jac: the jacobian of the constraints with respect to the curves DoFs (nConstrains, nCurvesDoF)
    '''
    
    jac = torch.zeros(size=(len(cshell.jointsQuads), cshell.GetNumDoF()))
    
    cshell.CleanGradients()
    currCurvesDoF = cshell.curvesDoF
    
    def MapCurvesDoFsToConstraints(curvesDoF):
        cshell.curvesDoF = curvesDoF
        return ConstraintsQuadsTriangulationsAreas(cshell)
    
    # Usually more output than input => use jacfwd instead of jacrev
    jac[:, :cshell.curvesDoF.shape[0]] = torch.func.jacfwd(MapCurvesDoFsToConstraints)(currCurvesDoF)
    cshell.CleanGradients()
    
    return jac

########################################################################
##  CURVES DOF OPTIMIZER
########################################################################

class CurvesDoFOptimizer():
    '''
    Attributes:
        cshell               : a cshell object to be optimized
        configPath           : a string giving the absolute path to the .opt file used for the optimization
        applyAngleConstraint : whether we apply the min angle constraint
        minAngle             : the minimum value to use for the minimum angle constraint
        useHVP               : whether we use the exact HVP or not or not (filled when calling OptimizeDoF())
        numSteps             : maximum number of iterations (filled when calling OptimizeDoF())
        readDescription      : whether the optimizer read a description and should use its parameters
    '''

    def __init__(self, cshell, configPath, minAngle=None, minRL=None, minArea=None):
        self.cshell               = cshell
        self.configPath           = configPath
        self.applyAngleConstraint = not minAngle is None
        self.minAngle             = minAngle
        self.applyAreaConstraint  = not minArea is None
        self.minArea              = minArea
        self.useHVP               = None
        self.numSteps             = None
        self.readDescription      = False

        if minRL is None:
            height = cshell.linkageOptimizer.getDeployedLinkage().homogenousMaterial().crossSectionHeight
            width  = cshell.linkageOptimizer.getDeployedLinkage().homogenousMaterial().area / height
            self.minRL = width
        else:
            self.minRL = minRL
        
        lengthCurves = np.array([sum([np.linalg.norm(cshell.joints[crv[i+1]] - cshell.joints[crv[i]]) for i in range(len(crv)-1)]) for crv in cshell.curves])
        self.scaleLengthA = max(lengthCurves[[fam == 0 for fam in cshell.curvesFamily]])
        self.scaleLengthB = max(lengthCurves[[fam == 1 for fam in cshell.curvesFamily]])
        if self.applyAreaConstraint:
            self.areaThresh = self.minArea * (self.scaleLengthA * self.scaleLengthB) / (2 * len(cshell.jointsQuads))
        else:
            self.areaThresh = 0.0
            
        self.nConstraints = self.applyAngleConstraint + self.applyAreaConstraint * len(cshell.jointsQuads)

    def callbackEvalF (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro. The different terms in the
        optimization are hardcoded for simplicity, since some terms might use different 
        representations of the cshell
        '''
        if evalRequest.type != KN_RC_EVALFC:
            print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
            return -1
        currDoF = torch.tensor(evalRequest.x)

        if self.cshell.optimizeAlpha: newAlpha = currDoF[-1]
        else                        : newAlpha = self.cshell.alphaTar
        
        self.cshell.UpdateCShell(currDoF[:self.cshell.curvesDoF.shape[0]], newAlpha, force=False, skipRL=False, resetRL=False, 
                                 commitLinesearchLinkage=False, cacheDeployed=False, useCached=True)
        currObj = self.cshell.ComputeObjective().item()
        
        evalResult.obj = currObj

        constraints = np.zeros(shape=(self.nConstraints,))
        if self.applyAngleConstraint:
            constraints[0] = self.cshell.linkageOptimizer.angle_constraint(ToNumpy(self.cshell.GetFullDP()))
        if self.applyAreaConstraint:
            areasMaxMin = ConstraintsQuadsTriangulationsAreas(self.cshell)
            constraints[self.applyAngleConstraint:] = ToNumpy(areasMaxMin)
        if self.nConstraints != 0:
            evalResult.c = constraints.copy()
        
        return 0

    def callbackEvalC (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro. The different terms in the
        optimization are hardcoded for simplicity, since some terms might use different 
        representations of the cshell
        '''
        if evalRequest.type != KN_RC_EVALFC:
            print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
            return -1
        currDoF = torch.tensor(evalRequest.x)

        if self.cshell.optimizeAlpha: newAlpha = currDoF[-1]
        else                        : newAlpha = self.cshell.alphaTar
        
        self.cshell.UpdateCShell(currDoF[:self.cshell.curvesDoF.shape[0]], newAlpha, force=False, skipRL=False, resetRL=False, 
                                 commitLinesearchLinkage=False, cacheDeployed=False, useCached=True)
        
        constraints = np.zeros(shape=(self.nConstraints,))
        if self.applyAngleConstraint:
            constraints[0] = self.cshell.linkageOptimizer.angle_constraint(ToNumpy(self.cshell.GetFullDP()))
        if self.applyAreaConstraint:
            areasMaxMin = ConstraintsQuadsTriangulationsAreas(self.cshell)
            constraints[self.applyAngleConstraint:] = ToNumpy(areasMaxMin)
        if self.nConstraints != 0:
            evalResult.c = constraints.copy()

        return 0

    def callbackEvalG (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro
        '''
        if evalRequest.type != KN_RC_EVALGA:
            print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1
        currDoF = torch.tensor(evalRequest.x)

        if self.cshell.optimizeAlpha: newAlpha = currDoF[-1]
        else                        : newAlpha = self.cshell.alphaTar
        
        self.cshell.UpdateCShell(currDoF[:self.cshell.curvesDoF.shape[0]], newAlpha, force=False, skipRL=False, resetRL=False, 
                                 commitLinesearchLinkage=False, cacheDeployed=False, useCached=True) 

        jacConstraints = np.zeros(shape=(self.nConstraints, self.cshell.GetNumDoF()))
        if self.applyAngleConstraint:
            # Could be improved by calling backward only once? No batching available... yet?
            gradAngle = self.cshell.linkageOptimizer.gradp_angle_constraint(ToNumpy(self.cshell.GetFullDP()))
            jacConstraints[0, :] = ToNumpy(self.cshell.PullDesignParametersToCurvesDoF(torch.tensor(gradAngle)))
        if self.applyAreaConstraint:
            jacConstraints[self.applyAngleConstraint:, :] = ToNumpy(JacConstraintsQuadsTriangulationsAreasCurvesDoFs(self.cshell))
        
        evalResult.jac = jacConstraints.reshape(-1,)

        grad = ToNumpy(self.cshell.ComputeGradient())
        evalResult.objGrad = grad
        return 0

    def callbackEvalCGrad (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro
        '''
        if evalRequest.type != KN_RC_EVALGA:
            print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1
        currDoF = torch.tensor(evalRequest.x)

        if self.cshell.optimizeAlpha: newAlpha = currDoF[-1]
        else                        : newAlpha = self.cshell.alphaTar
        
        self.cshell.UpdateCShell(currDoF[:self.cshell.curvesDoF.shape[0]], newAlpha, force=False, skipRL=False, resetRL=False, 
                                 commitLinesearchLinkage=False, cacheDeployed=False, useCached=True) 

        jacConstraints = np.zeros(shape=(self.nConstraints, self.cshell.GetNumDoF()))
        if self.applyAngleConstraint:
            # Could be improved by calling backward only once? No batching available... yet?
            gradAngle = self.cshell.linkageOptimizer.gradp_angle_constraint(ToNumpy(self.cshell.GetFullDP()))
            jacConstraints[0, :] = ToNumpy(self.cshell.PullDesignParametersToCurvesDoF(torch.tensor(gradAngle)))
        if self.applyAreaConstraint:
            jacConstraints[self.applyAngleConstraint:, :] = ToNumpy(JacConstraintsQuadsTriangulationsAreasCurvesDoFs(self.cshell))

        evalResult.jac = jacConstraints.reshape(-1,)
        return 0

    def callbackEvalHV (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro
        '''
        if evalRequest.type != KN_RC_EVALHV and evalRequest.type != KN_RC_EVALHV_NO_F:
            print ("*** callbackEvalHV incorrectly called with eval type %d" % evalRequest.type)
            return -1
        
        currDoF = torch.tensor(evalRequest.x)

        if self.cshell.optimizeAlpha: newAlpha = currDoF[-1]
        else                        : newAlpha = self.cshell.alphaTar

        dDof  = torch.tensor(evalRequest.vec)
        sigma = evalRequest.sigma
        lbda  = evalRequest.lambda_

        self.cshell.UpdateCShell(currDoF[:self.cshell.curvesDoF.shape[0]], newAlpha, force=False, skipRL=False, resetRL=False, 
                                 commitLinesearchLinkage=False, cacheDeployed=False, useCached=True)
        
        coeffAC = lbda[0] if self.applyAngleConstraint else 0.
        hvp = ToNumpy(self.cshell.ComputeHessianVectorProduct(dDof, coeffJ=sigma, coeffC=0., coeffAC=coeffAC))
        evalResult.hessVec = hvp

        return 0

    def newPtCallback(self, kc, x, lbda, userParams):
        currDoF = torch.tensor(x)

        if self.cshell.optimizeAlpha: newAlpha = currDoF[-1]
        else                        : newAlpha = self.cshell.alphaTar
        
        self.cshell.UpdateCShell(currDoF[:self.cshell.curvesDoF.shape[0]], newAlpha, force=False, resetCPFixed=True, skipRL=False, resetRL=False, 
                                 commitLinesearchLinkage=True, cacheDeployed=True, useCached=True)
        self.cshell.optimizationCallback()

        return 0

    def GetConstraintsDescription(self):
        descrConstraints = "Using min angle constraint: " + str(self.applyAngleConstraint)
        if self.applyAngleConstraint:
            descrConstraints += " (eps = {})".format(self.minAngle)
        descrConstraints += "."
        return descrConstraints

    def GetOptimizationDescription(self):
        if self.useHVP is None: hess = "unknown"
        elif self.useHVP      : hess = "HVP"
        else                  : hess = "L-BFGS"

        descrOptimization = "Hessian used: " + hess + " (max iterations: {}, trust radius: {}).".format(self.numSteps, self.trustRegionScale)

        return descrOptimization

    def GetFullDescription(self):
        descrFull  = self.GetOptimizationDescription() + " // "
        descrFull += self.cshell.GetWeightDescription() + " // "
        descrFull += self.GetConstraintsDescription()
        return descrFull

    def ReadFullDescription(self, descr):
        alg, weights, constraints = descr.split(" // ")

        if "HVP" in alg   : self.useHVP = False
        elif "Newton" in alg : self.useHVP = True
        else                 : self.useHVP = None

        self.numSteps         = int(alg.split("(max iterations: ")[1].split(", trust radius: ")[0])
        self.trustRegionScale = float(alg.split(", trust radius: ")[1].replace(").", ""))

        listWeights      = weights.replace("Weights used: ", "").split(", ")
        listWeights[-1]  = listWeights[-1][:-1]
        splitListWeights = [w.split(": ") for w in listWeights]
        dictWeights      = {w[0]: float(w[1]) for w in splitListWeights}
        self.cshell.SetWeights(**dictWeights)

        self.applyAngleConstraint = "True" in constraints
        if self.applyAngleConstraint:
            self.minAngle = float(constraints.split("eps = ")[-1].replace(").", ""))

        self.readDescription = True

    def OptimizeDoF(self, numSteps=None, trustRegionScale=1.0, 
                    optTol=1.0e-2, ftol=1.0e-6, ftol_iters=2, honorbounds=1,
                    maxEqSteps=None, updateCShell=False, useCB=False, 
                    computeGradMags=False, 
                    screenshot=None, visDeviations=False, saveGeometryPath=None, saveGeometryFrequency=1):
        '''
        Args:
            numSteps         : number of steps. If None, knitro figures it out itself
            trustRegionScale : radius of the trust region when taking the newton step (according to some defined metric, usually the mass matrix)
            maxEqSteps       : maximum number of steps for each equilibrium solve (if not set, )
            updateCShell     : whether we directly update the cshell or not
            useCB            : whether to use the new point callback or not
            screenshot       : a dictionnary containing "takeScreenshot", "camParams", "pathToSaveFolder"
            visDeviations    : whether we visualize target deviations of bending energies
            honorbounds      : whether we always enforce feasibility during design optimization
            saveGeometryPath      : where we want the CShell to be saved (json)
            saveGeometryFrequency : how frequently (in terms of iterations) we want to save the geometry

        Returns:
            optDoF : torch tensor of shape (nDoF,) giving the optimized degrees of freedom
            alphaT : the optimized target angle
        '''

        try:
            kc = KN_new()
        except:
            print("Failed to find a valid license.")
            
        KN_load_param_file(kc, self.configPath)
        if not self.readDescription:
            self.useHVP    = KN_get_int_param (kc, "hessopt") == 5
            self.numSteps = numSteps
        else:
            useHVP    = KN_get_int_param (kc, "hessopt") == 5
            if self.useHVP != useHVP:
                print("WARNING: The algorithm specified in the description read does not match the one given in the file used to configure the optimizer. Using CG: {}".format(useCG))
                self.useHVP = useHVP

        if not self.readDescription:
            self.trustRegionScale = trustRegionScale
        self.optTol = optTol
        KN_set_double_param(kc, "delta",  self.trustRegionScale)
        KN_set_double_param(kc, "opttol", self.optTol)
        KN_set_double_param(kc, "ftol",   ftol)
        KN_set_int_param(kc, "ftol_iters",     ftol_iters)
        KN_set_int_param(kc, "par_numthreads", 12)
        KN_set_int_param(kc, "honorbnds",      honorbounds) # 1: always enforce feasibility
        KN_set_int_param(kc, "presolve",       0)  # 0: no presolve

        if not maxEqSteps is None:
            self.cshell.newtonOptimizerOptions.niter = maxEqSteps
            self.cshell.linkageOptimizer.setEquilibriumOptions(self.cshell.newtonOptimizerOptions)
        
        # Specify the dimensionality of the input vector
        nDoF = self.cshell.GetNumDoF()
        dofIndices = KN_add_vars(kc, nDoF)

        # Set bounds on the target angle
        if self.cshell.optimizeAlpha:
            idxAlpha = [self.cshell.curvesDoF.shape[0]]
            lbAlpha  = [- 2 * math.pi]
            ubAlpha  = [  2 * math.pi]
            KN_set_var_lobnds(kc, indexVars=idxAlpha, xLoBnds=lbAlpha)
            KN_set_var_upbnds(kc, indexVars=idxAlpha, xUpBnds=ubAlpha)

        # Set the initial guess to be the current degrees of freedom
        initDoF = ToNumpy(self.cshell.GetFullDoF())
        KN_set_var_primal_init_values(kc, xInitVals=initDoF)

        if self.nConstraints != 0:
            cIndices = KN_add_cons(kc, self.nConstraints)
        if self.applyAngleConstraint:
            # Constraint smin(alpha) >= self.minAngle; note that linkageOptimizer.LOMinAngleConstraint::eval() returns smin(alpha) - eps.
            KN_set_con_lobnds(kc, cIndices[0], self.minAngle - self.cshell.linkageOptimizer.getEpsMinAngleConstraint()) # Set bounds
        if self.applyAreaConstraint:
            for i in range(self.applyAngleConstraint, self.nConstraints):
                KN_set_con_lobnds(kc, cIndices[i], self.areaThresh) # Set bounds

        # Set the objective callbacks
        cb = KN_add_eval_callback(kc, evalObj=None, indexCons=None, funcCallback=self.callbackEvalF)
        KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE, jacIndexVars=KN_DENSE_ROWMAJOR, gradCallback=self.callbackEvalG)
        if self.useHVP:
            if self.applyAreaConstraint:
                print("Cannot use HVP and area constraints, please disable either one of the two.")
                raise NotImplementedError
            KN_set_int_param(kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW)
            KN_set_cb_hess(kc, cb, hessIndexVars1=KN_DENSE_ROWMAJOR, hessCallback=self.callbackEvalHV)
        
        # Set the new point callback
        cocb = CShellOptimizationCallback(
            self.cshell, updateColor=True, full=True,
            applyAngleConstraint=self.applyAngleConstraint, 
            applyFlatnessConstraint=False,
            computeGradMags=computeGradMags,
            screenshot=screenshot,
            visDeviations=visDeviations, 
            saveGeometryPath=saveGeometryPath, 
            saveGeometryFrequency=saveGeometryFrequency
        )
        cocb.SetTurnOnCB(useCB)
        self.cshell.SetOptimizationCallback(cocb)
        KN_set_newpt_callback(kc, self.newPtCallback, userParams=None)

        KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE) # 0 minimize, 1 maximize
        if not numSteps is None:
            KN_set_int_param(kc, "maxit", numSteps)

        # Solve the problem
        nStatus = KN_solve(kc)

        # An example of obtaining solution information.
        nStatus, objSol, optDoF, lambda_ = KN_get_solution(kc)
        if nStatus==0:
            print("The solution has converged.\nOptimal objective value: {:.2e}".format(objSol))
        else:
            print("The solution has not converged.\nThe status is {}".format(nStatus))

        # Delete the Knitro solver instance.
        KN_free(kc)

        # Update the cshell accordingly
        if updateCShell:
            if self.cshell.optimizeAlpha:
                self.cshell.alphaTar = optDoF[-1]

            self.cshell.UpdateCShell(optDoF, self.cshell.alphaTar)

        return optDoF, self.cshell.linkageOptimizer.getTargetAngle()
