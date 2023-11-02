from knitro.numpy import *
import math
from VisUtils import CShellOptimizationCallback

def ToNumpy(tensor):
    return tensor.cpu().detach().clone().numpy()

class RestQuantitiesOptimizer():
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

    def __init__(self, linkageOptimizer, configPath, minAngle=None, applyFlatnessConstraint=False, minRL=None):
        self.linkageOptimizer        = linkageOptimizer
        self.configPath              = configPath
        self.applyFlatnessConstraint = applyFlatnessConstraint # Cannot handle that
        self.applyAngleConstraint    = not minAngle is None
        self.minAngle                = minAngle
        self.useHVP                  = None
        self.numSteps                = None
        self.readDescription         = False

        if minRL is None:
            height = linkageOptimizer.getDeployedLinkage().homogenousMaterial().crossSectionHeight
            width  = linkageOptimizer.getDeployedLinkage().homogenousMaterial().area / height
            self.minRL = width
        else:
            self.minRL = minRL

    def callbackEvalF (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro. The different terms in the
        optimization are hardcoded for simplicity, since some terms might use different 
        representations of the cshell
        '''
        if evalRequest.type != KN_RC_EVALFC:
            print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
            return -1
        currDP = np.array(evalRequest.x)

        currObj = self.linkageOptimizer.J(currDP)
        evalResult.obj = currObj

        if self.applyAngleConstraint:
            evalResult.c[0] = self.linkageOptimizer.angle_constraint(currDP)
        if self.applyFlatnessConstraint:
            evalResult.c[self.applyAngleConstraint] = self.linkageOptimizer.c(currDP)

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
        currDP = np.array(evalRequest.x)
        
        evalResult.c[0] = self.linkageOptimizer.angle_constraint(currDP)
        evalResult.c[self.applyAngleConstraint] = self.linkageOptimizer.c(currDP)

        return 0

    def callbackEvalG (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro
        '''
        if evalRequest.type != KN_RC_EVALGA:
            print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1
        currDP = np.array(evalRequest.x)
        
        if self.applyAngleConstraint:
            evalResult.jac[:currDP.shape[0]] = self.linkageOptimizer.gradp_angle_constraint(currDP)
        if self.applyFlatnessConstraint:
            evalResult.jac[self.applyAngleConstraint*currDP.shape[0]:] = self.linkageOptimizer.gradp_c(currDP)
        evalResult.objGrad = self.linkageOptimizer.gradp_J(currDP)
        return 0

    def callbackEvalCGrad (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro
        '''
        if evalRequest.type != KN_RC_EVALGA:
            print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1
        currDP = np.array(evalRequest.x)
        
        if self.applyAngleConstraint:
            evalResult.jac[:currDP.shape[0]] = self.linkageOptimizer.gradp_angle_constraint(currDP)
        if self.applyFlatnessConstraint:
            evalResult.jac[self.applyAngleConstraint*currDP.shape[0]:] = self.linkageOptimizer.gradp_c(currDP)
        return 0

    def callbackEvalHV (self, kc, cb, evalRequest, evalResult, userParams):
        '''
        This respects the function signature imposed by knitro
        '''
        if evalRequest.type != KN_RC_EVALHV and evalRequest.type != KN_RC_EVALHV_NO_F:
            print ("*** callbackEvalHV incorrectly called with eval type %d" % evalRequest.type)
            return -1
        
        currDP = np.array(evalRequest.x)
        dDP    = np.array(evalRequest.vec)
        sigma  = evalRequest.sigma
        lbda   = evalRequest.lambda_
        
        coeffAC = lbda[0] if self.applyAngleConstraint else 0.0
        coeffC  = lbda[(1 if self.applyAngleConstraint else 0)] if self.applyFlatnessConstraint else 0.0
        hvp     = self.linkageOptimizer.apply_hess(currDP, dDP, coeff_J=sigma, coeff_c=coeffC, coeff_angle_constraint=coeffAC)
        evalResult.hessVec = hvp

        return 0

    def newPtCallback(self, kc, x, lbda, userParams):

        currDP = np.array(x)
        self.linkageOptimizer.newPt(currDP)
        self.cocb()

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
        descrFull += self.GetConstraintsDescription()
        return descrFull

    def ReadFullDescription(self, descr):
        alg, weights, constraints = descr.split(" // ")

        if "HVP" in alg   : self.useHVP = False
        elif "Newton" in alg : self.useHVP = True
        else                 : self.useHVP = None

        self.numSteps         = int(alg.split("(max iterations: ")[1].split(", trust radius: ")[0])
        self.trustRegionScale = float(alg.split(", trust radius: ")[1].replace(").", ""))

        self.applyAngleConstraint = "True" in constraints
        if self.applyAngleConstraint:
            self.minAngle = float(constraints.split("eps = ")[-1].replace(").", ""))

        self.readDescription = True

    def OptimizeDP(self, numSteps=None, trustRegionScale=1.0, optTol=1.0e-2, maxEqSteps=None, useCB=False, 
                   computeGradMags=False, screenshot=None, flatView=None, deployedView=None, visDeviations=False, honorbounds=1):
        '''
        Args:
            numSteps         : number of steps. If None, knitro figures it out itself
            trustRegionScale : radius of the trust region when taking the newton step (according to some defined metric, usually the mass matrix)
            optTol           : optimality tolerance
            maxEqSteps       : maximum number of steps for each equilibrium solve (if not set, )
            useCB            : whether to use the new point callback or not
            screenshot       : a dictionnary containing "takeScreenshot", "camParams", "pathToSaveFolder"
            visDeviations    : whether we visualize target deviations of bending energies
            honorbounds      : whether we always enforce feasibility during design optimization

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
            self.useHVP   = KN_get_int_param (kc, "hessopt") == 5
            self.numSteps = numSteps
        else:
            useHVP    = KN_get_int_param (kc, "hessopt") == 5
            if self.useHVP != useHVP:
                print("WARNING: The algorithm specified in the description read does not match the one given in the file used to configure the optimizer. Using CG: {}".format(useCG))
                self.useHVP = useHVP

        if not self.readDescription:
            self.trustRegionScale = trustRegionScale
        self.optTol = optTol
        KN_set_double_param(kc, "delta", self.trustRegionScale)
        KN_set_double_param(kc, "opttol", self.optTol)
        KN_set_int_param(kc, "par_numthreads", 12)
        KN_set_int_param(kc, "honorbnds", honorbounds) # 1: always enforce feasibility
        KN_set_int_param(kc, "presolve", 0)  # 0: no presolve

        if not maxEqSteps is None:
            opt = self.linkageOptimizer.getEquilibriumOptions()
            opt.niter = maxEqSteps
            self.linkageOptimizer.setEquilibriumOptions(opt)
        
        # Specify the dimensionality of the input vector
        nDoF = self.linkageOptimizer.getFullDesignParameters().shape[0]
        dofIndices = KN_add_vars(kc, nDoF)

        # Set lower bounds on the rest lengths
        nRL   = self.linkageOptimizer.getDeployedLinkage().getPerSegmentRestLength().shape[0]
        idxRL = list(range(nRL))
        lbRL  = nRL * [self.minRL]
        KN_set_var_lobnds(kc, indexVars=idxRL, xLoBnds=lbRL)

        # Set bounds on the target angle
        if self.linkageOptimizer.getOptimizeTargetAngle():
            idxAlpha = [nDoF-1] # Last index
            lbAlpha  = [- 2.0 * math.pi]
            ubAlpha  = [  2.0 * math.pi]
            KN_set_var_lobnds(kc, indexVars=idxAlpha, xLoBnds=lbAlpha)
            KN_set_var_upbnds(kc, indexVars=idxAlpha, xUpBnds=ubAlpha)

        # Set the initial guess to be the current degrees of freedom
        initDoF = self.linkageOptimizer.getFullDesignParameters()
        KN_set_var_primal_init_values(kc, xInitVals=initDoF)

        if self.applyAngleConstraint:
            # Constraint smin(alpha) >= self.minAngle; note that linkageOptimizer.LOMinAngleConstraint::eval() returns smin(alpha) - eps.
            cIndices = KN_add_cons(kc, 1+self.applyFlatnessConstraint)
            KN_set_con_lobnds(kc, cIndices[0], self.minAngle - self.linkageOptimizer.getEpsMinAngleConstraint()) # Set bounds
            if self.applyFlatnessConstraint:
                KN_set_con_eqbnds(kc, cIndices[1], 0.0)
        elif self.applyFlatnessConstraint:
            cIndices = KN_add_cons(kc, 1)
            KN_set_con_eqbnds(kc, cIndices[0], 0.0)

        cb = KN_add_eval_callback(kc, evalObj=None, indexCons=None, funcCallback=self.callbackEvalF)
        KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE, jacIndexVars=KN_DENSE_ROWMAJOR, gradCallback=self.callbackEvalG)
        if self.useHVP:
            KN_set_int_param(kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW)
            KN_set_cb_hess(kc, cb, hessIndexVars1=KN_DENSE_ROWMAJOR, hessCallback=self.callbackEvalHV)


        # Set the new point callback
        optimPackage                    = {}
        optimPackage["optimizer"]       = self.linkageOptimizer
        optimPackage["flatView"]        = flatView
        optimPackage["deployedView"]    = deployedView

        self.cocb = CShellOptimizationCallback(
            optimPackage=optimPackage, updateColor=True, full=False,
            applyAngleConstraint=self.applyAngleConstraint, 
            applyFlatnessConstraint=self.applyFlatnessConstraint,
            computeGradMags=computeGradMags,
            screenshot=screenshot,
            visDeviations=visDeviations
        )

        self.cocb.ReinitializeVars()
        self.cocb.SetTurnOnCB(useCB)
        
        KN_set_newpt_callback(kc, self.newPtCallback, userParams=None)

        KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE) # 0 minimize, 1 maximize
        if not numSteps is None:
            KN_set_int_param(kc, "maxit", numSteps)

        # Solve the problem
        self.cocb.ReinitializeTime()
        nStatus = KN_solve(kc)

        # An example of obtaining solution information.
        nStatus, objSol, optDP, lambda_ = KN_get_solution(kc)
        if nStatus==0:
            print("The solution has converged.\nOptimal objective value: {:.2e}".format(objSol))
        else:
            print("The solution has not converged.\nThe status is {}".format(nStatus))

        # Delete the Knitro solver instance.
        KN_free(kc)

        return optDP, self.cocb

