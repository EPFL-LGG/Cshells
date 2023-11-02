import MeshFEM, py_newton_optimizer
import elastic_rods
import numpy as np
from numpy.linalg import norm
import math, random
from elastic_rods import EnergyType, SurfaceAttractionEnergyType, compute_equilibrium
from bending_validation import suppress_stdout as so
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pickle

# Rotate v around axis using Rodrigues' rotation formula
def rotatedVector(sinThetaAxis, cosTheta, v):
    sinThetaSq = np.dot(sinThetaAxis, sinThetaAxis)
    # More robust handling of small rotations:
    # Use identity (1 - cosTheta) / (sinTheta^2) = 1/2 sec(theta / 2)^2
    #  ~= 1 / 2 sec(sinTheta / 2)^2
    # (small angle approximation for theta)
    normalization = 0
    if (sinThetaSq > 1e-6):
        normalization = (1 - cosTheta) / sinThetaSq
    else:
        tmp = math.cos(0.5 * math.sqrt(sinThetaSq))
        normalization = 0.5 / (tmp * tmp)
    return sinThetaAxis * (np.dot(sinThetaAxis, v) * normalization) + cosTheta * v + np.cross(sinThetaAxis, v)

# Apply a random perturbation to the joint z positions to try to break symmetry.
def perturb_joints(linkage, zPerturbationEpsilon = 1e-3):
    dofs = np.array(linkage.getDoFs())
    zCoordDoFs = np.array(linkage.jointPositionDoFIndices())[2::3]
    dofs[zCoordDoFs] += 2 * zPerturbationEpsilon * (np.random.random_sample(len(zCoordDoFs)) - 0.5)
    linkage.setDoFs(dofs)

# Drive the linkage open either by opening a particular joint or by setting an
# average opening angle.
class AngleStepper:
    def __init__(self, useTargetAngleConstraint, linkage, jointIdx, fullAngle, numSteps):
        self.linkage = linkage
        self.joint = linkage.joint(jointIdx) # Ignored if useTargetAngleConstraint

        self.useTargetAngleConstraint = useTargetAngleConstraint
        if (useTargetAngleConstraint):
            self.currentAngle = self.linkage.averageJointAngle;
        else:
            self.currentAngle = self.joint.alpha

        self.thetaStep = fullAngle / numSteps

    def step(self):
        self.currentAngle += self.thetaStep
        if (not self.useTargetAngleConstraint):
            self.joint.alpha = self.currentAngle

# Drive open the linkage by opening the angle at jointIdx
def open_linkage(linkage, jointIdx, fullAngle, numSteps, view = None,
                 zPerturbationEpsilon = 0, equilibriumSolver = compute_equilibrium,
                 finalEquilibriumSolver = None, earlyStopIt = None, verbose = True,
                 maxNewtonIterationsIntermediate = 15,
                 useTargetAngleConstraint = False,
                 outPathFormat = None,
                 iterationCallback = None):

    useSAL = isinstance(linkage, elastic_rods.SurfaceAttractedLinkage)
    eTypes = (SurfaceAttractionEnergyType if useSAL else EnergyType)

    if not useSAL:
        j = linkage.joint(jointIdx)
        jdo = linkage.dofOffsetForJoint(jointIdx)
        if (useTargetAngleConstraint):
            fixedVars = list(range(jdo, jdo + 6)) # fix rigid motion by constrainting the joint orientation only
        else:
            fixedVars = list(range(jdo, jdo + 7)) # fix the orientation and angle at the driving joint
    else:
        fixedVars = []

    stepper = AngleStepper(useTargetAngleConstraint, linkage, jointIdx, fullAngle, numSteps)

    def reportIterate(it):
        print("\t".join(map(str, [stepper.joint.alpha, linkage.energy()]
                            + [linkage.energy(t) for t in eTypes.__members__.values()])))

    convergenceReports = []
    finalEnergies      = []
    actuationForces    = [linkage.gradient()[linkage.jointAngleDoFIndices()[jointIdx]]]
    average_angles     = [linkage.averageJointAngle]

    opts = py_newton_optimizer.NewtonOptimizerOptions()
    opts.verbose = verbose
    opts.niter = maxNewtonIterationsIntermediate

    # Before taking the first step
    finalEnergies = {'{}'.format(t):[linkage.energy(t)] for t in eTypes.__members__.values()}

    for it in range(1, numSteps + 1):
        if ((earlyStopIt != None) and it == earlyStopIt):
            return

        if (iterationCallback is not None): iterationCallback(it - 1, linkage)
        stepper.step()
        perturb_joints(linkage, zPerturbationEpsilon)

        tgtAngle = stepper.currentAngle if useTargetAngleConstraint else elastic_rods.TARGET_ANGLE_NONE
        print("target angle: ", tgtAngle)
        r = equilibriumSolver(tgtAngle, linkage, opts, fixedVars)
        # pickle.dump(linkage, open('open_post_step_{}.pkl'.format(it), 'wb'))

        convergenceReports.append(r)
        for t in eTypes.__members__.values(): finalEnergies['{}'.format(t)].append(linkage.energy(t))
        actuationForces.append(linkage.gradient()[linkage.jointAngleDoFIndices()[jointIdx]])
        average_angles.append(linkage.averageJointAngle)

        if (view is not None):
            view.update(False)

        if (outPathFormat is not None):
            linkage.saveVisualizationGeometry(outPathFormat.format(it), averagedMaterialFrames=True)

        reportIterate(it)

    if (finalEquilibriumSolver is None):
        finalEquilibriumSolver = equilibriumSolver
    opts.niter = 1000;
    tgtAngle = stepper.currentAngle if useTargetAngleConstraint else elastic_rods.TARGET_ANGLE_NONE
    r = finalEquilibriumSolver(tgtAngle, linkage, opts, fixedVars)
    convergenceReports.append(r)

    if (iterationCallback is not None):
        iterationCallback(len(convergenceReports) - 1, linkage)

    for t in eTypes.__members__.values(): finalEnergies['{}'.format(t)].append(linkage.energy(t))
    actuationForces.append(linkage.gradient()[linkage.jointAngleDoFIndices()[jointIdx]])
    average_angles.append(linkage.averageJointAngle)
    if (view is not None):
        view.update(False)
    
    return convergenceReports, finalEnergies, actuationForces, average_angles

def RunAndAnalyzeDeployment(flatLinkage, deployedLinkage, numOpeningSteps=40, maxNewtonIterIntermediate=20):
    '''
    Takes a flat linkage and run the deployment again so that we can record some values
    '''
    useSAL = isinstance(deployedLinkage, elastic_rods.SurfaceAttractedLinkage)
    
    if useSAL:
        tsf = deployedLinkage.get_target_surface_fitter()
        openLinkage = elastic_rods.SurfaceAttractedLinkage(tsf.V, tsf.F, False, flatLinkage)
        openLinkage.attraction_weight = deployedLinkage.attraction_weight
        openLinkage.scaleJointWeights(jointPosWeight=deployedLinkage.get_attraction_tgt_joint_weight())
        openLinkage.setTargetJointsPosition(deployedLinkage.getTargetJointsPosition())
        openLinkage.set_holdClosestPointsFixed(deployedLinkage.get_holdClosestPointsFixed())
    else:
        openLinkage = elastic_rods.RodLinkage(flatLinkage)
    alphaTar = deployedLinkage.averageJointAngle
    driver   = deployedLinkage.centralJoint()
        
    def equilibriumSolver(tgtAngle, l, opts, fv):
        opts.gradTol = 1e-4
        return elastic_rods.compute_equilibrium(l, tgtAngle, options=opts, fixedVars=fv)
    
    openView = None
        
    with so(): 
        lstOut = open_linkage(openLinkage, driver, alphaTar - openLinkage.averageJointAngle, numOpeningSteps, 
                              openView, equilibriumSolver=equilibriumSolver, 
                              maxNewtonIterationsIntermediate=maxNewtonIterIntermediate, useTargetAngleConstraint=True)
        
    dicOut = {}
    dicOut["ConvergenceReports"] = lstOut[0]
    dicOut["finalEnergies"]      = lstOut[1]
    dicOut["actuationForces"]    = lstOut[2]
    dicOut["averageAngles"]      = lstOut[3]
    return dicOut

def PlotDeploymentQuantities(dicOut):
    '''
    Plot the energies/actuation forces at each opening step

    Input:
    - dicOut : dictionnary that contains "finalEnergies", "actuationForces", and "averageAngles" as output by RunAndAnalyzeDeployment
    '''
    gs = gridspec.GridSpec(nrows=1, ncols=3, height_ratios=[1], width_ratios=[1, 0.05, 1])
    fig = plt.figure(figsize=(16, 6))

    axTmp = plt.subplot(gs[0, 0])
    labels = [t.split("EnergyType.")[-1] for t in dicOut["finalEnergies"].keys()]
    for i, t in enumerate(dicOut["finalEnergies"].keys()):
        axTmp.plot(dicOut["finalEnergies"][t], label=labels[i])
    axTmp.set_title("Energies as deployment goes", fontsize=14)
    axTmp.set_xlabel(r"Step (from $\alpha_t^i$={:.3f} to $\alpha_t^f$={:.3f})".format(dicOut["averageAngles"][0], dicOut["averageAngles"][-1]), fontsize=12)
    axTmp.set_ylabel("Energy values", fontsize=12)
    axTmp.legend(fontsize=12)
    axTmp.grid()

    axTmp = plt.subplot(gs[0, -1])
    labels = [t.split("EnergyType.")[-1] for t in dicOut["finalEnergies"].keys()]
    axTmp.plot(dicOut["actuationForces"])
    axTmp.set_title("Actuation forces as deployment goes", fontsize=14)
    axTmp.set_xlabel(r"Step (from $\alpha_t^i$={:.3f} to $\alpha_t^f$={:.3f})".format(dicOut["averageAngles"][0], dicOut["averageAngles"][-1]), fontsize=12)
    axTmp.set_ylabel("Force values", fontsize=12)
    axTmp.grid()
    plt.show()




