import json
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np

import average_angle_linkages
from average_angle_linkages import SurfaceAttractionEnergyType, compute_equilibrium
from bending_validation import suppress_stdout as so
from compute_vibrational_modes import compute_vibrational_modes
from CShellToJSON import ExtractQuantitiesPerSegment
import elastic_rods
from elastic_rods import EnergyType
from StiffnessAnalysis import ComputeRelativeStiffnessGap, ComputeCompliance
import py_newton_optimizer


# Apply a random perturbation to the joint z positions to try to break symmetry.
def perturb_joints(linkage, zPerturbationEpsilon=1.0e-3):
    dofs = np.array(linkage.getDoFs())
    zCoordDoFs = np.array(linkage.jointPositionDoFIndices())[2::3]
    dofs[zCoordDoFs] += 2 * zPerturbationEpsilon * (np.random.random_sample(len(zCoordDoFs)) - 0.5)
    linkage.setDoFs(dofs)
    
# Drive the linkage open either by directly setting the average angle variable.
class AverageAngleStepper:
    def __init__(self, linkage, fullAngle, numSteps):
        self.linkage = linkage
        self.averageAngleIndex = self.linkage.getAverageAngleIndex()

        self.currentAngle = self.linkage.getDoFs()[self.averageAngleIndex]
        self.initAverageAngle = self.currentAngle

        self.thetaStep = fullAngle / numSteps

    def step(self):
        self.currentAngle += self.thetaStep
        dof = self.linkage.getDoFs()
        dof[self.averageAngleIndex] = self.currentAngle
        print("Target angle: ", self.currentAngle)
        self.linkage.setDoFs(dof)

# Drive open the linkage by opening the angle at jointIdx
def open_average_angle_linkage(linkage, jointIdx, fullAngle, numSteps, view=None,
                 zPerturbationEpsilon=0.0, equilibriumSolver=compute_equilibrium,
                 finalEquilibriumSolver=None, earlyStopIt=None, verbose=True,
                 maxNewtonIterationsIntermediate=15, computeStresses=True,
                 saveIntermediate=False, rodEdgesFamily=None, tsf=None, pathToJsonFormat=None,
                 outPathFormat=None, additionalFixedVars=None, releaseFixedVarsAngle=None,
                 iterationCallback=None, computeModes=False):
    '''Opens a linkage incrementally.
    
    Args:
        linkage: the average angle linkage to open
        jointIdx: the index of the joint to fix if the linkage is not a surface attracted linkage
        fullAngle: the final average opening angle of the linkage
        numSteps: the number of steps to take to open the linkage
        view: the viewer to update after each step
        zPerturbationEpsilon: the amount of random perturbation to apply to the z coordinate of the joints
        equilibriumSolver: the function to use to solve for equilibrium at each step
        finalEquilibriumSolver: the function to use to solve for equilibrium at the end of the opening process
        earlyStopIt: if not None, the opening process will stop after earlyStopIt steps
        verbose: whether to print the energy at each step
        maxNewtonIterationsIntermediate: the maximum number of Newton iterations to take at each opening step
        computeStresses: whether to compute stresses at each step
        saveIntermediate: whether to save intermediate json files for the geometry
        rodEdgesFamily: the family for each rod segment (necessary if saveIntermediate is True)
        tsf: the target surface fitter object that serves computing deviation to a surface (necessary if saveIntermediate is True)
        pathToJsonFormat: the path to the json files folder (can be formatted to welcome the opening step) (necessary if saveIntermediate is True)
        outPathFormat: the path to the output files folder (can be formatted to welcome the opening step)
        additionalFixedVars: list that contains additional fixed vars during deployment
        releaseFixedVarsAngle: if provided, the fixed vars will be released when the average angle reaches this value
        iterationCallback: a function that takes the current iteration and the linkage as arguments and is called at each iteration
        computeModes: whether to compute the vibrational modes at each step
        
    Returns:
        convergenceReports: the convergence reports from the equilibrium solver at each step
        finalEnergies: the energies of the linkage at each step
        actuationTorque: the actuation torque at each step
        average_angles: the average angle at each step
        stresses: the stresses at each step
        finalModes: the vibrational modes at the end of the opening process
        finalStiffnessGap: the relative stiffness gap at the end of the opening process
        finalStiffnesses: the stiffnesses at the end of the opening process
        jointsPositions: the joints positions at each step
        perSegEnergies: the energies per segment at each step (bending, stretching, twisting)
    '''

    useSAL = isinstance(linkage, average_angle_linkages.AverageAngleSurfaceAttractedLinkage)
    eTypes = (SurfaceAttractionEnergyType if useSAL else EnergyType)
    
    if saveIntermediate:
        assert pathToJsonFormat is not None
        assert rodEdgesFamily is not None
        assert tsf is not None

        deployFamilyA, deployFamilyB = ExtractQuantitiesPerSegment(
            tsf, 
            rodEdgesFamily, 
            linkage.segment(0).rod.numEdges(), # subdivision
            linkage
        )
        
        jsonLinkages = {
            'TargetSurface': [{
                'Vertices': tsf.V.tolist(),
                'Faces': tsf.F.tolist(),
            }],
            'CrossSection': [linkage.homogenousMaterial().crossSection().params()],
            'Flat_FamilyA': [],
            'Flat_FamilyB': [],
            'Deploy_FamilyA': deployFamilyA,
            'Deploy_FamilyB': deployFamilyB,
        }
    
        with open(pathToJsonFormat.format(str(0).zfill(5)), "w") as f:
            json.dump(jsonLinkages, f) 

    if not useSAL:
        j = linkage.joint(jointIdx)
        jdo = linkage.dofOffsetForJoint(jointIdx)
        fixedVars = list(range(jdo, jdo + 6)) # fix rigid motion by constrainting the joint orientation only
        # Fix the average angle variable. 
        fixedVars.append(linkage.getAverageAngleIndex())
    else:
        jdo = linkage.dofOffsetForJoint(jointIdx)
        fixedVars = [linkage.getAverageAngleIndex()]
    if additionalFixedVars is None:
        additionalFixedVars = []
    fullFixedVars = fixedVars + additionalFixedVars

    stepper = AverageAngleStepper(linkage, fullAngle, numSteps)

    def reportIterate(it):
        print("\t".join(map(str, [stepper.currentAngle, linkage.energy()]
                            + [linkage.energy(t) for t in eTypes.__members__.values()])))
    convergenceReports = []
    finalEnergies      = []
    finalModes         = []
    finalStiffnessGap  = []
    finalStiffnesses   = []
    actuationTorque    = [linkage.gradient()[stepper.averageAngleIndex]]
    if computeStresses:
        currStresses       = GetFlattenedStressesFromLinkage(linkage)
        stresses           = {t:[currStresses[t]] for t in currStresses.keys()}
    else:
        stresses = {}
    average_angles     = [linkage.getDoFs()[linkage.getAverageAngleIndex()]]
    jointsPositions    = [linkage.jointPositions().reshape(-1, 3)]
    perSegEnergies     = {
        "Bending": [[s.rod.energyBend() for s in linkage.segments()]],
        "Stretching": [[s.rod.energyStretch() for s in linkage.segments()]],
        "Twisting": [[s.rod.energyTwist() for s in linkage.segments()]],
    }

    opts = py_newton_optimizer.NewtonOptimizerOptions()
    opts.verbose = verbose
    opts.niter = maxNewtonIterationsIntermediate

    # Before taking the first step
    finalEnergies = {'{}'.format(t):[linkage.energy(t)] for t in eTypes.__members__.values()}
    fixedVarsModes = []
    if computeModes:
        lambdas, _ = compute_vibrational_modes(linkage, fixedVars=fixedVarsModes, n=8, sigma=-0.001)
        finalModes.append(lambdas)
        finalStiffnessGap.append(ComputeRelativeStiffnessGap(linkage, stepper.thetaStep, [], multMass=1e-4))
        deltaE, dirs, commonLinearTerm, commonQuadTerm = ComputeCompliance(linkage, stepper.thetaStep, [], nDirs=10, multMass=1e-4)
        finalStiffnesses.append(deltaE + commonQuadTerm)

    for it in range(1, numSteps + 1):
        if ((earlyStopIt != None) and it == earlyStopIt):
            return

        if (iterationCallback is not None): iterationCallback(it - 1, linkage)
        stepper.step()
        if not releaseFixedVarsAngle is None:
            if abs(stepper.currentAngle - stepper.initAverageAngle) >= abs(releaseFixedVarsAngle - stepper.initAverageAngle):
                fullFixedVars = fixedVars
        perturb_joints(linkage, zPerturbationEpsilon)

        r = equilibriumSolver(elastic_rods.TARGET_ANGLE_NONE, linkage, opts, fullFixedVars)

        convergenceReports.append(r)
        for t in eTypes.__members__.values(): finalEnergies['{}'.format(t)].append(linkage.energy(t))
        if computeStresses:
            currStresses = GetFlattenedStressesFromLinkage(linkage)
            for t in stresses.keys(): stresses[t].append(currStresses[t])
        actuationTorque.append(linkage.gradient()[stepper.averageAngleIndex])
        average_angles.append(linkage.getDoFs()[linkage.getAverageAngleIndex()])
        jointsPositions.append(linkage.jointPositions().reshape(-1, 3))
        perSegEnergies["Bending"].append([s.rod.energyBend() for s in linkage.segments()])
        perSegEnergies["Stretching"].append([s.rod.energyStretch() for s in linkage.segments()])
        perSegEnergies["Twisting"].append([s.rod.energyTwist() for s in linkage.segments()])

        if computeModes:
            lambdas, _ = compute_vibrational_modes(linkage, fixedVars=fixedVarsModes, n=8, sigma=-0.001)
            finalModes.append(lambdas)
            finalStiffnessGap.append(ComputeRelativeStiffnessGap(linkage, stepper.thetaStep, [], multMass=1e-4))
            deltaE, dirs, commonLinearTerm, commonQuadTerm = ComputeCompliance(linkage, stepper.thetaStep, [], nDirs=10, multMass=1e-4)
            finalStiffnesses.append(deltaE + commonQuadTerm)

        if (view is not None):
            view.update(False)

        if (outPathFormat is not None):
            linkage.saveVisualizationGeometry(outPathFormat.format(it), averagedMaterialFrames=True)
            
        if saveIntermediate:
            assert pathToJsonFormat is not None
            assert rodEdgesFamily is not None
            assert tsf is not None

            deployFamilyA, deployFamilyB = ExtractQuantitiesPerSegment(
                tsf, 
                rodEdgesFamily, 
                linkage.segment(0).rod.numEdges(), # subdivision
                linkage
            )
            
            jsonLinkages = {
                'TargetSurface': [{
                    'Vertices': tsf.V.tolist(),
                    'Faces': tsf.F.tolist(),
                }],
                'CrossSection': [linkage.homogenousMaterial().crossSection().params()],
                'Flat_FamilyA': [],
                'Flat_FamilyB': [],
                'Deploy_FamilyA': deployFamilyA,
                'Deploy_FamilyB': deployFamilyB,
            }
        
            with open(pathToJsonFormat.format(str(it).zfill(5)), "w") as f:
                json.dump(jsonLinkages, f) 
            

        reportIterate(it)

    if (finalEquilibriumSolver is None):
        finalEquilibriumSolver = equilibriumSolver
    opts.niter   = 1000
    opts.gradTol = 1e-8
    r = finalEquilibriumSolver(elastic_rods.TARGET_ANGLE_NONE, linkage, opts, fullFixedVars)
    convergenceReports.append(r)

    if (iterationCallback is not None):
        iterationCallback(len(convergenceReports) - 1, linkage)

    actuationTorque.append(linkage.gradient()[stepper.averageAngleIndex])
    for t in eTypes.__members__.values(): finalEnergies['{}'.format(t)].append(linkage.energy(t))
    if computeStresses:
        currStresses = GetFlattenedStressesFromLinkage(linkage)
        for t in stresses.keys(): stresses[t].append(currStresses[t])
    average_angles.append(linkage.getDoFs()[linkage.getAverageAngleIndex()])
    jointsPositions.append(linkage.jointPositions().reshape(-1, 3))
    perSegEnergies["Bending"].append([s.rod.energyBend() for s in linkage.segments()])
    perSegEnergies["Stretching"].append([s.rod.energyStretch() for s in linkage.segments()])
    perSegEnergies["Twisting"].append([s.rod.energyTwist() for s in linkage.segments()])
    if computeModes:
        lambdas, _ = compute_vibrational_modes(linkage, fixedVars=fixedVarsModes, n=8, sigma=-0.001)
        finalModes.append(lambdas)
        finalStiffnessGap.append(ComputeRelativeStiffnessGap(linkage, stepper.thetaStep, [], multMass=1e-4))
        deltaE, _, commonLinearTerm, commonQuadTerm = ComputeCompliance(linkage, stepper.thetaStep, [], nDirs=10, multMass=1e-4)
        finalStiffnesses.append(deltaE + commonQuadTerm)
    if (view is not None):
        view.update(False)
    return convergenceReports, finalEnergies, actuationTorque, average_angles, stresses, finalModes, finalStiffnessGap, finalStiffnesses, jointsPositions, perSegEnergies

def GetFlattenedStressesFromLinkage(linkage):
    '''Extracts quantities for visualization from a linkage.
    
    Args:
        linkage: the linkage to extract quantities from
    '''

    stresses = {
        "Max von Mises Stress" : [],
        "Stretching Stress"    : [],
        "Min Bending Stress"   : [],
        "Max Bending Stress"   : [],
        "Sqrt Bending Energy"  : [],
        "Twisting Stress"      : []
    }
    NAS = 18446744073709551615 # NotASegment

    if linkage.hasCrossSection() and linkage.hasCrossSectionMesh():
        iterable = zip(linkage.segments(), linkage.maxVonMisesStresses(), linkage.stretchingStresses(), linkage.minBendingStresses(), 
                    linkage.maxBendingStresses(), linkage.sqrtBendingEnergies(), linkage.twistingStresses())
    else:
        iterable = zip(linkage.segments(), linkage.stretchingStresses(), linkage.stretchingStresses(), linkage.minBendingStresses(), 
                    linkage.maxBendingStresses(), linkage.sqrtBendingEnergies(), linkage.twistingStresses())
    
    for segID, listIters in enumerate(iterable):
        (seg, vm, stretch, minB, maxB, sqrtB, twist) = listIters
        startJoint = linkage.joint(seg.startJoint)
        if set(startJoint.segments_A) == {segID, NAS}:
            vm      = vm[1:]
            stretch = stretch[1:]
            minB    = minB[1:]
            maxB    = maxB[1:]
            sqrtB   = sqrtB[1:]
            twist   = twist[1:]
        elif set(startJoint.segments_B) == {segID, NAS}:
            vm      = vm[1:]
            stretch = stretch[1:]
            minB    = minB[1:]
            maxB    = maxB[1:]
            sqrtB   = sqrtB[1:]
            twist   = twist[1:]

        endJoint = linkage.joint(seg.endJoint)
        if set(endJoint.segments_A) == {segID, NAS}:
            vm      = vm[:-1]
            stretch = stretch[:-1]
            minB    = minB[:-1]
            maxB    = maxB[:-1]
            sqrtB   = sqrtB[:-1]
            twist   = twist[:-1]
        elif set(endJoint.segments_B) == {segID, NAS}:
            vm      = vm[:-1]
            stretch = stretch[:-1]
            minB    = minB[:-1]
            maxB    = maxB[:-1]
            sqrtB   = sqrtB[:-1]
            twist   = twist[:-1]

        if linkage.hasCrossSection() and linkage.hasCrossSectionMesh():
            stresses["Max von Mises Stress"] += list(vm)
        else:
            stresses["Max von Mises Stress"] += [np.nan for i in list(vm)]
        stresses["Stretching Stress"]    += list(stretch)
        stresses["Min Bending Stress"]   += list(minB)
        stresses["Max Bending Stress"]   += list(maxB)
        stresses["Sqrt Bending Energy"]  += list(maxB)
        stresses["Twisting Stress"]      += list(twist)
    return stresses

def RunAndAnalyzeDeployment(flatLinkage, deployedLinkage, numOpeningSteps=40, maxNewtonIterIntermediate=20, 
                            additionalFixedVars=None, releaseFixedVarsAngle=None, tsf=None, attractionWeight=None, jointPosWeight=None,
                            saveIntermediate=False, rodEdgesFamily=None, pathToJsonFormat=None):
    '''Takes a flat linkage and run the deployment again so that we can record some values

    Args:
        flatLinkage               : an undeployed AAL or AASAL
        deployedLinkage           : a deployed AAL or AASAL (should have the same type as flatLinkage)
        numOpeningSteps           : number of opening steps, note that there will be a final equilibrium solve
        maxNewtonIterIntermediate : maximum number of Newton step taken at each opening step
        additionalFixedVars       : list that contains additional fixed vars during deployment
        tsf                       : the target surface fitter object that serves computing deviation to a surface (if deployedLinkage is not a AverageAngleSurfaceAttractedLinkage)
        attractionWeight          : if provided, overrides the one in deployedLinkage
        jointPosWeight            : if provided, overrides the one in deployedLinkage
        saveIntermediate          : whether we want to save intermediate json files for the geometry
        rodEdgesFamily            : the family for each rod segment
        pathToJsonFormat          : the path to the json files folder (can be formatted to welcome the opening step)

    Returns:
        dicOut : a dictionnary with various metrics
    '''
    useSAL = isinstance(deployedLinkage, average_angle_linkages.AverageAngleSurfaceAttractedLinkage)

    if useSAL:
        tsf = deployedLinkage.get_target_surface_fitter()
        openLinkage = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(tsf.V, tsf.F, False, flatLinkage)

        if attractionWeight is not None: openLinkage.attraction_weight = attractionWeight
        else: openLinkage.attraction_weight = deployedLinkage.attraction_weight

        if jointPosWeight is not None: openLinkage.scaleJointWeights(jointPosWeight=jointPosWeight)
        else: openLinkage.scaleJointWeights(jointPosWeight=deployedLinkage.get_attraction_tgt_joint_weight())

        openLinkage.setTargetJointsPosition(deployedLinkage.getTargetJointsPosition())
        openLinkage.set_holdClosestPointsFixed(deployedLinkage.get_holdClosestPointsFixed())
    else:
        openLinkage = average_angle_linkages.AverageAngleLinkage(flatLinkage)
    alphaTar = deployedLinkage.getAverageActuatedJointsAngle()
    driver   = openLinkage.centralJoint()
        
    def equilibriumSolver(tgtAngle, l, opts, fv):
        opts.gradTol = 1.0e-6
        return average_angle_linkages.compute_equilibrium(l, tgtAngle, options=opts, fixedVars=fv)
    
    openView = None

    if not openLinkage.hasCrossSectionMesh(): 
        if not openLinkage.hasCrossSection():
            E  = openLinkage.homogenousMaterial().youngModulus
            nu = openLinkage.homogenousMaterial().youngModulus / (2 * openLinkage.homogenousMaterial().shearModulus) - 1
            height = openLinkage.homogenousMaterial().crossSectionHeight
            width  = openLinkage.homogenousMaterial().area / height
            cs = elastic_rods.CrossSection.construct("RECTANGLE", E, nu, [height, width])
            rm = elastic_rods.RodMaterial(cs)
            openLinkage.setMaterial(rm)
        openLinkage.meshCrossSection(0.001)
        openLinkage.setMaterial(openLinkage.homogenousMaterial())
        
    with so(): 
        lstOut = open_average_angle_linkage(openLinkage, driver, alphaTar - openLinkage.getAverageActuatedJointsAngle(), numOpeningSteps, 
                                openView, equilibriumSolver=equilibriumSolver, additionalFixedVars=additionalFixedVars, 
                                releaseFixedVarsAngle=releaseFixedVarsAngle,
                                saveIntermediate=saveIntermediate, rodEdgesFamily=rodEdgesFamily, tsf=tsf, pathToJsonFormat=pathToJsonFormat,
                                maxNewtonIterationsIntermediate=maxNewtonIterIntermediate, computeModes=True)
        
    dicOut = {}
    dicOut["ConvergenceReports"] = lstOut[0]
    dicOut["finalEnergies"]      = lstOut[1]
    dicOut["actuationTorque"]    = lstOut[2]
    dicOut["averageAngles"]      = lstOut[3]
    dicOut["stresses"]           = lstOut[4]
    dicOut["modes"]              = lstOut[5]
    dicOut["stiffnessGap"]       = lstOut[6]
    dicOut["stiffnesses"]        = lstOut[7]
    dicOut["jointsPositions"]    = lstOut[8]
    dicOut["perSegmentEnergies"] = lstOut[9]
    return dicOut

def ComputeStressesHistogram(stresses, minBounds=None, maxBounds=None, nBins=100):
    '''
    Args:
        stresses  : a dictionnary of lists as output by RunAndAnalyzeDeployment (key "stresses")
        minBounds : a dictionnary that contains the same keys as stresses
        maxBounds : a dictionnary that contains the same keys as stresses
        nBins     : the number of bins between minBounds and maxBounds

    Returns:
        stressHist : dictionnary that contains "totalHist", "mean", "std", and "yedges".
    '''
    stressHist = {}

    for t in stresses.keys():
        stressArr = np.array(stresses[t])
        if minBounds is None: minBound = np.min(stressArr)
        else                : minBound = minBounds[t]
        if maxBounds is None: maxBound = np.max(stressArr)
        else                : maxBound = maxBounds[t]

        bins = np.linspace(minBound, maxBound, nBins)

        totalHist = []
        for stress in np.array(stressArr):
            hist, yedges = np.histogram(stress, bins=bins)
            totalHist.append(list(hist))
            
        totalHist = np.array(totalHist)
        stressHist[t] = {
            "totalHist" : totalHist,
            "mean"      : np.mean(stressArr, axis=1),
            "median"    : np.median(stressArr, axis=1),
            "std"       : np.std(stressArr, axis=1),
            "95"        : np.percentile(stressArr, 95, axis=1),
            "5"         : np.percentile(stressArr, 5, axis=1),
            "min"       : np.min(stressArr, axis=1),
            "max"       : np.max(stressArr, axis=1),
            "yedges"    : yedges
        }

    return stressHist

def PlotDeploymentQuantities(dicOut, stressHist, showMeanStd=False):
    '''Plot the energies/actuation forces at each opening step

    Args:
        dicOut      : dictionnary that contains "finalEnergies", "actuationForces", and "averageAngles" as output by RunAndAnalyzeDeployment
        stressHist  : dictionnary that contains "totalHist", "mean", "std", and "yedges" as output by ComputeStressesHistogram
        showMeanStd : whether we want to overlay mean and standard deviation or not
    '''
    from copy import copy
    from matplotlib.colors import LogNorm

    gs = gridspec.GridSpec(nrows=5, ncols=5, height_ratios=[1, 0.05, 1, 0.05, 1], width_ratios=[1, 0.05, 1, 0.05, 1])
    fig = plt.figure(figsize=(20, 16))

    axTmp = plt.subplot(gs[0, 0])
    labels = [t.split("EnergyType.")[-1] for t in dicOut["finalEnergies"].keys()]
    for i, t in enumerate(dicOut["finalEnergies"].keys()):
        axTmp.plot(dicOut["finalEnergies"][t], label=labels[i])
    axTmp.set_title("Energies as deployment goes", fontsize=14)
    axTmp.set_xlabel(r"Step (from $\alpha_t^i$={:.3f} to $\alpha_t^f$={:.3f})".format(dicOut["averageAngles"][0], dicOut["averageAngles"][-1]), fontsize=12)
    axTmp.set_ylabel("Energy values", fontsize=12)
    axTmp.legend(fontsize=12)
    axTmp.grid()

    axTmp = plt.subplot(gs[0, 2])
    labels = [t.split("EnergyType.")[-1] for t in dicOut["finalEnergies"].keys()]
    axTmp.plot(dicOut["actuationTorque"])
    axTmp.set_title("Actuation torque as deployment goes", fontsize=14)
    axTmp.set_xlabel(r"Step (from $\alpha_t^i$={:.3f} to $\alpha_t^f$={:.3f})".format(dicOut["averageAngles"][0], dicOut["averageAngles"][-1]), fontsize=12)
    axTmp.set_ylabel("Torque values", fontsize=12)
    axTmp.grid()

    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))

    xedges = dicOut["averageAngles"] + [2 * dicOut["averageAngles"][-1] - dicOut["averageAngles"][-3]]
    xmeans = dicOut["averageAngles"][:-1] + [2 * dicOut["averageAngles"][-1] - dicOut["averageAngles"][-3]]

    locations = [[2, 0], [2, 2], [2, 4], [4, 0], [4, 2], [4, 4]]
    for (loc, t) in zip(locations, stressHist.keys()):
        axTmp = plt.subplot(gs[loc[0], loc[1]])
        currHist = stressHist[t]
        pcm = axTmp.pcolormesh(xedges, currHist["yedges"], currHist["totalHist"].T, cmap=cmap, norm=LogNorm(vmax=np.max(currHist["totalHist"])), rasterized=True)
        if showMeanStd:
            axTmp.plot(xmeans, currHist["mean"], '-', color='grey')
            axTmp.fill_between(xmeans, currHist["mean"] - currHist["std"], currHist["mean"] + currHist["std"], color='gray', alpha=0.3)
        fig.colorbar(pcm, ax=axTmp, label="Number of points")
        axTmp.set_title(t + " as deployment goes", fontsize=14)
        axTmp.set_xlabel(r"Step (from $\alpha_t^i$={:.3f} to $\alpha_t^f$={:.3f})".format(dicOut["averageAngles"][0], dicOut["averageAngles"][-1]), fontsize=12)
        axTmp.set_ylabel(t, fontsize=12)

    plt.show()