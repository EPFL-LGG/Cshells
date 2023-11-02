from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import time
import torch

from CShellToJSON import ConvertCShellToJSON
import linkage_optimization
from linkage_vis import LinkageViewer
from LaplacianSmoothing import LeastSquaresLaplacian, LeastSquaresLaplacianFullGradient
from VisUtilsCurveLinkage import ScalarFieldDeviations
from vis.fields import ScalarField # In mesh fem

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

###########################################################################
#############          CALLBACK FOR THE OPTIMIZATION          #############
###########################################################################

class CShellOptimizationCallback:
    def __init__(self, cshell=None, optimPackage=None, full=True, applyAngleConstraint=False, applyFlatnessConstraint=False, 
                 updateColor=False, callbackFreq=1, computeGradMags=True, screenshot=None, visDeviations=False,
                 saveGeometryPath=None, saveGeometryFrequency=1):
        '''
        Args:
            cshell                  : a FreeCShell object
            optimPackage            : a dictionnary that contains an "optimizer", "flatView", and "deployedView"
            full                    : whether we optimize the curves DoF or just the rest quantities
            applyAngleConstraint    : whether we impose the undeployed opening angle to be at least some value (only relevant when full is False)
            applyFlatnessConstraint : whether we impose the undeployed state to be flat (only relevant when full is False)
            updateColor             : whether we represent some scalar field in the viewers
            callbackFreq            : how frequently do we update the linkage viewers
            computeGradMags         : whether or not to compute the gradient magnitude at each time step
            screenshot              : a dictionnary containing "camParamsFlat", "camParamsDep", "pathToSaveFolder"
            visDeviations           : whether we visualize target deviations or bending energies
            saveGeometryPath        : where we want the CShell to be saved (json)
            saveGeometryFrequency   : how frequently (in terms of iterations) we want to save the geometry
        '''
        
        self.cshell                  = cshell
        usePackage                   = cshell is None
        assert (not usePackage) or (usePackage and (not optimPackage is None)) # Should have either a cshell or an optimPackage
        self.optimizer               = (optimPackage["optimizer"] if usePackage else cshell.linkageOptimizer)
        self.flatView                = (optimPackage["flatView"] if usePackage else cshell.flatView)
        self.deployedView            = (optimPackage["deployedView"] if usePackage else cshell.deployedView)
        self.full                    = full and (not usePackage) # Since we won't know what the curves DoF will be
        self.applyAngleConstraint    = applyAngleConstraint      # False in case we run the full optim
        self.applyFlatnessConstraint = applyFlatnessConstraint
        self.updateColor             = updateColor
        self.callbackFreq            = callbackFreq
        self.computeGradMags         = computeGradMags
        self.screenshot              = screenshot
        self.takeScreenShot          = not screenshot is None
        self.iterateData             = []
        self.visDeviations           = visDeviations
        self.saveGeometryPath        = saveGeometryPath
        self.saveGeometryFrequency   = saveGeometryFrequency

        # In case the viewers are not specified, fall back to basic ones
        flatLinkage     = self.optimizer.getBaseLinkage()
        deployedLinkage = self.optimizer.getDeployedLinkage()
        if self.flatView is None:
            self.flatView     = LinkageViewer(flatLinkage, width=768, height=480)
        if self.deployedView is None:
            self.deployedView = LinkageViewer(deployedLinkage, width=768, height=480)

        # Might want to call self.ReinitializeTime() and self.ReinitializeVars() before running the optimization
        self.prevVars      = None
        self.prevTimeStamp = None
        self.turnOnCB      = True

        self.ReinitializeVars()
        self.ReinitializeTime()

        # Initial state: record values of all objective terms, plus timestamp and variables.
        self.maxBEFlatInit     = None
        self.maxBEDeployedInit = None
        self.UpdateIterateData()
        self.UpdateViews(initialize=True)
        
        if self.saveGeometryPath is not None:
            self.SaveGeometry()

    def __call__(self, *args, **kwargs):

        if not self.turnOnCB:
            return 0

        if self.full:
            currVars = ToNumpy(self.cshell.GetFullDoF())
        else:
            currVars = self.optimizer.getFullDesignParameters()
        self.UpdateIterateData()
        self.prevVars      = currVars
        self.prevTimeStamp = time.time()
        
        if (self.NumIterations() % self.callbackFreq == 0):
            self.UpdateViews(initialize=False)
            
        if (self.NumIterations() % self.saveGeometryFrequency == 0):
            if self.saveGeometryPath is not None:
                self.SaveGeometry()

        return 0

    def SetTurnOnCB(self, turnOn): self.turnOnCB = turnOn
    def ReinitializeVars(self)   : 
        if self.full:
            self.prevVars  = ToNumpy(self.cshell.GetFullDoF())
        else:
            self.prevVars  = self.optimizer.getFullDesignParameters()
    def ReinitializeTime(self)   : self.prevTimeStamp = time.time()
    def SaveGeometry(self): ConvertCShellToJSON(self.cshell, self.saveGeometryPath.format(str(self.NumIterations()).zfill(5)))
    def NumIterations(self): return len(self.iterateData) - 1 # the initial state doesn't count as a step

    def UpdateIterateData(self):
        # Record values of all objective terms, plus timestamp and variables.
        idata  = {t.name: t.term.value() for t in self.optimizer.objective.terms}
        if self.full :# Since Laplacian term is only visible when we have the full pipeline
            lapVal = self.cshell.cpRegWeight * LeastSquaresLaplacian(self.cshell.controlPoints, self.cshell.controlPointsFixed, self.cshell.lapCP)
            if self.computeGradMags:
                gradCP     = self.cshell.cpRegWeight * LeastSquaresLaplacianFullGradient(self.cshell.controlPoints, self.cshell.controlPointsFixed, self.cshell.lapCP)
                lapGradMag = torch.linalg.norm(self.cshell.PullControlPointsToCurvesDoF(gradCP))
            else:
                lapGradMag = 0
        else:
            lapVal     = 0
            lapGradMag = 0
        idata.update({'LaplacianCP': lapVal,
                        'LaplacianCP_grad_norm': lapGradMag})

        # Get the deviations
        tsf = self.optimizer.get_target_surface_fitter()
        W_diag = np.copy(tsf.W_diag_joint_pos)
        useCenterline = np.copy(tsf.getUseCenterline())
        deployedLinkage = self.optimizer.getDeployedLinkage()
        tsf.setUseCenterline(deployedLinkage, False, 0.1, jointPosValence2Multiplier=1.0)
        devJoints = np.linalg.norm((deployedLinkage.jointPositions() - tsf.linkage_closest_surf_pts).reshape(-1, 3), axis=-1)
        tsf.setUseCenterline(deployedLinkage, True, 0.1, jointPosValence2Multiplier=1.0)
        devCP  = np.linalg.norm((deployedLinkage.centerLinePositions() - tsf.linkage_closest_surf_pts).reshape(-1, 3), axis=-1)
        devTot = np.concatenate([devJoints, devCP], axis=0) 
        devReport = {
            "mean"      : np.mean(devTot),
            "median"    : np.median(devTot),
            "std"       : np.std(devTot),
            "95"        : np.percentile(devTot, 95),
            "5"         : np.percentile(devTot, 5),
            "min"       : np.min(devTot),
            "max"       : np.max(devTot)
        }
        tsf.setUseCenterline(deployedLinkage, useCenterline, sum(W_diag), jointPosValence2Multiplier=max(W_diag) / min(W_diag))
        idata.update({'TargetDeviationReport': devReport})

        if self.applyAngleConstraint:
            angleVal     = self.optimizer.angle_constraint(self.optimizer.getFullDesignParameters())
            if self.computeGradMags:
                angleGrad    = self.optimizer.gradp_angle_constraint(self.optimizer.getFullDesignParameters())
                if self.full:
                    angleGradMag = torch.linalg.norm(self.cshell.PullDesignParametersToCurvesDoF(torch.tensor(angleGrad)))
                else:
                    angleGradMag = np.linalg.norm(angleGrad)
            else:
                angleGradMag = 0
            idata.update({'MinAngleConstraint':           angleVal,
                          'MinAngleConstraint_grad_norm': angleGradMag})

        if self.applyFlatnessConstraint:
            cVal  = self.optimizer.c(self.optimizer.getFullDesignParameters())
            if self.computeGradMags:
                cGrad = self.optimizer.gradp_c(self.optimizer.getFullDesignParameters())
                if self.full:
                    cGradMag = torch.linalg.norm(self.cshell.PullDesignParametersToCurvesDoF(torch.tensor(cGrad)))
                else:
                    cGradMag = np.linalg.norm(cGrad)
            else:
                cGradMag = 0
            idata.update({'FlatnessConstraint':           cVal,
                          'FlatnessConstraint_grad_norm': cGradMag})
        idata.update({'iteration_time':   time.time() - self.prevTimeStamp,
                      'extendedDoFsPSRL': self.optimizer.getFullDesignParameters()})
        if self.computeGradMags:
            idata.update({'{}_grad_norm'.format(t.name): get_component_gradient_norm(self.optimizer, t.type, self.cshell, self.full) for t in self.optimizer.objective.terms})
        else:
            idata.update({'{}_grad_norm'.format(t.name): 0. for t in self.optimizer.objective.terms})

        # Compute the gradient of the total objective
        totalGradRQ = self.optimizer.gradp_J(self.optimizer.getFullDesignParameters())
        idata.update({'total_grad_norm_rest_quants': np.linalg.norm(totalGradRQ)})
        totalGradMag = -1.
        if self.full:
            totalGradMag = torch.linalg.norm(self.cshell.PullDesignParametersToCurvesDoF(torch.tensor(totalGradRQ)))
        idata.update({'total_grad_norm': totalGradMag})
        self.iterateData.append(idata)

        pass

    def UpdateViews(self, initialize=False):

        flatLinkage     = self.optimizer.getBaseLinkage()
        deployedLinkage = self.optimizer.getDeployedLinkage()
        tsf             = self.optimizer.get_target_surface_fitter()

        usePercent = True
        useSurfDim = True

        if self.deployedView:
            if self.updateColor:
                if initialize:
                    if self.visDeviations:
                        deployedFieldInit = ScalarFieldDeviations(deployedLinkage, tsf, useSurfDim=useSurfDim, usePercent=usePercent)
                        self.maxDeployedFieldInit = np.max(np.array(deployedFieldInit))
                    else:
                        deployedFieldInit = ScalarField(deployedLinkage, deployedLinkage.sqrtBendingEnergies())
                        self.maxDeployedFieldInit = np.max(deployedFieldInit.data)
                if self.visDeviations:
                    devField = ScalarFieldDeviations(deployedLinkage, tsf, useSurfDim=useSurfDim, usePercent=usePercent)
                    deployedField = ScalarField(deployedLinkage, devField, vmin=0.0, vmax=self.maxDeployedFieldInit)
                else:
                    deployedField = ScalarField(deployedLinkage, deployedLinkage.sqrtBendingEnergies(), vmin=0.0, vmax=self.maxDeployedFieldInit)
                self.deployedView.update(preserveExisting=False, mesh=deployedLinkage, scalarField=deployedField)
            else:
                self.deployedView.update(preserveExisting=False, mesh=deployedLinkage)

        if self.flatView:
            if self.updateColor:
                if self.visDeviations:
                    flatField = deployedField
                else:
                    flatField = ScalarField(flatLinkage, flatLinkage.sqrtBendingEnergies(), vmin=0.0, vmax=self.maxDeployedFieldInit)
                self.flatView.update(preserveExisting=False, mesh=flatLinkage, scalarField=flatField)
            else:
                self.flatView.update(preserveExisting=False, mesh=flatLinkage)

        if self.takeScreenShot:
            self.flatView.setCameraParams(self.screenshot["camParamsFlat"])
            offRend = self.flatView.offscreenRenderer(width=540, height=440)
            offRend.render()
            offRend.save(self.screenshot["pathToSaveFolder"]+"flat_optim_{}.png".format(len(self.iterateData)))

            self.deployedView.setCameraParams(self.screenshot["camParamsDep"])
            offRend = self.deployedView.offscreenRenderer(width=540, height=440)
            offRend.render()
            offRend.save(self.screenshot["pathToSaveFolder"]+"deployed_optim_{}.png".format(len(self.iterateData)))

###########################################################################
#############        VISUALIZATION OF CONVERGENCE PLOTS       #############
###########################################################################

class VisualizationSetting():
    def __init__(self):
        self.cmap = plt.get_cmap("Set2")

        self.plotOptions = {
            "ElasticEnergyFlat"     : {"color":self.cmap(0), "label": 'Elastic Energy Base'},
            "ElasticEnergyDeployed" : {"color":'#555358',    "label": 'Elastic Energy Deployed'},
            "TargetFitting"         : {"color":self.cmap(1), "label": 'Target Surface Fitting'},
            "RestCurvatureSmoothing": {"color":self.cmap(2), "label": 'Curvature Variation'},
            "RestLengthMinimization": {"color":self.cmap(3), "label": 'Rest Length Sum'},
            "LaplacianCP"           : {"color":self.cmap(4), "label": 'Control Points Alignment'},

            "MinAngleConstraint": {"color":self.cmap(5), "label": 'Minimum Angle Constraint'},
            "FlatnessConstraint": {"color":self.cmap(6), "label": 'Flatness Constraint'}
        }

        self.elastic_color = '#555358'
        self.target_color = self.cmap(1)
        self.rest_length_color = self.cmap(2)
        self.smoothness_color = self.cmap(3)
        self.regularization_color = self.cmap(2)
        self.curvature_color = self.cmap(4)
        self.contact_color = self.cmap(5)
        self.separation_color = self.cmap(5)
        self.tangential_color = self.cmap(6)
        self.joint_color = self.cmap(7)

        self.elastic_label = 'Elastic Energy'
        self.target_label = 'Target Surface Fitting'
        self.dist_surf_label = 'Avg. Dist. to Surface'
        self.rest_length_label = 'Rest Length Sum'
        self.smoothness_label = 'Curvature Variation'
        self.curvature_label = 'Curvature Sum'
        self.contact_label = 'Contact Forces'
        self.separation_label = 'Max. Separation Force'
        self.tangential_label = 'Max. Tangential Force'
        self.joint_label = 'Avg. Dist. to Nodes'
        self.regularization_label = 'Regularization'

        self.x_label = 'Iteration'
        self.figure_size = (17, 6)
        self.figure_label_size = 22
        self.figure_title_size = 25

        self.optim_label = 'Design Optimization'

class ConvergencePlotsVisualizer:

    def __init__(self, cshellCB):
        self.cshellCB = cshellCB
        self.vs = VisualizationSetting()
        self.dpsObjective = None

        # Filled with self.UnpackData
        self.dpsObjective = None
        self.dpsGradMag   = None
        self.times        = None

        self.UnpackData()

    def UnpackData(self):
        self.dpsObjective = {
            "ElasticEnergyFlat"     : [],
            "ElasticEnergyDeployed" : [],
            "TargetFitting"         : [],
            "RestCurvatureSmoothing": [],
            "RestLengthMinimization": [],
            "LaplacianCP"           : []
        }

        self.dpsMetric = {
            "TargetDeviationReport" : [],
            "ElasticEnergyFlat"     : [], # Removes the scaling factor in front
            "ElasticEnergyDeployed" : [],
        }

        self.dpsConstraints = {
            "MinAngleConstraint"    : [],
            "FlatnessConstraint"    : []
        }

        self.dpsGradMag = {
            "ElasticEnergyFlat"     : [],
            "ElasticEnergyDeployed" : [],
            "TargetFitting"         : [],
            "RestCurvatureSmoothing": [],
            "RestLengthMinimization": [],
            "LaplacianCP"           : [],

            "Full (rest quantities)": [],
            "Full (curves DoF)"     : [],

            "MinAngleConstraint"    : [],
            "FlatnessConstraint"    : []
        }

        self.times = []

        for iter_idx in range(len(self.cshellCB.iterateData)):
            # Unpack the objectives
            for key in self.dpsObjective.keys():
                if key in self.cshellCB.iterateData[iter_idx].keys():
                    self.dpsObjective[key].append(self.cshellCB.iterateData[iter_idx][key])
            
            # Unpack the constraints
            for key in self.dpsConstraints.keys():
                if key in self.cshellCB.iterateData[iter_idx].keys():
                    self.dpsConstraints[key].append(self.cshellCB.iterateData[iter_idx][key])

            # Unpack the grad magnitudes
            for key in self.dpsGradMag.keys():
                if (key + "_grad_norm") in self.cshellCB.iterateData[iter_idx].keys():
                    self.dpsGradMag[key].append(self.cshellCB.iterateData[iter_idx][key + "_grad_norm"])
            self.dpsGradMag["Full (rest quantities)"].append(self.cshellCB.iterateData[iter_idx]["total_grad_norm_rest_quants"])
            self.dpsGradMag["Full (curves DoF)"].append(self.cshellCB.iterateData[iter_idx]["total_grad_norm"])

            # Unpack the metrics
            # targetObj = 2 * self.cshellCB.iterateData[iter_idx]["TargetFitting"] / (self.cshellCB.optimizer.beta + 1e-16)
            self.dpsMetric["TargetDeviationReport"].append(self.cshellCB.iterateData[iter_idx]["TargetDeviationReport"])
            if self.cshellCB.optimizer.gamma != 0.:
                self.dpsMetric["ElasticEnergyFlat"].append(self.dpsObjective["ElasticEnergyFlat"][iter_idx] * self.cshellCB.optimizer.get_E0() / self.cshellCB.optimizer.gamma)
            else:
                self.dpsMetric["ElasticEnergyFlat"].append(0.)
            if self.cshellCB.optimizer.gamma != 1.:
                self.dpsMetric["ElasticEnergyDeployed"].append(self.dpsObjective["ElasticEnergyDeployed"][iter_idx] * self.cshellCB.optimizer.get_E0() / (1. - self.cshellCB.optimizer.gamma))
            else:
                self.dpsMetric["ElasticEnergyDeployed"].append(0.)
            self.times.append(self.cshellCB.iterateData[iter_idx]["iteration_time"])

        self.cumTimes = list(np.cumsum(self.times))

        # print(self.dpsObjective.keys())
        # print([self.dpsObjective[key] for key in self.dpsObjective.keys()])
        lstObj = []
        for key in self.dpsObjective.keys():
            if key == "LaplacianCP": 
                # In case it is not a tensor
                try: obj = [elt.item() for elt in self.dpsObjective[key]]
                except: obj = [elt for elt in self.dpsObjective[key]]
            else                   : obj = self.dpsObjective[key]
            lstObj.append(obj)
        self.dpsObjective["Total"] = list(np.array(lstObj).sum(axis=0))
        colors = [self.vs.elastic_color, self.vs.regularization_color]
        labels = [self.vs.elastic_label, self.vs.regularization_label]
        return colors, labels

    def PlotObjective(self, figure_name, label, plotAll=False, wrtTime=False):
        fig, host = plt.subplots()

        try:
            x = range(len(self.dpsObjective["Total"]))
        except:
            raise ValueError("Please call UnpackData first!")

        if wrtTime:
            x = self.cumTimes
            plt.xlabel('Time (s)', fontsize=self.vs.figure_label_size)
        else:
            plt.xlabel(self.vs.x_label, fontsize=self.vs.figure_label_size)


        if plotAll:
            colors = [self.vs.plotOptions[key]["color"] for key in self.dpsObjective.keys() if key!="Total"]
            
            lstObj = []
            for key in self.dpsObjective.keys():
                if key == "LaplacianCP": 
                    # In case it is not a tensor
                    try: lstObj.append([elt.item() for elt in self.dpsObjective[key]])
                    except: lstObj.append([elt for elt in self.dpsObjective[key]])
                elif key != "Total": 
                    lstObj.append(self.dpsObjective[key])
            y = np.array(lstObj)
            plt.stackplot(x, y, labels=[self.vs.plotOptions[key]["label"] for key in self.dpsObjective.keys() if key!="Total"], colors=colors)
            plt.legend(loc='upper right', prop={'size': 15}, fancybox=True)
        else:
            y = np.array(self.dpsObjective["Total"])
            plt.plot(x, y)

        fig.set_size_inches(self.vs.figure_size)
        plt.ylabel('Objective Value', fontsize=self.vs.figure_label_size)
        plt.title(label, fontsize=self.vs.figure_title_size)
        fig.set_size_inches(self.vs.figure_size)
        plt.show()
        fig.savefig(figure_name, dpi=200)
        plt.close()

    def PlotMetrics(self, figure_name, label, wrtTime=False):
        fig, host = plt.subplots()

        try:
            x = range(len(self.dpsMetrics["ElasticEnergyFlat"]))
        except:
            raise ValueError("Please call UnpackData first!")

        if wrtTime:
            x = self.cumTimes
            plt.xlabel('Time (s)', fontsize=self.vs.figure_label_size)
        else:
            plt.xlabel(self.vs.x_label, fontsize=self.vs.figure_label_size)

        colors = [self.vs.plotOptions[key]["color"] for key in self.dpsMetrics.keys()]
        y = np.array([self.dpsMetrics[key] for key in self.dpsMetrics.keys()])
        plt.stackplot(x, y, labels=[self.vs.plotOptions[key]["label"] for key in self.dpsMetrics.keys()], colors=colors)
        plt.legend(loc='upper right', prop={'size': 15}, fancybox=True)

        fig.set_size_inches(self.vs.figure_size)
        plt.ylabel('Metrics Values', fontsize=self.vs.figure_label_size)
        plt.title(label, fontsize=self.vs.figure_title_size)
        fig.set_size_inches(self.vs.figure_size)
        plt.show()
        fig.savefig(figure_name, dpi=200)
        plt.close()

    def PlotConstraints(self, figure_name, label, wrtTime=False):

        if not (self.cshellCB.applyAngleConstraint or self.cshellCB.applyFlatnessConstraint):
            print("The optimization did not have constraints.")
            return

        fig, host = plt.subplots()

        try:
            x = range(len(self.dpsObjective["Total"]))
        except:
            raise ValueError("Please call UnpackData first!")

        if wrtTime:
            x = self.cumTimes
            plt.xlabel('Time (s)', fontsize=self.vs.figure_label_size)
        else:
            plt.xlabel(self.vs.x_label, fontsize=self.vs.figure_label_size)

        colors = [self.vs.plotOptions[key]["color"] for key in self.dpsConstraints.keys()]
        y = np.array([self.dpsConstraints[key] for key in self.dpsConstraints.keys() if len(self.dpsGradMag[key]) == len(x)])
        plt.stackplot(x, y, labels=[self.vs.plotOptions[key]["label"] for key in self.dpsConstraints.keys() if len(self.dpsConstraints[key])!=0], colors=colors)
        plt.legend(loc='upper right', prop={'size': 15}, fancybox=True)

        fig.set_size_inches(self.vs.figure_size)
        plt.ylabel('Constraint Value', fontsize=self.vs.figure_label_size)
        plt.title(label, fontsize=self.vs.figure_title_size)
        fig.set_size_inches(self.vs.figure_size)
        plt.show()
        fig.savefig(figure_name, dpi=200)
        plt.close()

    def PlotGradMags(self, figure_name, label, wrtTime=False):

        if not (self.cshellCB.computeGradMags):
            print("The gradient magnitudes have not been computed.")
            return

        
        fig, host = plt.subplots()

        try:
            x = range(len(self.dpsObjective["Total"]))
        except:
            raise ValueError("Please call UnpackData first!")

        if wrtTime:
            x = self.cumTimes
            plt.xlabel('Time (s)', fontsize=self.vs.figure_label_size)
        else:
            plt.xlabel(self.vs.x_label, fontsize=self.vs.figure_label_size)

        colors = [self.vs.plotOptions[key]["color"] for key in self.dpsGradMag.keys()]
        y = np.array([self.dpsGradMag[key] for key in self.dpsGradMag.keys() if len(self.dpsGradMag[key]) == len(x)])
        plt.stackplot(x, y, labels=[self.vs.plotOptions[key]["label"] for key in self.dpsGradMag.keys() if len(self.dpsGradMag[key])!=0], colors=colors)
        plt.legend(loc='upper right', prop={'size': 15}, fancybox=True)

        fig.set_size_inches(self.vs.figure_size)
        plt.ylabel('Gradient Magnitude', fontsize=self.vs.figure_label_size)
        plt.title(label, fontsize=self.vs.figure_title_size)
        fig.set_size_inches(self.vs.figure_size)
        plt.show()
        fig.savefig(figure_name, dpi=200)
        plt.close()
        
# That one is less object oriented, just feed the right dictionaries
def PlotStackedConvergencePlots(dpsObjectives, dpsTimings, listNames, listLabels, 
                                showText=True, againstTime=False, normalizeWithInit=False,
                                filename=None, transparent=False, logscale=False,
                                showGrid=True, removeTicks=False, preserveLegend=False,
                                removeLegendBox=False, removePlot=False):
    '''
    Args:
        dpsObjectives     : a dictionary containing the metrics to stack
        dpsTimings        : a list containing the cumulated time
        listNames         : the term names as they appear in dpsObjectives.keys()
        listLabels        : the associated list of labels to display
        showText          : whether we want to show text
        againstTime       : whether we plot against the timings or the step number
        normalizeWithInit : whether we normalize the data so that the initial cumulated objective equals 100
        filename          : the image's file name in case we want to save the figure
        transparent       : whether we want the background to be transparent
        logscale          : whether we want to use a log scale for the y axis
        showGrid          : whether we want to show the grid or not
        removeTicks       : whether we want to remove ticks or not
        preserveLegend    : whether we still want to show the legend despite showText being False
        removeLegendBox   : whether we want to remove the whole legend box
        removePlot        : whether we want to remove the stackplot, keeping the axes still
    '''
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
    fig = plt.figure(figsize=(16, 6))

    # First energies
    axTmp = plt.subplot(gs[0, 0])

    # Colors handling
    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(listNames))]

    normFactor = 1.0
    if normalizeWithInit:
        normFactor = 100.0 / dpsObjectives["Total"][0]
    xs = np.arange(0, len(dpsTimings))
    if againstTime:
        xs = dpsTimings
    lstObj = []
    for key in listNames:
        if key == "LaplacianCP": 
            lstObj.append([elt.item() for elt in dpsObjectives[key]])
        elif key != "Total": 
            lstObj.append(dpsObjectives[key])
    y = normFactor * np.array(lstObj)
    sp = plt.stackplot(xs, y, labels=listLabels, colors=colors)

    axTmp.set_frame_on(False)

    if logscale: axTmp.set_yscale('log')
    axTmp.set_xlim(0.0, xs[-1])
    
    if showText:
        axTmp.set_title("Objective as optimization goes", fontsize=22)
        if againstTime:
            axTmp.set_xlabel("Time (s)", fontsize=16)
        else:
            axTmp.set_xlabel("Number of steps", fontsize=16)
        if normalizeWithInit:
            axTmp.set_ylabel("Objective values (% of initial objective value)", fontsize=16)
        else:
            axTmp.set_ylabel("Objective values", fontsize=16)
        if not removeLegendBox: axTmp.legend(loc=0, fontsize=18, framealpha=1.0)
    else:
        axTmp.xaxis.set_ticklabels([])
        axTmp.yaxis.set_ticklabels([])
        if not removeLegendBox:
            leg = axTmp.legend(loc=0, fontsize=18, framealpha=1.0)
            for text in leg.get_texts():
                text.set_color("white")
                if preserveLegend: text.set_color("black")
            for handle in leg.legendHandles:
                handle.set_height(18.0) # hard coded for the figure dimensions
                handle.set_y(handle.get_y() - 4.5)
    if showGrid: axTmp.grid()
    if removePlot: 
        for artist in sp:
            artist.remove()
    if removeTicks:
        axTmp.xaxis.set_ticks_position('none') 
        axTmp.yaxis.set_ticks_position('none') 
    axTmp.set_axisbelow(True)
    
    if not filename is None: plt.savefig(filename, transparent=transparent, dpi=300)
    plt.show()
    
def CompareStatistics(listMetrics, listNames, metricName, colors=None, xlim=None, 
                      minBounds=None, maxBounds=None, useMedian=False, 
                      title=None, xlabel=None,
                      filename=None, showText=True):
    '''
    Args:
        listMetrics : a list of dictionnaries that contain the metric name associated with the metric evolution
        listNames   : a list of names corresponding to each list of metrics
        metricName  : the name of the metric to study
        colors      : a list of colors for each deployments
        xlim        : [xMin, xMax] used for all plots
        minBounds   : a dictionnary containing the keys in stressList and giving the yMin
        maxBounds   : a dictionnary containing the keys in stressList and giving the yMax
        useMedian   : if true show median and 5-95 percentile region, else show mean and an area with height equal to the std
        title       : string or None, title of the plot
        xlabel      : string or None, xlabel used
        filename    : the path to where we want to save the figure
        showText    : whether we show text or not
    '''
    assert metricName in listMetrics[0].keys()
    
    listStatistics = []

    for metrics in listMetrics:
        metricsArr = np.array(metrics[metricName])
        listStatistics.append({
            "mean"      : np.mean(metricsArr, axis=1),
            "median"    : np.median(metricsArr, axis=1),
            "std"       : np.std(metricsArr, axis=1),
            "95"        : np.percentile(metricsArr, 95, axis=1),
            "5"         : np.percentile(metricsArr, 5, axis=1),
            "min"       : np.min(metricsArr, axis=1),
            "max"       : np.max(metricsArr, axis=1),
        })

    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
    fig = plt.figure(figsize=(8, 6))

    xs = np.arange(0, metricsArr.shape[0])
    
    # Colors handling
    if (colors is None):
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(len(listMetrics))]
    
    axTmp = plt.subplot(gs[0, 0])
    for stats, name, color in zip(listStatistics, listNames, colors):
        if useMedian:
            median    = stats["median"]
            lowerPerc = stats["5"]
            upperPerc = stats["95"]
            axTmp.plot(xs, median, '-', color=color, label=name, linewidth=3.5)
            axTmp.fill_between(xs, lowerPerc, upperPerc, color=color, alpha=0.3)
        else:
            mean     = stats["mean"]
            std      = stats["std"]
            axTmp.plot(xs, mean, '-', color=color, label=name, linewidth=3.5)
            axTmp.fill_between(xs, mean - std / 2, mean + std / 2, color=color, alpha=0.3)
        if showText:
            if title is None:
                axTmp.set_title("{} during optimization".format(metricName), fontsize=14)
            else:
                axTmp.set_title(title, fontsize=14)
            if xlabel is None:
                axTmp.set_xlabel("Optimization step", fontsize=12)
            else:
                axTmp.set_xlabel(xlabel, fontsize=12)
            axTmp.set_ylabel(metricName, fontsize=12)
            axTmp.legend(loc=0, fontsize=12)
        else:
            axTmp.xaxis.set_ticklabels([])
            axTmp.yaxis.set_ticklabels([])
        if not xlim is None: axTmp.set_xlim(xlim)
        if (not minBounds is None) and (not maxBounds is None): axTmp.set_ylim([minBounds[metricName], maxBounds[metricName]])
        axTmp.grid()
        axTmp.set_axisbelow(True)
    
    if not filename is None: plt.savefig(filename)
    plt.show()

###########################################################################
#############         VISUALIZATION OF DISSIMILARITIES        #############
###########################################################################

def PlotDissimilarities(dissims, dissimsAll, title=None):
    '''
    Args:
        dissims    : dictionnary of tensors of shape (?,) containing the dissimilarity for each term
        dissimsAll : a torch tensor of shape (?, nCrits) contrainin the dissimilarities for each criterion for each direction
    '''

    def GetCumulatedTensor(data, **kwargs):
        cum = data.clip(**kwargs)
        cum = torch.cumsum(cum, dim=1)
        d = torch.zeros(data.shape)
        d[:, 1:] = cum[:, :-1]
        return d  

    cumulatedDissims    = GetCumulatedTensor(dissimsAll, min=0)
    cumulatedDissimsNeg = GetCumulatedTensor(dissimsAll, max=0)

    # Re-merge negative and positive data.
    rowMask = (dissimsAll < 0)
    cumulatedDissims[rowMask] = cumulatedDissimsNeg[rowMask]
    dataStack = cumulatedDissims

    vs = VisualizationSetting()
    cols = [vs.plotOptions[key]["color"] for key in dissims.keys()]
    labels = [vs.plotOptions[key]["label"] for key in dissims.keys()]

    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
    fig = plt.figure(figsize=vs.figure_size)

    ax = plt.subplot(gs[0, 0])

    for i in range(dissimsAll.shape[1]):
        ax.bar(np.arange(dissimsAll.shape[0]), dissimsAll[:, i], bottom=dataStack[:, i], 
               color=cols[i], label=labels[i])

    if not title is None:
        ax.set_title(title, fontsize=vs.figure_title_size)
    ax.set_xlabel("Modification ID", fontsize=vs.figure_label_size)
    ax.set_ylabel("Cumulated dissimilarity", fontsize=vs.figure_label_size)
    ax.legend(fontsize=15, loc="lower right")
    ax.legend(fontsize=15, loc="lower right")
    ax.set_xticks(np.arange(dissimsAll.shape[0]))
    if dissimsAll.shape[0] <= 6:
        center = (dissimsAll.shape[0] - 1) / 2
        ax.set_xlim(center - 4.0, center + 4.0)

    ax.set_axisbelow(True)
    ax.grid(color="gray", linestyle="dashed", alpha=0.3)

    plt.show()

def PlotStatisticsGradient(metrics, metricName, xs=None, cmap=None, xlim=None, 
                      minBounds=None, maxBounds=None, useMedian=False, 
                      title=None, xlabel=None,
                      filename=None, showText=True):
    '''
    Args:
        metrics     : a dictionnary that contain the metric name associated with the metric evolution
        metricName  : the name of the metric to study
        xs          : data on the x-axis
        cmap        : the colour map to use
        xlim        : [xMin, xMax] used for all plots
        minBounds   : a dictionnary containing the keys in stressList and giving the yMin
        maxBounds   : a dictionnary containing the keys in stressList and giving the yMax
        useMedian   : if true show median and 5-95 percentile region, else show mean and an area with height equal to the std
        title       : string or None, title of the plot
        xlabel      : string or None, xlabel used
        filename    : the path to where we want to save the figure
        showText    : whether we show text or not
    '''
    assert metricName in metrics.keys()

    metricsArr = np.array(metrics[metricName])
    stats = {
        "mean"      : np.mean(metricsArr, axis=1),
        "median"    : np.median(metricsArr, axis=1),
        "std"       : np.std(metricsArr, axis=1),
        "95"        : np.percentile(metricsArr, 95, axis=1),
        "5"         : np.percentile(metricsArr, 5, axis=1),
        "min"       : np.min(metricsArr, axis=1),
        "max"       : np.max(metricsArr, axis=1),
    }

    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
    fig = plt.figure(figsize=(8, 6))

    if xs is None:
        xs = np.arange(0, metricsArr.shape[0])
    
    axTmp = plt.subplot(gs[0, 0])

    if useMedian:
        mid   = stats["median"]
        lower = stats["5"]
        upper = stats["95"]
    else:
        mid     = stats["mean"]
        std     = stats["std"]
        lower = mid - std / 2
        upper = mid + std / 2

    points = np.array([xs, mid]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(np.linspace(0.0, 1.0, segments.shape[0]))
    lc.set_linewidth(4.5)
    axTmp.add_collection(lc)
    polygon = axTmp.fill_between(xs, lower, upper, lw=0, color='none')
    xlim = axTmp.get_xlim()
    ylim = axTmp.get_ylim()
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    gradient = axTmp.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap, aspect='auto', alpha=0.3,
                            extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
    axTmp.set_xlim(xlim)
    axTmp.set_ylim(ylim)

    if showText:
        if title is None:
            axTmp.set_title("{} during optimization".format(metricName), fontsize=14)
        else:
            axTmp.set_title(title, fontsize=14)
        if xlabel is None:
            axTmp.set_xlabel("Optimization step", fontsize=12)
        else:
            axTmp.set_xlabel(xlabel, fontsize=12)
        axTmp.set_ylabel(metricName, fontsize=12)
    else:
        axTmp.xaxis.set_ticklabels([])
        axTmp.yaxis.set_ticklabels([])
    if not xlim is None: axTmp.set_xlim(xlim)
    ylim = list(axTmp.get_ylim())
    if (not minBounds is None): ylim[0] = minBounds
    if (not maxBounds is None): ylim[1] = maxBounds
    axTmp.set_ylim(ylim)
    axTmp.grid()
    axTmp.set_axisbelow(True)
    
    if not filename is None: plt.savefig(filename)
    plt.show()
    

###########################################################################
#############                 MISC FUNCTIONS                  #############
###########################################################################
    
def PlotHist(data1, bins,
             col1="tab:blue", showText=True,
             fn=None, xTicks=None):
    """Plots the histograms of two sets of data upside down.
    
    Args:
        data1: np array of shape (nData1,)
        bins: np array of shape (nBins+1,)
        col1: the color used to represent data1
        col2: the color used to represent data2
        showText: whether we want to show text on the plot
        fn: filename of the plot
        xTicks: ticks along the x axis
    """
    
    hist1 = np.histogram(data1, bins=bins)
    maxCount = np.max(hist1[0])
    ylim = [-0.1 * maxCount, 1.1 * maxCount]
    
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
    fig = plt.figure(figsize=(10, 8))
    axTmp = plt.subplot(gs[0, 0])
    axTmp.hist(data1, bins=bins, alpha=1.0, color=col1)
    axTmp.set_xlim(0.0, np.pi)
    axTmp.set_ylim(ylim)
    
    if showText:
        # Have integer values on the y-axis
        maxYTicks = int(maxCount - maxCount % 3)
        incrementYTicks = int(maxYTicks // 3)
        yTicks = [incrementYTicks * j for j in range(4)]
        axTmp.set_yticks(yTicks)
        ticks =  axTmp.get_yticks()
        axTmp.set_yticklabels([int(abs(tick)) for tick in ticks])
    else:
        axTmp.set_yticklabels([])
    
    # Have the x-axis in the middle
    axTmp.set_yticks([])
    axTmp.spines['left'].set_visible(False)
    axTmp.spines['top'].set_visible(False)
    axTmp.spines['right'].set_visible(False)
    axTmp.spines['bottom'].set_position('zero')
    axTmp.spines['bottom'].set_linewidth(5.0)
    
    if xTicks is not None:
        axTmp.set_xticks(xTicks)
    axTmp.tick_params(direction='inout', width=5.0, length=25.0)
    if not showText:
        axTmp.set_xticklabels([])
    
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', transparent=True)
    plt.show()



###########################################################################
#############               UTILITARY FUNCTIONS               #############
###########################################################################


def get_component_gradient_norm(optimizer, currType, cshell, full):
    '''Change the weight so that we only get one gradient at a time

    Args:
        optimizer : a linkageOptimizer
        currType  : the current objective term we want to evaluate the gradient of
        cshell    : a FreeCShell instance
        full      : whether we run the full optimization (on the curves DoF) or not (just the rest quantities)
    '''
    save_beta = optimizer.beta
    save_gamma = optimizer.gamma
    save_smoothing_weight = optimizer.smoothing_weight
    save_rl_regularization_weight = optimizer.rl_regularization_weight

    optimizer.beta = 0.0
    optimizer.gamma = 0.0
    optimizer.smoothing_weight = 0.0
    optimizer.rl_regularization_weight = 0.0
    if currType == linkage_optimization.OptEnergyType.Full:
        optimizer.beta = save_beta
        optimizer.gamma = save_gamma
        optimizer.smoothing_weight = save_smoothing_weight
        optimizer.rl_regularization_weight = save_rl_regularization_weight
    elif currType == linkage_optimization.OptEnergyType.Target:
        optimizer.beta = save_beta
    elif currType == linkage_optimization.OptEnergyType.Smoothing:
        optimizer.smoothing_weight = save_smoothing_weight
    elif currType == linkage_optimization.OptEnergyType.Regularization:
        optimizer.rl_regularization_weight = save_rl_regularization_weight
    elif currType == linkage_optimization.OptEnergyType.ElasticBase:
        optimizer.gamma = save_gamma
    elif currType == linkage_optimization.OptEnergyType.ElasticDeployed:
        optimizer.gamma = save_gamma

    optimizer.invalidateAdjointState()
    gradp = optimizer.gradp_J(optimizer.getFullDesignParameters(), currType)
    if full:
        gradNorm = torch.linalg.norm(cshell.PullDesignParametersToCurvesDoF(torch.tensor(gradp)))
    else:
        gradNorm = np.linalg.norm(gradp)

    optimizer.beta = save_beta
    optimizer.gamma = save_gamma
    optimizer.smoothing_weight = save_smoothing_weight
    optimizer.rl_regularization_weight = save_rl_regularization_weight
    optimizer.invalidateAdjointState()
    return gradNorm
