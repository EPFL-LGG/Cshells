import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from open_average_angle_linkage import ComputeStressesHistogram

matplotlib.rcParams['savefig.dpi'] = 300

def CompareDeploymentQuantities(listDeployments, listNames, colors=None, xlimEnergies=None, ylimEnergies=None, 
                                xlimTorque=None, ylimTorque=None, addStartDot=None, addEndDot=None, 
                                showTorque=True, showAnnotations=True, showText=True, filename=None,):
    '''
    Args:
        listDeployments : a list of dictionnaries that are given by RunAndAnalyzeDeployment in open_average_angle_linkage.py
        listNames       : a list of names corresponding to each deployment
        colors          : a list of colors for each deployments
        xlimEnergies    : [xMinEnergy, xMaxEnergy]
        ylimEnergies    : [yMinEnergy, yMaxEnergy]
        xlimTorque      : [xMinTorque, xMaxTorque]
        ylimTorque      : [yMinTorque, yMaxTorque]
        addStartDot     : list of bools indicating whether or not we indicate the first state
        addEndDot       : list of bools indicating whether or not we indicate the last state
        showTorque      : whether we show the actuation torque during deployment
        showAnnotations : whether we show annotation or not
        showText        : whether we show text or not
        filename        : the path to where we want to save the figure
    '''

    widthRatios = ([1, 0.05, 1] if showTorque else [1])
    gs = gridspec.GridSpec(nrows=1, ncols=1+2*showTorque, height_ratios=[1], width_ratios=widthRatios)
    fig = plt.figure(figsize=((1+showTorque)*8, 6))
    
    if addStartDot is None: addStartDot = len(listDeployments) * [False]
    if addEndDot is None:   addEndDot   = len(listDeployments) * [False]

    # First energies
    axTmp = plt.subplot(gs[0, 0])

    # Colors handling
    if (colors is None):
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(len(listDeployments))]

    t = "SurfaceAttractionEnergyType.Elastic"
    for i, (dep, name, color) in enumerate(zip(listDeployments, listNames, colors)):
        if "Surface" in list(dep["finalEnergies"].keys())[0]: t = "SurfaceAttractionEnergyType.Elastic"
        else: t = "EnergyType.Full"
        
        alphas = dep["averageAngles"]

        if name == "Table Trick":
            maskDotted = np.array(dep["averageAngles"]) <= 1.42
            axTmp.plot(np.array(dep["averageAngles"])[maskDotted], np.array(dep["finalEnergies"][t])[maskDotted], label=r"{}".format(name), linestyle="--", color=color, linewidth=3.5)
            axTmp.plot(np.array(dep["averageAngles"])[~maskDotted], np.array(dep["finalEnergies"][t])[~maskDotted], label=r"{}".format(name), linestyle="-", color=color, linewidth=3.5)
        else:
            axTmp.plot(alphas, dep["finalEnergies"][t], label=r"{}".format(name), linestyle="-", color=color, linewidth=3.5)
        if addStartDot[i]:
            axTmp.scatter([alphas[0]], [dep["finalEnergies"][t][0]], color=color, s=130)
        if addEndDot[i]:
            axTmp.scatter([alphas[-1]], [dep["finalEnergies"][t][-1]], color=color, s=130)
        # axTmp.plot(alphas, dep["finalEnergies"][t], label="Elastic" + " ({})".format(name), linestyle="-", color=color)
    
    if not xlimEnergies is None: axTmp.set_xlim(xlimEnergies[0], xlimEnergies[1])
    if not ylimEnergies is None: axTmp.set_ylim(ylimEnergies[0], ylimEnergies[1])

    xlim = axTmp.get_xlim()
    ylim = axTmp.get_ylim()
    if showAnnotations:
        for i, (dep, name, color) in enumerate(zip(listDeployments, listNames, colors)):
            alphas = dep["averageAngles"]
            axTmp.annotate('Undeployed '+name, xy=(alphas[0], dep["finalEnergies"][t][0]), 
                        xytext=(alphas[0]+0.01 * (xlim[1] - xlim[0]), dep["finalEnergies"][t][0] + (0.05+0.05*i) * (ylim[1] - ylim[0])),
                        arrowprops=dict(arrowstyle="-", facecolor='black', connectionstyle="arc3,rad=0.3"))
    if showText:
        axTmp.set_title("Energies as deployment goes", fontsize=14)
        axTmp.set_xlabel("Average opening angle", fontsize=12)
        axTmp.set_ylabel("Energy values", fontsize=12)
        axTmp.legend(loc=2, fontsize=12)
    else:
        axTmp.xaxis.set_ticklabels([])
        axTmp.yaxis.set_ticklabels([])
    axTmp.grid()
    axTmp.set_axisbelow(True)
    
    # Second torques
    if showTorque:
        axTmp = plt.subplot(gs[0, -1])
        for dep, name, color in zip(listDeployments, listNames, colors):
            alphas = dep["averageAngles"]
            axTmp.plot(alphas, dep["actuationTorque"], label=name, linestyle="-", color=color, linewidth=3.5)
            if addStartDot[i]:
                axTmp.scatter([alphas[0]], [dep["actuationTorque"][0]], color=color, s=130)
            if addEndDot[i]:
                axTmp.scatter([alphas[-1]], [dep["actuationTorque"][-1]], color=color, s=130)
        if not xlimTorque is None: axTmp.set_xlim(xlimTorque[0], xlimTorque[1])
        if not ylimTorque is None: axTmp.set_ylim(ylimTorque[0], ylimTorque[1])
        xlim = axTmp.get_xlim()
        ylim = axTmp.get_ylim()
        if showAnnotations:
            for i, (dep, name) in enumerate(zip(listDeployments, listNames)):
                alphas = dep["averageAngles"]
                axTmp.annotate('Undeployed '+name, xy=(alphas[0], dep["actuationTorque"][0]), 
                            xytext=(alphas[0]+(0.01) * (xlim[1] - xlim[0]), dep["actuationTorque"][0] + (0.05+0.05*i) * (ylim[1] - ylim[0])),
                            arrowprops=dict(arrowstyle="-", facecolor='black', connectionstyle="arc3,rad=0.3"))
        if showText:
            axTmp.set_title("Actuation torque as deployment goes", fontsize=14)
            axTmp.set_xlabel("Average opening angle", fontsize=12)
            axTmp.set_ylabel("Torque [N.m]", fontsize=12)
            axTmp.legend(loc=2, fontsize=12)
        else:
            axTmp.xaxis.set_ticklabels([])
            axTmp.yaxis.set_ticklabels([])
        axTmp.grid()
        axTmp.set_axisbelow(True)

    if not filename is None: plt.savefig(filename)
    plt.show()

def CompareStressHistsQuantities(depXShell, depCShell, nameFirstHist="XShell", nameSecondHist="CShell", showMeanStd=False, 
                                 minBounds=None, maxBounds=None, minCounts=None, maxCounts=None):
    '''
    Args:
        depXShell      : a dictionnary given by RunAndAnalyzeDeployment in open_average_angle_linkage.py
        depCShell      : a dictionnary given by RunAndAnalyzeDeployment in open_average_angle_linkage.py
        nameFirstHist  : a name in case we want to compare something else than an XShell and a CShell
        nameSecondHist : a name in case we want to compare something else than an XShell and a CShell
        showMeanStd    : whether we want to overlay the mean and std on the histograms
        minBounds      : a dictionnary containing the keys in stressList and giving the yMin
        maxBounds      : a dictionnary containing the keys in stressList and giving the yMax
        minCounts      : a number that gives the min value for scaling the histograms
        maxCounts      : a number that gives the max value for scaling the histograms
    '''
    # stressList = depXShell["stresses"].keys()
    stressList = ["Max von Mises Stress", "Stretching Stress", "Sqrt Bending Energy", "Twisting Stress"]
    
    if (minBounds is None) or (maxBounds is None):
        minBounds = {}
        maxBounds = {}
        for t in depXShell["stresses"].keys():
            stressXShellArr = np.array(depXShell["stresses"][t])
            stressCShellArr = np.array(depCShell["stresses"][t])
            minBounds[t] = min(np.min(stressXShellArr), np.min(stressCShellArr))
            maxBounds[t] = max(np.max(stressXShellArr), np.max(stressCShellArr))
        
    xShellHist = ComputeStressesHistogram(depXShell["stresses"], minBounds=minBounds, maxBounds=maxBounds)
    cShellHist = ComputeStressesHistogram(depCShell["stresses"], minBounds=minBounds, maxBounds=maxBounds)
    
    if (minCounts is None) or (maxCounts is None):
        minCounts = {}
        maxCounts = {}
        for t in xShellHist.keys():
            minCounts[t] = max(1, min(np.min(xShellHist[t]["totalHist"]), np.min(cShellHist[t]["totalHist"])))
            maxCounts[t] = max(1, np.max(xShellHist[t]["totalHist"]), np.max(cShellHist[t]["totalHist"]))
    
    from copy import copy
    from matplotlib.colors import LogNorm
    
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    
    nrows = 2 * len(stressList) - 1
    height_ratios = (len(stressList) - 1) * [1, 0.05] + [1]
    gs = gridspec.GridSpec(nrows=nrows, ncols=5, height_ratios=height_ratios, width_ratios=[1, 0.05, 1, 0.05, 1])
    fig = plt.figure(figsize=(20, len(stressList) * 5))
    
    
    locsXShell = [[2 * i, 0] for i in range(len(stressList))]
    locsCShell = [[2 * i, 2] for i in range(len(stressList))]
    
    xedgesXShell = depXShell["averageAngles"] + [2 * depXShell["averageAngles"][-1] - depXShell["averageAngles"][-3]]
    xmeansXShell = depXShell["averageAngles"][:-1] + [2 * depXShell["averageAngles"][-1] - depXShell["averageAngles"][-3]]
    
    xedgesCShell = depCShell["averageAngles"] + [2 * depCShell["averageAngles"][-1] - depCShell["averageAngles"][-3]]
    xmeansCShell = depCShell["averageAngles"][:-1] + [2 * depCShell["averageAngles"][-1] - depCShell["averageAngles"][-3]]
    
    for (locXShell, locCShell, t) in zip(locsXShell, locsCShell, stressList):
        axTmp = plt.subplot(gs[locXShell[0], locXShell[1]])
        currHist = xShellHist[t]
        pcm = axTmp.pcolormesh(xedgesXShell, currHist["yedges"], currHist["totalHist"].T, cmap=cmap, norm=LogNorm(vmin=minCounts[t], vmax=maxCounts[t]), rasterized=True)
        if showMeanStd:
            axTmp.plot(xmeansXShell, currHist["mean"], '-', color='grey')
            axTmp.fill_between(xmeansXShell, currHist["mean"] - currHist["std"], currHist["mean"] + currHist["std"], color='gray', alpha=0.3)
        fig.colorbar(pcm, ax=axTmp, label="Number of points")
        axTmp.set_title(t + " as deployment goes ({})".format(nameFirstHist), fontsize=14)
        axTmp.set_xlabel(r"Step (from $\alpha_t^i$={:.3f} to $\alpha_t^f$={:.3f})".format(depXShell["averageAngles"][0], depXShell["averageAngles"][-1]), fontsize=12)
        axTmp.set_ylabel(t, fontsize=12)
        
        axTmp = plt.subplot(gs[locCShell[0], locCShell[1]])
        currHist = cShellHist[t]
        pcm = axTmp.pcolormesh(xedgesCShell, currHist["yedges"], currHist["totalHist"].T, cmap=cmap, norm=LogNorm(vmin=minCounts[t], vmax=maxCounts[t]), rasterized=True)
        if showMeanStd:
            axTmp.plot(xmeansCShell, currHist["mean"], '-', color='grey')
            axTmp.fill_between(xmeansCShell, currHist["mean"] - currHist["std"], currHist["mean"] + currHist["std"], color='gray', alpha=0.3)
        fig.colorbar(pcm, ax=axTmp, label="Number of points")
        axTmp.set_title(t + " as deployment goes ({})".format(nameSecondHist), fontsize=14)
        axTmp.set_xlabel(r"Step (from $\alpha_t^i$={:.3f} to $\alpha_t^f$={:.3f})".format(depCShell["averageAngles"][0], depCShell["averageAngles"][-1]), fontsize=12)
        axTmp.set_ylabel(t, fontsize=12)

    plt.show()
    
    return minBounds, maxBounds, minCounts, maxCounts
    
def CompareDeploymentStatistics(listDeployments, listNames, colors=None, xlim=None, 
                                minBounds=None, maxBounds=None, useMedian=False, 
                                filename=None, showText=True, vmOnly=False):
    '''
    Args:
        listDeployments : a list of dictionnaries that are given by RunAndAnalyzeDeployment in open_average_angle_linkage.py
        listNames       : a list of names corresponding to each deployment
        colors          : a list of colors for each deployments
        xlim            : [xMin, xMax] used for all plots
        minBounds       : a dictionnary containing the keys in stressList and giving the yMin
        maxBounds       : a dictionnary containing the keys in stressList and giving the yMax
        useMedian       : if true show median and 5-95 percentile region, else show mean and an area with height equal to the std
        filename        : the path to where we want to save the figure
        showText        : whether we show text or not
    '''
    if vmOnly:
        stressList = ["Max von Mises Stress"]
    else:
        stressList = ["Max von Mises Stress", "Stretching Stress", "Sqrt Bending Energy", "Twisting Stress"]
    
    listHists = [ComputeStressesHistogram(dep["stresses"]) for dep in listDeployments]

    nrows = 2 * len(stressList) - 1
    height_ratios = (len(stressList) - 1) * [1, 0.05] + [1]
    if vmOnly:
        gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
        fig = plt.figure(figsize=(8, 6))
    else:
        gs = gridspec.GridSpec(nrows=nrows, ncols=5, height_ratios=height_ratios, width_ratios=[1, 0.05, 1, 0.05, 1])
        fig = plt.figure(figsize=(20, len(stressList) * 5))
    locations = [[2 * (i // 2), 2 * (i % 2)] for i in range(len(stressList))]
    
    # Create a plot for each energy
    listXMeans = [dep["averageAngles"][:-1] for dep in listDeployments]
    
    # Colors handling
    if (colors is None):
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(len(listDeployments))]
    
    for (loc, t) in zip(locations, stressList):
        axTmp = plt.subplot(gs[loc[0], loc[1]])
        for hists, xmeans, name, color in zip(listHists, listXMeans, listNames, colors):
            currHist = hists[t]
            if useMedian:
                median    = currHist["median"][:-1]
                lowerPerc = currHist["5"][:-1]
                upperPerc = currHist["95"][:-1]
                if vmOnly:
                    axTmp.plot(xmeans, median, '-', color=color, label=name, linewidth=3.5)
                else:
                    axTmp.plot(xmeans, median, '-', color=color, label=name, linewidth=2.5)
                axTmp.fill_between(xmeans, lowerPerc, upperPerc, color=color, alpha=0.3)
            else:
                mean     = currHist["mean"][:-1]
                std      = currHist["std"][:-1]
                if vmOnly:
                    axTmp.plot(xmeans, mean, '-', color=color, label=name, linewidth=3.5)
                else:
                    axTmp.plot(xmeans, mean, '-', color=color, label=name, linewidth=2.5)
                axTmp.fill_between(xmeans, mean - std / 2, mean + std / 2, color=color, alpha=0.3)
        if showText:
            axTmp.set_title(t + " as deployment goes", fontsize=14)
            axTmp.set_xlabel("Average opening angle", fontsize=12)
            axTmp.set_ylabel(t, fontsize=12)
            axTmp.legend(loc=2, fontsize=12)
        else:
            axTmp.xaxis.set_ticklabels([])
            axTmp.yaxis.set_ticklabels([])
        if not xlim is None: axTmp.set_xlim(xlim)
        if (not minBounds is None) and (not maxBounds is None): axTmp.set_ylim([minBounds[t], maxBounds[t]])
        axTmp.grid()
        axTmp.set_axisbelow(True)
    
    if not filename is None: plt.savefig(filename)
    plt.show()

def CompareTargetFittingStatistics(listDeviations, listNames, colors=None, ylim=None, 
                                   filename=None, showText=True):
    '''
    Args:
        listDeployments : a list of arrays giving the data for each model
        listNames       : a list of names corresponding to each model
        colors          : a list of colors for each deployments
        ylim            : [yMin, yMax] used for all plots
        filename        : the path to where we want to save the figure
        showText        : whether we show text or not
    '''
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
    fig = plt.figure(figsize=(10, 8))

    axTmp = plt.subplot(gs[0, 0])
    parts = axTmp.violinplot(
            listDeviations, showmeans=False, showmedians=False,
            showextrema=False)
    
    if colors is None:
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(len(listDeviations))]

    for c, pc in zip(colors, parts['bodies']):
        # pc.set_facecolor('#8797B2')
        pc.set_facecolor(c)
        # pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    perc5, quartile1, med, quartile3, perc95 = np.percentile(listDeviations, [5, 25, 50, 75, 95], axis=1)
    inds = np.arange(1, len(quartile1) + 1)
    axTmp.scatter(inds, med, marker='o', color='white', s=90, zorder=3)
    axTmp.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=13)
    axTmp.vlines(inds, perc5, perc95, color='k', linestyle='-', lw=3.5)

    axTmp.xaxis.set_tick_params(direction='out')
    axTmp.xaxis.set_ticks_position('bottom')
    axTmp.set_xticks(np.arange(1, len(listNames) + 1))
    if showText:
        axTmp.set_title("Target deviations measured for each model", fontsize=14)
        axTmp.set_xlabel("Models", fontsize=12)
        axTmp.set_ylabel("Target deviation (% of the surface's bounding box diagonal)", fontsize=12)
        axTmp.set_xticklabels(listNames)
    else:
        axTmp.xaxis.set_ticks([])
        axTmp.yaxis.set_ticklabels([])
    axTmp.set_xlim(0.25, len(listNames) + 0.75)
    axTmp.grid(axis="y")
    axTmp.set_axisbelow(True)
    if not ylim is None: axTmp.set_ylim(ylim)
    if not filename is None: plt.savefig(filename)

    plt.show()

def CompareModes(listDeployments, listNames, colors=None, xlim=None, ylim=None, showText=True, filename=None,):
    '''
    Args:
        listDeployments : a list of dictionnaries that are given by RunAndAnalyzeDeployment in open_average_angle_linkage.py
        listNames       : a list of names corresponding to each deployment
        colors          : a list of colors for each deployments
        xlim            : [xMin, xMax]
        ylim            : [yMin, yMax]
        showText        : whether we show text or not
        filename        : the path to where we want to save the figure
    '''

    widthRatios = [1]
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=widthRatios)
    fig = plt.figure(figsize=(8, 6))

    # Remove the pinned degrees of freedom
    listModes    = []
    listModesGap = []
    for dep in listDeployments:
        idx = np.argsort(abs(np.array(dep["modes"])))[:, 6:]
        smallestModes = np.take_along_axis(np.array(dep["modes"]), idx, axis=1)
        listModes.append(smallestModes)
        listModesGap.append(smallestModes[:, 1] - smallestModes[:, 0])

    axTmp = plt.subplot(gs[0, 0])

    # Colors handling
    if (colors is None):
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(len(listDeployments))]

    for dep, modes, gaps, name, color in zip(listDeployments, listModes, listModesGap, listNames, colors):
        axTmp.plot(dep["averageAngles"], gaps, label=name, linestyle="-", color=color, linewidth=3.5)
        # axTmp.plot(dep["averageAngles"], modes[:, 0], label=name, linestyle="-", color=color, linewidth=3.5)
    if not xlim is None: axTmp.set_xlim(xlim[0], xlim[1])
    if not ylim is None: axTmp.set_ylim(ylim[0], ylim[1])
    xlim = axTmp.get_xlim()
    ylim = axTmp.get_ylim()
    if showText:
        axTmp.set_title("Modes gaps as deployment goes", fontsize=14)
        axTmp.set_xlabel("Average opening angle", fontsize=12)
        axTmp.set_ylabel("Modes gaps", fontsize=12)
        # axTmp.set_title("Smallest mode as deployment goes", fontsize=14)
        # axTmp.set_xlabel("Average opening angle", fontsize=12)
        # axTmp.set_ylabel("Smallest mode", fontsize=12)
        axTmp.legend(loc=2, fontsize=12)
    else:
        axTmp.xaxis.set_ticklabels([])
        axTmp.yaxis.set_ticklabels([])
    axTmp.grid()
    axTmp.set_axisbelow(True)

    if not filename is None: plt.savefig(filename)
    plt.show()

def CompareStiffnessGap(listDeployments, listNames, colors=None, xlim=None, ylim=None, showText=True, filename=None,):
    '''
    Args:
        listDeployments : a list of dictionnaries that are given by RunAndAnalyzeDeployment in open_average_angle_linkage.py
        listNames       : a list of names corresponding to each deployment
        colors          : a list of colors for each deployments
        xlim            : [xMin, xMax]
        ylim            : [yMin, yMax]
        showText        : whether we show text or not
        filename        : the path to where we want to save the figure
    '''

    widthRatios = [1]
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=widthRatios)
    fig = plt.figure(figsize=(8, 6))

    axTmp = plt.subplot(gs[0, 0])

    # Colors handling
    if (colors is None):
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(len(listDeployments))]

    for dep, name, color in zip(listDeployments, listNames, colors):
        axTmp.plot(dep["averageAngles"], 100. * np.array(dep["stiffnessGap"]), label=name, linestyle="-", color=color, linewidth=3.5)
    if not xlim is None: axTmp.set_xlim(xlim[0], xlim[1])
    if not ylim is None: axTmp.set_ylim(ylim[0], ylim[1])
    xlim = axTmp.get_xlim()
    ylim = axTmp.get_ylim()
    if showText:
        axTmp.set_title("Stiffness gaps as deployment goes", fontsize=14)
        axTmp.set_xlabel("Average opening angle", fontsize=12)
        axTmp.set_ylabel("Stiffness gaps (%)", fontsize=12)
        axTmp.legend(loc=2, fontsize=12)
    else:
        axTmp.xaxis.set_ticklabels([])
        axTmp.yaxis.set_ticklabels([])
    axTmp.grid()
    axTmp.set_axisbelow(True)

    if not filename is None: plt.savefig(filename)
    plt.show()

def CompareStiffnesses(listDeployments, listNames, colors=None, xlim=None, ylim=None, showText=True, filename=None,):
    '''
    Args:
        listDeployments : a list of dictionnaries that are given by RunAndAnalyzeDeployment in open_average_angle_linkage.py
        listNames       : a list of names corresponding to each deployment
        colors          : a list of colors for each deployments
        xlim            : [xMin, xMax]
        ylim            : [yMin, yMax]
        showText        : whether we show text or not
        filename        : the path to where we want to save the figure
    '''

    nModes = 4
    widthRatios = [1]
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=widthRatios)
    fig = plt.figure(figsize=(8, 6))

    axTmp = plt.subplot(gs[0, 0])

    # Colors handling
    if (colors is None):
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(len(listDeployments))]

    for dep, name, color in zip(listDeployments, listNames, colors):
        stiff = np.array(dep["stiffnesses"])
        axTmp.plot(dep["averageAngles"], np.std(stiff[:, :nModes], axis=1), label=name, linestyle="-", color=color, linewidth=3.5)
        # axTmp.plot(dep["averageAngles"], (stiff[:, 1] - stiff[:, 0]) / stiff[:, 0], label=name, linestyle="-", color=color, linewidth=3.5)
    if not xlim is None: axTmp.set_xlim(xlim[0], xlim[1])
    if not ylim is None: axTmp.set_ylim(ylim[0], ylim[1])
    xlim = axTmp.get_xlim()
    ylim = axTmp.get_ylim()
    if showText:
        # axTmp.set_title("Stiffnesses as deployment goes", fontsize=14)
        axTmp.set_title("Standard Deviation of the {} first modes as deployment goes".format(nModes), fontsize=14)
        axTmp.set_xlabel("Average opening angle", fontsize=12)
        # axTmp.set_ylabel("Stiffnesses", fontsize=12)
        axTmp.set_ylabel("Standard Deviation", fontsize=12)
        axTmp.legend(loc=2, fontsize=12)
    else:
        axTmp.xaxis.set_ticklabels([])
        axTmp.yaxis.set_ticklabels([])
    axTmp.grid()
    axTmp.set_axisbelow(True)

    if not filename is None: plt.savefig(filename)
    plt.show()
    
def CompareStackedEigenvalues(listEigenvalues, listNames, ylim=None, filename=None, showText=True):
    '''
    Args:
        listEigenvalues : a list of arrays giving the data for each model
        listNames       : a list of names corresponding to each model
        ylim            : [yMin, yMax] used for all plots
        filename        : the path to where we want to save the figure
        showText        : whether we show text or not
    '''
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
    fig = plt.figure(figsize=(10, 8))

    axTmp = plt.subplot(gs[0, 0])

    xsPlot = np.arange(len(listEigenvalues))
    colorsSet = plt.cm.Set2(xsPlot)

    allVals = np.stack(listEigenvalues, axis=0)
    bottom = np.zeros(shape=(allVals.shape[0],))
    alphas = np.linspace(0.3, 1.0, allVals.shape[1])[::-1]
    if alphas.shape[0] == 1: alphas[0] = 1.0

    for i in range(allVals.shape[1]):
        axTmp.bar(xsPlot, allVals[:, i], width=0.6, bottom=bottom, color=colorsSet, alpha=alphas[i], log=False)
        bottom += allVals[:, i]

    axTmp.xaxis.set_tick_params(direction='out')
    axTmp.xaxis.set_ticks_position('bottom')
    axTmp.set_xticks(xsPlot)
    if showText:
        axTmp.set_title("Eigenvalues of increasing magnitude", fontsize=14)
        axTmp.set_xlabel("Models", fontsize=12)
        axTmp.set_ylabel("Cumulated eigenvalues", fontsize=12)
        axTmp.set_xticklabels(listNames)
    else:
        axTmp.xaxis.set_ticks([])
        axTmp.yaxis.set_ticklabels([])
        
    axTmp.set_xlim(-0.5, len(listNames) - 0.5)
    axTmp.grid(axis="y")
    axTmp.set_axisbelow(True)
    if not ylim is None: axTmp.set_ylim(ylim)
    if not filename is None: plt.savefig(filename)

    plt.show()
    
def CompareEigenvalues(listEigenvalues, listNames, ylim=None, filename=None, showText=True):
    '''
    Args:
        listEigenvalues : a list of arrays giving the data for each model
        listNames       : a list of names corresponding to each model
        ylim            : [yMin, yMax] used for all plots
        filename        : the path to where we want to save the figure
        showText        : whether we show text or not
    '''
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1], width_ratios=[1])
    fig = plt.figure(figsize=(10, 8))

    axTmp = plt.subplot(gs[0, 0])

    xsPlot = np.arange(len(listEigenvalues))
    colorsSet = plt.cm.Set2(xsPlot)

    allVals = np.stack(listEigenvalues, axis=0)
    allVals[:, 1:] -= allVals[:, :-1]
    bottom = np.zeros(shape=(allVals.shape[0],))
    alphas = np.linspace(0.3, 1.0, allVals.shape[1])[::-1]
    if alphas.shape[0] == 1: alphas[0] = 1.0

    for i in range(allVals.shape[1]):
        axTmp.bar(xsPlot, allVals[:, i], width=0.6, bottom=bottom, color=colorsSet, alpha=alphas[i], log=False)
        bottom += allVals[:, i]

    axTmp.xaxis.set_tick_params(direction='out')
    axTmp.xaxis.set_ticks_position('bottom')
    axTmp.set_xticks(xsPlot)
    if showText:
        axTmp.set_title("Eigenvalues of increasing magnitude", fontsize=14)
        axTmp.set_xlabel("Models", fontsize=12)
        axTmp.set_ylabel("Eigenvalues", fontsize=12)
        axTmp.set_xticklabels(listNames)
    else:
        axTmp.xaxis.set_ticks([])
        axTmp.yaxis.set_ticklabels([])
        
    axTmp.set_xlim(-0.5, len(listNames) - 0.5)
    axTmp.grid(axis="y")
    axTmp.set_axisbelow(True)
    if not ylim is None: axTmp.set_ylim(ylim)
    if not filename is None: plt.savefig(filename)

    plt.show()