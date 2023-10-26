import matplotlib.pyplot as plt
import numpy as np

def PlotUV(uv, curves, pathToSave=None, 
           jointMarkerParams=None, curveWidthParams=None):
    '''Plots the uv parameters.
    
    Args:
        uv: the uv coordinates, np array of shape (nJ, 2)
        curves: the linkages' curves (list of list of joint indices)
        pathToSave: the name of the output file
        jointMarkerParams: a dictionnary that contains "s" and "lw"
        curveWidthParams: a dictionnary that contains "lwInner" and "lwOuter"
    '''
    
    if jointMarkerParams is None:
        jointMarkerParams = {"s": 180, "lw": 4.0}
    if curveWidthParams is None:
        curveWidthParams = {"lwInner": 10.0, "lwOuter": 12.0}

    colOrange = np.array([255, 191, 105]) / 255.0

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot()
    for crv in curves:
        ax.plot(uv[crv, 0], uv[crv, 1], color=colOrange, linewidth=curveWidthParams["lwInner"], zorder=1)
        ax.plot(uv[crv, 0], uv[crv, 1], color='k', linewidth=curveWidthParams["lwOuter"], zorder=0)
    ax.scatter(uv[:, 0], uv[:, 1], s=jointMarkerParams["s"], facecolors='w', edgecolors='k', linewidth=jointMarkerParams["lw"], zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # manual arrowhead width and length
    hw = 0.06
    hl = 0.07
    lw = 1.5 # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    ax.arrow(0.0, 0.0, 1.15, 0.0, fc='k', ec='k', lw = lw, ls='-',
                head_width=hw, head_length=hl, overhang = ohg, 
                length_includes_head=True, clip_on = False, zorder=0) 

    ax.arrow(0.0, 0.0, 0.0, 1.15, fc='k', ec='k', lw = lw, ls='-',
            head_width=hw, head_length=hl, overhang = ohg, 
            length_includes_head= True, clip_on = False, zorder=0)

    tickLength = 0.025
    tickWidth = 3.0
    ax.plot([1.0, 1.0], [-tickLength, tickLength], c='k', lw = tickWidth, zorder=0)
    ax.plot([-tickLength, tickLength], [1.0, 1.0], c='k', lw = tickWidth, zorder=0)
    ax.plot([1.0, 1.0], [0.0, 1.0], c='k', lw = 0.5*tickWidth, ls='--', dashes=(5, 10), zorder=-1)
    ax.plot([0.0, 1.0], [1.0, 1.0], c='k', lw = 0.5*tickWidth, ls='--', dashes=(5, 10), zorder=-1)

    ax.axis('equal')

    if pathToSave is not None:
        plt.savefig(pathToSave, dpi=500, transparent=True)
    plt.show()


def PlotFlatLayout(jointsFlat, curves, pathToSave=None):
    '''Plots the flat layout
    
    Args:
        jointsFlat: the joints positions, np array of shape (nJ, 2)
        curves: the linkages' curves (list of list of joint indices)
        pathToSave: the name of the output file
    '''
    
    colOrange = np.array([255, 191, 105]) / 255.0

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot()
    for crv in curves:
        ax.plot(jointsFlat[crv, 0], jointsFlat[crv, 1], color=colOrange, linewidth=8.0, zorder=1)
        ax.plot(jointsFlat[crv, 0], jointsFlat[crv, 1], color='k', linewidth=10.0, zorder=0)
    ax.scatter(jointsFlat[:, 0], jointsFlat[:, 1], s=200, facecolors='w', edgecolors='k', linewidth=4.0, zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    ax.axis('equal')

    plt.savefig(pathToSave, transparent=True, dpi=600)
    plt.show()