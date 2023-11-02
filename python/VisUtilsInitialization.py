from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def set_axes_equal(ax: plt.Axes):
    '''Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    '''
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

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


def PlotFlatLayout(jointsFlat, curves, pathToSave=None, jointMarkerParams=None, curveWidthParams=None):
    '''Plots the flat layout
    
    Args:
        jointsFlat: the joints positions, np array of shape (nJ, 2)
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
        ax.plot(jointsFlat[crv, 0], jointsFlat[crv, 1], color=colOrange, linewidth=curveWidthParams["lwInner"], zorder=1)
        ax.plot(jointsFlat[crv, 0], jointsFlat[crv, 1], color='k', linewidth=curveWidthParams["lwOuter"], zorder=0)
    ax.scatter(jointsFlat[:, 0], jointsFlat[:, 1], s=jointMarkerParams['s'], facecolors='w', edgecolors='k', linewidth=jointMarkerParams['lw'], zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    ax.axis('equal')

    if pathToSave is not None:
        plt.savefig(pathToSave, transparent=True, dpi=600)
    plt.show()

def PlotDeployedLayout(
        jointsDep, curves, surf, 
        nPlot=100, pathToSave=None, jointMarkerParams=None, curveWidthParams=None,
        azim=0.0, elev=30.0,
    ):
    '''Plots the deployed layout

    Args:
        jointsDep: the joints positions, np array of shape (nJ, 2)
        curves: the linkages' curves (list of list of joint indices)
        surf: a BSpline.Surface() object from geomdl
        nPlot: the resolution of the mesh to plot
        pathToSave: the name of the output file
        jointMarkerParams: a dictionnary that contains "s"
        curveWidthParams: a dictionnary that contains "lw"
        azim: the azimuth for the plot
        elev: the elevation for the plot
    '''

    colOrange = np.array([255, 191, 105]) / 255.0
    if jointMarkerParams is None:
        jointMarkerParams = {"s": 180}
    if curveWidthParams is None:
        curveWidthParams = {"lw": 10.0}

    uPlot = np.linspace(0.0, 1.0, nPlot)
    vPlot = np.linspace(0.0, 1.0, nPlot)
    uPlot, vPlot = np.meshgrid(uPlot, vPlot)
    uvPlot = np.stack([uPlot.reshape(-1,), vPlot.reshape(-1,)], axis=1)
    surfPointsPlot = np.array(surf.evaluate_list(uvPlot.tolist()))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d', azim=azim, elev=elev)
    ax.set_box_aspect([1, 1, 1])
    for crv in curves:
        ax.plot(jointsDep[crv, 0], jointsDep[crv, 1], jointsDep[crv, 2], color=colOrange, lw=curveWidthParams['lw'])
    ax.scatter(jointsDep[:, 0], jointsDep[:, 1], jointsDep[:, 2], c='k', s=jointMarkerParams['s'])
    surfPlot = ax.plot_surface(
        surfPointsPlot[:, 0].reshape(nPlot, nPlot), 
        surfPointsPlot[:, 1].reshape(nPlot, nPlot), 
        surfPointsPlot[:, 2].reshape(nPlot, nPlot), 
        cmap=cm.bone,
        linewidth=0.0, antialiased=True, alpha=0.4
    )
    set_axes_equal(ax)
    
    plt.axis('off')
    plt.grid(b=None)
    
    if pathToSave is not None:
        plt.savefig(pathToSave, transparent=True, dpi=600)
    plt.show()
