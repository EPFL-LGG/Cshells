import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import numpy as np

class ParametricFamily:
    
    def __init__(self, minVal, maxVal) -> None:
        self.params = None
        self.minVal = minVal
        self.maxVal = maxVal
    
    def eval(self, locs):
        """Evaluates the function at the provided locations.
        
        Args:
            locs: a np array of shape (nLocs, 2)
        
        Returns:
            vals: the values at the prescribed locations (nLocs,)        
        """

        print("Please specify the kind of parametric family you want to use.")
        raise NotImplementedError
    
    def plot(self, nPlots=100, ax=None, nContours=10, showCB=True, showAxes=True, locs=None, figName=None):
        """Plots the height field produced by the function.
        
        Args:
            nPlots: the plot resolution
            ax: a matplotlib ax in which things should be plot
            nContours: the number of isolines
            showCB: whether we want to show the colorbar
            showAxes: whether we show the axes or not
            locs: whether we want to plot the locations
            figName: name of the figure when saving it (None to prevent saving)
        """
        
        axProvided = ax is not None
        if not axProvided:
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])

        xsPlot = np.linspace(0.0, 1.0, nPlots)
        ysPlot = np.linspace(0.0, 1.0, nPlots)

        XsPlot, YsPlot = np.meshgrid(xsPlot, ysPlot)
        ZsPlot = self.eval(np.stack([XsPlot, YsPlot], axis=2).reshape(-1, 2)).reshape(nPlots, nPlots)

        if nContours > 0:
            contours = ax.contour(XsPlot, YsPlot, ZsPlot, nContours, colors='black')
            ax.clabel(contours, inline=True, fontsize=8)
            
        if locs:
            ax.scatter(locs[:, 0], locs[:, 1], c='k', s=50.0, marker='.')

        im = ax.imshow(ZsPlot, extent=[xsPlot[0], xsPlot[-1], ysPlot[0], ysPlot[-1]], origin='lower',
                       cmap='RdGy', alpha=0.5)
        
        if not showAxes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        if not axProvided:
            if showCB:
                plt.colorbar(im)
            plt.show()
            
        if figName is not None:
            plt.savefig(figName)
    
    
class GaussianRBF(ParametricFamily):
    
    def __init__(self, params, eps, minVal, maxVal) -> None:
        super().__init__(minVal, maxVal)
        assert params.shape[0] % 3 == 0
        self.params = params
        self.nMix = int(params.shape[0] / 3)
        self.centers = self.params[:2*self.nMix].reshape(-1, 2)
        self.weights = self.params[2*self.nMix:]
        self.eps = eps
        
    def eval(self, locs):
        nLocs = locs.shape[0]
        distsSq = np.sum((locs.reshape(nLocs, 1, 2) - self.centers.reshape(1, self.nMix, 2)) ** 2, axis=-1)
        return self.minVal + (self.maxVal - self.minVal) * (np.exp(-(self.eps ** 2) * distsSq) @ self.weights)
    
    def plot(self, nPlots=100, ax=None, nContours=10, showCB=True, 
             xLimPlot=None, yLimPlot=None,
             showCenters=True, locs=None, showAxes=True, figName=None):
        axProvided = ax is not None
        if not axProvided:
            fig = plt.figure(figsize=(8 + 2 * showCB, 8))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
        
        if xLimPlot is None: xLimPlot = [0.0, 1.0]
        if yLimPlot is None: yLimPlot = [0.0, 1.0]
        xsPlot = np.linspace(xLimPlot[0], xLimPlot[1], nPlots)
        ysPlot = np.linspace(yLimPlot[0], yLimPlot[1], nPlots)

        XsPlot, YsPlot = np.meshgrid(xsPlot, ysPlot)
        ZsPlot = self.eval(np.stack([XsPlot, YsPlot], axis=2).reshape(-1, 2)).reshape(nPlots, nPlots)
        ZsPlot = ZsPlot

        cmapBackground = cmap.plasma

        if locs is not None:
            ax.scatter(locs[:, 0], locs[:, 1], c='k', s=50.0, marker='.')
        else:
            if showCenters:
                ax.scatter(self.centers[:, 0], self.centers[:, 1], c='k', s=15.0 + 500.0 * self.weights, marker='.')

        if nContours > 0:
            contours = ax.contour(XsPlot, YsPlot, ZsPlot, nContours, colors='black')
            ax.clabel(contours, inline=True, fontsize=8)

        im = ax.imshow(ZsPlot, extent=[xsPlot[0], xsPlot[-1], ysPlot[0], ysPlot[-1]], origin='lower',
                       cmap=cmapBackground, alpha=0.7)
        
        if not showAxes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        if figName is not None:
            plt.savefig(figName, dpi=100)
        
        if not axProvided:
            if showCB:
                plt.colorbar(im)
            plt.show()
            
