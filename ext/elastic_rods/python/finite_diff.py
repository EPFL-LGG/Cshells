from elastic_rods import EnergyType
from MeshFEM import sparse_matrices
import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt
import numpy.linalg as la
import inspect

class DesignOptimizationTermFDWrapper:
    def __init__(self, term, linkage):
        self.term = term
        self.linkage = linkage
        self.term.update()

    def setDoFs(self, v):
        self.linkage.setExtendedDoFsPSRL(v)
        self.term.update()
        #self.linkage.updateSourceFrame()

    def getDoFs(self):
        return self.linkage.getExtendedDoFsPSRL()

    def numDoF(self):          return self.linkage.numExtendedDoFPSRL()
    def energy(self):          return self.term.value()
    def gradient(self):        return self.term.grad()
    # def gradient(self):        return self.term.computeGrad()
    def applyHessian(self, v): return self.term.computeDeltaGrad(v)

# All functions here work with both Elastic Rod object and Rod Linkage object despite the parameter names.
def getVars(l, variableDesignParameters=False, perSegmentRestLen=False):
    if perSegmentRestLen:
        return l.getExtendedDoFsPSRL()
    if (variableDesignParameters):
        return l.getExtendedDoFs()
    return l.getDoFs()

def setVars(l, dof, variableDesignParameters=False, perSegmentRestLen=False):
    if perSegmentRestLen:
        return l.setExtendedDoFsPSRL(dof)
    if (variableDesignParameters):
        return l.setExtendedDoFs(dof)
    return l.setDoFs(dof)

def energyAt(l, dof, etype = EnergyType.Full, variableDesignParameters=False, perSegmentRestLen=False, restoreDoF=True):
    if restoreDoF: prevDoF = getVars(l, variableDesignParameters, perSegmentRestLen)
    setVars(l, dof, variableDesignParameters, perSegmentRestLen)
    energy = guardedEval(l.energy, energyType = etype)
    if restoreDoF: setVars(l, prevDoF, variableDesignParameters, perSegmentRestLen)
    return energy

# Pybind11 methods/funcs apparently don't support `inspect.signature`,
# but at least their arg names are guaranteed to appear in the docstring... :(
def hasArg(func, argName):
    if (func.__doc__ is not None):
        return argName in func.__doc__
    return argName in inspect.signature(func).parameters

def guardedEval(func, *args, **kwargs):
    '''
    Evaluate `func`, on the passed arguments, filtering out any unrecognized keyword arguments.
    '''
    return func(*args, **{k: v for k, v in kwargs.items() if hasArg(func, k)})

def gradientAt(l, dof, etype = EnergyType.Full, variableDesignParameters=False, perSegmentRestLen=False, updatedSource=False, restoreDoF=True):
    if restoreDoF: prevDoF = getVars(l, variableDesignParameters, perSegmentRestLen)
    setVars(l, dof, variableDesignParameters, perSegmentRestLen)
    geval = l.gradientPerSegmentRestlen if perSegmentRestLen else l.gradient
    g = guardedEval(geval, updatedSource=updatedSource, energyType=etype, variableDesignParameters=variableDesignParameters)
    if restoreDoF: setVars(l, prevDoF, variableDesignParameters, perSegmentRestLen)
    return g

def fd_gradient_test(obj, stepSize, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, precomputedAnalyticalGradient=None, x=None, restoreDoF=True):
    if (x is None):
        x = getVars(obj, variableDesignParameters, perSegmentRestLen)
    if (direction is None): direction = np.random.uniform(-1, 1, x.shape)
    step = stepSize * direction

    an = precomputedAnalyticalGradient
    if (an is None):
        if perSegmentRestLen:
            grad = guardedEval(obj.gradientPerSegmentRestlen, updatedSource=False, energyType=etype, variableDesignParameters=variableDesignParameters)
        else:
            grad = guardedEval(obj.gradient, updatedSource=False, energyType=etype, variableDesignParameters=variableDesignParameters)
        an = np.dot(direction, grad)

    energyPlus  = energyAt(obj, x + step, etype, variableDesignParameters, perSegmentRestLen, restoreDoF=False)
    energyMinus = energyAt(obj, x - step, etype, variableDesignParameters, perSegmentRestLen, restoreDoF=False)

    if restoreDoF:
        setVars(obj, x, variableDesignParameters, perSegmentRestLen)

    return [(energyPlus - energyMinus) / (2 * stepSize), an]

def gradient_convergence(linkage, minStepSize=1e-12, maxStepSize=1e-2, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False):
    origDoF = getVars(linkage, variableDesignParameters, perSegmentRestLen)

    if (direction is None): direction = np.random.uniform(-1, 1, origDoF.shape)

    epsilons = np.logspace(np.log10(minStepSize), np.log10(maxStepSize), 100)
    errors = []

    an = None
    for eps in epsilons:
        fd, an = fd_gradient_test(linkage, eps, etype=etype, direction=direction, variableDesignParameters = variableDesignParameters, perSegmentRestLen=perSegmentRestLen, precomputedAnalyticalGradient=an, restoreDoF=False, x=origDoF)
        err = np.abs((an - fd) / an)
        errors.append(err)

    setVars(linkage, origDoF, variableDesignParameters, perSegmentRestLen)

    return (epsilons, errors, an)

def gradient_convergence_plot(linkage, minStepSize=1e-12, maxStepSize=1e-2, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, plot_name=None):
    eps, errors, ignore = gradient_convergence(linkage, minStepSize, maxStepSize, etype, direction, variableDesignParameters, perSegmentRestLen)
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()
    if (plot_name is not None): plt.savefig(plot_name, dpi = 300)


def fd_hessian_test(obj, stepSize, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, infinitesimalTransportGradient=False, hessianVectorProduct=False, x=None, precomputedAnalyticalHessVec=None, restoreDoF=True):
    if (x is None):
        x = getVars(obj, variableDesignParameters, perSegmentRestLen)
    if (direction is None): direction = np.random.uniform(-1, 1, x.shape)

    an = precomputedAnalyticalHessVec
    if (an is None):
        # Use hessian-vector product if requested, or if it's all we have.
        if hessianVectorProduct or not callable(getattr(obj, 'hessian', None)):
            an = guardedEval(obj.applyHessianPerSegmentRestlen if perSegmentRestLen else obj.applyHessian, v=direction, energyType=etype, variableDesignParameters=variableDesignParameters)
        else:
            hessEval = obj.hessianPerSegmentRestlen if perSegmentRestLen else obj.hessian
            h = guardedEval(hessEval, energyType=etype, variableDesignParameters=variableDesignParameters)
            h.reflectUpperTriangle()
            H = csc_matrix(h.compressedColumn())
            an = H * direction

    gradPlus  = gradientAt(obj, x + stepSize * direction, etype, variableDesignParameters=variableDesignParameters, perSegmentRestLen=perSegmentRestLen,  updatedSource=infinitesimalTransportGradient, restoreDoF=False)
    gradMinus = gradientAt(obj, x - stepSize * direction, etype, variableDesignParameters=variableDesignParameters, perSegmentRestLen=perSegmentRestLen,  updatedSource=infinitesimalTransportGradient, restoreDoF=False)

    if restoreDoF:
        setVars(obj, x, variableDesignParameters, perSegmentRestLen)

    return [(gradPlus - gradMinus) / (2 * stepSize), an]

def fd_hessian_test_relerror_max(linkage, stepSize, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, hessianVectorProduct=False):
    fd, an = fd_hessian_test(linkage, stepSize, etype, direction, variableDesignParameters, perSegmentRestLen, hessianVectorProduct = hessianVectorProduct)
    relErrors = np.nan_to_num(np.abs((fd - an) / an), 0.0)
    idx = np.argmax(relErrors)
    return (idx, relErrors[idx], fd[idx], an[idx])

def fd_hessian_test_relerror_norm(linkage, stepSize, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, infinitesimalTransportGradient=False, hessianVectorProduct=False, x=None, precomputedAnalyticalHessVec=None, restoreDoF=True):
    fd, an = fd_hessian_test(linkage, stepSize, etype, direction, variableDesignParameters, perSegmentRestLen, infinitesimalTransportGradient=infinitesimalTransportGradient, hessianVectorProduct = hessianVectorProduct, x=x, precomputedAnalyticalHessVec=precomputedAnalyticalHessVec, restoreDoF=restoreDoF)
    return [norm(fd - an) / norm(an), an]

def hessian_convergence(linkage, minStepSize=1e-12, maxStepSize=1e-2, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, infinitesimalTransportGradient=False, hessianVectorProduct=False, nsteps=40):
    origDoF = getVars(linkage, variableDesignParameters, perSegmentRestLen)
    if (direction is None): direction = np.random.uniform(-1, 1, origDoF.shape)

    epsilons = np.logspace(np.log10(minStepSize), np.log10(maxStepSize), nsteps)

    an = None
    errors = []
    for eps in epsilons:
        err, an = fd_hessian_test_relerror_norm(linkage, eps, etype=etype, direction=direction, variableDesignParameters=variableDesignParameters, perSegmentRestLen=perSegmentRestLen, infinitesimalTransportGradient=infinitesimalTransportGradient, hessianVectorProduct=hessianVectorProduct, x=origDoF, precomputedAnalyticalHessVec=an, restoreDoF=False)
        errors.append(err)

    setVars(linkage, origDoF, variableDesignParameters, perSegmentRestLen)

    return (epsilons, errors)

def hessian_convergence_plot(linkage, minStepSize=1e-12, maxStepSize=1e-2, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, infinitesimalTransportGradient=False, plot_name='hessian_validation.png', hessianVectorProduct=False, nsteps=40):
    from matplotlib import pyplot as plt
    eps, errors = hessian_convergence(linkage, minStepSize, maxStepSize, etype, direction, variableDesignParameters, perSegmentRestLen=perSegmentRestLen, infinitesimalTransportGradient=infinitesimalTransportGradient, hessianVectorProduct=hessianVectorProduct, nsteps=nsteps)
    plt.title('Directional derivative fd test for hessian')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()
    if (plot_name is not None): plt.savefig(plot_name, dpi = 300)

def block_error(linkage, var_indices, va, vb, grad, etype=EnergyType.Full, eps=1e-6, perturb=None, variableDesignParameters=False, perSegmentRestLen=False, hessianVectorProduct=False):
    '''
    Report the error in the (va, vb) block of the Hessian, where
    va and vb are members of the `var_types` array.
    '''
    if (perturb is None):
        perturb = np.random.normal(0, 1, len(grad))
    block_perturb = np.zeros_like(perturb)
    block_perturb[var_indices[vb]] = perturb[var_indices[vb]]
    fd_delta_grad, an_delta_grad = fd_hessian_test(linkage, eps, etype=etype, direction=block_perturb, variableDesignParameters=variableDesignParameters, perSegmentRestLen=perSegmentRestLen, hessianVectorProduct=hessianVectorProduct)
    fd_delta_grad = fd_delta_grad[var_indices[va]]
    an_delta_grad = an_delta_grad[var_indices[va]]
    return (la.norm(an_delta_grad - fd_delta_grad) / la.norm(an_delta_grad),
            fd_delta_grad, an_delta_grad)

def hessian_convergence_block_plot(linkage, var_types, var_indices, etype=EnergyType.Full, variableDesignParameters=False, perSegmentRestLen=False, plot_name='rod_linkage_hessian_validation.png', hessianVectorProduct=False):
    # The perSegmentRestLen flag should take priority over the variableDesignParameter since the perSegmentRestLen automatically assume design parameter exists (in particular the rest length exists).
    if perSegmentRestLen:
        grad = linkage.gradientPerSegmentRestlen()
    else:
        grad = linkage.gradient(variableDesignParameters = variableDesignParameters)
    perturb = np.random.normal(0, 1, len(grad))
    numVarTypes = len(var_types)
    epsilons = np.logspace(np.log10(1e-12), np.log10(1e2), 50)
    fig = plt.figure(figsize=(16, 12))
    for i, vi in enumerate(var_types):
        for j, vj in enumerate(var_types):
            plt.subplot(numVarTypes, numVarTypes, i * numVarTypes + j + 1)
            errors = [block_error(linkage, var_indices, vi, vj, grad, etype, eps, perturb, variableDesignParameters, perSegmentRestLen, hessianVectorProduct=hessianVectorProduct)[0] for eps in epsilons]
            plt.loglog(epsilons, errors)
            plt.title(f'({vi}, {vj}) block')
            plt.grid()
            plt.tight_layout()
    plt.savefig(plot_name, dpi = 300)
    plt.show()
