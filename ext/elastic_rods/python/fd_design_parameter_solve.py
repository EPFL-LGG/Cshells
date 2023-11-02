from scipy.sparse import csc_matrix
import numpy as np
import numpy.linalg as la
from elastic_rods import EnergyType
import matplotlib.pyplot as plt

def energyAt(obj, dof, etype = EnergyType.Full, Positive_step=False):
    prevDoF = obj.getVars()
    obj.setVars(dof)
    energy = guardedEval(obj.energy, energyType = etype)
    obj.setVars(prevDoF)
    return energy

# Pybind11 methods/funcs apparently don't support `inspect.signature`,
# but at least their arg names are guaranteed to appear in the docstring... :(
def hasArg(func, argName):
    return argName in func.__doc__

def guardedEval(func, *args, **kwargs):
    '''
    Evaluate `func`, on the passed arguments, filtering out any unrecognized keyword arguments.
    '''
    return func(*args, **{k: v for k, v in kwargs.items() if hasArg(func, k)})

def fd_gradient_test(obj, stepSize, etype=EnergyType.Full, direction=None):
#     print(grad[linkage.numDoF():linkage.numDoF() + linkage.numRestKappaVars()])
    step = stepSize * direction
    x = obj.getVars()
    positive_energy = energyAt(obj, x + step, etype, True)
    negative_energy = energyAt(obj, x - step, etype)
#     print("Energy +: {}; Energy -: {}".format(positive_energy, negative_energy))
    energy_diff = positive_energy - negative_energy
#     print("diff: ", energy_diff)
#     print([energy_diff / (2 * stepSize), np.dot(direction, grad)])
    obj.setVars(x)
    grad = obj.gradient()
    return [energy_diff / (2 * stepSize), np.dot(direction, grad)]


def gradientAt(obj, dof, etype = EnergyType.Full, variableDesignParameters=False, perSegmentRestLen=False, updatedSource=False):
    prevDoF = obj.getVars()
    obj.setVars(dof)
    g = guardedEval(obj.gradient, updatedSource=updatedSource, energyType=etype, variableDesignParameters=variableDesignParameters)
    obj.setVars(prevDoF)
    return g

def gradient_convergence(obj, minStepSize=1e-8, maxStepSize=1e-2, etype=EnergyType.Full, direction = None):
    epsilons = np.logspace(np.log10(minStepSize), np.log10(maxStepSize), 100)
    errors = []

    for eps in epsilons:
        fd, an = fd_gradient_test(obj, eps, etype=etype, direction = direction)
        err = np.abs(an - fd) / np.abs(an)
        errors.append(err)
    return (epsilons, errors, an)

def gradient_convergence_plot(obj, minStepSize=1e-12, maxStepSize=1e-2, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, plot_name='gradient_validation.png'):
    eps, errors, ignore = gradient_convergence(obj, minStepSize, maxStepSize, etype, direction)
    plt.title('Directional derivative fd test for design parameter solve')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()
    plt.savefig(plot_name, dpi = 300)

def fd_hessian_test(obj, stepSize, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, infinitesimalTransportGradient=False):
    h = guardedEval(obj.hessian, energyType=etype).getTripletMatrix()
    h.reflectUpperTriangle()
    H = csc_matrix(h.compressedColumn())
    if (direction is None): 
        direction = np.array(guardedEval(obj.gradient, updatedSource=True, energyType=etype))

    dof = obj.getVars()
#     print(H*direction)
#     print(((gradientAt(obj, dof + stepSize * direction, etype,  updatedSource=infinitesimalTransportGradient))))
#     print(la.norm((gradientAt(obj, dof + stepSize * direction, etype,  updatedSource=infinitesimalTransportGradient)
#            - gradientAt(obj, dof - stepSize * direction, etype,  updatedSource=infinitesimalTransportGradient)) / (2 * stepSize)),
#             la.norm(H * direction))
    return [(gradientAt(obj, dof + stepSize * direction, etype,  updatedSource=infinitesimalTransportGradient)
           - gradientAt(obj, dof - stepSize * direction, etype,  updatedSource=infinitesimalTransportGradient)) / (2 * stepSize),
            H * direction]

def fd_hessian_test_relerror_max(obj, stepSize, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False):
    dgrad = fd_hessian_test(obj, stepSize, etype, direction)
    relErrors = np.abs((dgrad[0] - dgrad[1]) / dgrad[0])
    idx = np.argmax(relErrors)
    return (idx, relErrors[idx], dgrad[0][idx], dgrad[1][idx])

def fd_hessian_test_relerror_norm(obj, stepSize, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perSegmentRestLen=False, infinitesimalTransportGradient=False):
    dgrad = fd_hessian_test(obj, stepSize, etype, direction, infinitesimalTransportGradient=infinitesimalTransportGradient)
    return la.norm(dgrad[0] - dgrad[1]) / la.norm(dgrad[0])

def hessian_convergence(obj, minStepSize=1e-8, maxStepSize=1e-2, etype=EnergyType.Full, direction=None, infinitesimalTransportGradient=False):
    dof_size = obj.numVars()
    if (direction is None): direction = np.random.uniform(-1, 1, dof_size)
    
    epsilons = np.logspace(np.log10(minStepSize), np.log10(maxStepSize), 20)
    errors = [fd_hessian_test_relerror_norm(obj, eps, etype=etype, direction=direction, infinitesimalTransportGradient=infinitesimalTransportGradient) for eps in epsilons]
    return (epsilons, errors)

def hessian_convergence_plot(obj, minStepSize=1e-8, maxStepSize=1e-2, etype=EnergyType.Full, direction=None, infinitesimalTransportGradient=False, plot_name='hessian_validation.png'):
    from matplotlib import pyplot as plt
    eps, errors = hessian_convergence(obj, minStepSize, maxStepSize, etype, direction, infinitesimalTransportGradient=infinitesimalTransportGradient)
    plt.title('Directional derivative fd test for hessian')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()
    plt.savefig(plot_name, dpi = 300)

