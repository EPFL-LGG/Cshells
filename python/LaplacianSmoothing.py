from matplotlib import gridspec
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

################################################################
########              LAPLACIAN SMOOTHING             ##########
################################################################


def ComputeAdjacencyCP(edges, curvesWithCP, curveFamily, disconnectEnd=False):
    '''
    Args:
        edges         : torch tensor of shape (nEdges, 2) containing all the edges
        curvesWithCP  : list of list giving the control points used for each curve
        curveFamily   : list containing whether the curve is labeled as A (0) or B (1)
        disconnectEnd : wehther we disconnect the ends of the rods or not

    Returns:
        adjCP : torch tensor of shape (nCP, nCP) where nCP is the  total number of control points
    '''

    # Reflect each edge
    cpConnRefl  = edges.tolist() + [[lst[1], lst[0]] for lst in edges.tolist()]
    uniques, valence = torch.unique(edges, return_counts=True)
    nCP = uniques.shape[0]
    cpConnTorch = torch.tensor(cpConnRefl)

    if disconnectEnd:
        cpFamily = torch.zeros(size=(nCP, 2))
        for i, crv in enumerate(curvesWithCP):
            cpFamily[crv[1:], curveFamily[i]] +=1
            cpFamily[crv[:-1], curveFamily[i]] +=1
        # To make sure loops are correctly handled
        edgeEnd = [[crv[0], crv[1]] for i, crv in enumerate(curvesWithCP) if cpFamily[crv[0], curveFamily[i]] == 1]
        edgeEnd += [[crv[-1], crv[-2]] for i, crv in enumerate(curvesWithCP) if cpFamily[crv[-1], curveFamily[i]] == 1]
        cpConnDisc = [edge for edge in cpConnRefl if not edge in edgeEnd]
        cpConnTorch = torch.tensor(cpConnDisc)

    # Construct the adjacency matrix
    values      = torch.ones(size=(cpConnTorch.shape[0],))
    sparseAdjCP = torch.sparse_coo_tensor(cpConnTorch.T, values, size=(nCP, nCP))

    adjCP  = sparseAdjCP.to_dense()
    colSum    = torch.sum(adjCP, dim=1)
    whereZero = colSum.reshape(-1,) == 0

    adjCP[whereZero, :] = torch.eye(nCP)[whereZero, :]
    adjCP /= torch.sum(adjCP, dim=1, keepdim=True)

    return adjCP

def DifferentialAdjacencyControlPoints(controlPoints, fixedIdx, edges, curvesWithCP, curveFamily, adjCP=None):
    '''
    Args:
        controlPoints : torch tensor of size (nCP, 3) containing the control points
        fixedIdx      : list of indices to fix
        edges         : torch tensor of shape (nEdges, 2) containing all the edges
        curvesWithCP  : list of list giving the control points used for each curve
        curveFamily   : list containing whether the curve is labeled as A (0) or B (1)
        adjCP         : torch tensor of shape (nCP, nCP) where nCP is the  total number of control points

    Returns:
        dAllCP : torch tensor of shape (nCP, 3) containing the differential to pass around
    '''

    if adjCP is None:
        adjCP = ComputeAdjacencyCP(edges, curvesWithCP, curveFamily, disconnectEnd=True)

    dAllCP = adjCP @ controlPoints - controlPoints
    dAllCP[fixedIdx] = 0.0

    return dAllCP

def ComputeLaplacianCP(edges, curvesWithCP, curveFamily, disconnectEnd=False, fixedIdx=[]):
    '''
    Args:
        edges         : torch tensor of shape (nEdges, 2) containing all the edges
        curvesWithCP  : list of list giving the control points used for each curve
        curveFamily   : list containing whether the curve is labeled as A (0) or B (1)
        disconnectEnd : wehther we disconnect the ends of the rods or not
        fixedIdx      : list of fixed control points

    Returns:
        lapCP : torch tensor of shape (nCP, nCP) where nCP is the  total number of control points
    '''

    # Reflect each edge
    cpConnRefl  = edges.tolist() + [[lst[1], lst[0]] for lst in edges.tolist()]
    uniques, valence = torch.unique(edges, return_counts=True)
    nCP = uniques.shape[0]
    cpConnTorch = torch.tensor(cpConnRefl)

    if disconnectEnd:
        cpFamily = torch.zeros(size=(nCP, 2))
        for i, crv in enumerate(curvesWithCP):
            cpFamily[crv[1:], curveFamily[i]] +=1
            cpFamily[crv[:-1], curveFamily[i]] +=1
        # To make sure loops are correctly handled
        edgeEnd = [[crv[0], crv[1]] for i, crv in enumerate(curvesWithCP) if cpFamily[crv[0], curveFamily[i]] == 1]
        edgeEnd += [[crv[-1], crv[-2]] for i, crv in enumerate(curvesWithCP) if cpFamily[crv[-1], curveFamily[i]] == 1]
        cpConnDisc = [edge for edge in cpConnRefl if not edge in edgeEnd]
        cpConnTorch = torch.tensor(cpConnDisc)

    # Construct the adjacency matrix
    values      = torch.ones(size=(cpConnTorch.shape[0],))
    sparseAdjCP = torch.sparse_coo_tensor(cpConnTorch.T, values, size=(nCP, nCP))

    adjCP  = sparseAdjCP.to_dense()
    colSum    = torch.sum(adjCP, dim=1)
    whereZero = colSum.reshape(-1,) == 0

    adjCP[~whereZero, :] /= torch.sum(adjCP, dim=1, keepdim=True)[~whereZero, :]
    lapCP = adjCP - torch.eye(nCP)
    
    # To fix points that are not connected
    lapCP[whereZero, :] = torch.eye(nCP)[whereZero, :]
    lapCP[fixedIdx, :]  = torch.eye(nCP)[fixedIdx, :]

    return lapCP

def SolveLaplacian(controlPoints, lapCP, invLapCP=None):
    '''
    Solves the equation L.cp = [cp_f 0]^T where L is the Laplacian, cp are the control points,
    and cp_f are the fixed control points

    Args:
        controlPoints : torch tensor of size (nCP, 3) containing the control points
        lapCP         : torch tensor of shape (nCP, nCP) where nCP is the  total number of control points
        fixedIdx      : list of fixed control points
        invLapCP      : torch tensor of shape (nCP, nCP), the inverse of lapCP (will be computed if not provided)

    Returns:
        newCP : torch tensor of size (nCP, 3) containing the new control points
    '''
    if invLapCP is None:
        invLapCP = torch.linalg.inv(lapCP)

    idxFix      = torch.diag(lapCP) == 1.
    rhs         = torch.zeros_like(controlPoints)
    rhs[idxFix] = controlPoints[idxFix]
    
    newCP = invLapCP @ rhs # Not numerically stable, but still enable
    return newCP

def LeastSquaresLaplacian(controlPoints, controlPointsFixed, lapCP):
    '''
    Computes 1/2||L.cp - [cp_f 0]^T ||^2

    Args:
        controlPoints      : torch tensor of size (nCP, 3) containing the control points
        controlPointsFixed : torch tensor of size (nCP, 3) containing the fixed control points
        lapCP              : torch tensor of shape (nCP, nCP) where nCP is the  total number of control points

    Returns:
        ls : the value 1/2||L.cp - [cp_f 0]^T ||^2
    '''

    idxFix      = torch.diag(lapCP) == 1.0
    rhs         = torch.zeros_like(controlPoints)
    rhs[idxFix] = controlPointsFixed[idxFix]
    residual    = lapCP @ controlPoints - rhs
    ls          = 0.5 * torch.sum(residual ** 2)
    
    return ls

def LeastSquaresLaplacianFullGradient(controlPoints, controlPointsFixed, lapCP):
    '''
    Computes L^T.L.cp - L^T.[cp_f 0]^T, the gradient of LeastSquaresLaplacian wrt the control points

    Args:
        controlPoints      : torch tensor of size (nCP, 3) containing the control points
        controlPointsFixed : torch tensor of size (nCP, 3) containing the fixed control points
        lapCP              : torch tensor of shape (nCP, nCP) where nCP is the  total number of control points

    Returns:
        grad : torch tensor of size (nCP, 3) containing the gradient
    '''

    idxFix      = torch.diag(lapCP) == 1.0
    rhs         = torch.zeros_like(controlPoints)
    rhs[idxFix] = controlPointsFixed[idxFix]
    grad = lapCP.T @ (lapCP @ controlPoints - rhs)
    
    return grad

def LeastSquaresLaplacianFullHVP(controlPoints, dControlPoints, controlPointsFixed, lapCP):
    '''
    Computes L^T.L.dcp, the HVP of LeastSquaresLaplacian wrt the control points

    Args:
        dControlPoints     : torch tensor of size (nCP, 3) containing the perturbation of the control points
        controlPointsFixed : torch tensor of size (nCP, 3) containing the fixed control points
        lapCP              : torch tensor of shape (nCP, nCP) where nCP is the  total number of control points

    Returns:
        hvp : torch tensor of size (nCP, 3) containing the hvp
    '''

    idxFix      = torch.diag(lapCP) == 1.0
    rhs         = torch.zeros_like(controlPoints)
    rhs[idxFix] = controlPointsFixed[idxFix]
    hvp = lapCP.T @ (lapCP @ dControlPoints)

    return hvp

def LeastSquaresLaplacianGradient(controlPoints, lapCP):
    '''
    Computes L_free_free^T.(L_free_free.cp_free + L_free_fixed.cp_fixed), and creates the gradient of Dirichlet
    wrt all the control points

    Args:
        controlPoints : torch tensor of size (nCP, 3) containing the control points
        lapCP         : torch tensor of shape (nCP, nCP) where nCP is the  total number of control points
        fixedIdx      : list of fixed control points
        invLapCP      : torch tensor of shape (nCP, nCP), the inverse of lapCP (will be computed if not provided)

    Returns:
        grad : torch tensor of size (nCP, 3) containing the gradient
    '''

    idxFixed = torch.diag(lapCP) == 1.0
    idxFree  = ~idxFixed
    rhs  = - lapCP[idxFree, :][:, idxFixed] @ controlPoints[idxFixed, :]
    lhs  = lapCP[idxFree, :][:, idxFree]
    grad = torch.zeros_like(controlPoints)
    grad[idxFree]  = lhs.T @ (lhs @ controlPoints[idxFree, :] - rhs)
    grad[idxFixed] = 0.0
    
    return grad

def LeastSquaresLaplacianHVP(dControlPoints, lapCP):
    '''
    Computes L_free_free^T.(L_free_free.cp_free + L_free_fixed.cp_fixed), and creates the gradient of Dirichlet
    wrt all the control points

    Args:
        controlPoints : torch tensor of size (nCP, 3) containing the control points
        lapCP         : torch tensor of shape (nCP, nCP) where nCP is the  total number of control points
        fixedIdx      : list of fixed control points
        invLapCP      : torch tensor of shape (nCP, nCP), the inverse of lapCP (will be computed if not provided)

    Returns:
        grad : torch tensor of size (nCP, 3) containing the gradient
    '''
    
    return LeastSquaresLaplacianGradient(dControlPoints, lapCP)

def fdValidationLaplacian(cpGlobal, lapCP):

    idxFix = torch.diag(lapCP) == 1.0

    epsilons = torch.logspace(-10, -0.5, 50)
    
    torch.manual_seed(1)
    cpGlobal       = cpGlobal.detach().clone()
    cpGlobal      += 0.001 * torch.rand(size=cpGlobal.shape)
    cpGlobal[:, 2] = 0.0
    torch.manual_seed(0)
    perturbCP         = torch.rand(size=cpGlobal.shape)
    perturbCP        /= torch.linalg.norm(perturbCP)
    perturbCP[:, 2]   = 0.0

    gradGlobal = LeastSquaresLaplacianFullGradient(cpGlobal, cpGlobal, lapCP)
    anDeltaObj = torch.sum(gradGlobal * perturbCP)

    hvp     = LeastSquaresLaplacianFullHVP(cpGlobal, perturbCP, cpGlobal, lapCP)
    normHVP = torch.linalg.norm(hvp)

    errorsFullGrad  = []
    errorsFullHVP   = []
    for i, eps in enumerate(epsilons):

        # One step forward
        obj1  = LeastSquaresLaplacian(cpGlobal + eps * perturbCP, cpGlobal, lapCP)
        grad1 = LeastSquaresLaplacianFullGradient(cpGlobal + eps * perturbCP, cpGlobal, lapCP)

        # Two steps backward
        obj2  = LeastSquaresLaplacian(cpGlobal - eps * perturbCP, cpGlobal, lapCP)
        grad2 = LeastSquaresLaplacianFullGradient(cpGlobal - eps * perturbCP, cpGlobal, lapCP)

        # Compute error
        fdDeltaObj = (obj1 - obj2) / (2.0*eps) 
        errorsFullGrad.append(abs(fdDeltaObj - anDeltaObj) / abs(anDeltaObj))

        fdDeltaGrad = (grad1 - grad2) / (2.0*eps)
        errorsFullHVP.append(torch.linalg.norm(fdDeltaGrad - hvp) / (normHVP + 1.0e-14))

        if i%5 == 0:
            print("Done with {} out of {} epsilons.".format(i+1, epsilons.shape[0]))

    gs = gridspec.GridSpec(nrows=1, ncols=3, height_ratios=[1], width_ratios=[1, 0.05, 1])
    fig = plt.figure(figsize=(16, 5))

    axTmp = plt.subplot(gs[0, 0])
    axTmp.loglog(epsilons, errorsFullGrad)
    axTmp.set_xlabel("Step size")
    axTmp.set_ylabel("Relative error")
    axTmp.set_title("Gradient of the Laplacian residuals")
    axTmp.grid()

    axTmp = plt.subplot(gs[0, -1])
    axTmp.loglog(epsilons, errorsFullHVP)
    axTmp.set_xlabel("Step size")
    axTmp.set_ylabel("Relative error")
    axTmp.set_title("HVP of the Laplacian residuals")
    axTmp.grid()
    pass
