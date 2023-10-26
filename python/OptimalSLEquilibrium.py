from bending_validation import suppress_stdout as so
from elastic_rods import ElasticRod
from elastic_rods import compute_equilibrium
import py_newton_optimizer
import numpy as np
import torch

torch_dtype = torch.float64
torch.set_default_dtype(torch_dtype)

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

newtonOptimizerOptions = py_newton_optimizer.NewtonOptimizerOptions()
newtonOptimizerOptions.gradTol = 1.0e-7
newtonOptimizerOptions.verbose = 1
newtonOptimizerOptions.beta = 1.0e-8
newtonOptimizerOptions.niter = 50
newtonOptimizerOptions.verboseNonPosDef = False

def ComputeJointAngles(joints, idxEdgesA, idxEdgesB):
    '''
    Args:
        joints: torch tensor containing the joints positions (nJ, 3)
        idxEdgesA: np array containing the edges for each joint along curves of family A
        idxEdgesB: np array containing the edges for each joint along curves of family B

    Returns:
        angles: torch tensor containing the opening angle at each joint (nJ,)
    '''
    edgesA = joints[idxEdgesA[:, 1], :] - joints[idxEdgesA[:, 0], :]
    edgesB = joints[idxEdgesB[:, 1], :] - joints[idxEdgesB[:, 0], :]

    sinJoints = torch.linalg.norm(torch.cross(edgesA, edgesB), dim=1)
    cosJoints = torch.einsum('ij, ij -> i', edgesA, edgesB)
    
    angles = torch.atan2(sinJoints, cosJoints)
    
    return angles

def ComputeAverageJointAnglesFromQuads(joints, quads):
    '''
    Args:
        joints: torch tensor containing the joints positions (nJ, 3)
        quads: list of list of 4 elements containing the quads ordered in a clockwise fashion

    Returns:
        averageAngle: torch tensor containing the average of angles between 0-1 and 0-3
    
    Notes:
        This is how vertices are labelled within a quad
                    3 +----+ 2
                      |    |
                    0 +----+ 1
    '''
    
    quadsTorch = torch.tensor(quads)
    
    averageAngle = torch.mean(ComputeJointAngles(
        joints, 
        torch.stack([quadsTorch[:, 0], quadsTorch[:, 1]], axis=1), 
        torch.stack([quadsTorch[:, 0], quadsTorch[:, 3]], axis=1),
    ), dim=0)
    
    return averageAngle

def ComputeGradientAverageJointAnglesFromQuads(joints, quads):
    '''
    Args:
        joints: torch tensor containing the joints positions (nJ, 3)
        quads: list of list of 4 elements containing the quads ordered in a clockwise fashion

    Returns:
        gradAAtoJoints: torch tensor containing the average of angles between 0-1 and 0-3
    
    Notes:
        This is how vertices are labelled within a quad
                    3 +----+ 2
                      |    |
                    0 +----+ 1
    '''
    
    quadsTorch = torch.tensor(quads)
    
    idxEdgesA = torch.stack([quadsTorch[:, 0], quadsTorch[:, 1]], axis=1)
    idxEdgesB = torch.stack([quadsTorch[:, 0], quadsTorch[:, 3]], axis=1)
    
    edgesA = joints[idxEdgesA[:, 1], :] - joints[idxEdgesA[:, 0], :]
    edgesB = joints[idxEdgesB[:, 1], :] - joints[idxEdgesB[:, 0], :]
    
    cross = torch.linalg.norm(torch.cross(edgesA, edgesB), dim=1).reshape(-1, 1)
    dot = torch.einsum('ij, ij -> i', edgesA, edgesB).reshape(-1, 1)
    normSqEdgesA = torch.sum(edgesA ** 2, dim=1).reshape(-1, 1)
    normSqEdgesB = torch.sum(edgesB ** 2, dim=1).reshape(-1, 1)
    
    gradAngletoEdgesA = (edgesA * (dot / normSqEdgesA) - edgesB) / cross
    gradAngletoEdgesB = (edgesB * (dot / normSqEdgesB) - edgesA) / cross
    
    gradAAtoJoints = torch.zeros_like(joints)
    gradAAtoJoints[idxEdgesA[:, 1], :] = gradAAtoJoints[idxEdgesA[:, 1], :] + gradAngletoEdgesA
    gradAAtoJoints[idxEdgesA[:, 0], :] = gradAAtoJoints[idxEdgesA[:, 0], :] - gradAngletoEdgesA
    gradAAtoJoints[idxEdgesB[:, 1], :] = gradAAtoJoints[idxEdgesB[:, 1], :] + gradAngletoEdgesB
    gradAAtoJoints[idxEdgesB[:, 0], :] = gradAAtoJoints[idxEdgesB[:, 0], :] - gradAngletoEdgesB
    
    return gradAAtoJoints / quadsTorch.shape[0]

def ComputeDeployedLinkageStability(jointsDep, gradEnergy, quads):
    '''Computes the deployed equilibrium first order optimality gap as a measure of stability.
    
    Args:
        jointsDep: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        gradEnergy: np array containing the gradient of the energy with respect to the joints positions (nJ, 3)
        quads: list of list of 4 elements containing the quads ordered in a clockwise fashion
    
    Returns:
        stability: torch tensor measuring the stability gap using the optimal torque magnitude (lmbd)
        residuals: torch tensor of shape (nJ, 3) giving the residuals so they can be reused
        lmbd: optimal global torque magnitude to apply at the joints, to be reused
    '''
    
    gradAngles = ComputeGradientAverageJointAnglesFromQuads(jointsDep, quads).reshape(-1,)
    gradEnergy = torch.tensor(gradEnergy, dtype=torch_dtype).reshape(-1,)
    # Gradient should not flow through lambda since it is optimal
    lmbd = (gradAngles.detach() @ gradEnergy) / torch.sum(gradAngles.detach() ** 2)
    residuals = gradEnergy - lmbd * gradAngles
    return 0.5 * torch.sum(residuals ** 2), residuals.reshape(-1, 3), lmbd

def JointsToRestQuantities(joints, curves):
    '''Extracts rest quantities from the joints (assumed to lie in the xy-plane!!).
    
    Args:
        joints: torch tensor of shape (nJ, 3), should have a null component in the z-axis
        curves: list of lists that contain the joints indices for each curve
        
    Returns:
        restQuantities: torch tensor containing the [rest_lengths, rest_kappas] concatenated over the straight linkage
    '''
    nRQ = sum([len(crv) - 1 + len(crv) - 2 for crv in curves])
    rqPerCurve = torch.zeros(size=(nRQ,), dtype=torch_dtype)
    idRQ = 0
    for crv in curves:
        nRLCrv = len(crv) - 1
        nRKCrv = len(crv) - 2

        jointsTmp = joints[crv, :]
        tangentsTmp = jointsTmp[1:, :] - jointsTmp[:-1, :]
        rlTmp = torch.linalg.norm(tangentsTmp, dim=-1)
        rqPerCurve[idRQ:idRQ+nRLCrv] = rlTmp
        idRQ += nRLCrv

        dotsConsecutive = torch.einsum('ij, ij -> i', tangentsTmp[1:, :], tangentsTmp[:-1, :])
        crossConsecutive = torch.cross(tangentsTmp[:-1, :], tangentsTmp[1:, :])
        rkTmp = 2.0 * crossConsecutive[:, -1] / (rlTmp[1:] * rlTmp[:-1] + dotsConsecutive)
        rqPerCurve[idRQ:idRQ+nRKCrv] = rkTmp
        idRQ += nRKCrv

    return rqPerCurve


def ComputeAdjoint(rod, rhs):
    '''Compute the adjoint vector given the right hand side
    
    Args:
        rod: the current rod which should have optimal thetas in the DoFs already! Call OptimizeThetasUpdateRod first
        rhs: a numpy array of shape (nThetas,)
        
    Returns
        adjoint: the adjoint vector of shape (nThetas)
    '''
    hess = rod.hessian(variableDesignParameters=True)
    hess.reflectUpperTriangle()
    hess = hess.compressedColumn().toarray()
    d2E_dtheta2 = hess[rod.numDoF() - rod.numEdges():rod.numDoF(), rod.numDoF() - rod.numEdges():rod.numDoF()] 
    adjoint = np.linalg.solve(d2E_dtheta2, rhs)
    return adjoint

def ComputeEnergy(jointsFlat, jointsDep, curves, rodMaterial=None, cachedThetas=None):
    '''Computes the total energy gradient at the deployed state using the "jointsFlat" configuration to compute rest quantities.
    
    Args:
        jointsFlat: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        jointsDep: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        curves: list of lists that contain the joints indices for each curve
        rodMaterial: the rod material used
        cachedThetas: list of np array containing the thetas
    
    Returns:
        energy: sum of all energies in the rods composing the straight linkage
        rodsList: list of deformed elastic rods to be reused
    '''
    
    rodsList = []
    energy = 0.0
    for i, crv in enumerate(curves):
        jointsFlatCrv = jointsFlat[crv, :]
        jointsDepCrv = jointsDep[crv, :]

        # Create a rod and deform
        rodTmp = ElasticRod(list(jointsFlatCrv))
        if rodMaterial is not None:
            rodTmp.setMaterial(rodMaterial)
        dofs = rodTmp.getDoFs()
        dofs[:3 * jointsDepCrv.shape[0]] = jointsDepCrv.reshape(-1,)
        if cachedThetas is not None:
            dofs[3 * jointsDepCrv.shape[0]:] = cachedThetas[i]
        rodTmp.setDoFs(dofs)
        with so(): compute_equilibrium(rodTmp, options=newtonOptimizerOptions, fixedVars=list(range(3 * rodTmp.numVertices())))

        # Collect the gradients at the right spots
        energy += rodTmp.energy()
        rodsList.append(rodTmp)
        
    return energy, rodsList

def ComputeEnergyGradients(jointsFlat, jointsDep, curves, rodMaterial=None, cachedThetas=None):
    '''Computes the total energy gradient at the deployed state using the "jointsFlat" configuration to compute rest quantities.
    
    Args:
        jointsFlat: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        jointsDep: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        curves: list of lists that contain the joints indices for each curve
        rodMaterial: the rod material used
        cachedThetas: list of np array containing the thetas
    
    Returns:
        gradEnergy: np array containing the gradient of the energy with respect to the joints positions (nJ, 3)
        rodsList: list of deformed elastic rods to be reused
    '''
    
    rodsList = []
    gradEnergy = np.zeros(shape=(jointsFlat.shape[0], 3))
    for i, crv in enumerate(curves):
        jointsFlatCrv = jointsFlat[crv, :]
        jointsDepCrv = jointsDep[crv, :]

        # Create a rod and deform
        rodTmp = ElasticRod(list(jointsFlatCrv))
        if rodMaterial is not None:
            rodTmp.setMaterial(rodMaterial)
        dofs = rodTmp.getDoFs()
        dofs[:3 * jointsDepCrv.shape[0]] = jointsDepCrv.reshape(-1,)
        if cachedThetas is not None:
            dofs[3 * jointsDepCrv.shape[0]:] = cachedThetas[i]
        rodTmp.setDoFs(dofs)
        with so(): compute_equilibrium(rodTmp, options=newtonOptimizerOptions, fixedVars=list(range(3 * rodTmp.numVertices())))

        # Collect the gradients at the right spots
        gradEnergy[crv, :] = gradEnergy[crv, :] + rodTmp.gradient()[:3*len(crv)].reshape(-1, 3)
        rodsList.append(rodTmp)
    
    return gradEnergy, rodsList

def ComputeEnergyGradientsToFlat(jointsFlat, jointsDep, curves, rodMaterial=None, cachedThetas=None):
    '''Computes the total energy gradient at the deployed state using the "jointsFlat" configuration to compute rest quantities.
    
    Args:
        jointsFlat: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        jointsDep: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        curves: list of lists that contain the joints indices for each curve
        rodMaterial: the rod material used
        cachedThetas: list of np array containing the thetas
    
    Returns:
        gradEnergyToFlat: np array containing the gradient of the energy with respect to the flat joints positions (nJ, 3)
        rodsList: list of deformed elastic rods to be reused
    '''
    
    rodsList = []
    nRQ = sum([len(crv) - 1 + len(crv) - 2 for crv in curves])
    gradEnergy = np.zeros(shape=(nRQ,))
    idRQ = 0
    for i, crv in enumerate(curves):
        nRQCrv = len(crv) - 1 + len(crv) - 2
        jointsFlatCrv = jointsFlat[crv, :]
        jointsDepCrv = jointsDep[crv, :]

        # Create a rod and deform
        rodTmp = ElasticRod(list(jointsFlatCrv))
        if rodMaterial is not None:
            rodTmp.setMaterial(rodMaterial)
        dofs = rodTmp.getDoFs()
        dofs[:3 * jointsDepCrv.shape[0]] = jointsDepCrv.reshape(-1,)
        if cachedThetas is not None:
            dofs[3 * jointsDepCrv.shape[0]:] = cachedThetas[i]
        rodTmp.setDoFs(dofs)
        with so(): compute_equilibrium(rodTmp, options=newtonOptimizerOptions, fixedVars=list(range(3 * rodTmp.numVertices())))

        # Collect the gradients at the right spots
        gradEnergy[idRQ:idRQ+nRQCrv] = rodTmp.gradient(variableDesignParameters=True, designParameterOnly=True)[-nRQCrv:]
        rodsList.append(rodTmp)
        idRQ += nRQCrv
    
    jointsFlatCopy = jointsFlat.detach().clone()
    jointsFlatCopy.requires_grad = True
    rq = JointsToRestQuantities(jointsFlatCopy, curves)
    rq.backward(torch.tensor(gradEnergy, dtype=torch_dtype))
    gradEnergyToFlat = ToNumpy(jointsFlatCopy.grad)
        
    return gradEnergyToFlat, rodsList

def ComputeDeployedLinkageStabilityFull(jointsFlat, jointsDep, curves, quads, rodMaterial=None, cachedThetas=None):
    '''Same as ComputeDeployedLinkageStability, full pipeline.
    
    Args:
        jointsFlat: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        jointsDep: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        curves: list of lists that contain the joints indices for each curve
        quads: list of list of 4 elements containing the quads ordered in a clockwise fashion
        rodMaterial: the rod material used
        cachedThetas: list of np array containing the thetas
    
    Returns:
        stability: torch tensor measuring the stability gap using the optimal torque magnitude
        rodsList: list of deformed elastic rods to be reused
    '''
    
    gradEnergy, rodsList = ComputeEnergyGradients(jointsFlat, jointsDep, curves, rodMaterial=rodMaterial, cachedThetas=cachedThetas)
    stability, _, _ = ComputeDeployedLinkageStability(jointsDep, gradEnergy, quads)
    
    return stability, rodsList
    
def ComputeGradientDeployedLinkageStabilityFull(jointsFlat, jointsDep, curves, quads, rodMaterial=None, cachedThetas=None):
    '''Compute the gradients of the stability loss with respect to jointsFlat and jointsDep
    
    Args:
        jointsFlat: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        jointsDep: torch tensor of shape (nJ, 3) containing the joints in their deployed state
        curves: list of lists that contain the joints indices for each curve
        quads: list of list of 4 elements containing the quads ordered in a clockwise fashion
        rodMaterial: the rod material used
        cachedThetas: list of np array containing the thetas
    
    Returns:
        totalGradFlat: np array of shape (nJ, 3) containing the gradient of our stability measure with respect to flat joints
        totalGradDep: np array of shape (nJ, 3) containing the gradient of our stability measure with respect to deployed joints
    '''
    
    gradEnergy, rodsList = ComputeEnergyGradients(jointsFlat, jointsDep, curves, rodMaterial=rodMaterial, cachedThetas=cachedThetas)

    jointsDepCopy = jointsDep.clone()
    jointsDepCopy.requires_grad = True
    rec, res, _ = ComputeDeployedLinkageStability(jointsDepCopy, gradEnergy, quads)
    rec.backward()

    hvpEnergy = np.zeros(shape=(jointsFlat.shape[0], 3))
    hvpEnergyThetasPos = np.zeros(shape=(jointsFlat.shape[0], 3))
    nRQ = sum([len(crv) - 1 + len(crv) - 2 for crv in curves])
    hvpEnergyRQ = np.zeros(shape=(nRQ,))
    hvpEnergyThetasRQ = np.zeros(shape=(nRQ,))
    res = ToNumpy(res)
    idRQ = 0
    for crv, rod in zip(curves, rodsList):
        nRQCrv = len(crv) - 1 + len(crv) - 2
        nThetas = len(crv) - 1
        resTmp = np.zeros(shape=(3 * len(crv) + nThetas + nRQCrv,))
        resTmp[:3 * len(crv)] = res[crv, :].reshape(-1,)
        # no need to run: rod = OptimizeThetasUpdateRod(rod)
        # this was already done in ComputeEnergyGradients
        hvpTmp = rod.applyHessian(resTmp, variableDesignParameters=True)
        
        adjointPadded = np.zeros(shape=(3 * len(crv) + nThetas + nRQCrv,))
        adjoint = ComputeAdjoint(rod, - hvpTmp[3*len(crv):3*len(crv)+nThetas])
        adjointPadded[3*len(crv):3*len(crv)+nThetas] = adjoint
        hapTmp = rod.applyHessian(adjointPadded, variableDesignParameters=True)
        
        hvpEnergy[crv, :] = hvpEnergy[crv, :] + hvpTmp[:3*len(crv)].reshape(-1, 3)
        hvpEnergyThetasPos[crv, :] = hvpEnergyThetasPos[crv, :] + hapTmp[:3*len(crv)].reshape(-1, 3)
        hvpEnergyRQ[idRQ:idRQ+nRQCrv] = hvpTmp[-nRQCrv:]
        hvpEnergyThetasRQ[idRQ:idRQ+nRQCrv] = hapTmp[3*len(crv)+nThetas:]
        idRQ += nRQCrv

    totalGradDep = hvpEnergy + hvpEnergyThetasPos + ToNumpy(jointsDepCopy.grad)

    jointsFlatCopy = jointsFlat.clone()
    jointsFlatCopy.requires_grad = True
    rq = JointsToRestQuantities(jointsFlatCopy, curves)
    rq.backward(torch.tensor(hvpEnergyRQ, dtype=torch_dtype))

    totalGradFlat = ToNumpy(jointsFlatCopy.grad)
    
    return totalGradFlat, totalGradDep