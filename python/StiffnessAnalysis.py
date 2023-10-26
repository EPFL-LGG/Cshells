import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu
import sparse_matrices

# Compute Mass Matrix
import enum

class MassMatrixType(enum.Enum):
    IDENTITY = 1
    FULL = 2
    LUMPED = 3

def ComputeMassMatrix(obj, fixedVars=[], mtype=MassMatrixType.FULL):
    M_scipy = None

    if (mtype != MassMatrixType.IDENTITY):
        objectMethods = dir(obj)
        if (mtype == MassMatrixType.FULL):
            if ("massMatrix" in objectMethods):
                M = obj.massMatrix()
                Mtrip = M if isinstance(M, sparse_matrices.TripletMatrix) else M.getTripletMatrix()
                Mtrip.reflectUpperTriangle()
                Mtrip.rowColRemoval(fixedVars)
                M_scipy = Mtrip.compressedColumn()
            else:
                M_scipy = sp.sparse.eye(obj.numDoF() - len(fixedVars))
                print("WARNING: object does not implement `massMatrix`; falling back to identity metric")
        elif (mtype == MassMatrixType.LUMPED):
            if ("lumpedMassMatrix" in objectMethods):
                M_scipy = sp.sparse.diags(np.delete(obj.lumpedMassMatrix(), fixedVars))
            else:
                M_scipy = sp.sparse.eye(obj.numDoF() - len(fixedVars))
                print("WARNING: object does not implement `lumpedMassMatrix`; falling back to identity metric")
        else: raise Exception('Unknown mass matrix type.')
    else:
        M_scipy = sp.sparse.eye(obj.numDoF() - len(fixedVars))
        
    return M_scipy

def ComputeCompliance(linkage, deltaAlpha, fixedVars, nDirs=1, multMass=1.0e-5):
    '''
    Args:
        linkage    : the linkage for which we want to compute the most efficient directions
        deltaAlpha : the angle increment we want to impose
        fixedVars  : a list of fixed degrees of freedom
        nDirs      : the number of M-orthogonal directions we want to compute
        multMass   : the factor in front of the Mass matrix to make the Hessian positive definite
    
    Returns:
        deltaE           : a np array of shape (nDirs,) giving the energy increment specific to each direction (sorted in increasing fashion)
        dirs             : a np array of shape (nDirs, numDoF) giving the most efficient direction (same sorting as deltaE)
        commonLinearTerm : the linear increment of energy due to deltaAlpha
        commonQuadTerm   : the quadratic increment of energy due to deltaAlpha
    '''

    deltaE = np.zeros(shape=(nDirs,))
    dirs   = np.zeros(shape=(nDirs, linkage.numDoF()))
    maskFixedVars = np.array([i in fixedVars for i in range(linkage.numDoF())])

    # Factorize the Hessian
    H = linkage.hessian()

    Htrip = H if isinstance(H, sparse_matrices.TripletMatrix) else H.getTripletMatrix()
    Htrip.rowColRemoval(fixedVars)
    Htrip.reflectUpperTriangle()
    Hfull = Htrip.compressedColumn()

    idxAverageAngle  = linkage.dofOffsetForJoint(linkage.numJoints() - 1) + 6
    idxAverageAngle -= sum(np.array(fixedVars) <= idxAverageAngle)

    maskAlpha = np.arange(Hfull.shape[0])!=idxAverageAngle
    H_all_alpha   = Hfull[:, idxAverageAngle].toarray().reshape(-1,)
    H_x_alpha     = H_all_alpha[maskAlpha]
    H_alpha_alpha = H_all_alpha[idxAverageAngle]

    Htrip.rowColRemoval([idxAverageAngle])
    Htrip.reflectUpperTriangle()
    Hfull = Htrip.compressedColumn() + multMass * ComputeMassMatrix(linkage, fixedVars=fixedVars+[linkage.dofOffsetForJoint(linkage.numJoints() - 1) + 6], 
                                                                    mtype=MassMatrixType.FULL)

    # Factorize the Hessian using LU
    solver = splu(Hfull)

    # Compute the first direction
    Hinv_H_x_alpha = solver.solve(H_x_alpha)
    d1 = np.zeros(shape=(Hfull.shape[0]+1,))
    d1[maskAlpha] = - deltaAlpha * Hinv_H_x_alpha
    d1[idxAverageAngle]  = deltaAlpha

    deltaE[0] = (deltaAlpha ** 2 / 2) * d1[maskAlpha] @ (Hfull @ d1[maskAlpha])
    dirs[0][~maskFixedVars] = d1

    commonLinearTerm = - deltaAlpha * linkage.gradient()[idxAverageAngle]
    commonQuadTerm   = (deltaAlpha ** 2 / 2) * H_alpha_alpha

    if nDirs==1: return deltaE, dirs, commonLinearTerm, commonQuadTerm

    M = ComputeMassMatrix(linkage, fixedVars=fixedVars, mtype=MassMatrixType.FULL)
    A = np.zeros(shape=(nDirs-1, Hfull.shape[0]))
    b = np.zeros(shape=(nDirs-1,))
    eAlpha = np.zeros(shape=(d1.shape[0]))
    eAlpha[idxAverageAngle] = 1.
    M_eAlpha = M @ eAlpha # Last column of the mass matrix

    # Compute the other directions

    for i in range(nDirs-1):
        dPrevPadded   = dirs[i][~maskFixedVars].copy()
        dPrevPadded[idxAverageAngle] = 0.
        M_dPrevPadded = M @ dPrevPadded

        A[i] = M_dPrevPadded[maskAlpha] + deltaAlpha * M_eAlpha[maskAlpha]
        b[i] = - deltaAlpha * (M_dPrevPadded[idxAverageAngle] - deltaAlpha * M_eAlpha[idxAverageAngle])

        Hinv_A    = solver.solve(A[:i+1, :].T)
        A_d1      = A[:i+1, :] @ d1[maskAlpha]
        A_Hinv_A  = A[:i+1, :] @ Hinv_A
        lu, piv   = lu_factor(A_Hinv_A)
        # solver_A_Hinv_A = splu(A_Hinv_A)
        # mu        = solver_A_Hinv_A.solve(A_d1 - b[:i+1])
        mu        = lu_solve((lu, piv), A_d1 - b[:i+1])
        A_mu      = A[:i+1, :].T @ mu
        Hinv_A_mu = solver.solve(A_mu)

        dNew = np.zeros(shape=(Hfull.shape[0]+1,))
        dNew[maskAlpha] = d1[maskAlpha] - Hinv_A_mu
        dNew[idxAverageAngle] = deltaAlpha

        deltaE[i+1] = deltaE[0] + 1/2 * (A_d1 - b[:i+1]) @ mu
        dirs[i+1][~maskFixedVars] = dNew

    return deltaE, dirs, commonLinearTerm, commonQuadTerm

def ComputeRelativeStiffnessGap(linkage, deltaAlpha, fixedVars, multMass=1e-5):
    deltaE, dirs, commonLinearTerm, commonQuadTerm = ComputeCompliance(linkage, deltaAlpha, fixedVars, nDirs=2, multMass=multMass)
    return (deltaE[1] - deltaE[0]) / (deltaE[0] + commonQuadTerm)
