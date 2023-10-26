import numpy as np
import torch

torch.set_default_dtype(torch.float64)

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

###########################################################################
#############    EXTRACTING QUANTITIES FROM THE DOF VECTOR    #############
###########################################################################

def IndicesDiscretePositions(linkage, edges, subdivision):
    '''
    Args:
        linkage     : a rod linkage object 
        edges       : the corresponding rod segments (list of pairs containing the start and end joint for 
                      each segment)
        subdivision : number of discrete elements for a single rod
    
    Returns:
        idxDoF : a list of nEdges list containing the indices extracting the positions
    '''
    
    idxPos = []
    for i, edge in enumerate(edges):
        lstEdge = []
        offsetJoint = linkage.dofOffsetForJoint(edge[0])
        offsetSegment = linkage.dofOffsetForSegment(i)
        lstEdge += 2 * list(range(offsetJoint, offsetJoint+3)) # For the two positions at the first end
        for j in range(subdivision-3):
            lstEdge += list(range(offsetSegment+j*3, offsetSegment+(j+1)*3))
        offsetJoint = linkage.dofOffsetForJoint(edge[1])
        lstEdge += 2 * list(range(offsetJoint, offsetJoint+3))
        
        idxPos.append(lstEdge)
        
    return idxPos

def IndicesDiscretePositionsNoDuplicate(linkage, edges, subdivision, flattenList=True):
    '''
    Args:
        linkage     : a rod linkage object 
        edges       : the corresponding rod segments (list of pairs containing the start and end joint for 
                      each segment)
        subdivision : number of discrete elements for a single rod
        flattenList : whether we end up flattening the list or not
    
    Returns:
        idxDoFflat : a list all the DoF indices related to the positions
    '''
    
    idxPos = []
    treatedJoint = []
    for i, edge in enumerate(edges):
        lstEdge = []
        offsetJoint = linkage.dofOffsetForJoint(edge[0])
        offsetSegment = linkage.dofOffsetForSegment(i)
        
        if not offsetJoint in treatedJoint:
            lstEdge += list(range(offsetJoint, offsetJoint+3))
            treatedJoint.append(offsetJoint)
        for j in range(subdivision-3):
            lstEdge += list(range(offsetSegment+j*3, offsetSegment+(j+1)*3))
        offsetJoint = linkage.dofOffsetForJoint(edge[1])
        if not offsetJoint in treatedJoint:
            lstEdge += list(range(offsetJoint, offsetJoint+3))
            treatedJoint.append(offsetJoint)
        
        idxPos.append(lstEdge)
    
    if flattenList:
        idxPosFlat = [idx for idxSubList in idxPos for idx in idxSubList]
        
    return idxPosFlat

def VectorFieldForDiscretePositions(dDoF, linkage, edges, subdivision, jointsOnly=True):
    '''
    Args:
        dDoF        : np array of shape (nDoF,) containing the dof modification
        linkage     : a rod linkage object 
        edges       : the corresponding rod segments (list of pairs containing the start and end joint for 
                      each segment)
        subdivision : number of discrete elements for a single rod
    
    Returns:
        field : a list of arrays of shape (rod.NumVertices, 3)
    '''
    
    idxPos = IndicesDiscretePositions(linkage, edges, subdivision)
    
    maskInner = np.zeros(shape=(subdivision+1,3))
    maskInner[0:2, :] = 1.0
    maskInner[-2:, :] = 1.0
    
    field    = [maskInner*dDoF[idx].reshape(-1, 3) if jointsOnly else dDoF[idx].reshape(-1, 3) for idx in idxPos]
    
    # Removing duplicates at the joints
    newField = []
    
    mask2Joints = np.ones_like(field[0])
    mask2Joints[0:2, :] = 0.0
    mask2Joints[-2:, :] = 0.0
    
    maskStartJoint = np.ones_like(field[0])
    maskStartJoint[0:2, :] = 0.0
    
    maskEndJoint = np.ones_like(field[0])
    maskEndJoint[-2:, :] = 0.0
    
    listVisitedJoint = []
    
    for rodField, edge in zip(field, edges):
        if edge[0] in listVisitedJoint and edge[1] in listVisitedJoint:
            newField.append(mask2Joints * rodField)
            
        elif edge[0] in listVisitedJoint and not edge[1] in listVisitedJoint:
            newField.append(maskStartJoint * rodField)
            listVisitedJoint.append(edge[1])
            
        elif not edge[0] in listVisitedJoint and edge[1] in listVisitedJoint:
            newField.append(maskEndJoint * rodField)
            listVisitedJoint.append(edge[0])
            
        else:
            newField.append(rodField)
            listVisitedJoint.append(edge[0])
            listVisitedJoint.append(edge[1])
        
    return newField

def ScalarFieldDeviations(linkage, tsf, useSurfDim=False, usePercent=False, perEdge=True):
    '''
    Args:
        linkage    : a rod linkage object 
        tsf        : a target surface fitter object
        useSurfDim : whether we prefer using the model's scale or the tsf mesh scale
        usePercent : whether we use percentages or not
        perEdge    : whether we send one value per edge or one value per vertex
    
    Returns:
        scalarField : a list of arrays of shape (rod.NumVertices,)
    '''

    W_diag = np.copy(tsf.W_diag_joint_pos)
    useCenterline = np.copy(tsf.getUseCenterline())

    if useSurfDim:
        l0 = np.linalg.norm(np.max(tsf.V, axis=0) - np.min(tsf.V, axis=0))
    else:
        defV = np.array(linkage.deformedPoints())
        l0 = np.linalg.norm(np.max(defV, axis=0) - np.min(defV, axis=0))

    percentScale = (100.0 if usePercent else 1.0)
    
    scalarField = []
    nCPperSeg   = int(linkage.numCenterlinePos() / linkage.numSegments())
    subdivision = nCPperSeg + 3

    tsf.setUseCenterline(linkage, False, 0.1, jointPosValence2Multiplier=1.0)
    devJoints = np.linalg.norm((linkage.jointPositions() - tsf.linkage_closest_surf_pts).reshape(-1, 3), axis=-1)

    tsf.setUseCenterline(linkage, True, 0.1, jointPosValence2Multiplier=1.0)
    dCP = (linkage.centerLinePositions() - tsf.linkage_closest_surf_pts).reshape(-1, 3)

    for i, seg in enumerate(linkage.segments()):
        if perEdge:
            field = np.zeros(shape=(subdivision,))
            field[0]    = devJoints[seg.startJoint]
            field[-1]   = devJoints[seg.endJoint]
            dMidPoints  = (dCP[i*nCPperSeg:(i+1)*nCPperSeg-1, :] + dCP[i*nCPperSeg+1:(i+1)*nCPperSeg, :]) / 2
            devCP       = np.linalg.norm(dMidPoints, axis=-1)
            field[2:-2] = devCP
            field[1]    = (field[0] + devCP[0]) / 2
            field[-2]   = (field[-1] + devCP[-1]) / 2
        else:
            field = np.zeros(shape=(subdivision+1,))
            field[1]    = devJoints[seg.startJoint]
            field[-2]   = devJoints[seg.endJoint]
            devCP       = np.linalg.norm(dCP[i*nCPperSeg:(i+1)*nCPperSeg, :], axis=-1)
            field[2:-2] = devCP

        scalarField.append(percentScale * field / l0)
    
    tsf.setUseCenterline(linkage, useCenterline, sum(W_diag), jointPosValence2Multiplier=max(W_diag) / min(W_diag))
    return scalarField

def HighlightJoints(linkage, curves, jointsIdx, subdivision):
    '''
    Args:
        linkage     : a rod linkage object 
        curves      : list of list giving the joints through which each curve passes
        jointsIdx   : list containing all the joints to be highlighted
        subdivision : number of discrete elements for a single rod
    
    Returns:
        scalarField : a list of arrays of shape (rod.NumVertices,)
    '''
    
    highlightBeginning = []
    highlightEnd       = []
    segmentCpt = 0
    for crv in curves:
        for i in range(len(crv)-1):
            if crv[i] in jointsIdx:
                highlightBeginning.append(segmentCpt)
            if crv[i+1] in jointsIdx:
                highlightEnd.append(segmentCpt)
            segmentCpt += 1
    
    scalarField = []
    for i in range(linkage.numSegments()):
        field = np.zeros(shape=(subdivision,))
        if i in highlightBeginning: field[0]  = 1.0
        if i in highlightEnd:       field[-1] = 1.0
        scalarField.append(field)
        
    return scalarField

def HighlightRestQuantities(dRestQuantities, linkage, subdivision, thresh=0.):
    '''
    Args:
        linkage     : a rod linkage object 
        curves      : list of list giving the joints through which each curve passes
        jointsIdx   : list containing all the joints to be highlighted
        subdivision : number of discrete elements for a single rod
        thresh      : threshold indicating when a DoF is considered active
    
    Returns:
        scalarField : a list of arrays of shape (rod.NumVertices,)
    '''
    
    dRestLengths = dRestQuantities[-linkage.numSegments():]
    dRestKappas  = dRestQuantities[:-linkage.numSegments()]
    
    scalarFieldLengths = []
    for i in range(linkage.numSegments()):
        field = (abs(dRestLengths[i]) > thresh) * np.ones(shape=(subdivision,))
        scalarFieldLengths.append(field)
        
    scalarFieldKappas = []
    for i in range(linkage.numSegments()):
        kappasActive = abs(dRestKappas[i*(subdivision-1):(i+1)*(subdivision-1)]) > thresh
        field = np.zeros(shape=(subdivision,))
        field[1:]  += kappasActive
        field[:-1] += kappasActive
        scalarFieldKappas.append(np.minimum(field, 1.0))
        
    return scalarFieldLengths, scalarFieldKappas

