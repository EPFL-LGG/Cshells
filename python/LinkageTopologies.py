# This file lists the main topology that can be reused over the notebooks

import torch
import math
import numpy as np
PI = math.pi

torch.set_default_dtype(torch.float64)
    
def ToNumpy(tensor):
    return tensor.cpu().detach().clone().numpy()

def RegularTopology(nJa, nJb):
    '''
    Args:
        nJa : number of joints for the first family of curves
        nJb : number of joints for the second family of curves

    Returns:
        nJ           : total number of joints
        curves       : list of list giving the joints through which each curve passes
        curvesFamily : list containing whether the curve is labeled as A (0) or B (1)
    '''

    nJ   = nJa * nJb
    curves  = [[i + j*nJa for i in range(nJa)] for j in range(nJb)]
    curves += [[i + j*nJa for j in range(nJb)] for i in range(nJa)]
    curvesFamily = [0 for j in range(nJb)] + [1 for i in range(nJa)]

    return nJ, curves, curvesFamily

def RegularTopologyRhombus(nJa, nJb):
    '''
    Args:
        nJa: maximum number of joints along curves of family A
        nJb: maximum number of joints along curves of family B
    
    Returns:
        nJ             : total number of joints
        curves         : list of list giving the joints through which each curve passes
        curvesFamily   : list containing whether the curve is labeled as A (0) or B (1)
        dictNewJointId : a dictionnary that filters points laid on a rectangular patch of a regular lattice
    '''
    
    assert nJa % 2 == 1
    assert nJb % 2 == 1
    
    _, curves, curvesFamily = RegularTopology(nJa, nJb)
    
    # Should not remove joints for the 3 middle rods
    rmA  = [int(nJb // 2) - i - 1 for i in range(int(nJb//2))] # Number of joints removed per rod of family A
    rmA += [0] + rmA[::-1]
    # Check if we do not remove too many of the joints
    minA = np.min([nJa - 2 * el for el in rmA]) # Minimum (number of joints left along A rods)
    rmA = [max(0, el + min(0, int((minA - 1) // 2) - 1)) for el in rmA]

    idStartA = rmA.copy()
    idEndA = [nJa - el for el in rmA]

    rmB  = [int(nJa // 2) - i - 1 for i in range(int(nJa//2))]
    rmB += [0] + rmB[::-1]
    minB = np.min([nJb - 2 * el for el in rmB])
    rmB = [max(0, el + min(0, int((minB-1) // 2) - 1)) for el in rmB]

    idStartB = rmB.copy()
    idEndB = [nJb - el for el in rmB]

    curvesA = [crv for crv, crvFam in zip(curves, curvesFamily) if crvFam == 0]
    curvesB = [crv for crv, crvFam in zip(curves, curvesFamily) if crvFam == 1]

    curvesA = [crv[ids:ide] for crv, ids, ide in zip(curvesA, idStartA, idEndA)]
    curvesB = [crv[ids:ide] for crv, ids, ide in zip(curvesB, idStartB, idEndB)]

    curves = curvesA + curvesB

    uniqueJointIds = np.sort(np.unique(np.concatenate([np.array(crv) for crv in curves], axis=0)))
    dictNewJointId = {uniqueJointIds[i]:i for i in range(uniqueJointIds.shape[0])}
    curves = [[dictNewJointId[idj] for idj in crv] for crv in curves]
        
    return len(dictNewJointId), curves, curvesFamily, dictNewJointId

def SpiralTopology(nJinner, nJalong):
    '''
    Args:
        nJinner : number of joints in the inner circle (or number of curves divided by 2)
        nJalong : number of joints along each curve

    Returns:
        nJ           : total number of joints
        curves       : list of list giving the joints through which each curve passes
        curvesFamily : list containing whether the curve is labeled as A (0) or B (1)
    '''

    nJ       = nJinner * nJalong
    curves  = [[i + j*nJalong for i in range(nJalong)] for j in range(nJinner)]
    curves += [[(-i*(nJalong-1) + j*nJalong)%nJ for i in range(nJalong)] for j in range(nJinner)]
    curvesFamily = [0 for j in range(nJinner)] + [1 for i in range(nJinner)]

    return nJ, curves, curvesFamily

def RadialTopology(nJr, nJthetas):
    '''
    Args:
        nJr      : number of joints in the radial direction
        nJthetas : number of joints in the orthoradial direction

    Returns:
        nJ           : total number of joints
        curves       : list of list giving the joints through which each curve passes
        curvesFamily : list containing whether the curve is labeled as A (0) or B (1)
    '''
    nJ           = nJr * nJthetas
    curves       = [[i%nJthetas + j*nJthetas for i in range(nJthetas+1)] for j in range(nJr)]
    curves      += [[i + j*nJthetas for j in range(nJr)] for i in range(nJthetas)]
    curvesFamily = [0 for j in range(nJr)] + [1 for i in range(nJthetas)]

    return nJ, curves, curvesFamily

def GenerateRegularRhombusLayout(nJa, nJb):
    '''
    Args:
        nJa: maximum number of joints along curves of family A
        nJb: maximum number of joints along curves of family B

    Returns:
        uv : the grid of joints laid out so that boundary joints are on the [0, 1]x[0, 1] square, shape (nJ, 3)
    '''

    _, _, _, dictNewJointId = RegularTopologyRhombus(nJa, nJb)
    uniqueJointIds = np.sort(list(dictNewJointId.keys()))
    
    u, v = np.meshgrid(np.linspace(0.0, 1.0, nJa), np.linspace(0.0, 1.0, nJb))
    uv = np.stack([u.reshape(-1,)[uniqueJointIds], v.reshape(-1,)[uniqueJointIds]], axis=1)
    uv -= np.mean(uv, axis=0, keepdims=True)
    uv[:, 1] *= (nJb / nJa)
    theta = np.pi / 4
    # theta = 0.0
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    uv = uv @ rot.T
    uv[:, 0] *= (0.5 / np.max(uv[:, 0]))
    uv[:, 1] *= (0.5 / np.max(uv[:, 1]))
    uv += np.array([0.5, 0.5])
    uv = np.clip(uv, 0.0, 1.0)
    
    return uv