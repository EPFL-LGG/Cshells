import os
import sys as _sys

SCRIPT_PATH = os.path.abspath(os.getcwd())
split = SCRIPT_PATH.split("Cshells")
if len(split)<2:
    print("Please rename the repository 'Cshells'")
    raise ValueError
PATH_TO_CUBICSPLINES = split[0] + "Cshells/ext/torchcubicspline"
_sys.path.append(PATH_TO_CUBICSPLINES)

import torch
from torchcubicspline import (natural_cubic_spline_coeffs, 
                              NaturalCubicSpline, NaturalCubicSplineWithVaryingTs)

torch.set_default_dtype(torch.float64)

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

def MakeConstantSpeed(initKnots, multiplier):
    '''
    Args:
        initKnots  : an array of shape (?, n_knots, 2 or 3) containing the initial knot positions
        multiplier : the factor by which the number of knot is multiplied

    Returns:
        newSplines     : ? new splines
        sKnots         : parameter associated to each knot shape (?, n_knots)
        refinedS       : new knots' parameter (?, n_knots*multiplier)
        newKnots       : new knots' (?, n_knots*multiplier, 2 or 3)
    '''
    initTs     = torch.linspace(0.0, 1.0, initKnots.shape[1])
    initCoeffs = natural_cubic_spline_coeffs(initTs, initKnots)
    splines    = NaturalCubicSpline(initCoeffs)
    
    # Start arc-length reparameterization
    refinedTs      = torch.linspace(0, 1, (initKnots.shape[1] - 1) * multiplier + 1)
    newKnots       = splines.evaluate(refinedTs)
    lengths        = torch.norm(newKnots[:, 1:, :] - newKnots[:, :-1, :], dim=2)
    cumLen         = torch.cumsum(lengths, dim=1)
    refinedS       = torch.cat([torch.zeros(size=(initKnots.shape[0], 1)), cumLen], dim=1) / cumLen[:, -1].reshape(-1, 1)
    
    # New splines
    newCoeffs  = natural_cubic_spline_coeffs(refinedS, newKnots)
    newSplines = NaturalCubicSplineWithVaryingTs(newCoeffs)

    # Parameters associated to the old knots
    sKnots = refinedS[:, ::multiplier].contiguous()
    
    return newSplines, sKnots, refinedS, newKnots
