import json
from typing import Union
import numpy as np

import elastic_rods
import average_angle_linkages
import linkage_optimization
from VisUtilsCurveLinkage import ScalarFieldDeviations

linkageType = Union[
    average_angle_linkages.AverageAngleLinkage,
    average_angle_linkages.AverageAngleSurfaceAttractedLinkage,
]

tsfType = linkage_optimization.TargetSurfaceFitter

def ExtractQuantitiesPerSegment(tsf: tsfType, rodEdgesFamily: np.ndarray, subd: int, linkage: linkageType):
    '''
    Args:
        tsf: a target surface fitter used to compute distances to the target surface
        rodEdgesFamily: an array that give the family of each rod segment in the linkage
        subd: the number of subdivisions per rod segment
        linkage: the linkage from which we want to extract quantities
    
    Returns:
        quantPerSegFamilyA: the per segment extracted quantities for segment of family A
        quantPerSegFamilyB: the per segment extracted quantities for segment of family B
    '''
    E  = linkage.homogenousMaterial().youngModulus
    nu = linkage.homogenousMaterial().youngModulus / (2 * linkage.homogenousMaterial().shearModulus) - 1.0
    height = linkage.homogenousMaterial().crossSectionHeight
    width  = linkage.homogenousMaterial().area / height
    
    if not linkage.hasCrossSection():
        cs = elastic_rods.CrossSection.construct("RECTANGLE", E, nu, [height, width])
        rm = elastic_rods.RodMaterial(cs)
        linkage.setMaterial(rm)
    linkage.meshCrossSection(0.001)
    devField                   = [list(elt) for elt in ScalarFieldDeviations(linkage, tsf, useSurfDim=True, usePercent=True, perEdge=False)]
    maxVMField                 = [list(elt) for elt in linkage.maxVonMisesStresses()]
    sqrtBendingEnergiesField   = [list(elt) for elt in linkage.sqrtBendingEnergies()]
    stretchingEnergiesField    = [list(elt) for elt in linkage.stretchingEnergies()]
    twistingEnergiesField      = [list(elt) for elt in linkage.twistingEnergies()]
    
    quantPerSeg = []

    for segment, vm, bend, stretch, twist, dev in zip(
        linkage.segments(), 
        maxVMField,
        sqrtBendingEnergiesField,
        stretchingEnergiesField,
        twistingEnergiesField,
        devField,
    ):

        segDataTmp = {}
        pos    = [list(defPt) for defPt in segment.rod.deformedPoints()]
        frameX = [list(segment.rod.deformedPoints()[i+1] - segment.rod.deformedPoints()[i]) for i in range(subd)]
        frameY = [list(d1) for d1 in segment.rod.deformedMaterialFramesD1D2()[:subd]]
        frameZ = [list(d2) for d2 in segment.rod.deformedMaterialFramesD1D2()[subd:]]
        segDataTmp['Pos']    = pos
        segDataTmp['FrameX'] = frameX
        segDataTmp['FrameY'] = frameY
        segDataTmp['FrameZ'] = frameZ

        segDataTmp['VonMises'] = vm
        segDataTmp['SqrtBend'] = bend
        segDataTmp['Stretching'] = stretch
        segDataTmp['Twisting'] = twist
        segDataTmp['TargetDeviation'] = dev

        quantPerSeg.append(segDataTmp)

    quantPerSegFamilyA = [quantPerSeg[i] for i in range(len(quantPerSeg)) if rodEdgesFamily[i] == 0]
    quantPerSegFamilyB = [quantPerSeg[i] for i in range(len(quantPerSeg)) if rodEdgesFamily[i] == 1]
    
    return quantPerSegFamilyA, quantPerSegFamilyB

def ConvertCShellToJSON(cshell, filepath, tsf=None):
    '''
    Args:
        cshell: the C-shell we would like to convert
        filepath: the destination
    '''

    if tsf:
        pass
    elif cshell.linkageOptimizer:
        tsf = cshell.linkageOptimizer.get_target_surface_fitter()
    elif cshell.useSAL:
        tsf = cshell.deployedLinkage.get_target_surface_fitter()
    
    flatFamilyA, flatFamilyB = ExtractQuantitiesPerSegment(
        tsf, 
        cshell.rodEdgesFamily, 
        cshell.subdivision, 
        cshell.flatLinkage
    )
    deployFamilyA, deployFamilyB = ExtractQuantitiesPerSegment(
        tsf, 
        cshell.rodEdgesFamily, 
        cshell.subdivision, 
        cshell.deployedLinkage
    )
    
    jointForceResiduals = cshell.deployedLinkage.gradient()[cshell.deployedLinkage.jointPositionDoFIndices()].reshape(-1, 3)
    
    # for backward compatibility
    if 'freeAngles' in dir(cshell): freeAngles = cshell.freeAngles
    else: freeAngles = []
    
    jsonCshell = {
        'TargetSurface': [{
            'Vertices': tsf.V.tolist(),
            'Faces': tsf.F.tolist(),
        }],
        'CrossSection': [cshell.deployedLinkage.homogenousMaterial().crossSection().params()],
        'Flat_FamilyA': flatFamilyA,
        'Flat_FamilyB': flatFamilyB,
        'Deploy_FamilyA': deployFamilyA,
        'Deploy_FamilyB': deployFamilyB,
        'free_joints_idx': freeAngles,
        'JointForceResiduals': jointForceResiduals.tolist(),
    }
    
    with open(filepath, "w") as f:
        json.dump(jsonCshell, f) 

    
    