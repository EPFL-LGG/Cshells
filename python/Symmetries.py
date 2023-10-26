import numpy as np
import torch

class CurvesDoFReducer:
    '''
    Attributes:
        __name        : name of the reducer used
        nJ            : total number of joint in the linkage
        curves        : list of list giving the joints through which each curve passes
        curvesFamily  : list containing whether the curve is labeled as A (0) or B (1)
        nCPperRodEdge : number of points to add between edges (list of list with nJcurve-1 elements)
    '''

    def __init__(self, name):
        self.__name        = name
        self.nJ            = None
        self.curves        = None
        self.curvesFamily  = None
        self.nCPperRodEdge = None

    def MapToFullCurvesDoF(self, reducedCurvesDoF):
        '''
        Args:
            reducedCurvesDoF : a tensor of shape (nReducedDoF,)

        Returns:
            curvesDoF : a tensor of shape (nCurvesDoF,)
        '''
        
        print("Please specify the kind of reducer you want to use.")
        raise NotImplementedError

    def GetName(self):
        return self.__name

class RadialDuplicator(CurvesDoFReducer):
    '''
    Additional attributes:
        nJinner       : number of joints in the inner-most circle
        nJalong       : number of joints along each rod 
        nCPperSplineA : number of intermediate control points on each segment along a spline A
        nTotCP_A      : total number of intermediate control points along a spline A
        nCPperSplineB : same as for splines A
        nTotCP_B      : same as for splines A
        rotAngle      : angle by which each spline is rotated
        rot           : all the rotation matrices stacked into a (nJinner, 2, 2) tensor
        rotAngles     : the different angles used to generate the rotation matrice: (nJinner,) tensor

    Note:
        The reducedCurvesDoF should have shape 2*nJalong + nTotCP_A + nTotCP_B
    '''

    def __init__(self, nJinner, nJalong, nCPperSplineA, nCPperSplineB, revertSymmetry=False):
        super().__init__("Radial Duplicator")
        self.nJinner = nJinner
        self.nJalong = nJalong
        self.nJ      = nJinner * nJalong
        self.curves  = [[i + j*nJalong for i in range(nJalong)] for j in range(nJinner)]
        self.curves += [[(i*(nJalong+1) + j*nJalong)%self.nJ for i in range(nJalong)] for j in range(nJinner)]
        self.nJinner = nJinner
        self.curvesFamily = [0 for j in range(nJinner)] + [1 for i in range(nJinner)]

        self.nTotCP_A      = sum(nCPperSplineA)
        self.nTotCP_B      = sum(nCPperSplineB)
        self.nCPperRodEdge = [nCPperSplineA for _ in range(nJinner)] + [nCPperSplineB for _ in range(nJinner)]

        self.rotAngle = 2 * torch.pi / nJinner
        self.rot = torch.zeros((nJinner, 2, 2))
        if revertSymmetry:
            self.rotAngles = torch.linspace(2*torch.pi - self.rotAngle, 0.0, nJinner)
        else:
            self.rotAngles = torch.linspace(0.0, 2*torch.pi - self.rotAngle, nJinner)
        self.rot[:, 0, 0] = torch.cos(self.rotAngles)
        self.rot[:, 0, 1] = - torch.sin(self.rotAngles)
        self.rot[:, 1, 0] = torch.sin(self.rotAngles)
        self.rot[:, 1, 1] = torch.cos(self.rotAngles)

    def MapToFullCurvesDoF(self, reducedCurvesDoF):
        curvesDoF = torch.zeros(size=(2*self.nJ + self.nJinner*(self.nTotCP_A+self.nTotCP_B),))
        rotPos = torch.einsum("ik, jlk->jil", reducedCurvesDoF[:2*self.nJalong].reshape(-1, 2), self.rot).reshape(-1)

        curvesDoF[:2*self.nJ] = rotPos
        curvesDoF[2*self.nJ : 2*self.nJ+self.nJinner*self.nTotCP_A] = reducedCurvesDoF[2*self.nJalong : 2*self.nJalong+self.nTotCP_A].repeat(self.nJinner)
        curvesDoF[-self.nJinner*self.nTotCP_B:] = reducedCurvesDoF[-self.nTotCP_B:].repeat(self.nJinner)

        return curvesDoF

class Reflector(CurvesDoFReducer):
    '''
    Additional attributes:
        nJfree           : number of joints that will be reflected
        nJcons           : number of joints that are constrained to lie on the reflection line
        nJAT             : number of joints after the transformation
        refLinePt        : a point on the reflection line (size (2,))
        refLineDir       : a **normalized** vector pointing in the direction of the reflection line (size (2,))
        refLineOrtho     : a **normalized** vector pointing in the direction orthogonal to the reflection line (size (2,))
        curvesRed        : list of list giving the joints through which each curve passes
        curvesFamilyRed  : list containing whether the curve is labeled as A (0) or B (1)
        nCPperRodEdgeRed : number of points to add between edges (list of list with nJcurve-1 elements)
        mapOffsets       : a matrix that maps the orthogonal offsets of the reduced linkage to the offsets of the full linkage (size (nIntCPtot, nIntCPtotRed)) 

    Note:
        The reducedCurvesDoF should have shape 2*nJfree + nJcons + nIntCPtotRed
    '''

    def __init__(self, nJfree, nJcons, refLinePt, refLineDir, curvesRed, curvesFamilyRed, nCPperRodEdgeRed):
        super().__init__("Reflector")
        self.nJfree          = nJfree
        self.nJcons          = nJcons
        self.nJ              = 2*nJfree + nJcons
        self.refLinePt       = refLinePt
        self.refLineDir      = refLineDir
        self.refLineOrtho    = torch.flip(refLineDir, dims=[0])
        self.refLineOrtho[0] = - self.refLineOrtho[0]

        self.curvesRed        = curvesRed
        self.curvesFamilyRed  = curvesFamilyRed
        self.nCPperRodEdgeRed = nCPperRodEdgeRed
        self.nIntCPcumRed     = list(np.cumsum([0] + [sum(nIntCP) for nIntCP in self.nCPperRodEdgeRed])[:-1])
        self.nIntCPtotRed     = sum([sum(nIntCP) for nIntCP in self.nCPperRodEdgeRed])

        # Handle the curves
        self.curves  = []
        newCurves = []
        self.curvesFamily = self.curvesFamilyRed
        self.nCPperRodEdge = []
        newNIntCP = []

        rowIdx    = 0
        rowIdxNew = 0
        idxMapIntCP = [[], []]
        valMapIntCP = []
        idxMapNewIntCP = [[], []]
        valMapNewIntCP = []

        def FindReflectedCurveIdx(jointID, currFam, curves, curvesFamily):
            '''
            This finds the index of the curve from curves containing jointsID such that 
            its curves family is different from the one specified by currFam
            '''
            for i, (crv, fam) in enumerate(zip(curves, curvesFamily)):
                if fam==1-currFam:
                    if jointID in crv:
                        return i
            raise ValueError("Could not find the reflected curve!")

        for idxCurrCurve, (crv, fam, nIntCP) in enumerate(zip(self.curvesRed, self.curvesFamilyRed, self.nCPperRodEdgeRed)):
            reflect   = sum(np.array(crv) >= nJfree) > 0
            nTotTmpCP = sum(nIntCP)
            # We extend the current curve
            # Here the curve will be reflected so that the reflection joint (aka pivot joint)
            # is at the end of the curve. We then look for the curve that will be reflected and 
            # have the pivot point at the beginning before stitching the two curves.
            if reflect:
                # Check if the first curve should be reflected
                reflectCurrCurve = crv[0] >= nJfree
                assert reflectCurrCurve or crv[-1] >= nJfree # Check if reflection wrt inner joint
                if reflectCurrCurve:
                    crv = crv[::-1]
                    nIntCP = nIntCP[::-1]
                pivotJoint  = crv[-1]
                idxRefCurve = FindReflectedCurveIdx(pivotJoint, fam, self.curvesRed, self.curvesFamilyRed) # Could be made more efficient by saving pairs
                refCurve    = self.curvesRed[idxRefCurve]
                
                # Check if the other curve should be reflected
                reflectOtherCurve = refCurve[-1] >= nJfree
                assert reflectOtherCurve or refCurve[0] >= nJfree # Check if reflection wrt inner joint
                newReflectedCurve = [nJfree+nJcons+jointID for jointID in refCurve]
                nIntCPReflected   = self.nCPperRodEdgeRed[idxRefCurve]
                nTotReflectedCP   = sum(nIntCPReflected)
                if reflectOtherCurve: 
                    newReflectedCurve = newReflectedCurve[::-1]
                    nIntCPReflected   = nIntCPReflected[::-1]
                    
                # Update the curves
                self.curves.append(crv + newReflectedCurve[1:])
                self.nCPperRodEdge.append(nIntCP + nIntCPReflected)
                
                # Add identity blocks.
                idxMapIntCP[0] += list(range(rowIdx, rowIdx+nTotTmpCP))
                colIdx = self.nIntCPcumRed[idxCurrCurve]
                if reflectCurrCurve:
                    idxMapIntCP[1] += list(range(colIdx, colIdx+nTotTmpCP))[::-1]
                else:
                    idxMapIntCP[1] += list(range(colIdx, colIdx+nTotTmpCP))
                valMapIntCP    += nTotTmpCP * [1.]
                rowIdx += nTotTmpCP
                
                idxMapIntCP[0] += list(range(rowIdx, rowIdx+nTotReflectedCP))
                colIdx = self.nIntCPcumRed[idxRefCurve]
                if reflectOtherCurve:
                    idxMapIntCP[1] += list(range(colIdx, colIdx+nTotReflectedCP))[::-1]
                else:
                    idxMapIntCP[1] += list(range(colIdx, colIdx+nTotReflectedCP))
                valMapIntCP    += nTotReflectedCP * [1.]
                rowIdx         += nTotReflectedCP
            
            # We create a new curve
            else:
                self.curves.append(crv)
                newCurves.append([nJfree+nJcons+jointID for jointID in crv])
                self.curvesFamily.append(1-fam)
                
                self.nCPperRodEdge.append(nIntCP)
                newNIntCP.append(nIntCP)
                
                # Add identity blocks.
                colIdx = self.nIntCPcumRed[idxCurrCurve]
                idxMapIntCP[0] += list(range(rowIdx, rowIdx+nTotTmpCP))
                idxMapIntCP[1] += list(range(colIdx, colIdx+nTotTmpCP))
                valMapIntCP    += nTotTmpCP * [1.]
                rowIdx += nTotTmpCP
                
                idxMapNewIntCP[0] += list(range(rowIdxNew, rowIdxNew+nTotTmpCP))
                idxMapNewIntCP[1] += list(range(colIdx,    colIdx+nTotTmpCP))
                valMapNewIntCP    += nTotTmpCP * [-1.]
                rowIdxNew += nTotTmpCP
                
        self.curves        += newCurves
        self.nCPperRodEdge += newNIntCP
        idxMapIntCP[0]     += [idx+rowIdx for idx in idxMapNewIntCP[0]]
        idxMapIntCP[1]     += idxMapNewIntCP[1]
        valMapIntCP        += valMapNewIntCP

        self.mapOffsets = torch.sparse_coo_tensor(idxMapIntCP, valMapIntCP, (len(idxMapIntCP[0]), self.nIntCPtotRed))

        # To avoid re-computing stuff
        self.repRefLineOrtho = self.refLineOrtho.repeat(self.nJfree, 1)
        self.repRefLineDir   = self.refLineDir.repeat(self.nJcons, 1)

    def MapToFullCurvesDoF(self, reducedCurvesDoF):
        # First handle the joints
        ## Compute symmetries
        jointsForSym = reducedCurvesDoF[:2*self.nJfree].reshape(-1, 2)
        jointsSym = 2 * ((self.refLinePt - jointsForSym) @ self.refLineOrtho).reshape(-1, 1) * self.repRefLineOrtho + jointsForSym

        ## Then compute the positions of the joints on the line
        paramsOnLine = reducedCurvesDoF[2*self.nJfree:2*self.nJfree+self.nJcons]
        jointsOnLine = self.refLinePt + paramsOnLine.reshape(-1, 1) * self.repRefLineDir

        ## Aggregate everything
        fullJoints = torch.zeros(size=(2*self.nJfree + 2*self.nJcons + 2*self.nJfree,))
        fullJoints[:2*self.nJfree] = reducedCurvesDoF[:2*self.nJfree]
        fullJoints[2*self.nJfree:2*self.nJfree + 2*self.nJcons] = jointsOnLine.reshape(-1,)
        fullJoints[-2*self.nJfree:] = jointsSym.reshape(-1,)

        # Then map the offsets
        offsets = self.mapOffsets @ reducedCurvesDoF[-self.nIntCPtotRed:]

        # Aggregate
        curvesDoF = torch.zeros(size=(2*self.nJ + offsets.shape[0],))
        curvesDoF[:2*self.nJ] = fullJoints
        curvesDoF[2*self.nJ:] = offsets

        return curvesDoF
