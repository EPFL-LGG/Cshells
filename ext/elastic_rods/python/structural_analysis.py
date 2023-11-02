import numpy as np
import numpy.linalg as la
import elastic_rods

class Load:
    def __init__(self, p = [0, 0, 0], F = [0, 0, 0], T = [0, 0, 0]):
        self.applicationPoint = np.array(p)
        self.netForce = np.array(F)
        self.netTorque = np.array(T)

    # Changes the torque to be around a particular point
    # WARNING: discards application point of net force! This means
    # subsequent `aroundPt` calls will compute the wrong torque.
    def aroundPt(self, c):
        return Load(c, self.netForce,
                    self.netTorque + np.cross(self.applicationPoint - c, self.netForce))

    def verifyZero(self, tol=1e-7):
        fmag = np.linalg.norm(self.netForce)
        tmag = np.linalg.norm(self.netTorque)
        if (fmag > tol): raise Exception('force  imbalance: ' + str(fmag))
        if (tmag > tol): raise Exception('torque imbalance: ' + str(tmag))

    def __add__(self, b):
        if (b == 0): return self # for compatibility with sum
        if (np.any(self.applicationPoint != b.applicationPoint)):
            raise Exception('Addition of loads only implemented for common application point')
        return Load(self.applicationPoint, np.array(self.netForce) + b.netForce, np.array(self.netTorque) + b.netTorque)
    def __radd__(self, b):
        return self.__add__(b)
    def __sub__(self, b):
        return self.__add__(Load(b.applicationPoint, -b.netForce, -b.netTorque))
    def __repr__(self):
        return f'Net force: {self.netForce}\n   torque: {self.netTorque} (around pt {self.applicationPoint})'

# Get index of entities `ti` away from the terminal
def terminalVtxIndex(rod, isStart, ti):
    return ti if isStart else rod.numVertices() - 1 - ti
def terminalEdgeIndex(rod, isStart, ti):
    return ti if isStart else rod.numEdges() - 1 - ti

# Get the net force/torque on an edge according to a particular
# gradient (generalized force vector) g
def getLoadOnEdge(rod, g, edgeIdx):
    F1 = g[3 * edgeIdx:3 * (edgeIdx + 1)]
    F2 = g[3 * (edgeIdx + 1):3 * (edgeIdx + 2)]
    dc = rod.deformedConfiguration()
    edgeMidpt = np.mean(rod.deformedPoints()[edgeIdx:edgeIdx+2], axis=0)
    t = dc.tangent[edgeIdx]
    e = t * dc.len[edgeIdx]
    torqueAround = g[rod.thetaOffset() + edgeIdx]
    return Load(edgeMidpt, F1 + F2, np.cross(-e / 2, F1) + np.cross(e / 2, F2) + torqueAround * t)

# Extract "free body diagram" for a small piece of rod around a joint
# (the centerline and the applied loads); useful for setting up an equivalent
# finite element simulation on the actual joint geometry.
# keepEdges: number of edges to keep on each incident rod segment;
#            in total, `2 * keepEdges - 1` will be kept (one edge is shared)
# verificationTol: how precisely the net force/torque on the piece should vanish.
# return: (polyline, material frame vectors d1, loads)
def isolateRodPieceAtJoint(linkage, ji, ABOffset, keepEdges = 4, verificationTol = 1e-6):
    joint = linkage.joint(ji)
    segmentIdxs = joint.segments_A if ABOffset == 0 else joint.segments_B

    # Return (polyline, [load on joint edge, load on cut material interface])
    # for rod "localIdx" in [0, 1] of segment ABOffset
    def processRod(localIdx):
        si = segmentIdxs[localIdx]
        r = linkage.segment(si).rod
        isStart = joint.isStartA[localIdx] if ABOffset == 0 else joint.isStartB[localIdx]
        tei = lambda ti: terminalEdgeIndex(r, isStart, ti)
        tvi = lambda ti: terminalVtxIndex (r, isStart, ti)
        jointEdgeLoad = getLoadOnEdge(r, r.gradient(), tei(0))

        # For keepEdges=2, we keep the contributions from entities:
        # o-----o-----x--/--x
        # 0  0  1  1
        gsm = elastic_rods.GradientStencilMaskCustom()
        mask = np.zeros(r.numVertices(), dtype=bool)
        mask[0:keepEdges] = True
        if not isStart: mask = np.flip(mask)
        gsm.vtxStencilMask = mask

        mask = np.zeros(r.numEdges(), dtype=bool)
        mask[0:keepEdges] = True
        if not isStart: mask = np.flip(mask)
        gsm.edgeStencilMask = mask

        # For keepEdges=2, we get force / torque imbalances on the entities:
        #       v  v  v
        # o-----o-----x--/--x
        # 0  0  1  1
        materialInterfaceLoad = getLoadOnEdge(r, r.gradient(stencilMask=gsm), tei(keepEdges - 1))

        polylineVertices = [r.deformedPoints()[tvi(ti)] for ti in range(keepEdges + 1)] # ordered points leading away from joint
        dc = r.deformedConfiguration()
        materialFrameVectors = [dc.materialFrame[tei(ti)].d1 for ti in range(keepEdges)]
        if (localIdx == 0):
            polylineVertices = polylineVertices[::-1] # The first rod's points should lead into the joint, not away...
            materialFrameVectors = materialFrameVectors[::-1] # The first rod's points should lead into the joint...

        return (polylineVertices, materialFrameVectors, [jointEdgeLoad, materialInterfaceLoad])

    # Get the net force/torque on the rod's joint edge by summing the overlapping rod gradients
    # Also collect the loads on the two material cut interfaces afterward
    jointEdgeLoad = 0
    loads = []
    polyline = []
    materialFrameVectors = []
    ns = [joint.numSegmentsA, joint.numSegmentsB][ABOffset]
    for localRodIdx in range(2):
        if (localRodIdx >= ns): continue
        rodPolyline, rodMaterialFrameD1, rodLoads = processRod(localRodIdx)
        jointEdgeLoad += rodLoads[0]
        loads += rodLoads[1:]
        polyline += rodPolyline
        materialFrameVectors += rodMaterialFrameD1
        # print("joint edge load contribution: ", rodLoads[0])
    loads = [jointEdgeLoad] + loads

    # Delete duplicate joint edge from middle (but not if there's just one segment...)
    polyline = np.array(polyline)
    materialFrameVectors = np.array(materialFrameVectors)
    if (ns == 2):
        polyline = np.delete(polyline, [len(polyline) // 2, len(polyline) // 2 + 1], axis=0)
        materialFrameVectors = np.delete(materialFrameVectors, [len(materialFrameVectors) // 2], axis=0)

    # Validate that they match the force/torque computed from the joint variable gradient components
    g = linkage.gradient()
    jdo = linkage.dofOffsetForJoint(ji)
    # load from rod A onto joint = - load on joint edge for rod A = load on joint edge for rod B
    forceAndTorqueOnJointFromA = linkage.rivetNetForceAndTorques()[ji, :]
    forceSign = [-1.0, 1.0][ABOffset]
    jointEdgeLoadFromLinkageGradient = Load(joint.position,
                                            forceAndTorqueOnJointFromA[0:3] * forceSign,
                                            forceAndTorqueOnJointFromA[3:6] * forceSign)
    # print(jointEdgeLoadFromLinkageGradient)
    # print(jointEdgeLoad)
    (jointEdgeLoadFromLinkageGradient - jointEdgeLoad).verifyZero(verificationTol)
    # print(jointEdgeLoad.netForce, jointEdgeLoad.netTorque)

    # print("Material cut interface loads:")
    # print(loads[1].aroundPt(joint.position) + loads[2].aroundPt(joint.position))

    # Verify that all forces/torques on the segment balance
    sum([load.aroundPt(joint.position) for load in loads]).verifyZero(verificationTol)
    return (polyline, materialFrameVectors, loads)

# Get the min/max bending and twisting stress acting on the linkage
def stressesOnJointRegions(linkage, edgeDist = 3):
    nj = linkage.numJoints()
    maxBendingStresses  = np.zeros((nj, 2))
    minBendingStresses  = np.zeros((nj, 2))
    maxTwistingStresses = np.zeros((nj, 2))
    for ji in range(nj):
        j = linkage.joint(ji)
        for local_si in range(2):
            ns = [j.numSegmentsA, j.numSegmentsB][local_si]
            for si in ([j.segments_A, j.segments_B][local_si])[:ns]:
                isStart = j.terminalEdgeIdentification(si)[2]
                r = linkage.segment(si).rod
                bs = r.bendingStresses()
                ts = r.twistingStresses()
                regionStiffnesses = np.array([[bs[vi, 0], bs[vi, 1], ts[vi]] for vi in [terminalVtxIndex(r, isStart, ti) for ti in range(1, edgeDist + 1)]])
                maxBendingStresses [ji, local_si] = max(maxBendingStresses [ji, local_si], np.max(regionStiffnesses[:, 0]))
                minBendingStresses [ji, local_si] = min(minBendingStresses [ji, local_si], np.min(regionStiffnesses[:, 1]))
                maxTwistingStresses[ji, local_si] = max(maxTwistingStresses[ji, local_si], np.max(regionStiffnesses[:, 2]))
    return (maxBendingStresses, minBendingStresses, maxTwistingStresses)

def freeBodyDiagramReport(l, ji, lsi, keepEdges = 4, verificationTol=5e-6):
    centerlinePts, materialFrameD1, loads = isolateRodPieceAtJoint(l, ji, lsi, keepEdges, verificationTol=verificationTol)
    j = l.joint(ji)
    print(f"Rod segment(s) {[j.segments_A, j.segments_B][lsi][:[j.numSegmentsA, j.numSegmentsB][lsi]]} around joint {ji} at {j.position}, normal {j.normal}")
    print("Centerline points:\n", centerlinePts)

    e = np.diff(centerlinePts, axis=0)
    print("\nCenterline tangents:\n", e / np.linalg.norm(e, axis=1)[:, np.newaxis])
    print("\nCross-section frame vectors d1:\n", materialFrameD1)

    print("\nActuation torque:\t", np.abs(np.dot(loads[0].netTorque, j.normal)))
    print("Out of plane torque from rivet (shear torque):\t", np.linalg.norm(loads[0].netTorque - np.dot(loads[0].netTorque, j.normal) * j.normal))
    print(f"Load on joint edge:\n{loads[0]}")
    print(f"\nLoad on sliced material interface 1:\n{loads[1]}")
    if (len(loads) > 2):
        print(f"\nLoad on sliced material interface 2:\n{loads[2]}")

def weavingCrossingForceMagnitudes(linkage, omitBoundary = False):
    """
    Get the separation force and tangential forces between the ribbons crossing at each joint.
    A positive separation force means the ribbons are trying to pull apart (and need pins).
    """
    AForceOnJoint = linkage.rivetNetForceAndTorques()[:, 0:3]
    ATorqueOnJoint = linkage.rivetNetForceAndTorques()[:, 3:6]
    if omitBoundary:
        for ji, j in enumerate(linkage.joints()):
            if (j.valence() < 4): AForceOnJoint[ji] = 0.0
            if (j.valence() < 4): ATorqueOnJoint[ji] = 0.0
    result = []
    for ji in range(len(AForceOnJoint)):
        f = AForceOnJoint[ji]
        torque = ATorqueOnJoint[ji]
        j = linkage.joint(ji)
        separationDirection = j.normal * (1 if j.type == j.Type.A_OVER_B else -1)
        separationForce = f.dot(separationDirection)
        tangentialForce = np.sqrt(f.dot(f) - separationForce**2)
        result.append([separationForce, tangentialForce, la.norm(torque)])
    return np.array(result)

from matplotlib import pyplot as plt
def weavingCrossingAnalysis(linkage, omitBoundary = False):
    cfm = weavingCrossingForceMagnitudes(linkage, omitBoundary)
    separationForce = cfm[:, 0]
    coefficientOfFriction = cfm[:, 1] / (-separationForce)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 4, 1)
    plt.xlim((separationForce.min(), max([0, separationForce.max()])))
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.title('Separation Forces')
    plt.xlabel('Separation Force Mag.')
    plt.ylabel('Number of Crossings')
    plt.hist(separationForce, 200);

    plt.subplot(1, 4, 2)
    plt.title('Tangential Forces')
    plt.xlabel('Tangential Force Mag.')
    plt.ylabel('Number of Crossings')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.hist(cfm[:, 1], 100);
    plt.tight_layout()

    plt.subplot(1, 4, 3)
    plt.title('Torques')
    plt.xlabel('Torque Mag.')
    plt.ylabel('Number of Crossings')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.hist(cfm[:, 2], 100);
    plt.tight_layout()

    plt.subplot(1, 4, 4)
    plt.title('Required Static Coefficient of Friction')
    plt.xlabel('Required $\mu$')
    plt.ylabel('Number of Crossings')
    plt.hist(coefficientOfFriction, 100);
    plt.tight_layout()
    print("Coefficient of Friction Percentiles 50, 75, 90: ",
            np.percentile(coefficientOfFriction, 50),
            np.percentile(coefficientOfFriction, 75),
            np.percentile(coefficientOfFriction, 90))

def crossingSegmentForceFields(linkage, omitBottom = False, omitBoundary = False):
    """
    Get the elastic forces acting from the rods on the joints
    as a per-segment per-edge vector field.

    omitBottom: only output vectors for the "top"/outer segment.
    omitBoundary: ignore forces on joints with valence < 4
    """
    AForceOnJoint = linkage.rivetNetForceAndTorques()[:, 0:3]
    ATorqueOnJoint = linkage.rivetNetForceAndTorques()[:, 3:6]
    if omitBoundary:
        for ji, j in enumerate(linkage.joints()):
            if (j.valence() < 4): AForceOnJoint[ji] = 0.0

    nj = linkage.numJoints()
    separationForceField = []
    tangentialForceField = []
    torqueField = []
    for si, s in enumerate(linkage.segments()):
        ne = s.rod.numEdges()
        sf = np.zeros((ne, 3))
        tf = np.zeros((ne, 3))
        torque = np.zeros((ne, 3))
        for endpt, ji in enumerate([s.startJoint, s.endJoint]):
            if (ji > nj): continue
            j = linkage.joint(ji)
            topSegmentIsB = 0 if j.type == j.Type.A_OVER_B else 1
            thisSegmentIsB = j.terminalEdgeIdentification(si)[1] != 0.0
            thisIsTop = (topSegmentIsB == thisSegmentIsB)
            if (omitBottom and not thisIsTop): continue # only draw arrows on top
            # if j.terminalEdgeIdentification(si)[topSegmentIsB] == 0.0: continue # only attach arrows to segment on top
            terminalEdge = ne - 1 if endpt else 0
            # Get the elastic forces acting from this segment on the crossing.
            f = (-1.0 if thisSegmentIsB else 1.0) * AForceOnJoint[ji]
            n = j.normal if thisIsTop else -j.normal # separation direction for this segment
            t = (-1.0 if thisSegmentIsB else 1.0) * ATorqueOnJoint[ji]
            sf[terminalEdge] = np.clip(f.dot(n), 0, None) * n
            tf[terminalEdge] = f - f.dot(n) * n
            torque[terminalEdge] = t

        separationForceField.append(sf)
        tangentialForceField.append(tf)
        torqueField.append(torque)
    return {'separation': separationForceField,
            'tangential': tangentialForceField, 
            'torque': torqueField}

import linkage_vis
from ipywidgets import HBox
class crossingForceFieldVisualization():
    def __init__(self, linkage, omitBottom = False, omitBoundary = False):
        self.forces = crossingSegmentForceFields(linkage, omitBottom, omitBoundary)
        self.separationView = linkage_vis.LinkageViewer(linkage, vectorField=self.forces['separation'])
        self.tangentialView = linkage_vis.LinkageViewer(linkage, vectorField=self.forces['tangential'])
        self.torqueView = linkage_vis.LinkageViewer(linkage, vectorField=self.forces['torque'])
        self.separationView.averagedMaterialFrames = True
        self.tangentialView.averagedMaterialFrames = True
        self.torqueView.averagedMaterialFrames = True

    def maxForce(self):
        return (np.max([np.linalg.norm(sf, axis=1) for sf in self.forces['separation']]),
                np.max([np.linalg.norm(tf, axis=1) for tf in self.forces['tangential']]),
                np.max([np.linalg.norm(tf, axis=1) for tf in self.forces['torque']]))

    def show(self):
        return HBox([self.separationView.show(), self.tangentialView.show(), self.torqueView.show()])
