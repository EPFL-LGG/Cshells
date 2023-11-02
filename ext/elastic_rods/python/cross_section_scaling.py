import numpy as np
from numpy.linalg import norm

import scipy
import scipy.interpolate
import elastic_rods

class CrossSectionInterpolator:
    def __init__(self, cs1, cs2, nFitSamples = 30):
        self.cs1 = cs1
        self.cs2 = cs2

        # Generate the material sample points for fitting
        self.sample_x    = np.linspace(0, 1, nFitSamples)
        self.sample_mats = [elastic_rods.RodMaterial(elastic_rods.CrossSection.lerp(cs1, cs2, x)) for x in self.sample_x]

        # Fit the material properties using quintic splines
        spline_interpolated_properties = [
            'area',
            'stretchingStiffness',
            'twistingStiffness',
            'B11', 'B22',
            'I11', 'I22',
            'torsionStressCoefficient',
            'youngModulus',
            'shearModulus'
        ]

        interpolationDegree = {p: 5 for p in spline_interpolated_properties}
        interpolationDegree['youngModulus'] = 1
        interpolationDegree['shearModulus'] = 1

        # Note: we will  manually interpolate cross-section parameters/height
        # with piecwise linear interpolation. (We cannot simply interpolate the
        # two endpoints since this will not account for the cross-section
        # rotation needed to diagonalize the B and I tensors.)

        self.splines = {}
        for p in spline_interpolated_properties:
            sample_data = [getattr(m, p) for m in self.sample_mats]
            spl = scipy.interpolate.splrep(self.sample_x, sample_data, k=interpolationDegree[p])
            self.splines[p] = lambda x, spl=spl: scipy.interpolate.splev(x, spl) # spl=spl is needed to make a local reference to the current value of spl!

    # Get the material at interpolation parameter alpha in [0, 1]
    def material(self, alpha):
        if (alpha < 0) or (alpha > 1): raise Exception('Extrapolation unsupported (alpha not in [0, 1])')

        samples = self.sample_mats
        # Avoid explicitly handling edge-cases of cross-section interpolation
        if (alpha == 0): return samples[0]
        if (alpha == 1): return samples[-1]

        result = elastic_rods.RodMaterial()
        # Set all calculated properties by interpolation
        for p, spline in self.splines.items():
            setattr(result, p, spline(alpha))

        # Manually interpolate the cross-section geometry/height with piecewise linear interpolation
        upperBound = np.searchsorted(self.sample_x, alpha, side='right')
        left, right = upperBound - 1, upperBound
        alphaRange = [self.sample_x[left], self.sample_x[right]]

        result.crossSectionHeight      = scipy.interpolate.interp1d(alphaRange, np.array([samples[left].crossSectionHeight,      samples[right].crossSectionHeight     ]), axis=0, bounds_error=True)(alpha)
        result.crossSectionBoundaryPts = scipy.interpolate.interp1d(alphaRange, np.array([samples[left].crossSectionBoundaryPts, samples[right].crossSectionBoundaryPts]), axis=0, bounds_error=True)(alpha)
        result.crossSectionBoundaryEdges = samples[0].crossSectionBoundaryEdges

        return result

    def __call__(self, alpha):
        return self.material(alpha)

# Gets scale factors in the normalized range [0, 1]
def density_based_scale_factors(linkage, smoothingIters = 10, smoothingStepSize = 0.25):
    adj = [j.neighbors() for j in linkage.joints()]
    incidentSegments = [[si for si in (j.segments_A + j.segments_B) if si < linkage.numSegments()] for j in linkage.joints()]

    # We define the "radius" at each joint as the average arclength to its neighbors
    # jointRadius = np.array([np.mean([norm(linkage.joint(ju).position - linkage.joint(jv).position) for jv in neighbors]) for ju, neighbors in enumerate(adj)])
    jointRadius = np.array([np.mean([linkage.segment(si).rod.totalRestLength() for si in segments]) for segments in incidentSegments])

    # We smooth the radius field by iteratively averaging with the joint neighbors
    # (Using uniform Laplacian)
    for i in range(smoothingIters):
        newJointRadius = (1 - smoothingStepSize) * jointRadius + smoothingStepSize * np.array([np.mean(jointRadius[n]) for n in adj])
        jointRadius = newJointRadius

    return np.interp(jointRadius, [jointRadius.min(), jointRadius.max()], [0.0, 1.0])

def apply_density_based_cross_sections(linkage, smallCrossSection, largeCrossSection, smoothingIters = 10, smoothingStepSize = 0.25):
    jointScales = density_based_scale_factors(linkage, smoothingIters, smoothingStepSize)

    csi = CrossSectionInterpolator(smallCrossSection, largeCrossSection)
    for s in linkage.segments():
        scales = []
        if (s.hasStartJoint()): scales.append(jointScales[s.startJoint])
        if (s.hasEndJoint()):   scales.append(jointScales[s.endJoint])
        if (len(scales) == 0): raise Exception('No incident joints')
        if (len(scales) == 1):
            s.rod.setMaterial(csi(scales[0]))
            continue

        # Linearly interpolate scale along the arclength from the first edge midpoint to the last
        rl = np.array(s.rod.restLengths())
        dualLengths = 0.5 * (rl[1:] + rl[:-1])           # distance between edge midpoints
        arclens = np.pad(np.cumsum(dualLengths), (1, 0)) # arclength position of each edge midpoint
        edgeScales = np.interp(arclens / arclens[-1], [0.0, 1.0], scales)
        edgeMaterials = [csi(scale) for scale in edgeScales]
        s.rod.setMaterial(edgeMaterials)
