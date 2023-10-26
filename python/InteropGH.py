import average_angle_linkages
from bending_validation import suppress_stdout as so
import contextlib
from CShell import CShell, IndicesDiscretePositionsNoDuplicate
from datetime import datetime
import elastic_rods
from functools import reduce
from linkage_vis import LinkageViewer, LinkageViewerWithSurface
import numpy as np
from open_average_angle_linkage import open_average_angle_linkage
from operator import concat
import py_newton_optimizer
import re
from specklepy.api.client import SpeckleClient
from specklepy.objects.base import Base
from specklepy.objects.geometry import Mesh, Polyline, Point
from specklepy.objects.other import RenderMaterial
from specklepy.transports.server import ServerTransport
from specklepy.api import operations
from Symmetries import RadialDuplicator
import torch
from typing import Any
from vis.fields import ScalarField
from VisUtilsCurveLinkage import ScalarFieldDeviations

import MeshFEM
import ElasticRods
import mesh

try:
    KNITRO_FOUND = True
    import cshell_optimization
except Exception as e:
    KNITRO_FOUND = False
    print("Knitro may not have been found: {}.".format(e))

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

class Interop():
    client : SpeckleClient = None
    stream_id : str = None
    transport : ServerTransport = None
    
    def __init__(self, token : str, stream_name : str, stream_index=0) -> None:
        with contextlib.redirect_stdout(None):
            # initialise the client and authenticate the client with the token
            self.client = SpeckleClient(host="https://speckle.xyz")
            self.client.authenticate(token)
            # stream index is used in case of different streams with the same name
            self.stream_id = self.client.stream.search(stream_name)[stream_index].id
            # Create a server transport (used for sending and receiving)
            self.transport = ServerTransport(self.stream_id, self.client)

    def get_stream(self):
        with contextlib.redirect_stdout(None):  
            return self.client.stream.get(self.stream_id)

    def send_data(self, data : Base, branch_name :str = "main", branch_description : str = ""):
        with contextlib.redirect_stdout(None):
            # Serialise the data and send to the transport
            data_id = operations.send(base=data, transports=[self.transport])

            if branch_name != "main": 
                # Check if the branch exist
                branches = self.client.branch.list(self.stream_id)
                has_res_branch = any(b.name == branch_name for b in branches)
                if not has_res_branch:
                    self.client.branch.create(self.stream_id, branch_name, branch_description)

            # Send the data 
            date_time = datetime.now()
            msg = date_time.strftime('%Y%m%d_%H%M%S_CShell')
            commid_id = self.client.commit.create(self.stream_id, data_id, branch_name, message=msg)

        return commid_id

    def receive_data(self, branch_name="main", commitItem : int = 0):
        with contextlib.redirect_stdout(None):
            # Get the last commit from the branch
            commit = self.client.branch.get(self.stream_id, branch_name).commits.items[commitItem]
            receivedData = operations.receive(commit.referencedObject, self.transport)
            
            if '@Data' in dir(receivedData): dataTag = '@Data'
            elif '@data' in dir(receivedData): dataTag = '@data'
            elif 'Data' in dir(receivedData): dataTag = 'Data'
            else: raise ValueError("Unable to read the data from Speckle.")
            
            data = receivedData[dataTag]
            
            r = re.compile("@\{([0-9];)+[0-9]\}") # To match the annoying {0;3;0}-like tags
            listTags = list(filter(r.match, dir(data)))
            if len(listTags) == 1:
                data = data[listTags[0]]

            flag = True
            while(flag):
                data = reduce(concat, data)
                flag = isinstance(data, list)
                
        return data

def pull_linkage(interop : Interop, branch : str, get_flat_linkage : bool, commitItem : int = 0, swapAxesCS : bool = False):
    data = interop.receive_data(branch, commitItem=commitItem)
    if(get_flat_linkage):
        data = data['FlatLinkage']
    else:
        data = data['DeployLinkage']

    # Joints
    joints_data = data['Joints']
    joints = []
    for jd in joints_data:
        pos = np.array(jd['Position'], order='F')
        omega = np.array(jd['Omega'], order='F', dtype=float)
        source_normal = np.array(jd['SourceNormal'], order='F', dtype=float)
        source_tangent = np.array(jd['SourceTangent'], order='F', dtype=float)
        isStartA = list(jd['IsStartA'])
        isStartB = list(jd['IsStartB'])
        segmentsA = list(jd['SegmentsA'])
        segmentsB = list(jd['SegmentsB'])
        normalSigns = list(jd['NormalSigns'])

        joints.append(elastic_rods.RodLinkage.Joint.fromGHState(pos, omega, float(jd['Alpha']), float(jd['LenA']), float(jd['LenB']), 
                                                                float(jd['SignB']), source_tangent, source_normal, segmentsA, segmentsB, 
                                                                isStartA, isStartB, int(jd['JointType']), normalSigns))
    #RodSegments
    segments_data = data['RodSegments']
    rodSegments = []
    for sg in segments_data:
        #Rest points
        pts = np.array(sg['RestPoints'])  
        #Rest Directors
        dirCoords = [coord for dir in sg['RestDirectors'] for dir_item in dir for coord in dir_item]
        # Rest kappas
        restKappas = sg['RestKappas']
        # Rest Twist
        restTwists = sg['RestTwists']
        # Rest Lengths
        restLengths = sg['RestLengths']
        # Twisting stiffness
        twistingStiffness = sg['TwistingStiffnesses']
        # Stretching stiffness
        stretchingStiffness = sg['StretchingStiffnesses']
        # Densities
        densities = sg['Densities']
        # Initial MinRestLength
        initialMinRestLen = sg['InitialMinRestLength']
        # Bending stiffnesses
        bendingStiffnesses1 = [item for item in sg['BendingStiffnesses'][0]]
        bendingStiffnesses2 = [item for item in sg['BendingStiffnesses'][1]]

        # Edge Material
        material_data = sg['EdgeMaterials']
        materials = []
        for mat in material_data:
            ptCoords = [coord for p2d in mat['CrossSectionBoundaryPts'] for coord in p2d ]
            edges = [idx for edge in mat['CrossSectionBoundaryEdges'] for idx in edge]
            materials.append(elastic_rods.RodMaterial.fromGHState(mat['Area'], mat['StretchingStiffness'], mat['TwistingStiffness'], mat['BendingStiffness'], mat['MomentOfInertia'],
                                                mat['TorsionStressCoefficient'], mat['YoungModulus'], mat['ShearModulus'], mat['CrossSectionHeight'], ptCoords, edges))

        # Deformed state
        deformed_data = sg['DeformedConfiguration']
        ptCoords = [coord for p3d in deformed_data['Points'] for coord in p3d ]
        tgtCoords = [coord for v3d in deformed_data['SourceTangent'] for coord in v3d ]
        dirCoords = [coord for v3d in deformed_data['SourceReferenceDirectors'] for coord in v3d ]
        deform_state = elastic_rods.ElasticRod.DeformedState.fromGHState(ptCoords, deformed_data['Thetas'], tgtCoords, dirCoords, deformed_data['SourceTheta'], deformed_data['SourceReferenceTwist'] )

        #Bending Energy Type
        energy_type = elastic_rods.BendingEnergyType.Bergou2008 if sg['BendingEnergyType'] == 0 else elastic_rods.BendingEnergyType.Bergou2010                            

        # Rod Segment 
        rodSegments.append(elastic_rods.RodLinkage.RodSegment.fromGHState(int(sg['StartJoint']), int(sg['EndJoint']), pts, dirCoords, restKappas, restTwists, restLengths, materials, bendingStiffnesses1, bendingStiffnesses2, 
                                                    twistingStiffness, stretchingStiffness, energy_type, deform_state, densities, initialMinRestLen))

    #Material
    section_data = int(data['CrossSectionType'])
    section  = 'RECTANGLE'
    if (section_data==1):
        section = 'ELLIPSE'
    elif (section_data==2):
        section = 'L'
    elif (section_data==3):
        section = 'I'
    elif (section_data==4):
        section = '+'
        
    if ('Width' in dir(data)) and ('Height' in dir(data)) and (section == 'RECTANGLE'):
        material = elastic_rods.RodMaterial('rectangle', 
                                            float(data['E']), 
                                            float(data['PoisonsRatio']), 
                                            [float(data['Width']), float(data['Height'])], 
                                            keepCrossSectionMesh=True)
    else:
        assert 'MatParameters' in dir(data)
        matParams = data['MatParameters']
        if swapAxesCS: stiffAxis = (elastic_rods.StiffAxis.D1 if matParams[0] < matParams[1] else elastic_rods.StiffAxis.D2)
        else         : stiffAxis = (elastic_rods.StiffAxis.D1 if matParams[0] > matParams[1] else elastic_rods.StiffAxis.D2)
        material = elastic_rods.RodMaterial(section, 
                                            float(data['E']), 
                                            float(data['PoisonsRatio']), 
                                            matParams,
                                            stiffAxis=stiffAxis,
                                            keepCrossSectionMesh=True)

    # MinRestLength
    initMinRestLength = float(data['InitialMinRestLength'])

    # SegmentRestLength
    perSegmentRestLength = [ float(rl) for rl in data['PerSegmentRestLength'] ]

    # SegmentRestLenToEdgeRestLenMapTranspose
    map_data = data['SegmentRestLenToEdgeRestLenMapTranspose']
    Ai = [ int(i) for i in map_data['Ai'] ]
    Ap = [ int(p) for p in map_data['Ap'] ]
    Ax = [ float(x) for x in map_data['Ax'] ]
    M = int(map_data['M'])
    N = int(map_data['N'])
    NZ = int(map_data['NZ'])

    # Design Parameters
    dpc_data = data['DesignParamConfig']
    use_restLen = bool(dpc_data['RestLength'])
    use_restKappa = bool(dpc_data['RestKappa'])

    l = elastic_rods.RodLinkage.fromGHState(joints, rodSegments, material, initMinRestLength, Ax, Ai, Ap, M, N, NZ, perSegmentRestLength, use_restLen, use_restKappa)

    if not get_flat_linkage:
        vertsFaces, tarJP = pull_target_surface(data)
        return l, vertsFaces, tarJP
    else:
        return l
 
def pull_target_surface(data):
    '''Pulls the target surface.
    
    Args:
        data: an object pulled from Speckle that contains elements related to the target surface
        
    Ouput:
        vertsFaces: a list of arrays containing the vertices and the faces of the target surface
        tarJP: the target joints positions
        featJointIndices: the indices of feature joints
    '''
    tarJP = None
    featJointIndices = None
    vertsFaces = [
        np.array(data['TargetSurface']['Mesh'].vertices).reshape(-1,3),
        np.array(data['TargetSurface']['Mesh'].faces).reshape(-1,4)[:,1:]
    ]
    if "JointPosOnMesh" in dir(data['TargetSurface']): 
        tarJP = np.array(data['TargetSurface']["JointPosOnMesh"]).reshape(-1,3)
    if "FeatureJoints" in dir(data['TargetSurface']):
        featJointIndices = data['TargetSurface']['FeatureJoints']
    return vertsFaces, tarJP, featJointIndices

def pull_full_linkage(interop : Interop, branchCShell : str, targetMeshUser : dict = None, optimizeAlpha : bool = True, multAlpha : float = 1.0,
                       linkagesGuess = None, commitItem : int = 0, rodMaterial : elastic_rods.RodMaterial = None,
                       flatOnly : bool = False, forceReconstruct : bool = False, swapAxesCS : bool = False, 
                       radialSymmetry : bool = False, revertSymmetry : bool = False,):
    '''
    Pull the curve network (curve data for CShell initialization) from the webapp (Grasshopper => Notebook)

    Args:
        interop          : interoperability object
        branch           : name of the branch from where to retreive the curve network
        targetMeshUser   : the target surface to use
        optimizeAlpha    : whether or not the cshell we keep the target average opening angle fixed
        multAlpha        : to over-open the linkage
        linkagesGuess    : dictionnary containing "flat" and "deployed" linkages
        commitItem       : the commit to look at on Speckle
        rodMaterial      : the rod material that will override the material fetched on Speckle
        flatOnly         : whether we deployed the flat state or not
        forceReconstruct : whether we want to try re-deploying the cshell or not
        swapAxesCS       : if True, the axis of the cross section orthogonal to the surface is made stronger
        radialSymmetry   : whether we apply radial symmetry to the design
        revertSymmetry   : in case of radial symmetry, reverts the curves ordering

    Ouput:
        cshell : cshell object
    '''
    # Receive data from the curve_netwrok branch as a tree structure. This handles the previous data
    data = interop.receive_data(branchCShell, commitItem=commitItem)
    
    curve_network_data = data['CurveNetwork']
    # CurvesDoFs
    temp = curve_network_data['CurvesDoF']
    curvesDoF = torch.tensor(temp)

    # Curves
    temp = curve_network_data['Curves']
    curves = []
    nCPperRodEdge = []
    curvesFamily = []
    for crv_data in temp:
        curves.append(crv_data['Indexes'])
        nCPperRodEdge.append(crv_data['NumControlPoints'])
        curvesFamily.append(crv_data['CurveFamily'])

    subdivision = int(curve_network_data['Subdivision'])
    alphaTar = multAlpha * np.deg2rad(float(curve_network_data['Angle']))
    numJoints = int(curve_network_data['NumJoints'])
    if not 'MatType' in dir(curve_network_data):
        print('Material is set to RECTANGLE by default, could not find MatType on Speckle.')
        matType = 'rectangle'
    else:
        matType = curve_network_data['MatType']
    if ('Width' in dir(curve_network_data)) and ('Height' in dir(curve_network_data)) and (matType.lower() == 'rectangle'):
        material = elastic_rods.RodMaterial('rectangle', 
                                            float(curve_network_data['E']), 
                                            float(curve_network_data['PoisonsRatio']), 
                                            [float(curve_network_data['Width']), float(curve_network_data['Height'])], 
                                            keepCrossSectionMesh=True)
    else:
        assert 'MatParameters' in dir(curve_network_data)
        matParams = curve_network_data['MatParameters']
        if swapAxesCS: stiffAxis = (elastic_rods.StiffAxis.D1 if matParams[0] < matParams[1] else elastic_rods.StiffAxis.D2)
        else         : stiffAxis = (elastic_rods.StiffAxis.D1 if matParams[0] > matParams[1] else elastic_rods.StiffAxis.D2)
        material = elastic_rods.RodMaterial(matType, 
                                            float(curve_network_data['E']), 
                                            float(curve_network_data['PoisonsRatio']), 
                                            matParams,
                                            stiffAxis=stiffAxis,
                                            keepCrossSectionMesh=True)
    target_srf = None
    attractionMesh = None
    
    if targetMeshUser is not None:
        attractionMesh = targetMeshUser
        target_srf = [targetMeshUser["V"], targetMeshUser["F"]]
        tarJP = targetMeshUser["targetJP"]

    if (linkagesGuess is None) and (not forceReconstruct):
        if data.ContainsFlatLinkage and data.ContainsDeployLinkage:
            linkagesGuess = {}
            # Flat first
            rodLinkage = pull_linkage(interop, branchCShell, True, commitItem=commitItem) 
            linkagesGuess['flat'] = average_angle_linkages.AverageAngleLinkage(rodLinkage)
            # Deployed then
            if targetMeshUser is not None:
                rodLinkage, _, tarJP = pull_linkage(interop, branchCShell, False, commitItem=commitItem)
            else:
                rodLinkage, target_srf, tarJP = pull_linkage(interop, branchCShell, False, commitItem=commitItem)
            useSAL = isinstance(rodLinkage, elastic_rods.SurfaceAttractedLinkage) or (not target_srf is None)
            l = average_angle_linkages.AverageAngleLinkage(rodLinkage)

            if not useSAL:
                linkagesGuess['deployed'] = l
            else:
                linkagesGuess['deployed'] = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(target_srf[0], target_srf[1], False, l)
                if not tarJP is None: linkagesGuess['deployed'].setTargetJointsPosition(tarJP.reshape(-1,))
                linkagesGuess['deployed'].set_holdClosestPointsFixed(False)
        else:
            print("Could not find both flat and deployed linkages in Speckle (flat: {}, deployed: {})".format(data.ContainsFlatLinkage, data.ContainsDeployLinkage))
    
    featJointIndices = None
    if (data['ContainsDeployLinkage']) and (targetMeshUser is None):
        target_srf, tarJP, featJointIndices = pull_target_surface(data['DeployLinkage'])
    if (featJointIndices is not None) and (featJointIndices):
        print("Feature joints are provided by Speckle! Don't forget to fetch them.")

    if tarJP is None and "JointPosOnMesh" in dir(data):
        jp = data["JointPosOnMesh"]
        tarJP = []
        for i in range(len(jp.get_dynamic_member_names())):
            tarJP.append([jp["{}".format(i)]["x"], jp["{}".format(i)]["y"], jp["{}".format(i)]["z"]])
        tarJP = np.array(tarJP).reshape(-1,)

    if target_srf is None and "TargetSurface" in dir(data):
        meshData = data["TargetSurface"]
        vertices = np.array(meshData.vertices).reshape(-1,3)
        faces = np.array(meshData.faces).reshape(-1,4)[:,1:]
        target_srf = [vertices, faces]

    attractionMesh = {
        "V"       : target_srf[0],
        "F"       : target_srf[1],
        "targetJP": tarJP.reshape(-1,)
    }
    targetMesh = {
        "V"       : target_srf[0],
        "F"       : target_srf[1],
        "targetJP": tarJP.reshape(-1,)
    }

    if forceReconstruct: linkagesGuess=None
    if rodMaterial: material = rodMaterial
    
    if radialSymmetry:
        nJalong = len(curve_network_data['Curves'][0]['NumControlPoints']) + 1
        nJinner = int(numJoints / nJalong)
        visited0 = False
        visited1 = False

        for crvData in curve_network_data['Curves']:
            if crvData['CurveFamily'] == 0 and not visited0:
                nCPperSplineA = crvData['NumControlPoints']
                visited0 = True
                
            if crvData['CurveFamily'] == 1 and not visited1:
                nCPperSplineB = crvData['NumControlPoints']
                visited1 = True
                
        symmetry = RadialDuplicator(nJinner, nJalong, nCPperSplineA, nCPperSplineB, revertSymmetry=revertSymmetry)
        redCurvesDoF = torch.zeros(size=(2*nJalong + sum(nCPperSplineA) + sum(nCPperSplineB),))

        redCurvesDoF[:2*nJalong] = curvesDoF[:2*nJalong]
        redCurvesDoF[2*nJalong:2*nJalong + sum(nCPperSplineA)] = curvesDoF[2*numJoints:2*numJoints + sum(nCPperSplineA)]
        redCurvesDoF[2*nJalong + sum(nCPperSplineA):] = curvesDoF[2*numJoints + nJinner*sum(nCPperSplineA) : 2*numJoints + nJinner*sum(nCPperSplineA) + sum(nCPperSplineB)]
        
        return CShell(redCurvesDoF, None, None, None, None, alphaTar, 5, subdivision, symmetry=symmetry,
                          rodMaterial=rodMaterial, optimizeAlpha=optimizeAlpha, linkagesGuess=linkagesGuess, 
                          attractionMesh=attractionMesh, targetMesh=targetMesh, 
                          useSAL=not (target_srf is None), flatOnly=flatOnly)
    
    cshell = CShell(curvesDoF, numJoints, curves, curvesFamily, nCPperRodEdge, alphaTar, 5, subdivision, 
                        rodMaterial=material, optimizeAlpha=optimizeAlpha, linkagesGuess=linkagesGuess, 
                        attractionMesh=attractionMesh, targetMesh=targetMesh, 
                        useSAL=not (target_srf is None), flatOnly=flatOnly)

    return cshell

def pull_morphing_experiment_data(interop : Interop, branch : str, nameMesh1 : str, nameMesh2 : str,
                                  optimizeAlpha : bool = True, multAlpha : float = 1.0,
                                  linkagesGuess = None, numOpSteps = None, commitItem : int = 0):
    '''
    Pull the curve network (curve data for CShell initialization) from the webapp (Grasshopper => Notebook)

    Args:
        interop       : interoperability object
        branch        : name of the branch from where to retreive the curve network
        nameMesh1     : name of the first mesh as uploaded on Speckle
        nameMesh2     : name of the second mesh as uploaded on Speckle
        optimizeAlpha : whether or not the cshell we keep the target average opening angle fixed
        multAlpha     : to over-open the linkage
        linkagesGuess : dictionnary containing "flat" and "deployed" linkages

    Ouput:
        cshell : cshell object
    '''

    cshell = pull_full_linkage(interop, branch, optimizeAlpha=optimizeAlpha, multAlpha=multAlpha,
                               linkagesGuess=linkagesGuess, numOpSteps=numOpSteps, commitItem=commitItem)
    
    data = interop.receive_data(branch, commitItem=commitItem)

    meshFlat = data[nameMesh1]
    vFlat = np.array(meshFlat.vertices).reshape(-1, 3)
    fFlat = np.array(meshFlat.faces).reshape(-1, 4)[:,1:]

    meshSphere = data[nameMesh2]
    vSphere = np.array(meshSphere.vertices).reshape(-1, 3)
    fSphere = np.array(meshSphere.faces).reshape(-1, 4)[:,1:]

    return cshell, vFlat, fFlat, vSphere, fSphere


def pull_curve_network(interop : Interop, branch : str, optimizeAlpha : bool = True, multAlpha : float = 1.0,
                       deployedDoF = None, linkagesGuess = None, numOpSteps = None, commitItem : int = 0, flatOnly = False):
    '''
    Pull the curve network (curve data for CShell initialization) from the webapp (Grasshopper => Notebook)

    Args:
        interop       : interoperability object
        branch        : name of the branch from where to retreive the curve network
        optimizeAlpha : whether or not the cshell we keep the target average opening angle fixed
        multAlpha     : to over-open the linkage
        deployedDoF   : initialize the deployed DoF with some guess
        linkagesGuess : dictionnary containing "flat" and "deployed" linkages

    Ouput:
        cshell        : cshell object
    '''
    # Receive data from the curve_netwrok branch as a tree structure. This handles the previous data
    data = interop.receive_data(branch,commitItem=commitItem)

    # CurvesDoFs
    temp = data['CurvesDoF']
    curvesDoF = torch.tensor(temp)

    # Curves
    temp = data['Curves']
    curves = []
    nCPperRodEdge = []
    curvesFamily = []
    for crv_data in temp:
        curves.append(crv_data['Indexes'])
        nCPperRodEdge.append(crv_data['NumControlPoints'])
        curvesFamily.append(crv_data['CurveFamily'])

    subdivision = int(data['Subdivision'])
    alphaTar = multAlpha * np.deg2rad(float(data['Angle']))
    numJoints = int(data['NumJoints'])
    material = elastic_rods.RodMaterial('rectangle', float(data['E']), float(data['PoisonsRatio']), [float(data['Width']), float(data['Height'])], keepCrossSectionMesh=True)

    linkagesGuess = {}

    cshell = CShell(curvesDoF, numJoints, curves, curvesFamily, nCPperRodEdge, alphaTar, 5, subdivision, 
                        rodMaterial=material, optimizeAlpha=optimizeAlpha, linkagesGuess=linkagesGuess, flatOnly=flatOnly)
    return cshell

def pull_xshell(interop : Interop, branchXShell : str, branchSurface : str, optimizeAlpha : bool = True, multAlpha : float = 1.0,
                useSAL : bool = False, linkagesGuess = None, commitItem : int = 0, forceReconstruct = False, attractionMesh = None):
    '''
    Pull the xshell from the webapp (Grasshopper => Notebook)

    Args:
        interop          : interoperability object
        branch           : name of the branch from where to retreive the curve network
        optimizeAlpha    : whether or not the cshell we keep the target average opening angle fixed
        multAlpha        : to over-open the linkage
        useSAL           : whether we want to deploy the shape using a given surface
        forceReconstruct : whether we want to re-deploy from scratch

    Ouput:
        flatLinkage      : a flat linkage object
        flatView         : the visualizer for the flat linkage
        deployedLinkage  : a deployed linkage object
        deployedView     : the visualizer for the deployed linkage
        linkageOptimizer : the linkage optimization we created from the flat and deployed linkages
    '''
    # Receive data from the curve_netwrok branch as a tree structure. This handles the previous data
    data = interop.receive_data(branchXShell, commitItem=commitItem)
    networkData = data["CurveNetwork"]

    # First get the target surface
    if forceReconstruct:
        pass
    else:
        if linkagesGuess is None:
            rodLinkage = pull_linkage(interop, branchXShell, True, commitItem=commitItem) 
            flatLinkage = average_angle_linkages.AverageAngleLinkage(rodLinkage)
            rodLinkage, target_srf, tarJP = pull_linkage(interop, branchXShell, False, commitItem=commitItem)
            useSAL = isinstance(rodLinkage, elastic_rods.SurfaceAttractedLinkage) or (not target_srf is None)
            l = average_angle_linkages.AverageAngleLinkage(rodLinkage)

            attractionMesh = {
                "V"       : target_srf[0],
                "F"       : target_srf[1],
                "targetJP": tarJP.reshape(-1,)
            }
            targetMesh = {
                "V"       : target_srf[0],
                "F"       : target_srf[1],
                "targetJP": tarJP.reshape(-1,)
            }
        
            if not useSAL:
                deployedLinkage = l
            else:
                deployedLinkage = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(attractionMesh["V"], attractionMesh["F"], False, l)
                deployedLinkage.set_holdClosestPointsFixed(False)
                if not tarJP is None: deployedLinkage.setTargetJointsPosition(attractionMesh["targetJP"].reshape(-1,))
                flatLinkage = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(attractionMesh["V"], attractionMesh["F"], False, flatLinkage)
                flatLinkage.attraction_weight = 0.0
            
        else:
            flatLinkage     = linkagesGuess["flat"]
            deployedLinkage = linkagesGuess["deployed"]
            deployedLinkage.set_holdClosestPointsFixed(False)
            useSAL = isinstance(deployedLinkage, average_angle_linkages.AverageAngleSurfaceAttractedLinkage)

        flatLinkage.set_design_parameter_config(True, False, True) # Keep length, remove rest curvature, and update the design parameters
        deployedLinkage.set_design_parameter_config(True, False, True) # Keep length, remove rest curvature, and update the design parameters

    # This will update the target surface/joints positions
    if attractionMesh is None:
        _, target_srf, tarJP = pull_linkage(interop, branchSurface, False, commitItem=commitItem)
        attractionMesh = {
            "V"       : target_srf[0],
            "F"       : target_srf[1],
            "targetJP": tarJP.reshape(-1,)
        }
        targetMesh = {
            "V"       : target_srf[0],
            "F"       : target_srf[1],
            "targetJP": tarJP.reshape(-1,)
        }
    else:
        targetMesh = attractionMesh

    if forceReconstruct:
        graphData      = networkData["Graph"]
        jointsPosition = np.array(graphData["Vertices"])
        rodEdges       = np.array(graphData["Edges"])
        rodMaterial    = elastic_rods.RodMaterial('rectangle', float(networkData['E']), float(networkData['PoisonsRatio']), 
                                                    [float(networkData['Width']), float(networkData['Height'])], keepCrossSectionMesh=True)

        # For the discretization
        subdivision = 10

        flatRodLinkage = elastic_rods.RodLinkage(jointsPosition, rodEdges,
                                            rod_interleaving_type=elastic_rods.InterleavingType.xshell, subdivision=subdivision)
        flatLinkage = average_angle_linkages.AverageAngleLinkage(flatRodLinkage)
        flatLinkage.setMaterial(rodMaterial)
        flatLinkage.set_design_parameter_config(True, False, True) # Keep length, remove rest curvature, and update the design parameters

    # Compute the flat equilibrium
    driver    = flatLinkage.centralJoint()
    jdo       = flatLinkage.dofOffsetForJoint(driver)
    fixedVars = list(range(jdo, jdo + 6)) # fix rigid motion for a single joint
    flatLinkage.setRestKappaVars(np.zeros_like(flatLinkage.getRestKappaVars()))
    with so(): average_angle_linkages.compute_equilibrium(flatLinkage, fixedVars=fixedVars)

    if forceReconstruct:
        alphaTar = multAlpha * np.deg2rad(float(networkData['Angle']))
    else:
        alphaTar = deployedLinkage.averageJointAngle

    # Create the deployed linkage in case it does not exist already
    if forceReconstruct:
        if useSAL:
            deployedLinkage = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(attractionMesh["V"], attractionMesh["F"], False, flatLinkage)
            flatLinkage = average_angle_linkages.AverageAngleSurfaceAttractedLinkage(attractionMesh["V"], attractionMesh["F"], False, flatLinkage)
            flatLinkage.attraction_weight = 0.0
        else:
            deployedLinkage = average_angle_linkages.AverageAngleLinkage(flatLinkage)
        deployedLinkage.set_design_parameter_config(True, False, True) # Keep length, remove rest curvature, and update the design parameters

    flatView       = LinkageViewer(flatLinkage, width=768, height=480)
    attractionSurf = mesh.Mesh(*(attractionMesh["V"], attractionMesh["F"]))
    deployedView   = LinkageViewerWithSurface(deployedLinkage, attractionSurf, wireframeSurf=False, transparent=True, width=768, height=480)

    # Compute the deployed equilibrium
    deployedLinkage.attraction_weight = 0.0001
    deployedLinkage.scaleJointWeights(jointPosWeight=0.1)
    deployedLinkage.set_holdClosestPointsFixed(False)
    deployedLinkage.setTargetSurface(attractionMesh["V"], attractionMesh["F"])
    deployedLinkage.setTargetJointsPosition(attractionMesh["targetJP"].reshape(-1,))
    deployedLinkage.setRestKappaVars(np.zeros_like(deployedLinkage.getRestKappaVars()))

    if forceReconstruct:
        def equilibriumSolver(tgtAngle, l, opts, fv):
            opts.gradTol = 1.0e-5
            return average_angle_linkages.compute_equilibrium(l, tgtAngle, options=opts, fixedVars=fv)

        numOpeningSteps = 40
        maxNewtonIterIntermediate = 50

        with so(): open_average_angle_linkage(deployedLinkage, driver, alphaTar - deployedLinkage.averageJointAngle, numOpeningSteps, 
                                deployedView, equilibriumSolver=equilibriumSolver, 
                                maxNewtonIterationsIntermediate=maxNewtonIterIntermediate)

    newtonOptimizerOptions = py_newton_optimizer.NewtonOptimizerOptions()
    newtonOptimizerOptions.gradTol = 1.0e-7
    newtonOptimizerOptions.verbose = 1
    newtonOptimizerOptions.beta = 1.0e-8
    newtonOptimizerOptions.niter = 500
    newtonOptimizerOptions.verboseNonPosDef = False

    idxAverageAngleDep = deployedLinkage.dofOffsetForJoint(deployedLinkage.numJoints() - 1) + 6
    fixedDepVars = []
    if not useSAL:
        fixedDepVars = list(range(jdo, jdo + 6))
    fixedDepVars.append(idxAverageAngleDep)
    with so(): average_angle_linkages.compute_equilibrium(deployedLinkage, elastic_rods.TARGET_ANGLE_NONE, options=newtonOptimizerOptions, fixedVars=fixedDepVars)
    deployedView.update()

    # Generate the linkage optimizer
    newtonOptimizerOptions.niter = 50

    if useSAL:
        linkageOptimizer = cshell_optimization.AverageAngleCShellOptimizationSAL(flatLinkage, deployedLinkage, newtonOptimizerOptions, 0., 
                                                                                 optimizeTargetAngle=optimizeAlpha, fixDeployedVars=not useSAL)
    else:
        linkageOptimizer = cshell_optimization.AverageAngleCShellOptimization(flatLinkage, deployedLinkage, newtonOptimizerOptions, 0., 
                                                                              optimizeTargetAngle=optimizeAlpha, fixDeployedVars=not useSAL)

    
    linkageOptimizer.setHoldClosestPointsFixed(False)
    linkageOptimizer.setTargetSurface(targetMesh["V"], targetMesh["F"])
    linkageOptimizer.scaleJointWeights(jointPosWeight=0.1)
    linkageOptimizer.setTargetJointsPosition(targetMesh["targetJP"].reshape(-1,))

    return flatLinkage, flatView, deployedLinkage, deployedView, linkageOptimizer

def push_cshell(interop : Interop, cshell : CShell, branch : str, deployed_field : ScalarField = None, centerline_positions : float = [], delta_centerline_positions : float = [],
                loop_subdivisions : int = 1, num_extension_layers :int = 1 , sendFlat : bool = True, 
                targetSurface = None, flat_field : ScalarField = None, 
                description : str = "", units :str = 'mm', crossSectionType : str = "RECTANGLE"):
    '''
    Push the deployed cshell (linkage + infer-surface) to the webapp (Notebook => Grasshopper)

    Args:
        interop                    : interoperability object
        cshell                     : cshell object
        branch                     : name of the branch where the cshell object is pushed
        deployed_field             : the scalar field to render on the deployed linkage
        joints_positions           : some joints positions to highlight (length 3*nJ)
        centerline_positions       : some centerline positions to keep track of (length 3*nCP)
        delta_centerline_positions : same as above except we provide displacements of the centerline positions
        loop_subdivisions          : useful for infering the target surface (refines the target surface)
        num_extension_layers       : useful for infering the target surface (expands the target surface)
        sendFlat                   : whether we should send the flat mesh too or not
        targetSurface              : dictionnary containing "V" and "F" as keys and np arrays as values
        flat_field                 : the scalar field to render on the flat linkage
        description                : description of the model
        units                      : units of the model
        crossSectionType           : cross section type (RECTANGLE, L, +, ELLIPSE, I)
    '''
    
    # deployed linkage
    deployed_mesh = get_interop_linkage_mesh(cshell.deployedLinkage, deployed_field, units)
    # centerline positions
    deployed_centerline_pos = GetCenterlinePositionsForAssembly(cshell, deployed=True) 

    base = Base()
    base.units = units

    # Send flat linkage
    flat_linkage = get_interop_linkage_mesh(cshell.flatLinkage, flat_field, units)
    # centerline positions
    flat_centerline_pos = GetCenterlinePositionsForAssembly(cshell, deployed=False) 

    if sendFlat:
        base.flat_linkage    = flat_linkage
        base.flat_centerline = flat_centerline_pos

    target_surface = None
    if not targetSurface is None:
        # Vertices
        vertices = []
        for pos in targetSurface["V"]:
            vertices.extend([float(pos[0]),float(pos[1]),float(pos[2])])
        # Faces
        faces = []
        for trias in targetSurface["F"]:
            faces.extend([0,int(trias[0]),int(trias[1]),int(trias[2])])
        
        target_surface = {
            "V": vertices,
            "F": faces
        }

    tsf = cshell.linkageOptimizer.target_surface_fitter
    devField = ScalarFieldDeviations(cshell.deployedLinkage, tsf, useSurfDim=True, usePercent=True, perEdge=False)
    E  = cshell.deployedLinkage.homogenousMaterial().youngModulus
    nu = cshell.deployedLinkage.homogenousMaterial().youngModulus / (2 * cshell.deployedLinkage.homogenousMaterial().shearModulus) - 1
    height = cshell.deployedLinkage.homogenousMaterial().crossSectionHeight
    width  = cshell.deployedLinkage.homogenousMaterial().area / height
    
    if sendFlat:
        if not cshell.flatLinkage.hasCrossSection():
            cs = elastic_rods.CrossSection.construct("RECTANGLE", E, nu, [height, width])
            rm = elastic_rods.RodMaterial(cs)
            cshell.flatLinkage.setMaterial(rm)
        cshell.flatLinkage.meshCrossSection(0.001)
        maxVMFieldFlat = cshell.flatLinkage.maxVonMisesStresses()
    if not cshell.deployedLinkage.hasCrossSection():
        cs = elastic_rods.CrossSection.construct("RECTANGLE", E, nu, [height, width])
        rm = elastic_rods.RodMaterial(cs)
        cshell.deployedLinkage.setMaterial(rm)
    cshell.deployedLinkage.meshCrossSection(0.001)
    maxVMField = cshell.deployedLinkage.maxVonMisesStresses()
    
    materialSpecs = {
        "E"     : E,
        "nu"    : nu,
        "height": height,
        "width" : width,
        "type"  : crossSectionType,
        "parameters": cshell.deployedLinkage.homogenousMaterial().crossSection().params(),
    }

    deformedFlatCP = []
    matFramesD1D2Flat = []
    for seg in cshell.flatLinkage.segments():
        deformedFlatCP.append([coor for defPts in seg.rod.deformedPoints() for coor in defPts])
        matFramesD1D2Flat.append([coor for matFrames in seg.rod.deformedMaterialFramesD1D2() for coor in matFrames])
        
    deformedDeployedCP = []
    matFramesD1D2Deployed = []
    for seg in cshell.deployedLinkage.segments():
        deformedDeployedCP.append([coor for defPts in seg.rod.deformedPoints() for coor in defPts])
        matFramesD1D2Deployed.append([coor for matFrames in seg.rod.deformedMaterialFramesD1D2() for coor in matFrames])

    base.materialSpecs              = materialSpecs
    base.targetDeviationField       = [list(elt) for elt in devField]
    base.maxVonMisesField           = [list(elt) for elt in maxVMField]
    if sendFlat: 
        base.maxVonMisesFieldFlat   = [list(elt) for elt in maxVMFieldFlat]
    base.sqrtBendingEnergiesField   = [list(elt) for elt in cshell.deployedLinkage.sqrtBendingEnergies()]
    base.stretchingEnergiesField    = [list(elt) for elt in cshell.deployedLinkage.stretchingEnergies()]
    base.twistingEnergiesField      = [list(elt) for elt in cshell.deployedLinkage.twistingEnergies()]
    base.rodEdgesFamily             = cshell.rodEdgesFamily
    base.curves                     = cshell.curves
    base.curvesFamily               = cshell.curvesFamily
    base.jointValence               = cshell.valence.tolist()
    base.deployed_centerline        = deployed_centerline_pos
    base.deformedFlatCenterline     = deformedFlatCP
    base.deformedDeployedCenterline = deformedDeployedCP
    base.matFramesD1D2Flat          = matFramesD1D2Flat
    base.matFramesD1D2Deployed      = matFramesD1D2Deployed
    base.deployed_linkage           = deployed_mesh
    base.target_surface             = target_surface
    base.flatJointsPositions        = list(cshell.flatLinkage.jointPositions())
    base.deployedJointsPositions    = list(cshell.deployedLinkage.jointPositions())
    base.centerline_positions       = centerline_positions
    base.delta_centerline_positions = delta_centerline_positions
    base.description                = description
    base.subdivision                = cshell.subdivision
    base.freeAngles                 = cshell.freeAngles

    interop.send_data(base, branch, description)

def push_xshell(interop : Interop, flatLinkage : elastic_rods.RodLinkage, deployedLinkage : elastic_rods.RodLinkage, linkageOptimizer : Any,
                branch : str, deployed_field : ScalarField = None, 
                joints_positions : float = [], centerline_positions : float = [], delta_centerline_positions : float = [],
                loop_subdivisions : int = 1, num_extension_layers :int = 1 , sendFlat : bool = True, sendTarget : bool = True,
                targetSurface = None, flat_field : ScalarField = None, 
                description : str = "", units :str = 'mm', crossSectionType : str = "RECTANGLE",
                curves = None):
    '''
    Push the deployed cshell (linkage + infer-surface) to the webapp (Notebook => Grasshopper)

    Args:
        interop                    : interoperability object
        flatLinkage                : the flat linkage
        deployedLinkage            : the deployed linkage
        branch                     : name of the branch where the cshell object is pushed
        deployed_field             : the scalar field to render on the deployed linkage
        joints_positions           : some joints positions to highlight (length 3*nJ)
        centerline_positions       : some centerline positions to keep track of (length 3*nCP)
        delta_centerline_positions : same as above except we provide displacements of the centerline positions
        loop_subdivisions          : useful for infering the target surface (refines the target surface)
        num_extension_layers       : useful for infering the target surface (expands the target surface)
        targetSurface              : dictionnary containing "V" and "F" as keys and np arrays as values
        sendFlat                   : whether we should send the flat mesh too or not
        flat_field                 : the scalar field to render on the flat linkage
        description                : description of the model
        units                      : units of the model
        crossSectionType           : cross section type (RECTANGLE, L, +, ELLIPSE, I)
        curves                     : list of list giving the joint indices per curve
    '''
    
    # deployed linkage
    deployed_mesh = get_interop_linkage_mesh(deployedLinkage, deployed_field, units)
    # centerline positions
    deployed_centerline_pos = GetCenterlinePositionsForAssemblyFromLinkage(deployedLinkage) 

    base = Base()
    base.units = units

    flat_linkage = get_interop_linkage_mesh(flatLinkage, flat_field, units)
    # centerline positions
    flat_centerline_pos = GetCenterlinePositionsForAssemblyFromLinkage(flatLinkage) 

    if sendFlat:
        base.flat_linkage    = flat_linkage
        base.flat_centerline = flat_centerline_pos

    target_surface = None
    if not targetSurface is None:
        # Vertices
        vertices = []
        for pos in targetSurface["V"]:
            vertices.extend([float(pos[0]),float(pos[1]),float(pos[2])])
        # Faces
        faces = []
        for trias in targetSurface["F"]:
            faces.extend([0,int(trias[0]),int(trias[1]),int(trias[2])])

        target_surface = {
            "V": vertices,
            "F": faces
        }

    if not linkageOptimizer is None:
        tsf = linkageOptimizer.target_surface_fitter
        devField = ScalarFieldDeviations(deployedLinkage, tsf, useSurfDim=True, usePercent=True, perEdge=False)
        base.targetDeviationField = [list(elt) for elt in devField]
    else:
        base.targetDeviationField = None
    E  = deployedLinkage.homogenousMaterial().youngModulus
    nu = deployedLinkage.homogenousMaterial().youngModulus / (2 * deployedLinkage.homogenousMaterial().shearModulus) - 1
    height = deployedLinkage.homogenousMaterial().crossSectionHeight
    width  = deployedLinkage.homogenousMaterial().area / height
    
    if sendFlat:
        if not flatLinkage.hasCrossSection():
            cs = elastic_rods.CrossSection.construct("RECTANGLE", E, nu, [height, width])
            rm = elastic_rods.RodMaterial(cs)
            flatLinkage.setMaterial(rm)
        flatLinkage.meshCrossSection(0.001)
        maxVMFieldFlat = flatLinkage.maxVonMisesStresses()
    if not deployedLinkage.hasCrossSection():
        cs = elastic_rods.CrossSection.construct("RECTANGLE", E, nu, [height, width])
        rm = elastic_rods.RodMaterial(cs)
        deployedLinkage.setMaterial(rm)
    deployedLinkage.meshCrossSection(0.001)
    maxVMField = deployedLinkage.maxVonMisesStresses()
    
    materialSpecs = {
        "E"     : E,
        "nu"    : nu,
        "height": height,
        "width" : width,
        "type"  : crossSectionType,
        "parameters": deployedLinkage.homogenousMaterial().crossSection().params(),
    }

    deformedFlatCP = []
    matFramesD1D2Flat = []
    for seg in flatLinkage.segments():
        deformedFlatCP.append([coor for defPts in seg.rod.deformedPoints() for coor in defPts])
        matFramesD1D2Flat.append([coor for matFrames in seg.rod.deformedMaterialFramesD1D2() for coor in matFrames])

    deformedDeployedCP = []
    matFramesD1D2Deployed = []
    for seg in deployedLinkage.segments():
        deformedDeployedCP.append([coor for defPts in seg.rod.deformedPoints() for coor in defPts])
        matFramesD1D2Deployed.append([coor for matFrames in seg.rod.deformedMaterialFramesD1D2() for coor in matFrames])
        
    rodEdges = []
    for s in flatLinkage.segments():
        rodEdges.append([s.startJoint, s.endJoint])
    crvToRodEdgeIdx = []
    crvSegmentRestLengths = []
    if curves is not None:
        crvToRodEdgeIdx = []
        for crv in curves:
            crvToRodEdgeIdxTmp = []
            for i in range(len(crv)-1):
                for j, re in enumerate(rodEdges):
                    if (re[0] == crv[i] and re[1] == crv[i+1]) or (re[0] == crv[i+1] and re[1] == crv[i]):
                        crvToRodEdgeIdxTmp.append(j)
                        
            crvToRodEdgeIdx.append(crvToRodEdgeIdxTmp)
        psrl = flatLinkage.getPerSegmentRestLength().tolist()
        crvSegmentRestLengths = [[psrl[idx] for idx in segIdx] for segIdx in crvToRodEdgeIdx]

    base.materialSpecs              = materialSpecs
    base.maxVonMisesField           = [list(elt) for elt in maxVMField]
    if sendFlat: 
        base.maxVonMisesFieldFlat   = [list(elt) for elt in maxVMFieldFlat]
    base.sqrtBendingEnergiesField   = [list(elt) for elt in deployedLinkage.sqrtBendingEnergies()]
    base.stretchingEnergiesField    = [list(elt) for elt in deployedLinkage.stretchingEnergies()]
    base.twistingEnergiesField      = [list(elt) for elt in deployedLinkage.twistingEnergies()]
    base.deployed_centerline        = deployed_centerline_pos
    base.deployed_linkage           = deployed_mesh
    if sendFlat: 
        base.deformedFlatCenterline = deformedFlatCP
    base.deformedDeployedCenterline = deformedDeployedCP
    if sendFlat: 
        base.matFramesD1D2Flat      = matFramesD1D2Flat
    base.matFramesD1D2Deployed      = matFramesD1D2Deployed
    if sendTarget: 
        base.target_surface         = target_surface
    base.joints_positions           = joints_positions
    base.per_segment_rest_lengths   = flatLinkage.getPerSegmentRestLength().tolist()
    base.centerline_positions       = centerline_positions
    base.delta_centerline_positions = delta_centerline_positions
    base.description                = description
    base.subdivision                = flatLinkage.segment(0).rod.numEdges()
    base.curves                     = curves
    base.rod_edges                  = rodEdges
    base.curves_to_rod_edge_idx     = crvToRodEdgeIdx
    base.curves_to_rod_rest_length  = crvSegmentRestLengths

    interop.send_data(base, branch, description)

def push_inferred_surface(interop : Interop, cshell : CShell, branch : str, loop_subdivisions : int = 1, num_extension_layers :int = 1, 
                          description : str = "", units :str = 'mm'):
    '''
    Push the deployed cshell (linkage + infer-surface) to the webapp (Notebook => Grasshopper)

    Args:
        interop                    : interoperability object
        cshell                     : cshell object
        branch                     : name of the branch where the cshell object is pushed
        loop_subdivisions          : useful for infering the target surface (refines the target surface)
        num_extension_layers       : useful for infering the target surface (expands the target surface)
        description                : description of the model
        units                      : units of the model
    '''

    base = Base()
    base.units = units
    infer_surface_mesh = get_interop_infer_surface_mesh(cshell.linkageOptimizer, loop_subdivisions, num_extension_layers, units, recomputeSurface=False)
    base.target_surface = infer_surface_mesh
    base.joints_positions = list(cshell.deployedLinkage.jointPositions())
    base.centerline_positions = get_interop_centerline_positions(cshell, deployed=True)

    interop.send_data(base, branch, description)

def pull_modified_surface(interop : Interop, cshell : CShell, branch : str):
    '''
    Pull the modifed surface into the cshell-optimizer from the webapp (Grasshopper => Notebook)

    Args:
        interop : interoperability object
        cshell  : cshell object
        branch  : name of the branch from where to retreive the curve network
    '''

    data = interop.receive_data(branch)
    meshData = data.modified_surface
    vertices = np.array(meshData.vertices).reshape(-1,3)
    faces = np.array(meshData.faces).reshape(-1,4)[:,1:]
    cshell.linkageOptimizer.setTargetSurface(vertices,faces)
    cshell.UpdateDeployedViewer()

    centerline_pos = data.modified_centerline_positions.vertices
    joints_pos = data.modified_joints_positions.vertices

    return centerline_pos, joints_pos

def pull_target_surface_and_update_cshell(interop : Interop, cshell : CShell, branch : str):
    '''
    Pull the modifed surface into the cshell-optimizer from the webapp (Grasshopper => Notebook)

    Args:
        interop : interoperability object
        cshell  : cshell object
        branch  : name of the branch from where to retreive the curve network
    '''

    data = interop.receive_data(branch)
    
    meshData = data.target_surface
    vertices = np.array(meshData.vertices).reshape(-1,3)
    faces = np.array(meshData.faces).reshape(-1,4)[:,1:]
    cshell.linkageOptimizer.setTargetSurface(vertices,faces)

    try:
        targetJointsPositions = np.array(data.joints_positions)
    except Exception as e:
        print("Couldn't find joints position, falling back to current joints position.")
        targetJointsPositions = cshell.deployedLinkage.jointPositions()
    cshell.linkageOptimizer.setTargetJointPosition(targetJointsPositions)
    cshell.UpdateDeployedViewer()

def get_interop_linkage_mesh(linkage : elastic_rods.RodLinkage, field : ScalarField = None, units :str = "mm"):
    '''
    Build mesh for interoperability

    Args:
        linkage : Rodlinkage object
        field   : scalar field for visualization
        units   : units of the model
    '''
    m = linkage.visualizationGeometry()

    # Vertices
    vertices = []
    for pos in m[0]:
        vertices.extend([float(pos[0]),float(pos[1]),float(pos[2])])
    # Faces
    faces = []
    for trias in m[1]:
        faces.extend([0,int(trias[0]),int(trias[1]),int(trias[2])])
        
    meshData = {
        "V": vertices,
        "F": faces
    }

    meshData = Mesh()
    meshData.vertices = vertices
    meshData.faces = faces
    meshData.units = units

    return meshData

def get_interop_infer_surface_mesh(linkageOptimizer, loop_subdivisions : int = 1, num_extension_layers : int = 1, 
                                   units :str = "mm", recomputeSurface : bool = True):
    # Infer target surface
    if recomputeSurface:
        linkageOptimizer.constructTargetSurface(loop_subdivisions,num_extension_layers,[1., 1., 1.])
    m = linkageOptimizer.target_surface_fitter

    # Vertices
    vertices = []
    for pos in m.V:
        vertices.extend([float(pos[0]),float(pos[1]),float(pos[2])])
    # Faces
    faces = []
    for trias in m.F:
        faces.extend([0,int(trias[0]),int(trias[1]),int(trias[2])])

    meshData = {
        "V": vertices,
        "F": faces
    }

    return meshData

def get_interop_infer_surface_edges(cshell : CShell, loop_subdivisions : int = 1, num_extension_layers : int = 1, recomputeSurface : bool = True):
    # Infer target surface
    if recomputeSurface:
        cshell.linkageOptimizer.constructTargetSurface(loop_subdivisions,num_extension_layers,[1.0, 1.0, 1.0])
    m = cshell.linkageOptimizer.target_surface_fitter

    # Vertices
    pos = m.V

    faces = []
    for trias in m.F:
        pts = []
        for idx in trias:
            p = Point(x=pos[idx][0],y=pos[idx][1],z=pos[idx][2], units='mm')
            pts.append(p)
        pts.append(pts[0])

        pl = Polyline()
        pl.from_points(pts)

        faces.append(pl)
        
    return faces

def get_interop_centerline_positions(cshell : CShell, deployed :bool = True):
    centerline_pos = []
    linkage = cshell.deployedLinkage
    if not deployed:
        linkage = cshell.flatLinkage
    
    dofs = IndicesDiscretePositionsNoDuplicate(linkage, cshell.rodEdges, cshell.subdivision)
    centerline_pos = list(linkage.getDoFs()[dofs])
    
    return centerline_pos

def get_interop_centerline_positions_from_linkage(linkage : elastic_rods.RodLinkage):
    centerline_pos = []
    subdivision = linkage.segment(0).rod.numEdges()
    rodEdgesList = []
    for seg in linkage.segments():
        rodEdgesList.append([seg.startJoint, seg.endJoint])
        
    rodEdges = np.array(rodEdgesList)

    dofs = IndicesDiscretePositionsNoDuplicate(linkage, rodEdges, subdivision)
    centerline_pos = list(linkage.getDoFs()[dofs])
    
    return centerline_pos

def IndicesDiscretePositionsForAssembly(linkage, edges, subdivision):
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
        lstEdge += list(range(offsetJoint, offsetJoint+3)) # For the two positions at the first end
        for j in range(subdivision-3):
            lstEdge += list(range(offsetSegment+j*3, offsetSegment+(j+1)*3))
        offsetJoint = linkage.dofOffsetForJoint(edge[1])
        lstEdge += list(range(offsetJoint, offsetJoint+3))
        
        idxPos.append(lstEdge)
        
    return idxPos

def GetCenterlinePositionsForAssembly(cshell : CShell, deployed :bool = True):
    centerlinePos = []
    linkage = cshell.deployedLinkage
    if not deployed:
        linkage = cshell.flatLinkage
    
    dofs = IndicesDiscretePositionsForAssembly(linkage, cshell.rodEdges, cshell.subdivision)
    fullLinkDoF = linkage.getDoFs()
    centerlinePos = [list(fullLinkDoF[dof]) for dof in dofs]
    
    return centerlinePos

def GetCenterlinePositionsForAssemblyFromLinkage(linkage : elastic_rods.RodLinkage, deployed :bool = True):
    centerlinePos = []
    subdivision = linkage.segment(0).rod.numEdges()
    rodEdgesList = []
    for seg in linkage.segments():
        rodEdgesList.append([seg.startJoint, seg.endJoint])
        
    rodEdges = np.array(rodEdgesList)
    
    dofs = IndicesDiscretePositionsForAssembly(linkage, rodEdges, subdivision)
    fullLinkDoF = linkage.getDoFs()
    centerlinePos = [list(fullLinkDoF[dof]) for dof in dofs]
    
    return centerlinePos
