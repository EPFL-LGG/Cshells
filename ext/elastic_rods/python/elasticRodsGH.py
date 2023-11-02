import sys
sys.path.append('../../python')

import numpy as np
from geomdl import NURBS
import elastic_rods
from elastic_rods import InterleavingType
import json

class linkageData:
    def __init__(self, filename : str):
        
        # Import converted data
        with open(filename) as json_file:
            data = json.load(json_file)
        
        # Interleaving
        # InterleavingType { xshell=0, weaving=1, noOffset=2, triaxialWeave=3 };
        itype = data['Interleaving']
        interleaving = InterleavingType.noOffset
        # if(itype==1):
        #     interleaving = InterleavingType.weaving
        # elif(itype==2):
        #     interleaving = InterleavingType.noOffset
        # elif(itype==3):
        #     interleaving = InterleavingType.triaxialWeave

        # Vertices data
        joints = data['Joints']
        num_joints = len(joints)
        self.joints = np.ndarray(shape=(num_joints, 3))
        self.normals = np.ndarray(shape=(num_joints, 3))
        for i in range(num_joints):
            j = joints[i]
            pos = j['Position']
            self.joints[i] = [pos[0],pos[1],pos[2]]
            norm = j['Normal']
            self.normals[i] = [norm[0],norm[1], norm[2]]
            
        # Edge data
        edges = data['Edges']
        num_edges = len(edges)
        self.edges = np.ndarray(shape=(num_edges, 2))
        self.rlengths = np.ndarray(shape=(num_edges,))
        curve_functions = []
        for i in range(num_edges):
            e = edges[i]
            idx = e['Indexes']
            self.edges[i] = [idx[0],idx[1]] 
            self.rlengths[i] = e['RestLength']
    
            crv = NURBS.Curve()
            crv.degree = e['Degree']
            crv.ctrlpts = e['ControlPoints']
            knots = e['Knots']
            knots[1:-1] = knots[:]
            crv.knotvector = knots
            curve_functions.append((lambda capture_crv: lambda alpha, correct_orientation: np.array(capture_crv.evaluate_single(clip_alpha(alpha))[:3]) if correct_orientation else np.array(capture_crv.evaluate_single(clip_alpha(1-alpha))[:3]))(crv))
        
        input_joint_normals = np.ones((num_joints, 3))
        input_joint_normals[:, :2] *= 0
        # Init linkage
        # TODO: Include all parameters from Grasshopper (Add binding for the new constructor)
        self.linkage = elastic_rods.RodLinkage(self.joints, self.edges, subdivision=20, rod_interleaving_type = interleaving, edge_callbacks = curve_functions, input_joint_normals = input_joint_normals)    

        # Material Data (after initialization)
        # CrossSectionType { rectangle=0, ellipse=1, I=2, L=3, cross=4 };
        # StiffAxis { tangent=0, normal=1 };
        materials = data['MaterialData']
        num_materials = len(materials)
        mat = []
        for i in range(num_materials):
            m = materials[i]

            section = 'rectangle'
            type = m['CrossSectionType']
            if(type==1):
                section = 'ellipse'
            elif(type==2):
                section = 'I'
            elif(type==3):
                section = 'L'
            elif(type==4):
                section = '+'

            axis = elastic_rods.StiffAxis.D1
            if(m['Orientation'] == 1):
                axis = elastic_rods.StiffAxis.D2

            mat.append(elastic_rods.RodMaterial(section, m['E'], m['PoisonsRatio'], [m['Width'],m['Height']], stiffAxis=axis))
        
        if(num_materials==1):
            self.linkage.setMaterial(mat[0])
        elif(num_materials>1):
            self.linkage.setJointMaterials(mat)
        
        # Support data
        anchors = data['Supports']
        num_anchors = len(anchors)
        anchors = []
        if num_anchors > 0:
            for i in range(num_anchors):
                a = anchors[i]
                idx = self.linkage.dofOffsetForJoint(a['Indexes'][0])
                idx_dof = []
                dof = a['LockedDOF']
                for offset in dof:
                    idx_dof.append(idx + offset)
                anchors.extend(idx_dof)
        self.supports = anchors
        
        # Force data
        forces = data['Forces']
        num_forces = len(forces)
        forces = []
        if num_forces > 0:
            forces = np.linspace(0,0,len(self.linkage.gradient()))
            for i in range(num_forces):
                f = forces[i]
                idx = self.linkage.dofOffsetForJoint(f['Indexes'][0])
                vec = f['Vector']
                for j in range(3):
                    forces[idx+j] = vec[j]
        self.forces = forces

        # Target deployment angle
        self.deployment_angle = data['TargetAngle']
    
    
def clip_alpha(a):
    if a > 1:
        return 1
    if a < 0:
        return 0
    return a    
