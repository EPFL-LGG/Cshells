import elastic_rods
import json
import numpy as np
import torch

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

class GH_Converter:
    def __init__(self,filename : str):
        
        self.filename = filename

        # Import converted data
        with open(filename) as json_file:
            data = json.load(json_file)

        # CurvesDoFs
        temp = data['CurvesDoF']
        self.curvesDoF = torch.tensor( [ temp[i] for i in range(len(temp)) ])

        # Curves
        temp = data['Curves']
        self.curves = [ temp[i] for i in range(len(temp)) ]

        # nCPperRodEdge
        temp = data['NumControlPoints']
        self.nCPperRodEdge = [ temp[i] for i in range(len(temp)) ]

        # CurvesFamily
        temp = data['CurvesFamily']
        self.curvesFamily = [ temp[i] for i in range(len(temp)) ]

        self.subdivision = int(data['Subdivision'])
        self.mult = int(data['Mult'])
        self.alphaTar = np.deg2rad(float(data['Angle']))
        self.numJoints = int(data['NumJoints'])

        self.material = elastic_rods.RodMaterial('rectangle', float(data['E']), float(data['PoisonsRatio']), [float(data['Width']), float(data['Height'])])

        self.numOpeningSteps = int(data['NumSteps'])
        self.maxNewtonIterIntermediate = int(data['NumIterations'])
    
    def saveVizLinkageJSON(self, filename :str, linkage):
        # Get visualization mesh
        m = linkage.visualizationGeometry()
        
        data = {}
        data['Vertices'] = []
        for pos in m[0]:
            data['Vertices'].append([float(pos[0]),float(pos[1]),float(pos[2])])

        data['Faces'] = []
        for quads in m[1]:
            data['Faces'].append([int(quads[0]),int(quads[1]),int(quads[2])])

        # Serializing json 
        json_object = json.dumps(data, indent = 4)

        # Import converted data
        with open(filename+'.json', 'w') as json_file:
            json_file.write(json_object)
