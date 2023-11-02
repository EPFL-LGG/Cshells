import pickle
linkage_rest = pickle.load(open('linkage_rest.pkl', 'rb'))
linkage_defo = pickle.load(open('linkage_defo.pkl', 'rb'))

import elastic_rods
elastic_rods.linkage_deformation_analysis(linkage_rest, linkage_defo, 'linkage_analysis.msh')
