from model import Model, write_to_doc
import os
import json

overwrite = False 
for rundir in os.listdir(os.path.join(os.getcwd(),'runs')):
    docfn = os.path.join(os.getcwd(),'runs',rundir,'run_doc.json')
    infofn = os.path.join(os.getcwd(),'runs',rundir,'run_info.json')
    with open(docfn, 'r+') as doc_file, open(infofn, 'r') as info_file:
        doc = json.load(doc_file)
        info = json.load(info_file)
        inst = Model(**info)
        #if inst.spatial and inst.fixed_habitat and inst.fixed_fire:
        if rundir == str(74):
            overwrite = True
        else:
            overwrite = False
        inst.write_abundance_stats(overwrite=overwrite)
        if inst.spatial:
            inst.write_patch_centroids(overwrite=overwrite)
            inst.write_fire_prob(overwrite=overwrite)
            inst.write_patch_K(overwrite=overwrite)
