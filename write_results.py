from model import Model, write_to_doc
import os
import json

'''Instead of below, move writing to doc and checking for doc key/overwrite into model.py
   Check if doc key False OR (overwrite True AND doc key True)
   If doc key False, run
   If overwrite True AND doc key True, delete existing data first, then run'''
for rundir in os.listdir(os.path.join(os.getcwd(),'runs')):
    docfn = os.path.join(os.getcwd(),'runs',rundir,'run_doc.json')
    infofn = os.path.join(os.getcwd(),'runs',rundir,'run_info.json')
    with open(docfn, 'r+') as doc_file, open(infofn, 'r') as info_file:
        doc = json.load(doc_file)
        info = json.load(info_file)
        inst = Model(**info)
        if doc.get('abundance_stats_written') != True:
            inst.write_abundance_stats()
            write_to_doc(docfn, 'abundance_stats_written', True)
        if inst.spatial:
            if doc.get('patch_centroids_written') != True:
                inst.write_patch_centroids()
                write_to_doc(docfn, 'patch_centroids_written', True)
            if doc.get('fire_prob_written') != True:
                inst.write_fire_prob()
                write_to_doc(docfn, 'fire_prob_written', True)
            if doc.get('patch_K_written') != True:
                inst.write_patch_K()
                write_to_doc(docfn, 'patch_K_written', True)
