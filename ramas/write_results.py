from model import Model, write_to_doc, open_run
import os
import json

overwrite = False 
for runnum in os.listdir(os.path.join(os.getcwd(),'runs')):
    run = open_run(runnum)
    #if runnum in ['76']:
    #    overwrite = True
    #else:
    #    overwrite = False
    run.write_abundance_stats(overwrite=overwrite)
    if run.spatial:
        run.write_patch_centroids(overwrite=overwrite)
        run.write_patch_K(overwrite=overwrite)
        if (run.fixed_fire and (run.fire_frame != None)) or (run.fixed_fire == False):
            run.write_fire_prob(overwrite=overwrite)
