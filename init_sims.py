import os
import json
from model import Model
from model import write_to_doc

def check_for_results(inst):
    res_check = False
    mpfn = os.path.join(inst.run_dir, 'final.mp')
    with open(mpfn, 'r') as mp:
        for line in mp:
            if 'Simulation results' in line:
                res_check = True
                break
    return res_check

# Whether or not to overwrite existing data
overwrite = False

# Read in shared ('project level') data
with open(os.path.join(os.getcwd(), 'project_info.json'), 'r') as info_file:
    proj_info = json.load(info_file)

batfn = os.path.join(os.getcwd(),'run_sims.bat')
with open(batfn, 'w') as bat:
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
            if inst.spatial:
                inst.init_patch_data(overwrite=overwrite)
                inst.init_fire(overwrite=overwrite)
                res_check = check_for_results(inst)
                if (res_check == False) or overwrite: 
                    batln = 'START /WAIT "title" "{}\Metapop.exe" "{}\\final.mp" /RUN=YES\n'.format(proj_info['ramas_loc'], inst.run_dir)
                    bat.write(batln)     
