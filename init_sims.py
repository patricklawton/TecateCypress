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
            #run_check = [doc.get('patchmaps_extracted') == True, 
            #             doc.get('pch_files_written') == True]
            #if not run_check == [True for i in range(len(run_check))]:
            #    inst = Model(**info)
			#    res_check = check_for_results(inst)
            inst = Model(**info)
            res_check = check_for_results(inst)
            if doc.get('patchmaps_extracted') != True:
                inst.init_patch_data()
                write_to_doc(docfn, 'patchmaps_extracted', True)
            if doc.get('pch_files_written') != True:
                inst.write_pch()
                write_to_doc(docfn, 'pch_files_written', True) 
            if res_check != True: 
                batln = 'START /WAIT "title" "{}\Metapop.exe" "{}\\final.mp" /RUN=YES\n'.format(proj_info['ramas_loc'], inst.run_dir)
                bat.write(batln)     
