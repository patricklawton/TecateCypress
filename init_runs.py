from model import Model
import numpy as np
import os
import itertools
import json

# Read in shared ('project level') data
with open(os.path.join(os.getcwd(), 'project_info.json'), 'r') as info_file:
    proj_info = json.load(info_file)

# Constants across all runs
run_info = {}
# Non-spatial parameters
run_info['spcode'] = 'tcypre'
run_info['spname'] = 'Hesperocyparis forbesii'
run_info['timestep'] = 2
run_info['K_cell'] = 10794 #Maximum carrying-capacity per cell (i.e. when habitat suitability = 1)
run_info['R_max'] = 1 #Maximum growth rate
run_info['init_ab_frac'] = 0.9 #Initial abundance per hs*K_cell
run_info['rel_F'] = 1 #Relative fecundity
run_info['rel_S'] = 1 #Relative survival
run_info['min_patch_hs'] = 5
run_info['burn_in_period'] = 30 #In timesteps
# Boolean for spatial simulations and additional parameters
run_info['spatial'] = True
if run_info['spatial']:
    run_info['ul_coord'] = [1500, 2800] #Upper-left corner, in pixels relative to upper left corner of full SDM
    run_info['lr_coord'] = [2723, 3905] #Lower-right corner
    run_info['FDM_type'] = 'ecoregion'
    run_info['grid'] = True
    run_info['grid_length'] = 90
    run_info['grid_sep'] = 4
    run_info['cell_length'] = 0.27
    run_info['hs_threshold'] = 0.084
    run_info['n_distance'] = 3.0 #Neighborhood distance
    run_info['dist_metric'] = 'Default: Center to center' #For inter-patch distance calcs
    run_info['habitat_change'] = 'same until next' #Assume the same for K, F, S and between all frames
    run_info['dispersal'] = False
    # Decide if running on a fixed fire prob and/or habitat map
    run_info['fixed_fire'] = True
    if run_info['fixed_fire']:
        # Frame here refers to the FDM period, e.g. frame 0 -> 1980-2009
        run_info['fire_frame'] = 0
    run_info['fixed_habitat'] = False
    if run_info['fixed_habitat']:
        run_info['burn_in_frame'] = 0
        run_info['habitat_frame'] = 1
else:
    # Scan over fire probabilities with correponding FRIs from 5 to 80 years
    fire_probs = run_info['timestep'] / np.arange(4,142,2)
    #fire_probs = [2/40]

# Varying model parameters
land_use = ['LU'] #RAW, LU, LULUC
climate_model = ['cnrm'] #cnrm, hades, None
climate_scenario = ['rcp45'] #rcp45, rcp85, None

# Whether or not to overwrite existing data
overwrite = False

# Initialize runs
if run_info['spatial']:
    # Erase bat file from previous set of runs
    batfn = os.path.join(os.getcwd(),'run_spatials.bat')
    if os.path.isfile(batfn):
        os.remove(batfn)
    # Initialize run for each scenario
    for comb in itertools.product(land_use, climate_model, climate_scenario):
        run_info.update({'land_use':comb[0]})
        run_info.update({'climate_model':comb[1]})
        run_info.update({'climate_scenario':comb[2]})
        model = Model(**run_info)
        model.init_popmodel(overwrite=overwrite)
        model.init_maps()
        model.init_spatial(overwrite=overwrite)
else:
    # No spatial batch file necessary, just create simulation run batch file
    batfn = os.path.join(os.getcwd(),'run_sims.bat')
    with open(batfn, 'w') as bat:
        for fire_prob in fire_probs:
            run_info.update({'fire_prob': fire_prob})
            model = Model(**run_info)
            model.init_popmodel(overwrite=overwrite)
            batln = 'START /WAIT "title" "{}\Metapop.exe" "{}\\final.mp" /RUN=YES\n'.format(proj_info['ramas_loc'], model.run_dir)
            bat.write(batln)     
