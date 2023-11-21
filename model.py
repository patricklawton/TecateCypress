import numpy as np
import pandas as pd
import json
import os
import sys
import shutil
import subprocess as sp
import pyautogui as pg
import itertools
import h5py
import timeit
import time
from matplotlib import pyplot as plt
from contextlib import ExitStack

# Global constants
pop_idx_i = 44 #Pop 1 always appears on this line in mp files

# Open run
def open_run(runnum):
    infofn = os.path.join(os.getcwd(), 'runs', str(runnum), 'run_info.json')
    with open(infofn, 'r') as info_file:
        run_info = json.load(info_file)
    run = Model(**run_info)
    return run

# Function to add data to run document (run_doc.json) 
def write_to_doc(doc_fn, key, val):
    with open(doc_fn,'r+') as doc_file:
        doc = json.load(doc_file)
        doc.update({key: val})
        doc_file.seek(0)
        json.dump(doc, doc_file)
        doc_file.truncate()

# Function to read data from run document
def read_from_doc(doc_fn, key):
    with open(doc_fn, 'r') as doc_file:
        doc = json.load(doc_file)
        val = doc.get(key)
        if val == None:
            if proj_info['verbose']: print("{} key not found in run document".format(key))
        return val

# For copying uncropped maps to crop directory
def crop_and_copy_maps(mapsdir, modeldir, ul_coord, lr_coord):
    # Get list of maps to crop (only .asc files) 
    mapsdir_uncropped = os.path.join(os.getcwd(),'maps','crop0',modeldir)
    mapfiles = [file for file in os.listdir(mapsdir_uncropped) if file[-4:]=='.asc']
    # Find, crop, and copy map (.asc) files
    for mapfile in mapfiles:
        fn = os.path.join(mapsdir_uncropped,mapfile)
        if fn[-4:] == '.asc':
            # Open map to desired crop and save as file
            usecols = np.arange(ul_coord[0],lr_coord[0])
            map_cropped = np.loadtxt(fn,skiprows=proj_info['hd_lns']+ul_coord[1], 
                                     max_rows=lr_coord[1], usecols=usecols)
            croppedfn = os.path.join(mapsdir,mapfile)
            # Change file type to txt
            croppedfn = croppedfn.replace('.asc','.txt')  
            np.savetxt(croppedfn, map_cropped)

# Function to crop multiple maps to the smallest size
'''Not very foolproof, but works for now'''
def adjustmaps(maps):
    dim_len = []
    for dim in range(2):
        dim_len.append(min([m.shape[dim] for m in maps]))
    for mi, m in enumerate(maps):
        maps[mi] = m[0:dim_len[0], 0:dim_len[1]]
    return maps

# Read in shared ('project level') data
with open(os.path.join(os.getcwd(), 'project_info.json'), 'r') as info_file:
    proj_info = json.load(info_file)

class Model:
    def __init__(self, **kwargs):
        self.spcode = kwargs['spcode']
        self.spname = kwargs['spname']
        self.timestep = kwargs['timestep']
        self.K_cell = kwargs['K_cell'] #Maximum carrying-capacity per cell
        self.R_max = kwargs['R_max'] #Maximum growth rate
        self.init_ab_frac = kwargs['init_ab_frac'] #Initial abundance per hs*K_cell
        self.rel_F = kwargs['rel_F'] #Relative fecundity
        self.rel_S = kwargs['rel_S'] #Relative survival
        self.burn_in_period = kwargs['burn_in_period'] #In timesteps
        self.spatial = kwargs['spatial']
        if self.spatial:
            # Coordinates in pixels relative to upper left corner of full SDM
            self.ul_coord = kwargs['ul_coord'] #Upper-left corner 		
            self.lr_coord = kwargs['lr_coord'] #Upper-right corner
            self.land_use = kwargs['land_use'] 
            self.climate_model = kwargs['climate_model']
            self.climate_scenario = kwargs['climate_scenario']
            self.FDM_type = kwargs['FDM_type']
            self.grid = kwargs['grid']
            self.grid_length = kwargs['grid_length']
            self.grid_sep = kwargs['grid_sep']
            self.cell_length = kwargs['cell_length']
            self.hs_threshold = kwargs['hs_threshold']
            self.n_distance = kwargs['n_distance'] #Neighborhood distance
            self.min_patch_hs = kwargs['min_patch_hs']
            self.dist_metric = kwargs['dist_metric'] #For inter-patch distance
            self.habitat_change = kwargs['habitat_change'] #Change in K, F,and S btwn frames
            self.dispersal = kwargs['dispersal']
            self.fixed_habitat = kwargs['fixed_habitat']
            if self.fixed_habitat:
                self.habitat_frame = kwargs['habitat_frame']
            self.fixed_fire = kwargs['fixed_fire']
            if self.fixed_fire:
                self.fire_frame = kwargs['fire_frame']
        else:
            self.fire_prob = kwargs['fire_prob']
        
        # Compare existing simulation runs to desired run
        runsdir = os.path.join(os.getcwd(),'runs')
        if not os.path.isdir(runsdir):
            os.mkdir(runsdir)
        rundirs = os.listdir(runsdir)
        try:
            rundirs.remove('.DS_Store')
        except:
            pass
        rundirs = sorted([int(d) for d in rundirs])
        if len(rundirs) == 0:
            run_exists = False
            run_num = '0'
        else:
            run_exists = False
            for d in rundirs:
                with open(os.path.join(os.getcwd(),'runs',str(d),'run_info.json'), 'r') as run_info_prev:
                    run_info_prev = json.load(run_info_prev)
                    # Check each key value pair from previous runs' info
                    info_check = []
                    for key, val in run_info_prev.items():
                        if (key in list(self.__dict__.keys())) and (self.__dict__[key] == val):
                            info_check.append(True)
                        else:
                            info_check.append(False)
                    # Also check for any new keys in current run
                    keys_match = np.all([key in list(run_info_prev.keys()) for key in self.__dict__.keys()])
                    info_check.append(keys_match)
                    if np.all(info_check):
                        run_exists = True
                        run_num = d
            if run_exists == False:
                run_num = str(d+1)
        run_dir = os.path.join(os.getcwd(),'runs',str(run_num))
        if proj_info['verbose']: 
            print('using existing run: {}, run_num={}, run_dir={}'.format(run_exists,run_num,run_dir))

        # Make dir and info file if new run 
        if run_exists == False:
            os.makedirs(run_dir)
            with open(os.path.join(run_dir,'run_info.json'), 'w') as info_file:
                json.dump(self.__dict__, info_file)
        
		# Create/check run doc
        run_doc_fn = os.path.join(run_dir,'run_doc.json')
        if not os.path.isfile(run_doc_fn):
            with open(run_doc_fn,'w') as run_doc_file:
                json.dump({'run_dir': run_dir}, run_doc_file)
        else:
            with open(run_doc_fn,'r+') as run_doc_file:
                doc = json.load(run_doc_file)
                if doc.get('run_dir') != run_dir:
                    if proj_info['verbose']: print("Inconsistent run_dir in run document")
                    
        # Add run_dir and docfn to instance variable
        self.run_dir = run_dir
        self.docfn = run_doc_fn

    def check_results(self, doc_key, data_keys, overwrite=False):
        # Check if data has already been written
        # Return True if writing/overwriting, False otherwise
        with open(self.docfn, 'r') as doc_file:
            doc = json.load(doc_file)
        if (overwrite == False) and doc.get(doc_key):
            print('Data written, not overwriting')
            return False
        else:
            resultsfn = os.path.join(self.run_dir, 'results.h5')
            # Delete old results before overwriting
            if overwrite and doc.get(doc_key):
                print('Overwriting data')
                with h5py.File(resultsfn, 'a') as results:
                    for data_key in data_keys:
                        del results[data_key]
            return True

    # Function to get number of populations
    def get_num_pops(self):
        mpfn = os.path.join(self.run_dir,'final.mp')
        with open(mpfn, encoding="utf8", errors='ignore') as mp:
            for ln_idx, ln in enumerate(mp):
                # Final population always appears on the line before "Migration\n"
                if ln == "Migration\n":
                    num_pops = ln_idx-pop_idx_i+1 #Plus 1 for metapopulation (i.e. Pop 0)
                    break
        return num_pops

    def init_popmodel(self, overwrite=False):
        # Copy template file into run folder, change as needed
        '''
        Mostly skip this for now, assume the template file is fine
        '''
        if self.spatial == False:
            # Skip if already run and not overwriting
            if os.path.isfile(os.path.join(self.run_dir, 'final.mp')) and (overwrite==False): 
                print("Pop model already initialized, not overwriting")
                return
            shutil.copy2('./template.mp', os.path.join(self.run_dir, 'final.mp'))
            mpfn = os.path.join(self.run_dir,'final.mp')
            with open(mpfn, 'r') as mp:
                mpdata = mp.readlines()
            ln_idx = 44 #Population 1 always appears on this line
            ln_splt = mpdata[ln_idx].split(',')
            # Update the initial abundance
            ln_splt[3] = str(1000 * self.K_cell * self.init_ab_frac)
            # Update the fire probability
            ln_splt[12] = str(self.fire_prob) 
            mpdata[ln_idx] = ','.join(ln_splt)
            # Rewrite mp file
            with open(mpfn, 'w') as mp:
                for ln in mpdata:
                    mp.write(ln)
        else:
            # Skip if already run and not overwriting
            if os.path.isfile(os.path.join(self.run_dir, 'template.mp')) and (overwrite==False):
                print("Pop model already initialized, not overwriting")
                return
            shutil.copy2('./template.mp', self.run_dir)
        # Copy the RAMAS config file
        shutil.copy2('./RAMASGIS.CFG', self.run_dir)

    def init_maps(self):
        # Write info file for uncropped maps (i.e. crop0)
        # Should read data from one of the maps, for now can just hardcode
        uncropped_info = os.path.join(os.getcwd(),'maps','crop0','crop_info.json')
        if not os.path.isfile(uncropped_info):
            with open(uncropped_info, 'w') as info_file:
                json.dump({'ul_coord': [0,0], 'lr_coord': [3905, 2723]}, info_file)

        # Copy current (1995) uncropped map into each uncropped climate model/scenario subdir
        for mapsub in [os.path.join(os.getcwd(),'maps','crop0','SDM',self.__dict__['land_use']),
                      os.path.join(os.getcwd(),'maps','crop0','FDM',self.__dict__['FDM_type'])]:
            dirlist = [d for d in os.listdir(mapsub) if os.path.isdir(os.path.join(mapsub,d))]
            dirlist.remove('current')
            curr_files = os.listdir(os.path.join(mapsub,'current'))
            for d in dirlist:
                for curr in curr_files: 
                    currfn = os.path.join(mapsub, 'current', curr)
                    dest = os.path.join(mapsub, d)
                    if os.path.isfile(os.path.join(dest,curr)) == False:
                        print('copying {} to {}'.format(currfn,dest))
                        shutil.copy2(currfn, dest)

        # Set sub-dirs for this run's land use & climate model/scenario
        if (self.__dict__['climate_model']!=None) and (self.__dict__['climate_scenario']!=None):
            SDM_dir = os.path.join('SDM',self.__dict__['land_use'],
                                   self.__dict__['climate_model']+'_'+self.__dict__['climate_scenario'])
            FDM_dir = os.path.join('FDM',self.__dict__['FDM_type'],
                                  self.__dict__['climate_model']+'_'+self.__dict__['climate_scenario'])
        else:
            SDM_dir = os.path.join('SDM',self.__dict__['land_use'],'current')
            FDM_dir = os.path.join('FDM',self.__dict__['FDM_type'],'current')    

        # See if directory for this crop size exists yet
        for cropdir in os.listdir('./maps'):
            with open(os.path.join('./maps',cropdir,'crop_info.json'), 'r') as info_file:
                crop_info = json.load(info_file)
                match = np.all([self.__dict__[key]==val for key,val in crop_info.items()])
            if match:
                break
        if proj_info['verbose']: print('Existing crop found:',match)
        if proj_info['verbose'] and match: print(cropdir)

        # Crop and copy needed maps
        for modeldir in [SDM_dir,FDM_dir]:
            if match:
                # See if cropped maps for land use & climate model/scenario exist
                mapsdir = os.path.join('./maps',cropdir,modeldir)
                if os.path.isdir(mapsdir):
                    # Use the existing directory
                    if proj_info['verbose']: print('Using maps from', mapsdir)
                    '''Check if all the files actually exist'''
                    pass
                else:
                    # Make a new directory
                    if proj_info['verbose']: print('Creating cropped maps in', mapsdir)
                    os.makedirs(mapsdir)
                    crop_and_copy_maps(mapsdir, modeldir, self.ul_coord, self.lr_coord)
            else:
                # Make a new dir for desired crop
                cropdir = 'crop' + str(int(cropdir[4:])+1)
                # Write info file for this crop
                with open(os.path.join('./maps',cropdir,'crop_info.json'), 'w') as info_file:
                    json.dump({'ul_coord': self.__dict__['ul_coord'], 
                               'lr_coord': self.__dict__['lr_coord']}, info_file)
                if proj_info['verbose']: print('New crop:',cropdir)
                mapsdir = os.path.join('./maps',cropdir,modeldir)
                if proj_info['verbose']: print('Creating cropped maps in', mapsdir)
                os.makedirs(mapsdir)
                crop_and_copy_maps(mapsdir, modeldir, self.ul_coord, self.lr_coord)

        # Find/create grid if using
        if self.__dict__['grid'] == True:
            '''
            Skipping this step for now, just use the 90 cell grid
            Checked that the full grid (for all of Cali) works with a cropped SDM
            So only need to make a given grid for the full map, no cropped grids
            '''
            for fn in os.listdir('./grids'):
                sp = fn[:-4].split('_')
                if int(sp[0])==self.__dict__['grid_length'] and int(sp[1])==self.__dict__['grid_sep']:
                    gridfn = os.path.join('./grids',fn)
            if proj_info['verbose']: print('Using grid file',gridfn)

        # Write SDM,FDM and crop dirs to run doc
        write_to_doc(self.docfn, 'SDM_dir', SDM_dir)
        write_to_doc(self.docfn, 'FDM_dir', FDM_dir)
        write_to_doc(self.docfn, 'cropdir', cropdir)

    def init_spatial(self, overwrite=False):
        # Get data locations from run document
        SDM_dir = read_from_doc(self.docfn, 'SDM_dir')
        FDM_dir = read_from_doc(self.docfn, 'FDM_dir')
        cropdir = read_from_doc(self.docfn, 'cropdir')

        # Define time array
        times = np.arange(proj_info['ti'],proj_info['tf'],self.timestep)
        frames = np.arange(len(times))
        if self.fixed_habitat:
            times = [times[self.habitat_frame]]
            frames = [self.habitat_frame]

        # Check for completion of this run's spatial data
        spatial_finished = []
        for frame in range(len(times)):
            # 5 ptc output files, only checking one here
            fn = os.path.join(self.run_dir, 'frame{}.SCL'.format(frame))
            spatial_finished.append(os.path.isfile(fn))
        if not self.fixed_habitat:
            fn = os.path.join(self.run_dir, 'final-hist.TXT')
            spatial_finished.append(os.path.isfile(fn))
        spatial_finished = np.all(spatial_finished)
        print('spatial_finished', spatial_finished)

        # Continue if spatial data incomplete or overwriting
        if (not spatial_finished) or overwrite:
            # Open/make the .bat and .pdy files
            with ExitStack() as stack:    
                batfn = os.path.join(os.getcwd(), 'run_spatials.bat')
                bat = stack.enter_context(open(batfn, 'a'))
                if not self.fixed_habitat:
                    pdyfn = os.path.join(self.run_dir, 'patchdynamics.pdy')
                    pdy = stack.enter_context(open(pdyfn, 'w'))
                    # Write initial lines to .pdy file
                    pdy.write('Habitat Dynamics (version 4.1)\n')
                    pdy.write('\n'*5)
                    pdy.write('final.mp'+'\n')
                    pdy.write(('pop'+'\n')*3)
                    pdy.write(str(len(times))+'\n')

                for frame, t in zip(frames, times):
                    # Write ptc file
                    ptcfn = os.path.join(self.run_dir, 'frame'+str(frame)+'.ptc')
                    with open(ptcfn, 'w') as ptc:
                        # Normally file contains "map=ÿ" at the end of the first line, I'm omitting 
                        ptc.write('Landscape input file (4.1)\n')
                        ptc.write('\n'*5)
                        ptc.write(str(self.cell_length)+'\n')
                        if self.__dict__.get('grid'):
                            ptc.write('[Habitat]*[Grid]'+'\n'*2)
                        else:
                            ptc.write('[Habitat]'+'\n'*2)
                        ptc.write(str(self.hs_threshold)+'\n')
                        ptc.write(str(self.n_distance)+'\n')
                        ptc.write('Blue,False\n')
                        ptc.write('2\n')
                        ptc.write('thr(ths*{0}, {1}*{0})\n'.format(self.K_cell,self.min_patch_hs))
                        ptc.write(str(self.R_max)+'\n')
                        ptc.write('{0}*ths*{1}\n'.format(self.init_ab_frac,self.K_cell))
                        ptc.write(str(self.rel_F)+'\n')
                        ptc.write(str(self.rel_S)+'\n')
                        ptc.write('\n')
                        ptc.write(os.path.join(self.run_dir, 'template.mp')+'\n')
                        ptc.write('0\n1\n'*2) #For catastrophes, will replace with .PCH files downstream
                        ptc.write('No\n\n') #Not sure what this is for
                        ptc.write(self.dist_metric+'\n')
                        if self.grid:
                            ptc.write('2\n')
                        else:
                            ptc.write('1\n')
                        ptc.write('Habitat\n') #Name of SDM input map
                        if cropdir == 'crop0':
                            mapfn = os.path.join(os.getcwd(),'maps',cropdir,SDM_dir,self.spname+'_'+str(t)+'.asc')
                            ptc.write(mapfn+'\n')
                            ptc.write('ARC/INFO,ConstantMap\n')
                        else:
                            mapfn = os.path.join(os.getcwd(),'maps',cropdir,SDM_dir,self.spname+'_'+str(t)+'.txt')
                            ptc.write(mapfn+'\n')
                            ptc.write('fixed grid,ConstantMap\n')
                        ptc.write('Blue\n')
                        ptc.write(str(self.lr_coord[0]-self.ul_coord[0])+'\n')
                        if self.grid:
                            ptc.write('Grid\n') #Name of grid input map
                            gridfn = os.path.join(os.getcwd(),'grids',str(self.grid_length)+'_'+str(self.grid_sep)+'.asc')
                            ptc.write(gridfn+'\n')
                            ptc.write('ARC/INFO,ConstantMap\n')
                            ptc.write('Blue\n')
                            with open(gridfn, 'r') as grid: #Read num cols from asc
                                line = grid.readline().split(' ')
                                ptc.write(line[1]+'\n')
                        '''This next line is a mess, I'll figure this out later
                        At least know 18 is the time since last fire, EX is the dens dep
                        I think most/all of these are things specified in the linked .mp'''
                        ptc.write(',0.000,0.000,,EX,,,0.0,0.0,,0.0,1,0,TRUE,1,1,1,0.0,1,0,1,0,18,0,1.0,\n') 
                        # Assuming no migration or correlation for now
                        ptc.write('Migration\nTRUE\n')
                        ptc.write('0.00000,0.10000,0.00000,0.00000\n')
                        ptc.write('Correlation\nTRUE\n')
                        ptc.write('0.00000,0.10000,0.00000\n')
                        ptc.write('-End of file-\n')

                    # Write line(s) to .bat and .pdy files
                    ptc_path = os.path.join(self.run_dir, 'frame'+str(frame)+'.ptc')
                    batln = 'START /WAIT "title" "{}\SpatialData.exe" "{}" /RUN=YES\n'.format(proj_info['ramas_loc'], ptc_path)
                    bat.write(batln)
                    if not self.fixed_habitat:
                        pdy.write('frame'+str(frame)+'.ptc\n')
                        if frame == 0:
                            pdy.write('1\n')
                        else:
                            pdy.write(str(frame+self.burn_in_period)+'\n')
                        pdy.write((self.habitat_change+'\n')*3)
                # Write final lines to .bat file
                if not self.fixed_habitat:
                    pdy_path = os.path.join(self.run_dir, 'patchdynamics.pdy')
                    batln = 'START /WAIT "title" "{}\HabDyn.exe" "{}" /RUN=YES\n'.format(proj_info['ramas_loc'], pdy_path)
                    bat.write(batln)
                init_sims_path = os.path.join(os.getcwd(), 'init_sims.py')
                batln = 'START /WAIT "title" "python" "{}" /RUN=YES\n'.format(init_sims_path)
                bat.write(batln)
    
    def init_patch_data(self, overwrite=False):
        if read_from_doc(self.docfn, 'patch_data_initialized') and (overwrite == False):
            if proj_info['verbose']: print('patch data initialized, not overwriting')
            return

        if not self.fixed_habitat:
		    # Process hist file created by pdy file
            with open(os.path.join(self.run_dir, 'final-hist.TXT')) as hf:
                for ln_i,ln in enumerate(hf):
                    if ln_i < round((proj_info['tf']-proj_info['ti'])/self.timestep)+2:
                        pass
                    elif ln_i == round((proj_info['tf']-proj_info['ti'])/self.timestep)+2:
                        hist_df = pd.DataFrame(columns=ln.split())
                    else:
                        # Parse line
                        hist_df.loc[len(hist_df)] = ln.split()
            hist_df.to_pickle(os.path.join(self.run_dir, 'final-hist.pkl')) 
        
        # Define time array
        times = np.arange(proj_info['ti'],proj_info['tf'],self.timestep)
        frames = np.arange(len(times))
        if self.fixed_habitat:
            times = [times[self.habitat_frame]]
            frames = [self.habitat_frame]

        # Extract patch maps from ptc files
        for frame, t in zip(frames, times):
            # Remove existing patchmap if it exists
            patchfn = os.path.join(self.run_dir,'frame'+str(frame)+'_patchmap')
            if os.path.isfile(patchfn+'.ASC'):
                os.remove(patchfn+'.ASC')

            # Open up the ptc file as a subprocess
            ptcfn = os.path.join(self.run_dir,'frame'+str(frame)+'.ptc')
            p = sp.Popen(['{}\SpatialData.exe'.format(proj_info['ramas_loc']), ptcfn])
            while pg.locateOnScreen("ptc_file.png", confidence=0.9) is None:
                pass
            # Use the gui's upper left corner image as a reference point
            menuloc = pg.locateOnScreen("ptc_file.png", confidence=0.9)
            left, top, width, height = menuloc

            # Export the patchmap ASC file
            pg.click(x=left+0.5*width, y=top+0.75*height)
            pg.click(x=left+0.5*width, y=top+0.75*height+190)
            pg.write(patchfn)
            pg.press('enter')
            # Wait for file to finish writing before continuing
            while not os.path.isfile(patchfn+'.ASC'):
                pass
            numlines = 0
            numlines_final = self.lr_coord[1] - self.ul_coord[1] + proj_info['hd_lns'] + 1
            while numlines < numlines_final:
                with open(patchfn+'.ASC', 'r') as patchmap:
                    numlines = sum(1 for line in patchmap)
            rowlen = 0
            rowlen_final = self.lr_coord[0]-self.ul_coord[0]
            while rowlen < rowlen_final:
                with open(patchfn+'.ASC', 'r') as patchmap:
                    # Should find last line more efficiently, but following works for now
                    for row in patchmap:
                        pass
                    rowlen = len(row.split())

            if self.fixed_habitat:
                # Remove existing metapop file if it exists
                mpfn = os.path.join(self.run_dir,'final')
                if os.path.isfile(mpfn+'.mp'):
                    os.remove(mpfn+'.mp')
                # Export the metapopulation file
                pg.click(x=left+0.5*width, y=top+0.75*height)
                pg.click(x=left+0.5*width, y=top+0.75*height+150)
                pg.write(mpfn)
                pg.press('enter')
                # Wait for file to write before continuing
                while not os.path.isfile(patchfn+'.ASC'):
                    pass
                check = False
                while check == False:
                    with open(mpfn+'.mp', 'r') as mp:
                        # Check that last line is the final mp file line
                        for ln in mp:
                            pass
                        if ln == "-End of file-":
                            check = True
                    
            # Terminate and wait for window to close before moving onto next frame
            p.terminate()
            while pg.locateOnScreen("ptc_file.png", confidence=0.9) is not None:
                pass

        # Update run doc
        write_to_doc(self.docfn, 'patch_data_initialized', True)

    def init_fire(self, overwrite=False):
        if read_from_doc(self.docfn, 'fire_initialized') and (overwrite == False):
            if proj_info['verbose']: print('fire initialized, not overwriting')
            return

        # Get data locations from run document
        SDM_dir = read_from_doc(self.docfn, 'SDM_dir')
        FDM_dir = read_from_doc(self.docfn, 'FDM_dir')
        cropdir = read_from_doc(self.docfn, 'cropdir')

        num_pops = self.get_num_pops() 

        # Process the fdm time ranges from filenames
        fdmfns = os.listdir(os.path.join(os.getcwd(),'maps',cropdir,FDM_dir))
        fdm_rngs = []
        for fdmfn in fdmfns:
            spltfn = fdmfn.split('_')
            try:
                fdm_rngs.append([int(spltfn[-2]), int(spltfn[-1][:4])])
            except:
                # Hardcoded bc info is not in filename, also missing period 2010-2039
                fdm_rngs.append([1980,2039])
        sorted_indices = sorted(range(len(fdm_rngs)), key=fdm_rngs.__getitem__)
        fdm_rngs = [fdm_rngs[i] for i in sorted_indices]
        fdmfns = [fdmfns[i] for i in sorted_indices]
            
        if self.fixed_fire:
            # Read in fixed fire map 
            fdmfn = os.path.join(os.getcwd(),'maps',cropdir,FDM_dir,fdmfns[self.fire_frame])
            if fdmfn[-3:] == 'txt':
                fdm = np.loadtxt(fdmfn)
            else:
                # Assume these are uncropped .asc maps
                fdm = np.loadtxt(fdmfn, skiprows=6)
            # Translate 30-yr prob in FDM to 1-yr prob
            '''For now using Gregs approach'''
            fdm = np.ones(fdm.shape)-(np.ones(fdm.shape)-fdm)**(self.timestep/30)

        if not self.fixed_habitat:
            # Read in hist file dataframe
            hist_df = pd.read_pickle(os.path.join(self.run_dir, 'final-hist.pkl')) 

            # Calculate and write per patch fire probabilities frame by frame
            for frame, t in enumerate(np.arange(proj_info['ti'],proj_info['tf'],self.timestep)): 
                if not self.fixed_fire:
                    # Read in fire map for this frame
                    '''Can probably find fdm index using np.nonzero, but this works for now'''
                    fdm_idx = [i for i,rng in enumerate(fdm_rngs) if ((t<=max(rng)) and (t>=min(rng)))][0]
                    fdmfn = os.path.join(os.getcwd(),'maps',cropdir,FDM_dir,fdmfns[fdm_idx])
                    if fdmfn[-3:] == 'txt':
                        fdm = np.loadtxt(fdmfn)
                    else:
                        # Assume these are uncropped .asc maps
                        fdm = np.loadtxt(fdmfn, skiprows=6)
                    # Translate 30-yr prob in FDM to 1-yr prob
                    '''For now using Gregs approach'''
                    fdm = np.ones(fdm.shape)-(np.ones(fdm.shape)-fdm)**(self.timestep/30)

                # Read in patchmap and get patch ids used in this frame
                patchmapfn = os.path.join(self.run_dir,'frame'+str(frame)+'_patchmap.ASC')
                patchmap = np.loadtxt(patchmapfn, skiprows=6)
                num_patches_frame = int(max(np.unique(patchmap)))
                hist_df_sub = hist_df.loc[(hist_df['iter'] == str(int(frame + 1)))]
                patch_ids_frame = np.array(hist_df_sub['new2old'])[:num_patches_frame]
                # Store the initial patchmap
                if frame == 0:
                    patchmap_0 = patchmap
                patchmap_0, patchmap, fdm = adjustmaps([patchmap_0,patchmap,fdm])

                # Loop over all possible patch ids
                for patch_new in np.arange(num_pops)[1:]:
                    if (str(int(patch_new)) in patch_ids_frame) or ((frame == 0) and (patch_new <= num_patches_frame)):
                        # Find id in patchmap
                        if frame == 0:
                            patch = int(patch_new)
                        else:
                            row = hist_df.loc[(hist_df['new2old']==str(int(patch_new))) & (hist_df['iter']==str(int(frame+1)))]
                            patch = int(row['patch'].values[0])
                        # Calculate average probability
                        if self.dispersal:
                            #patchmap, fdm = adjustmaps([patchmap,fdm])
                            prob = np.mean(fdm[patchmap==patch])
                        else:
                            #patchmap_0, patchmap, fdm = adjustmaps([patchmap_0,patchmap,fdm])
                            prob = np.mean(fdm[(patchmap==patch) & (patchmap_0>0)])
                        if np.isnan(prob):
                            prob = 0.0
                    else:
                        prob = 0.0

                    pchfn = os.path.join(self.run_dir,'pop'+str(int(patch_new))+'.PCH')
                    pch_check = os.path.isfile(pchfn)
                    fmode = 'w' if frame == 0 else 'a'
                    with open(pchfn, fmode) as pch:
                        if frame == 0:
                            pch.write('1.0 {}\n'.format(prob))
                            # Write timesteps for the rest of the burn in period
                            for b_i in range(self.burn_in_period - 1):
                                pch.write('  {}\n'.format(prob))
                        else:
                            pch.write('  {}\n'.format(prob))

            # Specify pch files in final mp file
            mpfn = os.path.join(self.run_dir,'final.MP')
            with open(mpfn, 'r') as mp:
                mpdata = mp.readlines()
            for patch in range(num_pops)[1:]:
                #ln_idx = pop_idx_i + patch - 1
                ln = mpdata[pop_idx_i + patch - 1]
                ln_splt = ln.split(',')
                # Assume the following give the only options for pop number sep string
                for sep in [' ','A','S']:
                    pop_elem = ln_splt[0].split(sep)
                    if (len(pop_elem)==2) and (sep==' '):
                        pop_num = pop_elem[-1]
                    elif len(pop_elem) == 2:
                        pop_num = pop_elem[0]
                pchfn = 'pop'+pop_num+'.PCH'
                # Assume .PCH file specified in element 12 of the Pop line
                ln_splt[12] = pchfn
                mpdata[pop_idx_i + patch - 1] = ','.join(ln_splt)
            with open(mpfn, 'w') as mp:
                for ln in mpdata:
                    mp.write(ln)

        elif self.fixed_habitat and self.fixed_fire: 
            # Open full mp file for editing
            mpfn = os.path.join(self.run_dir,'final.MP')
            with open(mpfn, 'r') as mp:
                mpdata = mp.readlines()

            # Read in patchmap 
            patchmapfn = os.path.join(self.run_dir,'frame'+str(self.habitat_frame)+'_patchmap.ASC')
            patchmap = np.loadtxt(patchmapfn, skiprows=6)
            patchmap, fdm = adjustmaps([patchmap,fdm])

            # Loop over all possible patch ids
            for patch in range(num_pops)[1:]:
                # Calculate average fire probability
                prob = np.mean(fdm[patchmap==patch])
                # Edit mp file line
                ln = mpdata[pop_idx_i + patch - 1]
                ln_splt = ln.split(',') 
                ln_splt[12] = str(prob)
                mpdata[pop_idx_i + patch - 1] = ','.join(ln_splt)

            # Writed edited mp file
            with open(mpfn, 'w') as mp:
                for ln in mpdata:
                    mp.write(ln)

        else:
            sys.exit("Dynamic fire with fixed SDM not implemented ")

        # Update run doc
        write_to_doc(self.docfn, 'fire_initialized', True)

    def write_abundance_stats(self, overwrite=False):
        abundance = {}
        abund_keys = ['mean','stdev','min','max']

        # Check for previous results, delete if overwriting
        check_results = self.check_results('abundance_stats_written', abund_keys, overwrite)
        if not check_results: return 

        num_frames = int(np.round((proj_info['tf']-proj_info['ti'])/self.timestep))+self.burn_in_period
        mpfn = os.path.join(self.run_dir,'final.mp')
        init_abundances = []
        with open(mpfn, 'r') as mp:
            for ln_idx, ln in enumerate(mp):
                # Final population always appears on the line before "Migration\n"
                if ln == "Migration\n":
                    num_pops = ln_idx-pop_idx_i+1 #Plus 1 for metapopulation (i.e. Pop 0)
                if ln[:4] == 'Pop ':
                    ln_splt = ln.split(',')
                    init_abundances.append(int(ln_splt[3]))
                if ln == 'Pop. ALL\n':
                    res_idx_i = ln_idx
                    break
        for key in abund_keys:
            abundance[key] = {}
            for pop_idx in range(num_pops):
                if pop_idx == 0:
                    if key == 'stdev':
                        abundance[key]['pop'+str(pop_idx)] = [0]
                    else:
                        abundance[key]['pop'+str(pop_idx)] = [sum(init_abundances)]
                elif pop_idx <= len(init_abundances):
                    if key == 'stdev':
                        abundance[key]['pop'+str(pop_idx)] = [0]
                    else:
                        abundance[key]['pop'+str(pop_idx)] = [init_abundances[pop_idx-1]]
                else:
                    abundance[key]['pop'+str(pop_idx)] = [0]
        pop_idx = -1
        with open(mpfn, 'r') as mp:
            for ln_idx, ln in enumerate(mp):
                frame_idx = (ln_idx-res_idx_i)%(num_frames+1)
                if ln_idx < res_idx_i:
                    pass
                elif frame_idx == 0:
                    pop_idx += 1
                    # Stop reading lines when this appears
                    if ln == 'Occupancy\n':
                        break
                else:
                    '''Allowing for floats, but shouldnt these always be integers?'''
                    ln_splt = []
                    for val in ln.split():
                        try:
                            ln_splt.append(int(val))
                        except:
                            ln_splt.append(float(val))
                    for key_idx, abund_key in enumerate(abund_keys):
                        abundance[abund_key]['pop'+str(pop_idx)].append(ln_splt[key_idx])

        # Write data to h5 file in run directory
        resultsfn = os.path.join(self.run_dir, 'results.h5')
        with h5py.File(resultsfn, 'a') as results:
            for abund_key in abundance.keys():
                for pop_idx in range(num_pops):
                    dset_name = abund_key+'/pop'+str(pop_idx)
                    dset = abundance[abund_key]['pop'+str(pop_idx)]
                    results.create_dataset(dset_name, data=dset)
        # Update run doc
        write_to_doc(self.docfn, 'abundance_stats_written', True)

    def write_patch_centroids(self, overwrite=False):
        # Store patch centroid coordinates at each timestep
        '''Could combine this with init_fire to save time from reading in patchmap'''
        # Check for previous results, delete if overwriting
        check_results = self.check_results('patch_centroids_written', ['patch_centroids'], overwrite)
        if not check_results: return 

        # Initialize data
        num_pops = self.get_num_pops()
        centroids = {}
        for pop_idx in range(num_pops):
            centroids['pop'+str(pop_idx)] = []

        if not self.fixed_habitat:
            # Read in hist file dataframe
            hist_df = pd.read_pickle(os.path.join(self.run_dir, 'final-hist.pkl')) 

            # Calculate the centroids of each patch
            for frame, t in enumerate(np.arange(proj_info['ti'],proj_info['tf'],self.timestep)):
                # Get this frames' patchmap
                patchmapfn = os.path.join(self.run_dir,'frame'+str(frame)+'_patchmap.ASC')
                patchmap = np.loadtxt(patchmapfn, skiprows=6)
                num_patches_frame = int(max(np.unique(patchmap)))
                hist_df_sub = hist_df.loc[(hist_df['iter'] == str(int(frame + 1)))]
                patch_ids_frame = np.array(hist_df_sub['new2old'])[:num_patches_frame]
                # Loop over all possible patch ids
                for patch_new in np.arange(num_pops)[1:]:
                    # If patch exists in this frame, find its centroid
                    if (str(int(patch_new)) in patch_ids_frame) or ((frame == 0) and (patch_new <= num_patches_frame)):
                        # Find id in patchmap
                        if frame == 0:
                            patch = int(patch_new)
                        else:
                            row = hist_df.loc[(hist_df['new2old']==str(int(patch_new))) & (hist_df['iter']==str(int(frame+1)))]
                            patch = int(row['patch'].values[0])
                        patch_coords = np.argwhere(patchmap==patch)
                        length = patch_coords.shape[0]
                        sum_x = np.sum(patch_coords[:, 0])
                        sum_y = np.sum(patch_coords[:, 1])
                        x, y = sum_x/length, sum_y/length
                    # Otherwise, set the centroid to the origin as a placeholder
                    else:
                        x, y = (0.0, 0.0)
                    if frame == 0:
                        for b_i in range(self.burn_in_period):
                            centroids['pop'+str(int(patch_new))].append([x,y])
                    else:
                        centroids['pop'+str(int(patch_new))].append([x,y])
        else:
            # Read in patchmap 
            patchmapfn = os.path.join(self.run_dir,'frame'+str(self.habitat_frame)+'_patchmap.ASC')
            patchmap = np.loadtxt(patchmapfn, skiprows=6) 

            # Loop over all patches
            for patch in range(num_pops)[1:]:
                patch_coords = np.argwhere(patchmap==patch)
                length = patch_coords.shape[0]
                sum_x = np.sum(patch_coords[:, 0])
                sum_y = np.sum(patch_coords[:, 1])
                x, y = sum_x/length, sum_y/length
                centroids['pop'+str(patch)].append([x,y])

        # Write to results file
        resultsfn = os.path.join(self.run_dir, 'results.h5')
        with h5py.File(resultsfn, 'a') as results:
            for pop_idx in range(num_pops):
                dset_name = 'patch_centroids/pop'+str(pop_idx)
                dset = centroids['pop'+str(pop_idx)]
                results.create_dataset(dset_name, data=dset)
        # Update run doc
        write_to_doc(self.docfn, 'patch_centroids_written', True)

    def write_fire_prob(self, overwrite=False):
        # Check for previous results, delete if overwriting
        check_results = self.check_results('fire_prob_written', ['fire_prob'], overwrite)
        if not check_results: return 

        # Initialize data
        num_pops = self.get_num_pops()
        fire_probs = {}
        for pop_idx in range(num_pops):
            fire_probs['pop'+str(pop_idx)] = []

        if self.fixed_habitat and self.fixed_fire:
            mpfn = os.path.join(self.run_dir,'final.MP')
            with open(mpfn, 'r') as mp:
                mpdata = mp.readlines()
            for patch in range(num_pops)[1:]:
                ln = mpdata[pop_idx_i + patch - 1]
                ln_splt = ln.split(',') 
                fire_probs['pop'+str(patch)].append(float(ln_splt[12]))
        else:
            for patch in range(num_pops)[1:]:
                pchfn = os.path.join(self.run_dir, 'pop'+str(patch)+'.PCH')
                with open(pchfn, 'r') as pchfile:
                    pch = pchfile.readlines()
                #patch_probs = [float(ln.split()[-1]) for ln in pch]
                for ln in pch:
                    fire_probs['pop'+str(patch)].append(float(ln.split()[-1]))

        # Write to results file
        resultsfn = os.path.join(self.run_dir, 'results.h5')
        with h5py.File(resultsfn, 'a') as results:
            for pop_idx in range(num_pops):
                dset_name = 'fire_prob/pop'+str(pop_idx)
                dset = fire_probs['pop'+str(pop_idx)]
                results.create_dataset(dset_name, data=dset)
        # Update run doc
        write_to_doc(self.docfn, 'fire_prob_written', True)
    
    def write_patch_K(self, overwrite=False):
        # Check for previous results, delete if overwriting
        check_results = self.check_results('patch_K_written', ['K'], overwrite)
        if not check_results: return 

        # Initialize data
        num_pops = self.get_num_pops()
        Ks = {}
        for pop_idx in range(num_pops):
            Ks['pop'+str(pop_idx)] = []

        if self.fixed_habitat and self.fixed_fire:
            mpfn = os.path.join(self.run_dir,'final.MP')
            with open(mpfn, 'r') as mp:
                mpdata = mp.readlines()
            for patch in range(num_pops)[1:]:
                ln = mpdata[pop_idx_i + patch - 1]
                ln_splt = ln.split(',') 
                Ks['pop'+str(patch)].append(float(ln_splt[6]))
        else:
            for patch in range(num_pops)[1:]:
                kchfn = os.path.join(self.run_dir, 'pop'+str(patch)+'.KCH')
                with open(kchfn, 'r') as kchfile:
                    kch = kchfile.readlines()
                for ln in kch:
                    Ks['pop'+str(patch)].append(float(ln.split()[-1]))

        # Write to results file
        resultsfn = os.path.join(self.run_dir, 'results.h5')
        with h5py.File(resultsfn, 'a') as results:
            for pop_idx in range(num_pops):
                dset_name = 'K/pop'+str(pop_idx)
                dset = Ks['pop'+str(pop_idx)]
                results.create_dataset(dset_name, data=dset)
        # Update run doc
        write_to_doc(self.docfn, 'patch_K_written', True)
