from model import *

runnum = sys.argv[1]
#print(runnum); sys.exit()
run = open_run(runnum)
frame = 0

# Remove existing patchmap if it exists
patchfn = os.path.join(run.run_dir,'frame'+str(frame)+'_patchmap')
if os.path.isfile(patchfn+'.ASC'):
    os.remove(patchfn+'.ASC')

# Open up the ptc file as a subprocess
ptcfn = os.path.join(run.run_dir,'frame'+str(frame)+'.ptc')
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
time.sleep(0.15)
pg.press('enter')
# Wait for file to finish writing before continuing
while not os.path.isfile(patchfn+'.ASC'):
    pass
numlines = 0
numlines_final = run.lr_coord[1] - run.ul_coord[1] + proj_info['hd_lns'] + 1
while numlines < numlines_final:
    with open(patchfn+'.ASC', 'r') as patchmap:
        numlines = sum(1 for line in patchmap)
rowlen = 0
rowlen_final = run.lr_coord[0]-run.ul_coord[0]
while rowlen < rowlen_final:
    with open(patchfn+'.ASC', 'r') as patchmap:
        # Should find last line more efficiently, but following works for now
        for row in patchmap:
            pass
        rowlen = len(row.split())
        
# Terminate and wait for window to close before moving onto next frame
p.terminate()
while pg.locateOnScreen("ptc_file.png", confidence=0.9) is not None:
    pass
