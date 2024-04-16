import numpy as np
import signac as sg

# Initialize signac project
project = sg.init_project()

fire_probs = np.arange(0, 0.2, 0.01)
for fp in fire_probs:
    sp = {'fire_prob': fp}
    job = project.open_job(sp)
    job.init()
