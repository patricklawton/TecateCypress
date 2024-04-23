import numpy as np
import signac as sg

# Initialize signac project
project = sg.init_project()

b_vec = np.arange(0, 122, 2)
for b in b_vec:
    sp = {'weibull_b': b}
    job = project.open_job(sp)
    job.init()
