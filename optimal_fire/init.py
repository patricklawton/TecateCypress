import numpy as np
import signac as sg
from itertools import product

# Initialize signac project
project = sg.init_project()

b_vec = np.arange(0, 78, 2)
init_age_vec = np.arange(1,49,8).astype(int)
for b, init_age in product(b_vec, init_age_vec):
    sp = {'weibull_b': b, 'init_age': init_age}
    job = project.open_job(sp)
    job.init()
