import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import copy

# Define line function to be used for fitting later
def line(x, m, b):
    return m*x + b

def adjustmaps(maps):
    dim_len = []
    for dim in range(2):
        dim_len.append(min([m.shape[dim] for m in maps]))
    for mi, m in enumerate(maps):
        maps[mi] = m[0:dim_len[0], 0:dim_len[1]]
    return maps
