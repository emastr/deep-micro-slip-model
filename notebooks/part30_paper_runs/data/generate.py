import sys
import torch
from boundary_solvers.gauss_grid_2d import *
from boundary_solvers.geometry import *
import numpy as np
import matplotlib.pyplot as plt
from util.plot_tools import *
from boundary_solvers.blobs import *
import os


path_data = "/home/emastr/github/deep-micro-slip-model/data"
path_geometries = "/home/emastr/github/deep-micro-slip-model/data/micro_geometries_boundcurv_high_variance/"

os.makedirs(path_data, exist_ok=True)
os.makedirs(path_geometries, exist_ok=True)
os.makedirs(path_torch, exist_ok=True)

np.random.seed(0)
N = 200000
plt.figure(figsize=(15,15))
riesz = []
zvec = []

worst_error = 0.
worst_idx = 0

default_parameters = {"kernel": "exp", "shape":.05, "num":20, "scale":.05, "bound":.3, "verbose": False, "width":1, "height":1, "corner_w":0.3, "line_pos":0.1, "n_refine":2, "n_corner_refine":0}
#randomisers = {"width": lambda: 0.5 + np.random.rand()*0.5, "scale": lambda: 0.01 + 10 ** (-np.random.rand()-1), "height": lambda: 0.5 + np.random.rand()*0.5, "corner_w": lambda: np.random.rand()*0.3 + 0.1, "line_pos": lambda: np.random.rand()*0.2 + 0.05}
randomisers = {"width": lambda: 0.2 + np.random.rand()*1., 
               "scale": lambda: 10**(-0.4-1.*np.random.rand()),#0.01 + 10 ** (-np.random.rand()-1), 
               "height": lambda: 0.7 + np.random.rand()*0.5, 
               "corner_w": lambda: np.random.rand()*0.2 + 0.1, 
               "line_pos": lambda: np.random.rand()*0.1 + 0.00}
to_randomise = ["height", "corner_w", "line_pos", "scale"]
#to_randomise = ["height", "line_pos", "corner_w"]
#to_randomise = ["line_pos"]
i = 0
while i < N:
    kwargs = {**default_parameters}
    for key in to_randomise:
        kwargs[key] = randomisers[key]()
    try:
        mg = GPDomain(**kwargs)
        
        r1, func = mg.precompute_line_avg(derivative=0, tol=1e-12, maxiter=200, verbose=False)
        r2, _    = mg.precompute_line_avg(derivative=1, tol=1e-12, maxiter=200, verbose=False)
        r3, _    = mg.line_eval_adjoint(derivative=0, tol=1e-12, maxiter=200, verbose=False)
        r4, _    = mg.line_eval_adjoint(derivative=1, tol=1e-12, maxiter=200, verbose=False)
        
        
        c = 1j + 1
        vec = np.conjugate(mg.eval_param()) - c
        a = mg.line_left - c
        b = mg.line_right - c
        
        num = func(vec)
        tru = np.real(b**2 - a**2)/2
        dif = abs(num - tru)
        worst_error = max(worst_error, dif)
        worst_idx = i if dif == worst_error else worst_idx
        print(f"{i+1}/{N} done. TEST > numerical: {num:.2f}, true: {tru:.2f}, difference: {dif:.2e}, worst: {worst_error:.2e}, idx: {worst_idx}", end="\r")
        
        mg.data["precomp"]     = r1
        mg.data["precomp_der"] = r2
        mg.data["intermediate_precomp"] = r3
        mg.data["intermediate_precomp_der"] = r4
        mg.data["error"] = dif
        
        assert dif < 1e-4
        
        #mg.plot(ax=plt.gca(), shownormals=True, showpts=True, npts=500)
        #mg.plot(ax=plt.gca(), npts=300)
        
        mg.save(f"{path_geometries}/domain_{i}.GPDomain")
        i = i + 1
    except:
        pass