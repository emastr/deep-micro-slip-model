import sys

sys.path.append("/home/emastr/phd/")


import numpy as np
from architecture.session import svdfno, fakesvdfno, egeofno
import torch
import torch.nn as nn
import torch.autograd as agrad
import matplotlib.pyplot as plt
from util.plot_tools import *
from architecture.fno_1d import *
from boundary_solvers.blobs import *
from boundary_solvers.geometry import *
from scipy.sparse.linalg import gmres, LinearOperator
from operators.stokes_operator import StokesAdjointBoundaryOp
from util.unet import *
import torch.nn as nn
import os


######
from boundary_solvers.geometry_torch import to_device, concat_dict_entries, geometry_to_net_input, unpack, GeomData, concat_dicts, subdict
from boundary_solvers.geometry import GPDomain, RoundedMicroGeomV2
from architecture.fno_1d import FNO1d
from hmm.stokes import StokesMicProb
from hmm.stokes_deep import DeepMicroSolver, get_net
from dataclasses import dataclass
import time


#######
DEVICE = "cpu"
DTYPETORCH = torch.float32
net_factory = egeofno #fakesvdfno

# Load the data using torch
old = False
if not old:
    #data = torch.load("/home/emastr/phd/data/article_training/svdfno_ver3_20000.Torch")
    #data = torch.load("/home/emastr/phd/data/article_training/svdfnof_ver1_20000.Torch")
    data = torch.load("/home/emastr/phd/data/article_training/fno_ver2_20000.Torch")
    print(data.keys())
    settings = data["settings"]
    settings["num_pts"] = 256
    settings["device"] = DEVICE
    settings["dtype"] = DTYPETORCH
    state_dict = data["state dict"]
    net = net_factory(settings, DEVICE, DTYPETORCH) # choice=2
    net.load_state_dict(state_dict)

if old:
    path = f"/mnt/data/emanuel/data/data/runs/fnoskip_big_data_49_{40000}.Torch"
    save = torch.load(path)
    print(save['settings'].keys())
    net, settings = get_net(path, 256, DEVICE, "float")
    print(settings)



###################
## Random data
default_parameters = {"kernel": "exp", "shape":.05, "num":20, "scale":.05, "bound":.3, "verbose": False, "width":1, "height":1, "corner_w":0.3, "line_pos":0.1, "n_refine":2, "n_corner_refine":0}
randomisers = {"width": lambda: 0.5 + np.random.rand()*0.5, 
               "scale": lambda: 0.01 + 10 ** (-np.random.rand()-1), 
               "height": lambda: 0.7 + np.random.rand()*0.5, 
               "corner_w": lambda: np.random.rand()*0.2 + 0.1, 
               "line_pos": lambda: np.random.rand()*0.1 + 0.1}
to_randomise = ["height", "corner_w", "line_pos", "scale"]
kwargs = {**default_parameters}

@dataclass
class MicData:
    geom: GPDomain

def get_funcs(k, eps):
    func = lambda t: -0.2*(1.5+np.sin(k*t)) * eps
    dfunc = lambda t: -0.2*k*np.cos(k*t) * eps
    ddfunc = lambda t: 0.2*k**2*np.sin(k*t) * eps
    return func, dfunc, ddfunc
    

geoms = []
fig = plt.figure(figsize=(10,10))
for i in range(4):
    for key in to_randomise:
        kwargs[key] = randomisers[key]()
        
    dom = GPDomain(**kwargs)
    
    eps = 0.01
    func, dfunc, ddfunc = get_funcs(i*4., eps)
    dom = RoundedMicroGeomV2(func, dfunc, ddfunc, width=1. * eps, height=1. * eps, corner_w=.2 * eps, line_pos=0.0*eps, center_x=10*i, n_refine=2)
    geoms.append(dom)
    
    plt.subplot(2,2,i+1)
    dom.plot(ax=plt.gca())
    
fig.savefig("/home/emastr/phd/data/figures/conditioning_geoms.png")


geom_data = [unpack(lambda x: torch.from_numpy(x)[None, :])(geometry_to_net_input(dom, 256, output=True)) for dom in geoms]
data = geom_data[0]
for g in geom_data[1:]:
    data = concat_dicts(data, g, axis=0)
data = GeomData.tfm_inp(data)
data = GeomData.tfm_out(data, data)


X = concat_dict_entries(subdict(data, settings["input_features"])).to(DEVICE).float()
solvers = [DeepMicroSolver(MicData(geom), net, settings, None) for geom in geoms]

fig = plt.figure()
solvers[1].plot()
fig.savefig("/home/emastr/phd/data/figures/conditioning_solution.png")


Y = net(X).detach().cpu().numpy()

i = 1
fig = plt.figure(figsize=(20, 20))
plt.subplot(221)
plt.plot(Y[i, 0, :])
plt.plot(data["rt"][i], '--')

plt.subplot(222)
plt.plot(Y[i, 1, :])
plt.plot(data["rn"][i], '--')

plt.subplot(223)
plt.plot(Y[i, 2, :])
plt.plot(data["drt_norm"][i], '--')

plt.subplot(224)
plt.plot(Y[i, 3, :])
plt.plot(data["drn_norm"][i], '--')

fig.savefig("/home/emastr/phd/data/figures/conditioning_solution_2.png")


    
