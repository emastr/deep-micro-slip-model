import sys
import torch
from boundary_solvers.gauss_grid_2d import *
from boundary_solvers.geometry import *
import numpy as np
import matplotlib.pyplot as plt
from util.plot_tools import *
from boundary_solvers.blobs import *
import os
from scipy.interpolate import interp1d

path_data = "/home/emastr/github/deep-micro-slip-model/data"
path_geometries = "/mnt/data0/emastr/geometries/micro_geometries_boundcurv_high_variance/"
#path_torch = "/home/emastr/deep-micro-slip-model/data/micro_geometries_boundcurv_repar_256_torch_high_variance/"
path_torch = "/mnt/data0/home/emastr/geometries_torch/variance_norepar_512/"


os.makedirs(path_data, exist_ok=True)
os.makedirs(path_geometries, exist_ok=True)
os.makedirs(path_torch, exist_ok=True)


def reparameterize_data(t, dt, data, M=None):
    """Reparameterize the data to be a function of t in [0,1].
    Uses fouier series with 2*k_max+1 terms."""    
    #dtype = torch.complex64

    if M is None:
        M = len(t)
    data_np = data.numpy()
    t_np = t
    f = interp1d(t_np, data_np, axis=-1)
    t_np = np.linspace(0, 2*np.pi, M+1)[:-1]
    data_np = f(t_np)
    
    return torch.from_numpy(data_np)

# 200 000 data points, we run through
J = 10
Ntot = 190000
N = Ntot // J
M = 512#512
reparameterize = "none" #"arclen" # uniform / arclen
Mout = 256
err = []
idx = []

for j in range(J):
    start = j * N
    

    in_ch = ["zr", "zi", "dzr", "dzi", "ddzr", "ddzi", "inter_precomp_r", "inter_precomp_i", "inter_precomp_der_r", "inter_precomp_der_i", "w", "t"]
    out_ch = ["precomp_r", "precomp_i", "precomp_der_r", "precomp_der_i"]

    in_data = torch.zeros(N, len(in_ch), M)
    out_data = torch.zeros(N, len(out_ch), M)

    in_tmp = torch.zeros(N, len(in_ch), Mout)
    out_tmp = torch.zeros(N, len(out_ch), Mout)
    
    for i in range(N):
        print(f"{start + i + 1}/{Ntot} done", end="\r")
        data = torch.load(f"{path_geometries}/domain_{start + i}.GPDomain")
        precomp = data.pop("precomp")
        inter_precomp = data.pop("intermediate_precomp")
        precomp_der = data.pop("precomp_der")
        inter_precomp_der = data.pop("intermediate_precomp_der")
        
        err.append(data.pop("error"))
        idx.append(start + i)
        
        geom = GPDomain.load(data)
        t, w = geom.grid.get_grid_and_weights()
        z = geom.eval_param(derivative=0)
        dz = geom.eval_param(derivative=1)
        ddz = geom.eval_param(derivative=2)

        #n = dz/np.abs(dz)
        #r = np.imag(dz * np.conjugate(ddz))/np.abs(dz)**3 * n

        # Set in data
        in_data[i, 0, :] = torch.from_numpy(np.real(z))
        in_data[i, 1, :] = torch.from_numpy(np.imag(z))
        in_data[i, 2, :] = torch.from_numpy(np.real(dz))
        in_data[i, 3, :] = torch.from_numpy(np.imag(dz))
        in_data[i, 4, :] = torch.from_numpy(np.real(ddz))
        in_data[i, 5, :] = torch.from_numpy(np.imag(ddz))
        in_data[i, 6, :] = torch.from_numpy(np.real(inter_precomp))
        in_data[i, 7, :] = torch.from_numpy(np.imag(inter_precomp))
        in_data[i, 8, :] = torch.from_numpy(np.real(inter_precomp_der))
        in_data[i, 9, :] = torch.from_numpy(np.imag(inter_precomp_der))
        in_data[i, 10, :] = torch.from_numpy(w)
        in_data[i, 11, :] = torch.from_numpy(t)

        # set out data
        out_data[i, 0, :] = torch.from_numpy(np.real(precomp))
        out_data[i, 1, :] = torch.from_numpy(np.imag(precomp))
        out_data[i, 2, :] = torch.from_numpy(np.real(precomp_der))
        out_data[i, 3, :] = torch.from_numpy(np.imag(precomp_der))


        if reparameterize == "arclen":
            dt = np.zeros_like(t)
            dt[:-1] = t[1:] - t[:-1]
            dt[-1] = 2*np.pi - t[-1]
            l = np.cumsum(np.abs(dz) * dt)
            L = l[-1]
            l = np.roll(l, 1) / L *2*np.pi
            l[0] = 0.
            dl = np.abs(dz) * 2*np.pi / L * dt
            
            out_tmp[i, :, :] = reparameterize_data(l, dl, out_data[i, :, :], M=Mout)
            in_tmp[i, :, :] = reparameterize_data(l, dl, in_data[i, :, :], M=Mout)
            in_tmp[i, -1, :] = torch.linspace(0, 2*np.pi, Mout+1)[:-1]
            in_tmp[i, -2, :] = torch.ones(Mout) * 2*np.pi / Mout
        
        if reparameterize == "uniform":
            dt = np.zeros_like(t)
            t_ext = np.concatenate([t-2*np.pi, t, t+2*np.pi])
            out_ext = torch.cat((out_data[i, :, :], out_data[i, :, :], out_data[i, :, :]), dim=1)
            inp_ext = torch.cat((in_data[i, :, :], in_data[i, :, :], in_data[i, :, :]), dim=1)

            out_tmp[i, :, :] = reparameterize_data(t_ext, dt, out_ext, M=Mout)
            in_tmp[i, :, :] = reparameterize_data(t_ext, dt, inp_ext, M=Mout)
            in_tmp[i, -1, :] = torch.linspace(0, 2*np.pi, Mout+1)[:-1]
            in_tmp[i, -2, :] = torch.ones(Mout) * 2*np.pi / Mout
            
    if not (reparameterize == "none"):
        out_data = out_tmp
        in_data = in_tmp
    data = {"X": in_data, "Y": out_data, "info": {"in_ch": in_ch, "out_ch": out_ch}}
    torch.save(data, f"{path_torch}data_big_{j}.torch")    
    print(f"saved {j}") #breakb
