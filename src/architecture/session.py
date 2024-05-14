import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from util.plot_tools import *
from architecture.fno_1d import *
from boundary_solvers.blobs import *
from boundary_solvers.geometry import *
from boundary_solvers.geometry_torch import GeomData
from architecture.unet import *
from util.logger import EventTracker
from torch.utils.data import Dataset, DataLoader, random_split
from util.dashboard import DashBoard
from boundary_solvers.geometry_torch import norm

# Meta
DTYPETORCH = torch.float
DTYPEPYTHN = "float"
CTYPETORCH = torch.cfloat
DEVICE = "cpu"

class SkipNet(nn.Module):
    def __init__(self, net, settings, device=DEVICE):
        super(SkipNet, self).__init__()
        print("Warning: old skipnet had no scale parameter")
        
        self.scale = nn.Parameter(torch.ones(4, dtype=DTYPETORCH).to(device))
        self.net = net
        self.settings_ = settings
        
    def forward(self, x):
        if self.settings_["skip"]:
            return self.net(x) + torch.einsum('ij,bjm->bim', torch.diag(self.scale), x[:, -4:, :])
        else:
            return self.net(x) 
    
    def settings(self):
        return self.settings_


class FNOsvd(nn.Module):
    def __init__(self, basis_size, **kwargs):
        super(FNOsvd, self).__init__()
        
        
        # Basis In
        self.fno_svd = FNO1d(out_channels=basis_size*2*4, **kwargs)
        self.basis_size = basis_size
        self.kwargs = kwargs
        
    def forward(self, x):
        """
        Last two arguments are vx and vy. Do Fourier on the rest.
        """
        v = x[:,-4:, :] # [m, 2, n]
        
        m = v.shape[0]
        b = self.basis_size
        n = v.shape[-1]
        
        # Compute learned basis
        svd = self.fno_svd(x[:, :-4, :]) # [m, b, n]
        svd_in =  svd[:, :4*b, :].view(m, 4, b, n) # [m, 2, b, n]
        svd_out = svd[:, 4*b:, :].view(m, 4, b, n) # [m, 2, b, n]
        
        # Normalise basis
        dots = torch.einsum('ijk,ijlk->ijl', v, svd_in) # [m, 2]
        out = torch.einsum('ijl,ijlk->ijk', dots, svd_out) # [m, 2, n]
        return out
    
    
    def assemble_matrix(self, x, stride=1):
        v = x[:,-4:, :] # [m, 2, n]
        m = v.shape[0]
        b = self.basis_size
        
        # Compute learned basis
        svd = self.fno_svd(x[:, :-4, :])[:, :, ::stride] # [m, b, n]
        
        n =svd.shape[-1]
        svd_in =  svd[:, :4*b, :].view(m, 4, b, n) # [m, 2, b, n]
        svd_out = svd[:, 4*b:, :].view(m, 4, b, n) # [m, 2, b, n]
        
        mat = torch.einsum('mijk,mabc->mijkabc', svd_out, svd_in) # [m, 2]
        #rint(mat.shape)
        #print(m, b, n)
        matview = mat.reshape(m, 4*b*n, 4*b*n)
        return matview
    
    def plot(self, ax, x):
        mat = self.assemble_matrix(x, stride=6).detach().cpu().squeeze()
        mat = torch.eye(mat.shape[0]) + mat
        matflat = mat[:, 0]
        std = torch.std(matflat)
        mean = torch.mean(matflat)
        
        ax.imshow(mat, vmin=mean-2*std, vmax=mean+2*std)
        ax.set_title(f"Cond: {np.linalg.cond(mat):.2e}")
        pass
    

class FNOsvd2(nn.Module):
    def __init__(self, basis_size, **kwargs):
        super(FNOsvd2, self).__init__()
        
        
        # Basis In
        self.fno_svd = FNO1d(out_channels=basis_size*2*2, **kwargs)
        self.basis_size = basis_size
        self.kwargs = kwargs
        
    def forward(self, x):
        """
        Last two arguments are vx and vy. Do Fourier on the rest.
        """
        v1 = x[:,-4:-2, :] # [m, 2, n]
        v2 = x[:,-2:, :] # [m, 2, n]
        
        m = v1.shape[0]
        b = self.basis_size
        n = v1.shape[-1]
        
        # Compute learned basis
        svd = self.fno_svd(x[:, :-4, :]) # [m, b, n]
        svd_in =  svd[:, :2*b, :].view(m, 2, b, n) # [m, 2, b, n]
        svd_out = svd[:, 2*b:, :].view(m, 2, b, n) # [m, 2, b, n]
        
        # Normalise basis
        v1 = torch.einsum('ijk,ijlk->ijl', v1, svd_in) # [m, 2]
        v2 = torch.einsum('ijk,ijlk->ijl', v2, svd_in) # [m, 2]
        
        v1 = torch.einsum('ijl,ijlk->ijk', v1, svd_out) # [m, 2, n]
        v2 = torch.einsum('ijl,ijlk->ijk', v2, svd_out) # [m, 2, n]
        
        return torch.cat([v1, v2], dim=1)
    
    
    def assemble_matrix(self, x, stride=1):
        v = x[:,-4:, :] # [m, 2, n]
        b = self.basis_size
        
        # Compute learned basis
        svd = self.fno_svd(x[:, :-4, :])[0, :, ::stride] # [m, b, n]
        
        n =svd.shape[-1]
        svd_in =  svd[:2*b, :].view(2, b, n)# [m, 2, b, n]
        svd_out = svd[2*b:, :].view(2, b, n) # [m, 2, b, n]
        
        mat = torch.einsum('rbi,sbj->rsij', svd_out, svd_in) # [m, 2]
        mat = torch.vstack([torch.hstack([mat[0,0], mat[0,1]]), torch.hstack([mat[1,0], mat[1,1]])])
        #rint(mat.shape)
        #print(m, b, n)

        return mat
    
    def plot(self, ax, x):
        mat = self.assemble_matrix(x, stride=3).detach().cpu().squeeze()
        mat = torch.eye(mat.shape[0]) + mat
        matflat = mat[:, 0]
        std = torch.std(matflat)
        mean = torch.mean(matflat)
        
        ax.imshow(mat, vmin=mean-2*std, vmax=mean+2*std)
        ax.set_title(f"Cond: {np.linalg.cond(mat):.2e}")
        pass
    
    
class FNOsvd3(nn.Module):
    def __init__(self, basis_size, scale=False, device=DEVICE, **kwargs):
        super(FNOsvd3, self).__init__()
        
        
        # Basis In
        self.scale1 = nn.Parameter(torch.ones(1, dtype=DTYPETORCH).to(device))
        self.scale2 = nn.Parameter(torch.ones(1, dtype=DTYPETORCH).to(device))
        self.fno_svd = FNO1d(out_channels=basis_size*2*2, **kwargs)
        self.basis_size = basis_size
        self.kwargs = kwargs
        
    def forward(self, x):
        """
        Last two arguments are vx and vy. Do Fourier on the rest.
        """
        v1 = x[:,-4:-2, :] # [m, 2, n]
        v2 = x[:,-2:, :] # [m, 2, n]
        
        m = v1.shape[0]
        b = self.basis_size
        n = v1.shape[-1]
        
        # Compute learned basis
        svd = self.fno_svd(x[:, :-4, :]) # [m, b, n]
        svd_in =  svd[:, :2*b, :].view(m, 2, b, n) # [m, 2, b, n]
        svd_out = svd[:, 2*b:, :].view(m, 2, b, n) # [m, 2, b, n]
        
        # Normalise basis
        v1 = torch.einsum('ijk,ijlk->ijl', v1, svd_in) # [m, 2]
        v2 = torch.einsum('ijk,ijlk->ijl', v2, svd_in) # [m, 2]
        
        v1 = torch.einsum('ijl,ijlk->ijk', v1, svd_out) # [m, 2, n]
        v2 = torch.einsum('ijl,ijlk->ijk', v2, svd_out) # [m, 2, n]
        
        return torch.cat([self.scale1 * v1, self.scale2 * v2], dim=1)
        
    
    
    def assemble_matrix(self, x, stride=1):
        v = x[:,-4:, :] # [m, 2, n]
        b = self.basis_size
        
        # Compute learned basis
        svd = self.fno_svd(x[:, :-4, :])[0, :, ::stride] # [m, b, n]
        
        n =svd.shape[-1]
        svd_in =  svd[:2*b, :].view(2, b, n)# [m, 2, b, n]
        svd_out = svd[2*b:, :].view(2, b, n) # [m, 2, b, n]
        
        mat = torch.einsum('rbi,sbj->rsij', svd_out, svd_in) # [m, 2]
        mat = torch.vstack([torch.hstack([mat[0,0], mat[0,1]]), torch.hstack([mat[1,0], mat[1,1]])])
        #rint(mat.shape)
        #print(m, b, n)

        return mat
    
    def plot(self, ax, x):
        mat = self.assemble_matrix(x, stride=3).detach().cpu().squeeze()
        mat = torch.eye(mat.shape[0]) + mat
        matflat = mat[:, 0]
        std = torch.std(matflat)
        mean = torch.mean(matflat)
        
        ax.imshow(mat, vmin=mean-2*std, vmax=mean+2*std)
        ax.set_title(f"Cond: {np.linalg.cond(mat):.2e}")
        pass
    
    
class FNOfakeSVD(nn.Module):
    def __init__(self, basis_size, scale=False, **kwargs):
        super(FNOfakeSVD, self).__init__()
        
        
        # Basis In
        self.fno_svd = FNO1d(out_channels=basis_size*2*2, **kwargs)
        self.basis_size = basis_size
        self.kwargs = kwargs
        
    def forward(self, x):
        """
        Last two arguments are vx and vy. Do Fourier on the rest.
        """
        v1 = x[:,-4:-2, :] # [m, 2, n]
        v2 = x[:,-2:, :] # [m, 2, n]
        
        m = v1.shape[0]
        b = self.basis_size
        n = v1.shape[-1]
        
        # Compute learned basis
        svd = self.fno_svd(x) # [m, b, n]
        svd_in =  svd[:, :2*b, :].view(m, 2, b, n) # [m, 2, b, n]
        svd_out = svd[:, 2*b:, :].view(m, 2, b, n) # [m, 2, b, n]
        
        # Normalise basis
        v1 = torch.einsum('ijk,ijlk->ijl', v1, svd_in) # [m, 2]
        v2 = torch.einsum('ijk,ijlk->ijl', v2, svd_in) # [m, 2]
        
        v1 = torch.einsum('ijl,ijlk->ijk', v1, svd_out) # [m, 2, n]
        v2 = torch.einsum('ijl,ijlk->ijk', v2, svd_out) # [m, 2, n]
        
        return torch.cat([v1, v2], dim=1)
        
    
    
    def assemble_matrix(self, x, stride=1):
        v = x[:,-4:, :] # [m, 2, n]
        b = self.basis_size
        
        # Compute learned basis
        svd = self.fno_svd(x)[0, :, ::stride] # [m, b, n]
        
        n =svd.shape[-1]
        svd_in =  svd[:2*b, :].view(2, b, n)# [m, 2, b, n]
        svd_out = svd[2*b:, :].view(2, b, n) # [m, 2, b, n]
        
        mat = torch.einsum('rbi,sbj->rsij', svd_out, svd_in) # [m, 2]
        mat = torch.vstack([torch.hstack([mat[0,0], mat[0,1]]), torch.hstack([mat[1,0], mat[1,1]])])
        #rint(mat.shape)
        #print(m, b, n)

        return mat
    
    def plot(self, ax, x):
        mat = self.assemble_matrix(x, stride=3).detach().cpu().squeeze()
        mat = torch.eye(mat.shape[0])*0.3 + mat
        matflat = mat[:, 0]
        std = torch.std(matflat)
        mean = torch.mean(matflat)
        
        ax.imshow(mat, vmin=mean-2*std, vmax=mean+2*std)
        ax.set_title(f"Cond: {np.linalg.cond(mat):.2e}")
        pass
   
# Try different models
# Data: dictionary with M x 256 resolution entry
# Model: input_Tfm -> Cat -> Net -> Output_Tfm -> unCat
# Training: (input_Tfm(data), inv(output_Tftm)(data)) -> loss(Net)

def egeofno(settings, device, dtype):
    
    net = FNO1d(modes=settings["modes"], 
                in_channels=len(settings["input_features"]), 
                out_channels=len(settings["output_features"]), 
                layer_widths=settings["layer_widths"],
                bias = settings["bias"],
                activation=settings["activation"],
                kernel_size=settings["kernel_size"],
                batch_norm=settings["batch_norm"],
                dtype=dtype)
    
    net = SkipNet(net, settings)
    net.to(device)   
    return net
    
    
def egeofno_ver1(device=DEVICE):
    # Features
    inp_features = ['c_norm', 'vt', 'vn', 'dvt_norm', 'dvn_norm']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']

    # Model
    settings = {"modes": 80,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [4*len(inp_features),] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "skip": True,
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True,
                "amsgrad": False}
    return egeofno(settings, device, DTYPETORCH)


def egeofno_ver2(device=DEVICE):
    # Features
    inp_features = ['c_norm', 'vt', 'vn', 'dvt_norm', 'dvn_norm']#GeomData.PREDEFINED_INPUTS['FIX:invariant-natural']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']

    # Model
    settings = {"modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [4*len(inp_features),] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "skip": True,
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True,
                "amsgrad": False}
    
    return egeofno(settings, device, DTYPETORCH)


def egeofno_ver3(device=DEVICE):
    # Features
    inp_features = ['c_norm', 'vt', 'vn', 'dvt_norm', 'dvn_norm']#GeomData.PREDEFINED_INPUTS['FIX:invariant-natural']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']

    # Model
    settings = {"modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [4*len(inp_features),] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "skip": True,
                "bias": True,
                'h1_weight': 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True,
                "amsgrad": False}
    return egeofno(settings, device, DTYPETORCH)
        
    
def egeofno_ver4(device=DEVICE):
    # Features
    inp_features = ['c_norm', 'vt', 'vn', 'dvt_norm', 'dvn_norm']#GeomData.PREDEFINED_INPUTS['FIX:invariant-natural']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']

    # Model
    settings = {"modes": 20,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [4*len(inp_features),] * 2, #(3,8) works, (2,8) worse. (8, 3) best so far
                "skip": True,
                "bias": True,
                "activation": F.gelu,
                "h1_weight": 1.0,
                "kernel_size": 1,
                "batch_norm": True,
                "amsgrad": False}
    return egeofno(settings, device, DTYPETORCH)   

   
def egeofno_ver5(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['FIX:invariant-natural-dir']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']

    # Model
    settings = {"modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [5*len(inp_features),] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "skip": True,
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True,
                "amsgrad": False}
    
    return egeofno(settings, device, DTYPETORCH)



## SVD FNO

def svdfno(settings, device, dtype, choice=1):
    fnosvds = [FNOsvd, FNOsvd2, FNOsvd3]
    FNOsvd_ = fnosvds[choice]
    net = FNOsvd_(basis_size=settings["basis_size"],
                 modes=settings["modes"], 
                 in_channels=len(settings["input_features"])-4, 
                 layer_widths=settings["layer_widths"],
                 bias = settings["bias"],
                 activation=settings["activation"],
                 kernel_size=settings["kernel_size"],
                 batch_norm=settings["batch_norm"],
                 dtype=dtype)
    
    net = SkipNet(net, settings)
    net.to(device)   
    return net


def svdfno_ver1(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['FIX:invariant-natural-dir']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']
    basis_size = 12
    # Model
    settings = {"basis_size": basis_size,
                "skip": True,
                "modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [4*s for s in [1, 1, 1, 2, 4, 8, basis_size]], #(3,8) works, (2,8) worse. (8, 3) best so far
                "bias": True,
                "activation": F.gelu,
                "kernel_size": 1,
                "h1_weight": 1.0,
                "batch_norm": True}
    
    return svdfno(settings, device, DTYPETORCH)


def svdfno_ver2(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['FIX:invariant-natural-dir']
    #out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural-dir']
    basis_size = 8
    # Model
    settings = {"basis_size": basis_size,
                "skip": True,
                "modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [basis_size * 2 * 2,] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True}
    
    return svdfno(settings, device, DTYPETORCH)


def svdfno_ver3(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['FIX:invariant-natural']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']
    basis_size = 5
    # Model
    settings = {"basis_size": basis_size,
                "skip": True,
                "modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [basis_size * 2 * 2,] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True}
    
    return svdfno(settings, device, DTYPETORCH, choice=2)


def svdfno_ver4(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['FIX:invariant-natural']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']
    basis_size = 5
    # Model
    settings = {"basis_size": basis_size,
                "skip": True,
                "modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [basis_size,]*4 + [basis_size * 2 * 2,] * 3, #(3,8) works, (2,8) worse. (8, 3) best so far
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True}
    
    return svdfno(settings, device, DTYPETORCH)


def fakesvdfno(settings, device, dtype):
    net = FNOfakeSVD(basis_size=settings["basis_size"],
                 modes=settings["modes"], 
                 in_channels=len(settings["input_features"]), 
                 layer_widths=settings["layer_widths"],
                 bias = settings["bias"],
                 activation=settings["activation"],
                 kernel_size=settings["kernel_size"],
                 batch_norm=settings["batch_norm"],
                 dtype=dtype)
    
    net = SkipNet(net, settings)
    net.to(device)   
    return net


def svdfnof_ver1(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['FIX:invariant-natural']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']
    basis_size = 5
    # Model
    settings = {"basis_size": basis_size,
                "skip": True,
                "modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [basis_size * 2 * 2,] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True}
    
    return fakesvdfno(settings, device, DTYPETORCH)


def svdfnof_ver2(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['FIX:invariant-natural']
    out_features = GeomData.PREDEFINED_OUTPUTS['FIX:invariant-natural']
    basis_size = 5
    # Model
    settings = {"basis_size": basis_size,
                "skip": True,
                "modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [basis_size * 2 * 2,] * 4, #(3,8) works, (2,8) worse. (8, 3) best so far
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True}
    
    return fakesvdfno(settings, device, DTYPETORCH)


# Non-invariant architectures
def fno_ver1(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['reduced-cartesian']
    out_features = GeomData.PREDEFINED_OUTPUTS['cartesian']
    
    # Model
    settings = {"modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [3*len(inp_features),] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "skip": True,
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True,
                "amsgrad": False}
    
    return egeofno(settings, device, DTYPETORCH)

def fno_ver2(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['full-cartesian']
    out_features = GeomData.PREDEFINED_OUTPUTS['cartesian']
    
    # Model
    settings = {"modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [4*len(inp_features),] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "skip": True,
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True,
                "amsgrad": False}
    
    return egeofno(settings, device, DTYPETORCH)

def fno_ver3(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['full-cartesian-norm']
    out_features = GeomData.PREDEFINED_OUTPUTS['cartesian-norm']
    
    # Model
    settings = {"modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [2*len(inp_features),] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "skip": True,
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True,
                "amsgrad": False}
    
    return egeofno(settings, device, DTYPETORCH)

def fno_ver4(device=DEVICE):
    # Features
    inp_features = GeomData.PREDEFINED_INPUTS['reduced-cartesian-norm']
    out_features = GeomData.PREDEFINED_OUTPUTS['cartesian-norm']
    
    # Model
    settings = {"modes": 40,
                "input_features": inp_features,
                "output_features": out_features,
                "weight_decay": 0,
                "layer_widths": [4*len(inp_features),] * 8, #(3,8) works, (2,8) worse. (8, 3) best so far
                "skip": True,
                "bias": True,
                "h1_weight": 1.0,
                "activation": F.gelu,
                "kernel_size": 1,
                "batch_norm": True,
                "amsgrad": False}
    
    return egeofno(settings, device, DTYPETORCH)




class Session:
    def __init__(self, 
                 net,
                 save_name="test",
                 weight_decay=0.000,
                 path_data="/home/emastr/phd/data/micro_geometries_boundcurv_repar_256_torch/data_1.torch",
                 save_dir="/home/emastr/phd/data/article_training/",
                 dash_dir="/home/emastr/phd/data/dashboard/",
                 device=DEVICE,
                 seed=None):
    
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        
        # Data
        self.dash_dir=dash_dir
        self.save_dir=save_dir
        self.device=device
        self.path_data = path_data
        self.data = self.get_data(net, path_data, device)   
        M = len(self.data)
        M_train, M_batch = int(0.8*M), 32
        self.train_data, self.test_data = random_split(self.data, [M_train, M-M_train])
        
        # Network
        self.net = net
        
        # Test data
        (self.X_test, self.Y_test)  = self.test_data[:]
        
        # Train data
        self.train_loader = DataLoader(self.train_data, batch_size=M_batch, shuffle=True)
        
        # Training settings
        self.optim = torch.optim.Adam(net.parameters(), weight_decay=weight_decay)
        
        # Loss function
        self.h1_weight = 0.0 if "h1_weight" not in net.settings() else net.settings()["h1_weight"]
        
        # Data setup
        self.trainloss = []
        self.testloss  = []
        self.logger = EventTracker()
        self.step_num = 0
        self.dashboard_setup()
        self.save_name = save_name
        
    @staticmethod
    def get_data(net, data_dir, device):
        # Data
        inp_features = net.settings()["input_features"]
        out_features = net.settings()["output_features"]
        return GeomData(data_dir, inp_features, out_features, random_roll=False, device=device, dtype=DTYPETORCH)       
        
        
    def train_nsteps(self, N):
        for i in range(N):
            self.train_step()
            self.step_num += 1
        self.save_state()
    
    
    def train_step(self):
        i = self.step_num
        self.logger.start_event("train")
        self.optim.zero_grad()
        (X_batch, Y_batch) = next(iter(self.train_loader))
        
        # Train on truncated net that expands as iterations progress
        loss = self.loss_fcn(self.net(X_batch), Y_batch, grad_weight=self.h1_weight, device=self.device)
        loss.backward()
        self.optim.step()
        self.logger.end_event("train")
        
        # Test 
        self.trainloss.append(loss.item() ** 0.5)
        self.testloss.append(self.loss_fcn(self.net(self.X_test), self.Y_test, grad_weight=0.0, device=self.device).item() ** 0.5)

        self.net.eval()
        # Print minor state info
        if i % 500 == 0:
            self.dashboard_update()
            iter_speed = int(1/self.logger["train"].time_mean)
            print(f"Step {i}. Train loss = {self.trainloss[-1]:.2e}, test loss = {self.testloss[-1]:.2e}, {iter_speed} iter/s, shape={X_batch.shape[-1]}", end="\r")

        # major state info
        if i % 1000 == 0:
            self.save_state()
            
        self.net.train()
        
    
    def dashboard_setup(self):
        fig = plt.figure(figsize=(10,10))
        self.ax1 = fig.add_subplot(2,3,1)
        self.ax2 = fig.add_subplot(2,3,2)
        self.ax3 = fig.add_subplot(2,3,3)
        self.ax4 = fig.add_subplot(2,3,4)
        self.ax5 = fig.add_subplot(2,3,5)
        self.ax6 = fig.add_subplot(2,3,6)
        
        self.dashboard = DashBoard(path=self.dash_dir) 
        self.dashboard.add_figure(fig)
        
    
    def dashboard_update(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()
        
        # Plot training loss
        self.dashboard.tracked_figures[0].fig.suptitle(self.save_name)
        self.ax1.plot(self.trainloss, label="train")
        self.ax1.plot(self.testloss, label="test")
        self.ax1.legend()
        self.ax1.set_yscale("log")
        
        # Plot example
        (x, y) = self.test_data[13]
        xc, yc = x[None, :, :], y[None, :, :]
        f = self.net(xc).squeeze().detach().cpu()
        x = xc.squeeze().detach().cpu()
        y = yc.squeeze().detach().cpu()
        t = np.linspace(0, 2*np.pi, x.shape[-1])
        
        # Data 1
        plot_periodic2(self.ax2, t, x[-1, :], label="d input n", alpha=0.5)
        plot_periodic2(self.ax2, t, y[-1, :], label="d true n")
        plot_periodic2(self.ax2, t, f[-1, :], label="d output n")
        self.ax2.legend()
        
        # Data 2
        plot_periodic2(self.ax3, t, x[-2, :], label="d input t", alpha=0.5)
        plot_periodic2(self.ax3, t, y[-2, :], label="d true t")
        plot_periodic2(self.ax3, t, f[-2, :], label="d output t")
        self.ax3.legend()
        
        # Data 2
        plot_periodic2(self.ax5, t, x[-3, :], label="input n", alpha=0.5)
        plot_periodic2(self.ax5, t, y[-3, :], label="true n")
        plot_periodic2(self.ax5, t, f[-3, :], label="output n")
        self.ax5.legend()
        
        # Data 2
        plot_periodic2(self.ax6, t, x[-4, :], label="input t", alpha=0.5)
        plot_periodic2(self.ax6, t, y[-4, :], label="true t")
        plot_periodic2(self.ax6, t, f[-4, :], label="output t")
        self.ax6.legend()
        
        # Matrix
        if hasattr(self.net.net, "plot"):
            self.net.net.plot(self.ax4, xc)
        else:
            pass
        
        # Boundary
        self.dashboard.update_all()
    
    
    def save_state(self):    
        torch.save({"state dict" : self.net.state_dict(), 
                    "settings"   : self.net.settings(),
                    "trainloss"  : self.trainloss,
                    "testloss"   : self.testloss}, 
                    f"{self.save_dir}{self.save_name}_{self.step_num}.Torch") 
        
    # Loss function
    @staticmethod
    def mse_normalized(x, y):
        y_norm = torch.linalg.norm(y, dim=-1)[:,:,None] / np.sqrt(y.shape[-1])
        return torch.mean((x - y)**2 / y_norm**2)
        
    @staticmethod
    def deriv(x, device=DEVICE):
        x_fft = torch.fft.rfft(x, dim=-1)
        freq = 1j * 2 * np.pi * torch.tensor(np.arange(x.shape[-1]//2 + 1))[None, None, :].to(device)
        x_deriv = torch.fft.irfft(x_fft * freq).to(CTYPETORCH).real
        return x_deriv

    @staticmethod
    def loss_fcn(x, y, grad_weight=0.0, **kwargs):
        x_der = Session.deriv(x, **kwargs)
        y_der = Session.deriv(y, **kwargs)
        return Session.mse_normalized(x, y) + grad_weight * Session.mse_normalized(x_der, y_der)