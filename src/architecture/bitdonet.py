"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).

Original author Zongyi Li, Modified to general layer widths by Emanuel Ström.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class FCNN(nn.Module):
    def __init__(self, layer_widths, activation=F.gelu, bias=True, dtype=torch.float):
        super(FCNN, self).__init__()
        self.layer_widths = layer_widths
        self.activation = activation
        self.bias = bias
        self.n_layers = len(layer_widths) - 1
        self.linears = nn.ModuleList([nn.Linear(layer_widths[i], layer_widths[i+1], bias=bias, dtype=dtype) for i in range(self.n_layers)])
        
    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < self.n_layers - 1:
                x = self.activation(x)
        return x

    
###############################################################################################################
#                                    FNO 1D Using spectral convolution module                                 #
###############################################################################################################
class BITDOnet(nn.Module):
    def __init__(self, modes, dtype="float", device="cpu"):
        super(BITDOnet, self).__init__()

        """       
                
        """
        if (dtype == "float") or (dtype is torch.float):
            self.dtype = torch.float
            self.cdtype = torch.cfloat
        else:
            self.dtype = torch.double
            self.cdtype = torch.cdouble
        
        D = 300
        self.modes = modes
        self.layer_widths = [[modes*6, D, D, D, D, modes*4], \
                             [modes*2, D, D, D, D, modes*4], \
                             [modes*6, D, D, D, D, modes*4]]
        
        self.V = FCNN(self.layer_widths[0], dtype=self.dtype)
        self.m = FCNN(self.layer_widths[1], dtype=self.dtype)
        self.U = FCNN(self.layer_widths[2], dtype=self.dtype)
            
        self.to(dtype=self.dtype)
        self.to(device)
        self.net = None
        
    def print_msg(self, msg):
        if self.verbose:
            print(msg)
        pass

    def forward(self, x):
        """Data shape should be (batchsize, 6, 256)."""
        # Project to FNO width
        #print(self.inp.weight.dtype)
        
        # take fft
        x_ft = torch.fft.rfft(x).to(dtype=self.cdtype)
        
        min_modes = (self.modes-1)//2
        
        # Truncate modes
        x_ft = x_ft[:, :, :min_modes+1]
        
        # Concatenate real and imaginary parts
        v = torch.cat((x_ft.real, x_ft.imag[:, :, 1:]), dim=-1)
        
        # Separate out the frst two channels
        g_ft = v[:, :2].view(x.size(0), -1) # Gamma FFT
        
        # Apply network to all data (collapse last two dimensions)
        v = self.V(v.view(x.size(0), -1)) * self.m(g_ft)
        v = self.U(torch.cat((g_ft, v), dim=-1)).view(x.size(0), 4, self.modes)
        
        #print(v.shape)
        out_ft = torch.zeros(x.size(0), 4, x.size(-1)//2+1,  device=x.device, dtype=self.cdtype)
        out_ft[:, :, 1:min_modes+1] = v[:, :, 1:min_modes+1] + 1j*v[:, :, min_modes+1:]
        out_ft[:, :, 0] = v[:, :, 0]

        #Return to physical space
        return torch.fft.irfft(out_ft, n=x.size(-1))

    def settings(self):
        
        inp_features = ['x', 'y', 'vx', 'vy', 'dvx', 'dvy']
        out_features = ['rx', 'ry', 'drx', 'dry']

        # Model
        settings = {"modes": self.modes,\
                    "input_features": inp_features,\
                    "output_features": out_features,\
                    "weight_decay": 0,\
                    "layer_widths": self.layer_widths,\
                    "h1_weight": 0.1,
                    "batch_norm": True,
                    "amsgrad": False}
        
        return settings