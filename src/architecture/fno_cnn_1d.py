"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from architecture.unet_1d import *

import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)
np.random.seed(0)


################################################################################################################
#                                              1d fourier layer                                                #
################################################################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, dtype=torch.cdouble):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=dtype))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    
###############################################################################################################
#                                    FNO 1D Using spectral convolution module                                 #
###############################################################################################################
class FNO1d(nn.Module):
    def __init__(self, modes, in_channels, out_channels, layer_widths=None, verbose=False):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        
        self.verbose = verbose
        
        if layer_widths is None:
            self.n_layers = 4
            self.layer_widths = [2 * in_channels,] * (self.n_layers+1)
            self.print_msg(f"Employing default layer structure, {self.layer_widths}")
        else:
            self.n_layers = len(layer_widths)-1
            self.layer_widths = layer_widths
            
        
        self.inp = nn.Linear(in_channels, self.layer_widths[0])        
        
        # Convolution layers
        self.conv_list = nn.ModuleList([SpectralConv1d(self.layer_widths[i], self.layer_widths[i+1], modes) for i in range(self.n_layers)])
        
        # Linear layers
        self.lin_list = nn.ModuleList([nn.Conv1d(self.layer_widths[i], self.layer_widths[i+1], 1) for i in range(self.n_layers)])
        
        # Local convolutions
#        self.loc_list = nn.ModuleList([nn.Conv1d(in_channels=self.layer_widths[i+1], 
#                                                 out_channels=self.layer_widths[i+1], 
#                                                 kernel_size=3, padding=1, padding_mode="circular", stride=1) 
#                                       for i in range(self.n_layers)])
        
        self.loc_list = nn.ModuleList([nn.Sequential(
                                        nn.Conv1d(in_channels=self.layer_widths[i+1], 
                                         out_channels=self.layer_widths[i+1], 
                                         kernel_size=3, padding=1, padding_mode="circular", stride=1),
                                        nn.GELU(),
                                        nn.Conv1d(in_channels=self.layer_widths[i+1], 
                                         out_channels=self.layer_widths[i+1], 
                                         kernel_size=3, padding=1, padding_mode="circular", stride=1)
                                        )
                                       for i in range(self.n_layers)])

        
        
#        self.loc_list = nn.ModuleList([Unet(depth=3,
#                                            width=2,
#                                            in_channels=self.layer_widths[i+1], 
#                                            min_out_channels=self.layer_widths[i+1], 
#                                            out_channels=self.layer_widths[i+1]) 
#                                       for i in range(self.n_layers)])
        
        self.out = nn.Linear(self.layer_widths[-1], out_channels)

        
    def print_msg(self, msg):
        if self.verbose:
            print(msg)
        pass

    def forward(self, x):
        
        # Project to FNO width
        x = self.inp(x.permute(0,2,1)).permute(0,2,1)
        
        # Evaluate FNO
        for conv_op, lin_op, loc_op in zip(self.conv_list, self.lin_list, self.loc_list):
            y = conv_op(x) + lin_op(x)
            x = F.gelu(y + loc_op(y))
        
        # Project to out_channels width
        return self.out(x.permute(0,2,1)).permute(0,2,1)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
