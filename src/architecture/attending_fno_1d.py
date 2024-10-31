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
import matplotlib.pyplot as plt

def identity(x):    
    return x

def potential(x):
    abs_x = torch.abs(x)
    return x / (1 + abs_x**2)
################################################################################################################
#                                              1d fourier layer                                                #
################################################################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, max_mode, dtype=None):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Input: (batches, in_channels, dimension) - pytorch Tensor x

        output: (batches, out_channels, dimension) - pytorch Tensor y,
                determined by the relation:
                
                y = iFFT(  W  @  trunc(  FFT( x ) ) )
                
                iFFT, FFT are the Fourier Transform and its inverse, respectively.
                trunc truncates the FFT of x to the lowest modes, determined by the max_mode variable.
                W is a matrix with dimensions (in_channels, out_channels, max_mode)
                @ is a matrix operation: (in_channels, max_mode) -> (out_channels, max_mode)
                
                
                In function space (imagine input x as a function), the operation corresponds to
                
                y = K * x,
                
                where K is a (out_channel, in_channel) matrix of convolutional kernels,
                which operate on the channels i=1,2,3 ... n of x as follows
                
                y_i = k1i * x1  +  k2i * x2  +  k3i * x3  +  ...  +  kni * xn
                
                where * is the convolution operator.                
        """
        #WARNING: Fixed bug to allow change in dimension, line 90 to in_channels, not out_channels.
        if (dtype == "float") or (dtype is torch.float):
            self.dtype = torch.float
            self.cdtype = torch.cfloat
        else:
            self.dtype = torch.double
            self.cdtype = torch.cdouble
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = max_mode  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        
        ## TODO: Make this a parameter, and make it trainable. (only relevant for shift-variant problems)
        #if spectral_bias:
            #self.bias = Parameter(self.scale * torch.rand(out_channels, dtype=self.dtype))
        #else:
            #self.bias = torch.zeros(out_channels, dtype=self.dtype)
            #self.bias = Parameter(torch.zeros(out_channels, dtype=self.dtype))
            #self.bias.requires_grad = False
        

        self.scale = (1 / (in_channels*out_channels))
        self.weights = Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=self.cdtype) +\
                                 1j*self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=self.cdtype))
        
        self.attention = Attention(in_channels, dtype=dtype)

    # Complex multiplication
    def compl_mul1d(self, inp, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", inp, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)        
        # Multiply relevant Fourier modes
        xmodes = x.size(-1)
        min_modes = min(xmodes, self.modes)
        x_ft = SpectralConv1d.truncfft(x, min_modes)
        out_ft = torch.zeros(batchsize, self.out_channels, self.modes,  device=x.device, dtype=self.cdtype)
        out_ft = self.compl_mul1d(x_ft, self.weights)
        #out_ft = self.attention(x_ft)

        #Return to physical space
        x = SpectralConv1d.itruncfft(out_ft, n=x.size(-1), modes=self.modes)
        return x
    
    @staticmethod
    def truncfft(x, modes):
        return torch.roll(torch.fft.fft(x), modes//2, dims=-1)[:, :, :modes]

    @staticmethod
    def itruncfft(xfft, n, modes):
        out = torch.zeros(xfft.size(0), xfft.size(1), n, device=xfft.device, dtype=xfft.dtype)
        out[:, :, :modes] = xfft
        return torch.fft.ifft(torch.roll(out, -(modes//2), dims=-1))

class Attention(nn.Module):
    def __init__(self, channels, dtype=None, version=3):
        super(Attention, self).__init__()

        """
        Channel wise 1D attention
        """
        #WARNING: Fixed bug to allow change in dimension, line 90 to in_channels, not out_channels.
        if (dtype == "float") or (dtype is torch.float):
            self.dtype = torch.float
            self.cdtype = torch.cfloat
        else:
            self.dtype = torch.double
            self.cdtype = torch.cdouble
            
        self.channels = channels
        self.version = version
        ## TODO: Make this a parameter, and make it trainable. (only relevant for shift-variant problems)
        #if spectral_bias:
            #self.bias = Parameter(self.scale * torch.rand(out_channels, dtype=self.dtype))
        #else:
            #self.bias = torch.zeros(out_channels, dtype=self.dtype)
            #self.bias = Parameter(torch.zeros(out_channels, dtype=self.dtype))
            #self.bias.requires_grad = False
        

        self.scale = (1 / channels / channels)
        self.KQ = Parameter(self.scale * torch.rand(channels, channels, dtype=self.cdtype))
        self.V = Parameter(self.scale * torch.rand(channels, channels, dtype=self.cdtype))
        self.M = Parameter(self.scale * torch.rand(channels, channels, dtype=self.cdtype))

    # Complex multiplication
    def compl_mul1d(self, inp, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", inp, weights)

    @staticmethod
    def activation(x, func=None, eps=0.00001):
        #func = lambda x: 1 / (1 + torch.exp(-x))
        #func = lambda x: F.gelu(x) ** 0.5
        #func = F.gelu
        #return func(torch.abs(x))*x/(eps + torch.abs(x))
        abs_x = torch.abs(x)
        return torch.max(abs_x, torch.ones_like(abs_x))*x/(eps + abs_x)
    
    def softmax(self, x):
        x = torch.exp(x)
        return x / x.sum(dim=-2, keepdim=True)

    def forward(self, x):
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        if self.version == 1:
            xQKx = torch.einsum("ij, bjx -> bix", self.KQ, x)
            xQKx = torch.einsum("bix, bjx -> bijx", torch.conj(x), xQKx) #/ x.size(-2) ** 0.5
            xQKx = xQKx + self.M[None, :, :, None]
            #xQKx = self.activation(xQKx)
            xQKx = self.softmax(xQKx)
            out = torch.einsum("ij, bjx -> bix", self.V, x)
            return  torch.einsum("bijx, bjx -> bix", xQKx, out)
        elif self.version == 2:
            return torch.einsum("ij, bjx -> bix", self.M, x)
        else:
            out = torch.einsum("ij, bjx -> bix", self.V, x)
            xQKx = torch.einsum("ij, bjx -> bix", self.KQ, x)
            xQKx = torch.einsum("bix, bix -> bx", xQKx, out)
            xQKx = torch.conj(x)*xQKx[:, None, :]/(0.00001 + torch.abs(x))# / x.size(-2) ** 0.5
            return xQKx + torch.einsum("ij, bjx -> bix", self.M, out)

    
###############################################################################################################
#                                    FNO 1D Using spectral convolution module                                 #
###############################################################################################################
class AFNO1d(nn.Module):
    def __init__(self, device, dtype=None, attention_version=3):
        super(AFNO1d, self).__init__()

        """
        The overall network. 
        1. Lift the input to the desired channel dimension by a channel wise linear operation (in_channels, layer_widths[0]).
        2. multiple layers of the integral operators u' = (W + K)(u),
           number of layers determined by the length of the vector layer_widths.
        3. Project from the channel space to the output space by another linear operation.
        
        
        input: A batch of vector valued functions x, represented as a discretised Tensor.
                
                functions:  a batch of b functions x: [0,1] -> R^d ,  
                representation: Tensor             x in  R^(b * d * N), 

                where b is number of batches, d=in_channels, N=discretisation points.
                that is, x_ijk is channel j of function i, evaluated at a point s_k.

        output: a batch of vector valued functions y, represented same as input, but with d=out_channels.
        
        parameters: layer_widths regulates the number of channels in the inner layers
                    modes regulates the frequency cutoff in the SpectralConv1 operations.
        
        Function:
                depends on input as:
                
                yi = wi1 * zK1  +  wi2 * zK2  +  wi3 * zK3  +  ...  +  wiL * zKL,  where wij are weights, i=1,2,..., out_channels,
                                                                                   and where j = 1,2,..., layer_widths[-1].
                
                zK is the final layer of the inner network (K = len(layer_widths) is the number of layers),
                where the layers are updated according to 
                
                zk+1 = gelu(  W  @  zk     +    SpectralConv1d(   zk   )    +  bias),
                
                with zk a batched set of functions with dimension (b,  d,  N), (b functions [0,1] -> R^layer_widths[k])
                W the same type of linear operation as used in SpectralConv1d
                bias is a discretised function bias: [0,1] -> R^layer_widths[k+1].
                gelu is the gelu activation function, operating element wise.
                
                The updates can be written in the continuum formulation like:
                
                z_{(k+1),i}(s) = gelu(  w_{ki1} * z_{k1}(s)  +  w_{ki2} * z_{k2}(s)  +  ...  +  w_{kin} * z_{kn}(s)        (Linear mixing)
                                      + (r_{ki1} x z_{k1})(s)  +  (r_{ki2} x z_{k2})(s)  +  ...  +  (r_{kin} x z_{kn})(s)  (Convolution)
                                       +  bias_i(s)  )                                                                     (bias)
                
                
                where n = layer_widths[k], and where i = 1,2, ... layer_widths[k+1].              
                
        """
        
        
        if (dtype == "float") or (dtype is torch.float):
            self.dtype = torch.float
            self.cdtype = torch.cfloat
        else:
            self.dtype = torch.double
            self.cdtype = torch.cdouble    
            
        self.settings_ = self.settings()
        self.attention_version = attention_version        
        in_channels = len(self.settings_["input_features"]) // 2
        out_channels = len(self.settings_["output_features"]) // 2
        self.n_layers = len(self.settings_["layer_widths"]) - 1 
        self.modes = self.settings_["modes"]
        self.layer_widths = self.settings_["layer_widths"]
        
        # input layer
        self.inp = nn.Parameter(torch.rand(self.layer_widths[0], in_channels, dtype=self.cdtype))     
        self.attention_list = nn.ModuleList([Attention(self.layer_widths[i], \
                              dtype=dtype, version=attention_version) for i in range(self.n_layers)])
        
        # Convolution layers
        self.conv_list = nn.ModuleList([SpectralConv1d(self.layer_widths[i], self.layer_widths[i+1], 
                                                       self.modes, dtype=self.dtype) for i in range(self.n_layers)])
        

        self.out = nn.Parameter(torch.rand(out_channels, self.layer_widths[-1], dtype=self.cdtype))     

        class NetPlotter:
            def __init__(self, net):
                self.net = net
                pass
            
            def plot(self, ax, xc):
                xc = AFNO1d.to_compl(xc)
                xc = torch.einsum("ij, bjx -> bix", self.net.inp, xc)
                plot_layer = torch.randint(1, self.net.n_layers, (1,)).item()
                for conv, att in zip(self.net.conv_list[:plot_layer] , self.net.attention_list[:plot_layer]):
                    xc = Attention.activation(conv(xc) + att(xc))
                att1 = self.net.attention_list[plot_layer]
                conv1 = self.net.conv_list[plot_layer]
                Vx = torch.einsum("ij, bjx -> bix", att1.V, xc)
                xQKx = torch.einsum("ij, bjx -> bix", att1.KQ, xc)
                xQKx = torch.einsum("bix, bjx -> bijx", torch.conj(xc), xQKx)/(0.00001 + torch.abs(xc))
                xQKx = torch.abs(torch.einsum("bijx, bjx -> bix", xQKx, Vx).squeeze())
                Vx = torch.abs(torch.einsum("ij, bjx -> bix", att1.M, Vx).squeeze())
                convx = torch.abs(conv1(xc)).squeeze()
                
                xQKxVx = torch.abs(torch.cat([xQKx, Vx, convx], dim=0))
                ax.imshow(xQKxVx.cpu().detach().numpy(), cmap='hot', interpolation='nearest', aspect='auto')
                ax.set_title("Attention layer %d, min (%d, %d, %d), max (%d,%d,%d)"\
                    %(plot_layer, torch.log10(xQKx).min().item(), torch.log10(Vx).min().item(), torch.log10(convx).min().item(), \
                        torch.log10(xQKx).max().item(), torch.log10(Vx).max().item(), torch.log10(convx).max().item()))
        self.net = NetPlotter(self)
        self.to(device)
        self.device = device
                
    def print_msg(self, msg):
        if self.verbose:
            print(msg)
        pass

    def forward(self, x):
        # Project to FNO width
        #print(self.inp.weight.dtype)
        
        # Convert to complex
        x = x[:, ::2] + 1j*x[:, 1::2]
        x = torch.einsum("ij, bjx -> bix", self.inp, x)
        
        #print(x.isnan().any(), x.norm().item())
        # Evaluate FNO
        for conv_op, attention in zip(self.conv_list, self.attention_list):
            x = Attention.activation(conv_op(x) + attention(x))
            #print(x.isnan().any(),x.norm().item())
        # Project to out_channels width
        x = torch.einsum("ij, bjx -> bix", self.out, x)
        out = torch.zeros(x.size(0), 2*x.size(1), x.size(-1), device=x.device, dtype=self.dtype)
        for i in range(x.size(1)):
            out[:, 2*i] = x[:, i].real
            out[:, 2*i+1] = x[:, i].imag
        return out

 
    @staticmethod   
    def from_compl(x):
        out = torch.zeros(x.size(0), 2*x.size(1), x.size(-1), device=x.device, dtype=x.dtype)
        for i in range(x.size(1)):
            out[:, 2*i] = x[:, i].real
            out[:, 2*i+1] = x[:, i].imag
        return out

    @staticmethod
    def to_compl(x):
        return x[:, ::2] + 1j*x[:, 1::2]

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=self.cdtype)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
    
    def settings(self):
        
        inp_features = ['cx_norm', 'cy_norm', 'vx','vy', 'dvx_norm', 'dvy_norm']
        out_features = ['rx', 'ry', 'drx_norm', 'dry_norm']

        # Model
        settings = {"input_features": inp_features,\
                    "output_features": out_features,\
                    "weight_decay": 0,\
                    "modes": 41,\
                    "layer_widths": 5*[15,],\
                    "h1_weight": 0.01,
                    "batch_norm": True,
                    "amsgrad": False}
        
        return settings
        
