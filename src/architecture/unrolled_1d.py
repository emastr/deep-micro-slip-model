import torch as pt
from architecture.unet_1d import *
import torch.nn as nn
from boundary_solvers.geometry_torch import unpack

DEFAULT_SETTINGS = {"in_channels":2, 
                    "min_out_channels":4,
                    "out_channels":2,
                    "depth":3, 
                    "batch_norm":2, 
                    "width":2,
                    "kernel_size":3,
                    "padding":1,
                    "padding_mode":'circular',
                    "activation":nn.ReLU}

class Unrolled(nn.Module):
    
    def __init__(self, n_iter, net_factory, net_args=None, device="cpu", debug=False):
        super(Unrolled, self).__init__()
        
        self.debug = debug
        self.device = device
        self.n_iter = n_iter
        if net_args is None:
            self.nets = nn.ModuleList([net_factory() for i in range(n_iter)])
        else:
            self.nets = nn.ModuleList([net_factory(net_args[i]) for i in range(n_iter)])
        self.nets.to(device)
        
        
    def printmsg(self, msg):
        if self.debug:
            print(msg)
        else:
            pass
        
            
    def forward(self, x: dict, op=None, subset=None, truncate=None) -> torch.Tensor:
        """
        param x: tuple (x, y) with x = (N,4,K) and y = (N, C, K) 
        where x = [z, dz, ddz, w, p0] (boundary data + initial data)
        and y = input data to feed the network together with the gradient of the residual.
        """
        # If input is 5-dimensional, treat as raw problem data
        if op is None:
            self.printmsg("Making operator...")
            op = op_factory(x)
        
        if truncate is None:
            truncate = self.n_iter
        
        # Extract riesz vector components to use as starting point
        
        self.printmsg("concatting..")
        v = concat_dict_entries(subdict(x, {'vx', 'vy'}))
        
        self.printmsg("Cloning...")
        y = v.clone()
        
        ##### CURRENTLY NOT USED
        # Extract subset to input alongside Kx - v0
        if subset is not None:
            self.printmsg("Subsetting")
            x = subdict(x, subset)

        ##### TODO: MAKE USE OF THESE QUANTITIES?
        self.printmsg("Entering fwd loop")
        for i in range(truncate):
            self.printmsg(f"Shape: {y.shape}")
            y = y + self.nets[i](op(y)-v)
                
        return y
    
    
def factory():
    return Unet(**settings)
    #return nn.Sequential(nn.Conv1d(2,2,3,padding=1, padding_mode='circular'),\
    #                     nn.ReLU(),\
    #                     nn.Conv1d(2,2,3,padding=1, padding_mode='circular'))

@unpack
def to_dtype(x, dtype):
    return x.to(dtype)