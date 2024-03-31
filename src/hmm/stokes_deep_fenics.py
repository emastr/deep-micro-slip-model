import torch
import torch.nn as nn
import numpy as np
from util.plot_tools import *
from hmm.hmm import Solver
from hmm.stokes_fenics import StokesMicProb as StokesMicProbFenics
from architecture.fno_1d import FNO1d
import torch.nn.functional as F
from boundary_solvers.geometry_torch import GeomData, concat_dict_entries, subdict, unpack, geometry_to_net_input, projection



def get_net(path, num_pts, device, dtype):
    save = torch.load(path)
    settings = save["settings"]
    bias = settings["bias"] if "bias" in settings else True
    activation = settings["activation"] if "activation" in settings else F.gelu
    kernel_size = settings["kernel_size"] if "kernel_size" in settings else 1
    batch_norm = settings["batch_norm"] if "batch_norm" in settings else False
    
    net = FNO1d(modes=settings["modes"], 
                in_channels=len(settings["input_features"]), 
                out_channels=len(settings["output_features"]), 
                layer_widths=settings["layer_widths"],
                bias = bias,
                activation=activation,
                kernel_size=kernel_size,
                batch_norm=batch_norm,
                dtype=dtype)
    
    net.to(device)
    if settings["skip"]:
        class SkipNet(nn.Module):
            def __init__(self, net):
                super(SkipNet, self).__init__()
                self.net = net
            
            def forward(self, x):
                return self.net(x) + x[:, -4:, :]
        net = SkipNet(net)
    net.load_state_dict(save["state dict"])
    return net, {"num_pts": num_pts, "input_features": settings["input_features"], "device": device, "dtype": torch.float if dtype=="float" else torch.double}


class DeepMicroSolverFenics(Solver):
    """Solve the micro problem at a specific position. Precompute."""
    def __init__(self, problem, net, net_settings, logger=None, **kwargs):
        _2np = lambda x: x.cpu().detach().numpy()
        
        geom = problem.geom
        
        ## Compute deep
        self.geom = geom
        self.net = net
        self.net_settings = net_settings
        self.logger = logger

        if self.logger is not None:
            self.logger.start_event("deep_micro_precompute")
        
        # Add features
        data = unpack(lambda x: torch.from_numpy(x)[None, :])(geometry_to_net_input(geom, net_settings['num_pts'], output=False))
        data = GeomData.tfm_inp(data)
        self.data = data
        
        # Run through network
        X = concat_dict_entries(subdict(data, net_settings['input_features'])).to(net_settings['device']).to(net_settings['dtype'])
        self.t = _2np(data["t"])[0]
        tx, ty = _2np(data['tx'])[0], _2np(data['ty'])[0]
        dv_norm = (np.linalg.norm(_2np(data['dvt'])[0])**2 + np.linalg.norm(_2np(data['dvn'])[0])**2)**0.5 /  len(self.t) ** 0.5
        
        
        if self.logger is not None:
            self.logger.start_event("deep_micro_net_eval")
        Y = _2np(net(X))[0]
        if self.logger is not None:
            self.logger.end_event("deep_micro_net_eval")
        
        
        # Project to cartesian
        rx, ry = projection(Y[0], Y[1], tx, ty, inv=True)        
        drx, dry = projection(Y[2]*dv_norm, Y[3]*dv_norm, tx, ty, inv=True)
        
        # Save as complex
        self.r = rx + 1j * ry
        self.dr = drx + 1j * dry

        
        self.avg = lambda cond: np.sum((self.r * np.conjugate(1j * cond(self.t))).real)
        self.davg = lambda cond: np.sum((self.dr * np.conjugate(1j * cond(self.t))).real)
        

        if self.logger is not None:
            self.logger.end_event("deep_micro_precompute")
    
    def can_solve(self, problem):
        return isinstance(problem, StokesMicProbFenics)
        
    def solve(self, problem: StokesMicProbFenics):
        # Log Solve time
        if self.logger is not None:
            self.logger.start_event("deep_micro_solve")
    
        out = MicroData(problem.center, -self.avg(problem.condition) / self.davg(problem.condition))

        # End log
        if self.logger is not None:
            self.logger.end_event("deep_micro_solve")

        return out
    
    def plot(self):
         # Compute true
        self.r_true, self.avg_true = self.geom.precompute_line_avg(derivative=0)
        self.dr_true, self.davg_true = self.geom.precompute_line_avg(derivative=1)
        
        t1, _ = self.geom.grid.get_grid_and_weights()
        
        plt.figure()
        plt.plot(self.data['x'][0], self.data['y'][0])
        
        
        plt.figure()
        plt.plot(self.t, self.r.real)
        plt.plot(t1, self.r_true.real)
        
        plt.plot(self.t, self.r.imag)
        plt.plot(t1, self.r_true.imag)
        
        plt.figure()
        plt.plot(self.t, self.dr.real)
        plt.plot(t1, self.dr_true.real)
        
        plt.plot(self.t, self.dr.imag)
        plt.plot(t1, self.dr_true.imag)
 

   
class MicroData():
            def __init__(self, x, a):
                self.alpha = a
                self.x = x
            