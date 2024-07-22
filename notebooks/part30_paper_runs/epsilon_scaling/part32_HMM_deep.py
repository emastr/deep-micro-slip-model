import numpy as np
import matplotlib.pyplot as plt
from util.logger import EventTracker
from util.basis_scaled import *
from util.plot_tools import *
from boundary_solvers.gauss_grid_2d import StokesDirichletProblem
from scipy.io import loadmat
import matplotlib.pyplot as plt
from util.interp import PiecewiseInterp2D
from hmm.stokes import *
from itertools import product
from hmm.stokes_deep import DeepMicroSolver, get_net
from boundary_solvers.geometry_torch import interp_periodic
import matplotlib
import torch.nn as nn
import matplotlib as mpl


#matplotlib.rcParams['text.usetex'] = True
#pretty_pyplot_layout()

np.random.seed(0)
net_dir = "/mnt/data0/emastr/article_training_hugedata/" #lowline/"
figures_dir = "/home/emastr/deep-micro-slip-model/data/figures/"
run_dir = "/home/emastr/deep-micro-slip-model/data/reference_2/"

#save_dir = "/home/emastr/deep-micro-slip-model/data/hmm_coupling_error/"
save_dir = "/mnt/data0/emastr/hmm_coupling_paper/"
deep_save_dir = "/mnt/data0/emastr/hmm_coupling_deep_paper"

from dataclasses import dataclass
from multiprocessing import Pool
from architecture.session import egeofno_ver1, egeofno_ver2, egeofno_ver3, egeofno_ver4,\
                                 svdfno_ver1, svdfno_ver2, svdfno_ver3, svdfno_ver4,\
                                 fno_ver1, fno_ver2, fno_ver3, fno_ver4
import torch

device = "cpu"
nets = {}
#nets.update({f"fno_svd_ver{i+1}": fno for i,fno in enumerate([svdfno_ver1, svdfno_ver2, svdfno_ver3, svdfno_ver4])})
nets.update({f"fno_ver{i+1}": fno for i, fno in enumerate([egeofno_ver1, egeofno_ver2, egeofno_ver3, egeofno_ver4])})
nets.update({f"fno_vanilla_ver{i+1}": fno for i, fno in enumerate([fno_ver1, fno_ver2, fno_ver3, fno_ver4])})


print(list(nets.keys()))
@dataclass
class HyperParams:
    nMic: int
    xDim: int
    yDim: int
    width: float
    height: float
    tol: float
    max_iter: int
    num_pts: int
    xDim_reduce: int = 5
    yDim_reduce: int = 5
    line_pos: float = 0.0
    net_path: str = None
    net_type: str = None
    
    
class Setup():
    def __init__(self, data: StokesData, param: HyperParams):#, net: nn.Module, net_settings: dict):
        """Create a new HMM setup with the given parameters."""
        self.params = param
        self.data = data
        self.data0 = data.copy()
        self.data.dom = [self.data.dom[0], [param.line_pos*param.height+self.data.dom[1][0], self.data.dom[1][1]]]
        
        # Net
        print(f"type {nets[param.net_type]}, loading {param.net_path}")
        net = nets[param.net_type](device=device)
        net_data = torch.load(param.net_path, map_location=torch.device(device))
        net.load_state_dict(net_data["state dict"])
        net_settings =  {"num_pts": param.num_pts, "input_features": net_data["settings"]["input_features"], "output_features": net_data["settings"]["output_features"], "device": device, "dtype": torch.float}
        
        # Naive
        self.macro0 = StokesMacProb(self.data0, lambda x,a: fft_interp(x, a, dom=self.data0.dom[0]))
        self.macro_solver0 = MacroSolver(param.xDim, param.yDim, tol=param.tol)
        
        # Macro
        self.macro = StokesMacProb(self.data, lambda x,a: fft_interp(x, a, dom=self.data.dom[0]))
        self.macro_solver = MacroSolver(param.xDim, param.yDim, tol=param.tol)
        
        # Micro problems
        self.xPos = np.linspace(self.data.dom[0][0], self.data.dom[0][1], param.nMic+1)[1:] - param.width/2 - (self.data.dom[0][1]-self.data.dom[0][0]) / param.nMic
        self.micros = [StokesMicProb(self.data, x, param.width, param.height, param.line_pos, 8, xDim_reduce=param.xDim, yDim_reduce=param.yDim, n_refine=1) for x in self.xPos]
        self.micro_solvers = [MicroSolver(m, tol=param.tol) for m in self.micros]
        self.deep_micro_solvers = [DeepMicroSolver(m, net, net_settings, tol=param.tol) for m in self.micros]
        
        # Convergence checker
        self.conv_checker = ConvergenceChecker(self.macro, self.micros, tol=param.tol)
        
        # HMM problem
        self.hmm_problem = StokesHMMProblem(self.macro, self.micros, self.data, convergence_checker=self.conv_checker)
        self.hmm_solver = IterativeHMMSolver(self.macro_solver, self.micro_solvers)
        self.hmm_deep_solver = IterativeHMMSolver(self.macro_solver, self.deep_micro_solvers)
        self.logger = EventTracker()
        
        
    def solve(self, **kwargs):
        # Naive
        print("Solving naive")
        self.logger.start_event("naive")
        macro_naive = self.macro_solver0.solve(self.macro0)
        self.logger.end_event("naive")
        
        # Deep
        print("Solving deep")
        self.logger.start_event("deep")
        deep_macro_guess = self.hmm_deep_solver.macro_solver.solve(self.hmm_problem.macro)
        (deep_macro_sol, deep_micro_sols) = self.hmm_deep_solver.solve(self.hmm_problem, macro_guess=deep_macro_guess, verbose=True, maxiter=self.params.max_iter, tol=self.params.tol)
        self.logger.end_event("deep")
        
        # Classical
        print("Solving Classical")
        self.logger.start_event("classical")
        macro_guess = self.hmm_solver.macro_solver.solve(self.hmm_problem.macro)
        (macro_sol, micro_sols) = self.hmm_solver.solve(self.hmm_problem, macro_guess=macro_guess, verbose=True, maxiter=self.params.max_iter, tol=self.params.tol)
        self.logger.end_event("classical")
        
        #(deep_macro_sol, deep_micro_sols) = self.hmm_deep_solver.solve(self.hmm_problem, macro_guess=macro_guess, verbose=True, maxiter=self.params.max_iter, tol=self.params.tol)
        
        return macro_naive, macro_sol, micro_sols, deep_macro_sol, deep_micro_sols #deep_micro_sols, deep_macro_sol, macro_guess
    
    
    def save_sol(self, naive_sol, macro_sol, deep_macro_sol, filename):
        data = self.params.__dict__
        data['deep_mic_sol'] = [(m.r, m.t) for m in self.hmm_deep_solver.micro_solvers]
        data['mic_sol'] = [(ms.avg_vec, m.geom.grid.get_grid_and_weights()[0]) for ms, m in zip(self.micro_solvers, self.micros)]
        data['naive_sol_u'] = naive_sol.u.eval_grid()
        data['naive_sol_v'] = naive_sol.v.eval_grid()
        data['sol_u'] = macro_sol.u.eval_grid()
        data['sol_v'] = macro_sol.v.eval_grid()
        data['deep_sol_u'] = deep_macro_sol.u.eval_grid()
        data['deep_sol_v'] = deep_macro_sol.v.eval_grid()
        np.save(filename, data)
        
    @staticmethod
    def load_sol(filename):
        data = np.load(filename, allow_pickle=True).flatten()[0]
        u = data.pop('sol_u')
        v = data.pop('sol_v')
        u_deep = data.pop('deep_sol_u', None)
        v_deep = data.pop('deep_sol_v', None)
        u_naive = data.pop('naive_sol_u', None)
        v_naive = data.pop('naive_sol_v', None)
        m = data.pop('deep_mic_sol', None)
        mt = data.pop('mic_sol', None)
        par = HyperParams(**data)
        return par, u_naive, v_naive, u, v, u_deep, v_deep, m, mt#, u_deep, v_deep
    
path_from_type_seed = lambda net_type, seed: f"{net_dir}{net_type}_seed{seed}_40000.Torch"
w_list = [4.5, 5, 7, 8]#[3,5,10]#[2,5,7]
height_list = [0.8, 0.7, 1.0]#[0.8, 0.5, 1.0]
line_pos_list = [0.0, 0.01, 0.05, 0.1, 0.2]#[0.0, 0.01, 0.1]
seeds = [0,1,2,3,4,5]#, 1, 2, 3, 4, 5]
types = ["fno_ver1", "fno_ver2", "fno_ver3", "fno_ver4", "fno_vanilla_ver1", "fno_vanilla_ver2", "fno_vanilla_ver4", "fno_vanilla_ver3"]
data_product = product(types, seeds, w_list, height_list, line_pos_list)
data_product_list = list(data_product)


from multiprocessing import Pool
i_list = [2, 42-16, 42, 42+16]#, 74]
e_list = [0.001, 0.005, 0.01, 0.04]


i_list = [42+16]
e_list = [0.04]

for i, e in zip(i_list, e_list):
    #i = i_list[0]
    #e = e_list[0]

    num_pts = 256

    data, info = StokesData.load_from_matlab(f"{run_dir}run_{i}.mat")
    data_big_domain = data
    dom = data.dom
    eps = info["eps"]
    n = 7
    #w = 5
    
    def loop_func(args):
        try:
            it, (net_type, net_seed, w, h, lp) = args
            with open("/home/emastr/deep-micro-slip-model/data/logs/log.txt", "a") as f:
                f.write(f"BEGUN ----- {it}, i={i}, e={e}, net_type={net_type}, net_seed={net_seed}, w={w}, h={h}, lp={lp}\n")
            params = HyperParams(nMic=n, xDim=21, yDim=21, width=w*eps, height=w*eps*h, tol=1e-8, max_iter=30, num_pts=num_pts,line_pos=lp, net_path=path_from_type_seed(net_type, net_seed), net_type=net_type)
            setup = Setup(data, params)
            macro_naive, macro_sol, micro_sols, deep_macro_sol, deep_micro_sols = setup.solve()
            setup.save_sol(macro_naive, macro_sol, deep_macro_sol, f"{deep_save_dir}hmm_coupling_{i}_{net_type}_{net_seed}_{w}_{h}_{lp}.npy")
            with open("/home/emastr/deep-micro-slip-model/data/logs/log.txt", "a") as f:
                f.write(f"----- ENDED {it}, i={i}, e={e}, net_type={net_type}, net_seed={net_seed}, w={w}, h={h}, lp={lp}\n")
        except:
            print("ERROR - requested net not found (or other bug)")
        

    
    p = Pool(16)
    p.map(loop_func, list(enumerate(data_product_list)))#[2700:])
    #for it, (net_type, net_seed, w, h, lp) in enumerate(data_product_list):
    #    print(f"################### ITER {it}, i={i}, e={e}, net_type={net_type}, net_seed={net_seed}, w={w}, h={h}, lp={lp} ###################")
    #    
    #    params = HyperParams(nMic=n, xDim=21, yDim=21, width=w*eps, height=w*eps*h, tol=1e-8, max_iter=30, num_pts=num_pts,line_pos=lp, net_path=path_from_type_seed(net_type, net_seed), net_type=net_type)
    #    setup = Setup(data, params)

        #macro_naive, macro_sol, micro_sols, deep_macro_sol, deep_micro_sols = setup.solve()#deep_micro_sols, deep_macro_sol, macro_guess
        #setup.save_sol(macro_naive, macro_sol, deep_macro_sol, f"{deep_save_dir}hmm_coupling_{i}_{it}.npy")