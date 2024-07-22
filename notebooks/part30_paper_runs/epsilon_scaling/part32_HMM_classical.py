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
import matplotlib
import torch.nn as nn

#matplotlib.rcParams['text.usetex'] = True
#pretty_pyplot_layout()

np.random.seed(0)
net_dir = "/mnt/data0/emastr/article_training_nodecay/"
MESH_PATH = "/home/emastr/deep-micro-slip-model/data/mesh/"
figures_dir = "/home/emastr/deep-micro-slip-model/data/figures/"
simulation_dir = "/home/emastr/deep-micro-slip-model/data/stokes_fenics/"
run_dir = "/home/emastr/deep-micro-slip-model/data/reference_2/"

#save_dir = "/home/emastr/deep-micro-slip-model/data/hmm_coupling_error/"
save_dir = "/mnt/data0/emastr/hmm_coupling_paper/"

from dataclasses import dataclass
from multiprocessing import Pool

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
    line_pos: float
    
    
class Setup():
    def __init__(self, data: StokesData, param: HyperParams):#, net: nn.Module, net_settings: dict):
        """Create a new HMM setup with the given parameters."""
        self.params = param
        self.data = data
        
        # Macro problem
        macro = StokesMacProb(data, lambda x,a: fft_interp(x, a, dom=data.dom[0]))
        macro_solver = MacroSolver(param.xDim, param.yDim, tol=param.tol)
        
        # Micro problems
        self.xPos = np.linspace(data.dom[0][0], data.dom[0][1], param.nMic+1)[1:] - param.width/2 - (data.dom[0][1]-data.dom[0][0]) / param.nMic/1
        micros = [StokesTrapezMicProb(data, x, param.width, param.height, param.line_pos, 8, param.xDim, param.yDim, num_pts=param.num_pts) for x in self.xPos]
        micro_solvers = [MicroSolver(m, tol=param.tol) for m in micros]
        #deep_micro_solvers = [DeepMicroSolver(m, net, net_settings, tol=param.tol) for m in micros]
        
        # Convergence checker
        conv_checker = ConvergenceChecker(macro, micros, tol=param.tol)
        
        # HMM problem
        self.hmm_problem = StokesHMMProblem(macro, micros, data, convergence_checker=conv_checker)
        self.hmm_solver = IterativeHMMSolver(macro_solver, micro_solvers)
        #self.hmm_deep_solver = IterativeHMMSolver(macro_solver, deep_micro_solvers)
        
        
    def solve(self, **kwargs):
        macro_guess = self.hmm_solver.macro_solver.solve(self.hmm_problem.macro)
        (macro_sol, micro_sols) = self.hmm_solver.solve(self.hmm_problem, macro_guess=macro_guess, verbose=True, maxiter=self.params.max_iter, tol=self.params.tol)
        #(deep_macro_sol, deep_micro_sols) = self.hmm_deep_solver.solve(self.hmm_problem, macro_guess=macro_guess, verbose=True, maxiter=self.params.max_iter, tol=self.params.tol)
        
        return micro_sols, macro_sol, macro_guess #deep_micro_sols, deep_macro_sol, macro_guess
    
    
    def save_sol(self, macro_sol, filename):
        data = self.params.__dict__
        data['mic_sol'] = [m.avg_vec for m in self.hmm_solver.micro_solvers]
        data['sol_u'] = macro_sol.u.eval_grid()
        data['sol_v'] = macro_sol.v.eval_grid()
        #data['deep_sol_u'] = deep_macro_sol.u.eval_grid()
        #data['deep_sol_v'] = deep_macro_sol.v.eval_grid()
        #print(np.linalg.norm(data['sol_u'] - data['deep_sol_u'])/np.linalg.norm(data['sol_u']))
        np.save(filename, data)
        
    @staticmethod
    def load_sol(filename):
        data = np.load(filename, allow_pickle=True).flatten()[0]
        u = data.pop('sol_u')
        v = data.pop('sol_v')
        m = data.pop('mic_sol')
        #u_deep = data.pop('deep_sol_u')
        #v_deep = data.pop('deep_sol_v')
        par = HyperParams(**data)
        return par, u, v, m#, u_deep, v_deep
    
    
w_list = [4.5, 5, 7, 8]#[3,5,10]#[2,5,7]
height_list = [0.8, 0.7, 1.0]#[0.8, 0.5, 1.0]
line_pos_list = [0.0, 0.01, 0.05, 0.1, 0.2]#[0.0, 0.01, 0.1]

from multiprocessing import Pool
#step = [1]#[1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 46, 64]#, 128]
step = [1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 46, 64]#, 128]
#M_list = [256]#[s * 20 for s in step]
M_list = [s * 20 for s in step]
#i_list = list(range(2,80,8))#:list(range(50,60,1))
i_list = [2, 42-16, 42, 42+16]#

data_product = product(M_list, w_list, height_list, line_pos_list)
data_product_list = list(data_product)

print(len(i_list)*len(data_product_list), (5952-192)/2)


for i in i_list:
    
    def run(args):
        m, w, h, lp = args
        with open(f"/home/emastr/deep-micro-slip-model/data/logs/classical_log.txt", "a") as f:
            f.write(f"START --- Running sim {i} with {m} pts, {w} width, {h} height, {lp} line pos \n")
        
        data, info = StokesData.load_from_matlab(f"{run_dir}run_{i}.mat")
        data_big_domain = data
        dom = data.dom
        eps = info["eps"]
        n = 7
        
        params = HyperParams(nMic=n, xDim=21, yDim=21, width=w*eps, height=w*eps*h, tol=1e-8, max_iter=30, num_pts=m, line_pos=lp)
        setup = Setup(data, params)
        micro_sols, macro_sol, macro_guess = setup.solve()
        setup.save_sol(macro_sol, f"{save_dir}big_run_{i}_{m}_{w}_{h}_{lp}_sol.npy")
                
        with open(f"/home/emastr/deep-micro-slip-model/data/logs/classical_log.txt", "a") as f:
            f.write(f"----- END Running sim {i} with {m} pts, {w} width, {h} height, {lp} line pos \n")
    
    p = Pool(16)
    p.map(run, data_product_list)