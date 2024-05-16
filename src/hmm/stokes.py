import typing
import time
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from util.basis_scaled import FourBasis, ChebBasis, ScaleShiftedBasis
from util.plot_tools import *
from hmm.hmm import MacroProblem, MicroProblem, HMMProblem, Solver, IterativeHMMSolver
from boundary_solvers.gauss_grid_2d import TrapezGrid
from boundary_solvers.geometry import Geometry, RoundedMicroGeom, ShiftScaledGeom, MacroGeom, MacroGeomGeneric, RoundedMicroGeomV2
from stokes2d.robin_solver import solveRobinStokes_fromFunc
from scipy.io import loadmat
from dataclasses import dataclass
from typing import Callable

# Decouple complex and real arguments (naive implementation, requires 2x fcn evaluations)
fromComplex = lambda func: (lambda *args: np.real(func(*args)), lambda *args: np.imag(func(*args)))

@dataclass
class StokesData:
    f : Callable
    df : Callable
    ddf : Callable
    g : Callable
    dom : List[List[float]]
    width : float
    height : float
        
    def plot(self, axis, npts=100, **kwargs):
        geom = MacroGeomGeneric(self.f, self.df, self.ddf, self.dom, 1, 1)
        geom.plot(axis, showpts=False, npts=npts, **kwargs)
        axis.plot(self.dom[0], [self.dom[1][0],self.dom[1][0]], '--', color='black')
        
    
    def geom(self, **kwargs):
        return MacroGeomGeneric(self.f, self.df, self.ddf, self.dom, **kwargs)

    def copy(self):
        return StokesData(self.f, self.df, self.ddf, self.g, self.dom, self.width, self.height)
        
    @staticmethod
    def load_from_matlab(path: str):
        """Load data from matlab file. Save additional info in info_dict."""
        matf = loadmat(path, struct_as_record=True)
        info = matf['info'][0][0]
        params = info['input_params'][0][0]
        amp = params['amplitude_condition'].squeeze()
        eps = params['amplitude_bottom'].squeeze()
        nper = params['n_periods_bottom'].squeeze()
        freq_g = params['n_periods_condition'].squeeze()

        # Hard coded
        eval_pos = 0.3 * eps #0.005 # 0.01
        dom = [[-1.,1.], [eval_pos, 0.5]] # before [eval_pos, 0.5]
        bbox = [dom[0], [-1., 1.]]
        k = 4 * nper
        Lx = dom[0][1] - dom[0][0]


        # Fnctions
        f = lambda x: eps * (-1 + np.sin(k*x))-eval_pos #1.6
        df = lambda x: eps * k * np.cos(k*x)
        ddf = lambda x: -eps * (k**2) * np.sin(k*x)
        g = lambda x: dom[1][1]*(1 + (np.sin(2*np.pi * x / Lx)*0.4 + np.sin(freq_g*2*np.pi * x / Lx)*0.6) * amp)#*np.ones_like(x)
        
        
        info_dict = {
            "amp": amp, 
            "eps": eps, 
            "nper": nper, 
            "freq_g": freq_g,
            "eval_pos": eval_pos,
            "Lx": Lx,
            "bbox": bbox,
            "X": info['X'],
            "Y": info['Y'],
            "Uc": info['Uc'],
            "Vc": info['Vc'],
            "Uyc": info['Uyc'],
            "Uxc": info['Uxc'],
            "Vyc": info['Vyc'],
            "Vxc": info['Vxc'],
            "Pc": info['Pc'],
        }

        data = StokesData(f, df, ddf, g, dom, dom[0][1]-dom[0][0], dom[1][1]-dom[1][0])
        return data, info_dict

@dataclass
class HyperParams:
    nMic: int
    n_refine: int
    xDim: int
    yDim: int
    width: float
    height: float
    tol: float
    max_iter: int
        
    
class StokesHMMWrapper():
    def __init__(self, data: StokesData, param: HyperParams):
        """Create a new HMM setup with the given parameters."""
        self.params = param
        self.data = data
        
        # Macro problem
        macro = StokesMacProb(data, lambda x,a: fft_interp(x, a, dom=data.dom[0]))
        macro_solver = MacroSolver(param.xDim, param.yDim, tol=param.tol)
        
        # Micro problems
        self.xPos = np.linspace(data.dom[0][0], data.dom[0][1], param.nMic+1)[1:] - param.width/2 - (data.dom[0][1]-data.dom[0][0]) / param.nMic/1
        micros = [StokesMicProb(data, x, param.width, param.height, 0.0, 8, param.xDim, param.yDim, n_refine=param.n_refine) for x in self.xPos]
        micro_solvers = [MicroSolver(m, tol=param.tol) for m in micros]
        
        # Convergence checker
        conv_checker = ConvergenceChecker(macro, micros, tol=param.tol)
        
        # HMM problem
        self.hmm_problem = StokesHMMProblem(macro, micros, data, convergence_checker=conv_checker)
        self.hmm_solver = IterativeHMMSolver(macro_solver, micro_solvers)
        
        
    def solve(self, **kwargs):
        macro_guess = self.hmm_solver.macro_solver.solve(self.hmm_problem.macro)
        (macro_sol, micro_sols) = self.hmm_solver.solve(self.hmm_problem, macro_guess=macro_guess, verbose=True, maxiter=self.params.max_iter, tol=self.params.tol)
        return micro_sols, macro_sol, macro_guess
    
    
    def save_sol(self, macro_sol, filename):
        data = self.params.__dict__
        data['sol_u'] = macro_sol.u.eval_grid()
        data['sol_v'] = macro_sol.v.eval_grid()
        np.save(filename, data)
        
        
    def load_sol(filename):
        data = np.load(filename, allow_pickle=True).flatten()[0]
        u = data.pop('sol_u')
        v = data.pop('sol_v')
        par = HyperParams(**data)
        return par, u, v

        
class StokesHMMProblem(HMMProblem):
    """HMM Stokes problem"""
    def __init__(self, macro, micros, data, *args, **kwargs):
        """Given a macro problem and a micro problem(s), construct a HMM solver."""
        self.data = data
        super(StokesHMMProblem, self).__init__(macro, micros, *args, **kwargs)
        
    def plot(self, ax, npts=1000, **kwargs):
        self.data.plot(ax, npts, **kwargs)
        #self.macro.plot(ax)
        for m in self.micros:
            m.plot(ax, **kwargs)

    
class StokesMacProb(MacroProblem):
    def __init__(self, stokes_data: StokesData, interp, alpha0=None):
        """Create Stokes Macro Problem."""
        self.stokes_data = stokes_data
        self.interp = interp
        if alpha0 is None:
            self.alpha = lambda x: np.zeros_like(x)
        else:
            self.alpha = alpha0
        pass
    
    def is_solution(self, macro_sol, tol = 1e-5):
        # Check alpha versus updated alpha.
        #du = macro_sol.u.diff(0, 1).reduce_eval(self.stokes_data.dom[1][0], axis=1)
        #u = macro_sol.u.reduce_eval(self.stokes_data.dom[1][0], axis=1)
        #alpha = (-1.) * u / du
        #err = np.mean((self.alpha - alpha).eval_grid()**2)
        #return err < tol
        return False
    
    def update(self, micro_sol):
        x = np.array([m.x for m in micro_sol])
        a = np.array([m.alpha for m in micro_sol])
        self.alpha = self.interp(x, a)
        pass
    

class MacroSolver(Solver):
    def __init__(self, xDim, yDim, tol=1e-5, logger=None, **kwargs):
        self.xDim = xDim
        self.yDim = yDim
        self.tol = tol
        self.logger = logger
    
    def can_solve(self, problem) -> bool:
        """Return True if solver can solve this problem."""
        return isinstance(problem, StokesMacProb)
    
    def solve(self, problem: MacroProblem, *args, **kwargs):
        """Solve and return solution"""
        
        zero2d = lambda x,y: np.zeros_like(x).astype(np.complex128)
        zero = lambda x: np.zeros_like(x).astype(np.complex128)
        alpha = problem.alpha
        gx = lambda x: np.real(problem.stokes_data.g(x))
        gy = lambda x: np.imag(problem.stokes_data.g(x))
        dom = problem.stokes_data.dom
        
        if self.logger is not None:
            self.logger.start_event("macro_solve")

        u, v, info = solveRobinStokes_fromFunc(zero2d, zero2d, alpha, gx, zero, gy, zero, dom,
                                               self.xDim, self.yDim, tol=self.tol, logger=self.logger)

        if self.logger is not None:
            self.logger.end_event("macro_solve")
        
        class MacroSol():
            def __init__(self, u, v, info):
                self.u = u
                self.v = v
                self.info = info
            
            def plot_stream(self, ax, npts=20, **kwargs):
                dom = self.u.domain()
                x = np.linspace(dom[0][0],dom[0][1],npts), 
                y = np.linspace(dom[1][0],dom[1][1],npts)
                x,y = np.meshgrid(x,y)
                u = self.u(x.flatten(), y.flatten()).reshape((npts,npts))
                v = self.v(x.flatten(), y.flatten()).reshape((npts,npts))
                
                #x = u.xBasis.grid(u.xDim)
                #y = x.yBasis.grid(u.yDim)
                #X,Y = u.grid()
                #U = u.eval_grid()
                #V = v.eval_grid()
                ax.streamplot(x, y, u, v, **kwargs)
                
        return MacroSol(u, v, info)
    
    
class StokesMicProb(MicroProblem):
    def __init__(self, stokes_data: StokesData, xPos, width, height, linePos, deg_project=None, xDim_reduce=None, yDim_reduce=None, logger=None, **kwargs):
        self.stokes_data = stokes_data
        self.xPos = xPos
        self.width = width
        self.height = height
        self.linePos = linePos
        self.deg_project = deg_project
        self.logger = logger
        self.xDim_reduce = xDim_reduce
        self.yDim_reduce = yDim_reduce
        
        # Rescale to [0, 0.5pi] domain
        dom = stokes_data.dom
        wid = stokes_data.width
        x0 = stokes_data.dom[0][0]  # Left point
        
        dom1 = [(xPos - x0)/wid*0.5*np.pi, (xPos+width-x0)/wid*0.5*np.pi]
        dom2 = [0, 0.5*np.pi]
        func = Geometry.change_domain([stokes_data.f, stokes_data.df, stokes_data.ddf], dom1, dom2)        
        #self.geom = RoundedMicroGeom(*func, 
                                     #width = width, 
                                     #height = height, 
                                     #corner_w = width*0.2, #0.2
                                     #center_x = xPos + 0.5*width,
                                     #shift_y = stokes_data.dom[1][0],
                                     #line_pos = linePos*height, **kwargs)
        
        self.geom = RoundedMicroGeomV2(*func, 
                                      width = width, 
                                      height = height, 
                                      corner_w = width*0.2,
                                      center_x = xPos + 0.5*width,
                                      shift_y = stokes_data.dom[1][0],
                                      line_pos = linePos*height, **kwargs)
        
        # Translate and scale to fit original 
        #shift = 0.5*width + xPos - 0.5j * stokes_data.height
        #scale = height
        #self.geom_scaled = ShiftScaledGeom(self.geom, shift, scale)
        self.condition = None
        
    
    def is_solution(self, micro_sol, tol = 1e-5):
        # Requires a solver to check. instead, check convergence.
        return False
    
    def update(self, macro_sol, **kwargs):
        """Given a solution macroSol to the macro problem, read off the solution at points in the micro problem,
        and extrapolate to missing parts of the micro boundary."""
        
        # Joint components
        u = macro_sol.u
        v = macro_sol.v

        if (self.yDim_reduce is None) or (self.xDim_reduce is None):
           yDim = u.yDim
           xDim = u.xDim
        else:
           yDim = self.yDim_reduce
           xDim = self.xDim_reduce
        u_ = u.change_dim(xDim, yDim)
        v_ = v.change_dim(xDim, yDim)
        
        def toComplex(fx, fy):
            return lambda z: fx(np.real(z), np.imag(z)) + 1j * fy(np.real(z), np.imag(z))
        
        if self.logger is not None:
            self.logger.start_event("micro_update_diff")

        U = toComplex(u_, v_)
        dxU = toComplex(u_.diff(1,0), v_.diff(1,0))
        dyU = toComplex(u_.diff(0,1), v_.diff(0,1))
        dxdxU = toComplex(u_.diff(2,0), v_.diff(2,0))
        dxdyU = toComplex(u_.diff(1,1), v_.diff(1,1))
        dydyU = toComplex(u_.diff(0,2), v_.diff(0,2))

        #dxU = toComplex(u.diff_inplace(1,0), v.diff_inplace(1,0))
        #dyU = toComplex(u.diff_inplace(0,1), v.diff_inplace(0,1))
        #dxdxU = toComplex(u.diff_inplace(2,0), v.diff_inplace(2,0))
        #dxdyU = toComplex(u.diff_inplace(1,1), v.diff_inplace(1,1))
       # dydyU = toComplex(u.diff_inplace(0,2), v.diff_inplace(0,2))
        
        if self.logger is not None:
            self.logger.end_event("micro_update_diff")

        z  = lambda t: self.geom.eval_param(t=t)
        dz = lambda t: self.geom.eval_param(t=t, derivative=1)
        ddz = lambda t: self.geom.eval_param(t=t, derivative=2)

        def g(t):
            return U(z(t))

        def dg(t):
            z_ = z(t)
            dz_ = dz(t)
            return np.real(dz_) * dxU(z_) + np.imag(dz_) * dyU(z_)

        def ddg(t):
            z_ = z(t)
            dz_ = dz(t)
            ddz_ = ddz(t)

            x, y = np.real(z_), np.imag(z_)
            dx, dy = np.real(dz_), np.imag(dz_)
            ddx, ddy = np.real(ddz_), np.imag(ddz_)    

            return ddx * dxU(z_) + ddy * dyU(z_) + (dx**2) * dxdxU(z_) + (2*dx*dy)*dxdyU(z_) + (dy**2)*dydyU(z_)
        
        if self.logger is not None:
            self.logger.start_event("micro_update_fit")

        deg_project = kwargs.pop("N", self.deg_project)
        g_extrap = self.geom.project(g, dg, ddg, N=deg_project, **kwargs)
        self.set_data(g_extrap)

        if self.logger is not None:
            self.logger.end_event("micro_update_fit")
    
    def set_data(self, condition):
        """Change the boundary condition of the micro problem."""
        self.condition = condition
    
    def plot(self, axis, **kwargs):
        """Plot micro domain. relative = True means plotting in true coordinates."""
        self.geom.plot(axis, **kwargs)
        
class StokesTrapezMicProb(MicroProblem):
    def __init__(self, stokes_data: StokesData, xPos, width, height, linePos, deg_project=None, xDim_reduce=None, yDim_reduce=None, num_pts=100, logger=None, **kwargs):
        self.stokes_data = stokes_data
        self.xPos = xPos
        self.width = width
        self.height = height
        self.linePos = linePos
        self.deg_project = deg_project
        self.logger = logger
        self.xDim_reduce = xDim_reduce
        self.yDim_reduce = yDim_reduce
        
        # Rescale to [0, 0.5pi] domain
        dom = stokes_data.dom
        wid = stokes_data.width
        x0 = stokes_data.dom[0][0]  # Left point
        
        dom1 = [(xPos - x0)/wid*0.5*np.pi, (xPos+width-x0)/wid*0.5*np.pi]
        dom2 = [0, 0.5*np.pi]
        func = Geometry.change_domain([stokes_data.f, stokes_data.df, stokes_data.ddf], dom1, dom2)        
        self.geom = RoundedMicroGeomV2(*func, 
                                     width = width, 
                                     height = height, 
                                     corner_w = width*0.2,
                                     center_x = xPos + 0.5*width,
                                     shift_y = stokes_data.dom[1][0],
                                     line_pos = linePos*height, **kwargs)
        self.geom.grid = TrapezGrid(num_pts)
        self.condition = None
        
    
    def is_solution(self, micro_sol, tol = 1e-5):
        # Requires a solver to check. instead, check convergence.
        return False
    
    def update(self, macro_sol, **kwargs):
        """Given a solution macroSol to the macro problem, read off the solution at points in the micro problem,
        and extrapolate to missing parts of the micro boundary."""
        
        # Joint components
        u = macro_sol.u
        v = macro_sol.v

        if (self.yDim_reduce is None) or (self.xDim_reduce is None):
           yDim = u.yDim
           xDim = u.xDim
        else:
           yDim = self.yDim_reduce
           xDim = self.xDim_reduce
        u_ = u.change_dim(xDim, yDim)
        v_ = v.change_dim(xDim, yDim)
        
        def toComplex(fx, fy):
            return lambda z: fx(np.real(z), np.imag(z)) + 1j * fy(np.real(z), np.imag(z))
        
        if self.logger is not None:
            self.logger.start_event("micro_update_diff")

        U = toComplex(u_, v_)
        dxU = toComplex(u_.diff(1,0), v_.diff(1,0))
        dyU = toComplex(u_.diff(0,1), v_.diff(0,1))
        dxdxU = toComplex(u_.diff(2,0), v_.diff(2,0))
        dxdyU = toComplex(u_.diff(1,1), v_.diff(1,1))
        dydyU = toComplex(u_.diff(0,2), v_.diff(0,2))
        
        if self.logger is not None:
            self.logger.end_event("micro_update_diff")

        z  = lambda t: self.geom.eval_param(t=t)
        dz = lambda t: self.geom.eval_param(t=t, derivative=1)
        ddz = lambda t: self.geom.eval_param(t=t, derivative=2)

        def g(t):
            return U(z(t))

        def dg(t):
            z_ = z(t)
            dz_ = dz(t)
            return np.real(dz_) * dxU(z_) + np.imag(dz_) * dyU(z_)

        def ddg(t):
            z_ = z(t)
            dz_ = dz(t)
            ddz_ = ddz(t)

            x, y = np.real(z_), np.imag(z_)
            dx, dy = np.real(dz_), np.imag(dz_)
            ddx, ddy = np.real(ddz_), np.imag(ddz_)    

            return ddx * dxU(z_) + ddy * dyU(z_) + (dx**2) * dxdxU(z_) + (2*dx*dy)*dxdyU(z_) + (dy**2)*dydyU(z_)
        
        if self.logger is not None:
            self.logger.start_event("micro_update_fit")

        deg_project = kwargs.pop("N", self.deg_project)
        g_extrap = self.geom.project(g, dg, ddg, N=deg_project, **kwargs)
        self.set_data(g_extrap)

        if self.logger is not None:
            self.logger.end_event("micro_update_fit")
    
    def set_data(self, condition):
        """Change the boundary condition of the micro problem."""
        self.condition = condition
    
    def plot(self, axis, **kwargs):
        """Plot micro domain. relative = True means plotting in true coordinates."""
        self.geom.plot(axis, **kwargs)
        
     
        
class MicroSolver(Solver):
    """Solve the micro problem at a specific position. Precompute."""
    def __init__(self, problem, logger=None, **kwargs):
        geom = problem.geom
        self.logger = logger

        if self.logger is not None:
            self.logger.start_event("micro_precompute")

        self.avg_vec, self.avg = geom.precompute_line_avg(with_gmres=True, tol=9*1e-3)#**kwargs)
        self.davg_vec, self.davg = geom.precompute_line_avg(derivative=1, **kwargs)

        if self.logger is not None:
            self.logger.end_event("micro_precompute")
        
    def can_solve(self, problem):
        return isinstance(problem, StokesMicProb) or isinstance(problem, StokesTrapezMicProb)
        
    def solve(self, problem: StokesMicProb):
        # Log Solve time
        if self.logger is not None:
            self.logger.start_event("micro_solve")

        class MicroData():
            def __init__(self, x, a):
                self.alpha = a
                self.x = x
        t,_ = problem.geom.grid.get_grid_and_weights()
        c = problem.condition(t)
        
        out = MicroData(problem.xPos + problem.width/2, -self.avg(c) / self.davg(c))

        # End log
        if self.logger is not None:
            self.logger.end_event("micro_solve")

        return out
    

class ConvergenceChecker():
    def __init__(self, macros, micros, tol=1e-5):
        self.converged = False
        self.tol = tol
        self.alphas = np.zeros((len(micros)))
    
    def __call__(self, macro_sol, micro_sols, *args, **kwargs):
        alphas = np.array([m.alpha for m in micro_sols])
        self.converged = np.mean((alphas - self.alphas)**2) < self.tol**2
        self.alphas = alphas
        return self.converged


def trig_interp(x, a, deg=11, L=1.):
    """Interpolate values in fourier series"""
    kmax = (deg - 1)// 2
    ks = np.arange(-kmax, kmax+1)
    A = np.kron(np.eye(2), np.diag(np.abs(ks))**2 + 0. * np.ones((deg,)))
    V = np.hstack([np.exp(2j * np.pi * x * k / L)[:, None] for k in ks])
    V = np.vstack([np.hstack([np.real(V), -np.imag(V)]),
                   np.hstack([np.imag(V), np.real(V)])])
    
    sys = np.vstack([np.hstack([A, V.T]), np.hstack([V, np.zeros((V.shape[0], V.shape[0]))])])
    rhs = np.vstack([np.zeros((2*deg, 1)), np.real(a[:, None]), np.imag(a[:, None])])
    
    #coef = np.linalg.solve((V.T @ V + reg * np.eye(deg)), V.T @ a[:, None])
    #coef = np.linalg.solve(sys, rhs)
    coef = np.linalg.solve(sys.T @ sys, sys.T @ rhs)
    
    coef = (coef[:deg] + coef[deg:2*deg] * 1j).flatten()
    def fcn(x):
        out = np.zeros_like(x).astype(np.complex128)
        for k, c in zip(ks, coef):
            out = out + np.exp(2j* np.pi * k  * x / L) * c
        return np.real(out)
    return fcn


def fft_interp(x, a, dom, dim=None):
    N = len(a)
    a = np.array(a)
    assert N % 2 != 0, "Must be odd."
    basis = FourBasis(FourBasis._interpolate(a))
    scale, shift = ScaleShiftedBasis._domain_to_scale_shift(FourBasis._domain(), dom)
    basis = ScaleShiftedBasis(basis, scale, shift)
    if dim is not None:
        basis = basis.change_dim(dim)
    return basis

