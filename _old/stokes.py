import sys
sys.path.append('/home/emastr/phd/')
import typing
import time
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from util.basis import FourBasis
from util.plot_tools import *
from hmm.hmm import MacroProblem, MicroProblem, HMMProblem, Solver, IterativeHMMSolver
from boundary_solvers.geometry import Geometry, RoundedMicroGeom, ShiftScaledGeom, MacroGeom
from old.robin_solver import testSolve, solveRobinStokes_fromFunc


# Decouple complex and real arguments (naive implementation, requires 2x fcn evaluations)
fromComplex = lambda func: (lambda *args: np.real(func(*args)), lambda *args: np.imag(func(*args)))

class StokesData:
    def __init__(self, f, df, ddf, g):
        """Stokes dirichlet data problem with periodicity in x-axis. 
           Problem is defined on [0,1]x[-1,1] U {(x, y): y in [f(x), 1], x in [0, 1]}.
           That is, the space between f(x) and 1 for all x in [0, 1].
           The function g sets a boundary condition on the ceiling of the channel. 
           g must be periodic."""
        
        self.f = f
        self.df = df
        self.ddf = ddf
        self.g = g
        self.width = 1
        self.height = 2
        
    def plot(self, ax):
        geom = MacroGeom(self.f, self.df, self.ddf, 1, 2, 1, 1)
        geom.plot(ax, showpts=False, npts=1000)
        ax.plot([0, 1], [-1, -1], '--', color='black')
    

        
class StokesHMMProblem(HMMProblem):
    """HMM Stokes problem"""
    def __init__(self, macro, micros, data, *args, **kwargs):
        """Given a macro problem and a micro problem(s), construct a HMM solver."""
        self.data = data
        super(StokesHMMProblem, self).__init__(macro, micros, *args, **kwargs)
        
    def plot(self, ax, **kwargs):
        self.data.plot(ax)
        #self.macro.plot(ax)
        for m in self.micros:
            m.plot(ax)

    
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
    
    def is_solution(self, x, tol = 1e-5):
        return False
    
    def update(self, micro_sol):
        x = np.array([m.x for m in micro_sol])
        a = np.array([m.alpha for m in micro_sol])
        self.alpha = self.interp(x, a)
        pass
    
    
class MacroSolver(Solver):
    def __init__(self, xDim, yDim):
        self.xDim = xDim
        self.yDim = yDim
    
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
        
        u, v, info = solveRobinStokes_fromFunc(zero2d, zero2d, alpha, gx, zero, gy, zero, 
                                               self.xDim, self.yDim, with_gmres=False, maxiter=10, tol=1e-5)
        
        class MacroSol():
            def __init__(self, u, v, info):
                self.u = u
                self.v = v
                self.info = info
            
            def plot_stream(self, ax, npts=20, **kwargs):
                x,y = np.linspace(0,1,npts), np.linspace(-1,1,npts)
                x,y = np.meshgrid(x,y)
                u = self.u(x.flatten(), y.flatten()).reshape((npts,npts))
                v = self.v(x.flatten(), y.flatten()).reshape((npts,npts))
                ax.streamplot(x, y, u, v, **kwargs)
                
        return MacroSol(u, v, info)
    
    
    
class StokesMicProb(MicroProblem):
    def __init__(self, stokes_data: StokesData, xPos, width, height, linePos):
        self.stokes_data = stokes_data
        self.xPos = xPos
        self.width = width
        self.height = height
        self.linePos = linePos
        
        # Rescale to [0, 0.5pi] domain
        dom1 = [xPos*0.5*np.pi, (xPos+width)*0.5*np.pi]
        dom2 = [0, 0.5*np.pi]
        func = Geometry.change_domain([stokes_data.f, stokes_data.df, stokes_data.ddf], dom1, dom2)        
        self.geom = RoundedMicroGeom(*func, 
                                     width = width, 
                                     height = height, 
                                     corner_w = width*0.3,
                                     center_x = xPos + 0.5*width,
                                     shift_y = -1,
                                     line_pos = linePos*height, 
                                     n_refine = 2, 
                                     n_corner_refine = 0)
        
        # Translate and scale to fit original 
        shift = 0.5*width + xPos - 0.5j * stokes_data.height
        scale = height
        #self.geom_scaled = ShiftScaledGeom(self.geom, shift, scale)
        self.condition = None
        
    
    def is_solution(self, micro_sol, tol = 1e-5):
        return False
    
    def update(self, macro_sol, **kwargs):
        """Given a solution macroSol to the macro problem, read off the solution at points in the micro problem,
        and extrapolate to missing parts of the micro boundary."""
        
        # Joint components
        u = macro_sol.u
        v = macro_sol.v
        
        def toComplex(fx, fy):
            return lambda z: fx(np.real(z), np.imag(z)) + 1j * fy(np.real(z), np.imag(z))
        
        U = toComplex(u, v)
        dxU = toComplex(u.diff(1,0), v.diff(1,0))
        dyU = toComplex(u.diff(0,1), v.diff(0,1))
        dxdxU = toComplex(u.diff(2,0), v.diff(2,0))
        dxdyU = toComplex(u.diff(1,1), v.diff(1,1))
        dydyU = toComplex(u.diff(0,2), v.diff(0,2))
        
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
        
        g_extrap = self.geom.project(g, dg, ddg, **kwargs)
        self.set_data(g_extrap)
    
    def set_data(self, condition):
        """Change the boundary condition of the micro problem."""
        self.condition = condition
    
    def plot(self, axis, **kwargs):
        """Plot micro domain. relative = True means plotting in true coordinates."""
        self.geom.plot(axis, **kwargs)
        
        
class MicroSolver(Solver):
    """Solve the micro problem at a specific position. Precompute."""
    def __init__(self, problem, **kwargs):
        geom = problem.geom
        self.avg_vec, self.avg = geom.precompute_line_avg(**kwargs)
        self.davg_vec, self.davg = geom.precompute_line_avg(derivative=1, **kwargs)
        
    def can_solve(self, problem):
        return isinstance(problem, StokesMicProb)
        
    def solve(self, problem: StokesMicProb):
        class MicroData():
            def __init__(self, x, a):
                self.alpha = a
                self.x = x
        t,_ = problem.geom.grid.get_grid_and_weights()
        c = problem.condition(t)
        return MicroData(problem.xPos + problem.width/2, -self.avg(c) / self.davg(c))
    

def trig_interp(x, a, deg=11):
    """Interpolate values in fourier series"""
    kmax = (deg - 1)// 2
    ks = np.arange(-kmax, kmax+1)
    A = np.kron(np.eye(2), np.diag(np.abs(ks))+ 0. * np.ones((deg,)))
    V = np.hstack([np.exp(2j * np.pi * x * k)[:, None] for k in ks])
    V = np.vstack([np.hstack([np.real(V), -np.imag(V)]),
                   np.hstack([np.imag(V), np.real(V)])])
    
    sys = np.vstack([np.hstack([A, V.T]), np.hstack([V, np.zeros((V.shape[0], V.shape[0]))])])
    rhs = np.vstack([np.zeros((2*deg, 1)), np.real(a[:, None]), np.imag(a[:, None])])
    
    #coef = np.linalg.solve((V.T @ V + reg * np.eye(deg)), V.T @ a[:, None])
    coef = np.linalg.solve(sys, rhs)
    
    coef = (coef[:deg] + coef[deg:2*deg] * 1j).flatten()
    def fcn(x):
        out = np.zeros_like(x).astype(np.complex128)
        for k, c in zip(ks, coef):
            out = out + np.exp(2j* np.pi * k  * x) * c
        return np.real(out)
    return fcn