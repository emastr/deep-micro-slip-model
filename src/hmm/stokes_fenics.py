import typing
import time
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from util.basis_scaled import FourBasis, ChebBasis, ScaleShiftedBasis
from util.plot_tools import *
from hmm.hmm import MacroProblem, MicroProblem, HMMProblem, Solver, IterativeHMMSolver
from boundary_solvers.geometry import Geometry, RoundedMicroGeom, ShiftScaledGeom, MacroGeom, MacroGeomGeneric, RoundedMicroGeomGeneric, RoundedMicroGeomGenericV2
from util.mesh_tools import find_intersection, curve_length, reparameterize_curve, find_intersection_on_segment, get_indicator, to_fenics_func
from util.mesh_tools import project_to_boundary, Box, boundary_bbox_tree, order_connected_vertices
from scipy.io import loadmat
from dataclasses import dataclass
from typing import Callable
import dolfin as dl
import fenics as fn
import mshr as ms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, RBF


# Decouple complex and real arguments (naive implementation, requires 2x fcn evaluations)
fromComplex = lambda func: (lambda *args: np.real(func(*args)), lambda *args: np.imag(func(*args)))

def macro_to_micro(macro_sol, geom, feasible_domain: List[float], **kwargs):
    u, v, gu, gv = macro_sol.u, macro_sol.v, macro_sol.grad_u, macro_sol.grad_v
    t = feasible_domain
    z = geom.eval_param(t=t)
    dz = geom.eval_param(t=t, derivative=1)
    ddz = geom.eval_param(t=t, derivative=2)
    
    pts = [dl.Point(z[i].real, z[i].imag) for i in range(len(z))]
    
    u_vals = np.array([u(p) for p in pts])
    v_vals = np.array([v(p) for p in pts])
    du_vals = np.array([gu(p) for p in pts])
    dv_vals = np.array([gv(p) for p in pts])
    
    
    g_bdry = [u_v + 1j * v_v for u_v, v_v in zip(u_vals, v_vals)]
    dg_bdry = [d.real*(du_v[0] + 1j * dv_v[0]) + d.imag*(du_v[1] + 1j * dv_v[1]) for d, du_v, dv_v in zip(dz, du_vals, dv_vals)]
    ddg_bdry = [dd.real*(du_v[0] + 1j * dv_v[0]) + dd.imag*(du_v[1] + 1j * dv_v[1]) for dd, du_v, dv_v in zip(ddz, du_vals, dv_vals)]

    def g(t):
        z = geom.eval_param(t=t)
        pts = [dl.Point(z[i].real, z[i].imag) for i in range(len(z))]
        return np.array([u(p) + 1j * v(p) for p in pts])
    
    return geom.project(g, g_bdry, dg_bdry, ddg_bdry, feasible_domain, **kwargs)


def micros_to_macro(x, a, scale=None):
    if scale is None:
        scale = 2 * min(np.linalg.norm(x[0, :] - x[1:, :], axis=1))
    kernel = RBF(scale, length_scale_bounds="fixed")
    regressor = GaussianProcessRegressor(kernel, normalize_y=True)
    regressor.fit(x, a)

    def interp(x, y):
        xy = np.hstack([x, y])[None, :]
        return regressor.predict(xy)[0]
    
    return interp

@dataclass
class StokesData:
    """Stokes data. Contains domain, boundary conditions, and forcing."""
    #rough_domain: int 
    macro_mesh: dl.Mesh
    smoothed_rough_boundary: dl.BoundaryMesh
    smoothed_rough_boundary_tree: dl.BoundingBoxTree
    smoothed_rough_idx_map: np.ndarray #Index Map #ms.BoundaryMap
    rough_domain: dl.SubDomain
    rough_boundary: dl.BoundaryMesh
    function_space: dl.FunctionSpace
    boundary_conditions: List[dl.DirichletBC]
    
    
    @staticmethod
    def from_mesh(mesh: dl.Mesh, rough_domain: dl.SubDomain, rough_boundary: np.ndarray, 
                  function_space: dl.FunctionSpace, boundary_conditions: List[dl.DirichletBC]):
        srb, srb_tree, srb_map = boundary_bbox_tree(mesh, rough_domain)
        return StokesData(mesh, srb, srb_tree, srb_map, rough_domain, rough_boundary, function_space, boundary_conditions)
    

    def get_micro(self, center, normal, pad=None, **geomargs):
        """Find closest point on boundary and normal to boundary at that point, 
        then construct a micro geometry from that point and normal with parameters specified in geomargs,
        and return the geometry and the feasible domain (in parameter space [0,2pi] that is covered by the mesh.
        """
        fine_boundary_mesh = self.rough_boundary
        tangent = np.array([-normal[1], normal[0]])
        width = geomargs.get('width', 1.0)
        height = geomargs.get('height', 1.0)
        if pad is None:
            pad = 2 * width
        corner_l = center - width * tangent/2
        corner_r = center + width * tangent/2

        bbox = Box(center, normal, width + 2*pad, height+2*pad)
        
        if isinstance(fine_boundary_mesh, dl.Mesh):
            boxMesh = dl.SubMesh(fine_boundary_mesh, bbox)
            indices = order_connected_vertices(boxMesh)
            points = np.array([v[:2] for v in boxMesh.coordinates()])[indices]
        elif isinstance(fine_boundary_mesh, np.ndarray):
            points = fine_boundary_mesh[[bbox.inside(p, True) for p in fine_boundary_mesh]]
            assert len(points) > 0, "No points found in boundary mesh"
        else:
            raise ValueError("fine_boundary_mesh must be either a mesh or a numpy array")
            
        length = curve_length(points)
        t = reparameterize_curve(points, length, corner_l, corner_r, normal, [0, 0.5*np.pi])
        

        f_x = ScaleShiftedBasis.fromSamplesInDomain(t, points[:, 0], [t[0], t[-1]], ChebBasis, dim=len(t)//2)
        df_x = f_x.diff(1)
        ddf_x = f_x.diff(2)
        
        f_y = ScaleShiftedBasis.fromSamplesInDomain(t, points[:, 1], [t[0], t[-1]], ChebBasis, dim=len(t)//2)
        df_y = f_y.diff(1)
        ddf_y = f_y.diff(2)
        
        
        f = lambda t: f_x(t) + 1j * f_y(t)
        df = lambda t: df_x(t) + 1j * df_y(t)
        ddf = lambda t: ddf_x(t) + 1j * ddf_y(t)
        
        center_c = center[0] + 1j * center[1]
        normal_c = normal[0] + 1j * normal[1]
        
        
        #geom = RoundedMicroGeomGeneric(f, df, ddf, center=center_c, normal=normal_c, simple_func=False, **geomargs)
        geomargs.pop("width")
        geom = RoundedMicroGeomGenericV2(f, df, ddf, center=center_c, normal=normal_c, **geomargs)
        
        ############## Find intersection with smooth rough boundary ################
        boxMesh = dl.SubMesh(self.smoothed_rough_boundary, bbox)
        indices = order_connected_vertices(boxMesh)
        points = np.array([v[:2] for v in boxMesh.coordinates()])[indices]
        feasible_domain = []
        for i in [2, 6]:
            dom = geom.dom[i:i+2]
            ab = geom.eval_param(t=np.array(geom.dom[i:i+2]))
            a, b = np.array([ab[0].real, ab[0].imag]), np.array([ab[1].real, ab[1].imag])
            print("Plot line 146 in stokes_fenics.py", end='\r')
            #geom.plot(plt.gca())
            #plt.plot(points[:, 0], points[:, 1], color='blue')
            #plt.scatter(a[0], a[1], color='red')
            #plt.scatter(b[0], b[1], color='red')
            _, t = find_intersection_on_segment(points, a, b)
            if t is None:
                raise ValueError("Micro problem is poorly dimensioned. Try increasing height.")
            feasible_domain.append(t*dom[1]+(1-t)*dom[0])
        
        return geom, np.array(feasible_domain)
        
        
    def plot(self, ax, **kwargs):
        dl.plot(self.macro_mesh, **kwargs)


@dataclass
class MicroSol:
    alpha: float
    x: np.ndarray    
    info: dict = None
    

@dataclass
class MacroSol:
    u: Callable # Object
    v: Callable # Object
    grad_u: Callable # Object
    grad_v: Callable # Object
    p: Callable # Object
    info: dict = None  # Object
            
    def plot_stream(self, ax, dom, bbox, npts=20, **kwargs):
        x = np.linspace(*bbox[0], npts), 
        y = np.linspace(*bbox[1], npts)
        X, Y = np.meshgrid(x,y)
        
        btree = dom.bounding_box_tree()
        inside = lambda x, y: len(btree.compute_entity_collisions(dl.Point(x, y)))>=1
        #help(btree)
        #U = np.array([self.u(x, y) if dom.inside(dl.Point(x, y)) else np.nan for x, y in zip(X.flatten(), Y.flatten())]).reshape((npts,npts))
        #V = np.array([self.v(x, y) if dom.inside(dl.Point(x, y)) else np.nan for x, y in zip(X.flatten(), Y.flatten())]).reshape((npts,npts))
        U = np.array([self.u(x, y) if inside(x, y) else np.nan for x, y in zip(X.flatten(), Y.flatten())]).reshape((npts,npts))
        V = np.array([self.v(x, y) if inside(x, y) else np.nan for x, y in zip(X.flatten(), Y.flatten())]).reshape((npts,npts))
        ax.streamplot(X, Y, U, V, **kwargs)    
    
 
class StokesMacProb(MacroProblem):
    def __init__(self, stokes_data: StokesData, interp, alpha0, g=None, f=None, lam=0.000):
        """Create Stokes Macro Problem."""
        self.stokes_data = stokes_data
        self.interp = interp
        self.interp_a = None
        
        if g is None:
            g = lambda x, y: 0.
        
        if f is None:
            f = dl.Constant((0,0))
    
        W = stokes_data.function_space
        self.U, self.P = W.split()
        self.U = self.U.collapse()
        self.P = self.P.collapse()
        (u, p) = dl.TrialFunctions(W)
        (v, q) = dl.TestFunctions(W)

        G = stokes_data.rough_domain

        # Forces
        V = self.U.sub(0).collapse()
        self.V = V
        self.indicator = get_indicator(G, V)
        self.g = to_fenics_func(g, V) 
        self.alpha_inv = to_fenics_func(lambda x, y: 1./alpha0(x, y), V)
        n = dl.FacetNormal(stokes_data.macro_mesh)
        
        # Slip length
        robin = dl.MeshFunction('size_t', stokes_data.macro_mesh, 1)
        G.mark(robin, 1)
        ds = dl.Measure('ds')[robin]
        
        def proj_n(u): return dl.dot(u, n)
        def grad_n_proj_n(u): return proj_n(dl.grad(proj_n(u)))
        
        def a(u, v, p, q): return dl.inner(dl.grad(u), dl.grad(v))*dl.dx - (p * dl.div(v) + q * dl.div(u))*dl.dx
        def b(u, v, p, q): return (proj_n(v) * (p - grad_n_proj_n(u)) + (proj_n(u) - self.g) * (q - grad_n_proj_n(v))) * ds(1)
        def c(u, v): return (dl.dot(u,v) - proj_n(u) * proj_n(v)) * self.alpha_inv * ds(1)
        def r(u, v): return (proj_n(u)  - self.g) * proj_n(v) * ds(1)

        
        self.F = a(u, v, p, q) + b(u, v, p, q) + c(u, v) + lam * r(u, v)  - dl.dot(f, v)*dl.dx
        self.a = dl.lhs(self.F)
        self.L = dl.rhs(self.F)
    
    def is_solution(self, macro_sol, tol = 1e-5):
        return False
    
    def update(self, micro_sol):
        G = self.stokes_data.rough_domain
        x = np.array([m.x for m in micro_sol])
        a = np.array([m.alpha for m in micro_sol])
        self.interp_a = self.interp(x, a)
        self.alpha_inv.vector()[:] = np.array([1./self.interp_a(x, y)  for x, y in self.V.tabulate_dof_coordinates()]) #if G.inside(np.array([x,y]), True) else 0.


class MacroSolver(Solver):
    def __init__(self, tol=1e-5, **kwargs):
        pass
    
    def can_solve(self, problem) -> bool:
        """Return True if solver can solve this problem."""
        return isinstance(problem, StokesMacProb)
    
    def solve(self, problem: MacroProblem, *args, **kwargs) -> MacroSol:
        """Solve and return solution"""
        w = dl.Function(problem.stokes_data.function_space)
        dl.solve(problem.a == problem.L, w, problem.stokes_data.boundary_conditions)
        uv, p = w.split() # Don't need pressure
        u, v = uv.split()
        gu = dl.project(dl.grad(u))
        gv = dl.project(dl.grad(v))
        
        for f in [u, v, gu, gv]:
            f.set_allow_extrapolation(True)
        
        return MacroSol(u, v, gu, gv, p)
    
    
class StokesMicProb(MicroProblem):
    def __init__(self, setup_data: StokesData, x, projection_op, debug=False, **kwargs):
        self.setup_data = setup_data
        self.center, self.normal = project_to_boundary(dl.Point(x[0], x[1]),\
                                                       setup_data.smoothed_rough_boundary,\
                                                       setup_data.smoothed_rough_boundary_tree, \
                                                       setup_data.smoothed_rough_idx_map, \
                                                       setup_data.macro_mesh)
        if debug:
            plt.scatter(self.center[0], self.center[1], color='red')
            plt.quiver(self.center[0], self.center[1], self.normal[0], self.normal[1])
            w, h, p = kwargs.get('width'), kwargs.get('height'), kwargs.get('pad')
            Box(self.center, self.normal, width=w, height=h).plot(plt.gca())
            Box(self.center, self.normal, width=w+2*p, height=h+2*p).plot(plt.gca())
            
        self.geom, self.feasible_domain = setup_data.get_micro(self.center, self.normal, **kwargs)
        self.width = kwargs.get('width', 1.0)
        self.height = kwargs.get('height', 1.0)
        self.projection_op = projection_op
        self.condition = None
        
    
    def is_solution(self, micro_sol, tol = 1e-5):
        # Requires a solver to check. instead, check convergence.
        return False
    
    def update(self, macro_sol, **kwargs):
        """Given a solution macroSol to the macro problem, read off the solution at points in the micro problem,
        and extrapolate to missing parts of the micro boundary."""      
        self.condition = self.projection_op(macro_sol, self.geom, self.feasible_domain, **kwargs)
        
    
    def set_data(self, condition):
        """Change the boundary condition of the micro problem."""
        self.condition = condition
    
    
    def plot(self, axis, **kwargs):
        """Plot micro domain. relative = True means plotting in true coordinates."""
        self.geom.plot(axis, **kwargs)
        z = self.geom.eval_param(t = self.feasible_domain)
        plt.scatter(z.real, z.imag, color='red')


class MicroSolver(Solver):
    """Solve the micro problem at a specific position. Precompute."""
    def __init__(self, problem, **kwargs):
        self.avg_vec, self.avg = problem.geom.precompute_line_avg(**kwargs)
        self.davg_vec, self.davg = problem.geom.precompute_line_avg(derivative=1, **kwargs)
        
    def can_solve(self, problem):
        return isinstance(problem, StokesMicProb)
        
    def solve(self, problem: StokesMicProb):
        # Log Solve time
        t, _ = problem.geom.grid.get_grid_and_weights()
        c = problem.condition(t)
        print("Bug fix in micro solver, removing - sign", end='\r')
        return MicroSol(-self.avg(c) / self.davg(c), problem.center)
  
        
class StokesHMMProblem(HMMProblem):
    """HMM Stokes problem"""
    def __init__(self, macro, micros, data, *args, **kwargs):
        """Given a macro problem and a micro problem(s), construct a HMM solver."""
        self.data = data
        super(StokesHMMProblem, self).__init__(macro, micros, *args, **kwargs)
        
    def plot(self, ax, npts=1000, **kwargs):
        self.data.plot(ax, **kwargs)
        #self.macro.plot(ax)
        for m in self.micros:
            m.plot(ax, **kwargs)
