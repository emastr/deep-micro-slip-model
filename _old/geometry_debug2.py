import numpy as np
import torch
from scipy.sparse.linalg import gmres, LinearOperator
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/emastr/phd/')
from boundary_solvers.gauss_grid_2d import *

def fix_scalar_input(func):
        def _func(x, *args, **kwargs):
            if isinstance(x, int) or isinstance(x, float):
                #print(np.array([x,]))
                return func(np.array([x,]), *args, **kwargs)[0]
            else:
                return func(x, *args, **kwargs)
        return _func
    

class Geometry():
    
    def __init__(self, param, grid):
        """
        Geometry: parametrisations of derivatives (complex valued) + grid.
        param: list of functions such that param[n](x) is the n:th derivative of the parametrisation.
        """
        
        self.diff_degree = len(param)-1 # Highest available derivative
        self.param = param              # Parameterisation
        self.grid = grid
        
        t, w = grid.get_grid_and_weights()
        z = param[0](t)
        x,y = np.real(z), np.imag(z)
        self.limits = [min(x), max(x), min(y), max(y)]
    
    
    def eval_param(self, derivative=0, t=None):
        """
        Evaluate parameterisation in derivative [i]
        """
        assert (self.diff_degree >= derivative), "Must provide higher order derivatives of boundary parameterisation."
        if t is None:
            t, _ = self.grid.get_grid_and_weights()
        return self.param[derivative](t)
    
    
    def normal(self, t=None):
        """
        Evaluate normal direction of boundary. If parameterisation is counter-clockwise, the normal points outwards.
        """
        return self.eval_param(derivative=1, t=t)/1j
            
        
    def mask(self, z, tol=0.0001, eps=0.0):
        """
          Given complex array z, return mask m such that 
          m[i] = {0 if z[i] is outside, 1 if z[i] inside of domain}
        """     
        
        tau = lambda t: self.eval_param(derivative=0, t=t)
        dtau = lambda t: self.eval_param(derivative=1, t=t)
        ddtau = lambda t: self.eval_param(derivative=2, t=t)
        
        n = lambda t: 1j * eps * dtau(t) / np.abs(dtau(t))
        dn = lambda t: 1j * eps * (ddtau(t)/np.abs(dtau(t)) \
                                   - dtau(t)*np.real(ddtau(t)*np.conjugate(dtau(t))) / np.abs(dtau(t))**3)
        
        func = lambda t: np.imag((dtau(t) - dn(t)) /(tau(t)-n(t) - z.repeat(repeats=len(t), axis=1)))
        return self.grid.integrate(func) > (0.5 + tol) * (2*np.pi)
        
    def plot(self, ax, showpts=False, shownormals=False, showmask=False, npts=200, maskpts=100, **kwargs):
        """Plot geometry"""
        t = np.linspace(self.grid.segments[0], self.grid.segments[-1], npts)
        z = self.eval_param(t=t)
        x,y = np.real(z),np.imag(z)
        ax.plot(x, y, **kwargs)
        if showmask:
            self.plot_field(func=self.mask, ax=ax, npts=maskpts, apply_mask=False, cmap="Greys", zorder=-1)
        if shownormals:
            norm = self.normal(t=t)
            norm = norm / np.abs(norm)
            nx, ny = np.real(norm), np.imag(norm)
            ax.quiver(x, y, nx, ny)
        if showpts:
            zpts = self.eval_param()
            ax.scatter(np.real(zpts), np.imag(zpts), zorder=2)
            
    def plot_field(self, ax, func, limits=None, npts=200, apply_mask=False, masktol=0.1, eps=0.0, **kwargs):
        """Plot field over geometry"""
        if limits == None:
            limits = self.limits
        xmin,xmax,ymin,ymax = limits[0],limits[1],limits[2],limits[3]
        
        xgrid = np.linspace(xmin, xmax, npts)[:, None]
        ygrid = np.linspace(ymin, ymax, npts)[None, :]
        zgrid = xgrid + 1j*ygrid
        
        funcvals = func(zgrid.flatten()[:,None]).reshape((npts,npts))
        if apply_mask:
            mask = self.mask(zgrid.flatten()[:,None],tol=masktol, eps=eps).reshape((npts,npts))
            funcvals = np.where(mask, funcvals, np.nan)
        
        ax.pcolormesh(np.real(zgrid), np.imag(zgrid), funcvals, **kwargs)
        
    def plot_field_on_boundary(self, ax, func, npts=200, **kwargs):
        """Plot vector field on the boundary"""
        t = np.linspace(self.grid.segments[0], self.grid.segments[-1], npts)
        z = self.eval_param(t=t)
        x,y = np.real(z),np.imag(z)
        f = func(t)
        fx,fy = np.real(f),np.imag(f)       
        ax.quiver(x,y,fx,fy,**kwargs)
        
    def plot_stream(self, ax, func, limits=None, npts=200, apply_mask=False, masktol=0.1, **kwargs):
        """Plot stream function (complex-valued)"""
        if limits == None:
            limits = self.limits
        xmin,xmax,ymin,ymax = limits[0],limits[1],limits[2],limits[3]
        
        xgrid = np.linspace(xmin, xmax, npts)[:, None]
        ygrid = np.linspace(ymin, ymax, npts)[None, :]
        zgrid = xgrid + 1j*ygrid
        
        funcvals = func(zgrid.flatten()[:,None]).reshape((npts,npts))
        if apply_mask:
            mask = self.mask(zgrid.flatten()[:,None],tol=masktol).reshape((npts,npts))
            funcvals = np.where(mask, funcvals, np.nan)
        
        ax.streamplot(np.real(zgrid).T, np.imag(zgrid).T, np.real(funcvals).T, np.imag(funcvals).T, **kwargs)
    
    def self_intersections(self, tol=1e-3):
        """
        Returns list of approximate points where the boundary self-intersects
        """
        z = self.eval_param() # segment positions.
        n = len(z) # number of segments
        intersections = []
        for i in range(n):
            for j in range(i+1,n):
                # line i:  zi(t) = z[i] + (z[i+1] - z[i])*t,  t in (0,1).
                # line j:  zj(s) = z[j+1] + dz[j]*s,  s in (0,1)
                # Want to find s,t such that zi(t) = zj(s)
                dir_i = z[(i+1)%n] - z[i] # mod n to wrap around to the start.
                dir_j = z[(j+1)%n] - z[j] 
                
                A = np.array([[np.real(dir_i), -np.real(dir_j)],
                              [np.imag(dir_i), -np.imag(dir_j)]])
                
                b = np.array([np.real(z[j] - z[i]),
                              np.imag(z[j] - z[i])])
                if np.abs(np.linalg.det(A)) > tol:
                    par = np.linalg.solve(A, b)

                    # If they intersect, return segment coordinate.
                    if (par < 1-tol).all() and (par > tol).all():
                        intersections.append(z[i] + par[0]*dir_i)
                    
        return intersections 
    
    def integrate_vol(self, func, npts=100, **kwargs):
        """Integrate over domain"""
        limits = self.limits
        xmin,xmax,ymin,ymax = limits[0],limits[1],limits[2],limits[3]
        xgrid = np.linspace(xmin, xmax, npts)[:, None]
        ygrid = np.linspace(ymin, ymax, npts)[None, :]
        zgrid = xgrid + 1j*ygrid
        z = zgrid.flatten()[:,None]
        fval = func(z).flatten()
        mask = self.mask(z, **kwargs).flatten()
        val =  np.sum(np.where(mask, fval, 0.)) / np.sum(mask) 
        return val
    
    def inner_vol(self, f, g, **kwargs):
        """Inner product on interior."""
        return self.integrate_vol(lambda z: np.real(f(z) * np.conjugate(g(z))), **kwargs)
    
    def inner(self, f, g):
        """Inner product on boundary. Not weighted by arc length, so depends on parameterisation."""
        return self.grid.integrate(lambda t: np.real(f * np.conjugate(g)) * np.abs(self.eval_param(derivative=1)))
    
    def net_flow(self, f):
        """Net flow across boundary"""
        n = self.eval_param(derivative=1) / 1j
        n = n / np.abs(n)
        return self.inner(f, n)
    
    def zero_flow(self, f):
        n = self.eval_param(derivative=1) / 1j
        return f - self.net_flow(f) / self.net_flow(n) * n
    
    def stokes_matrix(self):
        # Grid
        t, weights = self.grid.get_grid_and_weights()
        n_pts = len(t)
        
        # Parametrisation, derivatives
        param = self.eval_param(derivative=0)
        dparam = self.eval_param(derivative=1)
        ddparam = self.eval_param(derivative=2)
        
        param_conj = np.conjugate(param)
        dparam_conj = np.conjugate(dparam)
        
        # Create matrix
        K = np.eye(2 * n_pts, 2 * n_pts)
        
        # Create correction matrix;
        z = self.get_inner_pt()
        z_conj = np.conjugate(z)
        dz = param - z
        dz_conj = np.conjugate(dz)
        
        C = np.zeros_like(K)       
        corr_integrand =  dparam / dz ** 2 * weights
        corr_integrand_conj = np.conjugate(corr_integrand)
        
        
        idx = np.arange(0, n_pts)
        for n in range(n_pts):
            m = (idx != n)  # Non-singular indices
            assert sum(m) == n_pts - 1

             # K
            integrand = np.zeros(weights.shape, dtype=np.complex128)
            integrand[m] = np.imag(dparam[m] / (param[m] - param[n])) * weights[m] / np.pi
            integrand[n] = np.imag(ddparam[n] / (2 * dparam[n])) * weights[n] / np.pi

             # K conj
            integrand_conj = np.zeros(weights.shape, dtype=np.complex128)
            integrand_conj[m] = -np.imag(dparam[m] * np.conjugate(param[m] - param[n])) / \
                                np.conjugate(param[m] - param[n]) ** 2 * weights[m] / np.pi
            integrand_conj[n] = -np.imag(ddparam[n] * dparam_conj[n]) /\
                                (2 * dparam_conj[n] ** 2) * weights[n] / np.pi
            
            # Assemble
            k11 =  np.real(integrand) + np.real(integrand_conj)
            k21 =  np.imag(integrand) + np.imag(integrand_conj)
            k12 = -np.imag(integrand) + np.imag(integrand_conj)
            k22 =  np.real(integrand) - np.real(integrand_conj)
            K[n, :n_pts] += k11
            K[n, n_pts:] += k12
            K[n_pts + n, :n_pts] += k21
            K[n_pts + n, n_pts:] += k22
            
            
            # C      
            coef = -(1/dz[n] - 1/dz_conj[n] + dz[n]/dz_conj[n]**2) / (2 * np.pi * 1j)
            integrand =  corr_integrand * coef
            integrand_conj = corr_integrand_conj * coef
            C[n, :n_pts]         =  np.real(integrand)  + np.real(integrand_conj)
            C[n, n_pts:]         = -np.imag(integrand) + np.imag(integrand_conj)
            C[n_pts + n, :n_pts] =  np.imag(integrand) + np.imag(integrand_conj)
            C[n_pts + n, n_pts:] =  np.real(integrand) - np.real(integrand_conj)     
        return K + C
    
    def stokes_correction_matrix(self):
         # Grid
        t, weights = self.grid.get_grid_and_weights()
        n_pts = len(t)

        # Parametrisation, derivatives
        param = self.eval_param(derivative=0)
        dparam = self.eval_param(derivative=1)
        ddparam = self.eval_param(derivative=2)

        param_conj = np.conjugate(param)
        dparam_conj = np.conjugate(dparam)

        # Create correction matrix;
        z = self.get_inner_pt()
        z_conj = np.conjugate(z)
        C = np.zeros((2*n_pts, 2*n_pts))
        dz = param - z
        dz_conj = np.conjugate(dz)
        
        idx = np.arange(0, n_pts)

        corr_integrand =  dparam / dz ** 2 * weights
        corr_integrand_conj = np.conjugate(corr_integrand)

        for n in range(n_pts):
            m = (idx != n)  # Non-singular indices
            assert sum(m) == n_pts - 1

            # C
            #coef = -1/(dz_conj[n]*2*np.pi*1j)
            coef = (1/dz[n] - 1/dz_conj[n] + dz[n]/dz_conj[n]**2) / (2 * np.pi * 1j)
            integrand =  corr_integrand * coef
            integrand_conj = corr_integrand_conj * coef
            C[n, :n_pts]         =  np.real(integrand)  + np.real(integrand_conj)
            C[n, n_pts:]         =  -np.imag(integrand) + np.imag(integrand_conj)
            C[n_pts + n, :n_pts] =  np.imag(integrand) + np.imag(integrand_conj)
            C[n_pts + n, n_pts:] =  np.real(integrand) - np.real(integrand_conj)            
        return C
    
    
    def get_inner_pt(self):
        a = self.eval_param(t=np.array([0]))[0]
        b = self.eval_param(t=np.array([np.pi]))[0]
        c = (a + b)/2
        while self.mask(np.array([c])[:,None])[0] == False:
            b = c
            c = (a+b)/2
        return c
    
    def stokes_adjoint_matrix(self):
        # Grid
        t, weights = self.grid.get_grid_and_weights()
        n_pts = len(t)
        
        # Parametrisation, derivatives
        param = self.eval_param(derivative=0)
        dparam = self.eval_param(derivative=1)
        ddparam = self.eval_param(derivative=2)
        dparam_conj = np.conjugate(dparam)
        abs_dparam = np.abs(dparam)
        
        # Create correction matrix;
        z = self.get_inner_pt()
        z_conj = np.conjugate(z)
        dz = param - z
        dz_conj = np.conjugate(dz)
        
        # Create matrix
        K = np.eye(2 * n_pts, 2 * n_pts)
        idx = np.arange(0, n_pts)
        
        C = np.zeros_like(K)       
        corr_integrand =  dparam / dz ** 2 / abs_dparam
        corr_integrand_conj = np.conjugate(corr_integrand)
        
        for n in range(n_pts):
            m = (idx != n)  # Non-singular indices
            assert sum(m) == n_pts - 1

             # K
            integrand = np.zeros(weights.shape, dtype=np.complex128)
            integrand[m] = np.imag(dparam[m] / (param[m] - param[n])) * weights[n] / np.pi * abs_dparam[n] / abs_dparam[m]
            integrand[n] = np.imag(ddparam[n] / (2 * dparam[n])) * weights[n] / np.pi

             # K conj
            integrand_conj = np.zeros(weights.shape, dtype=np.complex128)
            integrand_conj[m] = -np.imag(dparam[m] * np.conjugate(param[m] - param[n])) / \
                                  np.conjugate(param[m] - param[n]) ** 2 * weights[n] / np.pi * abs_dparam[n] / abs_dparam[m]
            integrand_conj[n] = -np.imag(ddparam[n] * dparam_conj[n]) / (2 * dparam_conj[n] ** 2) * weights[n] / np.pi

            k11 =  np.real(integrand) + np.real(integrand_conj)
            k21 =  np.imag(integrand) + np.imag(integrand_conj)
            k12 = -np.imag(integrand) + np.imag(integrand_conj)
            k22 =  np.real(integrand) - np.real(integrand_conj)

            # Integrate                
            K[n, :n_pts] += k11
            K[n, n_pts:] += k12
            K[n_pts + n, :n_pts] += k21
            K[n_pts + n, n_pts:] += k22
            
            
            #coef = -1/(dz_conj[n]*2*np.pi*1j)
            coef = (1/dz[n] - 1/dz_conj[n] + dz[n]/dz_conj[n]**2) / (2 * np.pi * 1j) * weights[n] * abs_dparam[n]
            integrand =  corr_integrand * coef
            integrand_conj = corr_integrand_conj * coef
            C[n, :n_pts]         =  np.real(integrand)  + np.real(integrand_conj)
            C[n, n_pts:]         = -np.imag(integrand) + np.imag(integrand_conj)
            C[n_pts + n, :n_pts] =  np.imag(integrand) + np.imag(integrand_conj)
            C[n_pts + n, n_pts:] =  np.real(integrand) - np.real(integrand_conj) 
            
        return K.T + C.T
    
    
    def solve_adjoint(self, b, with_gmres=True, verbose=False, **kwargs):
        if 'tol' in kwargs.keys():
            tol = kwargs['tol']
        else:
            tol = 1e-10
        
        #abs_dg = np.abs(self.eval_param(derivative=1))
        #b = b
        # Divide out the kernel
        Kt = self.stokes_adjoint_matrix()
        
        # System + projection
        matvec = lambda x: Kt @ x
        sysVec = np.vstack([np.real(b)[:,None], np.imag(b)[:,None]])
        N = len(sysVec)
        
        if with_gmres:
            linop = LinearOperator(shape=(N, N), matvec=matvec)
            omega, info = gmres(linop, sysVec, **kwargs)
            
        else:
            assert False, "linalg.solve method not implemented"
        if verbose:
            print(f"Converged in {info} iterations")
        
        
        return (omega[0:N//2] + 1j * omega[N//2:])
    
    
    def precompute_line_avg(self, a, b, derivative=0, **kwargs):
        """
          Evaluate solution at points z in the complex plane.
          TODO: Implement FMM (Fast Multipole Method) here.
        """
        
        f, _ = self.line_eval_adjoint(a, b, derivative=derivative, **kwargs)
        f  = self.solve_adjoint(f, **kwargs)
        
        def line_avg_func(v, **kwargs):
            return self.inner(1j * v, f)
        
        return f, line_avg_func
    
    
    def line_eval_adjoint(self, a, b, derivative=0, **kwargs):
        """
          Evaluate solution at points z in the complex plane.
          TODO: Implement FMM (Fast Multipole Method) here.
        """

        
        ################### ANALYTIC FORMULA, specific to stokes kernel #################
        # Select branch? 
        log = lambda z: np.log(np.abs(z)) + 1j * np.angle(z)
        
        # Conjugate operator
        c = lambda z: np.conjugate(z)
        im = lambda z: np.imag(z)
        re = lambda z: np.real(z)
        
        # Tangent vector
        da = b-a 
        
        # Normal unit vectur
        en = da / np.abs(da) * 1j
        
        
        g = self.eval_param()
        dg = self.eval_param(derivative=1)
        ga = g - a
        gb = g - b
        log_ba = log(gb / ga) / da
        ga_gb = ga * gb
        e = dg / np.abs(dg)
        
        lgrid = GaussLegGrid(segments = np.linspace(0,1,2))
        lpos = lambda t: a + (b-a)*t
        lder = da
        
        t2, _ = self.grid.get_grid_and_weights()
        if derivative==0:
            term1 = dg  * log(np.abs(ga) / np.abs(gb))
            term2 = -c(dg) * (da/c(da)) * log(c(ga / gb))
            term3 = 1j * c(dg) * im(ga * c(gb)) / c(ga) / c(gb)
            f = (term1 + term2 + term3) / np.pi
                
        elif derivative == 1:
                
            f = -(- da * im(dg * en  / ga_gb) 
                  - im(c(en) * dg) * c(da / ga_gb) \
                  + 2*c(en) *im(dg * c(da)) / c(ga_gb)\
                  - im(ga * c(da)) * c(en * dg * (ga + gb) / ga_gb ** 2)) / np.pi 
            
            
            term1 = lambda z: dg * en / (g - z)
            term2 = lambda z: (3 * en * c(dg) + c(en) * dg) / c(g - z) 
            term3 = lambda z: -c(en * dg) * (g - z)/c(g - z)**2
            integ = lambda z: (term1(z) + term2(z) + term3(z))/(2 * np.pi)
            f = integ(b) - integ(a)
            #f = (term1 + term2 + term3) / np.pi
            
            f2 = np.zeros(g.size, np.complex128)
            ter1 = f2.copy()
            ter2 = f2.copy()
            ter3 = f2.copy()
            #t = np.linspace(0,1.,20)
            for i in range(len(f)):
                dl = lambda t: g[i] - lpos(t)
                int1 = lambda t: -(lder) / (1j * np.pi)* im(en * dg[i] / dl(t) ** 2) 
                #int2 = lambda t: -c(lder) / (1j * np.pi)* im(c(g[i] - lpos(t))*dg[i]) / (c(g[i] - lpos(t))**2)
                int2 = lambda t: -c(lder) / (1j * np.pi)* (im(-c(en)*dg[i])/c(dl(t))**2 + \
                                                           2*c(en)*im(c(dl(t))*dg[i]) / c(dl(t))**3)
                
                #int1 = lambda t: lder * (en * dg[i] / dl(t) ** 2 - c(en * dg[i] / dl(t)**2))
                #int2 = lambda t: -c(lder)/c(dl(t))**2 * (c(en) * dg[i] - en * c(dg[i]))
                #int3 = lambda t: 2*c(lder*en) / c(dl(t))**3 * (dg[i] * c(dl(t)) - c(dg[i]) * dl(t))
                
                int1 = lambda t: lder * en * dg[i] / dl(t) ** 2 * 0
                int2 = lambda t: 0*c(lder) / c(dl(t))**2  * (3 * en * c(dg[i]) + c(en) * dg[i]) 
                #int3 = lambda t:  - 2*c(lder*en) *c(dg[i])/ c(dl(t))**3  * dl(t)
                int3 = lambda t:  c(lder)*en*c(dg[i])/ c(dl(t))**2 * 0
                
                #delt2 = ()
                delt1 = -en * dg[i] * (-1/dl(1) + 1/dl(0)) / (2*np.pi)
                delt2 = (3*en *c(dg[i]) + c(en)*dg[i])*c(1/dl(1) - 1/dl(0)) / (2*np.pi)
                delt3 = - c(en*dg[i]) * (dl(1)/c(dl(1))**2 - dl(0)/c(dl(0))**2) / (2 * np.pi) 
                delt = delt3 + delt1 + delt2
                #ter1[i] = lambda t: lder * en * dg / dl(t)**2
                #ter2[i] = lambda t: 
                #int2 = lambda t: -c(dg[i]) / np.pi * re(c(g[i] - lpos(t))*lder) / c(g[i] - lpos(t))**2
                #int2 = lambda t: c(dg[i]) /(2 * np.pi) * (lder / c(lpos(t) - g[i])\
                 #                                    + (lpos(t) - g[i])*c(lder)/ c(lpos(t)-g[i])**2)
                
                
                #ter3i = lambda t: c(dg[i]) /2/np.pi * (-lder / c(lpos(t) - g[i]) + (lpos(t) - g[i])*c(lder)/ c(lpos(t)-g[i])**2)
                
                #ter1[i] = lgrid.integrate(ter1i)
                #ter2[i] = lgrid.integrate(ter2i)
                #ter3[i] = lgrid.integrate(ter3i)
                
                #f2[i] = lgrid.integrate(lambda t: int1(t) + int2(t)+int3(t))
                f2[i] = lgrid.integrate(lambda t: (int1(t) + int2(t)+int3(t))/(2*np.pi)) + delt
                  
            #f =f2
            
        f = f / np.abs(dg)
        
        def line_avg_func(v, **kwargs):
            return self.inner(v, f)
        
        # Remove net flow of f
        return f, line_avg_func
    
    def curve_length(self):
        """Curve length as a function of the parameter: int(|dz(s)|ds, s from 0 to t)"""
        _, w = self.grid.get_grid_and_weights()
        return np.cumsum(np.abs(self.eval_param(derivative=1)) * w)
    
    @staticmethod
    def interp(a, b, ta, tb):   
        """Interpolate points a=[a,da,dda], b=[b,db,ddb] to match 2nd derivatives."""
        ## Right hand side
        rhs = np.array([[a[i], b[i]] for i in range(3)]).flatten()[:, None]

        # System matrix
        deg = 5
        n = np.arange(deg+1)
        n1 = np.maximum(n-1, 0)
        n2 = np.maximum(n-2, 0)

        one = np.ones_like(n)
        T = np.array([ta, tb])[:, None].astype(float)
        A = np.vstack([T ** n, \
                       n * (T ** n1), \
                       n * (n-1) * (T ** n2)])

        c = np.linalg.solve(A, np.real(rhs)) + 1j * np.linalg.solve(A, np.imag(rhs))
        c1 = n[:,None] * c
        c2 = (n * (n-1))[:,None] * c


        z = lambda t: ((t.flatten()[:,None] ** n) @ c).reshape(t.shape)
        dz = lambda t: ((t.flatten()[:,None] ** n1) @ c1).reshape(t.shape)
        ddz = lambda t: ((t.flatten()[:,None] ** n2) @ c2).reshape(t.shape)
        return z, dz, ddz
    
    @staticmethod
    def change_domain(param, dom, new_dom):
        scale = (dom[1]-dom[0]) / (new_dom[1] - new_dom[0])
        shift = (dom[1]*new_dom[0] - new_dom[1]*dom[0]) / (dom[1] - dom[0])
        return Geometry.shift_then_scale(param, shift, scale)
    
    @staticmethod
    def shift_then_scale(param, shift, scale):
        ret = []
        ret.append(lambda x: 1 * param[0](scale*(x-shift)))
        ret.append(lambda x: (scale**1) * param[1](scale*(x-shift)))
        ret.append(lambda x: (scale**2) * param[2](scale*(x-shift)))
        return ret
    
    @staticmethod
    def scale_then_shift_output(param, shift, scale):
        ret = []
        ret.append(lambda x: param[0](x)*scale + shift)
        ret.append(lambda x: param[1](x)*scale)
        ret.append(lambda x: param[2](x)*scale)
        return ret
    
    @staticmethod
    def get_line_2d(a,b,t0,t1):
        z = lambda t: a + (b-a)*(t-t0)/(t1-t0)            
        dz = lambda t: (b-a)*np.ones_like(t)/(t1-t0)
        ddz = lambda t: np.zeros_like(t)
        return z, dz, ddz
    
    @staticmethod
    def get_circle(rad, center, t0, t1, a0):
        scale = 1j * 2 * np.pi / (t1 - t0)
        c = lambda t: rad * np.exp(1j*a0 + (t - t0) * scale)
        z = lambda t: c(t) + center
        dz = lambda t: c(t) * scale
        ddz = lambda t: c(t) * scale ** 2
        return z, dz, ddz
    
    @staticmethod
    def stitch_functions_1d(funcs, domains):
        """
        Help function to stitch functions on 1d together.
        funcs: list of n functions
        domains: numpy array of n+1 numbers marking the endpoints of the function domains.
        """
        dom = domains
        assert len(funcs)+1 == len(dom), "Mismatch, len(funcs) != len(domains)-1."
        
        @fix_scalar_input
        def stitched_func(t):
            f = np.zeros(t.shape, dtype=np.complex64)
            
            if isinstance(t, float) or isinstance(t, int):
                t = np.array([t])
                
            idx = np.clip(np.sum(t[:, None] > dom[None, :], axis=1)-1, 0, len(dom)).astype(int)
            for i in range(len(dom)-1):
                idx_i = (idx == i)
                f[idx_i] = funcs[i](t[idx_i])
            return f
                                                  
        return stitched_func
    
    
    @staticmethod
    def stitch_functions_1d_old(funcs, domains):
        """
        Help function to stitch functions on 1d together.
        funcs: list of n functions
        domains: numpy array of n+1 numbers marking the endpoints of the function domains.
        """
        assert len(funcs)+1 == len(domains), "Mismatch, len(funcs) != len(domains)-1."
        
        @fix_scalar_input
        def stitched_func(t):
            f = np.zeros(t.shape, dtype=np.complex64)
            
            if isinstance(t, float) or isinstance(t, int):
                t = np.array([t])
                
            fidx = 0
            tidx = 0
            for i in range(len(t)):
                # If we arrive at a new panel, switch function
                if (t[i] >= domains[fidx+1]):
                    f[tidx:i] = funcs[fidx](t[tidx:i])
                    tidx = i
                    fidx = fidx + 1
                if fidx == len(funcs)-1:
                    break
            # Arriving at the end, add the last panel.
            f[tidx:] = funcs[fidx](t[tidx:]) #Close the loop
            
            return f
                                                  
        return stitched_func
    
    @staticmethod
    def stitch_functions(funcs, domains):
        # Two functions are easy to stitch.
        if len(funcs)==1:
            return funcs[0]
        
        elif len(funcs)==2:
            def func(t):
                return np.where(t<domains[1], funcs[0](t), funcs[1](t))
            return func
        
        else:
            # Number of  functions to stitch together.
            n1 = len(funcs)//2
            #n2 = len(funcs)-n1
            
            func1 = Geometry.stitch_functions(funcs[0:n1], domains=domains[0:(n1+1)])
            func2 = Geometry.stitch_functions(funcs[n1:], domains=domains[n1:])
            return Geometry.stitch_functions([func1, func2], domains=domains[[0, n1, len(funcs)]])
    
    
    
class ShiftScaledGeom(Geometry):
    def __init__(self, geom, shift, scale):
        self.scale = scale
        self.shift = shift
        param = Geometry.scale_then_shift_output(geom.param, shift, scale)
        super(ShiftScaledGeom, self).__init__(param, geom.grid)
        

class RoundedMicroGeom(Geometry):
    def __init__(self, func, dfunc, ddfunc, width, height, corner_w, line_pos, 
                 center_x=0, shift_y=0, n_refine=1, n_corner_refine=0):
        """
        Rounded microscopic problem. Functions func, dfunc and ddfunc are defined on [0, 0.5pi].
        Domain is encompassed in the bounding box [-w/2-cw, w/2+cw] x [min(func), h+cw].
        In the region [-w/2, -w/2+cw] and [w/2-cw, w/2], we round the corners using a 5th deg poly interp.
        """
        
        self.height = height #Remember this
        eps = corner_w/(0.5*width) * 0.25*np.pi * 1.
        dom = np.array([0,\
                        0.5*np.pi,\
                        0.5*np.pi+eps,\
                        np.pi-eps,\
                        np.pi,\
                        1.5*np.pi,\
                        1.5*np.pi+eps, \
                        2*np.pi-eps,\
                        2*np.pi])
        
        # parameterise func 
        f = lambda t: (func(t) + shift_y)*1j + width * (t / dom[1] - 1/2) + center_x
        df = lambda t: dfunc(t)*1j + width / dom[1]
        ddf = lambda t: ddfunc(t)*1j
        
        # Define segments to stitch
        pts = [f(dom[1]), 
               (center_x + width/2 + corner_w) + shift_y*1j,#f(dom[1])+corner_w * (1 + 1j),
               (center_x + width/2) + (shift_y + height)*1j + corner_w*(1 - 1j),
               (center_x + width/2) + (shift_y + height)*1j,
               (center_x - width/2) + (shift_y + height)*1j,
               (center_x - width/2) + (shift_y + height)*1j + corner_w*(-1-1j),
               (center_x - width/2 - corner_w) + shift_y*1j,#f(dom[0]) + corner_w*(-1 + 1j),
               f(dom[0])]
        
        #plt.scatter([np.real(p) for p in pts], [np.imag(p) for p in pts])
        z, dz, ddz = [f], [df], [ddf]
        
        # Build panels
        for i in range(3):
            a = pts[2*i+1]
            b = pts[2*i+2]
            ta = dom[2*i+2]
            tb = dom[2*i+3]
            r, dr, ddr =  Geometry.get_line_2d(a, b, ta, tb)
            z.append(r)
            dz.append(dr)
            ddz.append(ddr)
            
        # Build smooth segments
        z2, dz2, ddz2 = [], [], []
        for i in range(4):           
            ta = dom[2*i+1]
            tb = dom[(2*i+2)%(len(dom)-1)]
            
            a = [z[i](ta)      , dz[i](ta)      , ddz[i](ta)]
            b = [z[(i+1)%4](tb), dz[(i+1)%4](tb), ddz[(i+1)%4](tb)]
            
            r, dr, ddr = Geometry.interp(a, b, ta, dom[2*i+2])
            
            z2.append(z[i])
            dz2.append(dz[i])
            ddz2.append(ddz[i])
            
            z2.append(r)
            dz2.append(dr)
            ddz2.append(ddr)
            
            
        # Stitch segments together
        z = Geometry.stitch_functions_1d(z2, dom)
        dz = Geometry.stitch_functions_1d(dz2, dom)
        ddz = Geometry.stitch_functions_1d(ddz2, dom)
        
        # Define grid
        grid = GaussLegGrid(dom, np.ones(dom.shape).astype(int))
        grid.refine_all_nply(n_refine)
        grid.refine_corners_nply(n_corner_refine)
        
        # Main properties
        self.width = width
        self.line_left = center_x - 0.45 * width + 1j * (line_pos + shift_y)
        self.line_right = center_x + 0.45 * width + 1j * (line_pos + shift_y)
        self.inner_point = center_x + (height * 0.5 + shift_y) * 1j 
        self.eps = eps
        self.dom = dom
        
        super().__init__([z, dz, ddz], grid)
    
    
    def get_inner_pt(self):
        """Overwrite inner_point to be independent of lower boundary."""
        return self.inner_point
    
    def precompute_line_avg(self, derivative=0, **kwargs):
        """
          Evaluate solution at points z in the complex plane.
          TODO: Implement FMM (Fast Multipole Method) here.
        """
        
        f, _ = self.line_eval_adjoint(derivative=derivative, **kwargs)
        f  = self.solve_adjoint(f, **kwargs)
        
        def line_avg_func(v, **kwargs):
            return self.inner(1j * v, f)
        
        return f, line_avg_func
    
    
    def line_eval_adjoint(self, **kwargs):
        return super().line_eval_adjoint(self.line_left, self.line_right, **kwargs)
    
    def plot(self, ax, show_hline=True, **kwargs):
        super().plot(ax, **kwargs)
        color = kwargs.pop("color", "black")
        if show_hline:
            ax.plot([np.real(self.line_left), np.real(self.line_right)],
                    [np.imag(self.line_left), np.imag(self.line_right)], linewidth=4, color=color)
    
    
    def project(self, g, dg, ddg, N = 13, normdeg=1, reg=0):
        """Given a function g with derivatives dg, ddg defined on the positive half plane,
        extrapolate g such that g = 0 on the part of the domain that corresponds to the lower boundary,
        and that the total flow over the boundary is zero. This function solves a constrained least squares problem
        where the objective function is the sobolev norm of degree "normdeg". """
        from numpy.polynomial.chebyshev import chebvander, chebpts1, chebgauss, chebder, chebval
        
        dom = self.dom
    
        # Some usefull functions
        from_blocks = lambda blocks: np.vstack([np.hstack(row) for row in blocks])
        _x = lambda g: np.real(g)
        _y = lambda g: np.imag(g)

        ns = np.arange(N)
        deg = N-1

        # Interval widths
        l0 = 2*(dom[2] - dom[1])**(-1)
        l1 = 2*(dom[-1] - dom[-2])**(-1)

        grid = GaussLegGrid([-1, 1])
        grid0 = GaussLegGrid([dom[1], dom[2]])
        grid1 = GaussLegGrid([dom[-2], dom[-1]])

        # Gauss Legendre points
        x, w = grid.get_grid_and_weights()
        x0, w0 = grid0.get_grid_and_weights()
        x1, w1 = grid1.get_grid_and_weights()

        n0 = self.normal(t=x0)
        n1 = self.normal(t=x1)

        # Chebyshev matrix
        V = chebvander(x, deg)
        D = [chebder(np.arange(n)==(n-1), m=normdeg) for n in ns]
        D = np.hstack([np.pad(d, (0, N-len(d)))[:, None] for d in D])

        # Matrix for computing the sobolev 1-norm
        #@ V.T @ np.diag(w) @ V
        A = D.T @ D + reg * np.eye(N)
        A = np.kron(np.eye(4), A)

        # Net flow integral
        v0x = (_x(n0) * w0) @ V
        v0y = (_y(n0) * w0) @ V
        v1x = (_x(n1) * w1) @ V
        v1y = (_y(n1) * w1) @ V

        # Evaluation of derivatives up to 2nd order
        # zero deriv
        Bm1 = (-1.) ** ns
        Bp1 = np.ones((N,))
        # First deriv
        dBm1 = ns ** 2 * (-1.) ** (ns-1)
        dBp1 = ns ** 2
        # Second deriv
        ddBm1 = (ns ** 4 - ns**2)/3 * (-1.) ** ns
        ddBp1 = (ns ** 4 - ns**2)/3

        # Stack blocks
        Bpm = from_blocks([[Bm1], [Bp1]])
        dBpm = from_blocks([[dBm1], [dBp1]])
        ddBpm = from_blocks([[ddBm1], [ddBp1]])

        # Expand system for multiple x-and-y components as well as evaluation points
        B = from_blocks([[np.kron(np.eye(4), Bpm)],
                         [np.kron(np.diag([l0, l0, l1, l1]), dBpm)],
                         [np.kron(np.diag([l0**2, l0**2, l1**2, l1**2]), ddBpm)]])

        # Add net row for the net flow
        B = from_blocks([[v0x, v0y, v1x, v1y], [B]])

        ##### Right hand side:
        t = np.array([dom[2], dom[-2]])
        stack_yx = lambda z: np.array([0, _x(z[0]), 0, _y(z[0]), _x(z[1]), 0, _y(z[1]), 0])[:, None]
        b = np.vstack([np.zeros((A.shape[0]+1,1))] + [stack_yx(g) for g in [g(t), dg(t), ddg(t)]])

        ##### FULL SYSTEM

        M = from_blocks([[A, B.T], [B, np.zeros((B.shape[0], B.shape[0]))]])
        c = np.linalg.solve(M, b)

        ###### BUILD FUNCTIONS
        c0x, c0y, c1x, c1y = c[:N], c[N:2*N], c[2*N:3*N], c[3*N:4*N]

        g0 = lambda t: (chebval(t, c0x) + 1j * chebval(t, c0y)).flatten()
        g1 = lambda t: (chebval(t, c1x) + 1j * chebval(t, c1y)).flatten()

        g0 = Geometry.change_domain([g0, g0, g0], [-1, 1], [dom[1], dom[2]])[0]
        g1 = Geometry.change_domain([g1, g1, g1], [-1, 1], [dom[-2], dom[-1]])[0]

        zer = lambda t: np.zeros(t.shape, dtype=np.complex128)
        domain = np.array([0, dom[1], dom[2], dom[-2], dom[-1], 2*np.pi])
        g_extrap = Geometry.stitch_functions_1d([zer, g0, g, g1, zer], domain)

        return g_extrap
    



class GPDomain(RoundedMicroGeom):
    def __init__(self, kernel, shape, scale, num, bound=1, X=None, Y=None, K_inv_dy=None, **kwargs):
        
        self.k, self.dk, self.ddk = self.get_kernel(kernel, scale, shape)
        width = kwargs["width"]
        
        if (X is None):
            X, Y, K_inv_dy = self.sample_bounded_gp(self.k, width, bound, num)
        
        r = width / (0.5 * np.pi)
        dr = -width/2
        y = lambda t: (K_inv_dy.T @ self.k(dr + r*t.flatten(), X)).reshape(t.shape)-bound
        dy = lambda t: (K_inv_dy.T @ self.dk(dr + r*t.flatten(), X)*r).reshape(t.shape)
        ddy = lambda t: (K_inv_dy.T @ self.ddk(dr + r*t.flatten(), X)*r**2).reshape(t.shape)
        
        # Data that is used in function methods.
        self.X = X
        self.Y = Y
        
        # Hidden data
        self.bound = bound
        self.kwargs = kwargs
        self.data = {"kernel": kernel,
                     "shape": shape,
                     "scale": scale,
                     "num": num,
                     "bound": bound,
                     "Y": Y,
                     "X": X,
                     "K_inv_dy": K_inv_dy}
        super().__init__(y, dy, ddy, **kwargs)
    
    def save(self, path):
        data = self.data.copy()
        data.update(self.kwargs)
        torch.save(data, path)
    
    def plot(self, show_samples=True, show_hline=True, **kwargs):
        super().plot(**kwargs)
        color = kwargs.pop("color", "black")
        ax = kwargs.pop("ax")
        if show_hline:
            ax.plot([-0.45*self.width, 0.45*self.width],[self.line_pos, self.line_pos], linewidth=4, color=color)
        if show_samples:
            ax.scatter(self.X, self.Y-self.bound)
            #plt.scatter(self.X, self.K_inv_df)
    
    @staticmethod
    def load(path, **kwargs):
        if type(path) == str:
            data = torch.load(path)
        elif type(path) == dict:
            data = path
        gp_dom = GPDomain(**data)
        return gp_dom
    
    @staticmethod
    def get_kernel(name, scale, shape):
        if name == "exp":
            # Do smth else
            k = lambda x, y: (scale * np.exp(-((x - y) ** 2) / shape))
            dk = lambda x, y: (- 2 / shape * (x - y) * k(x, y))
            ddk = lambda x, y: (- 2 / shape * (1 - 2 / shape * (x - y)**2) * k(x, y))
        else:
            assert False,f"Kernel {name} not implemented"
        return k, dk, ddk
    
    @staticmethod
    def sample_bounded_gp(k, width, bound, num, resample=0):
            X = np.linspace(-0.6*width, 0.6*width, num)[:,None]#np.sort(np.random.uniform(-0.1*width, 1.1*width, num))[:,None]
            K = k(X, X.T)
            Y = np.random.multivariate_normal(np.zeros_like(X).flatten(), K)[:, None]
            K_inv_dy = np.linalg.solve(K, Y)
            
            #print(K_inv_dy)
            if np.max(np.abs(Y)) >= bound:
                print(f"Resample # {resample}", end="\r")
                return GPDomain.sample_bounded_gp(k, width, bound, num, resample+1)
            else:
                return X, Y, K_inv_dy
            
class MacroGeom(Geometry):
    def __init__(self, func, dfunc, ddfunc, width, height, n_refine, n_corner_refine):
        
        dom = np.linspace(0,2*np.pi,5)
        
        # parameterise func 
        f = lambda t: func(t)*1j + width * t / dom[1] - 0.5j*height
        df = lambda t: dfunc(t)*1j + width / dom[1]
        ddf = lambda t: ddfunc(t)*1j
        
        # Define segments to stitch
        pts = [f(dom[1]), width + 0.5 * height*1j, 0.5*height*1j, f(dom[0])]
        z, dz, ddz = [f], [df], [ddf]
        
        for i in range(3):
            r, dr, ddr = Geometry.get_line_2d(pts[i], pts[i+1], dom[i+1], dom[i+2])
            z.append(r)
            dz.append(dr)
            ddz.append(ddr)
        
        # Stitch segments together
        z = Geometry.stitch_functions_1d(z, dom)
        dz = Geometry.stitch_functions_1d(dz, dom)
        ddz = Geometry.stitch_functions_1d(ddz, dom)
        
        # Define grid
        grid = GaussLegGrid(dom, np.ones((5,)).astype(int))
        grid.refine_all_nply(n_refine)
        grid.refine_corners_nply(n_corner_refine)
        
        # Main properties
        self.width = width
        super().__init__([z, dz, ddz], grid)
        
        
class MacroGeomGeneric(Geometry):
    def __init__(self, func, dfunc, ddfunc, dom, n_refine, n_corner_refine):
        
        width = dom[0][1] - dom[0][0]
        
        t_dom = np.linspace(0,2*np.pi,5)
        
        
        # parameterise func 
        f = lambda t: func(t)*1j + width * t / t_dom[1] + dom[0][0] + 1j * dom[1][0]
        df = lambda t: dfunc(t)*1j + width / t_dom[1]
        ddf = lambda t: ddfunc(t)*1j
        
        # Define segments to stitch
        pts = [f(t_dom[1]), dom[0][1] + 1j*dom[1][1], dom[0][0] + 1j*dom[1][1], f(t_dom[0])]
        z, dz, ddz = [f], [df], [ddf]
        
        for i in range(3):
            r, dr, ddr = Geometry.get_line_2d(pts[i], pts[i+1], t_dom[i+1], t_dom[i+2])
            z.append(r)
            dz.append(dr)
            ddz.append(ddr)
        
        # Stitch segments together
        z = Geometry.stitch_functions_1d(z, t_dom)
        dz = Geometry.stitch_functions_1d(dz, t_dom)
        ddz = Geometry.stitch_functions_1d(ddz, t_dom)
        
        # Define grid
        grid = GaussLegGrid(t_dom, np.ones((5,)).astype(int))
        grid.refine_all_nply(n_refine)
        grid.refine_corners_nply(n_corner_refine)
        
        # Main properties
        self.width = width
        super().__init__([z, dz, ddz], grid)
            