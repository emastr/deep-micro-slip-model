import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import matplotlib.pyplot as plt

class GaussLegGrid:
    

    def __init__(self, segments, corners=None):
        """
        Discretisation of grid
        :param segments: on the form [t_0, t_1, t_2, ..., t_n]
                         where each segment [t_i, t_{i+1}] is smooth.
        :param tau:
        :param taup:
        :param taupp:
        """
        self.ABSCISSA = np.array([-0.0950125098376374,
                                   0.0950125098376374,
                                  -0.2816035507792589,
                                   0.2816035507792589,
                                  -0.4580167776572274,
                                   0.4580167776572274,
                                  -0.6178762444026438,
                                   0.6178762444026438,
                                  -0.7554044083550030,
                                   0.7554044083550030,
                                  -0.8656312023878318,
                                   0.8656312023878318,
                                  -0.9445750230732326,
                                   0.9445750230732326,
                                  -0.9894009349916499,
                                   0.9894009349916499])

        self.WEIGHTS = np.array([0.1894506104550685,
                                 0.1894506104550685,
                                 0.1826034150449236,
                                 0.1826034150449236,
                                 0.1691565193950025,
                                 0.1691565193950025,
                                 0.1495959888165767,
                                 0.1495959888165767,
                                 0.1246289712555339,
                                 0.1246289712555339,
                                 0.0951585116824928,
                                 0.0951585116824928,
                                 0.0622535239386479,
                                 0.0622535239386479,
                                 0.0271524594117541,
                                 0.0271524594117541])
        args = np.argsort(self.ABSCISSA)
        self.ABSCISSA = self.ABSCISSA[args]
        self.WEIGHTS = self.WEIGHTS[args]

        self.segments = segments
        if corners is None:
            self.corners = np.ones_like(segments).astype(int) # This can be refined.
        else:
            self.corners = corners

    def get_grid_and_weights(self):
        n_segments = len(self.segments) - 1
        segment_size = len(self.ABSCISSA)
        grid = np.zeros((n_segments * segment_size,))
        weights = np.zeros((n_segments * segment_size,))
        for n in range(n_segments):
            grid[n * segment_size: (n+1)*segment_size] = self.rescale_abscissa(self.segments[n], self.segments[n+1])
            weights[n * segment_size: (n+1)*segment_size] = self.rescale_weights(self.segments[n], self.segments[n+1])

        return grid, weights

    def rescale_abscissa(self, a, b):
        return (a + b)/2 + self.ABSCISSA * (b - a)/2

    def rescale_weights(self, a, b):
        return self.WEIGHTS * (b - a)/2

    def refine_all(self):
        self.segments, _ = GaussLegGrid.refine_segments_at_corners(self.segments, np.ones_like(self.corners))
        old_corners = self.corners
        self.corners = np.zeros_like(self.segments).astype(int)
        for n in range(len(old_corners)):
            self.corners[n * 2] = old_corners[n]

    def refine_all_nply(self, n):
        for i in range(n):
            self.refine_all()

    def refine_corners(self):
        self.segments, self.corners = GaussLegGrid.refine_segments_at_corners(self.segments, self.corners)

    def refine_corners_nply(self, n):
        for i in range(n):
            self.refine_corners()
            
    def integrate(self, func):
        t, weights = self.get_grid_and_weights()
        return self.integrate_vec(func(t), weights)
    
    
    @staticmethod
    def refine_segments_at_corners(segments, corners):
        old_size = len(corners)
        tight_corners = np.roll(corners, 1) * corners
        num_tight_corners = np.sum(tight_corners)
        num_corners = np.sum(corners)

        # No need to refine twice if corners are directly connected (tight) # Remove last element, should be a cycle.
        new_size = old_size + 2 * num_corners - num_tight_corners
        new_corners = np.zeros(new_size)
        new_segments = np.zeros(new_size)

        new_idx = np.arange(0, old_size, 1)# + np.cumsum(2 * self.sharp_corners)
        new_idx = new_idx\
                  + np.roll(np.cumsum(corners), 1)\
                  + np.roll(np.cumsum(corners), 0)\
                  - np.cumsum(tight_corners)\
                  + corners[-1]*corners[0] - 1
        new_idx[0] = 0 # First corner never moves.
        new_corners[new_idx] = corners
        new_segments[new_idx] = segments

        for i in range(1, len(new_corners)-1):
            if new_corners[i] != 1:
                if new_corners[i+1] == 1 or new_corners[i-1] == 1:
                    new_segments[i] = (new_segments[i-1] + new_segments[i+1])/2

        # Ignore the last grid point
        segments = new_segments[:-1]
        corners = new_corners.astype(int)[:-1]

        return segments, corners

    @staticmethod
    def refine_segments_at_corners_nply(self, n, segments, corners):
        for i in range(n):
            segments, corners = GaussLegGrid.refine_segments_at_corners(segments, corners)

    @staticmethod
    def integrate_vec(fvals, weights):
        if len(fvals.shape) == 2:
            #print((fvals * weights).shape)
            return np.sum(fvals * weights, axis=1)
        else:
            #print((fvals * weights).shape)
            return np.sum(fvals * weights, axis=0)

    @staticmethod
    def integrate_func(func, gridpts, weights):
        return GaussLegGrid.integrate_vec(func(gridpts), weights)


    
    
class GaussLegGeometry():
    
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
            
        
    def mask(self, z, tol=0.0001):
        """
          Given complex array z, return mask m such that 
          m[i] = {0 if z[i] is outside, 1 if z[i] inside of domain}
        """     
        
        func = lambda t: np.imag(self.eval_param(derivative=1, t=t) /\
                                 (self.eval_param(derivative=0,t=t) - z.repeat(repeats=len(t), axis=1)) )
        return self.grid.integrate(func) > (0.5 + tol) * (2*np.pi)
        
    def plot(self, ax, showpts=False, shownormals=False, showmask=False, npts=200, maskpts=100, **kwargs):
        t = np.linspace(self.grid.segments[0], self.grid.segments[-1], npts)
        z = self.eval_param(t=t)
        x,y = np.real(z),np.imag(z)
        ax.plot(x, y, **kwargs)
        if showmask:
            self.plot_field(func=self.mask, ax=ax, npts=maskpts, apply_mask=False, cmap="Greys", zorder=-1)
        if shownormals:
            norm = self.normal(t=t)
            nx, ny = np.real(norm), np.imag(norm)
            ax.quiver(x, y, nx, ny)
        if showpts:
            zpts = self.eval_param()
            ax.scatter(np.real(zpts), np.imag(zpts), zorder=2)
            
    def plot_field(self, ax, func, limits=None, npts=200, apply_mask=False, masktol=0.1, **kwargs):
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
        
        ax.pcolormesh(np.real(zgrid), np.imag(zgrid), funcvals, **kwargs)
        
    def plot_stream(self, ax, func, limits=None, npts=200, apply_mask=False, masktol=0.1, **kwargs):
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
    
    def inner(self, f, g):
        return self.grid.integrate(lambda t: np.real(f * np.conjugate(g)))
    
    def net_flow(self, f):
        return self.inner(f, self.eval_param(derivative=1) / 1j)
    
    def zero_flow(self, f):
        normal = self.eval_param(derivative=1) / 1j
        return f - self.net_flow(f) / self.net_flow(normal) * normal
    
    def stokes_matrix(self):
        # Grid
        t, weights = self.grid.get_grid_and_weights()
        n_pts = len(t)
        
        # Parametrisation, derivatives
        param = self.eval_param(derivative=0)
        dparam = self.eval_param(derivative=1)
        ddparam = self.eval_param(derivative=2)
        dparam_conj = np.conjugate(dparam)
        
        # Create matrix
        K = np.eye(2 * n_pts, 2 * n_pts)
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
        return K
    
    
    def stokes_adjoint_matrix(self):
        # Grid
        t, weights = self.grid.get_grid_and_weights()
        n_pts = len(t)
        
        # Parametrisation, derivatives
        param = self.eval_param(derivative=0)
        dparam = self.eval_param(derivative=1)
        ddparam = self.eval_param(derivative=2)
        dparam_conj = np.conjugate(dparam)
        
        # Create matrix
        K = np.eye(2 * n_pts, 2 * n_pts)
        idx = np.arange(0, n_pts)
        for n in range(n_pts):
            m = (idx != n)  # Non-singular indices
            assert sum(m) == n_pts - 1

             # K
            integrand = np.zeros(weights.shape, dtype=np.complex128)
            integrand[m] = np.imag(dparam[m] / (param[m] - param[n])) * weights[n] / np.pi
            integrand[n] = np.imag(ddparam[n] / (2 * dparam[n])) * weights[n] / np.pi

             # K conj
            integrand_conj = np.zeros(weights.shape, dtype=np.complex128)
            integrand_conj[m] = -np.imag(dparam[m] * np.conjugate(param[m] - param[n])) / \
                                  np.conjugate(param[m] - param[n]) ** 2 * weights[n] / np.pi
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
        return K.T
    
    
    @staticmethod
    def get_line_2d(a,b,t0,t1):
        z = lambda t: a + (b-a)*(t-t0)/(t1-t0)
        dz = lambda t: (b-a)*np.ones_like(t)/(t1-t0)
        ddz = lambda t: np.zeros_like(t)
        return z, dz, ddz
    
    
    @staticmethod
    def stitch_functions_1d(funcs, domains):
        """
        Help function to stitch functions on 1d together.
        funcs: list of n functions
        domains: numpy array of n+1 numbers marking the endpoints of the function domaints.
        """
        def stitched_func(t):
            f = np.zeros(t.shape, dtype=np.complex64)
            
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
            
            func1 = GaussLegGeometry.stitch_functions(funcs[0:n1], domains=domains[0:(n1+1)])
            func2 = GaussLegGeometry.stitch_functions(funcs[n1:], domains=domains[n1:])
            return GaussLegGeometry.stitch_functions([func1, func2], domains=domains[[0, n1, len(funcs)]])
    
    
    
    
class StokesDirichletProblem():
    def __init__(self, geometry, condition):
        """
        Wrapper for Stokes Dirichlet Problem.
        geometry: GaussLegGeometry object.
        condition: complex-valued function for boundary condition. should integrate to zero along boundary.
        
        DEPRECATED BELOW
        param_type: can be "t", "xy" or "z" types. 
        if "t", will call condition(t)
        if "z", will call condition(param(t))
        if "xy", will call condition(real(param(t)), imag(param(t)))
        """
        self.geometry = geometry
        self.condition = condition
        self.density = None # Will be updated if solve() is called
     
    def net_flow(self):
        def point_flow(t):
            normal = self.geometry.eval_param(derivative=1, t=t) / 1j
            return np.real(self.condition(t) * np.conjugate(normal))
        return self.geometry.grid.integrate(point_flow)
    
    def solve(self, with_gmres=True, verbose=False, **kwargs):
        if 'tol' in kwargs.keys():
            tol = kwargs['tol']
        else:
            tol = 1e-10
        netflow = self.net_flow()
        if np.abs(netflow) > tol:
            print(f"Warning: Net flow must be zero for unique solution, flow={netflow}")
        
        t, weights = self.geometry.grid.get_grid_and_weights()
        n_pts = len(t)
        
        # Parametrisation, derivatives
        param = self.geometry.eval_param(derivative=0,t=t)
        dparam = self.geometry.eval_param(derivative=1,t=t)
        ddparam = self.geometry.eval_param(derivative=2,t=t)
        dparam_conj = np.conjugate(dparam)
        
        u = self.condition(t)
        
        # Assert boundary integral is zero (or sufficiently close to it)
        b = 1j*u
        sysVec = np.vstack([np.real(b)[:,None], np.imag(b)[:,None]])
        
        # SYSTEM:
        # Ax + Bxc = b -> (Ar + jAi)(xr+jxi) + (Br + jBi)(xr - jxi) = br + jbi
        # (Arxr - Aixi + Brxr + Bixi) + j(Arxi + Aixr + Bixr - Brxi) = br + jbi
        def matvec(v):
            if len(v.shape) == 1:
                v = v[:, None]

            w = v.copy()  # eye * v
            v_real = v[0:n_pts]
            v_imag = v[n_pts:]

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
                integrand_conj[n] = -np.imag(ddparam[n] * dparam_conj[n]) / (2 * dparam_conj[n] ** 2) * weights[n] / np.pi
                
                k11 =  np.real(integrand) + np.real(integrand_conj)#+(np.arange(0,n_pts)==n)
                k21 =  np.imag(integrand) + np.imag(integrand_conj)
                k12 = -np.imag(integrand) + np.imag(integrand_conj)
                k22 =  np.real(integrand) - np.real(integrand_conj)#+(np.arange(0,n_pts)==n)

                # Integrate                
                w[n]         += k11 @ v_real + k12 @ v_imag
                w[n_pts + n] += k21 @ v_real + k22 @ v_imag
            return w
             
        
        if with_gmres:
            linop = LinearOperator(shape=(2*n_pts, 2*n_pts), matvec=matvec)
            omega, info = gmres(linop, sysVec, **kwargs)            
        
        else:
            assert False, "linalg.solve method not implemented"
        #    omega = np.linalg.solve(sysMat, sysVec).flatten()
        #    info = None
        if verbose:
            print(info)
        
        
        self.density = omega[0:n_pts] + 1j * omega[n_pts:2*n_pts]
        
    def solve_adjoint(self, b, with_gmres=True, verbose=False, **kwargs):
        if 'tol' in kwargs.keys():
            tol = kwargs['tol']
        else:
            tol = 1e-10
        netflow = self.net_flow()
        if np.abs(netflow) > tol:
            print(f"Warning: Net flow must be zero for unique solution, flow={netflow}")
        
        
        t, weights = self.geometry.grid.get_grid_and_weights()
        n_pts = len(t)
        
        # Parametrisation, derivatives
        param = self.geometry.eval_param(derivative=0,t=t)
        dparam = self.geometry.eval_param(derivative=1,t=t)
        ddparam = self.geometry.eval_param(derivative=2,t=t)
        dparam_conj = np.conjugate(dparam)
        
        # Assert boundary integral is zero (or sufficiently close to it)
        sysVec = np.vstack([np.real(b)[:,None], np.imag(b)[:,None]])
        def matvec(v):
            if len(v.shape) == 1:
                v = v[:, None]

            w = v.copy()  # eye * v
            v_real = v[0:n_pts]
            v_imag = v[n_pts:]

            idx = np.arange(0, n_pts)
            for n in range(n_pts):
                m = (idx != n)  # Non-singular indices
                assert sum(m) == n_pts - 1

                 # K
                integrand = np.zeros(weights.shape, dtype=np.complex128)
                integrand[m] = np.imag(dparam[n] / (param[n] - param[m])) * weights[m] / np.pi
                integrand[n] = np.imag(ddparam[n] / (2 * dparam[n])) * weights[n] / np.pi

                 # K conj
                integrand_conj = np.zeros(weights.shape, dtype=np.complex128)
                integrand_conj[m] = -np.imag(dparam[n] * np.conjugate(param[n] - param[m])) / \
                                      np.conjugate(param[n] - param[m]) ** 2 * weights[m] / np.pi
                integrand_conj[n] = -np.imag(ddparam[n] * dparam_conj[n]) / (2 * dparam_conj[n] ** 2) * weights[n] / np.pi
                
                k11 =  np.real(integrand) + np.real(integrand_conj)
                k21 =  np.imag(integrand) + np.imag(integrand_conj)
                k12 = -np.imag(integrand) + np.imag(integrand_conj)
                k22 =  np.real(integrand) - np.real(integrand_conj)

                # Integrate                
                w[n]          +=  k11 @ v_real  +  k21 @ v_imag
                w[n_pts + n]  +=  k12 @ v_real  +  k22 @ v_imag
            return w
             
        
        if with_gmres:
            linop = LinearOperator(shape=(2*n_pts, 2*n_pts), matvec=matvec)
            omega, info = gmres(linop, sysVec, **kwargs)  
        
        else:
            assert False, "linalg.solve method not implemented"
        if verbose:
            print(info)
        
        
        return omega[0:n_pts] + 1j * omega[n_pts:2*n_pts]
        
        
    def plot(self, ax, **kwargs):
        t, _ = self.geometry.grid.get_grid_and_weights()
        z = self.geometry.eval_param(derivative=0, t=t)
        x,y = np.real(z),np.imag(z)
        u = self.condition(t)
        self.geometry.plot(ax=ax,**kwargs)
        ax.quiver(x, y, np.real(u), np.imag(u), color='red', label='dirichlet')
        if self.density is not None:
            ax.quiver(x, y, np.real(self.density/1j), np.imag(self.density/1j), color='blue', label='density')
            
    def fourier_density(self, K):
        assert self.density is not None, "Must use solve() first!"
        T = self.geometry.grid.segments[-1] - self.geometry.grid.segments[0]
        freq = np.linspace(-K,K,2*K+1)[:, None]
        basis = lambda t: self.density * np.exp(-1j * freq * t * 2 * np.pi / T)
        return self.geometry.grid.integrate(basis)
    
    def evaluate(self,z):
        """
          Evaluate solution at points z in the complex plane.
          TODO: Implement FMM (Fast Multipole Method) here.
        """
        assert (self.density is not None), "Must solve the system before evaluating"
        def integrand(t):
            param = self.geometry.eval_param(derivative=0,t=t)
            dparam = self.geometry.eval_param(derivative=1,t=t)
            
            dz = param - z
            dz_conj = np.conjugate(dz)
            
            dens = self.density
            dens_conj = np.conjugate(self.density)
            return (np.imag(dparam / dz) * dens - np.imag(dparam * dz_conj) / (dz_conj ** 2) * dens_conj)/ 1j / np.pi
           
        return self.geometry.grid.integrate(integrand)
    
    
    def half_precompute_line_avg(self, a, b, derivative=0, **kwargs):
        """
          Evaluate solution at points z in the complex plane.
          TODO: Implement FMM (Fast Multipole Method) here.
        """
        
        grid = GaussLegGrid([0., 1.])
        grid.refine_all_nply(2)
        
        tau = lambda t: a + (b-a)*t
        dtau = lambda t: (b-a) * np.ones_like(t)
        
        gamma_t = self.geometry.eval_param(derivative=0)[:,None]
        dgamma_t = self.geometry.eval_param(derivative=1)[:,None]
            
        def integrand(s):
            tau_s = tau(s)[None,:]
            dtau_s = dtau(s)[None,:]
            dtau_s_conj = np.conjugate(dtau_s)
            en = dtau_s / np.abs(dtau_s) / 1j
            enc = np.conjugate(en)
            
            abs_dtau = np.abs(dtau_s)
            
            dz = (gamma_t - tau_s)
            dz_conj = np.conjugate(dz)
            
            if derivative == 0:
                k1 = np.imag(dgamma_t / dz) * dtau_s
                k2 = np.imag(dgamma_t * dz_conj) / dz_conj ** 2 * dtau_s_conj
            elif derivative == 1:
                k1 = np.imag(en * dgamma_t / dz ** 2) * dtau_s
                k2 = -(-np.imag(enc * dgamma_t)/ dz_conj ** 2  + \
                       2 * enc * np.imag(dgamma_t * dz_conj) / (dz_conj ** 3)) * dtau_s_conj

            return (k1 + k2) / (np.pi * 1j)
        
        # Remove net flow of f
        f  = grid.integrate(integrand)
        print(f"Flow: {self.geometry.net_flow(f)}, Norm: {self.geometry.inner(f,f)**0.5}")

        def line_avg_func(v, **kwargs):
            self.density = v
            self.solve(**kwargs)
            #print(f"f1 avg: {self.geometry.inner(self.density,f1)} f2 avg: {self.geometry.inner(self.density,f2)}")
            return self.geometry.inner(self.density, f)
        
        return f, line_avg_func
    
    
    def precompute_line_avg(self, a, b, derivative=0, **kwargs):
        """
          Evaluate solution at points z in the complex plane.
          TODO: Implement FMM (Fast Multipole Method) here.
        """
        
        grid = GaussLegGrid([0., 1.])
        grid.refine_all_nply(2)
        
        tau = lambda t: a + (b-a)*t
        dtau = lambda t: (b-a) * np.ones_like(t)
        
        gamma_t = self.geometry.eval_param(derivative=0)[:,None]
        dgamma_t = self.geometry.eval_param(derivative=1)[:,None]
            
        def integrand(s):
            tau_s = tau(s)[None,:]
            dtau_s = dtau(s)[None,:]
            dtau_s_conj = np.conjugate(dtau_s)
            en = dtau_s / np.abs(dtau_s) / 1j
            enc = np.conjugate(en)
            
            abs_dtau = np.abs(dtau_s)
            
            dz = (gamma_t - tau_s)
            dz_conj = np.conjugate(dz)
            
            if derivative == 0:
                k1 = np.imag(dgamma_t / dz) * dtau_s
                k2 = -np.imag(dgamma_t * dz_conj) / dz_conj ** 2 * dtau_s_conj

            elif derivative == 1:
                k1 = np.imag(en * dgamma_t / dz ** 2) * dtau_s
                k2 = -(-np.imag(enc * dgamma_t)/ dz_conj ** 2  + \
                       2 * enc * np.imag(dgamma_t * dz_conj) / (dz_conj ** 3)) * dtau_s_conj

            return (k1 + k2) / (np.pi * 1j)
        
        
        # Remove net flow of f
        f  = grid.integrate(integrand)
        if derivative == 1:
            f  = self.geometry.zero_flow(f)
        f = self.solve_adjoint(f / (-1j), **kwargs)
                            
        # Solve adjoint system for riesz_vector
        #line_avg_func = lambda v: inner(v, riesz_vec)
        
        def line_avg_func(v, **kwargs):
            return self.geometry.inner(v, f)
        
        return f, line_avg_func
    
    
    def precompute_line_avg_2(self, a, b, derivative=0, **kwargs):
        """
          Evaluate solution at points z in the complex plane.
          TODO: Implement FMM (Fast Multipole Method) here.
        """
        
        grid = GaussLegGrid([0., 1.])
        grid.refine_all_nply(2)
        
        tau = lambda t: a + (b-a)*t
        dtau = lambda t: (b-a) * np.ones_like(t)
        
        gamma_t = self.geometry.eval_param(derivative=0)[:,None]
        dgamma_t = self.geometry.eval_param(derivative=1)[:,None]
            
        def integrand(s):
            tau_s = tau(s)[None,:]
            dtau_s = dtau(s)[None,:]
            dtau_s_conj = np.conjugate(dtau_s)
            en = dtau_s / np.abs(dtau_s) / 1j
            enc = np.conjugate(en)
            
            abs_dtau = np.abs(dtau_s)
            
            dz = (gamma_t - tau_s)
            dz_conj = np.conjugate(dz)
            
            if derivative == 0:
                k1 = np.imag(dgamma_t / dz) * dtau_s
                k2 = -np.imag(dgamma_t * dz_conj) / dz_conj ** 2 * dtau_s_conj

            elif derivative == 1:
                k1 = np.imag(en * dgamma_t / dz ** 2) * dtau_s
                k2 = -(-np.imag(enc * dgamma_t)/ dz_conj ** 2  + \
                       2 * enc * np.imag(dgamma_t * dz_conj) / (dz_conj ** 3)) * dtau_s_conj

            return (k1 + k2) / (np.pi * 1j)
        
        # Remove net flow of f
        f  = grid.integrate(integrand) / (-1j)
        
        # Divide out the kernel
        def coimage(A):
            A = np.atleast_2d(A)
            _, _, vh = np.linalg.svd(A)
            ns = vh[:-1].conj().T
            return ns
        
        # Solve system by projecting onto range and removing kernel.
        Kt = self.geometry.stokes_adjoint_matrix()
        V = coimage(Kt)
        U = Kt @ V
        B = V.T @ Kt.T @ Kt @ V
        vec = V @ np.linalg.solve(B, U.T @ np.hstack([np.real(f), np.imag(f)])[:,None])
        n = len(f)
        f = (vec[:n] + 1j * vec[n:]).flatten()
                            
        # Solve adjoint system for riesz_vector
        #line_avg_func = lambda v: inner(v, riesz_vec)
        
        def line_avg_func(v, **kwargs):
            return self.geometry.inner(v, f)
        
        return f, line_avg_func
        
    
    def fourier_flow_correction(self, K, regpar):
        """Correct the total flow across the boundary so that it is zero, using 2K+1 fourier modes.
        Return list of coefficients"""
        
        
        # Flow of different fourier bases.
        T = self.geometry.grid.segments[-1] - self.geometry.grid.segments[0]
        tangent = self.geometry.param[1]
        fflow = T * self.fourier_tfm(tangent, K).T # Flow accross boundary of each fourier basis.
        
        # Create minimisation problem.
        a = np.hstack([np.imag(fflow), -np.real(fflow)])
        b = -self.net_flow()
        d = (1+np.abs(np.linspace(-K,K,2*K+1)))**regpar
        d = np.hstack([d,d])
        
        # Create correction, return coefficients
        coef = self.constrained_min(d, a, b)
        coef = coef[0:(2*K+1)] + 1j * coef[(2*K+1):] # Turn into complex valued.
        #T = self.geometry.grid.segments[-1] - self.geometry.grid.segments[0]
        #self.condition = lambda t: self.condition(t) + \
                                   #sum([coef[k + K] * np.exp(1j * t * 2*np.pi/T) for k in range(-K, K+1)])
        return coef # Return coefficients.
        
    def fourier_tfm(self, func, K):
        """Fourier transform along boundary."""
        T = self.geometry.grid.segments[-1] - self.geometry.grid.segments[0]
        freq = np.linspace(-K,K,2*K+1)[:, None]
        basis = lambda t: func(t) * np.exp(-1j * freq * t * 2 * np.pi / T) / T
        return self.geometry.grid.integrate(basis)
    
    def inv_fourier_tfm(self, coefs, K):
        T = self.geometry.grid.segments[-1] - self.geometry.grid.segments[0]
        freq = np.linspace(-K,K,2*K+1)[:, None]
        return lambda t: sum([coefs[k] * np.exp(1j * freq[k] * t * 2 * np.pi / T) for k in range(2*K+1)])
    
    def get_matvec():
        return 0
    
    def get_mat():
        return 0
    
    @staticmethod
    def constrained_min(d, a, b, tol=1e-6):
        """Minimise x'diag(d) x subject to a'x = b.
        D is psd matrix, a vector and b constant.
        """
        n = len(a)
        pivot = None
        for i in range(n):
            if abs(a[i]) > tol:
                pivot = i
                break
        
        assert pivot is not None, "a must be nonzero"
        
        x = np.zeros((n,))
        nonpivot = np.arange(0,n)!= pivot
        
        # Divide the elements of a into a[p] and a[not p]
        ap = a[pivot]
        a = a[nonpivot][:,None]/ap # also turn into column vector.
        b = b/ap
        
        # Calculate x.
        x[nonpivot] = np.linalg.solve(np.diag(d[nonpivot]) + d[pivot]*a @ a.T, d[pivot]*b*a).flatten()               
        x[pivot] = b - x[nonpivot] @ a
        
        return x
        