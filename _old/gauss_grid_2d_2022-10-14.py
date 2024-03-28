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
        if corners is None:
            return segments, corners
        
        old_size = len(corners)
        tight_corners = np.roll(corners, 1) * corners
        num_tight_corners = np.sum(tight_corners)
        num_corners = np.sum(corners)
        
        # If no corners, do nothing (hotfix).
        if num_corners == 0:
            return segments, corners
        
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
        
        # Create correction matrix;
        z = self.get_inner_pt()
        z_conj = np.conjugate(z)
        dz = param - z
        dz_conj = np.conjugate(dz)
        
        # Create matrix
        K = np.eye(2 * n_pts, 2 * n_pts)
        idx = np.arange(0, n_pts)
        
        C = np.zeros_like(K)       
        corr_integrand =  dparam / dz ** 2 * weights
        corr_integrand_conj = np.conjugate(corr_integrand)
        
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
            
            
            #coef = -1/(dz_conj[n]*2*np.pi*1j)
            coef = (1/dz[n] - 1/dz_conj[n] + dz[n]/dz_conj[n]**2) / (2 * np.pi * 1j) * weights[n]
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
        
        
        return omega[0:N//2] + 1j * omega[N//2:]
    
    
    def precompute_line_avg(self, a, b, derivative=0, **kwargs):
        """
          Evaluate solution at points z in the complex plane.
          TODO: Implement FMM (Fast Multipole Method) here.
        """
        
        f, _ = self.line_eval_adjoint(a,b,derivative=derivative, **kwargs)
        f  = self.solve_adjoint(f, **kwargs)
        
        def line_avg_func(v, **kwargs):
            return self.inner(v, f)
        
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
        
        # Tangent vector
        da = b-a      
        e = da / np.abs(da) / 1j
       
        g = self.eval_param()
        dg = self.eval_param(derivative=1)
        ga = g - a
        gb = g - b
        log_ba = log(gb / ga) / da
        ga_gb = ga * gb
        
        if derivative==0:
            f = (-da * im(dg * log_ba) + im(c(da) * dg) * c(log_ba) + im(ga * c(da)) * c(dg) / c(ga_gb)) / np.pi
        elif derivative == 1:
            f = -(- da * im(dg * e  / ga_gb) 
                  - im(c(e) * dg) * c(da / ga_gb) \
                  + 2*c(e) *im(dg * c(da)) / c(ga_gb)\
                  - im(ga * c(da)) * c(e * dg * (ga + gb) / ga_gb ** 2)) / np.pi 
        
        def line_avg_func(v, **kwargs):
            return self.inner(v, f)
        
        # Remove net flow of f
        return f, line_avg_func
    
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
        assert len(funcs)+1 == len(domains), "Mismatch, len(funcs) != len(domains)-1."
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
        K = self.geometry.stokes_matrix()
        matvec = lambda v: K @ v
        
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
        