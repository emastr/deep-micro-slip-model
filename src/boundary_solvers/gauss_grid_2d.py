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


    

class TrapezGrid:
    def __init__(self, n_pts):
        """
        Discretisation of grid
        :param segments: on the form [t_0, t_1, t_2, ..., t_n]
                         where each segment [t_i, t_{i+1}] is smooth.
        :param tau:
        :param taup:
        :param taupp:
        """
        self.segments = [0, 2*np.pi]
        self.n_pts = n_pts

    def get_grid_and_weights(self):
        grid = np.linspace(0, 2*np.pi, self.n_pts+1)[:-1]
        weights = np.ones_like(grid) * 2*np.pi/self.n_pts
        return grid, weights
            
    def integrate(self, func):
        t, weights = self.get_grid_and_weights()
        return self.integrate_vec(func(t), weights)
    
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
            return (np.imag(dparam / dz) * dens - np.imag(dparam * dz_conj) / (dz_conj ** 2) * dens_conj)/ (1j * np.pi)
           
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
        