import numpy as np
from boundary_solvers.gauss_grid_2d import *
import torch

class ProblemData():
    def __init__(self, bound_coef, u_coef, density_coef=None):
        
        self.bound_coef = bound_coef
        self.u_coef = u_coef
        self.density_coef = density_coef
    
    def get_param(self):
        return [self.get_bound(deriv=0), self.get_bound(deriv=1), self.get_bound(deriv=2)]
    
    def get_bound(self, deriv=0):
        return self.fourier_series(self.bound_coef, deriv)
    
    def get_u(self, deriv=0):
        return self.fourier_series(self.u_coef, deriv)
    
    def get_density(self, deriv=0):
        return self.fourier_series(self.density_coef, deriv)
    
    def to_problem(self, **kwargs):
        problem = self.problemdata_to_problem(self, **kwargs)
        return problem
   
    
    def set_density(self, n=3, tol=1e-14, K_density=40, **kwargs):
        problem = self.to_problem(n=n)
        problem.solve(tol=tol, **kwargs)
        self.density_coef = problem.fourier_tfm(func=lambda t: problem.density, K=K_density)
        
        # DEBUGGING
        #dens = problem.inv_fourier_tfm(self.density_coef,K_density)
        #t, _ = problem.geometry.grid.get_grid_and_weights()
        #err = problem.density - dens(t)
        #print(err)
        
    def save(self, name, path):
        state_dict = {"boundary": self. bound_coef,
                      "condition": self.u_coef,
                      "density": self.density_coef}
        
        if not (path[-1] == '\\') or (path[-1] == '/'):
                path += '/'
        torch.save(state_dict, path + name)
        
        
    @staticmethod
    def load(name, path):
        state_dict = torch.load(path)[0]
        data = ProblemData(bound_coef=state_dict["boundary"], 
                           u_coef=state_dict["condition"],
                           density_coef=state_dict["density"])
        return data
           
    
    
    @staticmethod
    def random_problemdata(K, 
                           bound_amplitude=0.1, 
                           bound_decay=0.2, 
                           cond_amplitude=0.1,
                           cond_decay=0.2,
                           decay_type="exp",
                           callback=None,
                           allow_intersections=True, allow_net_flow=False, tol=1e-10, refine=3):
        
        bound_coef = ProblemData.random_fourier_coefs(K, amplitude=bound_amplitude, decay=bound_decay, decay_type=decay_type)
        bound_coef[K] = 0
        bound_coef[K-1] = 0.0
        bound_coef[K+1] = 1.0
        u_coef = ProblemData.random_fourier_coefs(K, amplitude=cond_amplitude, decay=cond_decay, decay_type=decay_type)
        data = ProblemData(bound_coef, u_coef)
        problem = data.to_problem(n=refine)
        
        # If net flow should be zero, make sure it is.
        if not allow_net_flow:
            coef = problem.fourier_flow_correction(K=K, regpar=3)
            u_coef += coef
        
        if callback is not None:
            callback()
            
        # Do rejection sampling to avoid self intersections.
        if not allow_intersections:
            if len(problem.geometry.self_intersections(tol=tol)) > 0:
                return ProblemData.random_problemdata(K=K,
                                                      bound_amplitude=bound_amplitude, 
                                                      bound_decay=bound_decay, 
                                                      cond_amplitude=cond_amplitude,
                                                      cond_decay=cond_decay,
                                                      decay_type=decay_type,
                                                      allow_intersections=allow_intersections,
                                                      allow_net_flow=allow_net_flow, 
                                                      tol=tol,
                                                      callback=callback,
                                                      refine=refine)
            else:
                return data
        else:
            return data

    
    @staticmethod
    def random_fourier_coefs(K, amplitude=1.0, decay=1.0, decay_type="exp"):
        """
        decay_type: "exp" or "pow", followed by 
        decay: positive real number.
        """
        freqs = np.linspace(-K,K,2*K+1)
        rand = np.random.randn(2*K+1) + 1j * np.random.randn(2*K+1)
        
        if decay_type == "exp":
            coef = np.exp(-decay * np.abs(freqs)) * rand * amplitude
        elif decay_type == "pow":
            coef = (1+np.abs(freqs))**(-decay) * rand * amplitude
        else:
            assert False, "Illegal decay_type."
        
        return coef
        
    
    @staticmethod
    def problemdata_to_problem(data, n=4):
        """
        Convert ProblemData object to a problem.
        """
        params = data.get_param()
        condition = data.get_u()
        
        grid = GaussLegGrid([0,2*np.pi], corners=None)
        grid.refine_all_nply(n)
        geom = GaussLegGeometry(params, grid)
        problem = StokesDirichletProblem(condition=condition, geometry=geom)
        if data.density_coef is not None:
            t, _ = grid.get_grid_and_weights()
            density = data.get_density()
            problem.density = density(t)
        return problem
    
    @staticmethod 
    def fourier_series(coef, deriv=0):
        K = len(coef)//2
        c = coef
        d = deriv
        return lambda t: sum([(1j * k)**d * np.exp(1j * k * t) * c[k + K] for k in range(-K,K+1)])
    