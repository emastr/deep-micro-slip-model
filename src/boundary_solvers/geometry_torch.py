import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


class GeomData(Dataset):
    PREDEFINED_INPUTS = {'full-cartesian':  ['x', 'y', 'dx','dy','ddx','ddy','vx','vy', 'dvx', 'dvy'],\
                         'reduced-cartesian':  ['x', 'y','vx','vy', 'dvx', 'dvy'],\
                         'full-natural':  ['x', 'y', 'tx', 'ty', 'c', 'vt', 'vn', 'dvt', 'dvn'],\
                         'equivariant-natural': ['c', 'dist', 'vt', 'vn', 'dvt', 'dvn'],\
                         'equivariant-natural-rad': ['rad', 'dist', 'vt', 'vn', 'dvt', 'dvn'],\
                         'invariant-natural': ['c_norm', 'dist_norm', 'vt', 'vn', 'dvt', 'dvn'],\
                         'FIX:invariant-natural': ['c_norm', 'dist_norm', 'vt', 'vn', 'dvt_norm', 'dvn_norm'],
                         'FIX:invariant-natural-dir': ['c_norm', 'dct_norm', 'dcn_norm', 'vt', 'vn', 'dvt', 'dvn']}
    
    PREDEFINED_OUTPUTS = {'cartesian': ['rx', 'ry', 'drx', 'dry'],\
                          'natural': ['rt', 'rn', 'drt', 'drn'],
                          'FIX:invariant-natural': ['rt', 'rn', 'drt_norm', 'drn_norm']}
    
    def __init__(self, path, input_features, output_features, random_roll=True, device="cuda:0", dtype=torch.double):
        
        self.path = path
        if isinstance(input_features, str):
            self.input_features = self.PREDEFINED_INPUTS[input_features]
        else:
            self.input_features = input_features
            
        if isinstance(output_features, str):
            self.output_features = self.PREDEFINED_OUTPUTS[output_features]
        else:
            self.output_features = output_features
        
        self.random_roll = random_roll
        self.inp, self.out, self.N = self.load_and_tfm(path, device, dtype)
        self.X = concat_dict_entries(subdict(self.inp, self.input_features))
        self.Y = concat_dict_entries(subdict(self.out, self.output_features))
        
        
        super(GeomData, self).__init__()
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        X, Y = self.X[idx], self.Y[idx]
        if self.random_roll:
            roll = np.random.randint(0, X.shape[-1])
            X = torch.roll(X, roll, dims=-1)
            Y = torch.roll(Y, roll, dims=-1)
        return X, Y
    
    def concat_to_tensor(self, idx, inp_features=None, out_features=None):
        X = None if inp_features is None else concat_dict_entries(subsample(subdict(self.inp, inp_features), idx))
        Y = None if out_features is None else concat_dict_entries(subsample(subdict(self.out, out_features), idx))
        return X, Y
    
    @staticmethod
    def load_and_tfm(path, device, dtype=torch.double):
        data = subdict(torch.load(path), ['X', 'Y'])
        data = to_dtype(to_device(data, device), dtype)
        inp, out = unpack_data(data)
                
        inp = GeomData.tfm_inp(inp)
        out = GeomData.tfm_out(out, inp)
        return inp, out, inp['x'].shape[0]

    @staticmethod
    def tfm_out(out, inp):
        out.update(project_to_natural(['r', 'dr'], inp['tx'], inp['ty'], out))
        
        dv_norm = (norm(inp['dvt'])**2 + norm(inp['dvn'])**2)**0.5
        out['drt_norm'] = out['drt'] / dv_norm
        out['drn_norm'] = out['drn'] / dv_norm
        return out

    @staticmethod
    def tfm_inp(inp):
        inp.update(invariant_quantities(inp))
        
        dist, dx, dy = mean_distance(inp)
        inp.update({'dist': dist, 'dcx': dx, 'dcy': dy})
        
        
        # Project to natural coordinates.
        inp.update(project_to_natural(['v', 'dv'], inp['tx'], inp['ty'], inp))
        inp.update(project_to_natural(['dc'], inp['tx'], inp['ty'], inp))
        
        ### Add normalised quantities
        # Normalise dv 
        dv_norm = (norm(inp['dvt'])**2 + norm(inp['dvn'])**2)**0.5
        inp['dvt_norm'] = inp['dvt'] / dv_norm
        inp['dvn_norm'] = inp['dvn'] / dv_norm
        
        dc_norm = (norm(inp['dct'])**2 + norm(inp['dcn'])**2)**0.5
        inp['dct_norm'] = inp['dct'] / dc_norm
        inp['dcn_norm'] = inp['dcn'] / dc_norm
        
        
        # Normalise curvature using monotone function sigmoid(x/2) - 1/2.
        inp['c_norm'] = sigmoid_normalisation(inp['c'] / norm(inp['c']))
        #inp['c_norm'] = inp['c'] / norm(inp['c'])
        inp['dist_norm'] = inp['dist'] / norm(inp['dist'])
        
        inp['rad'] = sigmoid_normalisation(1/inp['c'])
        inp['c'] = sigmoid_normalisation(inp['c'])
        return inp
        

def reparameterize_data(t, dt, data, M=None):
    """Reparameterize the data to be a function of t in [0,1].
    Uses fouier series with 2*k_max+1 terms."""    
    #dtype = torch.complex64

    if M is None:
        M = len(t)
    data_np = data.numpy()
    t_np = t
    f = interp1d(t_np, data_np, axis=-1)
    t_np = np.linspace(0, 2*np.pi, M+1)[:-1]
    data_np = f(t_np)
    
    return torch.from_numpy(data_np)

def reparameterize_dict(t, data, M=None):
    """Reparameterize the data to be a function of t in [0,1].
    Uses fouier series with 2*k_max+1 terms."""    
    #dtype = torch.complex64

    if M is None:
        M = len(t)
    t_np = np.linspace(0, 2*np.pi, M+1)[:-1]
    
    # Add the first point to the end to make it periodic.
    if abs(t[-1]- t[0])%(2*np.pi) > 1e-8:
        t = np.concatenate([[t[-1]-2*np.pi], t, [2*np.pi+t[0]]])
        for key in data.keys():
            data[key] = np.concatenate([data[key][-1:], data[key], data[key][0:1]], axis=-1)
    
        
    data_out = {}
    for key in data.keys():
        f = interp1d(t, data[key], axis=-1)
        data_out[key] = f(t_np)
    
    return data_out

def geometry_to_net_input(geom, Mout, output=False):
    v, _ = geom.line_eval_adjoint(derivative=0, tol=1e-12, maxiter=200, verbose=False)
    dv, _ = geom.line_eval_adjoint(derivative=1, tol=1e-12, maxiter=200, verbose=False)
    t, w = geom.grid.get_grid_and_weights()
    z = geom.eval_param(derivative=0)
    dz = geom.eval_param(derivative=1)
    ddz = geom.eval_param(derivative=2)


    # TODO: Change this to use the grid and weights instead.
    dt = np.zeros_like(t)
    dt[:-1] = t[1:] - t[:-1]
    dt[-1] = 2*np.pi - t[-1]
    l = np.cumsum(np.abs(dz) * dt)
    L = l[-1]
    l = np.roll(l, 1) / L *2*np.pi
    l[0] = 0.
    w = np.ones_like(l) * 2*np.pi / Mout
    
    # NOTE: Below we include the original parameter t, which is not the same as the reparameterized l.
    data = {"x": z.real, "y": z.imag, "dx": dz.real, "dy": dz.imag, "ddx": ddz.real, "ddy": ddz.imag, 
            "vx": v.real, "vy": v.imag, "dvx": dv.real, "dvy": dv.imag, "t": t, "w": w}
    
    if output:
        r, _ = geom.precompute_line_avg(derivative=0, tol=1e-12, maxiter=200, verbose=False)
        dr, _ = geom.precompute_line_avg(derivative=1, tol=1e-12, maxiter=200, verbose=False)
        data.update({"rx": r.real, "ry": r.imag, "drx": dr.real, "dry": dr.imag})
        
    data = reparameterize_dict(l, data, M=Mout)
    return data
  
def concat_dicts(dict1, dict2, axis):
    return {key: torch.cat([dict1[key], dict2[key]], axis=axis) for key in dict1.keys()}  

def sigmoid_normalisation(x):
    return 1/(1 + torch.exp(-0.5 * x)) - 0.5

def norm(x):
    return torch.linalg.norm(x, dim=-1, keepdim=True) / x.shape[-1]**0.5

def spectral_rescale(x, factor):
    #Compute Fourier coeffcients up to factor of e^(- something constant)
    x_ft = torch.fft.rfft(x)
    x = torch.fft.irfft(x_ft, n=int(factor * x.shape[-1]))
    return x

def plot_data(data, n_plot):    
    plt.figure(figsize=(10,10))
    for i in range(n_plot*n_plot):
        plt.subplot(n_plot, n_plot, i+1)
        x = data.inp["x"][i, :].cpu().numpy()
        y = data.inp["y"][i, :].cpu().numpy()
        plt.plot(x, y)
        plt.axis("equal")
        
def unpack(transform):
    """Decorator for if a method is supposed to act on the values of a dict."""
    def up_transform(data, *args, **kwargs):
        if type(data) == dict:
            return {k: transform(data[k], *args, **kwargs) for k in data.keys()}
        else:
            return transform(data, *args, **kwargs)
    return up_transform

@unpack
def subsample(x, idx):
    return x[idx]

def to_device(data, device):
    for k in data.keys():
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(device)
    return data

@unpack
def to_dtype(tensor, dtype):
    return tensor.to(dtype)

def subdict(dic, keys):
    return {k: dic[k] for k in keys if k in dic.keys()}

def concat_dict_entries(data):
    return torch.cat(tuple((d[:, None, :] for d in data.values())), dim=1)

@unpack
def integrate(x, w):
    return torch.cumsum(x * w, axis=len(x.shape)-1)

def integrate_reduce(x, w):
    return torch.sum(x * w, axis=-1)

def unpack_data(data):
    xlabels = ['x', 'y', 'dx', 'dy', 'ddx', 'ddy', 'vx', 'vy', 'dvx', 'dvy', 'w', 't']
    ylabels = ['rx', 'ry', 'drx', 'dry']
    inp = {xlabels[i]: data['X'][:, i, :] for i in range(12)}
    out = {ylabels[i]: data['Y'][:, i, :] for i in range(4)}
    return (inp, out)
    
def arclength(dx, dy, w):
    return integrate((dx**2 + dy**2)**0.5, w)    

def normalize(dx, dy):
    """Normalize 2-dim vector"""
    mag = (dx**2 + dy**2)**0.5
    return dx/mag, dy/mag
    
def curvature(dx, dy, ddx, ddy):
    """Find curvature of line segment given points"""
    mag = (dx**2 + dy**2)**0.5
    return (dx * ddy - dy * ddx) / (mag ** 3)

def invariant_quantities(inp):
    labels = ('tx', 'ty', 'c')
    tx, ty = normalize(inp['dx'], inp['dy'])
    c = curvature(inp['dx'], inp['dy'], inp['ddx'], inp['ddy'])
    data = (tx, ty, c)
    return {labels[i]: data[i] for i in range(len(data))}

def project_to_natural(keys, tx, ty, inp):
    """Project the input data to the natural coordinates of the boundary."""
    data = {}
    for k in keys:
        kx = k + 'x'
        ky = k + 'y'
        kt = k + 't'
        kn = k + 'n'
        data.update({kt: inp[kx] * tx + inp[ky] * ty,
                     kn: inp[ky] * tx - inp[kx] * ty})
    return data

def projection(ux, uy, nx, ny, inv=False):
    """Project vector (ux, uy) onto unit vector n=(nx, ny) and orth, t=(ny, -nx)."""
    if inv:
        wn = ux * nx - uy * ny
        wt = ux * ny + uy * nx
    else:
        wn = ux * nx + uy * ny
        wt = ux * ny - uy * nx
    return wn, wt

def mean_distance(data):
    w_sum = torch.sum(data['w'], axis=-1)
    center_x = integrate_reduce(data['x'], data['w'])[:, None]  / w_sum[:, None]
    center_y = integrate_reduce(data['y'], data['w'])[:, None] / w_sum[:, None]
    dx = data['x'] - center_x
    dy = data['y'] - center_y
    return (dx**2 + dy**2)**0.5, dx, dy

def avg(f, g, dx, dy, w):
    return np.sum(np.real(np.conjugate(1j * f) * g) * (dx**2 + dy**2) ** 0.5 * w)
