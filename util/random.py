import numpy as np
from util.interp import PiecewiseInterp2D

############################################
# 1D
############################################


def sample_fourierGP_1D(x, coef):
    """Generate a gaussian process over a (periodic) 2D domain in the points x, y.
       diag is a vector containing the real-valued fourier basis coefficients, to be interpreted as variance.
    """
    coef_rnd = np.random.multivariate_normal(np.zeros_like(coef), np.diag(coef))
    basis_eval = [np.cos(x * np.pi * k) for k in range(len(coef))]
    return sum([b * c for b, c in zip(basis_eval, coef_rnd)])


def get_coef_1D(kmax, eps):
    ks = np.arange(kmax+1)
    return eps * np.exp(-ks**2 * eps**2)


def get_multiscale_coef_1D(kmax, L, eps):
    k = np.arange(kmax+1)
    return L * np.exp(-k**2 * L**2) + eps * np.exp(-(k-1/eps)**2 * eps / 20)


############################################
# 2D
############################################


def sample_fourierGP_2D(x, y, coef):
    """Generate a gaussian process over a (periodic) 2D domain in the points x, y.
       diag is a vector containing the real-valued fourier basis coefficients, to be interpreted as variance.
    """
    coef_rnd = np.random.randn(*coef.shape) * coef
    ans = np.zeros_like(x)
    for kx in range(coef.shape[0]):
        for ky in range(coef.shape[1]):
            ans += np.cos(x * np.pi * kx) * np.cos(y * np.pi * ky) * coef_rnd[kx, ky]
    return ans


def sample_fourierGP_2D_FFT(c, seed=None):
    """Generate a gaussian process over a (periodic) 2D domain in the points x, y.
    Using the FFT method. This method is much faster than the other one."""
    
    if seed is not None:
        np.random.seed(seed)
    c2 = np.vstack([np.hstack([c[:,:], c[:,:0:-1]]), 
                np.hstack([c[:0:-1,:], c[:0:-1,:0:-1]])])
    c2 = np.random.randn(*c2.shape) * c2
    return np.fft.ifft2(c2).real


def get_multiscale_coef_2D(kmax_x, kmax_y, L, LAmp, eps, epsAmp, anisotropy=1):
    a = anisotropy
    Kx, Ky = np.meshgrid(np.arange(kmax_x+1), np.arange(kmax_y+1))
    return LAmp * np.exp(-((a*Kx)**2 + Ky**2) * L**2) + epsAmp * np.exp(-(np.abs((a*Kx)**2 + Ky**2-1/eps**2)) * eps / 100)


def fourierGP_2D_lerp(kmax_x, kmax_y, L, LAmp, eps, epsAmp, dom, seed=0):
    a = (dom[1][1]-dom[1][0])/(dom[0][1]-dom[0][0])
    c = get_multiscale_coef_2D(kmax_x, kmax_y, L, LAmp, eps, epsAmp, a)
    f = sample_fourierGP_2D_FFT(c, seed=seed)
    (Nx, Ny) = f.shape
    x = np.linspace(dom[0][0], dom[0][1], Nx+1)[:-1]
    y = np.linspace(dom[1][0], dom[1][1], Ny+1)[:-1]
    f = f-np.min(f)
    f = f/np.max(np.abs(f)) * epsAmp
    return PiecewiseInterp2D(x, y, f), x, y


############################################
# Loop erased random walk
############################################

def loop_erased_rw():
    N = 100
    Nmax = 200

    x,y = [0],[0]
    steps = [(0,1),(0,-1),(-1,0),(1,0)]


    for n in range(100000):
        step_idx = np.random.randint(0,4)
        dx,dy = steps[step_idx]
        xn = x[-1]+dx
        yn = y[-1]+dy

        intersects = [i for i in range(len(y)-1) if x[i]==xn and y[i]==yn]
        assert len(intersects) <= 1, "Something is wrong"

        if len(intersects) == 0:
            x.append(xn)
            y.append(yn)

        elif len(intersects) == 1:
            i = intersects[0]
            if (len(x) - i) >= Nmax:
                x = x[i:]
                y = y[i:]
                break
            else:
                x = x[:i+1]
                y = y[:i+1]        


    # Define numpy arrays
    x = np.array(x)
    y = np.array(y)
    # Center
    x = 2*(x-(min(x)+max(x))/2)/(max(x)-min(x))
    y = 2*(y-(min(y)+max(y))/2)/(max(y)-min(y))
    return x, y


def smooth_random_loop_fourier_coef(K):
    x, y = loop_erased_rw()
    t = np.linspace(0,2*np.pi,len(x))
    #X,Y
    ks = np.arange(-K, K)

    E = np.exp(1j * ks[None, :] * t[:, None])
    c = np.linalg.solve(np.conjugate(E.T) @ E, np.conjugate(E.T) @ np.hstack([x[:,None], y[:,None]]))
    return c


