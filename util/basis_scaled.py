import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.fft import fft, ifft, fftshift, ifftshift


######################################################################################
####################  Basis 1D ########################################################
######################################################################################

class Basis():
    def __init__(self, *args, **kwargs):
        """A basis for function spaces. """
        self.dim # dimension of basis
        self.coef # coefficients for each basis function
        raise NotImplementedError("Subclasses should implement this!")

        
    def __call__(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this!")

    
    def get_dim(self) -> int:
        """An integer or tuple of integers corresponding to the basis dimension in each coordinate."""
        return self.dim
    
    def get_coef(self) -> np.ndarray:
        """Coefficients for the bases. This is only relevant if the basis is instantiated with some function."""
        raise NotImplementedError("Subclasses should implement this!")
    
    def change_dim(self, dim: int):
        """Change dimenstion to the one specified, either by zero padding or truncation"""
        selfType = type(self)
        return selfType(self._change_dim(self.coef, dim))
    
    def grid(self):
        selfType = type(self)
        return selfType._grid(self.dim)
    
    def eval_grid(self):
        selfType = type(self)
        return selfType._eval_grid(self.coef)
    
    def diff(self, deg):
        raise NotImplementedError("Subclasses should implement this!")
        
    def plot(self, ax, dim, **kwargs):
        self = self.change_dim(dim)
        ax.plot(self.grid(), np.real(self.eval_grid()), **kwargs)
    
    @staticmethod
    def match_dim_first(func):
        """Match dimensions, then apply function."""
        def _func(basis1, basis2, *args, **kwargs):
            assert isinstance(basis1, type(basis2)), f"Must be the same basis, got {type(basis1)}, {type(basis2)}"
            basis1, basis2 = basis1._match_dim(basis1, basis2)
            return func(basis1, basis2, *args, **kwargs)
        return _func
    
    @staticmethod
    def _match_dim(basis1, basis2):
        """Match dimensions of basis1 and basis2"""
        dim1 = basis1.dim
        dim2 = basis2.dim
        dim = max(dim1, dim2)
        return basis1.change_dim(dim), basis2.change_dim(dim)
    
    
    @staticmethod
    def _interpolate_func(f, dim: int) -> np.ndarray:
        """Interpolate function using basis, with a fixed number of dim basis functions."""
        grid = Basis.grid(dim)
        funcval = f(*grid)
        return Basis.interpolate(funcval)
        
    
    @staticmethod
    def _interpolate(f) -> np.ndarray:
        """From interpolation points to coefficients. Coefficients and interpolation points will have the same dimension"""
        raise NotImplementedError("Subclasses should implement this!")
    
    @staticmethod
    def _eval(coef: np.ndarray, *args) -> np.ndarray:
        """From coefficients, evaluate on unstructured arrays *args. Can't use fft and stuff here"""
        raise NotImplementedError("Subclasses should implement this!")
    
    @staticmethod
    def _eval_grid(coef: np.ndarray) ->  np.ndarray:
        """From coefficients to interpolation points. Interpolation points will have same dimensions as coef."""
        raise NotImplementedError("Subclasses should implement this!")
        
    @staticmethod
    def _diff(coef: np.ndarray, deg: int) -> np.ndarray:
        """From coefficients + degree to coefficients.  Output has same dimension as input."""
        raise NotImplementedError("Subclasses should implement this!")
    
    @staticmethod
    def _grid(dim: int) -> np.ndarray:
        """Interpolation grid. Return a grid on which a function can be evaluated to yield the correct interpolation points."""
        raise NotImplementedError("Subclasses should implement this!")
    
    @staticmethod
    def _change_dim(coef: np.ndarray, dim: int) -> np.ndarray:
        """Change dimension of coef matrix by adding new basis functions."""
        raise NotImplementedError("Subclasses should implement this!")
    
    @staticmethod
    def add(coef: np.ndarray, coef2: np.ndarray) -> np.ndarray:
        """Add coefs from same basis. Needs to handle truncation."""
        raise NotImplementedError("Subclasses should implement this!")
    
    

class ScaleShiftedBasis(Basis):
    def __init__(self, basis, scale, shift, *args, **kwargs):
        """A basis for function spaces. """
        self.basis = basis
        self.shift = shift
        self.scale = scale
        self.shift_scale_op = lambda x: scale*(x - shift)
        self.inv_shift_scale_op = lambda y: y/scale + shift
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.basis(self.inv_shift_scale_op(x))

    def __add__(self, other):
        if isinstance(other, ScaleShiftedBasis):
            return ScaleShiftedBasis(self.basis + other.basis, self.scale, self.shift)
        else:
            return ScaleShiftedBasis(self.basis + other, self.scale, self.shift)
    
    def __mul__(self, other):
        if isinstance(other, ScaleShiftedBasis):
            return ScaleShiftedBasis(self.basis * other.basis, self.scale, self.shift)
        else:
            return ScaleShiftedBasis(self.basis * other, self.scale, self.shift)
            
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return ScaleShiftedBasis(self.basis / other.basis, self.scale, self.shift)
    
    def __sub__(self, other):
        return self + (-1.) * other
    
    def get_dim(self) -> int:
        """An integer or tuple of integers corresponding to the basis dimension in each coordinate."""
        return self.basis.get_dim()
    
    def get_coef(self) -> np.ndarray:
        """Coefficients for the bases. This is only relevant if the basis is instantiated with some function."""
        return self.basis.get_coef()
    
    def change_dim(self, dim: int):
        """Change dimenstion to the one specified, either by zero padding or truncation"""
        basis = self.basis.change_dim(dim)
        return ScaleShiftedBasis(basis, self.scale, self.shift)
    
    def grid(self):
        return self.shift_scale_op(self.basis.grid())
    
    def eval_grid(self):
        return self.basis.eval_grid()
    
    def diff(self, deg):
        basis = (self.scale**(-deg)) * self.basis.diff(deg)
        return ScaleShiftedBasis(basis, self.scale, self.shift)
    
    
    @staticmethod
    def _interpolate_func(f, basis, dim, scale, shift) -> np.ndarray:
        """Interpolate function using basis, with a fixed number of dim basis functions."""
        #grid = self.grid()
        #funcval = f(*grid)
        #return ScaleShiftedBasis(self.basis._interpolate(funcval), self.scale, self.shift)
        grid = basis._grid(dim)
        funcval = f(*grid)
        return ScaleShiftedBasis(basis(basis._interpolate(funcval)), scale, shift)
    
    @staticmethod
    def fromSamplesInDomain(x, f, dom, basis, *args, **kwargs):
        scale, shift = ScaleShiftedBasis._domain_to_scale_shift(basis._domain(), dom)
        basis_obj = basis(basis._interpolate_generic(x/scale + shift, f, dim=kwargs.get('dim', len(x))))
        return ScaleShiftedBasis(basis_obj, scale, shift, *args, **kwargs)
    
    @staticmethod
    def fromFunction(func, scale, shift, basis, *args, **kwargs):
        """Instantiate a basis from a function. 
        The function should take a single argument, 
        which is a tuple of arrays, one for each coordinate."""
        basis = basis.fromFunction(lambda x: func((x - shift)*scale), *args, **kwargs)
        return ScaleShiftedBasis(basis, scale, shift)
    
    @staticmethod
    def fromFunctionInDomain(func, dom, basis, *args, **kwargs):
        """Instantiate a basis from a function. Domain is a tuple of (a,b) for the domain of the function."""
        scale, shift = ScaleShiftedBasis._domain_to_scale_shift(basis._domain(), dom)
        return ScaleShiftedBasis.fromFunction(func, scale, shift, basis, *args, **kwargs)
    
    
    @staticmethod
    def _domain_to_scale_shift(dom_from, dom_to):
        # [a,b] -> [c,d] with linear:
        # d*(x-a)/(b-a) + c*(b-x)/(b-a)
        # = (d-c)/(b-a)(x - (da-cb)/(c-d))
        # = scale * (x - shift)
        # = 
        L_from = dom_from[1]-dom_from[0]
        L_to = (dom_to[1]-dom_to[0])
        scale = L_to/L_from
        shift = (dom_to[1]*dom_from[0] - dom_from[1]*dom_to[0])/L_to
        return scale, shift
        
        
        
    
######################################################################################
##################### CHEBYSHEV ######################################################
######################################################################################
    
class ChebBasis(Basis):
    def __init__(self, coef):
        self.coef = coef
        self.dim = len(coef)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ChebBasis._eval(self.coef, x)
    
    
    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return  ChebBasis(self.coef * other)
        elif isinstance(other, ChebBasis):
            prod, other = Basis._match_dim(self, other)
            f = prod.eval_grid()
            f *= other.eval_grid()
            prod.coef = ChebBasis._interpolate(f)
            return prod
        else:
            raise NotImplementedError("Chebshevs can only be multiplied by scalars or other Chebyshevs")
        
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return self * (other ** (-1.))
    
    def __pow__(self, p):
        raise NotImplementedError("Not Implemented for ChebBasis")
    
    
    #@Basis.match_dim_first
    def __add__(self, other):
        NotImplementedError("Not Implemented for ChebBasis")
    
    @Basis.match_dim_first
    def __sub__(self, other):
        NotImplementedError("Not Implemented for ChebBasis")
   
    def diff(self, deg):
        coef = self._diff(self.coef, deg)
        return ChebBasis(coef)

    def get_coef(self):
        return self.coef

    @staticmethod
    def fromFunction(func, dim):
        return ChebBasis(ChebBasis._interpolate_func(func, dim))
    
    @staticmethod
    def _eval(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
        return cheb.chebval(x, coef)
        
    
    @staticmethod
    def _interpolate_func(f, dim: int) -> np.ndarray:
        fval = f(ChebBasis._grid(dim))
        return ChebBasis._interpolate(fval)
    
    @staticmethod
    def _interpolate(f) -> np.ndarray:
        dim = len(f)
        xcheb = cheb.chebpts1(dim)
        m = cheb.chebvander(xcheb, dim-1)
        c = np.dot(m.T, f)
        c[0] /= dim
        c[1:] /= 0.5*dim
        return c
    
    @staticmethod
    def _interpolate_generic(x, f, dim=None) -> np.ndarray:
        if dim is None:
            dim = len(x)
        assert len(x) == len(f), f"x and f must have the same length, got {len(x)} and {len(f)}"
        m = cheb.chebvander(x, dim-1)
        m = m[:, :dim]
        c = np.linalg.solve(m.T @ m, m.T @ f)
        return c
    
    @staticmethod
    def _eval_grid(coef: np.ndarray) -> np.ndarray:
        return cheb.chebval(cheb.chebpts1(len(coef)), coef)
    
    @staticmethod
    def _diff(coef: np.ndarray, deg: int) -> np.ndarray:
        dif = np.pad(cheb.chebder(coef, deg), (0, deg))
        return dif
    
    @staticmethod
    def _grid(dim: int) -> np.ndarray:
        return cheb.chebpts1(dim)
    
    @staticmethod
    def _change_dim(coef: np.ndarray, dim: int) -> np.ndarray:
        if dim <= len(coef):
            nCoef = coef[:dim]
        else:
            nCoef = np.hstack([coef, np.zeros((dim - len(coef),))])
        return nCoef
    
    @staticmethod
    def _domain():
        return [-1.,1.]
######################################################################################
####################  FOURIER ########################################################
######################################################################################

class FourBasis(Basis):
    def __init__(self, coef):
        self.coef = coef
        self.dim = len(coef)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return FourBasis._eval(self.coef, x)
    
    
    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return  FourBasis(self.coef * other)
        elif isinstance(other, FourBasis):
            prod, other = Basis._match_dim(self, other)
            f = prod.eval_grid()
            f *= other.eval_grid()
            prod.coef = FourBasis._interpolate(f)
            return prod
        else:
            raise NotImplementedError("Must be instance of int, float or FourBasis.")
        
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return self * (other ** (-1.))
    
    def __pow__(self, p):
        coef = FourBasis._interpolate(self.eval_grid() ** p)
        return FourBasis(coef)
    
    
    #@Basis.match_dim_first
    def __add__(self, other):
        if isinstance(other, FourBasis):
            self, other = Basis._match_dim(self, other)
            return FourBasis(self.coef + other.coef)
        else:
            cons = np.zeros_like(self.coef)
            kmax = (len(cons)-1)//2
            cons[kmax] = other
            return FourBasis(self.coef + cons)
    
    @Basis.match_dim_first
    def __sub__(self, other):
        return FourBasis(self.coef - other.coef)
   
    def diff(self, deg):
        coef = self._diff(self.coef, deg)
        return FourBasis(coef)

    def get_coef(self):
        return self.coef

    @staticmethod
    def fromFunction(func, dim):
        return FourBasis(FourBasis._interpolate_func(func, dim))
    
    @staticmethod
    def _interpolate_generic(x, f, dim=None) -> np.ndarray:
        raise NotImplemented("Not implemeted yet.")
    
    @staticmethod
    def _eval(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
        dim = len(coef)
        freq = FourBasis._freq(dim)
        if isinstance(x, float) or isinstance(x, int):
            y = 0
        else:
            y = np.zeros(shape=x.shape).astype(np.complex128)
        for i in range(dim):
            y += coef[i] * np.exp(freq[i] * x)
        return y
    
    @staticmethod
    def _interpolate_func(f, dim:int):
        fval = f(FourBasis._grid(dim))
        return FourBasis._interpolate(fval)
    
    @staticmethod
    def _interpolate(f: np.ndarray) -> np.ndarray:
        return fftshift(fft(f)) / len(f)
        
    @staticmethod
    def _eval_grid(coef: np.ndarray) -> np.ndarray:
        return ifft(ifftshift(coef)) * len(coef)
    
    @staticmethod
    def _diff(coef: np.ndarray, deg: int = 0) -> np.ndarray:
        if deg == 0:
            return coef
        else:
            freq = FourBasis._freq(len(coef))
            coef = (freq ** deg) * coef
            return coef
    
    @staticmethod
    def _grid(dim: int) -> np.ndarray:
        return np.linspace(0, 1, dim + 1)[:-1]

    @staticmethod
    def _freq(dim: int) -> np.ndarray:
        kmax = (dim-1)//2
        return 2j * np.pi * np.arange(-kmax, kmax+1)
    
    @staticmethod
    def _change_dim(coef: np.ndarray, dim: int) -> np.ndarray:
        curDim = len(coef)
        curKmax = (curDim - 1) // 2
        kmax = (dim - 1) // 2
        
        if dim <= len(coef):
            nCoef = coef[curKmax-kmax:curKmax+kmax+1]
        else:
            kDiff = kmax - curKmax
            zer = np.zeros((kDiff,)).astype(np.complex128)
            nCoef = np.hstack([zer, coef, zer])
        return nCoef
    
    @staticmethod
    def _domain():
        return [0, 1]
    
######################################################################################
################################ Basis 2D ############################################
######################################################################################

class BasisProduct(Basis):
    def __init__(self, coef, xDim, yDim, xBasis: Basis, yBasis: Basis):
        """Combine two bases on a grid."""
        self.xBasis = xBasis
        self.yBasis = yBasis
        
        if coef is None:
            self.coef = np.zeros((xDim, yDim)).astype(np.complex128)
            self.xDim = xDim
            self.yDim = yDim
        else:
            self.coef = coef
            self.xDim = coef.shape[0]
            self.yDim = coef.shape[1]            
        pass
        
    def __call__(self, x, y):
        """Evaluate in arbitrary points."""
        return self._eval(self.coef, x, y, self.xBasis, self.yBasis)
    
    def __mul__(self, other):
        """Multiply with function of the same kind, or a scalar"""
        
        if isinstance(other, float) or isinstance(other, int):
            prod = BasisProduct(self.coef * other, self.xDim, self.yDim, self.xBasis, self.yBasis)
        
        if isinstance(other, BasisProduct):
            prod, other = BasisProduct._match_dim(self, other)

            # Product in interpolation space
            val = prod.eval_grid()
            val = val * other.eval_grid()

            # Back to coefficients
            coef = BasisProduct._interpolate(val, self.xBasis, self.yBasis)
            prod.coef = coef
            
        return prod
    
    def __rmul__(self, other):
        return self.__mul__(other)        
        
    
    def __pow__(self, p):
        if isinstance(p, float) or isinstance(p, int):
            # Fractional power, compute slow using interpolation
            val = self.eval_grid()
            val = val ** float(p)
            
            # Back to coefficients
            coef = BasisProduct._interpolate(val, self.xBasis, self.yBasis)
            return BasisProduct(coef, self.xDim, self.yDim, self.xBasis, self.yBasis)
        else:
            raise NotImplementedError("Can only raise to integer or float powers.")
        
    @Basis.match_dim_first
    def __add__(self, other):
        """Addition, not in place."""
        self.coef = self.coef + other.coef
        return self
    
    @Basis.match_dim_first
    def __sub__(self, other):
        """Subtraction"""
        return self + (-1.)*other
    
    def domain(self):
        return self._domain(self.xBasis, self.yBasis)
     
    def change_dim(self, xDim, yDim):
        """Change the dimension. Return new basis."""
        coef = BasisProduct._change_dim(self.coef, xDim, yDim, self.xBasis, self.yBasis)
        return BasisProduct(coef, xDim, yDim, self.xBasis, self.yBasis)
        
    def reduce_eval(self, val, axis):
        """Evaluate along axis and reduce to 1d Basis"""
        if axis == 0:
            coef = BasisProduct._eval_reduce_x(self.coef, self.xBasis, val)
            basis = self.yBasis(coef)
        elif axis == 1:
            coef = BasisProduct._eval_reduce_y(self.coef, self.yBasis, val)
            basis = self.xBasis(coef)
        else:
            assert False, "axis must be either 0 or 1."
        return basis
     
    def diff(self, xDeg, yDeg) -> Basis:
        """Evaluate derivatives."""
        coef = BasisProduct._diff(self.coef, self.xBasis, self.yBasis, xDeg, yDeg)
        return BasisProduct(coef, self.xDim, self.yDim, self.xBasis, self.yBasis)
    
    def eval_grid(self) -> np.ndarray:
        """Evaluate on preset interpolation grid."""
        return BasisProduct._eval_grid(self.coef, self.xBasis, self.yBasis)
        
    def grid(self):
        """Grid"""
        return BasisProduct._grid(self.xDim, self.yDim, self.xBasis, self.yBasis)
        
    
    def plot(self, ax, **kwargs):
        """Plot on axis"""
        X, Y = self.grid()
        F = self.eval_grid()
        ax.pcolormesh(X, Y, F, **kwargs)
        
    
    @staticmethod
    def fromFunction(func, xDim, yDim, xBasis, yBasis) -> Basis:
        """Create a basis from a 2d function."""
        basis = BasisProduct(None, xDim, yDim, xBasis, yBasis)
        basis.coef = BasisProduct._interpolate_func(func, xDim, yDim, xBasis, yBasis)
        return basis
    
    @staticmethod
    def _match_dim(basis1, basis2):
        """Match the dimensions of two bases."""
        assert basis1.xBasis == basis2.xBasis, "incompatible x-basis"
        assert basis1.yBasis == basis2.yBasis, "incompatible y-basis"
        
        xDim = max(basis1.xDim, basis2.xDim)
        yDim = max(basis1.yDim, basis2.yDim)
        
        return basis1.change_dim(xDim, yDim), basis2.change_dim(xDim, yDim)
    
    @staticmethod
    def _eval_reduce_x(coef:np.ndarray, xBasis:Basis, val:float) -> np.ndarray:
        """Evaluate along x-axis and reduce to 1d function"""
        yDim = coef.shape[1]
        res = np.zeros((yDim,)).astype(np.complex128)
        for i in range(yDim):
            res[i] = xBasis._eval(coef[:, i], val)
        return res
    
    @staticmethod
    def _eval_reduce_y(coef:np.ndarray, yBasis:Basis, val:float) -> np.ndarray:
        """Evaluate along y-axis and reduce to 1d function"""
        xDim = coef.shape[0]
        res = np.zeros((xDim,)).astype(np.complex128)
        for i in range(xDim):
            res[i] = yBasis._eval(coef[i, :], val)
        return res
        
    @staticmethod
    def _eval(coef: np.ndarray, x: np.ndarray, y: np.ndarray, xBasis: Basis, yBasis:Basis) -> np.ndarray:
        """Evaluate basis at points (x, y)"""
        (xDim, yDim) = coef.shape
        N = len(x)
        
        xVal = np.zeros((N, yDim)).astype(np.complex128)
        out = np.zeros((N,)).astype(np.double)
        
        for j in range(yDim):
            xVal[:,j] = xBasis._eval(coef[:, j], x)
        
        for i in range(N):
            out[i] = np.real(yBasis._eval(xVal[i, :], y[i]))
        
        return out
    
    @staticmethod
    def _interpolate_func(f, xDim: int, yDim: int, xBasis: Basis, yBasis: Basis) -> np.ndarray:    
        """Interpolate from function given bases in each direction"""
        xG, yG = BasisProduct._grid(xDim, yDim, xBasis, yBasis)
        fG = f(xG.flatten(), yG.flatten()).reshape(xDim, yDim).astype(np.complex128)
        return BasisProduct._interpolate(fG, xBasis, yBasis)
        
    @staticmethod
    def _interpolate(fG: np.ndarray, xBasis: Basis, yBasis: Basis) -> np.ndarray:
        """Interpolate from interpolation points"""
        coef = BasisProduct._product_eval(fG, xBasis._interpolate, yBasis._interpolate, dtype=np.complex128)
        return coef
    
    @staticmethod
    def _eval_grid(coef: np.ndarray, xBasis: Basis, yBasis: Basis) -> np.ndarray:
        """Evaluate function on interpolation grid"""
        fG = BasisProduct._product_eval(coef, xBasis._eval_grid, yBasis._eval_grid, dtype=np.complex128)
        return np.real(fG)
            
    
    @staticmethod
    def _diff(coef: np.ndarray, xBasis: Basis, yBasis: Basis, xDeg: int = 0, yDeg: int = 0) -> np.ndarray:
        """Differentiate and return coefficients of derivatives."""
        fx = lambda c: xBasis._diff(c, xDeg)
        fy = lambda c: yBasis._diff(c, yDeg)
        coef = BasisProduct._product_eval(coef, fx, fy, dtype=np.complex128)
        return coef
        
    @staticmethod
    def _grid(xDim: int, yDim: int, xBasis: Basis, yBasis: Basis) -> np.ndarray:
        """Return grid corresponding to output of evaluate()."""
        xG, yG = np.meshgrid(xBasis._grid(xDim), yBasis._grid(yDim))
        return xG.T, yG.T
    
    @staticmethod
    def _change_dim(coef, xDim, yDim, xBasis, yBasis) -> np.ndarray:
        fx = lambda c: xBasis._change_dim(c, xDim)
        fy = lambda c: yBasis._change_dim(c, yDim)
        return BasisProduct._product_eval(coef, fx, fy, dtype=np.complex128)
    
    @staticmethod
    def _product_eval(xy, fx, fy, dtype):
        """Evaluate functions on a product grid by applying row wise and then column wise transforms.
           This function is used for diff, interpolate and evaluate."""
        
        tmp = []
        for i in range(xy.shape[0]):
            tmp.append(fy(xy[i, :]))
        tmp = np.vstack(tmp)
        
        out = []
        for j in range(tmp.shape[1]):
            out.append(fx(tmp[:, j])[:, None])
        
        return np.hstack(out)
    
    
    @staticmethod
    def _domain(xBasis, yBasis):
        """Return domain of basis. Example: [0, 1] x [0, 1] if xBasis and yBasis are on [0, 1]."""
        return [xBasis._domain(), yBasis._domain()]
    
    

class ScaleShiftedBasisProduct(Basis):
    def __init__(self, basis, scale, shift):
        """Combine two bases on a grid."""
        self.basis = basis
        self.shift = shift
        self.scale = scale
        self.scale_shift_x = lambda x: self.scale[0]*(x - self.shift[0])
        self.scale_shift_y = lambda y: self.scale[1]*(y - self.shift[1])
        self.inv_scale_shift_x = lambda x: x/self.scale[0] + self.shift[0]
        self.inv_scale_shift_y = lambda y: y/self.scale[1] + self.shift[1]
        self.__dict__.update(basis.__dict__)
        
    def __call__(self, x, y):
        """Evaluate in arbitrary points."""
        x = self.inv_scale_shift_x(x)
        y = self.inv_scale_shift_y(y)
        return self.basis(x, y)
    
    #@ScaleShiftedBasisProduct.match_scaleshift
    def __mul__(self, other):
        """Multiply with function of the same kind, or a scalar"""
        if isinstance(other, ScaleShiftedBasisProduct):
            return ScaleShiftedBasisProduct(self.basis * other.basis, self.scale, self.shift)
        else:
            return ScaleShiftedBasisProduct(self.basis * other, self.scale, self.shift)
    
    #@ScaleShiftedBasisProduct.match_scaleshift
    def __rmul__(self, other):
        """Multiply with function of the same kind, or a scalar"""
        return self.__mul__(other)
    
    
    #@ScaleShiftedBasisProduct.match_scaleshift
    def __add__(self, other):
        """Addition, not in place."""
        return ScaleShiftedBasisProduct(self.basis + other.basis, self.scale, self.shift)
    
    #@ScaleShiftedBasisProduct.match_scaleshift
    def __sub__(self, other):
        """Subtraction"""
        other = (-1.)*other
        return self + other
        
     
    def change_dim(self, xDim, yDim):
        """Change the dimension. Return new basis."""
        return ScaleShiftedBasisProduct(self.basis.change_dim(xDim, yDim), self.scale, self.shift)
        
    def reduce_eval(self, val, axis):
        """Evaluate along axis and reduce to 1d Basis"""
        if axis==0:
            valss = self.inv_scale_shift_x(val)
            ax2 = 1
            #print(f"Eval at x = {val}, native = {valss}")
        else:
            valss = self.inv_scale_shift_y(val)
            ax2 = 0
            #print(f"Eval at y = {val}, native = {valss}")
                
        return ScaleShiftedBasis(self.basis.reduce_eval(valss, axis), self.scale[ax2], self.shift[ax2])
     
    def diff(self, xDeg, yDeg) -> Basis:
        """Evaluate derivatives."""
        rescale = 1/((self.scale[0]**xDeg) * (self.scale[1]**yDeg))
        return ScaleShiftedBasisProduct(rescale * self.basis.diff(xDeg, yDeg), self.scale, self.shift)
    
    def __pow__(self, p):
        """Raise to power p."""
        return ScaleShiftedBasisProduct(self.basis**p, self.scale, self.shift)
    
    def eval_grid(self) -> np.ndarray:
        """Evaluate on preset interpolation grid."""
        return self.basis.eval_grid()
        
    def grid(self):
        """Grid"""
        x, y = self.basis.grid()
        return self.scale_shift_x(x), self.scale_shift_y(y)
        
    
    def plot(self, ax, cbar=False, **kwargs):
        """Plot on axis"""
        
        X, Y = self.grid()
        F = self.eval_grid()
        ax.pcolormesh(X, Y, F, **kwargs)
        
    def domain(self):
        """Return domain of basis. Example, [0, 1] for a basis on [0, 1]."""
        dom = self.basis.domain()
        scale_ops = [self.scale_shift_x, self.scale_shift_y]
        return [[scale_op(x) for x in subdom] for subdom, scale_op in zip(dom, scale_ops)]
        
    
    @staticmethod
    def fromFunction(func, scale, shift, xDim, yDim, xBasis, yBasis) -> Basis:
        """Create a basis from a 2d function."""
        func_ss = lambda x,y: func(scale[0]*(x-shift[0]), scale[1]*(y-shift[1]))
        basis = BasisProduct.fromFunction(func_ss, xDim, yDim, xBasis, yBasis)
        return ScaleShiftedBasisProduct(basis, scale, shift)
    
    @staticmethod
    def fromFunctionInDomain(func, domain, xDim, yDim, xBasis, yBasis) -> Basis:
        """Create a basis from a 2d function."""
        # Calculate shifts from native domains of bases to target.
        dom_from = [xBasis._domain(), yBasis._domain()]
        dom_to = domain
        
        scale, shift = ScaleShiftedBasisProduct._domain_to_scale_shift(dom_from,
                                                                      dom_to)
        return ScaleShiftedBasisProduct.fromFunction(func, scale, shift,
                                                     xDim, yDim, xBasis, yBasis)
    
    @staticmethod
    def _domain_to_scale_shift(dom_from, dom_to):
        """Calculate scale and shift for a linear transformation from one domain to another."""
        cls = ScaleShiftedBasis
        sc_x, sh_x = cls._domain_to_scale_shift(dom_from[0], dom_to[0])
        sc_y, sh_y = cls._domain_to_scale_shift(dom_from[1], dom_to[1])
        return [sc_x, sc_y], [sh_x, sh_y]
        
    
    @staticmethod
    def match_scale_shift(func):
        """Decorator to match scale and shift of two ScaleShiftedBasisProduct objects."""
        def func_match(one, other):
            if isinstance(other, ScaleShiftedBasisProduct):
                assert one.scale[0] == other.scale[0], "Mismatch in x scale"
                assert one.scale[1] == other.scale[1], "Mismatch in y scale"
                assert one.shift[0] == other.shift[0], "Mismatch in x shift"
                assert one.shift[1] == other.shift[1], "Mismatch in y shift"
            else:
                pass
            return func(one, other)
        return func_match