from boundary_solvers.geometry import *
import numpy as np
import pytest as pt

@pt.fixture
def geometry():
    k = 10
    width = 0.1
    height = 0.1
    center = 1+2j
    normal = -(1+1j)/2**0.5
    corner_w = 0.02
    eps = 0.01

    func = lambda t: np.sin(k*t) * eps + eps * 1.2
    dfunc = lambda t: k*np.cos(k*t) * eps
    ddfunc = lambda t: -k**2*np.sin(k*t) * eps
    return RoundedMicroGeomGeneric(func, dfunc, ddfunc, center, width, height, normal, corner_w, n_refine=3, n_corner_refine=0)
    #return RoundedMicroGeom(lambda t: -func(t), lambda t: -dfunc(t), lambda t: -ddfunc(t), 
                            #width, height, corner_w, line_pos=0, center_x=center.real, shift_y=center.imag, n_refine=3, n_corner_refine=0)


def test_adjoint(geometry):
        
    solution = lambda z: np.conjugate(z)
    condition = lambda t: solution(geometry.eval_param(t=t))
    
    a = geometry.line_left
    b = geometry.line_right
    
    _, avgFunc = geometry.precompute_line_avg(derivative=0,tol=1e-10, maxiter=100)
    _, avgFuncDer = geometry.precompute_line_avg(derivative=1,tol=1e-10, maxiter=100)
    t, _ = geometry.grid.get_grid_and_weights()
    
    avg = (b**2 - a**2).real/2
    davg = -((b - a)**2).imag/abs(b - a)
    
    TOL = 1e-5
    delta = abs(avgFunc(condition(t)) - avg)
    ddelta = abs(avgFuncDer(condition(t)) - davg)
    assert delta < TOL, f"Expected difference={delta} in average to be smaller than TOL={TOL}"
    assert ddelta < TOL, f"Expected difference={ddelta} in average of derivative to be smaller than TOL={TOL}"
    
    return delta, ddelta

    