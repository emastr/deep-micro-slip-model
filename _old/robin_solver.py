import scipy.sparse as sp
from scipy.sparse.linalg import gmres
import numpy as np
from scipy.sparse.linalg import spsolve
import time
import sys
sys.path.append('/home/emastr/phd/')
from util.basis import Basis, ChebBasis, FourBasis, BasisProduct

############ TOOLS ####################################

def speye(n):
    return spdiags(np.ones(1, n), [0], (n, n))

def kron(A, B):
    cols = []
    for i in range(A.shape[1]):
        rows = []
        for j in range(A.shape[0]):
            rows.append(A[i,j] * B)
        cols.append(torch.vstack(rows))
    return torch.hstack(cols)

def sparse_coo(indx, data, *args, **kwargs):
    return sp.coo_matrix((data, indx), *args, **kwargs).tocsr()    

spdiags = lambda *args, **kwargs: sp.diags(*args, **kwargs).tocsr()
kron = lambda *args, **kwargs: sp.kron(*args, **kwargs).tocsr()
speye = lambda *args, **kwargs: sp.eye(*args, **kwargs).tocsr()

hstack = lambda *args, **kwargs: np.hstack(*args, **kwargs)
vstack = lambda *args, **kwargs: np.vstack(*args, **kwargs)

hstackSP = lambda *args, **kwargs: sp.hstack(*args, **kwargs)
vstackSP = lambda *args, **kwargs: sp.vstack(*args, **kwargs)

solve = lambda A, b: spsolve(A.tocsr(), b)
ones = lambda *args, **kwargs: np.ones(*args, **kwargs)
zeros = lambda *args, **kwargs: np.zeros(*args, **kwargs)
arange = lambda *args, **kwargs: np.arange(*args, **kwargs)
    

def flip(vec):
    return vec[::-1]

######################### END OF TOOLS ###########################


#================================================================#
######################## SOLVER ##################################
def solveRobinStokes_fromStreamFunction(psi, pressure, xDim, yDim):    
    xBasis = FourBasis
    yBasis = ChebBasis
    
    # Convert to basis
    p_basis = BasisProduct._fromFunction(p, xDim, yDim, xBasis, yBasis)
    psi_basis = BasisProduct._fromFunction(psi, xDim, yDim, xBasis, yBasis)
    
    # U and V velocity components
    u_basis = psi_basis.diff(0, 1)
    v_basis = -1. * psi_basis.diff(1, 0)

    # Laplacian and pressure gradients
    lapu_basis = u_basis.diff(2, 0) + u_basis.diff(0, 2)
    lapv_basis = v_basis.diff(2, 0) + v_basis.diff(0, 2)
    dxp_basis = p_basis.diff(1,0)
    dyp_basis = p_basis.diff(0,1)


    ### Problem parameters

    # Inner force
    f_basis = lapu_basis - dxp_basis
    g_basis = lapv_basis - dyp_basis


    # Upper boundary condition
    uTop_basis = u_basis.reduce_eval(1, axis=1)
    vTop_basis = v_basis.reduce_eval(1, axis=1)


    # Lower boundary velocities
    uBot_basis = u_basis.reduce_eval(-1, axis=1)
    vBot_basis = v_basis.reduce_eval(-1, axis=1)


    # Lower boundary slip coefficient
    duBot_basis = u_basis.diff(0,1).reduce_eval(-1, axis=1)
    alpha_basis = (-1.) * (uBot_basis / duBot_basis)
    beta_basis = 0. * uBot_basis
    
    u, v, info = solveRobinStokes_fromBasis(f_basis, 
                                           g_basis, 
                                           alpha_basis, 
                                           uTop_basis, 
                                           beta_basis, 
                                           vTop_basis,             
                                           vBot_basis, 
                                           xDim, yDim)
    
    return u, v, info

def solveRobinStokes_fromFunc(f, g, alpha, uTop, beta, vTop, vBot, xDim, yDim, **kwargs):
    """Solve from function handles"""
    f = BasisProduct.fromFunction(f, xDim, yDim, FourBasis, ChebBasis)
    g = BasisProduct.fromFunction(g, xDim, yDim, FourBasis, ChebBasis)
    alpha = FourBasis.fromFunction(alpha, xDim)
    uTop = FourBasis.fromFunction(uTop, xDim)
    beta = FourBasis.fromFunction(beta, xDim)
    vTop = FourBasis.fromFunction(vTop, xDim)
    vBot = FourBasis.fromFunction(vBot, xDim)
    return solveRobinStokes_fromBasis(f, g, alpha, uTop, beta, vTop, vBot, xDim, yDim, **kwargs)


def solveRobinStokes_fromBasis(f, g, alpha, uTop, beta, vTop, vBot, xDim, yDim, **kwargs):
    """Solve from Basis Function objects"""
    uC,vC,info = solveRobinStokes(f.coef, g.coef, alpha.coef, uTop.coef, beta.coef, vTop.coef, vBot.coef, xDim, yDim, **kwargs)
    u = BasisProduct(uC, xDim, yDim, FourBasis, ChebBasis)
    v = BasisProduct(vC, xDim, yDim, FourBasis, ChebBasis)
    return u, v, info


def solveRobinStokes(f, g, alpha, uTop, beta, vTop, vBot, Nx, Ny, with_gmres=False, **gmresargs):
    """SOLVEROBINSTOKES   Solve 1-periodic Stokes equations on 2D domain [0,1]x[-1,1].
     The inputs must be ChebFourFun - objects.

     f:[0,1]x[-1,1] -> R  force in u-direction
     g:[0,1]x[-1,1] -> R  force in v-direction
     alpha:[0,1] -> R     variable shear constant
     uTop:[0,1] -> R      Dirichlet condition for u at y=1
     beta:[0,1] -> R      Inhom robin condition for u at y=-1
     vTop:[0,1] -> R      Dirichlet condition for v at y=1
     vBot:[0,1] -> R      Dirichlet condition for v at y=-1

     solveRobinStokes then solves The following boundary value problem:

         PDE: Moment equations
           (DxDx + DyDy)*u - Dx*p = f on [0,1]x[-1,1]
           (DxDx + DyDy)*v - Dy*p = g on [0,1]x[-1,1]
         PDE: Continuity Equations
                      Dx*u + Dy*v = 0 on [0,1]x[-1,1]
         BC: Boundary conditions
                         u - uTop = 0 on [0,1]x{1}
            u + alpha*Dy*u - beta = 0 on [0,1]x{-1}
                         v - vTop = 0 on [0,1]x{1}
                         v - vBot = 0 on [0,1]x{-1}
     Returns
      u, v. p must be solved using a pressure solver."""
    
    
    info = {"Dim": (Nx, Ny)}
    kmax = (Nx-1) // 2
    
    # Make sure the dimensions match. Truncate or zero pad.
    f = f.reshape(Nx, Ny)
    g = g.reshape(Nx, Ny)
    alpha = alpha.reshape(Nx,1)
    uTop = uTop.reshape(Nx,1)
    beta = beta.reshape(Nx,1)
    vTop = vTop.reshape(Nx,1)
    vBot = vBot.reshape(Nx,1) 
    
    t = time.time()
    # Laplace system 
    ns = arange(0, Ny)
    ks = arange(-kmax,kmax+1)
    cn = lambda n: 1. + (n==0) 
    # To correct low order coefficients!

    # Inverse d/dy matrix
    nVals = arange(1,Ny)[None, :]
    dyValm1 = cn(nVals-1) / (2*nVals)
    dyValp1 = -1 / (2*nVals)
    By = spdiags(vstack([dyValm1, dyValp1]), [0,2], (Ny-1,Ny))

    # Inverse dd/ddy matrix
    nVals = arange(2,Ny)[None, :]
    ddyValp2 = 1./(4.*nVals * (nVals+1))
    ddyVal0  = -1./(2.*(nVals ** 2 - 1))
    ddyValm2 = cn(nVals-2)/(4 * nVals * (nVals-1))
    By2 = spdiags(vstack([ddyValm2, ddyVal0, ddyValp2]), \
                         [0,2,4], (Ny-2, Ny))

    
    # Inverse ddd/dddy matrix
    nVals = arange(3, Ny)[None, :]
    dddyValp3 = -1 / (8*nVals * (nVals+1) * (nVals+2))
    dddyValp1 =  3 / (8*(nVals+2) * nVals * (nVals-1))
    dddyValm1 = -3 / (8*(nVals-2) * nVals *(nVals+1)) 
    dddyValm3 = cn(nVals-3) / (8*nVals * (nVals-1) * (nVals-2))

    By3 = spdiags(vstack([dddyValm3, dddyValm1, dddyValp1, dddyValp3]),\
                          [0,2,4,6], (Ny-3, Ny))

    Dx  = spdiags((2j*np.pi*(ks[None, :])) ** 1, [0], (Nx, Nx))
    Dx2 = spdiags((2j*np.pi*(ks[None, :])) ** 2, [0], (Nx, Nx))
    Dx3 = spdiags((2j*np.pi*(ks[None, :])) ** 3, [0], (Nx, Nx))

    Idx  = speye(Nx)
    Idy  = spdiags(ones((1, Ny-1)), [1], (Ny-1, Ny))
    Idy2 = spdiags(ones((1, Ny-2)), [2], (Ny-2, Ny))
    Idy3 = spdiags(ones((1, Ny-3)), [3], (Ny-3, Ny))

    # Eval
    botEval = kron(Idx, (-1) ** ns)
    
    topEval = kron(Idx, ones(ns.shape))
    dirZero = sparse_coo([[], []], [], (Nx, Ny*Nx))

    # Create B_upper
    AMat = spdiags(flip(alpha).repeat(Nx, 1), ks, (Nx, Nx))
    
    dyEval  = (ns ** 2) *(-1) ** (ns+1)
    botRobi = botEval + kron(AMat, dyEval)
    unit_const = (ks == 0)
    zero_const = sparse_coo([[],[]],[],((Ny-2),Nx*Ny))
    
    # B system
    #print(By.shape, By2.shape, By.shape)
    B11 = kron(Dx2[:kmax,:], By2[1:, :])+kron(Idx[:kmax,:], Idy3)
    B12 = -(kron(Dx3[:kmax,:], By3)+kron(Dx[:kmax,:], By[2:, :]))
    B21 = kron(unit_const, Idy2)
    B22 = zero_const
    B31 = kron(Dx2[kmax+1:,:], By2[1:, :])+kron(Idx[kmax+1:,:], Idy3)
    B32 = -(kron(Dx3[kmax+1:,:], By3)+kron(Dx[kmax+1:,:],By[2:, :]))
    B41 = kron(Dx, By)
    B42 = kron(Idx, Idy)
                  
    B_pde = vstackSP([hstackSP([B11, B12]),
                      hstackSP([B21, B22]),
                      hstackSP([B31, B32]),
                      hstackSP([B41, B42])])
    
    
    #### BUG? ####
    # MIGHT HAVE TO PERMUTE f.C DIMENSIONS
    ##############
    
    b_mom = kron(Idx, By2[1:,:]) @ f.reshape(Ny*Nx, 1) \
                  - kron(Dx, By3) @ g.reshape(Ny*Nx, 1)
    b_pde = vstack([b_mom[0:(kmax*(Ny-3)),:],
                      By2 @ (f[kmax,:, None]),
                      b_mom[(kmax+1)*(Ny-3):,:],
                      zeros(((Ny-1)*Nx,1))])

    B_dir = vstackSP([hstackSP([botRobi, dirZero]),\
                      hstackSP([dirZero, botEval]),\
                      hstackSP([topEval, dirZero]),\
                      hstackSP([dirZero[ks!=0,:], topEval[ks!=0, :]])\
                    ])

    b_dir = vstack([beta, vBot, uTop, vTop[ks!=0,:]])
    
    # solve
    B = vstackSP([B_pde, B_dir])    
    b = vstack([b_pde, b_dir])
    
    info["buildTime"] = time.time() - t;
    
    t = time.time()
    if not with_gmres:
        uv = solve(B, b)
    else:
        uv, _ = gmres(B, b, **gmresargs)
    info["solveTime"] = time.time() - t
    
    
    L = Nx*Ny
    idx = arange(0, L)
    uC = uv[idx+0*L].reshape(Nx, Ny)
    vC = uv[idx+1*L].reshape(Nx, Ny)
    
   
    
    return uC, vC, info


def testSolve(xDim, yDim):
    import matplotlib.pyplot as plt

    xBasis = FourBasis
    yBasis = ChebBasis

    psi = lambda x, y: np.exp(y*np.cos(2*np.pi*x))
    p = lambda x,y: 1. / (y**2 + 1) * np.arctan(np.cos(2*np.pi*x));
    a = lambda x: 1. / (np.sin(2*np.pi*x)+2);
    
    
    p_basis = BasisProduct.fromFunction(p, xDim, yDim, xBasis, yBasis)
    psi_basis = BasisProduct.fromFunction(psi, xDim, yDim, xBasis, yBasis)
    alpha_basis = xBasis.fromFunction(a, xDim)

    # U and V velocity components
    u_basis = psi_basis.diff(0, 1)
    v_basis = -1. * psi_basis.diff(1, 0)

    # Divergence field
    div_basis = u_basis.diff(1, 0) + v_basis.diff(0, 1)

    # Laplacian and pressure gradients
    lapu_basis = u_basis.diff(2, 0) + u_basis.diff(0, 2)
    lapv_basis = v_basis.diff(2, 0) + v_basis.diff(0, 2)
    dxp_basis = p_basis.diff(1,0)
    dyp_basis = p_basis.diff(0,1)


    ### Problem parameters

    # Inner force
    f_basis = lapu_basis - dxp_basis
    g_basis = lapv_basis - dyp_basis

    # PDE x component
    pde_basis = lapu_basis - dxp_basis - f_basis

    # Upper boundary condition
    uTop_basis = u_basis.reduce_eval(1, axis=1)
    vTop_basis = v_basis.reduce_eval(1, axis=1)


    # Lower boundary velocities
    uBot_basis = u_basis.reduce_eval(-1, axis=1)
    vBot_basis = v_basis.reduce_eval(-1, axis=1)


    # Lower boundary slip coefficient
    duBot_basis = u_basis.diff(0,1).reduce_eval(-1, axis=1)
    beta_basis = uBot_basis + alpha_basis * duBot_basis

    
    uSol_basis,vSol_basis,info = solveRobinStokes_fromBasis(f_basis, 
                                  g_basis, 
                                  alpha_basis, 
                                  uTop_basis, 
                                  beta_basis, 
                                  vTop_basis, 
                                  vBot_basis, 
                                  xDim, yDim)
    
    #plotargs = {"ymax="}
    ylim=[-5,5]
    plotargs={"linewidth": 3}
    
    # VELOCITY AT Y = -1
    plt.figure(figsize=(15,4))
    plt.subplot(121)
    plt.title("U component at y=-1")
    uBot_basis.plot(plt.gca(), 101)#, **plotargs)
    plt.ylim(ylim)

    plt.subplot(122)
    plt.title("V component at y=-1")
    vBot_basis.plot(plt.gca(), 101)#, **plotargs)
    plt.ylim(ylim)

    # Y DERIVATIVE AND SLIP COEF AT Y = -1
    plt.figure(figsize=(15,4))
    plt.subplot(121)
    plt.title("dyU at y=-1")
    duBot_basis.plot(plt.gca(), 101, **plotargs)
    plt.ylim(ylim)

    plt.subplot(122)
    plt.title("beta at y=-1")
    beta_basis.plot(plt.gca(), 101, **plotargs)
    plt.ylim(ylim)

    # VELOCITY AT Y = 1
    plt.figure(figsize=(15,4))
    plt.subplot(121)
    plt.title("U component at y=1")
    uTop_basis.plot(plt.gca(), 101, **plotargs)
    plt.ylim(ylim)

    plt.subplot(122)
    plt.title("V component at y=1")
    vTop_basis.plot(plt.gca(), 101, **plotargs)
    plt.ylim(ylim)

    
    ### TESTING THAT THE BASIS WORKS
    plt.figure(figsize=(15,20))
    plt.subplot(321)
    plt.title("Lap U component")
    lapu_basis.plot(plt.gca())
    plt.subplot(322)
    plt.title("Lap V component")
    lapv_basis.plot(plt.gca())
    plt.subplot(323)
    plt.title("gradP x component")
    dxp_basis.plot(plt.gca())
    plt.subplot(324)
    plt.title("gradP y component")
    dyp_basis.plot(plt.gca())
    plt.subplot(325)
    plt.title("fx component")
    f_basis.plot(plt.gca())
    plt.subplot(326)
    plt.title("fy component")
    g_basis.plot(plt.gca())
    
    plt.figure(figsize=(15,4))
    plt.subplot(121)
    plt.title("Divergence")
    div_basis.plot(plt.gca(), vmax=1e-9, vmin=-1e-9)
    plt.subplot(122)
    plt.title("PDE x component")
    pde_basis.plot(plt.gca(), vmax=1e-9, vmin=-1e-9)
    
    
    ### TESTING THAT THE SOLVER WORKS
    tol = 1e-6
    plt.figure(figsize=(15,20))
    plt.subplot(321)
    plt.title("True U component")
    u_basis.plot(plt.gca())
    plt.subplot(322)
    plt.title("True V component")
    v_basis.plot(plt.gca())
    plt.subplot(323)
    plt.title("Solver U component")
    uSol_basis.plot(plt.gca())
    plt.subplot(324)
    plt.title("Solver V component")
    vSol_basis.plot(plt.gca())
    plt.subplot(325)
    plt.title("Delta U component")
    (u_basis - uSol_basis).plot(plt.gca(), vmax=tol, vmin=-tol)
    plt.subplot(326)
    plt.title("Delta V component")
    (v_basis - vSol_basis).plot(plt.gca(), vmax=tol, vmin=-tol)
    
    
    
    