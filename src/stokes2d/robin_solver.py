import scipy.sparse as sp
from scipy.sparse.linalg import gmres
import numpy as np
from scipy.sparse.linalg import spsolve
import time
from util.basis_scaled import Basis, ChebBasis, FourBasis, BasisProduct, ScaleShiftedBasis, ScaleShiftedBasisProduct

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

def solveRobinStokes_fromFunc(f, g, alpha, uTop, beta, vTop, vBot, dom, xDim, yDim, **kwargs):
    """Solve from function handles"""

    logger = kwargs.get("logger", None)

    if logger is not None:
        logger.start_event("solver_project_to_basis")

    f = ScaleShiftedBasisProduct.fromFunctionInDomain(f, dom, xDim, yDim, FourBasis, ChebBasis)
    g = ScaleShiftedBasisProduct.fromFunctionInDomain(g, dom, xDim, yDim, FourBasis, ChebBasis)
    uTop = ScaleShiftedBasis.fromFunctionInDomain(uTop, dom[0], FourBasis, xDim)
    beta = ScaleShiftedBasis.fromFunctionInDomain(beta, dom[0], FourBasis, xDim)
    vTop = ScaleShiftedBasis.fromFunctionInDomain(vTop, dom[0], FourBasis, xDim)
    vBot = ScaleShiftedBasis.fromFunctionInDomain(vBot, dom[0], FourBasis, xDim)

    if isinstance(alpha, ScaleShiftedBasis):
        alpha = alpha.change_dim(xDim)
    else:
        alpha = ScaleShiftedBasis.fromFunctionInDomain(alpha, dom[0], FourBasis, xDim)

    if logger is not None:
        logger.end_event("solver_project_to_basis")
    
    return solveRobinStokes_fromBasis(f, g, alpha, uTop, beta, vTop, vBot, dom, xDim, yDim, **kwargs)


def solveRobinStokes_fromBasis(f, g, alpha, uTop, beta, vTop, vBot, dom, xDim, yDim, **kwargs):
    """Solve from Basis Function objects"""
    alpha_0 = np.mean(alpha.eval_grid())
    uC,vC,info = solveRobinStokesIterative(f, g, alpha, uTop, beta, vTop, vBot, dom, xDim, yDim, alpha0=alpha_0, **kwargs)
    
    u = BasisProduct(uC, xDim, yDim, FourBasis, ChebBasis)
    v = BasisProduct(vC, xDim, yDim, FourBasis, ChebBasis)
    
    scale, shift = ScaleShiftedBasisProduct._domain_to_scale_shift(u.domain(), dom)
    ub = ScaleShiftedBasisProduct(u, scale, shift)
    vb = ScaleShiftedBasisProduct(v, scale, shift)
    
    return ub, vb, info

def get_pde_matrix(f, g, Lx, Ly, Nx, Ny):
    # Make sure the dimensions match. Truncate or zero pad.
    f = f.reshape(Nx, Ny)
    g = g.reshape(Nx, Ny)
    
    # Laplace system 
    kmax = (Nx-1) // 2
    ns = arange(0, Ny)
    ks = arange(-kmax,kmax+1)
    cn = lambda n: 1. + (n==0) 
    # To correct low order coefficients!

    # Inverse d/dy matrix
    nVals = arange(1,Ny)[None, :]
    dyValm1 = cn(nVals-1) / (2*nVals)
    dyValp1 = -1 / (2*nVals)
    iDy = spdiags(vstack([dyValm1, dyValp1]), [0,2], (Ny-1,Ny)) * Ly

    # Inverse dd/ddy matrix
    nVals = arange(2,Ny)[None, :]
    ddyValp2 = 1./(4.*nVals * (nVals+1))
    ddyVal0  = -1./(2.*(nVals ** 2 - 1))
    ddyValm2 = cn(nVals-2)/(4 * nVals * (nVals-1))
    iDy2 = spdiags(vstack([ddyValm2, ddyVal0, ddyValp2]), \
                         [0,2,4], (Ny-2, Ny)) * Ly ** 2

    
    # Inverse ddd/dddy matrix
    nVals = arange(3, Ny)[None, :]
    dddyValp3 = -1 / (8*nVals * (nVals+1) * (nVals+2))
    dddyValp1 =  3 / (8*(nVals+2) * nVals * (nVals-1))
    dddyValm1 = -3 / (8*(nVals-2) * nVals *(nVals+1)) 
    dddyValm3 = cn(nVals-3) / (8*nVals * (nVals-1) * (nVals-2))

    iDy3 = spdiags(vstack([dddyValm3, dddyValm1, dddyValp1, dddyValp3]),\
                          [0,2,4,6], (Ny-3, Ny)) * Ly ** 3

    Dx  = spdiags((2j*np.pi*(ks[None, :])) ** 1, [0], (Nx, Nx)) / Lx**1
    Dx2 = spdiags((2j*np.pi*(ks[None, :])) ** 2, [0], (Nx, Nx)) / Lx**2
    Dx3 = spdiags((2j*np.pi*(ks[None, :])) ** 3, [0], (Nx, Nx)) / Lx**3

    Idx  = speye(Nx)
    Idy  = spdiags(ones((1, Ny-1)), [1], (Ny-1, Ny))
    Idy2 = spdiags(ones((1, Ny-2)), [2], (Ny-2, Ny))
    Idy3 = spdiags(ones((1, Ny-3)), [3], (Ny-3, Ny))

    unit_const = (ks == 0)
    zero_const = sparse_coo([[],[]],[],((Ny-2),Nx*Ny))
    
    # B system
    #print(By.shape, By2.shape, By.shape)
    B11 = kron(Dx2[:kmax,:], iDy2[1:, :])+kron(Idx[:kmax,:], Idy3)
    B12 = -(kron(Dx3[:kmax,:], iDy3)+kron(Dx[:kmax,:], iDy[2:, :]))
    B21 = kron(unit_const, Idy2)
    B22 = zero_const
    B31 = kron(Dx2[kmax+1:,:], iDy2[1:, :])+kron(Idx[kmax+1:,:], Idy3)
    B32 = -(kron(Dx3[kmax+1:,:], iDy3)+kron(Dx[kmax+1:,:],iDy[2:, :]))
    B41 = kron(Dx, iDy)
    B42 = kron(Idx, Idy)
                  
    B_pde = vstackSP([hstackSP([B11, B12]),
                      hstackSP([B21, B22]),
                      hstackSP([B31, B32]),
                      hstackSP([B41, B42])])
    
    
    b_mom = kron(Idx, iDy2[1:,:]) @ f.reshape(Ny*Nx, 1) \
                  - kron(Dx, iDy3) @ g.reshape(Ny*Nx, 1)
    b_pde = vstack([b_mom[0:(kmax*(Ny-3)),:],
                      iDy2 @ (f[kmax,:, None]),
                      b_mom[(kmax+1)*(Ny-3):,:],
                      zeros(((Ny-1)*Nx,1))])    
    
    return B_pde, b_pde


def get_bdry_matrix(alpha, Lx, Ly, Nx, Ny):
    kmax = (Nx-1) // 2
    ns = arange(0, Ny)
    ks = arange(-kmax,kmax+1)
     
    alpha = alpha.reshape(Nx,1)

    Idx  = speye(Nx)
    
    # Eval
    botEval = kron(Idx, (-1) ** ns)
    topEval = kron(Idx, ones(ns.shape))
    dirZero = sparse_coo([[], []], [], (Nx, Ny*Nx))
    dyEval  = ((ns ** 2) *(-1) ** (ns+1)) / Ly
    
    AMat = spdiags(flip(alpha).repeat(Nx, 1), ks, (Nx, Nx))
    botRobi = botEval + kron(AMat, dyEval)
     

    B_bdry = vstackSP([hstackSP([botRobi, dirZero]),\
                      hstackSP([dirZero, botEval]),\
                      hstackSP([topEval, dirZero]),\
                      hstackSP([dirZero[ks!=0,:], topEval[ks!=0, :]])\
                    ])
    return B_bdry


def get_bdry_vec(uTop, beta, vTop, vBot, Lx, Ly, Nx, Ny):
    kmax = (Nx-1) // 2
    ks = arange(-kmax,kmax+1)
    uTop = uTop.reshape(Nx,1)
    beta = beta.reshape(Nx,1)
    vTop = vTop.reshape(Nx,1)
    vBot = vBot.reshape(Nx,1) 
    b_bdry = vstack([beta, vBot, uTop, vTop[ks!=0,:]])
    return b_bdry



def solveRobinStokesIterative(f, g, alpha, uTop, beta, vTop, vBot, dom, Nx, Ny, alpha0=None, tol=1e-3, logger=None):
    """SOLVEROBINSTOKES   Solve 1-periodic Stokes equations on 2D domain [0,Lx]x[0, Ly].
     The inputs must be ChebFourFun - objects.

     f:[0,Lx]x[0,Ly] -> R  force in u-direction
     g:[0,Lx]x[0,Ly] -> R  force in v-direction
     alpha:[0,Lx] -> R     variable shear constant
     uTop:[0,Lx] -> R      Dirichlet condition for u at y=1
     beta:[0,Lx] -> R      Inhom robin condition for u at y=-1
     vTop:[0,Lx] -> R      Dirichlet condition for v at y=1
     vBot:[0,Lx] -> R      Dirichlet condition for v at y=-1

     solveRobinStokes then solves The following boundary value problem:

         PDE: Moment equations
           (DxDx + DyDy)*u - Dx*p = f on [0,Lx]x[0,Ly]
           (DxDx + DyDy)*v - Dy*p = g on [0,Lx]x[0,Ly]
         PDE: Continuity Equations
                      Dx*u + Dy*v = 0 on [0,Lx]x[0,Ly]
         BC: Boundary conditions
                         u - uTop = 0 on [0,Lx]x{1}
            u + alpha*Dy*u - beta = 0 on [0,Lx]x{-1}
                         v - vTop = 0 on [0,Lx]x{1}
                         v - vBot = 0 on [0,Lx]x{-1}
     Returns
      u, v. p must be solved using a pressure solver."""

    f_coef = f.basis.coef
    g_coef = g.basis.coef
    uTop_coef = uTop.basis.coef
    vTop_coef = vTop.basis.coef
    vBot_coef = vBot.basis.coef
    uBot = 0. * alpha
    beta0_coef = (beta - (alpha - alpha0) * uBot).basis.coef
    alpha0_coef = (alpha * 0. + alpha0).basis.coef
    
    
    info = dict()
    dom_from = [FourBasis._domain(), ChebBasis._domain()]
    scale, shift = ScaleShiftedBasisProduct._domain_to_scale_shift(dom_from, dom)
    Lx, Ly = scale[0], scale[1]
    
    
    if logger is not None:
        logger.start_event("solver_build_system")

    t = time.time()
    B_pde, b_pde = get_pde_matrix(f_coef, g_coef, Lx, Ly, Nx, Ny)
    B_bdry = get_bdry_matrix(alpha0_coef, Lx, Ly, Nx, Ny)
    B = vstackSP([B_pde, B_bdry])
    info["buildTime"] = time.time() - t

    if logger is not None:
        logger.end_event("solver_build_system")
    
    
    info["solveTime"] = []
    info["resList"] = []
    info["betaCompTime"] = []
    
    res = tol + 1
    L = Nx*Ny
    idx = arange(0, L)

    while res > tol:        
        # Compute rhs vector
        b_bdry = get_bdry_vec(uTop_coef, beta0_coef, vTop_coef, vBot_coef, Lx, Ly, Nx, Ny)
        b = vstack([b_pde, b_bdry])    
        
        # Solve the system

        if logger is not None:
            logger.start_event("solver_solve")

        t = time.time()
        uv = solve(B, b)
        info["solveTime"].append(time.time() - t)
        

        if logger is not None:
            logger.end_event("solver_solve")
            logger.start_event("solver_beta_compute")
        
        
        # Evaluate solution at boundary to obtain new solution
        t = time.time()
        uC = uv[idx+0*L].reshape(Nx, Ny)
        u = BasisProduct(uC, uC.shape[0], uC.shape[1], FourBasis, ChebBasis)
        u = ScaleShiftedBasisProduct(u, scale, shift)
        uBot = u.reduce_eval(dom[1][0], axis=1)
        duBot = u.diff(0, 1).reduce_eval(dom[1][0], axis=1)  ##  THIS IS EXPENSIVE AND UNNECESSARY
        rhs = beta - (alpha - alpha0) * duBot 
        beta0_coef = rhs.basis.coef
        info["betaCompTime"].append(time.time()-t)
        
        if logger is not None:
            logger.end_event("solver_beta_compute")
        
        res = np.linalg.norm((uBot + alpha * duBot - beta).eval_grid())
        info["resList"].append(res)
    
    info["solveTime"] = np.array(info["solveTime"])
    info["resList"] = np.array(info["resList"])
    info["betaCompTime"] = np.array(info["betaCompTime"])
    
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
    
    
    
    