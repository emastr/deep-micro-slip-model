import torch
from torch import Tensor
import typing
from torch.nn.functional import normalize
from torch import tensordot as dot
from torch.linalg import vector_norm as norm

def gram_schmidt(x: Tensor, B: Tensor):
    """
    Tensor x: shape (1, M) <- Representing NxC vectors in R^M
    Tensor B: shape (R, M) <- Representing NxC vectors in R^M
    
    h0 = B' * x (pointwise dot products) -> (C, R)
    """
    
    h0 = dot(x, B, dims=[[1], [1]])             # dim (1, R)
    x0 = x - dot(h0, B, dims=[[1], [0]])        # dim (1, M) - ((1, R) x (R, M)) = # dim (1, M)

    h1 = dot(x0, B, dims=[[1], [1]])            # dim (1, R)
    x1 = x0 - dot(h1, B, dims=[[1], [0]])       # dim (1, M) - ((1, R) x (R, M)) = # dim (1, M) 
    
    h = h0 + h1                                             # dim (1, R)
    b = norm(x1, dim=1, keepdim=True)   # dim (1, 1)        
    
    x1 = x1 / b
    
    return x1, h, b



def arnoldi(op, b, steps, callback=None):
    """
    nn.Module op: takes (C, M) input, outputs of shape (C, M).
    Tensor b: (1, M)
    """
    assert steps < b.shape[1], f"Hey, steps must be at most {b.shape[1]}"
    
    q = b / norm(b, dim=1)  # dim (C, M)
    Q = torch.zeros(steps+1, b.shape[1]) # dim (C, M)
    Q[0, :] = torch.ones(q.shape[1])
    Q = Q * q
    for m in range(steps):
        # Next element in krylov space
        x = op(q)
        
        # Orthogonalise against Q and update
        (q, h, beta) = gram_schmidt(x, Q[:(m+1), :])
        
        # Create H
        # Hij = dot(q_i, (Aq)_;j) = q_i' * Aq_j
        # Behöver endast beräkna q_m' * Aq_j  och q_j * Aqm
        AQ = op(Q[:(m+1), :]) # -> (C, M)
        H = dot(Q[:(m+1), :], AQ, dims=[[1],[1]]) # (C, C) # unnecessary double counting but ok
        
        # Do callback
        stop = False
        if callback is not None:
            stop = callback(Q[:(m+1),:], q, H, beta, m+1)
        if stop:
            break
        mask = torch.zeros_like(Q)
        mask[m+1, :] = torch.ones_like(q)
        Q = Q + mask * q
    return Q, q, H, beta



def gmres(op, b, steps, callback=None, verbose=False):
    """
    GMRES implementation in pytorch
    """
    Q, q, H, beta = arnoldi(op, b, steps, callback=callback)
    
    normb = norm(b, dim=1)
    m = steps
    Q = Q[:m, :]
    
    em = torch.zeros(m, m)
    em[m-1, m-1] = 1

    e1 = torch.zeros(1, m)
    e1[0, 0] = 1

    HmTHm = dot(H, H, dims=[[0], [0]]) + em * beta**2
    HmTbe = dot(normb*e1, H, dims=[[1], [0]])

    z = torch.linalg.solve(HmTHm[None,:,:], HmTbe)
    x = dot(z, Q, dims=[[1],[0]])
    return x