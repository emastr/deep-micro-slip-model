import torch
import numpy as np
import torch.nn as nn
import torch.autograd as agrad


class StokesBoundaryOp(nn.Module):
    
    def __init__(self, z, dz, ddz, w):
        """
        A linear transform from boundary density to dirichlet boundary conditions. 
        Evaluated using jump condition. 
        
        input z: Discretisation of the boundary
        input dz: Discretisation of derivative with respect to parameterisation.
        input ddz: Discretisation of second derivative.
        input w: Quadrature weights for discretisation points.
        """
        super(StokesBoundaryOp, self).__init__()
        self.K   = self.get_mat(z,dz,ddz,w)
        
        
    def forward(self, x):
        """
        Given a tensor input x with shape (M, 2, N), 
        maps each x[m,:,:] onto output y[m,:,:] using a linear map K[m,:,:,:,:]. 
        There is no interaction between values with different m-indices.
        """
        (M,C,N) = x.shape
        assert C == 2, f"Expected 2 channels (complex + real), but got {C}"
        assert M == self.K.shape[0], f"Wrong dimensions for input, must match" + \
        f"{self.K.shape[0]}x{self.K.shape[1]}x{self.K.shape[1]} matrix but found {x.shape[0]}x{x.shape[1]}x{x.shape[2]}"
        y = x.clone()
        for m in range(M):
            y[m,:,:] += torch.tensordot(x[m, :, :], self.K[m, :, :, :, :], [[0,1],[1,3]])
        return y
    
    
    def mse_grad(self, x, y):
        (M,C,N) = x.shape
        assert C == 2, f"Expected 2 channels (complex + real), but got {C}"
        assert M == self.K.shape[0], f"Wrong dimensions for input, must match" + \
        f"{self.K.shape[0]}x{self.K.shape[1]}x{self.K.shape[1]} matrix but found {x.shape[0]}x{x.shape[1]}x{x.shape[2]}"
        dy = self(x)-y
        grad = dy.clone()
        for m in range(M):
            grad[m,:,:] += torch.tensordot(dy[m, :, :], self.K[m, :, :, :, :], [[0,1],[0,2]])
        return grad
    
    
    def transpose(self, x):
        (M,C,N) = x.shape
        assert C == 2, f"Expected 2 channels (complex + real), but got {C}"
        assert M == self.K.shape[0], f"Wrong dimensions for input, must match" + \
                      f"{self.K.shape[0]}x{self.K.shape[1]}x{self.K.shape[1]} matrix but found" +\
                        f"{x.shape[0]}x{x.shape[1]}x{x.shape[2]}"
        
        y = x.clone()
        for m in range(M):
            y[m,:,:] += torch.tensordot(x[m, :, :], self.K[m, :, :, :, :], [[0,1],[0,2]])
        return y
    
    
    @staticmethod
    def get_mat(z:   torch.Tensor,
                dz:  torch.Tensor, 
                ddz: torch.Tensor, 
                w:   torch.Tensor) -> torch.Tensor:
        """
        param: z    (M, N) - discretised boundary
        param: dz   (M, N) - discretised boundary, 1st derivative
        param: ddz  (M, N) - discretised boundary, 2nd derivative
        param: w    (M, N) - quadrature weights
        
        return: 2xM matrices, NxN dimension. As a (M x 2N x 2N) Matrix.
        """
        if len(z.shape)==1:
            z = z[None,:]
            dz = dz[None,:]
            ddz = ddz[None,:]
            w = w[None,:]
        
        (M, N) = z.shape
        assert (z.shape[1] == N) and (dz.shape[1] == N) and (ddz.shape[1] == N) and (w.shape[1] == N), "Non-matching dimensions"
        assert (z.shape[0] == M) and (dz.shape[0] == M) and (ddz.shape[0] == M) and (w.shape[0] == M), "Non-matching dimensions"
        
        dzc = torch.conj(dz)
        
        K = torch.zeros(M, 2, 2, N, N, device=z.device)
        
        # out
        idx = np.arange(0, N)
        for i in range(M):
            for n in idx:
                m = (idx != n)  # Non-singular indices
                assert sum(m) == N - 1

                # K
                k = torch.zeros(1,N).to(dtype=z.dtype, device=z.device)
                
                k[0,m] = (torch.imag(dz[i,m] / (z[i,m] - z[i,n])) * w[i,m] / np.pi).to(z.dtype)
                k[0,n] = (torch.imag(ddz[i,n] / (2 * dz[i,n])) * w[i,n] / np.pi).to(z.dtype)
                
                # K conj
                kc = torch.zeros(1,N).to(dtype=z.dtype, device=z.device)
                
                kc[0,m] = -torch.imag(dz[i,m] * torch.conj(z[i,m] - z[i,n])) / torch.conj(z[i,m] - z[i,n]) ** 2 * w[i,m] / np.pi
                kc[0,n] = -torch.imag(ddz[i,n] * dzc[i,n]) / (2 * dzc[i,n] ** 2) * w[i,n] / np.pi
                
                
                # Put in matrix
                kr, ki = torch.real(k), torch.imag(k)
                kcr, kci = torch.real(kc), torch.imag(kc)
                K[i,0,0,n,:] =  (kr + kcr)
                K[i,0,1,n,:] = -(ki - kci)
                K[i,1,0,n,:] =  (ki + kci)
                K[i,1,1,n,:] =  (kr - kcr)
        
        return K 
    
    
    
class StokesAdjointBoundaryOp(nn.Module):
    
    def __init__(self, z, dz, ddz, w, a):
        """
        A linear transform from boundary density to dirichlet boundary conditions. 
        Evaluated using jump condition. 
        
        input z: Discretisation of the boundary
        input dz: Discretisation of derivative with respect to parameterisation.
        input ddz: Discretisation of second derivative.
        input w: Quadrature weights for discretisation points.
        """
        super(StokesAdjointBoundaryOp, self).__init__()
        self.K   = self.get_mat(z,dz,ddz,w, a)
        
        
    def forward(self, x, idx=None):
        """
        Given a tensor input x with shape (M, 2, N), 
        maps each x[m,:,:] onto output y[m,:,:] using a linear map K[m,:,:,:,:]. 
        There is no interaction between values with different m-indices.
        """
        (M,C,N) = x.shape
        if idx is None:
            idx = range(M)
        assert C == 2, f"Expected 2 channels (complex + real), but got {C}"
        assert M <= self.K.shape[0], f"Wrong dimensions for input, must match" + \
        f"{self.K.shape[0]}x{self.K.shape[1]}x{self.K.shape[1]} matrix but found {x.shape[0]}x{x.shape[1]}x{x.shape[2]}"
        assert M == len(idx), "idx does not match dimension of x"
        y = x.clone()
        for i, m in enumerate(idx):
            y[i,:,:] += torch.tensordot(x[i, :, :], self.K[m, :, :, :, :], [[0,1],[1,3]])
        return y
    
    
    def mse_grad(self, x, y):
        (M,C,N) = x.shape
        assert C == 2, f"Expected 2 channels (complex + real), but got {C}"
        assert M == self.K.shape[0], f"Wrong dimensions for input, must match" + \
        f"{self.K.shape[0]}x{self.K.shape[1]}x{self.K.shape[1]} matrix but found {x.shape[0]}x{x.shape[1]}x{x.shape[2]}"
        dy = self(x)-y
        grad = dy.clone()
        for m in range(M):
            grad[m,:,:] += torch.tensordot(dy[m, :, :] , self.K[m, :, :, :, :], [[0,1],[0,2]])
        return grad
    
    
    def transpose(self, x):
        (M,C,N) = x.shape
        assert C == 2, f"Expected 2 channels (complex + real), but got {C}"
        assert M == self.K.shape[0], f"Wrong dimensions for input, must match" + \
                      f"{self.K.shape[0]}x{self.K.shape[1]}x{self.K.shape[1]} matrix but found" +\
                        f"{x.shape[0]}x{x.shape[1]}x{x.shape[2]}"
        
        y = x.clone()
        for m in range(M):
            y[m,:,:] += torch.tensordot(x[m, :, :], self.K[m, :, :, :, :], [[0,1],[0,2]])
        return y
    
    
    @staticmethod
    def get_mat(z:   torch.Tensor,
                dz:  torch.Tensor, 
                ddz: torch.Tensor, 
                w:   torch.Tensor,
                a) -> torch.Tensor:
        """
        param: z    (M, N) - discretised boundary
        param: dz   (M, N) - discretised boundary, 1st derivative
        param: ddz  (M, N) - discretised boundary, 2nd derivative
        param: w    (M, N) - quadrature weights
        paran; a    (M,  ) - points in the interior of the domain
        
        return: 2xM matrices, NxN dimension. As a (M x 2N x 2N) Matrix.
        """
        if len(z.shape)==1:
            z = z[None,:]
            dz = dz[None,:]
            ddz = ddz[None,:]
            w = w[None,:]
        
        (M, N) = z.shape
        assert (z.shape[1] == N) and (dz.shape[1] == N) and (ddz.shape[1] == N) and (w.shape[1] == N), "Non-matching dimensions"
        assert (z.shape[0] == M) and (dz.shape[0] == M) and (ddz.shape[0] == M) and (w.shape[0] == M), "Non-matching dimensions"
        
        dzc = torch.conj(dz)
        abs_dz = torch.abs(dz)
        
        K = torch.zeros(M, 2, 2, N, N, device=z.device)
        
        
        # out
        idx = np.arange(0, N)
        for i in range(M):
            da = z[i,:] - a[i]
            dac = torch.conj(da)
            corr_integrand =  dz[i,:] / da ** 2 / abs_dz[i,:]
            corr_integrand_conj = torch.conj(corr_integrand)
            
            
            for n in idx:
                m = (idx != n)  # Non-singular indices
                assert sum(m) == N - 1
                
                # Correction
                coef = (1/da[n] - 1/dac[n] + da[n]/dac[n]**2) / (2 * np.pi * 1j) * w[i, n] * abs_dz[i, n]
            
                # K
                k = torch.zeros(1,N).to(dtype=z.dtype, device=z.device)
                
                k[0,m] = (torch.imag(dz[i,m] / (z[i,m] - z[i,n])) * w[i,n] / np.pi * abs_dz[i,n] / abs_dz[i,m]).to(z.dtype)
                k[0,n] = (torch.imag(ddz[i,n] / (2 * dz[i,n])) * w[i,n] / np.pi).to(z.dtype)
                k[0,:] += coef * corr_integrand
                
                # K conj
                kc = torch.zeros(1,N).to(dtype=z.dtype, device=z.device)
                
                kc[0,m] = -torch.imag(dz[i,m] * torch.conj(z[i,m] - z[i,n])) / torch.conj(z[i,m] - z[i,n]) ** 2 \
                            * w[i,n] / np.pi  * abs_dz[i,n] / abs_dz[i,m]
                kc[0,n] = -torch.imag(ddz[i,n] * dzc[i,n]) / (2 * dzc[i,n] ** 2) * w[i,n] / np.pi
                kc[0,:] += coef * corr_integrand_conj
                
                # Put in matrix
                kr, ki = torch.real(k), torch.imag(k)
                kcr, kci = torch.real(kc), torch.imag(kc)
                K[i,0,0,n,:] =  (kr + kcr)
                K[i,1,0,n,:] = -(ki - kci)
                K[i,0,1,n,:] =  (ki + kci)
                K[i,1,1,n,:] =  (kr - kcr)
        
            #K[i,0,0,:,:] += torch.eye(N)
            #K[i,1,1,:,:] += torch.eye(N)
        
        return torch.transpose(K, 3, 4)


class StokesDomainOp(nn.Module):
    def __init__(self, z, dz, w, z_domain, dz_domain, w_domain, grad=0):
        """
        Integral operator across a line inside the domain.
        """
        self.K = self.get_mat(z, dz, w, z_domain, dz_domain, w_domain, grad)
        
    def forward(self, x):
        
        return y
        
    @staticmethod
    def get_mat(z:   torch.Tensor,
                dz:  torch.Tensor, 
                ddz: torch.Tensor, 
                w:   torch.Tensor,
                z_dom: torch.Tensor,
                dz_dom: torch.Tensor,
                w_dom: torch.Tensor) -> torch.Tensor:
        """
        param: z    (M, N) - discretised boundary
        param: dz   (M, N) - discretised boundary, 1st derivative
        param: ddz  (M, N) - discretised boundary, 2nd derivative
        param: w    (M, N) - quadrature weights
        
        return: 2xM matrices, NxN dimension. As a (M x 2N x 2N) Matrix.
        """
        # CHECK VALIDITY OF INPUTS ########
        for ar in (z, dz, dz, w, z_dom, dz_dom, w_dom):
            if len(ar.shape)==1:
                z = z[None,:]
        for ar in (dz, w):
            assert ar.shape == z.shape, "Non-matching dimensions"
        for ar in (dz_dom, w_dom):
            assert ar.shape == z_dom.shape, "Non-matching dimensions"
        
        (M, N) = z.shape
        (M_dom, N_dom) = z_dom.shape
        assert N_dom == M, "First dimension of domain and boundary data must match."
        #############################################################
        
        K = torch.zeros(M, 2, 2, N, N, device=z.device)
        
        # out
        m = np.arange(0, N)
        for i in range(M):
            for n in range(N_dom):
                # K
                k = torch.zeros(1,N).to(dtype=z.dtype, device=z.device)
                k[0,m] = (torch.imag(dz[i,m] / (z[i,m] - z_dom[i,n])) * w[i,m] / np.pi).to(z.dtype)
                
                # K conj
                kc = torch.zeros(1,N).to(dtype=z.dtype, device=z.device)
                kc[0,m] = -torch.imag(dz[i,m] * torch.conj(z[i,m] - z_dom[i,n])) / \
                            torch.conj(z[i,m] - z_dom[i,n]) ** 2 * w[i,m] / np.pi
                
                
                # Put in matrix
                kr, ki = torch.real(k), torch.imag(k)
                kcr, kci = torch.real(kc), torch.imag(kc)
                K[i,0,0,n,:] =  (kr + kcr)
                K[i,0,1,n,:] = -(ki - kci)
                K[i,1,0,n,:] =  (ki + kci)
                K[i,1,1,n,:] =  (kr - kcr)
        
        return K         
        
        