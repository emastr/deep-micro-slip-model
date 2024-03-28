import torch
import numpy as np
import torch.nn as nn
import torch.autograd as agrad


class StokesOperator(nn.Module):
    
    def __init__(self, z, dz, ddz, w):
        super(StokesOperator, self).__init__()
        self.z   = z
        self.dz  = dz
        self.ddz = ddz
        self.w   = w
        self.K   = self.get_mat(z,dz,ddz,w)
        
    def forward(self, x):
        (M,C,N) = x.shape
        assert C == 2, f"Expected 2 channels (complex + real), but got {C}"
        assert M == self.K.shape[0], f"Wrong dimensions for input, must match" + \
        f"{self.K.shape[0]}x{self.K.shape[1]}x{self.K.shape[1]} matrix but found {x.shape[0]}x{x.shape[1]}x{x.shape[2]}"

        y = x.clone()
        for m in range(M):
            #y[m,:,:] += torch.matmul(x[m, :, :], self.K[m, :, :].permute(1,0))
            y[m,:,:] += torch.tensordot(x[m, :, :], self.K[m, :, :, :, :], [[0,1],[1,3]])
            
        #v = x[:,:C//2,:] + 1j * x[:,C//2:,:]
        #out = StokesOperator.matvec(v, self.z, self.dz, self.ddz, self.w)
        #return torch.cat([torch.real(out), torch.imag(out)], dim=1)
        return y
    
    def mse_grad(self, x, y):
        #y.requires_grad_(True)
        #x.requires_grad_(True)
        #mse = torch.norm(self(x) - y)**2
        #return agrad.grad(mse, x, retain_graph=True, create_graph=True)[0]
        (M,C,N) = x.shape
        assert C == 2, f"Expected 2 channels (complex + real), but got {C}"
        assert M == self.K.shape[0], f"Wrong dimensions for input, must match" + \
        f"{self.K.shape[0]}x{self.K.shape[1]}x{self.K.shape[1]} matrix but found {x.shape[0]}x{x.shape[1]}x{x.shape[2]}"
        
        dy = self(x)-y
        
        grad = dy.clone()
        for m in range(M):
            #y[m,:,:] += torch.matmul(x[m, :, :], self.K[m, :, :].permute(1,0))
            grad[m,:,:] += torch.tensordot(dy[m, :, :], self.K[m, :, :, :, :], [[0,1],[0,2]])
            
        #v = x[:,:C//2,:] + 1j * x[:,C//2:,:]
        #out = StokesOperator.matvec(v, self.z, self.dz, self.ddz, self.w)
        #return torch.cat([torch.real(out), torch.imag(out)], dim=1)
        return grad
    
    def eval_transpose(self, x):
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
    def matvec(v:   torch.Tensor, 
               z:   torch.Tensor,
               dz:  torch.Tensor, 
               ddz: torch.Tensor, 
               w:   torch.Tensor) -> torch.Tensor:
        """
        param: v (D, C, N) - discretised function, D data and C channels.
        param: z      (N,) - discretised boundary
        param: dz     (N,) - discretised boundary, 1st derivative
        param: ddz    (N,) - discretised boundary, 2nd derivative
        param: w      (N,) - quadrature weights
        """
        
        (_, C, N) = v.shape
        assert (z.shape[0] == N) and (dz.shape[0] == N) and (ddz.shape[0] == N) and (w.shape[0] == N), "Non-matching dimensions"
        
        dzc = torch.conj(dz)
        
        # out
        out = v.clone()  
        idx = np.arange(0, N)
        for n in idx:
            m = (idx != n)  # Non-singular indices
            assert sum(m) == N - 1

            # K
            K = torch.zeros_like(z)
            K[m] = (torch.imag(dz[m] / (z[m] - z[n])) * w[m] / np.pi).to(z.dtype)
            K[n] = (torch.imag(ddz[n] / (2 * dz[n])) * w[n] / np.pi).to(z.dtype)

            # K conj
            Kc = torch.zeros_like(z)
            Kc[m] = -torch.imag(dz[m] * torch.conj(z[m] - z[n])) / torch.conj(z[m] - z[n]) ** 2 * w[m] / np.pi
            Kc[n] = -torch.imag(ddz[n] * dzc[n]) / (2 * dzc[n] ** 2) * w[n] / np.pi

            # Integrate    
            out[:, :, n] += torch.tensordot(v, K, [[2],[0]]) + torch.tensordot(torch.conj(v), Kc, [[2], [0]])
        return out
    
    
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
                #k[0,m] = (torch.imag(dz[i,m] / (z[i,m] - z[i,n])) * w[i,m] / np.pi).to(z.dtype)
                
                # K conj
                kc = torch.zeros(1,N).to(dtype=z.dtype, device=z.device)
                
                kc[0,m] = -torch.imag(dz[i,m] * torch.conj(z[i,m] - z[i,n])) / torch.conj(z[i,m] - z[i,n]) ** 2 * w[i,m] / np.pi
                kc[0,n] = -torch.imag(ddz[i,n] * dzc[i,n]) / (2 * dzc[i,n] ** 2) * w[i,n] / np.pi
                #kc[0,m] = (1 / torch.abs(z[i,m] - z[i,n]) * w[i,m] / np.pi).to(z.dtype)
                
                
                # Put in matrix
                kr, ki = torch.real(k), torch.imag(k)
                kcr, kci = torch.real(kc), torch.imag(kc)
                K[i,0,0,n,:] =  (kr + kcr)
                K[i,0,1,n,:] = -(ki - kci)
                K[i,1,0,n,:] =  (ki + kci)
                K[i,1,1,n,:] =  (kr - kcr)
        
        return K 
             
        
        