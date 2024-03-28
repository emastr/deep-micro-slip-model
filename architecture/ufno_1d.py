import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.fno_1d import FNO1d
seed = 123
torch.manual_seed(seed)
torch.set_default_dtype(torch.double)

class SpectralRescale(nn.Module):
    def __init__(self, channels, modes1, rescale_factor):
        super(SpectralRescale, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.channels = channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.rescale_factor = rescale_factor

        self.scale = (1 / (channels*channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes1, dtype=torch.cdouble))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.channels, (x.size(-1)//2 + 1),  device=x.device, dtype=torch.cdouble)
        out_ft[:, :, :self.modes1] = x_ft[:, :, :self.modes1]

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=int(x.size(-1) * self.rescale_factor))
        return x


class UFNO1d(nn.Module):
    """
    U-net as presented in the article by Ronneberger,
    But with padding to conserve image dimension.
    """
    def __init__(self, 
                 in_channels, 
                 min_out_channels, 
                 out_channels, 
                 depth, 
                 modes=10, # Number of fourier modes
                 **kwargs):
        
        # Default arguments for Convnet
        DEFAULTARGS = {}
        
        for key in DEFAULTARGS.keys():
            if not key in kwargs.keys():
                kwargs[key] = DEFAULTARGS[key]
        
        super(UFNO1d, self).__init__()
        self.expansion = Expansion(min_out_channels, depth, modes, **kwargs)
        self.contraction = Contraction(in_channels, min_out_channels, depth, modes, **kwargs)
        self.segmentation = nn.Conv1d(in_channels=min_out_channels, out_channels=out_channels, kernel_size=1)

        
    def forward(self, x):
        cont = self.contraction(x)
        #print([c.shape for c in cont])
        exp = self.expansion(cont)
        return self.segmentation(exp)



class Contraction(nn.Module):
    def __init__(self, in_channels, min_out_channels, depth, modes, **kwargs):
        super(Contraction, self).__init__()
        self.convBlocks = nn.ModuleList([])
        self.maxPools = nn.ModuleList([])
        self.depth = depth

        out_channels = min_out_channels
        for d in range(depth):
            self.convBlocks.append(FNO1d(modes, in_channels, out_channels, **kwargs))
            if d < depth:
                self.maxPools.append(SpectralRescale(out_channels, modes, rescale_factor=0.5))
            in_channels = out_channels
            out_channels = out_channels * 2


    def forward(self, x):
        outputs: list = [self.convBlocks[0](x)]
        for d in range(1, self.depth):
            outputs.append(self.convBlocks[d](self.maxPools[d-1](outputs[-1])))
        return outputs


class Expansion(nn.Module):
    def __init__(self, min_out_channels, depth, modes, **kwargs):
        super(Expansion, self).__init__()
        self.convBlocks = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        self.depth = depth

        out_channels = min_out_channels
        for d in range(depth-1):
            self.convBlocks.append(FNO1d(modes, 3 * out_channels, out_channels, **kwargs))
            self.upConvs.append(SpectralRescale(2 * out_channels, modes, rescale_factor=2))
            out_channels = out_channels * 2

    def forward(self, x: list):
        out = x[-1]
        for d in reversed(range(self.depth - 1)):
            #print(torch.cat([x[d], self.upConvs[d](out)], dim=1).shape)
            #print(self.convBlocks[d].inp.weight.shape)
            #print(self.convBlocks[d].out.weight.shape)
            out = self.convBlocks[d](torch.cat([x[d], self.upConvs[d](out)], dim=1))
        return out


