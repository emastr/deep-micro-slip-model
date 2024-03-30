import torch
import torch.nn as nn
# SET SEED
seed = 123
torch.manual_seed(seed)
torch.set_default_dtype(torch.double)



class Unet(nn.Module):
    """
    U-net as presented in the article by Ronneberger,
    But with padding to conserve image dimension.
    """
    def __init__(self, 
                 in_channels, 
                 min_out_channels, 
                 out_channels, 
                 depth, 
                 width=2, # Number of layers in each block
                 #batch_norm=0, 
                 #activation=nn.ReLU, 
                 **kwargs):
        
        # Default arguments for Convnet
        DEFAULTARGS = {"padding_mode" : "circular", 
                       "kernel_size"  : 3, 
                       "padding"      : 1, 
                       "stride"       : 1,
                       "activation"   : nn.ReLU}
        
        for key in DEFAULTARGS.keys():
            if not key in kwargs.keys():
                kwargs[key] = DEFAULTARGS[key]
        
        super(Unet, self).__init__()
        self.expansion = Expansion(min_out_channels, depth, width, **kwargs)
        self.contraction = Contraction(in_channels, min_out_channels, depth, width, **kwargs)
        self.segmentation = nn.Conv1d(in_channels=min_out_channels, out_channels=out_channels, kernel_size=1)

        
    def forward(self, x):
        cont = self.contraction(x)
        exp = self.expansion(cont)
        return self.segmentation(exp)




class Contraction(nn.Module):
    def __init__(self, in_channels, min_out_channels, depth, width, **kwargs):
        super(Contraction, self).__init__()
        self.convBlocks = nn.ModuleList([])
        self.maxPools = nn.ModuleList([])
        self.depth = depth

        out_channels = min_out_channels
        for d in range(depth):
            self.convBlocks.append(ConvBlock(in_channels, out_channels, width, **kwargs))
            if d < depth:
                self.maxPools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels = out_channels * 2


    def forward(self, x):
        outputs: list = [self.convBlocks[0](x)]
        for d in range(1, self.depth):
            outputs.append(self.convBlocks[d](self.maxPools[d-1](outputs[-1])))
        return outputs


class Expansion(nn.Module):
    def __init__(self, min_out_channels, depth, width, **kwargs):
        super(Expansion, self).__init__()
        self.convBlocks = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        self.depth = depth

        out_channels = min_out_channels
        for d in range(depth-1):
            self.convBlocks.append(ConvBlock(2 * out_channels, out_channels, width, **kwargs))
            self.upConvs.append(nn.ConvTranspose1d(2 * out_channels, out_channels, kernel_size=2, stride=2))
            out_channels = out_channels * 2

    def forward(self, x: list):
        out = x[-1]
        for d in reversed(range(self.depth - 1)):
            out = self.convBlocks[d](torch.cat([x[d], self.upConvs[d](out)], dim=1))
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, width, **kwargs):
        """
        Convolution block (C_i, X  , Y  ) -> conv2d
                       -> (C_o, X-2, Y-2) -> reLU
                       -> (C_o, X-2, Y-2) -> conv2d
                       -> (C_o, X-4, Y-4) -> reLu
                       -> (C_o, X-4, Y-4) -> interpolate
                       -> (C_o, X  , Y  )

        :param in_channels: Number of channels in input image
        :param out_channels: Number of features in output
        """
        activation = kwargs.pop("activation", nn.ReLU)
        batch_norm = kwargs.pop("batch_norm", 0)
        
        super(ConvBlock, self).__init__()
        
        sequential = []
        ch = in_channels
        for i in range(width):
            if batch_norm == 0:
                sequential.append(nn.Conv1d(in_channels=ch,
                                            out_channels=out_channels,
                                            **kwargs))
            elif batch_norm==1:
                sequential.append(nn.utils.weight_norm(nn.Conv1d(in_channels=ch,
                                                       out_channels=out_channels,
                                                       **kwargs)))
            elif batch_norm==2:
                sequential.append(nn.BatchNorm1d(num_features=ch))
                sequential.append(nn.Conv1d(in_channels=ch,
                                            out_channels=out_channels,
                                            **kwargs))
            ch = out_channels
                
            
            sequential.append(activation())
            
        self.sequential = nn.Sequential(*sequential)
        

    def forward(self, x):
        return self.sequential(x)

