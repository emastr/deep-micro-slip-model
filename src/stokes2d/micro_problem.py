from typing import Dict, List, Tuple
import deepxde as dde
import deepxde.geometry as dxGeom
import deepxde.geometry.geometry as dxGeom2
from deepxde.geometry.geometry import *
from stokes2d.deepxde.geometry import MicroGeometry
import matplotlib.pyplot as plt
import torch.nn as nn

        
        
    
class MicroDomain:
    
    def __init__(self, y: nn.Module, xlims: List[float], ylims: List[float], **kwargs):
        """
        Micro Problem For the Multi Scale Stokes Flow.
        :param y:   Parametrisation of the lower boundary. Torch function.
        :param xlims:  List of floats for left and right edges of domain.
        :param ylims:  List of floats for max and min y value attained along boundary.
 
         xlims[0]                xlims[1]
            |                       |
            v                       v
            _________________________   ___   <-- ylims[1]
            |                       |    |
            |                       |    |
            |        Geometry       |    |
            |                       |    | 
            |-----------------------|   -|-   <-- y=0
            |         _____      ___|    |
            |        /     \    /        |
             \______/       \__/        _|_   <-- ylims[0]
                ^
                |
              y = yfunc(x)
        """
        
        # Base properties
        self.xlims = xlims
        self.ylims = ylims
        self.yfunc = yfunc
        self.yedge = [yfunc(xlims[0]), yfunc(xlims[1])]   
        
        return None
    
    # Base functions
    
    
    
    # Transform to other libraries
    
    def to_deepxde(self) -> MicroGeometry:
        """
        Transform to deepxde domain object
        """
        return MicroGeometry(self.y, self.xlim, self.ylim)
    
    
    def to_fenics(self):
        """
        Transform to fenics mesh object
        """
        return MicroDomainFenics(self)
    
    
    def to_matlab(self):
        """
        Transform to matlab compatible object
        """
        return domain_mlab


    @staticmethod
    def test():
        help(dxGeom2)

    

    
                
            
    
