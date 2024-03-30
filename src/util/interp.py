import numpy as np

class PiecewiseInterp2D(object):
    def __init__(self, x, y, Z):
        self.xmax = max(x)
        self.xmin = min(x)
        self.ymax = max(y)
        self.ymin = min(y)
        self.xnum = Z.shape[1]
        self.ynum = Z.shape[0]
        self.dx = (self.xmax-self.xmin)/(self.xnum-1)
        self.dy = (self.ymax-self.ymin)/(self.ynum-1)
        self.Z = Z.copy()
    
    def get_idx(self, x):
        return self.get_id(x, self.xmin, self.xmax, self.xnum)
    
    def get_idy(self, y):
        return self.get_id(y, self.ymin, self.ymax, self.ynum)
    
    def get_y(self, idx):
        return self.get_val(idx, self.ymin, self.ymax, self.ynum)
    
    def get_x(self, idx):
        return self.get_val(idx, self.xmin, self.xmax, self.xnum)
    
    def __call__(self, x, y):
        idx = self.get_idx(x)
        idy = self.get_idy(y)
        
        idxp1 = (idx + 1) % self.xnum
        idyp1 = (idy + 1) % self.ynum
        
        xlist = [self.get_x(idx), self.get_x(idxp1)]
        ylist = [self.get_y(idy), self.get_y(idyp1)]
        Zlist = [self.Z[idy, idx], self.Z[idy, idxp1], self.Z[idyp1, idxp1], self.Z[idyp1, idx]]
        
        return self.interp(x, y, xlist, ylist, Zlist, self.weight)
    
    @staticmethod
    def get_id(x, xmin, xmax, n):
        return np.floor((n-1) * (x - xmin)/(xmax - xmin) + 0.1).astype(int)
    
    @staticmethod
    def get_val(idx, xmin, xmax, n):
        return idx/(n-1) * (xmax - xmin) + xmin
    
    def interp(self, x, y, x0, y0, z0, weight):
        w1 = self.weight(x-x0[0], y-y0[0])
        w2 = self.weight(x-x0[1], y-y0[0])
        w3 = self.weight(x-x0[1], y-y0[1])
        w4 = self.weight(x-x0[0], y-y0[1])
        return (w1 * z0[0] + w2 * z0[1] + w3 * z0[2] + w4 * z0[3]) / (w1 + w2 + w3 + w4)
    
    def weight(self, dx, dy):
        dx = np.abs(dx) % (self.xmax-self.xmin)
        dy = np.abs(dy) % (self.ymax-self.ymin)
        w = (self.dx-dx) * (self.dy-dy)
        return w
                     