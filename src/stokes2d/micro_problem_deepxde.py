import deepxde as dde
from deepxde import geometry as gty
from deepxde.geometry.geometry import *
import matplotlib.pyplot as plt


class MicroGeometry(Geometry):

    def __init__(self, y, xlim, ylim):
        """
        Micro Problem For the Multi Scale Stokes Flow.
        :param tau      parametrisation of the lower boundary
        :param epsilon  positive real number such that max(tau) <= epsilon (important for sampling)
        :param height:  height above the boundary to put the top bounding box.

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

        self.dim = 2
        self.xlim = xlim
        self.ylim = ylim
        self.y = y


        # Weights for sampling uniformly from boundaries: [left, top, right, bot]
        x = np.linspace(xlim[0], xlim[1], 200)
        y = self.y(x)
        ylength = np.sum(((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)**0.5)

        self.wall_lengths = np.array([ylim[1] - self.y(xlim[0]),
                                      xlim[1] - xlim[0],
                                      ylim[1] - self.y(xlim[1]),
                                      ylength])
        self.wall_weights = self.wall_lengths/np.sum(self.wall_lengths)

        # Construct bounding box to sample from (We don't need points from the boundary)
        self.bbox = gty.Rectangle(xmin=[xlim[0], ylim[0]], xmax=[xlim[1], ylim[1]])


    def inside(self, x):
        inside_bbox = self.bbox.inside(x)
        inside_bdry = np.where(inside_bbox, x[:, 1] >= self.y(x[:, 0]),
                               np.full(inside_bbox.shape, False, dtype=bool))
        return inside_bdry

    def on_boundary(self, x, tol=1e-8):

        # If on bbox boundary
        on_bbox_boundary = self.bbox.on_boundary(x)#, tol=tol) Change in source code for this to work.
        on_rough_boundary = np.isclose(x[:, 1], self.y(x[:, 0]), atol=tol)
        inside = self.inside(x)
        return np.logical_and(inside, np.logical_or(on_bbox_boundary, on_rough_boundary))

    def random_boundary_points(self, n, random="pseudo"):
        x = np.zeros((n, 2))
        segment_no = np.random.choice(a=list(range(4)), size=(n), p=self.wall_weights)
        bdry_param = np.random.rand(n)

        # Left bdry

        for idx in range(n):
            if segment_no[idx] == 0:
                x[idx, 0] = self.xlim[0]
                x[idx, 1] = bdry_param[idx] * self.wall_lengths[0] + self.y(self.xlim[0])
            elif segment_no[idx] == 1:
                x[idx, 0] = bdry_param[idx] * self.wall_lengths[1] + self.xlim[0]
                x[idx, 1] = self.ylim[1]
            elif segment_no[idx] == 2:
                x[idx, 0] = self.xlim[1]
                x[idx, 1] = bdry_param[idx] * self.wall_lengths[2] + self.y(self.xlim[1])
            else:
                x_rand = bdry_param[idx] * self.wall_lengths[0] + self.y(self.xlim[0])
                x[idx, 0] = x_rand
                x[idx, 1] = self.y(x_rand)

        return x

    def random_points(self, n, random="pseudo"):
        # Importance sampling
        # Slow and steady
        x = self.bbox.random_points(n)
        not_inside = np.logical_not(self.inside(x))
        while np.any(not_inside):
            x[not_inside] = self.bbox.random_points(np.sum(not_inside))
            not_inside = np.logical_not(self.inside(x))
        return x

    def uniform_points(self, n, boundary=True):
        """Uniform grid of points in the bounding box. Apply 'inside' mask to see which is inside the domain"""
        return self.bbox.uniform_points(n, boundary=boundary)

    @staticmethod
    def test_indicators():

        # Create Test Geometry
        xlim = [0,1]
        ylim= [-0.1,1]
        boundary = lambda x: np.sin(20*x)/10
        microProb = MicroGeometry(y=boundary, xlim=xlim, ylim=ylim)


        # Discretise domain with grid
        N = 200
        eps = 0.01
        x = np.linspace(xlim[0]-eps, xlim[1]+eps, N)[None, :]
        y = np.linspace(ylim[0]-eps, ylim[1]+eps, N)[:, None]

        # Convert to mesh matrices
        z = x + 1j*y
        X = np.real(z).flatten()[:, None]
        Y = np.imag(z).flatten()[:, None]
        xy = np.hstack([X,Y])

        # Plot "inside" function
        inside = microProb.inside(xy)

        # Only some should be on the boundary
        on_bdry = microProb.on_boundary(xy, tol=1e-2)

        fig=plt.figure(figsize=(500,100))

        fig.add_subplot(1, 2, 1)
        plt.pcolormesh(np.real(z), np.imag(z), inside.reshape(N, N))
        plt.plot(x.T, boundary(x).T, 'red')

        fig.add_subplot(1, 2, 2)
        plt.pcolormesh(np.real(z), np.imag(z), on_bdry.reshape(N, N))

    @staticmethod
    def test_samplers():
        xlim = [0,1]
        ylim= [-0.1,1]
        boundary = lambda x: np.sin(20*x)/10
        microProb = MicroGeometry(y=boundary, xlim=xlim, ylim=ylim)


        N = 1000
        
        plt.figure(figsize=(500,100))
        
        x = microProb.random_boundary_points(N)
        plt.scatter(x[:,0], x[:, 1], s=5000)


        x = microProb.random_points(N)
        plt.scatter(x[:, 0], x[:, 1], s=5000)



