# Import architecture modules
from architecture.fno_1d import *
from architecture.fno_cnn_1d import *
from architecture.session import *
from architecture.ufno_1d import *
from architecture.unet_1d import *
from architecture.unrolled_1d import *


# Import boundary solvers
from boundary_solvers.blobs import *
from boundary_solvers.gauss_grid_2d import *
from boundary_solvers.geometry_torch import *
from boundary_solvers.geometry import *

# Import hmm modules
from hmm.hmm import *
from hmm.stokes_deep import *
from hmm.stokes import *
    #from hmm.stokes_fenics import *
    #from hmm.stokes_deep_fenics import *

# Import operator modules
from operators.stokes_operator import *

# Import stokes2d modules
from stokes2d.robin_solver import *
from stokes2d.navier_stokes_robin import *

# Import util modules
from util.basis_scaled import *
from util.dashboard import *
from util.gmres import *
from util.interp import *
from util.logger import *
from util.plot_tools import *
from util.random import *
    #from util.mesh_tools import *
