import sys
sys.path.append('/home/emastr/phd/')

import pygmsh as pg
import dolfin as dl
import mshr as ms
import numpy as np
import matplotlib.pyplot as plt
from hmm.stokes_fenics import *
from hmm.hmm import *
from util.basis_scaled import *
from util.mesh_tools import *
from util.random import *
from util.logger import *
from util.plot_tools import *
from dataclasses import dataclass
import meshio
from typing import *