import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

DEFAULT_PATH = "/home/emastr/phd/data/figures_dashboard/"

class DashFigure():
    
    def __init__(self, fig: Figure, path: str):
        self.path = path
        self.fig = fig
        self.save()
        
    def save(self):
        self.fig.savefig(self.path)
        

class DashBoard():
    
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self.tracked_figures = []
        self.num_figures = 0
        os.makedirs(self.path, exist_ok=True)
    
    def add_figure(self, fig: Figure):    
        self.tracked_figures.append(DashFigure(fig, self.path + f"figure_{len(self.tracked_figures)}.jpeg"))
    
    def update_all(self):
        for fig in self.tracked_figures:
            fig.save()
    
        
    
        