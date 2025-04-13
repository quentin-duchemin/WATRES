import itertools
import numpy as np
import pandas as pd
import scipy.optimize
from .dms_variants.ispline import Isplines
import matplotlib.pyplot as plt
import torch

class BasisW():
    def __init__(self, nsplines=10, Tmax=40000):
        order = 3
        self.Tmax = Tmax
        step = 1/(nsplines-2)
        mesh = np.arange(0,1.01,step)
        xplot = np.linspace(0, 1, Tmax)
        self.isplines = Isplines(order, mesh, xplot)
        self.n = self.isplines.n
        self.init_mat_basisw()
        self.init_mat_basisw_dT()
    
    def init_mat_basisw(self):
        self.mat = torch.zeros((self.n,self.Tmax))
        for i in range(1,self.n + 1):
            self.mat[i-1,:] = torch.flip(torch.tensor(self.isplines.I(i)),[0])
            
    def init_mat_basisw_dT(self):
        self.mat_dT = torch.zeros((self.n,self.Tmax))
        for i in range(1,self.n + 1):
            self.mat_dT[i-1,:] = -torch.flip(torch.tensor(self.isplines.dI_dx(i)),[0])
    
    
