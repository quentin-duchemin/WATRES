import math
import numpy as np
import pytorch_lightning as pl

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from typing import Any, Callable, Optional
from collections import OrderedDict


import itertools
import numpy as np
import pandas as pd
import scipy.optimize
import sys, os
dms_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(dms_folder)
from .dms_variants.ispline import Isplines
import matplotlib.pyplot as plt
import torch


class BasisW():
    def __init__(self, nsplines=10, Tmax=3000):
        order = 3
        self.Tmax = Tmax
        
        # Define a non-linear mesh, dense near 0 and sparser as we approach Tmax
        mesh = self.create_non_linear_mesh(nsplines)
        
        # xplot remains uniformly spaced
        xplot = np.linspace(0, 1, Tmax)
        
        # Initialize the I-splines using the modified mesh
        self.isplines = Isplines(order, mesh, xplot)
        self.n = self.isplines.n
        
        # Initialize the matrix basis (mat)
        self.init_mat_basisw()
    
    def create_non_linear_mesh(self, nsplines):
        """
        Create a non-linear mesh that is dense near 0 and sparser near Tmax.
        We will reverse the exponential scaling for the mesh to achieve this.
        """
        # Linear space for a baseline
        linear_space = np.linspace(0, 1, nsplines)
        
        # Inverted exponential transformation to create dense mesh near 0
        exponent = 3  # You can adjust this exponent to control density near 0
        non_linear_mesh = 1 - np.power(1 - linear_space, exponent)
        
        return non_linear_mesh
    
    def init_mat_basisw(self):
        # Initialize a matrix to hold the basis functions
        self.mat = torch.zeros((self.n, self.Tmax))
        
        # Fill the matrix with I-spline basis functions
        for i in range(1, self.n + 1):
            # Flip the tensor for each basis function and assign to the matrix
            self.mat[i-1, :] = torch.flip(torch.tensor(self.isplines.I(i)), [0])


import torch
import torch.special  # For the gamma function (torch.special.gammaln)

def gamma_pdf(x, a, b):
    """
    Compute the Gamma PDF manually.
    :param x: Input tensor (must be positive)
    :param a: Shape parameter (concentration)
    :param b: Rate parameter (inverse scale)
    :return: Gamma PDF
    """
    a = torch.tensor(a)
    b = torch.tensor(b)
    # Compute the Gamma PDF using the direct formula
    # Avoid calculating log and exp to prevent numerical instability
    coeff = (b ** a) / torch.exp(torch.special.gammaln(a))  # Normalization factor
    pdf = coeff * (x ** (a - 1)) * torch.exp(-b * x)
    
    # Handle cases where x <= 0 by setting PDF to 0
    pdf = torch.where(x > 0, pdf, torch.tensor(0.0, device=x.device))
    
    return pdf


class Watres(nn.Module):

    def __init__(
        self,
        input_size,
        dmodel: int=10,
        Tmax: int = 24*365*5,
        KpQ: int = 6, 
        **kwargs,
    ):
        super().__init__()
            
            
        self.KpQ = KpQ
        self.Tmax = Tmax
        self.dmodel = dmodel 
        
        self.seq_len = 150
        #self.STnormalization = torch.nn.Parameter(torch.zeros(KpQ))
        
        
        
        def init_weights(m):
            if type(m) in [nn.Conv1d,nn.Linear]:
                m.weight.data.fill_(0.)

        
        
        self.params_distri = []
#         for a in [1.,3.,6.]:
#             for b in [1.,2.]:
#                 self.params_distri.append({'distri':'beta', 'a':a, 'b':b})  
        for a in [1.,2.,3.,4., 6.]:
            self.params_distri.append({'distri':'beta', 'a':a, 'b':1.})          

        hidden_sizes=[64, 128]

        self.pQcoeffs = nn.Sequential(
            # First fully connected layer with BatchNorm and ReLU
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            
            # Second fully connected layer with BatchNorm and ReLU
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            
            # Final output layer to get the embedding
            nn.Linear(hidden_sizes[1], self.seq_len),
            nn.LeakyReLU(),
            nn.Linear(self.seq_len, KpQ),
            nn.Softmax(dim=1)
            )

        #self.STnormalization = nn.Parameter(2000*torch.ones(len(self.params_distri)))

        basisw = BasisW(nsplines=17, Tmax=Tmax)
        self.basis_values = basisw.mat
        self.n_splines = self.basis_values.shape[0]

        
        self.weights_basis = nn.Sequential(
            # First fully connected layer with BatchNorm and ReLU
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            
            # Second fully connected layer with BatchNorm and ReLU
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            
            # Final output layer to get the embedding
            nn.Linear(hidden_sizes[1], self.seq_len),
            nn.LeakyReLU(),
            nn.Linear(self.seq_len, self.n_splines),
            nn.Softmax(dim=1)
            )

    def forward_w(self, x):
        weights = self.weights_basis.forward(x)
        w = torch.mm(weights, self.basis_values) 
        return w
        
    def forward(self, x,  J, CJ, returnpQ=False, EHS=False, idxsEHS=None):
        ztemp = self.weights_basis.forward(x)
        w = torch.mm(ztemp, self.basis_values) 
        Jw = torch.flip(J, [1]) * w
        ST = torch.cumsum(Jw,1)
        pQcoeffs = self.pQcoeffs(x)
        pQ = torch.zeros((x.shape[0], self.Tmax))
        for k, param in enumerate(self.params_distri):
            a = self.params_distri[k]['a']
            b = self.params_distri[k]['b']
            distgamma = torch.distributions.gamma.Gamma(a, b)
            pdf1 = gamma_pdf(ST/2000, a, b)

            pdf = pdf1 * Jw #Stot
            pdf = torch.nn.functional.normalize(pdf, p=1, dim=1)
            pQ = pQ + pQcoeffs[:,k].unsqueeze(1)*pdf
        Chat = torch.sum( torch.flip(CJ, [1]) * pQ, dim=1)
        if EHS:
            return Chat, torch.cumsum(pQ, 1)[:,idxsEHS]
        else:
            if returnpQ:
                return Chat, ST[:,[24*30*j for j in range(1,10)]], torch.cumsum(pQ, 1)[:,[24*30*j for j in range(1,10)]], pQ
            else:
                return Chat, ST[:,[24*30*j for j in range(1,10)]], torch.cumsum(pQ, 1)[:,[24*30*j for j in range(1,10)]]        
