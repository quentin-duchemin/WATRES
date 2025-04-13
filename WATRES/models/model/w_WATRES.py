import math
import numpy as np
import pytorch_lightning as pl

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from collections import OrderedDict


import itertools
import numpy as np
import pandas as pd
import scipy.optimize
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

import torch


class BasisW():
    def __init__(self, Tmax=3000):
        order = 3
        self.Tmax = Tmax
        
        # xplot remains uniformly spaced
        xplot = np.linspace(0, 1, Tmax)
                
        # Initialize the matrix basis (mat)
        self.init_mat_basisw()
    
    def init_mat_basisw(self):
        m = self.Tmax
        knots_ref = []
        val = 0
        step = 50
        while val<=m:
            knots_ref.append(val)
            val += step
            step *= 1.5
        
        nk = len(knots_ref)
        degree = 2
        n = nk+degree+1
        knots = np.concatenate((min(knots_ref)*np.ones(degree),knots_ref,max(knots_ref)*np.ones(degree)))
        c = np.zeros(n)
        
        # Generate B-spline basis functions
        basis_functions = []
        evaluation_points = np.linspace(min(knots), max(knots), m)
        
        basis_values = []
        for i in range(n):
            c[i] = 1
            basis = BSpline(knots, c, degree)
            basis_values.append(basis(evaluation_points))
            c[i] = 0
        self.basis_values = np.array(basis_values)[1:-3,:]
        self.n_splines = (self.basis_values).shape[0]
        for i in range(self.n_splines):
            self.basis_values[i,:] /= np.sum(self.basis_values[i,:])
            self.basis_values[i,:] = np.flip(np.cumsum(np.flip(self.basis_values[i,:])))





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
        def init_weights(m):
            if type(m) in [nn.Conv1d,nn.Linear]:
                m.weight.data.fill_(0.)        
        
        self.params_distri = []
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

        self.basisw = BasisW(Tmax=Tmax)
        self.n_splines = self.basisw.basis_values.shape[0]
        self.basis_values = torch.tensor(self.basisw.basis_values).float()
        
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