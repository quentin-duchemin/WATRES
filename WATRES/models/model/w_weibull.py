

import math
import numpy as np
import pytorch_lightning as pl

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from typing import Any, Callable, Optional
from collections import OrderedDict

  

class Weibull(nn.Module):

    def __init__(
        self,
        input_size,
        Tmax: int = 24*365*5,
        KpQ: int = 6, 
        power: int = 1,
        **kwargs,
    ):
        super().__init__()
            

        self.seq_len = 150
            
        self.KpQ = KpQ
        self.Tmax = Tmax
        self.power = power
        
        hidden_sizes_w =[24, 12]

                    
        self.STnormalization = nn.Sequential(
            # First fully connected layer with BatchNorm and ReLU
            nn.Linear(input_size, hidden_sizes_w[0]),
            nn.BatchNorm1d(hidden_sizes_w[0]),
            nn.ReLU(),
            
            # Second fully connected layer with BatchNorm and ReLU
            nn.Linear(hidden_sizes_w[0], hidden_sizes_w[1]),
            nn.BatchNorm1d(hidden_sizes_w[1]),
            nn.ReLU(),
            
            # Final output layer to get the embedding
            nn.Linear(hidden_sizes_w[1], self.seq_len),
            nn.LeakyReLU(),
            nn.Linear(self.seq_len, KpQ),
            nn.Softplus()
            )
        
        def init_weights(m):
            if type(m) in [nn.Conv1d,nn.Linear]:
                m.weight.data.fill_(0.)
                
        self.STnormalization.apply(init_weights)
        
        self.params_distri = []
        for a in [1.,2.,3.,4., 6.]:
            self.params_distri.append({'distri':'beta', 'a':a, 'b':1.})          

            
        self.coeffs_w = nn.Sequential(
            # First fully connected layer with BatchNorm and ReLU
            nn.Linear(input_size, hidden_sizes_w[0]),
            nn.BatchNorm1d(hidden_sizes_w[0]),
            nn.ReLU(),
            
            # Second fully connected layer with BatchNorm and ReLU
            nn.Linear(hidden_sizes_w[0], hidden_sizes_w[1]),
            nn.BatchNorm1d(hidden_sizes_w[1]),
            nn.ReLU(),
            
            # Final output layer to get the embedding
            nn.Linear(hidden_sizes_w[1], self.seq_len),
            nn.LeakyReLU(),
            nn.Linear(self.seq_len, 1),
            )
        
        
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
            nn.Linear(self.seq_len, self.KpQ),
            nn.Softmax(dim=1)
            )

    def forward_w(self, x):
        
        pQcoeffs = self.pQcoeffs(x)
        
        STnormalization = self.STnormalization(x)*2000. +1.
        alphas = self.coeffs_w(x)
        w = torch.exp(-torch.pow(alphas*torch.arange(1,self.Tmax+1).unsqueeze(0)/10000.,self.power))   
        return w
                
                
 
    def forward(self, x,  J, CJ, returnpQ=False, EHS=False, idxsEHS=None):
        
        pQcoeffs = self.pQcoeffs(x)
        
        STnormalization = self.STnormalization(x)*2000. +1.
        alphas = self.coeffs_w(x)
        w = torch.exp(-torch.pow(alphas*torch.arange(1,self.Tmax+1).unsqueeze(0)/10000.,self.power))   
        Jw = torch.flip(J, [1]) * w
        ST = torch.cumsum(Jw,1)
        pQ = torch.zeros((x.shape[0], self.Tmax))
        for k, param in enumerate(self.params_distri):
            a = self.params_distri[k]['a']
            b = self.params_distri[k]['b']
            distgamma = torch.distributions.gamma.Gamma(a, b)
            pdf1 = torch.exp(distgamma.log_prob(ST/STnormalization[:,k].unsqueeze(1)))
            pdf = pdf1 * Jw #Stot
            with torch.no_grad():
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
