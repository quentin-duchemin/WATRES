import math
import numpy as np
import pytorch_lightning as pl

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from typing import Any, Callable, Optional
from collections import OrderedDict



def mixture_component(params, t, T=80000, basis=False):
    if params['distri'] == 'beta':
        init = params['init']
        scale = params['scale']
        a = params['a']
        b = params['b']
        #x = np.linspace(beta.ppf(0.01, a, b),min(1,beta.ppf(0.99, a,b)*scale),4*self.age_window[0])
        length0 = int(T*init)
        length1 = int(T*scale)
        x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.99, a,b),T-length0-length1)
        pdf = beta.pdf(x, a=a, b=b)
        pdf = np.concatenate((np.zeros(length0),pdf))
        pdf = np.concatenate((pdf,np.zeros(length1)))
    elif params['distri'] == 'gamma':
        init = params['init']
        scale = params['scale']
        a = params['a']
        length0 = int(T*init)
        length1 = int(T*scale)
        x = np.linspace(gamma.ppf(0.01, a),gamma.ppf(0.99, a),T-length0-length1)
        pdf = gamma.pdf(x, a=a)
        pdf = np.concatenate((np.zeros(length0),pdf))
        pdf = np.concatenate((pdf,np.zeros(length1)))
    elif params['distri'] == 'lognorm':
        init = params['init']
        scale = params['scale']
        a = params['a']
        length0 = int(T*init)
        length1 = int(T*scale)
        x = np.linspace(lognorm.ppf(0.01, a),lognorm.ppf(0.99, a),T-length0-length1)
        pdf = lognorm.pdf(x, a)
        pdf = np.concatenate((np.zeros(length0),pdf))
        pdf = np.concatenate((pdf,np.zeros(length1)))
    elif params['distri'] == 'sinus':
        init = params['init']
        scale = params['scale']
        f = params['f']
        length0 = int(T*init)
        length1 = int(T*scale)
        x = np.linspace(length0, T-length1, T-length0-length1)
        pdf = (1+np.sin(x*2*np.pi*f/8000))
        pdf = np.concatenate((np.zeros(length0),pdf))
        pdf = np.concatenate((pdf,np.zeros(length1)))
    elif params['distri'] == 'cosinus':
        init = params['init']
        scale = params['scale']
        f = params['f']
        length0 = int(T*init)
        length1 = int(T*scale)
        x = np.linspace(length0, T-length1, T-length0-length1)
        pdf = (1+np.cos(x*2*np.pi*f/8000))
        pdf = np.concatenate((np.zeros(length0),pdf))
        pdf = np.concatenate((pdf,np.zeros(length1)))
    pdf /= np.sum(pdf)
    if not(basis):
        J = np.load(path+'J.npy')
        pdf = pdf * J[int(t)-len(pdf):int(t)][::-1]
        pdf /= np.sum(pdf)
    return (np.cumsum(pdf)).reshape(-1,1)
  
        
def compute_convs(params_distri, J, CJ):
    """
    Compute convolutions
    """
    import numpy as np
    import scipy
    from scipy.stats import beta, gamma, lognorm

    n,T = J.shape
    K = len(params_distri)
    conv = torch.zeros((K, n))
    pQ = torch.zeros((K,n,T))
    for k in range(K):
        if params_distri[k]['distri'] == 'beta':
            init = params_distri[k]['init']
            scale = params_distri[k]['scale']
            a = params_distri[k]['a']
            b = params_distri[k]['b']
            length0 = int(T*init)
            length1 = int(T*scale)
            x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.99, a,b),T-length0-length1)
            pdf = beta.pdf(x, a=a, b=b)
            pdf = np.concatenate((np.zeros(length0),pdf))
            pdf = np.concatenate((pdf,np.zeros(length1)))
        elif params_distri[k]['distri'] == 'gamma':
            init = params_distri[k]['init']
            scale = params_distri[k]['scale']
            a = params_distri[k]['a']
            length0 = int(T*init)
            length1 = int(T*scale)
            x = np.linspace(gamma.ppf(0.01, a),gamma.ppf(0.99, a),T-length0-length1)
            pdf = gamma.pdf(x, a=a)
            pdf = np.concatenate((np.zeros(length0),pdf))
            pdf = np.concatenate((pdf,np.zeros(length1)))
        elif params_distri[k]['distri'] == 'lognorm':
            init = params_distri[k]['init']
            scale = params_distri[k]['scale']
            a = params_distri[k]['a']
            length0 = int(T*init)
            length1 = int(T*scale)
            x = np.linspace(lognorm.ppf(0.01, a),lognorm.ppf(0.99, a),T-length0-length1)
            pdf = lognorm.pdf(x, a)
            pdf = np.concatenate((np.zeros(length0),pdf))
            pdf = np.concatenate((pdf,np.zeros(length1)))
        elif params_distri[k]['distri'] == 'sinus':
            init = params_distri[k]['init']
            scale = params_distri[k]['scale']
            f = params_distri[k]['f']
            length0 = int(T*init)
            length1 = int(T*scale)
            x = np.linspace(length0, T-length1, T-length0-length1)
            pdf = (1+np.sin(x*2*np.pi*f/8000))
            pdf = np.concatenate((np.zeros(length0),pdf))
            pdf = np.concatenate((pdf,np.zeros(length1)))
        elif params_distri[k]['distri'] == 'cosinus':
            init = params_distri[k]['init']
            scale = params_distri[k]['scale']
            f = params_distri[k]['f']
            length0 = int(T*init)
            length1 = int(T*scale)
            x = np.linspace(length0, T-length1, T-length0-length1)
            pdf = (1+np.cos(x*2*np.pi*f/8000))
            pdf = np.concatenate((np.zeros(length0),pdf))
            pdf = np.concatenate((pdf,np.zeros(length1)))
        pdf /= np.sum(pdf)
        pdf = torch.tensor(pdf)
        pdfJ =  J * torch.flip(pdf, [0]).unsqueeze(0)
        pdfJnorm = pdfJ / torch.sum(pdfJ, dim=1).unsqueeze(1)
        pQ[k,:,:] = pdfJnorm
        conv[k, :] = torch.sum(CJ*pdfJnorm, dim=1)
    return conv, pQ

def get_distris():
    params_distri_beta = []
    for a in np.linspace(0.3,4,3):
        for scale in [0, 0.1,0.2]:
            if a>1:
                for init in [0]:
                    params_distri_beta.append({'distri':'beta', 'init': init, 'scale':scale, 'a':a, 'b':1})
            else:
                params_distri_beta.append({'distri':'beta', 'init': 0, 'scale':scale, 'a':a, 'b':1})
                
    params_distri_gamma = []
    for a in np.linspace(1,10,3):
        for scale in [0, 0.1,0.2]:
            lsinit = [0]
            if a>1:
                lsinit = [0] 
            for init in lsinit:
                params_distri_gamma.append({'distri':'gamma', 'init': init,  'scale':scale, 'a':a})

    params_distri_lognorm = []
    for a in np.linspace(0.0001,0.9,3):
        for scale in [0, 0.1,0.2]:
            lsinit = [0]
            if a<0.5:
                lsinit = [0]
            for init in lsinit:
                params_distri_lognorm.append({'distri':'lognorm', 'init': init, 'scale':scale, 'a':a})

    params_distri_sinus = []
    for a in np.linspace(1,10,3):
        params_distri_sinus.append({'distri': 'sinus', 'f':a, 'scale':0, 'init':0})

    params_distri_cosinus = []
    for a in np.linspace(1,10,3):
        params_distri_cosinus.append({'distri': 'cosinus',   'f':a, 'scale':0, 'init':0 })

    params_distri = params_distri_beta + params_distri_gamma + params_distri_lognorm
    params_distri += params_distri_sinus + params_distri_cosinus
    return params_distri


   
class AgeDomain(nn.Module):

    def __init__(
        self,
        input_size,
        Tmax: int = 24*365*5,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = 150

            
        self.KpQ = len(get_distris())
        self.Tmax = Tmax

        def init_weights(m):
            if type(m) in [nn.Conv1d,nn.Linear]:
                m.weight.data.fill_(0.)
                
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
        
    
    def forward(self, x,  J, CJ, returnpQ=False, EHS=False, idxsEHS=None):
        pQcoeffs = self.pQcoeffs(x)
        params_distri = get_distris()
        convs, pQall = compute_convs(params_distri, J, CJ)
        Chat = torch.sum(pQcoeffs.T*convs, dim=0)
        pQ = torch.sum((pQcoeffs.T).unsqueeze(2)*pQall, dim=0)
        if EHS:
            return Chat, torch.cumsum(pQ, 1)[:,idxsEHS]
        else:
            if returnpQ:
                return Chat, pQ[:,[24*30*j for j in range(1,10)]], pQ
            else:
                return Chat, pQ[:,[24*30*j for j in range(1,10)]]