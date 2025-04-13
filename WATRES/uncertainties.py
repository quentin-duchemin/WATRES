from .bert_model import tst
from .bert_model import lightning_interface_standard_bert, lightning_interface
import numpy as np
import torch
from .bert_model import masking
import matplotlib.pyplot as plt
import random
import pyreadr
from .bert_model.model import BasisW
import torch.nn.functional as F
import pickle
import os
import pandas as pd

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import torch.nn.functional as F
import pickle
import os


def gamma_pdf(x, a, b):
    """
    Compute the Gamma PDF manually.
    :param x: Input tensor (must be positive)
    :param a: Shape parameter
    :param b: Rate parameter
    :return: Gamma PDF
    """
    a = torch.tensor(a)
    b = torch.tensor(b)
    coeff = (b ** a) / torch.exp(torch.special.gammaln(a))
    pdf = coeff * (x ** (a - 1)) * torch.exp(-b * x)
    pdf = torch.where(x > 0, pdf, torch.tensor(0.0, device=x.device))
    return pdf


class Uncertainties(PyroModule):
    def __init__(self):
        super().__init__()
        

    def forward(self, x, J, CJ, Cout):
        """
        Forward function with Gumbel sampling for each batch element.
        """
        self.gumbel = pyro.param("gumbel", torch.ones(14), constraint=dist.constraints.positive)
        self.sigma = pyro.param("sigma", torch.tensor(1.0), constraint=dist.constraints.positive)
        with torch.no_grad():
            z = self.model.weights_basis.forward(x)

        tau = torch.cumsum(self.gumbel, 0)
        gumbel_dist = dist.Gumbel(0, 1)
        gumbel_samples = gumbel_dist.sample(z.size())

        ztemp = torch.exp((torch.log(z) + gumbel_samples) / tau.unsqueeze(0))
        ztemp = F.normalize(ztemp, p=1, dim=1)

        w = torch.mm(ztemp, self.model.basis_values) 
        Jw = torch.flip(J, [1]) * w
        ST = torch.cumsum(Jw, dim=1)

        with torch.no_grad():
            pQcoeffs = self.model.pQcoeffs(x)

        pQ = torch.zeros((x.shape[0], self.Tmax))
        for k, param in enumerate(self.model.params_distri):
            a = param['a']
            b = param['b']
            pdf1 = gamma_pdf(ST / 2000, a, b)
            pdf = pdf1 * Jw
            pdf = F.normalize(pdf, p=1, dim=1)
            pQ += pQcoeffs[:, k].unsqueeze(1) * pdf

        with pyro.plate("data", Cout.shape[0]):
            Chat = pyro.sample("Chat", dist.Normal(torch.sum(torch.flip(CJ, [1]) * pQ, dim=1), self.sigma), obs=Cout)

    def guide(self, x, J, CJ, Cout):
        pass

    def learn_uncertainties(self, num_iterations=1000):
        lst_train = self.lst_train
        Tmax = self.Tmax

        J, Q, ET, CJ, Cout = self.get_data(self.pathsite, self.site, include_concentration=True)
        data_train, timeyear_train = self.get_data_noBERT(self.pathsite, self.site, lst_train, BATCH_SIZE=len(lst_train))

        model = lightning_interface.LightningSumSquares_noBERT2_bayesian3(data_train.shape[1], Tmax=Tmax)
        model.load_state_dict(torch.load(self.path_model)['state_dict'])

        self.model = model.model
        self.model.eval()

        Cout_train = torch.zeros(len(lst_train))
        CJ_train = torch.zeros((len(lst_train), Tmax))
        J_train = torch.zeros((len(lst_train), Tmax))

        for i, t in enumerate(lst_train):
            Cout_train[i] = Cout[t]
            CJ_train[i, :] = CJ[t - Tmax:t]
            J_train[i, :] = J[t - Tmax:t]

        start_index = (self.path_model).find("BERT4TRANSIT")
        name_model = self.path_model[start_index:-8] # -8 to remove '.pth.tar'

        loss_fn = lambda model, guide: pyro.infer.Trace_ELBO().differentiable_loss(self.forward, self.guide, data_train, J_train, CJ_train, Cout_train)
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss = loss_fn(self.forward, self.guide)
            params = set(site["value"].unconstrained()
                            for site in param_capture.trace.nodes.values())
        optimizer = torch.optim.Adam(params, lr=1e-1, betas=(0.90, 0.999))
        step = 0
        error = False
        while (step<=num_iterations) and not(error):
            try:
                loss = loss_fn(self.forward, self.guide)/len(Cout_train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if step % 1 == 0:
                    print(f"Step {step}: Loss = {loss}")
    
                    # Save parameters
                    params_svi = {name: value.detach().cpu().numpy() for name, value in pyro.get_param_store().items()}
    
                    f = open(os.path.join(self.pathsite, 'save', "params_uncertainties_{0}.pkl".format(name_model)),"wb")
                    pickle.dump(params_svi,f)
                    f.close()
                step += 1
            except:
                error = True

    def get_uncertainties(self, x, J, CJ, n_samples=100):
        # Load saved parameters
        start_index = self.path_model.find("BERT4TRANSIT")
        name_model = self.path_model[start_index:-8]
        filename = os.path.join(self.pathsite, 'save', f"params_uncertainties_{name_model}.pkl")

        model = lightning_interface.LightningSumSquares_noBERT2_bayesian3(x.shape[1], Tmax=self.Tmax)
        model.load_state_dict(torch.load(self.path_model)['state_dict'])

        self.model = model.model
        self.model.eval()
        
        with open(filename, 'rb') as handle:
            params_svi = pickle.load(handle)

        for key, value in params_svi.items():
            params_svi[key] = torch.tensor(value)
        
        # Extract Gumbel parameters and pre-compute values
        tau = torch.cumsum(params_svi['gumbel'], 0)
        batch_size = J.shape[0]
        
        # Compute weights for z and cumulative sum for w (no loops)
        with torch.no_grad():
            z = self.model.weights_basis.forward(x)
            gumbel_dist = dist.Gumbel(0, 1)
            gumbel_samples = gumbel_dist.sample((n_samples,) + z.size())
            
            # Compute ztemp, weights w, and Jw in batch mode
            ztemp = torch.exp((torch.log(z).unsqueeze(0) + gumbel_samples) / tau.unsqueeze(0).unsqueeze(0))
            ztemp = F.normalize(ztemp, p=1, dim=2)
            w = torch.matmul(ztemp, self.model.basis_values)
            Jw = torch.flip(J.unsqueeze(0), [2]) * w
            ST = torch.cumsum(Jw, dim=2)
            
            # Compute pQcoeffs and initialize pQ in batch mode
            pQcoeffs = self.model.pQcoeffs(x).unsqueeze(0).expand(n_samples, -1, -1)
            pQ = torch.zeros((n_samples, batch_size, self.Tmax))
            
            # Compute PDFs and pQ without loops
            for k, param in enumerate(self.model.params_distri):
                a, b = param['a'], param['b']
                pdf1 = gamma_pdf(ST / 2000, a, b)
                pdf = pdf1 * Jw
                pdf = F.normalize(pdf, p=1, dim=2)
                pQ += pQcoeffs[:, :, k].unsqueeze(2) * pdf
            
            # Compute Chat samples using vectorized operations
            Chat = torch.sum(torch.flip(CJ.unsqueeze(0), [2]) * pQ, dim=2)#dist.Normal(torch.sum(torch.flip(CJ.unsqueeze(0), [2]) * pQ, dim=2), params_svi['sigma']).sample()
        
        return Chat, pQ