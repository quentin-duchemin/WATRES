from typing import Callable, Optional
import numpy as np
import torch
from torch.functional import Tensor
from torch.optim import Adam
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from .model.basisw import *
    
from .model.w_weibull import Weibull
class LightningWeibull(pl.LightningModule):
    def __init__(
        self,
        input_size,
        Tmax: int = 24*365*5,
        KpQ: int = 6,
        power: int = 1
    ):
     
        super().__init__()
        self.model = Weibull(
            input_size,
            Tmax=Tmax,
            KpQ=KpQ,
            power=power
        )        
        self.KpQ = KpQ
        self.Tmax = Tmax

    def forward_w(self, x):
        return self.model.forward_w(x)
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, Cout, CJ, J):
        Chat, STref, pQ_ywf_hat = self.model.forward(batch, J, CJ)
        loss =  F.mse_loss(Cout, Chat)
        return loss

    
    

from .model.w_WATRES import Watres
class LightningWatres(pl.LightningModule):
    def __init__(
        self,
        input_size,
        mean_input_tracer,
        Tmax: int = 24*365*5,
        KpQ: int = 6,
    ):
     
        super().__init__()
        self.model = Watres(
            input_size,
            mean_input_tracer,
            Tmax=Tmax
        )      
        self.KpQ = self.model.KpQ
        self.Tmax = Tmax
        self.input_size = input_size
        
    def forward(self, x):
        return self.model(x)

    def forward_w(self, x):
        return self.model.forward_w(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def training_step_w(self, batch, J, rSTO):
        STref = self.model.forward_w(batch, J)
        loss = F.mse_loss(rSTO, STref)
        return loss
    
    def training_step(self, batch, Cout, CJ, J):
        Chat, STref, pQ_ywf_hat = self.model.forward(batch, J, CJ)
        loss =  F.mse_loss(Cout, Chat)
        return loss

    def training_step_EHS(self, batch, Cout, CJ, J, pQ_ywf_EHS, idxsEHS, clusters, lambEHS=1.):
        Chat, pQ_ywf_hat = self.model.forward(batch, J, CJ, EHS=True, idxsEHS=idxsEHS)
        loss =  F.mse_loss(Cout, Chat)
        for k, clust in enumerate(clusters):
            loss += lambEHS * F.mse_loss(torch.mean(pQ_ywf_hat[np.array(clust),:], dim=0), pQ_ywf_EHS[k,:])
        return loss    


from .model.w_age_domain import AgeDomain
class LightningAgeDomain(pl.LightningModule):
    def __init__(
        self,
        input_size,
        Tmax: int = 24*365*5
    ):
     
        super().__init__()
        self.model = AgeDomain(
            input_size,
            Tmax=Tmax
        )      
        self.KpQ = self.model.KpQ
        self.Tmax = Tmax
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
        
    def training_step(self, batch, Cout, CJ, J):
        Chat, pQ_ywf_hat = self.model.forward(batch, J, CJ)
        loss =  F.mse_loss(Cout, Chat)
        return loss


