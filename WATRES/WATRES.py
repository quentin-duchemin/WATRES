import os
from .models import lightning_interface
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import random
import pyreadr
import pickle
from .dataset import Dataset
from .results import Results

class WATRES(Dataset, Results):
    def __init__(self, pathsite=None, site=None, algo='WATRES', Tmax = 43200, path_model=None, site_name2save=None):
        """
        Main class of the WATRES package to learn transit time distributions of a given watershed.
        
        ...

        Attributes
        ----------
        pathsite: str
            Path of the site considered.
        site: str
            Name of the site considered.
        Tmax: int
            Maximum lag time considered.
        algo : str
            Method used to estimte the TTDs. By default, it is set to the WATRES model. Other options are 'Weibull' or 'AgeDomain' for the J-weighted model.          
        path_model: str
            If not None, we load the saved model.
        site_name2save: str, default None
            If None, the name of the site used to save the model will be `site`. Otherwise, we use the customized name provided: `site_name2save`.
        
        Methods
        -------
        load_model(pathfile)
            Load the model.
        train(BATCH_SIZE, nb_epochs, lr, n_test, n_validation, use_cout4batch, seed, subsampling_resolution, std_input_noise, std_output_noise)
            Train the model.
        """
        if site_name2save is None:
            self.site_name2save = site
        else:
            self.site_name2save = site_name2save
        Dataset.__init__(self)
        Results.__init__(self)
        self.path_model = path_model
        if not(path_model is None):
            self.load_model(self.path_model)
        else:
            assert not(pathsite is None), "Provide the path to the folder of the studied site and the name of the site"
            assert not(site is None), "Provide the path to the folder of the studied site and the name of the site"
        self.algo = algo
        self.Tmax = Tmax
        self.site = site
        self.pathsite = pathsite
        
        for subfolder in ['save', 'data']:
            directory = os.path.join(pathsite, subfolder)
            if not os.path.exists(directory):
                os.makedirs(directory)

    def load_model(self, pathfile):
        """
        Load a pretrained model.
        
        Parameters
        ----------
        pathfile : str
            Path of the pretrained model to load. The extension of the file should be .pth.tar
        """
        dic = torch.load(pathfile, weights_only=False)
        if self.path_model is None:
            self.path_model = dic.get('path_model', None)
        self.Tmax = dic['Tmax']
        self.algo =  dic['algo']
        self.site = dic['site']
        self.site_name2save = dic['site_name2save']
        self.pathsite = dic['pathsite']
        self.lst_train = dic.get('lst_train', None)
        self.timeyear_train = dic.get('timeyear_train', None)
        self.subsampling_resolution = dic.get('subsampling_resolution', None)
        self.std_input_noise = dic.get('std_input_noise', None)
        self.std_output_noise = dic.get('std_output_noise', None)

    def train(self, BATCH_SIZE = 1000, nb_epochs=200, lr=1e-3, n_test=0, n_validation=24*200, n_train=None, use_cout4batch=False, seed=4, subsampling_resolution=None, std_input_noise=None, std_output_noise=None):
        """
        Train a model.
        
        Parameters
        ----------
        BATCH_SIZE : int
            Size of the training set.
        nb_epochs: int
            Number of epochs to train the model.
        lr: float
            Learning rate used to train the model.
        n_test: int
            Number of time points at the end of the dataset to keep as test points (and thus to not use to train the model).
        n_validation: int
            Number of time points between the training set and the beginning of the test set to consider as the validation set.
        n_train: int
            Number of time points between the beginning of the training set and the beginning of the validation set to consider as the training set.
        use_cout4batch: bool
            If True, the training set is constructed sampling on equal number of points within the 4 equally space quantiles of the tracer output data. If False, the same procedure is done by relying on quantiles of the streamflow time series.
        seed: int
            Fixing the seed of the random generator, for reproducibility.
        subsampling_resolution: int, default None
            If not None, the training set is sampled doing as output tracer is available only every `subsampling_resolution` time steps. Warning: this is not sufficient to simulate an experiment where all tracer data would be sample at the resolution `subsampling_resolution`: one needs to modify the intput tracer data by computing a weighted average of the fine resolution input tracer time series.
        std_input_noise: float, default None
            Standard deviation of the zero mean Gaussian noise added to the input tracer data. If None, no noise is added.
        std_output_noise: float, default None
            Standard deviation of the zero mean Gaussian noise added to the output tracer data. If None, no noise is added.
        """
        pathsite = self.pathsite
        site = self.site
        batch_size_backprop = 100
        Tmax = self.Tmax
        algo = self.algo

        J, Q, ET, CJ, Cout = self.get_data(pathsite, site, include_concentration=True, input_noise=None, output_noise=None)
        
        if not(std_input_noise is None):
            CJ = CJ + np.random.normal(0, std_input_noise, size=len(CJ))
        if not(std_output_noise is None):
            Cout = Cout + np.random.normal(0, std_output_noise, size=len(Cout))
            
            
        if n_train is None:
            n_start = Tmax
        else:
            n_start = np.max([len(J)-(n_test+n_validation+n_train), Tmax])

        lst, BATCH_SIZE = self.get_time_points(pathsite, site, BATCH_SIZE, use_cout=use_cout4batch, n_start=n_start, n_end=(-n_test-n_validation-1), seed=seed, subsampling_resolution=subsampling_resolution)
        
        data_batch, timeyear_train = self.get_features(pathsite, site, lst)
        lst_train = lst
        input_size = data_batch.shape[1]
        self.timeyear_train = timeyear_train
            
            
        self.lst_train = lst_train
        Cout_batch = torch.zeros(BATCH_SIZE)
        CJ_batch = torch.zeros((BATCH_SIZE,Tmax))
        J_batch = torch.zeros((BATCH_SIZE,Tmax))
        Q_batch = torch.zeros(BATCH_SIZE)
        ET_batch = torch.zeros(BATCH_SIZE)
        Qinv_batch = torch.zeros((BATCH_SIZE,Tmax))
        ETinv_batch = torch.zeros((BATCH_SIZE,Tmax))
        for i,t in enumerate(lst_train):
            Cout_batch[i] = Cout[t]
            CJ_batch[i,:] = CJ[t-Tmax:t]
            J_batch[i,:]  = J[t-Tmax:t]
            Q_batch[i]  = torch.sum(Q[t-Tmax:t])
            ET_batch[i] = torch.sum(ET[t-Tmax:t])
            ETinv_batch[i,:] = torch.flip(ET[t-Tmax:t], [0])
            Qinv_batch[i,:] = torch.flip(Q[t-Tmax:t], [0])
            
        #EVAL
        lst_total = np.arange(len(CJ)-(n_test+n_validation),len(CJ)-n_test)
        random.shuffle(lst_total)
        lst = np.sort(lst_total[:min([n_validation,100])])


        lst_test = lst
        data_test, timeyear_test = self.get_features(pathsite, site, lst)
    
        Cout_test = torch.zeros(len(lst_test))
        CJ_test = torch.zeros((len(lst_test), Tmax))
        J_test = torch.zeros((len(lst_test), Tmax))
        Q_test = torch.zeros(len(lst_test))
        ET_test = torch.zeros(len(lst_test))
        Qinv_test = torch.zeros((len(lst_test),Tmax))
        ETinv_test = torch.zeros((len(lst_test),Tmax))
    
        for i,t in enumerate(lst_test):
            Cout_test[i] = Cout[t]
            CJ_test[i,:] = CJ[t-Tmax:t]
            J_test[i,:] = J[t-Tmax:t]
            Q_test[i]  = torch.sum(Q[t-Tmax:t])
            ET_test[i] = torch.sum(ET[t-Tmax:t])
            ETinv_test[i,:] = torch.flip(ET[t-Tmax:t], [0])
            Qinv_test[i,:] = torch.flip(Q[t-Tmax:t], [0])
            
            
        if algo=='Weibull':
            model = lightning_interface.LightningWeibull(input_size, Tmax=Tmax)
        elif algo=='WATRES':
            model = lightning_interface.LightningWatres(input_size, Tmax=Tmax)
        elif algo=='AgeDomain':
            model = lightning_interface.LightningAgeDomain(input_size, Tmax=Tmax)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        losses = []
        losses_val = []
        lst_batch = lst_train

        for epoch in range(nb_epochs):
            loss = model.training_step(data_batch, Cout_batch, CJ_batch, J_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            print(epoch, site+'_'+self.algo, loss.item())
            
            if epoch%10==0:
                model.eval()
                with torch.no_grad():
                    loss = model.training_step(data_test, Cout_test, CJ_test, J_test)
                    losses_val.append(loss.item())
                model.train()
            
            if epoch%10==0:
                state = {
                        'BATCH_SIZE': BATCH_SIZE,
                        'algo': self.algo,
                        'Tmax': self.Tmax,
                        'site': self.site,
                        'site_name2save': self.site_name2save,
                        'pathsite': self.pathsite,
                        'loss': losses,
                        'loss_val': losses_val,
                        'state_dict': model.state_dict(),
                        'lst_train': self.lst_train,
                        'timeyear_train': self.timeyear_train,
                        'std_input_noise':std_input_noise,
                        'std_output_noise':std_output_noise,
                        'subsampling_resolution':subsampling_resolution
                        }
                torch.save(state, os.path.join(pathsite, 'save', 'save_{0}_'.format(self.site_name2save)+self.algo+'.pth.tar'))
