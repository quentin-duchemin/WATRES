import numpy as np
import torch
import pandas as pd
import os
import random

class Dataset():
    """
    Dataset class of the WATRES package to learn transit time distributions of a given watershed.
    """
    def __init__(self):
        pass

    def datetime2year_month(self, dt):
        a = str(dt.year)
        a = a+'-'+str(dt.month)
        return(a)

    def get_time_points(self, pathsite, site,  BATCH_SIZE, use_cout=False, n_start=0, n_end=-1, seed=4, subsampling_resolution=None):
        """
        Selects time points to make sure that the batch contains a representative sample of the discharge data (or of the output tracer data is `use_cout` is True).
        """
        df = pd.read_csv(os.path.join(pathsite, 'data', site+'.txt'), sep='\t')
        if df.shape[1]==1:
            df = pd.read_csv(os.path.join(pathsite, 'data', site+'.txt'))
            
        Cout = df.loc[:,'Cq']
        lst = np.arange(0,len(Cout),1).astype(int)[:n_end][n_start:]
        if not(subsampling_resolution is None):
            n = len(lst)
            filter_idxs = np.array([i*subsampling_resolution for i in range(int(n//subsampling_resolution))])
            lst = lst[filter_idxs]
        
        idxs = np.where(~np.isnan(Cout))[0]
        lst = np.intersect1d(idxs, lst)

        BATCH_SIZE = np.min([BATCH_SIZE, len(lst)])
        batch = BATCH_SIZE // 4
        if use_cout:
            Q = Cout
        else:
            Q = df.loc[:, 'q']
        Qsorted = np.sort(np.array(Q)[lst])
        quantiles = []
        for l in [0.25*i for i in range(1,5)]:
             quantiles.append(Qsorted[max([0,int(l*len(Qsorted)-1)])])    
    
        clustersQ = {k:[] for k in range(len(quantiles))}
        for i in range(len(lst)):
            k = 0
            while quantiles[k]+1e-4<=Q[int(lst[i])]:
                k += 1
            clustersQ[k].append(lst[i])
        lst_train = []
        for k in range(4):
            #print(len(clustersQ[k]))
            random.Random(seed).shuffle(clustersQ[k])
            lst_train += list(clustersQ[k][:batch])
        return np.sort(np.array(lst_train)), len(lst_train)



    def get_features(self, pathsite, site, lst):
        """
        Compute the feature vectors to feed to the neural networks.
        
        Parameters
        ----------
        pathsite: str
            Path of the site considered.
        site: str
            Name of the site considered.
        lst: array of int
            List of indexes of the time points for which we want to compute the feature vector.
        """
        J, Q, ET, timeyear = self.get_data(pathsite, site, include_concentration=False, input_noise=None, output_noise=None, get_timeyear=True)
        features = [6, 12, 24, 24*15, 24*30, 24*30*6, 24*30*12, 24*30*18, 24*30*24]
        data_batch = torch.zeros((len(lst), 2+3*len(features)))
        dates = timeyear[lst]
        arg_perio = (dates-(dates).to(torch.int))*2*np.pi
        for j, lag in enumerate(features):
            ls = np.cumsum(np.ones(lag+1))
            weights = np.flip( np.exp(-4*ls/ls[-1]) )
            weights /= np.sum(weights)
            for i, t in enumerate(lst):
                data_batch[i,3*j] = torch.sum(J[t-lag:t+1]*weights)
                data_batch[i,3*j+1] = torch.sum(ET[t-lag:t+1]*weights)
                data_batch[i,3*j+2] = torch.sum(Q[t-lag:t+1]*weights)
        data_batch[:,-2] = np.cos(arg_perio)
        data_batch[:,-1] = np.sin(arg_perio)
        return data_batch, timeyear[lst]
        
    def get_data(self, pathsite, site, include_concentration=True, input_noise=None, output_noise=None, get_timeyear=False):
        """
        Return the flow time series (and the tracer ones also if `include_concentration`=True).
        
        Parameters
        ----------
        pathsite: str
            Path of the site considered.
        site: str
            Name of the site considered.
        include_concentration: bool
            If True, the tracer data is also provided (with precipitation, streamflow and PET).
        input_noise: float, default None
            Standard deviation of the zero mean Gaussian noise added to the input tracer data. If None, no noise is added.
        output_noise: float, default None
            Standard deviation of the zero mean Gaussian noise added to the output tracer data. If None, no noise is added.
        get_timeyear: bool
            If True, fractional year of the data is also returned.
        """
        df = pd.read_csv(os.path.join(pathsite, 'data', site+'.txt'), sep='\t')
        if df.shape[1]==1:
            df = pd.read_csv(os.path.join(pathsite, 'data', site+'.txt'))
        Cp = df.loc[:,'Cp']
        if not(input_noise is None):
            Cp += np.random.normal(0,input_noise, len(Cp))
        Cp = torch.tensor(np.nan_to_num(Cp)).float()
        J = df.loc[:, 'p']
        J = torch.tensor(J).float()
        Q = df.loc[:, 'q']
        Q = torch.tensor(Q).float()
        ET = df.loc[:, 'pet']
        ET = torch.tensor(ET).float()
        Cout =  df.loc[:, 'Cq']
        if not(output_noise is None):
            Cout += np.random.normal(0,output_noise, len(Cout))
        if include_concentration:
            if get_timeyear:
                return J, Q, ET, Cp, torch.tensor(Cout).float(), torch.tensor(df.loc[:,'t'].to_numpy()).float()
            else:
                return J, Q, ET, Cp, torch.tensor(Cout).float()
        else:
            if get_timeyear:
                return J, Q, ET, torch.tensor(df.loc[:,'t'].to_numpy()).float()
            else:
                return J, Q, ET