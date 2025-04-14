from .models import lightning_interface
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import pyreadr
import torch.nn.functional as F
import pickle
import os
import pandas as pd

def global_average(PQ_hat, ttd):
    x = PQ_hat[:6900,:]
    PQx = np.mean(x, axis=1)
    x = np.cumsum(ttd, axis=1)
    x = np.mean(x, axis=0)
    x = x[:6900]
    return np.linalg.norm(PQx-x, ord=1)/6900
    
def average_by_Qquantiles(PQ_hat, lst, Q, ttd):
    length_time = len(Q)
    Qsorted = np.sort(np.array(Q)[lst])
    quantiles = []
    
    errors = np.zeros(4)
    for l in [0.25*i for i in range(1,5)]:
         quantiles.append(Qsorted[int(l*len(Qsorted)-1)])    

    clustersQ = {k:[] for k in range(len(quantiles))}
    for i in range(len(lst)):
        k = 0
        while quantiles[k]+1e-4<=Q[int(lst[i])]:
            k += 1
        clustersQ[k].append(i)
    for k in range(len(quantiles)):
        x = PQ_hat[:6900,:]
        PQx = np.mean(x[:,clustersQ[k]], axis=1)
        
        x = np.cumsum(ttd[clustersQ[k],:], axis=1)
        x = np.mean(x, axis=0)
        x = x[:6900]
        errors[k] = np.linalg.norm(PQx-x, ord=1)/6900
    return errors
        

class Results():
    """
    Result class of the WATRES package to learn transit time distributions of a given watershed.
    """
    def __init__(self):
        pass
        
    def model_estimate(self, filter_dates=None, BATCH_SIZE=None):
        """
        Forward method of the model. Return the estimated cumulative TTDs and the predicted output tracer time series.
        
        Parameters
        ----------
        filter_dates: function
            Function taking as input an array of fractional year (such as [2020.43, 2020.47, ...] and returns the set of indexes that should to keep. A basic example to keep only time points after 2020 is:
            def filter_dates(dates):
                return np.where(dates>=2020)[0]
        BATCH_SIZE: int
            Among the time points filtered by `filter_dates`, BATCH_SIZE specifies the size of the set used to evaluate the model. If BATCH_SIZE = None, all the prefiltered time points will be considered.
        """
        result = {}
        Tmax = self.Tmax
        algo = self.algo
        pathsite = self.pathsite
        site = self.site

        J, Q, ET, CJ, Cout, timeyear = self.get_data(pathsite, site, include_concentration=True, get_timeyear=True)

        
        if filter_dates is None:
            lst = timeyear
        else:
            lst = filter_dates(timeyear)
        if BATCH_SIZE is None:
            BATCH_SIZE = len(lst)
        BATCH_SIZE = np.min([BATCH_SIZE, len(lst)])
        indices = [round(i * (len(lst) - 1) / (BATCH_SIZE - 1)) for i in range(BATCH_SIZE)]

        lst = lst[indices]

        result['time_index'] = lst
        data_train, timeyear_train = self.get_features(pathsite, site, lst)
        input_size = data_train.shape[1]

        result['timeyear'] = timeyear_train
        if algo=='WATRES':
                model = lightning_interface.LightningWatres(input_size, Tmax=Tmax)
        elif algo=='AgeDomain':
                model = lightning_interface.LightningAgeDomain(input_size, Tmax=Tmax)
        elif algo=='Weibull':
                model = lightning_interface.LightningWeibull(input_size, Tmax=Tmax)
        model.load_state_dict(torch.load(self.path_model, weights_only=False)['state_dict'])
        model.eval()
    
        with torch.no_grad():            
            Cout_train = torch.zeros(len(lst))
            CJ_train = torch.zeros((len(lst), Tmax))
            J_train = torch.zeros((len(lst), Tmax))
            Q_train = torch.zeros(len(lst))
            ET_train = torch.zeros(len(lst))
            Qinv_train = torch.zeros((len(lst),Tmax))
            ETinv_train = torch.zeros((len(lst),Tmax))
    
            for i,t in enumerate(lst):
                Cout_train[i] = Cout[t]
                CJ_train[i,:] = CJ[t-Tmax:t]
                J_train[i,:] = J[t-Tmax:t]
                Q_train[i]  = torch.sum(Q[t-Tmax:t])
                ET_train[i] = torch.sum(ET[t-Tmax:t])
                ETinv_train[i,:] = torch.flip(ET[t-Tmax:t], [0])
                Qinv_train[i,:] = torch.flip(Q[t-Tmax:t], [0])
    
            if algo=='AgeDomain':
                Chat, ywfhat, pQ = model.model.forward(data_train, J_train, CJ_train, returnpQ=True)
            elif algo in ['WATRES', 'Weibull']:
                Chat, SThat, ywfhat, pQ = model.model.forward(data_train, J_train, CJ_train, returnpQ=True)

    
            Chat, ywfhat, pQ = Chat.detach().numpy(), ywfhat.detach().numpy(), pQ.detach().numpy()
    
            result['Cout'] = Cout_train
            result['Chat'] = Chat
            result['ywfhat'] = ywfhat        
    
            PQ_hat = np.cumsum(pQ, axis=1).T
            result['PQhat'] = PQ_hat
            result['timeyear'] = timeyear_train
            result['ERROR_Cout'] = np.linalg.norm(Chat-Cout_train.numpy())/len(lst)
        return result
        
    def compute_results(self, BATCH_SIZE = None, n_test=360*24*3, pathsite_ground_truth=None, site_ground_truth=None, save_training_results=False):
        """
        Method precomputing and saving different statistics of interest to make faster and easier visualizations.
        """
        result = {}    

        if pathsite_ground_truth is None:
            pathsite_ground_truth = os.path.join(self.pathsite, 'data')
            site_ground_truth = self.site
        Tmax = self.Tmax
        algo = self.algo
        pathsite = self.pathsite
        site = self.site
         

        if BATCH_SIZE is None:
            BATCH_SIZE = n_test-5

        J, Q, ET, CJ, Cout = self.get_data(pathsite, site, include_concentration=True)

        ################################################################### BEGIN TRAINING
        if save_training_results:
            lst_train = self.lst_train
            BATCH_SIZE_train = len(lst_train)
            result['lst_train'] = lst_train
    
            data_train, timeyear_train = self.get_features(pathsite, site, lst_train)
            input_size = data_train.shape[1]

            result['timeyear_train'] = timeyear_train
            
            if algo=='WATRES':
                model = lightning_interface.LightningWatres(input_size, Tmax=Tmax)
            elif algo=='AgeDomain':
                model = lightning_interface.LightningAgeDomain(input_size, Tmax=Tmax)
            elif algo=='Weibull':
                model = lightning_interface.LightningWeibull(input_size, Tmax=Tmax)
            model.load_state_dict(torch.load(self.path_model, weights_only=False)['state_dict'])
            model.eval()
        
            with torch.no_grad():            
                Cout_train = torch.zeros(len(lst_train))
                CJ_train = torch.zeros((len(lst_train), Tmax))
                J_train = torch.zeros((len(lst_train), Tmax))
                Q_train = torch.zeros(len(lst_train))
                ET_train = torch.zeros(len(lst_train))
                Qinv_train = torch.zeros((len(lst_train),Tmax))
                ETinv_train = torch.zeros((len(lst_train),Tmax))
        
                for i,t in enumerate(lst_train):
                    Cout_train[i] = Cout[t]
                    CJ_train[i,:] = CJ[t-Tmax:t]
                    J_train[i,:] = J[t-Tmax:t]
                    Q_train[i]  = torch.sum(Q[t-Tmax:t])
                    ET_train[i] = torch.sum(ET[t-Tmax:t])
                    ETinv_train[i,:] = torch.flip(ET[t-Tmax:t], [0])
                    Qinv_train[i,:] = torch.flip(Q[t-Tmax:t], [0])
        
                if algo=='AgeDomain':
                    Chat, ywfhat, pQ = model.model.forward(data_train, J_train, CJ_train, returnpQ=True)
                elif algo in ['WATRES', 'Weibull']:
                    Chat, SThat, ywfhat, pQ = model.model.forward(data_train, J_train, CJ_train, returnpQ=True)

        
        
                Chat, ywfhat, pQ = Chat.detach().numpy(), ywfhat.detach().numpy(), pQ.detach().numpy()
        
                result['Cout_train'] = Cout_train
                result['Chat_train'] = Chat
                result['ywfhat_train'] = ywfhat        
        
                PQ_hat = np.cumsum(pQ, axis=1).T
                result['PQhat_train'] = PQ_hat
    
                result['ERROR_Cout_train'] = np.linalg.norm(Chat-Cout_train.numpy())/len(lst_train)
        ################################################################### END TRAINING

        lst_test, BATCH_SIZE = self.get_time_points(pathsite, site, BATCH_SIZE, n_start=-n_test, n_end=-1)
        result['lst_test'] = lst_test

        data_test, timeyear_test = self.get_features(pathsite, site, lst_test)
        input_size = data_test.shape[1]

        result['timeyear_test'] = timeyear_test
        


        try:
            print('Start loading true TTDs')

            if False:
                ls_ywf_true = []
                ls_ttds_true = []
                for i in range(100):  
                    ttd_temp = np.load(os.path.join(self.pathsite, 'data', 'TTD', f'{site}_TTD_{i}.npy'),  allow_pickle=True)
                    meanTTD += np.sum(np.cumsum(ttd_temp, axis=1), axis=0)
                    count += ttd_temp.shape[0]
                    ls_ywf_true.append(np.cumsum(ttd_true, axis=1)[:,[24*30*j for j in range(1,10)]])
                    
                    ywf_true += list(np.cumsum(ttd_temp, axis=1)[:,24*30*3])
                    ls_ttds_true.append(ttd_temp[:,:7000])
                    del ttd_temp
                ywf_true = np.concatenate(ls_ywf_true, axis=0)
                ttds_true = np.concatenate(ls_ttds_true, axis=0)
                # keeping relevant ground truth values
                idxs2keep_true_ttd = lst_test - (len(CJ)-ttd.shape[0])
                ywf_true = ywf_true[idxs2keep_true_ttd]
                ttds_true = ttds_true[idxs2keep_true_ttd]
            else:
                rows_to_load = lst_test - (len(CJ)-365*24*2)
                npy_file = os.path.join(pathsite_ground_truth, 'TTD.npy')
                ttds = np.load(npy_file)
                # keeping relevant ground truth values
                idxs2keep_true_ttd = lst_test - (len(CJ)-ttds.shape[0])
                ttds_true = ttds[idxs2keep_true_ttd,:]
                ywf_true = np.cumsum(ttds_true, axis=1)[:,[24*30*j for j in range(1,10)]]
            true_ttd_loaded = True
            print('True TTDs loaded')
        except:
            print('WARNING: True TTDs not loaded')
            true_ttd_loaded = False
        

        try:
            rSTO = pyreadr.read_r(os.path.join(pathsite_ground_truth, site_ground_truth+'_rank_storage.rda'))
            rSTO = rSTO["rank_storage"]
            rSTO = rSTO.to_numpy()
    
            idxs2keep_true_rSTO = lst_test - (len(CJ)-rSTO.shape[0])
            rSTO_true = rSTO[idxs2keep_true_rSTO,:]
            rSTO_loaded = True
        except:
            rSTO_loaded = False
            pass
        
        
        if algo=='AgeDomain':
            model = lightning_interface.LightningAgeDomain(input_size, Tmax=Tmax)
        elif algo=='Weibull':
            model = lightning_interface.LightningWeibull(input_size, Tmax=Tmax)
        elif algo=='WATRES':
            model = lightning_interface.LightningWatres(input_size, Tmax=Tmax)
            
        model.load_state_dict(torch.load(self.path_model, weights_only=False)['state_dict'])
        model.eval()
    
        with torch.no_grad():            
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
    
            if algo=='AgeDomain':
                Chat, ywfhat, pQ = model.model.forward(data_test, J_test, CJ_test, returnpQ=True)
            elif algo in ['WATRES', 'Weibull']:
                Chat, SThat, ywfhat, pQ = model.model.forward(data_test, J_test, CJ_test, returnpQ=True)
                w = model.model.forward_w(data_test)
                w = w.detach().numpy()
                frac_year = timeyear_test
                frac_year = [el-int(el) for el in frac_year]
                frac_year = np.array([min([1-el, el]) for el in frac_year])
                winter_idxs = np.where(frac_year<=(2*30/365))[0]
                summer_idxs = np.where(np.abs(0.5-frac_year)<=(2*30/365))[0]
                result['w_winter'] = np.mean(w[winter_idxs.astype(int),:], axis=0)
                result['w_summer'] = np.mean(w[summer_idxs.astype(int),:], axis=0)


            ####### 
            Q_test = np.array([Q[t] for t in lst_test])            
            result['Q_test'] = Q_test
    
            Chat, ywfhat, pQ = Chat.detach().numpy(), ywfhat.detach().numpy(), pQ.detach().numpy()
    
            result['Cout'] = Cout_test
            result['Chat'] = Chat
            result['ywfhat'] = ywfhat        
    
            PQ_hat = np.cumsum(pQ, axis=1).T
            result['global_PQhat'] = np.mean(PQ_hat.T, axis=0)
            
            data_Q = pd.DataFrame({'q': Q})
            q_quantiles = data_Q['q'].quantile([0.25, 0.5, 0.75]).to_dict()
            # Add columns for each quartile with 0 or 1 depending on which quartile the 'q' value belongs to
            data_Q['Q_quantile_0'] = (data_Q['q'] <= q_quantiles[0.25]).astype(int)
            data_Q['Q_quantile_1'] = ((data_Q['q'] > q_quantiles[0.25]) & (data_Q['q'] <= q_quantiles[0.5])).astype(int)
            data_Q['Q_quantile_2'] = ((data_Q['q'] > q_quantiles[0.5]) & (data_Q['q'] <= q_quantiles[0.75])).astype(int)
            data_Q['Q_quantile_3'] = (data_Q['q'] > q_quantiles[0.75]).astype(int)

            for k in range(4):
                idxs_q_quantiles = np.where(data_Q['Q_quantile_{0}'.format(k)])[0]
                idxs_q_quantiles = np.intersect1d(idxs_q_quantiles, lst_test)
                idxs = np.array([i for i, t in enumerate(lst_test) if t in idxs_q_quantiles])
                result['quantile{0}_PQhat'.format(k)] = np.mean(PQ_hat.T[idxs.astype(int),:], axis=0)
                

            if true_ttd_loaded:
                #result['PQtrue'] = np.cumsum(ttds_true, axis=1)
                result['global_PQtrue'] = np.mean(np.cumsum(ttds_true, axis=1), axis=0)
                for k in range(4):
                    idxs_q_quantiles = np.where(data_Q['Q_quantile_{0}'.format(k)])[0]
                    idxs_q_quantiles = np.intersect1d(idxs_q_quantiles, lst_test)
                    idxs = np.array([i for i, t in enumerate(lst_test) if t in idxs_q_quantiles])
                    result['quantile{0}_PQtrue'.format(k)] = np.mean(np.cumsum(ttds_true, axis=1)[idxs.astype(int),:], axis=0)
                result['ywf_true'] = ywf_true
                result['ERROR_ywf'] = np.linalg.norm(ywfhat-ywf_true, axis=0, ord=1)/len(lst_test)
                result['ERROR_global_ywf'] = np.abs(np.mean(ywfhat, axis=0)-np.mean(ywf_true, axis=0))
                result['ERROR_global_PQ'] = global_average(PQ_hat, np.cumsum(ttds_true, axis=1))
                result['ERROR_quantilesQ_PQ'] = average_by_Qquantiles(PQ_hat, lst_test, Q, ttds_true)
            
            result['ERROR_Cout'] = np.linalg.norm(Chat-Cout_test.numpy())/len(lst_test)
    
    
    
            if (algo!='AgeDomain') and (rSTO_loaded):
                SThat = SThat.detach().numpy()
                result['SThat'] = SThat
                result['ST_true'] = rSTO_true
                max_age = min([SThat.shape[1], rSTO_true.shape[1]])
                result['ERROR_ST'] = np.linalg.norm(SThat[:,:max_age]-rSTO_true[:,:max_age], axis=0, ord=1)/len(lst_test)
    


            start_index = (self.path_model).find("save_")
            name_model = self.path_model[start_index+5:-8] # -8 to remove '.pth.tar' 5 to remove save_

            
            f = open(os.path.join(pathsite, 'save', "results_{0}.pkl".format(name_model)),"wb")
            pickle.dump(result,f)
            f.close()
        return result
