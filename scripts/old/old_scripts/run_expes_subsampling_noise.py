import os
import multiprocessing

def trainf(x): 
    import sys
    sys.path.append('/mydata/watres/quentin/code/TRANSPORT/')
    from BERT4Transit import BERT4Transit
    model_bert = BERT4Transit(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], site_name2save=x['site']+'_noise_0.1_yearstrain_'+str(x['years_training'])+'_'+str(x['seed']))
    stride = x['stride']
    model_bert.train(BATCH_SIZE=4000, n_validation = 365*24*2, subsampling_resolution=x['stride'], n_train=365*24*x['years_training'], seed = x['seed'], nb_epochs=500, std_input_noise=x['input_std'], std_output_noise=x['output_std'])
    return 1

def testf(x): 
    return 1

if __name__ == "__main__":
    os.chdir('/mydata/watres/quentin/code/TRANSPORT/BERT4Transit/')

    # Define the sites and algorithms
    
    input_std = 0.1
    output_std = 0.1
    for years_training in [10,4]:
        for seed in range(0,10):
            mode2stride = { '2_weeks': 24*7*2,
                   '1_week': 24*7,
                  'daily': 24,
                  #'12_hours': 12,
                   '6_hours': 6,
                   # '3_hours':3,
                   # '2_hours':2,
                   'hourly': 1}
            algos = ['SumSquares_noBERT2_bayesian3']
            sites = list(mode2stride.keys())
            
            settings_algos = []
            for site in sites:
                pathsite = f'/mydata/watres/quentin/code/TRANSPORT/data/Pully_small_storage/subsampling/{site}/'
                for algo in algos:
                    settings_algos.append({
                        'site': site,
                        'pathsite': pathsite,
                        'algo': algo,
                        'stride': mode2stride[site],
                        'seed': seed,
                        'input_std':input_std,
                        'output_std':output_std,
                        'years_training':years_training
                    })
    
            for sett in settings_algos:
                trainf(sett)