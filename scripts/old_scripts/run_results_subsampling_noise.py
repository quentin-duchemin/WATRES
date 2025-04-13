import os
import multiprocessing

def get_results(x): 
    import sys
    sys.path.append('/mydata/watres/quentin/code/TRANSPORT/')
    from BERT4Transit import BERT4Transit
    model_bert = BERT4Transit(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], path_model=x['path_model'])
    stride = x['stride']
    res = model_bert.compute_results(BATCH_SIZE=365*5, n_test=365*24, pathsite_ground_truth='/mydata/watres/quentin/code/TRANSPORT/data/Pully_small_storage/data/', site_ground_truth='Pully_small_storage', save_training_results=False)
    return 1

def testf(x): 
    return 1

if __name__ == "__main__":
    os.chdir('/mydata/watres/quentin/code/TRANSPORT/BERT4Transit/')

    # Define the sites and algorithms
    
    input_std = 0.1
    output_std = 0.1
    for years_training in [10]:
        for seed in range(7,9):
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
                site_name2save=site+'_noise_0.1_yearstrain_'+str(years_training)+'_'+str(seed)
                for algo in algos:
                    settings_algos.append({
                        'site': site,
                        'pathsite': pathsite,
                        'algo': algo,
                        'stride': mode2stride[site],
                        'seed': seed,
                        'input_std':input_std,
                        'output_std':output_std,
                        'years_training':years_training,
                        'path_model': os.path.join(pathsite, 'save', f'save_BERT4TRANSIT_{site_name2save}_no_c_{algo}.pth.tar')

                    })
    
            for sett in settings_algos:
                get_results(sett)