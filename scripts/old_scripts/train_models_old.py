import os
import multiprocessing

path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

def trainf(x): 
    import sys
    sys.path.append(os.path.join(path_root))
    from WATRES import WATRES
    model_bert = WATRES(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], site_name2save=x['site']+'_seed_'+str(x['seed']))
    stride = x['stride']
    model_bert.train(BATCH_SIZE=1000, n_validation = 36, seed = x['seed'], subsampling_resolution=x['stride'])
    return 1

def testf(x): 
    return 1

if __name__ == "__main__":
    
    os.chdir(os.path.join(path_root, 'WATRES'))

    # Define the sites and algorithms
    
    if True:
        sites = ['Pully_small_storage'] #['Lugano_flashy', 'Lugano_notflashy', 'Basel_flashy', 'Basel_notflashy']
        algos = ['WATRES']  # List of algorithms
        
        # Create settings based on site and algo combinations
        settings_algos = []
        for site in sites:
            pathsite = os.path.join(path_root, f'data/{site}/')
            for algo in algos:
                settings_algos.append({
                    'site': site,
                    'pathsite': pathsite,
                    'algo': algo,
                     'seed': 0,
                    'stride':1
                })
    else:
        for seed in range(0,30):
            mode2stride = { '2_weeks': 24*7*2,
                   '1_week': 24*7,
                  'daily': 24,
                  #'12_hours': 12,
                   '6_hours': 6,
                   # '3_hours':3,
                   # '2_hours':2,
                   'hourly': 1}
            algos = ['WATRES']
            sites = list(mode2stride.keys())
            
            settings_algos = []
            for site in sites:
                pathsite = os.path.join(path_root, f'data/Pully_small_storage/subsampling/{site}/')
                for algo in algos:
                    settings_algos.append({
                        'site': site,
                        'pathsite': pathsite,
                        'algo': algo,
                        'stride': mode2stride[site],
                        'seed': seed
                    })
    
    for sett in settings_algos:
        trainf(sett)