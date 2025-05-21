import os
import multiprocessing
path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

def get_results(x):
    import sys
    sys.path.append(path_root)
    from WATRES import WATRES
    try:
        model = WATRES(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], path_model=x['path_model'])
        print(f"Training {x['site']} with {x['algo']} on process {os.getpid()}")
        pathsite_ground_truth = ''
        res = model.compute_results(BATCH_SIZE=365*5, n_test=365*24, pathsite_ground_truth=os.path.join(path_root, 'data/Pully_small_storage/data/'), site_ground_truth='Pully_small_storage')
        return 1
    except Exception as e:
        print(f"Error training {x['site']} with {x['algo']}: {e}")
        return 0

if __name__ == "__main__":
    os.chdir(os.path.join(path_root, 'WATRES'))
    
    mode2stride = { 
         '2_weeks': 24*7*2,
                '1_week': 24*7,
               'daily': 24,
        #       '12_hours': 12,
               '6_hours': 6,
        #       '3_hours':3,
        #       '2_hours':2,
               'hourly': 1}
    algos = ['WATRES']
    sites = list(mode2stride.keys())
    
    settings_algos = []
    for site in sites:
        for seed in range(12):
            pathsite = f'/mydata/watres/quentin/code/TRANSPORT/data/Pully_small_storage/subsampling/{site}/'
            for algo in algos:
                settings_algos.append({
                    'site': site,
                    'pathsite': pathsite,
                    'algo': algo,
                    'path_model': os.path.join(pathsite, 'save', f'save_{site}_{seed}_no_c_{algo}.pth.tar')
                })

    
    for setting in settings_algos:
        print("Computing results for the site {0} with algo {1}".format(setting['site'], setting['algo']))
        result = get_results(setting)  # Launch each job independently
