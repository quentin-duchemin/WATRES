import os
import multiprocessing
path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

def get_results(x):
    import sys
    sys.path.append(os.path.join(path_root))
    from WATRES import WATRES
    try:
        model_bert = WATRES(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], path_model=x['path_model'])
        pathsite_ground_truth = ''
        res = model_bert.compute_results(BATCH_SIZE=365*10, n_test=365*24)
        return 1
    except Exception as e:
        print(f"Error training {x['site']} with {x['algo']}: {e}")
        return 0

if __name__ == "__main__":
    os.chdir(os.path.join(path_root, 'WATRES'))
    sites = ['Pully_small_storage'] #['Basel_small_storage','Basel_large_storage','Pully_small_storage','Lugano_small_storage','Lugano_large_storage','Pully_large_storage']
    algos = ['WATRES']
    #algos = ['WATRES', 'AgeDomain', 'Weibull']
    settings_algos = []
    for site in sites:
        pathsite = os.path.join(path_root, f'data/{site}/')
        for algo in algos:
            settings_algos.append({
                'site': site,
                'pathsite': pathsite,
                'algo': algo,
                'path_model': os.path.join(pathsite, 'save', f'save_{site}_seed_0_{algo}.pth.tar')
            })

    
    for setting in settings_algos:
        print("Computing results for the site {0} with algo {1}".format(setting['site'], setting['algo']))
        result = get_results(setting)  # Launch each job independently
