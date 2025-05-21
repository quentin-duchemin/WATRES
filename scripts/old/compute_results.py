import os
import multiprocessing
path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

        
import os
import multiprocessing

def get_results(x):
    import sys
    sys.path.append(os.path.join(path_root))
    from WATRES import WATRES
    try:
        model = WATRES(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], path_model=x['path_model'])
        print(f"Training {x['site']} with {x['algo']} on process {os.getpid()}")
        pathsite_ground_truth = ''
        res = model.compute_results(BATCH_SIZE=365, n_test=365*24, save_training_results=True)
        return 1
    except Exception as e:
        print(f"Error training {x['site']} with {x['algo']}: {e}")
        return 0

if __name__ == "__main__":
    os.chdir(os.path.join(path_root, 'WATRES'))
    sites = ['Pully_small_storage'] #'Basel_small_storage','Basel_large_storage','Lugano_small_storage','Lugano_large_storage','Pully_large_storage']
    algos = ['WATRES']
    settings_algos = []
    input_std = 0.
    output_std = 0.

    for site in sites:
        pathsite = os.path.join(path_root, f'data/{site}/')
        for algo in algos:
            site_name2save = 'input_std_' + str(input_std) + '-output_std_' + str(output_std) 

            settings_algos.append({
                'site': site,
                'pathsite': pathsite,
                'algo': algo,
                'seed': 0,
                'path_model': os.path.join(pathsite, 'save', 'save_{0}_{1}.pth.tar'.format(site_name2save, algo))
            })

    for setting in settings_algos:
        print("Computing results for the site {0} with algo {1}".format(setting['site'], setting['algo']))
        result = get_results(setting)  # Launch each job independently