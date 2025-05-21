import os
import multiprocessing

path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

def trainf(x): 
    import sys
    sys.path.append(os.path.join(path_root))
    from WATRES import WATRES
    model_bert = WATRES(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], site_name2save=x['site_name2save'])
    model_bert.train(BATCH_SIZE=100, n_validation = 365*24*2, n_train=365*24*2, seed = x['seed'], nb_epochs=250, std_input_noise=x['input_std'], std_output_noise=x['output_std'])
    return 1

def testf(x): 
    return 1

if __name__ == "__main__":
    os.chdir(os.path.join(path_root, 'WATRES'))

    # Define the sites and algorithms
    
    input_std = 0.
    output_std = 0.
    sites = ['Pully_small_storage']#, 'Pully_large_storage', 'Lugano_small_storage','Lugano_large_storage','Basel_small_storage','Basel_large_storage'] 


    algos = ['WATRES']

    
    settings_algos = []
    for site in sites:
        pathsite = os.path.join(path_root, f'data/{site}/')
        
        for algo in algos:
            site_name2save = 'input_std_' + str(input_std) + '-output_std_' + str(output_std)

            settings_algos.append({
                'site': site,
                'site_name2save':site_name2save,
                'pathsite': pathsite,
                'algo': algo,
                'seed': 0,
                'input_std':input_std,
                'output_std':output_std
            })

    for sett in settings_algos:
        trainf(sett)