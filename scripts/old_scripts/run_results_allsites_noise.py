import os
import multiprocessing

def get_results(x):
    import sys
    sys.path.append('/mydata/watres/quentin/code/TRANSPORT/')
    from BERT4Transit import BERT4Transit
    try:
        model_bert = BERT4Transit(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], path_model=x['path_model'])
        print(f"Training {x['site']} with {x['algo']} on process {os.getpid()}")
        pathsite_ground_truth = ''
        res = model_bert.compute_results(BATCH_SIZE=365*10, n_test=365*24, save_training_results=True)
        return 1
    except Exception as e:
        print(f"Error training {x['site']} with {x['algo']}: {e}")
        return 0

if __name__ == "__main__":
    os.chdir('/mydata/watres/quentin/code/TRANSPORT/BERT4Transit/')
    sites = ['Basel_small_storage','Basel_large_storage','Pully_small_storage','Lugano_small_storage','Lugano_large_storage','Pully_large_storage']
    #sites = ['Pully_small_storage','Pully_large_storage']#
    algos = ['SumSquares_noBERT2_bayesian3']
    algos = ['AgeDomain', 'Weibull']
    settings_algos = []
    input_std = 0.1
    output_std = 0.1

    for site in sites:
        pathsite = f'/mydata/watres/quentin/code/TRANSPORT/data/{site}/'
        for algo in algos:
            site_name2save = 'input_std_' + str(input_std) + '-output_std_' + str(output_std) + '_'+algo

            settings_algos.append({
                'site': site,
                'pathsite': pathsite,
                'algo': algo,
                'seed': 0,
                'path_model': os.path.join(pathsite, 'save', 'save_BERT4TRANSIT_{0}_'.format(site_name2save)+'no_c_'+algo+'.pth.tar')
            })

    for setting in settings_algos:
        print("Computing results for the site {0} with algo {1}".format(setting['site'], setting['algo']))
        result = get_results(setting)  # Launch each job independently