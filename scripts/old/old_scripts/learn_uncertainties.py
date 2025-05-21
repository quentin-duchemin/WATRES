import os
import multiprocessing

def get_results(x):
    import sys
    sys.path.append('/mydata/watres/quentin/code/TRANSPORT/')
    from BERT4Transit import BERT4Transit
    model_bert = BERT4Transit(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], path_model=x['path_model'])
    print(f"Learning uncertainties {x['site']} with {x['algo']} on process {os.getpid()}")
    pathsite_ground_truth = ''
    res = model_bert.learn_uncertainties()
    return 1
    # except Exception as e:
    #     print(f"Error training {x['site']} with {x['algo']}: {e}")
    #     return 0

if __name__ == "__main__":
    os.chdir('/mydata/watres/quentin/code/TRANSPORT/BERT4Transit/')
    sites = ['Basel_small_storage','Basel_large_storage','Pully_small_storage','Lugano_small_storage','Lugano_large_storage','Pully_large_storage']
    algos = ['SumSquares_noBERT2_bayesian']
    #algos = ['SumSquares_noBERT2', 'AgeDomain', 'Weibull']
    settings_algos = []
    for site in sites:
        pathsite = f'/mydata/watres/quentin/code/TRANSPORT/data/{site}/'
        for algo in algos:
            settings_algos.append({
                'site': site,
                'pathsite': pathsite,
                'algo': algo,
                'path_model': os.path.join(pathsite, 'save', f'save_BERT4TRANSIT_{site}_no_c_{algo}.pth.tar')
            })

    
    for setting in settings_algos:
        print("Computing uncertainties for the site {0} with algo {1}".format(setting['site'], setting['algo']))
        result = get_results(setting)  # Launch each job independently
