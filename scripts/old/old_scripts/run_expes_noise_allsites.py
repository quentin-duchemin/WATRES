import os
import multiprocessing

def trainf(x): 
    import sys
    sys.path.append('/mydata/watres/quentin/code/TRANSPORT/')
    from BERT4Transit import BERT4Transit
    model_bert = BERT4Transit(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], site_name2save=x['site_name2save'])
    model_bert.train(BATCH_SIZE=4000, n_validation = 365*24*2, n_train=365*24*10, seed = x['seed'], nb_epochs=250, std_input_noise=x['input_std'], std_output_noise=x['output_std'])
    return 1

def testf(x): 
    return 1

if __name__ == "__main__":
    os.chdir('/mydata/watres/quentin/code/TRANSPORT/BERT4Transit/')

    # Define the sites and algorithms
    
    input_std = 0.1
    output_std = 0.1
    sites = ['Pully_small_storage', 'Pully_large_storage', 'Lugano_small_storage','Lugano_large_storage','Basel_small_storage','Basel_large_storage'] 


    #algos = ['SumSquares_noBERT2']
    algos = ['AgeDomain', 'Weibull']

    
    settings_algos = []
    for site in sites:
        pathsite = '/mydata/watres/quentin/code/TRANSPORT/data/{0}/'.format(site)

        for algo in algos:
            site_name2save = 'input_std_' + str(input_std) + '-output_std_' + str(output_std) + '_'+algo

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
    # Use multiprocessing to parallelize the trainf function
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(trainf, settings_algos)
