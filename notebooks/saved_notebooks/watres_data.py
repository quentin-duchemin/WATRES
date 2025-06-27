import os
notebook_path = os.getcwd()
import numpy as np
import sys
sys.path.append('../')
from WATRES import *
import pandas as pd
import itertools
savefolder = '/mydata/watres/quentin/PinballRT/final_watres/WATRES/data/watres_data/'


locations = ['Pully', 'Lugano', 'Basel']
sizes_storage = ['small_storage', 'large_storage']
sites = list(map(lambda x: '{0}_{1}'.format(x[0],x[1]), list(itertools.product(locations, sizes_storage))))

for site in sites:
    print('Processing site:', site)
    pathsite_ground_truth = f'/mydata/watres/quentin/PinballRT/final_watres/WATRES/data/{site}/data/'

    df = pd.read_csv(os.path.join(pathsite_ground_truth,f'{site}.txt'), sep='\t')
    n = df.shape[0]
    
    npy_file = os.path.join(pathsite_ground_truth, 'TTD.npy')
    ttds = np.load(npy_file, allow_pickle=True)
    dates_timeyear = np.array(df['t'][-(ttds.shape[0]):])
    dates = [fractional_year_to_datetime(el).strftime('%H:00-%m/%d/%Y') for el in dates_timeyear]
    columns = [f'age={i} hours' for i in range(ttds.shape[1])]
    ttds_df = pd.DataFrame(data=ttds[:,:24*30*5], index=dates, columns=columns[:24*30*5])
    # Save as Parquet (requires pyarrow or fastparquet)
    ttds_df.to_parquet(
        os.path.join(savefolder, f'{site}_true_ttds.parquet'),
        engine='pyarrow'
    )
        
        
    TTDs = []
    m = ttds.shape[0]
    step = 4000
    k=0
    algo = 'WATRES'
    while (k*step<m):
        low = k*step
        up = min([(k+1)*step,len(dates_timeyear)-1])
        def filter_dates(dates):
            if up==(len(dates_timeyear)-1):
                return np.where( (dates>=dates_timeyear[low]) & ((dates<=dates_timeyear[up])))[0]
            else:
                return np.where( (dates>=dates_timeyear[low]) & ((dates<dates_timeyear[up])))[0]
    
        x = {
            'pathsite': f"/mydata/watres/quentin/PinballRT/final_watres/WATRES//data/{site}/",
            'path_model': f"/mydata/watres/quentin/PinballRT/final_watres/WATRES/data/{site}/save/save_input_std_0.1-output_std_0.1_{algo}.pth.tar",
            'site': site,
            'algo': algo
        }
        model = WATRES(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], path_model=x['path_model'])
        results = model.model_estimate(filter_dates, BATCH_SIZE=None)
        TTDs.append(results['ttd'])
        k += 1
    allttds = np.concatenate(TTDs, axis=0)
    columns = [f'age={i} hours' for i in range(allttds.shape[1])]
    print(allttds.shape, ttds.shape)
    ttds_df = pd.DataFrame(data=allttds[:,:24*30*5], index=dates, columns=columns[:24*30*5])
    #ttds_df.to_csv(os.path.join(savefolder, f'{site}_WATRES_ttds.csv'), sep='\t')    
    # Save as Parquet (requires pyarrow or fastparquet)
    ttds_df.to_parquet(
        os.path.join(savefolder, f'{site}_WATRES_ttds.parquet'),
        engine='pyarrow'
    )





import os
import os
notebook_path = os.getcwd()
import numpy as np
import sys
sys.path.append('/mydata/watres/quentin/code/TRANSPORT/')
from BERT4Transit import *
import pandas as pd
import datetime
import itertools
savefolder = '/mydata/watres/quentin/PinballRT/final_watres/WATRES/data/watres_data/'



def fractional_year(dt: datetime) -> float:
    year = dt.year
    start_of_year = datetime.datetime(year, 1, 1)
    start_of_next_year = datetime.datetime(year + 1, 1, 1)
    year_length = (start_of_next_year - start_of_year).total_seconds()
    elapsed = (dt - start_of_year).total_seconds()
    return year + elapsed / year_length

x = {
    'pathsite': "/mydata/watres/quentin/code/TRANSPORT/data/Pully_small_storage/subsampling/2_weeks/",
    'path_model': "/mydata/watres/quentin/code/TRANSPORT/data/Pully_small_storage/subsampling/2_weeks/save/save_BERT4TRANSIT_2_weeks_noise_0.1_yearstrain_10_0_no_c_SumSquares_noBERT2_bayesian3.pth.tar",
    'site': "2_weeks",
    'algo': "SumSquares_noBERT2_bayesian3"
}
model = BERT4Transit(pathsite=x['pathsite'], site=x['site'], algo=x['algo'], path_model=x['path_model'])

dates_timeyear = np.array( pd.read_csv(os.path.join(savefolder,'Pully_small_storage_WATRES_ttds.csv'), usecols=[0], sep='\t')).flatten()


m=26281
step = 4000
k=0
TTDs = []
while (k*step<m):
    print('Step:', k)
    low = k*step
    up = min([(k+1)*step,len(dates_timeyear)-1])
    def filter_dates(dates):
        if up==(len(dates_timeyear)-1):
            return np.where( (dates>=dates_timeyear[low]) & ((dates<=dates_timeyear[up])))[0]
        else:
            return np.where( (dates>=dates_timeyear[low]) & ((dates<dates_timeyear[up])))[0]


    def convert2datetime(date):
        return fractional_year(datetime.datetime.strptime(date, '%H:00-%m/%d/%Y'))

    def filter_dates(dates):
        if up==(len(dates_timeyear)-1):
            return np.where( (dates>=convert2datetime(dates_timeyear[low])) & (dates<=convert2datetime(dates_timeyear[up])))[0]
        else:
            return np.where( (dates>=convert2datetime(dates_timeyear[low])) & (dates<convert2datetime(dates_timeyear[up])))[0]

    results = model.model_estimate(filter_dates, BATCH_SIZE=None)
    TTDs.append(results['ttd'])
    k += 1
# Showing prediction on output tracer data

allttds = np.concatenate(TTDs, axis=0)
columns = [f'age={i} hours' for i in range(allttds.shape[1])]
ttds_df = pd.DataFrame(data=allttds[:,:24*30*5], index=dates_timeyear, columns=columns[:24*30*5])
ttds_df.to_parquet(
        os.path.join(savefolder, 'Pully_small_storage_fortnightly_WATRES_ttds.parquet'),
        engine='pyarrow'
    )