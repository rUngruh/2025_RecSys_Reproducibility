import os
import ast
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path
env_path = Path('../..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")

import argparse
parser = argparse.ArgumentParser(description='Filter events by year and save. (Only Applicable for MLHD)')
parser.add_argument('--chunksize', type=int, default=1000000, help='Chunk size for processing listening events')
parser.add_argument('--start_date', type=str, help='Start date for the split', default='2012-10-31')
parser.add_argument('--end_date', type=str, help='End date for the split', default='2013-10-31')

args = parser.parse_args()

chunksize = args.chunksize

start_time = pd.to_datetime(args.start_date)
end_time = pd.to_datetime(args.end_date)

year = end_time.year

# mlhd_data_dir = dataset_dir + '/processed/mlhd_sampled_filtered'
mlhd_data_dir = '../../data'

save_dir = dataset_dir + f'/processed/mlhd_rec_{year}'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

interactions_path = mlhd_data_dir + '/interactions.tsv.bz2'

compressed = True if interactions_path.endswith('.bz2') else False

save_path = os.path.join(save_dir, 'interactions.tsv.bz2')

if os.path.exists(save_path):
    os.remove(save_path)
    
print(f"Loading interactions from {interactions_path}...")
num_unique_les = 0
for i, chunk in enumerate(pd.read_csv(interactions_path, sep='\t', chunksize=chunksize, compression='bz2' if compressed else None, header=None)):
    chunk.columns = ['user_id', 'timestamp', 'item_id', 'age_at_listen']
    if i % 1 == 0:  # Print status every 10 chunks
        print(f'Processed {i * chunksize:,} rows; current chunk size: {len(chunk):,}') #out of x billion ({i * chunksize / x:.2%})')

    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
    chunk = chunk[(chunk['timestamp'] >= start_time) & (chunk['timestamp'] < end_time)]
    
    chunk = chunk.sort_values(by=['user_id', 'item_id', 'timestamp'])
    
    # Add a 'count' column to track the number of listening events per user-track combination
    chunk['count'] = chunk.groupby(['user_id', 'item_id'])['timestamp'].transform('count')
    
    # Remove duplicate listening events, keeping only the first one per user-track combination
    chunk = chunk.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    chunk.reset_index(drop=True, inplace=True)
    
    num_unique_les += len(chunk)

    # Save the processed chunk to a new TSV file
    chunk.to_csv(save_path, sep='\t', index=False, header=False, mode='a')

del chunk


print('Now loading all data to check for duplicates...')
# Check whether duplicates remain
interactions = pd.read_csv(save_path, sep='\t', compression='bz2')
interactions.columns = ['user_id', 'timestamp', 'item_id', 'age_at_listen', 'count']


interactions = interactions.groupby(['user_id', 'item_id'], as_index=False).agg({
    'timestamp': 'min', 
    'age_at_listen': 'first',  # shouldn't matter either as timeframe should be within the same age bracket
    'count': 'sum'
})
interactions.reset_index(drop=True, inplace=True)

interactions = interactions[['user_id', 'timestamp', 'item_id', 'age_at_listen', 'count']]

interactions.to_csv(save_path, sep='\t', index=False, header=False)
