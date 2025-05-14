######################################################################

# This script takes the sampled listening events from the MLHD dataset and extracts the unique artists from them.

######################################################################

import os
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path


env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")


sample_directory = dataset_dir + '/processed/MLHD_sampled'


artists_path = os.path.join(sample_directory, 'artists.tsv')

batch_size = 1000000
artists = set()

le_paths = [os.path.join(sample_directory, f'listening_events-{i}.tsv.bz2') for i in list(range(0, 10)) + list(map(chr, range(ord('a'), ord('f')+1)))]


for path in le_paths:
    # load listening events in batches
    for i, chunk in enumerate(pd.read_csv(path, sep="\t", chunksize=batch_size, header=None)):
        if i == 0:
            print(chunk.head(), path)
        if i % 10:
            print(f'Processing chunk {i}...')

        chunk.columns = ['user_id', 'timestamp', 'artist_id', 'item_id']
        artist_chunk = chunk[['artist_id']].drop_duplicates()
        
        artists_unique = [artist for artist in artist_chunk['artist_id'] if ',' not in artist]
        multiple_artists = artist_chunk[artist_chunk['artist_id'].str.contains(',')]
        multiple_artists = multiple_artists['artist_id'].str.split(',').explode().drop_duplicates()
        
        artists_unique.extend(multiple_artists)
        artists_unique = set(artists_unique)

            
        # load artists
        artists.update(artists_unique)
        
# Save artists
artists_df = pd.DataFrame(artists, columns=['artist_id'])

artists_df.to_csv(artists_path, sep="\t", index=False, header=False)