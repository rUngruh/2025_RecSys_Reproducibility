import os
import pandas as pd
import numpy as np

import json

from dotenv import load_dotenv
from pathlib import Path
env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")

import argparse
argparser = argparse.ArgumentParser(description="Process MLHD listening events.")
argparser.add_argument('--dataset_dir', type=str, help="Path to the dataset directory. Only needed if not using the .env file.", default=dataset_dir)
args = argparser.parse_args()
dataset_dir = args.dataset_dir

sample_filtered_directory = dataset_dir + '/processed/MLHD_sampled_filtered'

listening_events_paths = [os.path.join(sample_filtered_directory, f'interactions_verbose-{i}.tsv.bz2') for i in 
                          list(range(0, 10)) + list(map(chr, range(ord('a'), ord('f')+1)))]

artists_path = os.path.join(sample_filtered_directory, 'artists_verbose.tsv')
users_path = os.path.join(sample_filtered_directory, 'users_verbose.tsv')

listening_events_save_paths = [os.path.join(sample_filtered_directory, f'interactions-{i}.tsv.bz2') for i in 
                                list(range(0, 10)) + list(map(chr, range(ord('a'), ord('f')+1)))]

all_listening_events_save_path = os.path.join(sample_filtered_directory, 'interactions.tsv.bz2')
all_listening_events_save_path = '../data/interactions.tsv.bz2'

artists_save_path = os.path.join(sample_filtered_directory, 'artists.tsv')
users_save_path = os.path.join(sample_filtered_directory, 'users.tsv')
tracks_save_path = os.path.join(sample_filtered_directory, 'tracks.tsv')

id_mbid_dict_artist_path = os.path.join(sample_filtered_directory, 'id_mbid_artist.json')
id_mbid_dict_user_path = os.path.join(sample_filtered_directory, 'id_mbid_user.json')
id_mbid_dict_item_path = os.path.join(sample_filtered_directory, 'id_mbid_item.json')

compressed = True if listening_events_save_paths[0].endswith('.bz2') else False
batch_size = 10000000

item_MBID_ID_dict = {}

print("Processing artists...")
# Process artists
artists = pd.read_csv(artists_path, sep="\t")
artist_mbids = artists['artist_id'].unique()
artist_MBID_ID_dict = {mbid : i for i, mbid in enumerate(artist_mbids)}
artists['artist_id'] = artists['artist_id'].map(artist_MBID_ID_dict)

artists.to_csv(artists_save_path, sep="\t", index=False, header=True)

del artists
with open(id_mbid_dict_artist_path, 'w') as f:
    json.dump({v:k for k, v in artist_MBID_ID_dict.items()}, f)

print("Processing users...")
# Process users
users = pd.read_csv(users_path, sep="\t")
user_mbids = users['user_id'].unique()
user_MBID_ID_dict = {mbid : i for i, mbid in enumerate(user_mbids)}
users['user_id'] = users['user_id'].map(user_MBID_ID_dict)                  

users.to_csv(users_save_path, sep="\t", index=False, header=True)

del users

with open(id_mbid_dict_user_path, 'w') as f:
    json.dump({v:k for k, v in user_MBID_ID_dict.items()}, f)

print("Combining all listening events...")
item_artists = pd.DataFrame(columns=['item_id', 'artist_id'])


for le_path, save_path in zip(listening_events_paths, listening_events_save_paths):
    if os.path.exists(save_path):
        os.remove(save_path)
    print(f"Processing dataset {le_path}...")
    
    skip_file = False
    while not os.path.exists(le_path):
        print(f"File {le_path} not found.")
        retry = input("Press Enter to check again, or type 'skip' to skip this file: ")
        if retry.lower() == 'skip':
            print(f"Skipping {le_path}")
            skip_file = True
            break

    if skip_file:
        continue
    
    chunk = pd.read_csv(le_path, sep="\t", header=None, compression='bz2' if compressed else None)

    print("Read chunk")
    
    # Process listening events
    chunk.columns = ['user_id', 'timestamp', 'artist_id', 'item_id', 'age_at_interaction']
    
    chunk['user_id'] = chunk['user_id'].map(user_MBID_ID_dict).astype(np.int32)

    chunk['artist_id'] = chunk['artist_id'].map(lambda x: ','.join([str(int(artist_MBID_ID_dict[a_id])) for a_id in x.split(',')]))
    
    print("Converted artist and user IDs")
    
    

    new_items = set(chunk['item_id']) - set(item_MBID_ID_dict.keys())
    start_id = len(item_MBID_ID_dict)
    item_MBID_ID_dict.update({item: idx for idx, item in enumerate(new_items, start=start_id)})
    
    
    chunk['item_id'] = chunk['item_id'].map(item_MBID_ID_dict).astype(np.int32)
    print("Converted item IDs")
    
    chunk = chunk.dropna(subset=['item_id', 'artist_id'])
    

    
    item_artist_pairs = chunk[['item_id', 'artist_id']].drop_duplicates()

    item_artists = pd.concat([item_artists, item_artist_pairs], ignore_index=True).drop_duplicates()
    
    print("Updated item artists")
    
    chunk.drop(columns=['artist_id'], inplace=True)
    chunk.to_csv(save_path, sep="\t", index=False, header=False, compression='bz2' if compressed else None)
    print("Saved chunk")

with open(id_mbid_dict_item_path, 'w') as f:
    json.dump({v:k for k, v in item_MBID_ID_dict.items()}, f)
    


item_artists.to_csv(tracks_save_path, sep="\t", index=False, header=True)
    
print("Combining all listening events...")
if os.path.exists(all_listening_events_save_path):
    os.remove(all_listening_events_save_path)
for le_path in listening_events_save_paths:
    print(f"Processing dataset {le_path}...")
    chunk = pd.read_csv(le_path, sep="\t", compression='bz2' if compressed else None, header=None)
    chunk.columns = ['user_id', 'timestamp', 'item_id', 'age_at_interaction']
    chunk.to_csv(all_listening_events_save_path, sep="\t", index=False, header=False, mode='a', compression='bz2' if compressed else None)