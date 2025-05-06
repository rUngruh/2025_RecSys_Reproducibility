import os
import pandas as pd
import argparse

import utils.age_processing as ap

from dotenv import load_dotenv
from pathlib import Path
env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")

mlhd_data_dir = dataset_dir + '/processed/MLHD_sampled_filtered'
bx_data_dir = dataset_dir + '/processed/Book-Crossing'
ml_data_dir = dataset_dir + '/processed/movielens-1m'

parser = argparse.ArgumentParser(description='Create user profiles in batches.')
parser.add_argument('--dataset', type=str, help='Dataset to use (MLHD, BX, or ML)', choices=['mlhd', 'bx',  'ml'], required=True)
parser.add_argument('--chunksize', type=int, default=10000000, help='Chunk size for processing listening events')
parser.add_argument('--weighted', type=bool, help='Use weighted listening events', default=True)
parser.add_argument('-in_batches', action='store_true', help='Process in batches')

args = parser.parse_args()

# Use the dataset argument
dataset = args.dataset
chunksize = args.chunksize
weighted = args.weighted
in_batches = args.in_batches

if dataset == 'mlhd':
    user_profile_stats_path = mlhd_data_dir + '/user_profile_stats.tsv' if not weighted else mlhd_data_dir + '/user_profile_stats_weighted.tsv'
    interactions_path = mlhd_data_dir + '/interactions.tsv.bz2'
    artist_path = mlhd_data_dir + '/artists.tsv'
    tracks_path = mlhd_data_dir + '/tracks.tsv'
    # user_path = mlhd_data_dir + '/users.tsv' # Not needed since column 'age_at_interaction' exists
    
    compressed = True if interactions_path.endswith('.bz2') else False
    
    artists = pd.read_csv(artist_path, sep='\t')
    artists['genres'] = artists['genres'].apply(lambda x: x.split(','))
    genre_dict = {str(artist_id): genres for artist_id, genres in artists[['artist_id', 'genres']].values}
    
    
    item_artists = pd.read_csv(tracks_path, sep='\t')
    item_artist_dict = item_artists.set_index('item_id')['artist_id'].to_dict()
    
    popularity_path = mlhd_data_dir + '/item_popularity.csv'
    

if dataset == 'bx':
    user_profile_stats_path = bx_data_dir + '/user_profile_stats.tsv' if not weighted else bx_data_dir + '/user_profile_stats_weighted.tsv'

    interactions_path = bx_data_dir + '/interactions.tsv.bz2'
    user_path = bx_data_dir + '/users.tsv'
    genre_path = bx_data_dir + '/books.tsv'
    
    genres = pd.read_csv(genre_path, sep='\t')
    genres['genres'] = genres['genres'].apply(lambda x: x.split(','))
    genre_dict = genres.set_index('item_id')['genres'].to_dict()
    
    users = pd.read_csv(user_path, sep='\t')
    user_age_dict = dict(zip(users['user_id'], users['age']))
    
    compressed = True if interactions_path.endswith('.bz2') else False
    
    popularity_path = bx_data_dir + '/item_popularity.csv'
    
if dataset == 'ml':
    user_profile_stats_path = ml_data_dir + '/user_profile_stats.tsv' if not weighted else ml_data_dir + '/user_profile_stats_weighted.tsv'
    interactions_path = ml_data_dir + '/interactions.tsv.bz2'
    genre_path = ml_data_dir + '/movies.tsv'
    user_path = ml_data_dir + '/users.tsv'
    
    genres = pd.read_csv(genre_path, sep='\t')
    genres['genres'] = genres['genres'].apply(lambda x: x.split(','))
    genre_dict = genres.set_index('item_id')['genres'].to_dict()
    
    users = pd.read_csv(user_path, sep='\t')
    user_age_dict = dict(zip(users['user_id'], users['age']))
    
    compressed = True if interactions_path.endswith('.bz2') else False
    
    popularity_path = ml_data_dir + '/item_popularity.csv'
    
    
    
   
   
#     interactions = pd.read_csv(interactions_path, sep='\t', compression='bz2' if compressed else None, chunksize=chunksize)

if in_batches:          
     
    stats_profile_data = []

    user_genre_sums = {}


    user_items = {}
         
    unique_items_per_user = {}
    interactions_per_user = {}
    interactions = 0
    for i, chunk in enumerate(pd.read_csv(interactions_path, sep='\t', compression='bz2' if compressed else None, chunksize=chunksize, header=None)):
        if dataset == 'mlhd':
            chunk.columns = ['user_id', 'timestamp', 'item_id', 'age_at_interaction']
        if dataset == 'bx':
            chunk.columns = ['user_id', 'item_id', 'rating']
            
        if dataset == 'ml':
            chunk.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        interactions += len(chunk)
        if i % 10 == 0:
            print(f"Processing batch {i}, current batch size: {chunksize}") #; out of x billion, percentage: {(i * chunksize / x) * 100:.3f}%")
        
        if dataset == 'bx' or dataset == 'ml':
            chunk['age_at_interaction'] = chunk['user_id'].map(user_age_dict)
            
            
        if weighted:
            chunk_user_interactions_dict = (chunk.copy()
            .groupby(['user_id', 'age_at_interaction'])
            .apply(lambda df: list(df['item_id']))
            .to_dict()
        )
        
        else:
            chunk_user_interactions_dict = (chunk.copy()
                .groupby(['user_id', 'age_at_interaction'])
                .apply(lambda df: list(set(df['item_id'])))
                .to_dict()
            )
            
        del chunk

        
        for key, new_items in chunk_user_interactions_dict.items():
            
            if key in user_items:
                new_unique_items = [t for t in list(set(new_items)) if t not in user_items[key]]
                    
            else:
                new_unique_items = [t for t in list(set(new_items))]
                # If the key does not exist in user_items, initialize it with the current track_ids
                user_items[key] = []
                user_genre_sums[key] = {}
                interactions_per_user[key] = 0
                unique_items_per_user[key] = 0
            
            interactions_per_user[key] += len(new_items)
            unique_items_per_user[key] += len(new_unique_items)
            
            if not weighted:
                new_items = new_unique_items # only add new unique items to genre computation
                    
            user_items[key].extend([t for t in new_unique_items])            
            
            for item_id in new_items:
                if dataset == 'mlhd':
                    genre_item_id = item_artist_dict.get(item_id, None)
                else:
                    genre_item_id = item_id
                if not isinstance(genre_item_id, int) and ',' in genre_item_id:
                    ids = genre_item_id.split(',')
                    ids = [id for id in ids]
                    genres = set()
                    for id in ids:
                        genres.update(genre_dict.get(id, []))
                else:
                    genres = genre_dict.get(genre_item_id, [])
                
                
                for genre in genres:
                    if genre in user_genre_sums[key]:
                        user_genre_sums[key][genre] += 1 / len(genres)
                    else:
                        user_genre_sums[key][genre] = 1 / len(genres)
    
    avg_popularity = {}
    avg_normalized_popularity = {}
    avg_age_group_popularity = {}
    avg_normalized_age_group_popularity = {}
    
    item_popularities = pd.read_csv(popularity_path, sep='\t')
    
    
    # Popularity Extension
    for key, items in user_items.items():
        age_group = ap.age_group(key[1], dataset, 'defined_ages')
        avg_popularity[key] = item_popularities[item_popularities['item_id'].isin(items)]['popularity'].mean()
        avg_normalized_popularity[key] = item_popularities[item_popularities['item_id'].isin(items)]['popularity_norm'].mean()
        avg_age_group_popularity[key] = item_popularities[item_popularities['item_id'].isin(items)][f'popularity_{age_group}'].mean()
        avg_normalized_age_group_popularity[key] = item_popularities[item_popularities['item_id'].isin(items)][f'popularity_{age_group}_norm'].mean()

    normalized_genre_sums = {}
        
    print(f"Processed {interactions} interactions.")
    print('Processed interactions. Now, computing genre distributions across users...')
    for key, user_genres in user_genre_sums.items():
        total_value = sum(user_genres.values())
        normalized_genre_sums[key] = {genre: value / total_value for genre, value in user_genres.items()}





    print('Creating dataframes...')
    
    stats_profile_data = []
    for key, user_genre_distribution in normalized_genre_sums.items():
            stats_profile_data.append({
                'user_id': key[0],
                'age': key[1],
                'num_interactions': interactions_per_user[key],
                'num_unique_items': unique_items_per_user[key],
                'normalized_genre_distribution': user_genre_distribution,
                'avg_popularity': avg_popularity[key],
                'avg_normalized_popularity': avg_normalized_popularity[key],
                'avg_age_group_popularity': avg_age_group_popularity[key],
                'avg_normalized_age_group_popularity': avg_normalized_age_group_popularity[key]                
            })


    print('Processing complete.')
    
    
    
else: # if not in_batches:
    
    user_profile_data = []
    stats_profile_data = []

    user_genre_sums = {}
    user_items = {}
    
    unique_items_per_user = {}
    interactions_per_user = {}
    
    avg_popularity = {}
    avg_normalized_popularity = {}
    avg_age_group_popularity = {}
    avg_normalized_age_group_popularity = {}
    
    item_popularities = pd.read_csv(popularity_path, sep='\t')

    chunk = pd.read_csv(interactions_path, sep='\t', compression='bz2' if compressed else None)
    
    if dataset == 'mlhd':
        chunk.columns = ['user_id', 'timestamp', 'item_id', 'age_at_interaction']
    if dataset == 'bx':
        chunk.columns = ['user_id', 'item_id', 'rating']
    if dataset == 'ml':
        chunk.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    interactions = len(chunk)

    if dataset == 'bx' or dataset == 'ml':
        chunk['age_at_interaction'] = chunk['user_id'].map(user_age_dict)
        
    if weighted:
        chunk_user_interactions_dict = (chunk
        .groupby(['user_id', 'age_at_interaction'])
        .apply(lambda df: list(df['item_id']))
        .to_dict()
    )
    
    else:
        chunk_user_interactions_dict = (chunk
            .groupby(['user_id', 'age_at_interaction'])
            .apply(lambda df: list(set(df['item_id'])))
            .to_dict()
        )
        
    del chunk

    
    for key, new_items in chunk_user_interactions_dict.items():
        
        new_unique_items = list(set(new_items))
        user_items[key] = new_unique_items
        interactions_per_user[key] = len(new_items)
        unique_items_per_user[key] = len(new_unique_items)
        user_genre_sums[key] = {}
        
        if not weighted:
            new_items = new_unique_items # only add new unique items to genre computation
        
        
        for item_id in new_items:
            if dataset == 'mlhd':
                genre_item_id = item_artist_dict.get(item_id, None)
            else:
                genre_item_id = item_id
            genres = genre_dict.get(genre_item_id, [])
        
            
            for genre in genres:
                if genre in user_genre_sums[key]:
                    user_genre_sums[key][genre] += 1 / len(genres)
                else:
                    user_genre_sums[key][genre] = 1 / len(genres)
                    
        # Normalize the aggregated dictionary
        current_user_genres = user_genre_sums.get(key, {})
        total_value = sum(current_user_genres.values())
        user_genre_sums[key] = {genre: value / total_value for genre, value in current_user_genres.items()}
        
        age_group = ap.age_group(key[1], dataset, 'defined_ages')
        avg_popularity[key] = item_popularities[item_popularities['item_id'].isin(new_unique_items)]['popularity'].mean()
        avg_normalized_popularity[key] = item_popularities[item_popularities['item_id'].isin(new_unique_items)]['popularity_norm'].mean()
        avg_age_group_popularity[key] = item_popularities[item_popularities['item_id'].isin(new_unique_items)][f'popularity_{age_group}'].mean()
        avg_normalized_age_group_popularity[key] = item_popularities[item_popularities['item_id'].isin(new_unique_items)][f'popularity_{age_group}_norm'].mean()
        
        
        
        
        
    
    print(f"Processed {interactions} interactions.")


    print('Creating dataframes...')

    stats_profile_data = []
    for key, user_genre_distribution in user_genre_sums.items():
            stats_profile_data.append({
                'user_id': key[0],
                'age': key[1],
                'num_interactions': interactions_per_user[key],
                'num_unique_items': unique_items_per_user[key],
                'normalized_genre_distribution': user_genre_distribution,
                'avg_popularity': avg_popularity[key],
                'avg_normalized_popularity': avg_normalized_popularity[key],
                'avg_age_group_popularity': avg_age_group_popularity[key],
                'avg_normalized_age_group_popularity': avg_normalized_age_group_popularity[key]                
            })

    print('Processing complete.')
    
print('Saving user profiles...')
if os.path.exists(user_profile_stats_path):
    os.remove(user_profile_stats_path)


stats_profile = pd.DataFrame(stats_profile_data)

stats_profile.to_csv(user_profile_stats_path, sep='\t', index=False)
