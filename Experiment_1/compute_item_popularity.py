import os
import pandas as pd
import argparse
import utils.age_processing as ap
from collections import defaultdict


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
parser.add_argument('-in_batches', action='store_true', help='Process the dataset in batches')
args = parser.parse_args()

# Use the dataset argument
dataset = args.dataset
in_batches = args.in_batches
chunk_size = 10000000

if dataset == 'mlhd':
    interactions_path = mlhd_data_dir + '/interactions.tsv.bz2'
    tracks_path = mlhd_data_dir + '/tracks.tsv'

    compressed = True if interactions_path.endswith('.bz2') else False

    save_path = mlhd_data_dir + '/item_popularity.csv'
    

if dataset == 'bx':
    interactions_path = bx_data_dir + '/interactions.tsv.bz2'
    user_path = bx_data_dir + '/users.tsv'
    genre_path = bx_data_dir + '/books.tsv'
    
    users = pd.read_csv(user_path, sep='\t')
    user_age_dict = dict(zip(users['user_id'], users['age']))
    
    compressed = True if interactions_path.endswith('.bz2') else False
    
    save_path = bx_data_dir + '/item_popularity.csv'
    
if dataset == 'ml':
    interactions_path = ml_data_dir + '/interactions.tsv.bz2'
    genre_path = ml_data_dir + '/movies.tsv'
    user_path = ml_data_dir + '/users.tsv'
    

    users = pd.read_csv(user_path, sep='\t')
    user_age_dict = dict(zip(users['user_id'], users['age']))
    
    compressed = True if interactions_path.endswith('.bz2') else False
    
    save_path = ml_data_dir + '/item_popularity.csv'
    
print(f'Loading interactions from {interactions_path}')
if dataset == 'mlhd':
    usecols = [0, 2, 3] if dataset == 'mlhd' else None
    dtype = {0: int, 2: int, 3: int} if dataset == 'mlhd' else {}
    
if not in_batches:
    interactions = pd.read_csv(interactions_path, sep='\t', compression='bz2' if compressed else None, header=None, usecols=usecols, dtype=dtype)

    if dataset == 'mlhd':
        interactions.columns = ['user_id', 'item_id', 'age_at_interaction']
    if dataset == 'bx':
        interactions.columns = ['user_id', 'item_id', 'rating']
    if dataset == 'ml':
        interactions.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        
    if dataset == 'bx' or dataset == 'ml':
        interactions['age_at_interaction'] = interactions['user_id'].map(user_age_dict)

    print('Adding age groups')
    interactions['age_group'] = interactions['age_at_interaction'].apply(lambda x: ap.age_group(x, dataset, 'defined_ages'))
    interactions = interactions[['item_id', 'user_id', 'age_group']]
    # Save popularities as dicts

    print('Computing popularity')
    song_interactions = interactions.groupby('item_id').size().to_dict()
    distinct_song_interactions = interactions.groupby('item_id')['user_id'].nunique().to_dict()


    song_interactions_by_age = interactions.groupby(['item_id', 'age_group']).size().unstack(fill_value=0).to_dict()
    distinct_song_interactions_by_age = interactions.groupby(['item_id', 'age_group'])['user_id'].nunique().unstack(fill_value=0).to_dict()

    del interactions

    print('Creating DataFrame')
    df = pd.DataFrame.from_dict(song_interactions, orient='index', columns=['popularity']).reset_index()
    df.rename(columns={'index': 'item_id'}, inplace=True)


    distinct_df = pd.DataFrame.from_dict(distinct_song_interactions, orient='index', columns=['popularity_distinct']).reset_index()
    distinct_df.rename(columns={'index': 'item_id'}, inplace=True)

    df = df.merge(distinct_df, on='item_id', how='outer')


    age_pop_df = pd.DataFrame(song_interactions_by_age)
    age_pop_df.columns = [f'popularity_{age}' for age in age_pop_df.columns]
    age_pop_df = age_pop_df.reset_index().rename(columns={'index': 'item_id'})
    df = df.merge(age_pop_df, on='item_id', how='outer')


    distinct_age_pop_df = pd.DataFrame(distinct_song_interactions_by_age)
    distinct_age_pop_df.columns = [f'popularity_distinct_{age}' for age in distinct_age_pop_df.columns]
    distinct_age_pop_df = distinct_age_pop_df.reset_index().rename(columns={'index': 'item_id'})
    df = df.merge(distinct_age_pop_df, on='item_id', how='outer')

    df = df.fillna(0).astype({col: int for col in df.columns if col != 'item_id'})

else:
    print('Processing in batches...')
    chunk_iter = pd.read_csv(interactions_path, sep='\t', compression='bz2' if compressed else None,
                             header=None, usecols=usecols, dtype=dtype, chunksize=chunk_size)

    total_counts = defaultdict(int)
    distinct_users = defaultdict(int)
    age_group_counts_total = defaultdict(lambda: defaultdict(int))
    age_group_users = defaultdict(lambda: defaultdict(int))

    for i, chunk in enumerate(chunk_iter):
        print(f'Processing chunk {i+1}')
        if dataset == 'mlhd':
            chunk.columns = ['user_id', 'item_id', 'age_at_interaction']
        if dataset == 'bx':
            chunk.columns = ['user_id', 'item_id', 'rating']
            chunk['age_at_interaction'] = chunk['user_id'].map(user_age_dict)
        if dataset == 'ml':
            chunk.columns = ['user_id', 'item_id', 'rating', 'timestamp']
            chunk['age_at_interaction'] = chunk['user_id'].map(user_age_dict)

        chunk['age_group'] = chunk['age_at_interaction'].apply(lambda x: ap.age_group(x, dataset, 'defined_ages'))
        chunk = chunk[['item_id', 'user_id', 'age_group']]

        # Count total interactions per item
        item_counts = chunk['item_id'].value_counts()

        # Count distinct users per item
        distinct_counts = chunk.groupby('item_id')['user_id'].nunique()

        # Count interactions per item per age group
        age_group_counts = chunk.groupby(['item_id', 'age_group']).size().unstack(fill_value=0)

        # Count distinct users per item per age group
        age_group_distinct = chunk.groupby(['item_id', 'age_group'])['user_id'].nunique().unstack(fill_value=0)
        
        for item_id, count in item_counts.items():
            total_counts[item_id] += count

        # Aggregate distinct user counts
        for item_id, n_users in distinct_counts.items():
            
            distinct_users[item_id] += n_users

        # Aggregate age-group interaction counts
        for item_id, row in age_group_counts.iterrows():
            for age_group, count in row.items():
                age_group_counts_total[item_id][age_group] += count

        # Aggregate distinct user counts per age group
        for item_id, row in age_group_distinct.iterrows():
            for age_group, n_users in row.items():
                age_group_users[item_id][age_group] += n_users
                
        
        
        
    print('Creating DataFrame')

    df_total = pd.DataFrame.from_dict(total_counts, orient='index', columns=['popularity']).reset_index().rename(columns={'index': 'item_id'}, inplace=True)


    df_distinct = pd.DataFrame(distinct_users.items(), columns=['item_id', 'popularity_distinct'])


    # Interactions per item per age group
    df_age = pd.DataFrame.from_dict(age_group_counts_total, orient='index').fillna(0).reset_index().rename(columns={'index': 'item_id'})
    df_age.columns = ['item_id'] + [f'popularity_{col}' for col in df_age.columns if col != 'item_id']

    # Distinct users per item per age group
    df_age_distinct = pd.DataFrame.from_dict({item: {age: len(users) for age, users in age_dict.items()}
                                            for item, age_dict in age_group_users.items()},
                                            orient='index').fillna(0).reset_index().rename(columns={'index': 'item_id'})
    df_age_distinct.columns = ['item_id'] + [f'popularity_distinct_{col}' for col in df_age_distinct.columns if col != 'item_id']

    # Final merge
    df = df_total.merge(df_distinct, on='item_id', how='outer')
    df = df.merge(df_age, on='item_id', how='outer')
    df = df.merge(df_age_distinct, on='item_id', how='outer')

    df = df.fillna(0).astype({col: int for col in df.columns if col != 'item_id'})



popularity_cols = [col for col in df.columns if col != 'item_id']
for col in popularity_cols:
    norm_col = f'{col}_norm'
    max = df[col].max()
    df[norm_col] = df[col] / max if max > 0 else 0
    
df.sort_values(by='popularity', ascending=False, inplace=True)
print(df.head())

df.to_csv(save_path, sep='\t', index=False)