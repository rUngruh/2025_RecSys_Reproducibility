######################################################################

# This script gathers the results of the experiment and computes preference profiles for users.
# It processes user and recommendation data, computes user profiles, and saves the results to a file.
# It computes the genre distribution for each user based on their interactions and the recommendations they received.
# It also computes the average popularity of items in the recommendations and saves the results to a file.
# The script uses argparse to handle command-line arguments for dataset selection, model selection, and other parameters.

######################################################################


import os
import ast
import pandas as pd
import utils.age_processing as ap
import utils.genre_processing as gp
import argparse
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='Gather the Results of the Experiment and compute Preference profiles.')
parser.add_argument('--age_type', type=str, help='Type of age grouping to use', choices=['finegrained_age', 'binary_age', 'finegrained_child_ages', 'defined_ages'], default='defined_ages')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml, bx, mlhd)', choices=['ml', 'bx', 'mlhd'], default='ml')
#parser.add_argument('--year', type=int, help='Year of the experiment data for identification (only MLHD)', default=2013)
parser.add_argument('--models', type=str, help='Best models from the experiment', nargs='+', default=["MostPop", "Random_seed=42"])
parser.add_argument('--child_models', type=str, help='Best child models from the experiment', nargs='+', default=["MostPop", "Random_seed=42"])
parser.add_argument('-filtered', action='store_true', help='Whether k-core filtering was applied')
parser.add_argument('--cutoff', type=int, help='Cutoff for the recommendation profiles', default=50)
args = parser.parse_args()


models_short = args.models
child_models_short = args.child_models


age_type = args.age_type  
dataset = args.dataset
#year = args.year
filtered = args.filtered
cut = args.cutoff


import json
with open('../../utils/best_models.json', 'r') as f:
    best_models = json.load(f)

models = [m for m in best_models[dataset]['all'] if m.split('_')[0] in models_short]
child_models = [m for m in best_models[dataset]['child'] if m.split('_')[0] in child_models_short]


import os
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('../..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")


if dataset == 'ml':
    data_dir = dataset_dir + f'/processed/ml_rec{"_filtered" if filtered else ""}'
    recommendations_path = f'../elliot/Results/'
    child_recommendations_path = f'../elliot/Results/'
    source_dir = dataset_dir + '/processed/movielens-1m'
    genre_path = source_dir + '/movies.tsv'
    genres = gp.ML_genres
    
elif dataset == 'mlhd':
    data_dir = dataset_dir + f'/processed/mlhd_rec{"_filtered" if filtered else ""}'
    recommendations_path = f'../elliot/Results/'
    child_recommendations_path = f'../elliot/Results/'
    source_dir = dataset_dir + '/processed/mlhd_sampled_filtered'
    genre_path = source_dir + '/artists.tsv'
    genres = gp.MLHD_genres
    
elif dataset == 'bx':
    data_dir = dataset_dir + f'/processed/bx_rec{"_filtered" if filtered else ""}'
    recommendations_path = f'../elliot/Results/'
    child_recommendations_path = f'../elliot/Results/'
    source_dir = dataset_dir + '/processed/Book-Crossing'
    genre_path = source_dir + '/books.tsv'
    genres = gp.BX_genres
    
    


results_path = f'../Results/{dataset}/user_and_recommendation_genre_distributions.tsv'

if not os.path.exists(f'../Results/{dataset}'):
    os.makedirs(f'../Results/{dataset}')

train_path = data_dir + f'/train.tsv'
validation_path = data_dir + f'/validation.tsv'
test_path = data_dir + f'/test.tsv'
user_info_path = data_dir + f'/user_info.tsv'
# listening_events_path =  + f'/listening-events.tsv.bz2'

columns = ['user_id', 'item_id', 'rating'] if dataset == 'bx' else ['user_id', 'item_id', 'rating', 'timestamp']


if not os.path.exists(train_path) or not os.path.exists(validation_path) or not os.path.exists(test_path):
    raise FileNotFoundError(f"One of the required files does not exist: {train_path}, {validation_path}, {test_path}")
for model in models:
    file_path = rf'{recommendations_path}/{model}.tsv'  # raw string to handle $ correctly
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Recommendation file does not exist: {file_path}")
for model in child_models:
    file_path = rf'{child_recommendations_path}/{model}.tsv' 
    if not os.path.exists(child_recommendations_path + f'/{model}.tsv'):
        raise FileNotFoundError(rf"Child recommendation file does not exist: {file_path}")
if not os.path.exists(user_info_path):
    raise FileNotFoundError(f"User info file does not exist: {user_info_path}")
if not os.path.exists(genre_path):
    raise FileNotFoundError(f"Genre file does not exist: {genre_path}")
if not os.path.exists(data_dir + '/item_popularity.tsv') or not os.path.exists(data_dir + '/item_popularity_child.tsv'):
    print('Please run the item popularity script first.')
    raise FileNotFoundError(f"Popularity file does not exist: {data_dir + '/item_popularity.tsv'} or {data_dir + '/item_popularity_child.tsv'}")

train_popularity = pd.read_csv(data_dir + '/item_popularity.tsv', sep='\t')
train_child_popularity = pd.read_csv(data_dir + '/item_popularity_child.tsv', sep='\t')

popularity_dict_raw = train_popularity.set_index('item_id').to_dict('index')
child_pop_dict_raw = train_child_popularity.set_index('item_id').to_dict('index')


popularity_dict = defaultdict(lambda: {'popularity': 0, 'popularity_norm': 0}, popularity_dict_raw)
child_pop_dict = defaultdict(lambda: {'popularity': 0, 'popularity_norm': 0}, child_pop_dict_raw)

del train_popularity
del train_child_popularity
del popularity_dict_raw
del child_pop_dict_raw

# Load the data
train = pd.read_csv(train_path, sep='\t', header=None, names=columns)
validation = pd.read_csv(validation_path, sep='\t', header=None, names=columns)
test = pd.read_csv(test_path, sep='\t', header=None, names=columns)
users = pd.read_csv(user_info_path, sep='\t')
    

train = pd.merge(train, users, on='user_id', how='left')
validation = pd.merge(validation, users, on='user_id', how='left')
test = pd.merge(test, users, on='user_id', how='left')

users = users[users['user_id'].isin(train['user_id'].unique())]
user_age_dict = dict(zip(users['user_id'], users['age']))

# les = pd.read_csv(listening_events_path, sep='\t')
# les = les[les['user_id'].isin(users['user_id'].unique())]




print('Loaded User Data and Recommendation Data')


if dataset == 'mlhd':
    tracks_path = source_dir + '/tracks.tsv'
    item_artists = pd.read_csv(tracks_path, sep='\t')
    item_artist_dict = item_artists.set_index('item_id')['artist_id'].to_dict()
    del item_artists
    
    artists = pd.read_csv(genre_path, sep='\t')
    artists['genres'] = artists['genres'].apply(lambda x: x.split(','))
    item_genre_dict = {str(artist_id): genres for artist_id, genres in artists[['artist_id', 'genres']].values}
    
else:
    item_genres = pd.read_csv(genre_path, sep='\t')
    item_genres['genres'] = item_genres['genres'].apply(lambda x: x.split(','))
    item_genre_dict = item_genres.set_index('item_id')['genres'].to_dict()
    del item_genres

print("Loaded track and genre information.")

def compute_user_profiles(frame, cutoff=None):
    # Compute the user profiles from a given frame

    user_genre_sums = {}
    interactions_per_user = {}
    
    avg_popularity = {}
    avg_normalized_popularity = {}
    avg_child_popularity = {}
    avg_child_normalized_popularity = {}
    
    user_interactions_dict = (frame
        .groupby('user_id')
        .apply(lambda df: list(df['item_id']))
        .to_dict()
        )
    
    # key == user_id, value == list of item_ids
    for key, new_items in user_interactions_dict.items():
        if cutoff is not None:
            new_items = new_items[:cutoff]
            
        interactions_per_user[key] = len(new_items)
        user_genre_sums[key] = {}
        
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
                    genres.update(item_genre_dict.get(id, []))
            else:
                genres = item_genre_dict.get(genre_item_id, [])
        
            for genre in genres:
                if genre in user_genre_sums[key]:
                    user_genre_sums[key][genre] += 1 / len(genres)
                else:
                    user_genre_sums[key][genre] = 1 / len(genres)

        
        avg_popularity[key] = float(np.mean([popularity_dict[item_id]['popularity'] for item_id in new_items]))    
        avg_normalized_popularity[key] = float(np.mean([popularity_dict[item_id]['popularity_norm'] for item_id in new_items]))
        avg_child_popularity[key] = float(np.mean([child_pop_dict[item_id]['popularity'] for item_id in new_items]))
        avg_child_normalized_popularity[key] = float(np.mean([child_pop_dict[item_id]['popularity_norm'] for item_id in new_items]))
        # avg_popularity[key] = train_popularity[train_popularity['item_id'].isin(new_items)]['popularity'].mean()
        # avg_normalized_popularity[key] = train_popularity[train_popularity['item_id'].isin(new_items)]['popularity_norm'].mean()
        # avg_child_popularity[key] = train_child_popularity[train_child_popularity['item_id'].isin(new_items)]['popularity'].mean()
        # avg_child_normalized_popularity[key] = train_child_popularity[train_child_popularity['item_id'].isin(new_items)]['popularity_norm'].mean()
        
    # Normalizing user profiles
    user_profiles = {}
    for key, genre_dict in user_genre_sums.items():
            total_value = sum(genre_dict.values())
            user_profiles[key] = {genre: value / total_value for genre, value in genre_dict.items()}
    
    genre_df = pd.DataFrame({
        'user_id': list(user_profiles.keys()),
        'genre_distribution': list(user_profiles.values())
    }).set_index('user_id')
    interactions_df = pd.DataFrame.from_dict(interactions_per_user, orient='index', columns=['interactions'])
    popularity_df = pd.DataFrame.from_dict(avg_popularity, orient='index', columns=['avg_popularity'])
    normalized_popularity_df = pd.DataFrame.from_dict(avg_normalized_popularity, orient='index', columns=['avg_normalized_popularity'])
    child_popularity_df = pd.DataFrame.from_dict(avg_child_popularity, orient='index', columns=['avg_child_popularity'])
    child_normalized_popularity_df = pd.DataFrame.from_dict(avg_child_normalized_popularity, orient='index', columns=['avg_child_normalized_popularity'])
    
    profile_df = genre_df \
    .join(interactions_df) \
    .join(popularity_df) \
    .join(normalized_popularity_df) \
    .join(child_popularity_df) \
    .join(child_normalized_popularity_df) \
    .reset_index()
    return profile_df




users.sort_values('age', inplace=True)
users['age_group'] = users['age'].apply(lambda x: ap.age_group(x, dataset, age_type))
users = users.reset_index(drop=True)

train_profiles = compute_user_profiles(train)
validation_profiles = compute_user_profiles(validation)
test_profiles = compute_user_profiles(test)

print("Computed user profiles")

# Compute the recommendation profiles

model_profiles = {}
child_model_profiles = {}


new_models = set()
for model in models:
    model_name = model
    print(f"Processing {model}")
    recommendations = pd.read_csv(recommendations_path + f'/{model}.tsv', sep='\t', header=None, names=['user_id', 'item_id', 'score'])
    new_model_name = model_name.split('_')[0]
    model_profiles[new_model_name] = compute_user_profiles(recommendations, cutoff=cut)

    new_models.update([new_model_name])
    
for model in child_models:
    model_name = model
    print(f"Processing {model}")
    child_recommendations = pd.read_csv(child_recommendations_path + f'/{model}.tsv', sep='\t', header=None, names=['user_id', 'item_id', 'score'])
    new_model_name = model_name.split('_')[0]
    child_model_profiles[new_model_name] = compute_user_profiles(child_recommendations, cutoff=cut)
    new_models.update([new_model_name])
models = new_models

def prefix_columns(df, prefix):
    df = df.copy()
    df.columns = [f"{prefix}_{col}" if col not in ['user_id', 'age', 'age_group'] else col for col in df.columns]
    return df

train_profiles = prefix_columns(train_profiles, 'train')
validation_profiles = prefix_columns(validation_profiles, 'validation')
test_profiles = prefix_columns(test_profiles, 'test')

combined_profiles = users \
    .merge(train_profiles, on='user_id', how='outer') \
    .merge(validation_profiles, on='user_id', how='outer') \
    .merge(test_profiles, on='user_id', how='outer')

for model in models:
    model_df = prefix_columns(model_profiles[model], f"{model}")
    child_model_df = prefix_columns(child_model_profiles[model], f"child_{model}")
    
    combined_profiles = combined_profiles \
        .merge(model_df, on='user_id', how='outer') \
        .merge(child_model_df, on='user_id', how='outer')

print("Computed recommendation profiles.")

combined_profiles.to_csv(results_path, sep='\t', index=False, header=True)