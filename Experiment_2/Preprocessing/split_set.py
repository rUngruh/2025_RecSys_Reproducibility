######################################################################

# This scipt creates train, validation and test split, including k-core filtering and removal of invalid profiles.
# It also creates a child dataset for child users.
# The details can be specified in the command line arguments.

######################################################################

import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Create train, validation and test split, including k-core filtering and removal of invalid profiles.')
parser.add_argument('--dataset', type=str, help='Dataset to use (mlhd, bx, or ml)', choices=['mlhd', 'bx', 'ml'], default='ml')
parser.add_argument('--year', type=int, help='Year to filter the dataset (indicator as per filtering for timeframe)', default=2013)
parser.add_argument('-remove_missing_profiles', action='store_true', help='Remove users with missing items in train, validation, or test sets')
parser.add_argument('--k_core_filtering_user', type=int, help='Minimum number of interactions per user, use None if no k_core_filtering', default=10)
parser.add_argument('--k_core_filtering_item', type=int, help='Minimum number of interactions per item, use None if no k_core_filtering', default=50)
parser.add_argument('--validation_start', type=str, help='Start date for the validation split', default='2013-09-01')
parser.add_argument('--test_start', type=str, help='Start date for the test split', default='2013-10-01')
parser.add_argument('--train_split_size', type=float, help='Train split size', default=0.8)
parser.add_argument('--validation_split_size', type=float, help='Validation split size', default=0.1)
parser.add_argument('--min_playcount', type=int, help='Minimum number of playcounts of user of a certain item.', default=1)
parser.add_argument('--min_rating', type=int, help='Minimum rating per item', default=0)
parser.add_argument('-binarize', action='store_true', help='Binarize the ratings')
parser.add_argument('-sample_users', action='store_true', help='Sample users from the dataset')
parser.add_argument('--sample_size', type=int, help='Number of users to sample', default=10000)
args = parser.parse_args()

# Use the dataset argument
dataset = args.dataset

remove_missing_profiles = args.remove_missing_profiles
k_core_filtering_user = args.k_core_filtering_user
k_core_filtering_item = args.k_core_filtering_item
binarize = args.binarize
sample_users = args.sample_users
sample_size = args.sample_size


if dataset == 'mlhd': # or dataset == 'ml':
    validation_start = pd.to_datetime(args.validation_start)
    test_start = pd.to_datetime(args.test_start)
    year = args.year
    
if dataset == 'bx' or dataset == 'ml':
    train_size = args.train_split_size
    validation_size = args.validation_split_size
    test_size = 1 - train_size - validation_size
    
min_playcount = args.min_playcount
min_rating = args.min_rating

from dotenv import load_dotenv
from pathlib import Path
env_path = Path('../..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")




if dataset == 'mlhd':
    data_dir = dataset_dir + f'/processed/mlhd_rec_{year}'
    save_path = dataset_dir + f'/processed/mlhd_rec{"_filtered" if k_core_filtering_user else ""}'
    user_stats_path = dataset_dir + f'/processed/mlhd_sampled_filtered'

elif dataset == 'bx':
    data_dir = dataset_dir + '/processed/Book-Crossing'
    save_path = dataset_dir + f'/processed/bx_rec{"_filtered" if k_core_filtering_user else ""}'
    user_stats_path = dataset_dir + f'/processed/Book-Crossing'
elif dataset == 'ml':
    data_dir = dataset_dir + '/processed/movielens-1m'
    save_path = dataset_dir + f'/processed/ml_rec{"_filtered" if k_core_filtering_user else ""}'
    user_stats_path = dataset_dir + f'/processed/movielens-1m'


        
train_path = save_path + f'/train.tsv'
validation_path = save_path + f'/validation.tsv'
test_path = save_path + f'/test.tsv'
user_info_path = save_path + f'/user_info.tsv'
item_artist_dict_path = save_path + f'/item_artist_dict.json'

train_child_path = save_path + f'/train_child.tsv'
validation_child_path = save_path + f'/validation_child.tsv'
test_child_path = save_path + f'/test_child.tsv'

if not os.path.exists(save_path):
    os.makedirs(save_path)


interactions_path = f'{data_dir}/interactions.tsv.bz2'

compressed = True if interactions_path.endswith('bz2') else False

if os.path.exists(train_path):
    os.remove(train_path)
if os.path.exists(validation_path):
    os.remove(validation_path)
if os.path.exists(test_path):
    os.remove(test_path)
if os.path.exists(user_info_path):
    os.remove(user_info_path)
if os.path.exists(train_child_path):
    os.remove(train_child_path)
if os.path.exists(validation_child_path):
    os.remove(validation_child_path)
if os.path.exists(test_child_path):
    os.remove(test_child_path)
    
interactions = pd.read_csv(interactions_path, sep='\t', compression='bz2', header=None)
if dataset == 'mlhd':
    interactions.columns = ['user_id', 'timestamp', 'item_id', 'age_at_interaction', 'count']
if dataset == 'bx':
    interactions.columns = ['user_id', 'item_id', 'rating']
if dataset == 'ml':
    interactions.columns = ['user_id', 'item_id', 'rating', 'timestamp']


    



if dataset == 'mlhd':
    print(interactions.head(n=5))
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    interactions = interactions.sort_values(by=['user_id', 'timestamp'])
    interactions['rating'] = interactions['count']
    user_info = interactions[['user_id', 'age_at_interaction']].drop_duplicates().rename(columns={'age_at_interaction': 'age'})
    
    interactions = interactions.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    
elif dataset == 'bx':
    interactions.sort_values(['user_id', 'item_id', 'rating'], inplace=True)
    
    
    
    users = pd.read_csv(data_dir + '/users.tsv', sep='\t')
    user_info = users[['user_id', 'age']]
    del users
    if min_playcount > 1:
        interactions['count'] = interactions.groupby(['user_id', 'item_id'])['rating'].transform('count')
        
    interactions = interactions.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
        
elif dataset == 'ml':
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'], unit='s')

    interactions = interactions.sort_values(by=['user_id', 'timestamp'])
    
    users = pd.read_csv(data_dir + '/users.tsv', sep='\t')
    
    user_info = users[['user_id', 'age']]
    
    if min_playcount > 1:
        interactions['count'] = interactions.groupby(['user_id', 'item_id'])['rating'].transform('count')
    
    interactions = interactions.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    
if sample_users:
    weighted = True # In our experiments, we always use weighted sampling
    #user_stats = pd.read_csv(user_stats_path + f'/user_profile_stats{"_weighted" if weighted else ""}.tsv', sep='\t')
    # age_distribution = user_stats['age'].value_counts(normalize=True).sort_index()
    # del user_stats
    user_with_sufficient_interactions = interactions.groupby('user_id').filter(lambda x: len(x) >= k_core_filtering_user)['user_id'].unique()
    
    user_info = user_info[user_info['user_id'].isin(user_with_sufficient_interactions)]
    user_info = user_info.groupby('age', group_keys=False).apply(lambda x: x.sample(frac=sample_size/len(user_info), random_state=42)) 
    
    print(f"User age information")
    print(user_info['age'].value_counts(normalize=True).sort_index())
    
    sampled_users = set(user_info['user_id'].unique())
    # all_users = user_info['user_id'].unique()
    # sampled_users = set()
    # for age, prob in age_distribution.items():
    #     available_users = user_info[user_info['age'] == age]['user_id']
    #     n_sample = min(len(available_users), int(sample_size * prob))
    #     if n_sample > 0:
    #         age_users = available_users.sample(n=n_sample, random_state=42).tolist()
    #         sampled_users.update(age_users)
            
    # user_info = user_info[user_info['user_id'].isin(sampled_users)]
    interactions = interactions[interactions['user_id'].isin(sampled_users)]
    print(f"Sampled {len(sampled_users)} users from the dataset")
    
        
if min_playcount > 1:
    interactions = interactions[interactions['count'] >= min_playcount]


if min_rating>0:
    interactions = interactions[interactions['rating'] >= min_rating]

if 'count' in interactions.columns:
    interactions.drop(columns=['count'], inplace=True)

if binarize:
    interactions['rating'] = 1

if k_core_filtering_user:
    nothing_removed = False
    while nothing_removed == False:
        init_users = interactions['user_id'].nunique()
        init_items = interactions['item_id'].nunique()
        init_interactions = interactions.shape[0]
        print(f"Initial number of users: {init_users}")
        print(f"Initial number of items: {init_items}")
        print(f"Initial number of interactions: {init_interactions}")
        
        # Filter users and items that meet the k-core threshold
        user_profile_counts = interactions.groupby('user_id').size()
        invalid_users = user_profile_counts[user_profile_counts < k_core_filtering_user].index
        print(f'Number of invalid users: {len(invalid_users)}')
        
        item_counts = interactions.groupby('item_id').size()
        invalid_items = item_counts[item_counts < k_core_filtering_item].index
        print(f'Number of invalid_items: {len(invalid_items)}')
        
        # Keep only listening events with valid users and items
        interactions = interactions[~interactions['user_id'].isin(invalid_users)]

        
        interactions = interactions[~interactions['item_id'].isin(invalid_items)]
        
        # Update the user_info DataFrame to keep only valid users
        user_info = user_info[~user_info['user_id'].isin(invalid_users)]

        final_users = interactions['user_id'].nunique()
        final_items = interactions['item_id'].nunique()
        final_interactions = interactions.shape[0]
        print(f"Final number of users: {final_users}")
        print(f"Final number of items: {final_items}")
        print(f"Final number of interactions: {final_interactions}")
        print()
        if final_users == init_users and final_items == init_items:
            nothing_removed = True
    print("Finished k-core filtering")

if dataset == 'mlhd':
    child_chunk = interactions[interactions['age_at_interaction'] < 17]
    child_chunk = child_chunk[['user_id', 'item_id', 'rating', 'timestamp']]
    interactions = interactions[['user_id', 'item_id', 'rating', 'timestamp']]
    print("Finished further processing")

    train_chunk = interactions[interactions['timestamp'] < validation_start]
    validation_chunk = interactions[(interactions['timestamp'] >= validation_start) & (interactions['timestamp'] < test_start)]
    test_chunk = interactions[interactions['timestamp'] >= test_start]
    print("Finished splitting")
    
elif dataset == 'bx' or dataset == 'ml': # Remove ml if temporal split wanted
    interactions = interactions[['user_id', 'item_id', 'rating']]
    train_chunks = []
    validation_chunks = []
    test_chunks = []

    for user_id, user_data in interactions.groupby('user_id'):
        user_data = user_data.sample(frac=1, random_state=42)  # Shuffle user data
        
        n = len(user_data)
        n_train = int(train_size * n)
        n_validation = int(validation_size * n)
        
        train = user_data.iloc[:n_train]
        validation = user_data.iloc[n_train:n_train + n_validation]
        test = user_data.iloc[n_train + n_validation:]

        train_chunks.append(train)
        validation_chunks.append(validation)
        test_chunks.append(test)

    train_chunk = pd.concat(train_chunks)
    validation_chunk = pd.concat(validation_chunks)
    test_chunk = pd.concat(test_chunks)
    print("Finished splitting")

# elif dataset == 'ml': # Use this if ml with temporal split
#     child_chunk = interactions[interactions['user_id'].isin(user_info[user_info['age'] < 18]['user_id'])]
#     child_chunk = child_chunk[['user_id', 'item_id', 'rating', 'timestamp']]
#     interactions = interactions[['user_id', 'item_id', 'rating', 'timestamp']]
    
#     train_chunk = interactions[interactions['timestamp'] < validation_start]
#     validation_chunk = interactions[(interactions['timestamp'] >= validation_start) & (interactions['timestamp'] < test_start)]
#     test_chunk = interactions[interactions['timestamp'] >= test_start]

del interactions

if remove_missing_profiles:
    print("Removing missing profiles")

    # Get unique user IDs from each dataset once
    train_user_ids = set(train_chunk['user_id'].unique())
    validation_user_ids = set(validation_chunk['user_id'].unique())
    test_user_ids = set(test_chunk['user_id'].unique())

    # Find valid user IDs that are present in all three datasets
    valid_user_ids = train_user_ids & validation_user_ids & test_user_ids

    print(f'Number of valid users: {len(valid_user_ids)}')

    # Filter user_info and chunks by valid user IDs
    user_info = user_info[user_info['user_id'].isin(valid_user_ids)]
    train_chunk = train_chunk[train_chunk['user_id'].isin(valid_user_ids)]
    validation_chunk = validation_chunk[validation_chunk['user_id'].isin(valid_user_ids)]
    test_chunk = test_chunk[test_chunk['user_id'].isin(valid_user_ids)]
    if dataset == 'mlhd' or dataset == 'ml':
        child_chunk = child_chunk[child_chunk['user_id'].isin(valid_user_ids)]
        


if dataset == 'mlhd': # or dataset == 'ml':
    train_child_chunk = child_chunk[child_chunk['timestamp'] < validation_start]
    validation_child_chunk = child_chunk[(child_chunk['timestamp'] >= validation_start) & (child_chunk['timestamp'] < test_start)]
    test_child_chunk = child_chunk[child_chunk['timestamp'] >= test_start]

elif dataset == 'bx' or dataset == 'ml':
    train_child_chunk = train_chunk[train_chunk['user_id'].isin(user_info[user_info['age'] < 18]['user_id'])]
    validation_child_chunk = validation_chunk[validation_chunk['user_id'].isin(user_info[user_info['age'] < 18]['user_id'])]
    test_child_chunk = test_chunk[test_chunk['user_id'].isin(user_info[user_info['age'] < 18]['user_id'])]
    print("Finished further processing")

    
train_chunk.to_csv(train_path, sep='\t', index=False, header=False)
validation_chunk.to_csv(validation_path, sep='\t', index=False, header=False)
test_chunk.to_csv(test_path, sep='\t', index=False, header=False)
user_info.to_csv(user_info_path, sep='\t', index=False, header=True)

train_child_chunk.to_csv(train_child_path, sep='\t', index=False, header=False)
validation_child_chunk.to_csv(validation_child_path, sep='\t', index=False, header=False)
test_child_chunk.to_csv(test_child_path, sep='\t', index=False, header=False)