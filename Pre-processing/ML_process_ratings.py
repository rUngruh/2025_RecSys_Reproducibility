######################################################################

# This script processes the MovieLens 1M dataset by filtering out items without genres and saving the processed data to a specified directory.
# It saves the interactions, genres, and user information in separate TSV files in line with other datasets used in the project.

######################################################################

import sys
import pandas as pd
import os

from dotenv import load_dotenv
from pathlib import Path
env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")

save_dir = dataset_dir + '/processed/movielens-1m'
data_dir = dataset_dir + '/raw/ml-1m'
ratingsSavePath = save_dir + '/interactions.tsv.bz2'
genresSavePath = save_dir + '/movies.tsv'
usersSavePath = save_dir + '/users.tsv'

genres_path = data_dir + '/movies.dat'
ratings_path = data_dir + '/ratings.dat'
users_path = data_dir + '/users.dat'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

    

print('Loading and processing batches...')
delimiter = '::'
compression= None

genre_subset = pd.DataFrame()

genre_subset = pd.read_csv(genres_path, delimiter=delimiter, header=None, encoding='latin-1')
genre_subset.columns = ['ItemID', 'Title', 'Genres']
genre_subset['genres'] = genre_subset['Genres'].apply(lambda x: x.split('|'))
genre_subset.drop(['Genres', 'Title'], axis=1, inplace=True)
print(genre_subset.head())

print(f'Number of items: {genre_subset.shape[0]}')    
empty_tags_count = genre_subset[genre_subset['genres'].apply(lambda x: len(x) == 0)].shape[0]
genre_subset = genre_subset[genre_subset['genres'].apply(lambda x: len(x) > 0)]
print(f"Number of rows without a tag: {empty_tags_count}")

valid_track_ids = genre_subset['ItemID'].unique()
genre_subset.rename(columns={'ItemID': 'item_id'}, inplace=True)
genre_subset['genres'] = genre_subset['genres'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
genres = set()
for genre_list in genre_subset['genres']:
    if isinstance(genre_list, str):
        genres.update(genre_list.split(','))

with open('../utils/ML_genres.txt', 'w') as f:
    for genre in genres:
        f.write(genre + '\n')


print('Processing complete.')


ratings = pd.read_csv(ratings_path, delimiter=delimiter, header=None)
ratings.columns = ['UserID', 'ItemID', 'Rating', 'Timestamp']
print(f'Number of ratings: {ratings.shape[0]}')
print(f'Number of unique users: {ratings["UserID"].nunique()}')
print(f'Number of unique items: {ratings["ItemID"].nunique()}')

ratings = ratings[ratings['ItemID'].isin(valid_track_ids)]
print(f'Number of ratings after filtering: {ratings.shape[0]}')
print(f'Number of unique users after filtering: {ratings["UserID"].nunique()}')
print(f'Number of unique items after filtering: {ratings["ItemID"].nunique()}')


ratings.rename(columns={'UserID': 'user_id',
                        'ItemID': 'item_id',
                        'Rating': 'rating',
                        'Timestamp' : 'timestamp'}, inplace=True)

genre_subset = genre_subset[genre_subset['item_id'].isin(ratings['item_id'])]
genre_subset.to_csv(genresSavePath, sep='\t', index=False, header=True, mode='w')

ratings.to_csv(ratingsSavePath, sep='\t', index=False, header=False, mode='w', compression='bz2' if ratingsSavePath.endswith('bz2') else None)

valid_users = ratings['user_id'].unique()

users = pd.read_csv(users_path, delimiter=delimiter, header=None)
users.columns = ['UserID', 'gender', 'age', 'Occupation', 'Zip-code']

users.drop(['Occupation', 'Zip-code'], axis=1, inplace=True)

users = users[users['UserID'].isin(valid_users)]
users.rename(columns={'UserID': 'user_id'}, inplace=True)

users.to_csv(usersSavePath, sep='\t', index=False, header=True, mode='w')