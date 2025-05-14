######################################################################

# This script computes item popularity on the training set of the dataset.

######################################################################


import os
import pandas as pd
import argparse
import utils.age_processing as ap
from collections import defaultdict


from dotenv import load_dotenv
from pathlib import Path
env_path = Path('../..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")

parser = argparse.ArgumentParser(description='Create user profiles in batches.')
parser.add_argument('--dataset', type=str, help='Dataset to use (MLHD, BX, or ML)', choices=['mlhd', 'bx',  'ml'], required=True)

args = parser.parse_args()

# Use the dataset argument
dataset = args.dataset


data_dir = dataset_dir + f'/processed/{dataset}_rec_filtered'
interactions_path = data_dir + '/train.tsv'
child_interactions_path = data_dir + '/train_child.tsv'

save_path_full = data_dir + '/item_popularity.tsv'
save_path_child = data_dir + '/item_popularity_child.tsv'

columns = ['user_id', 'item_id', 'rating'] if dataset == 'bx' else ['user_id', 'item_id', 'rating', 'timestamp']



#-----------------------
interactions = pd.read_csv(interactions_path, sep='\t', header=None, names=columns)

item_interactions = interactions.groupby('item_id').size().to_dict()

del interactions

print('Creating DataFrame')
df = pd.DataFrame.from_dict(item_interactions, orient='index', columns=['popularity']).reset_index()
df.rename(columns={'index': 'item_id'}, inplace=True)

df = df.fillna(0).astype({col: int for col in df.columns if col != 'item_id'})


popularity_cols = [col for col in df.columns if col != 'item_id']
for col in popularity_cols:
    norm_col = f'{col}_norm'
    max = df[col].max()
    df[norm_col] = df[col] / max if max > 0 else 0
    
df.sort_values(by='popularity', ascending=False, inplace=True)
print(df.head())

df.to_csv(save_path_full, sep='\t', index=False)

#-----------------------
child_interactions = pd.read_csv(child_interactions_path, sep='\t', header=None, names=columns)

item_interactions = child_interactions.groupby('item_id').size().to_dict()
del child_interactions

print('Creating DataFrame')
df = pd.DataFrame.from_dict(item_interactions, orient='index', columns=['popularity']).reset_index()
df.rename(columns={'index': 'item_id'}, inplace=True)

df = df.fillna(0).astype({col: int for col in df.columns if col != 'item_id'})


popularity_cols = [col for col in df.columns if col != 'item_id']
for col in popularity_cols:
    norm_col = f'{col}_norm'
    max = df[col].max()
    df[norm_col] = df[col] / max if max > 0 else 0
    
df.sort_values(by='popularity', ascending=False, inplace=True)
print(df.head())

df.to_csv(save_path_child, sep='\t', index=False)