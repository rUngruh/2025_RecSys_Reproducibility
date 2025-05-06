import pandas as pd
import os
import json

import os
import pandas as pd
import numpy as np
import argparse

from dotenv import load_dotenv
from pathlib import Path
env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")

bx_directory = dataset_dir + '/raw/Book-Crossing'
processed_directory = dataset_dir + '/processed/Book-Crossing'

books_save_path = os.path.join(processed_directory, 'books_w_genre.tsv')

print("Loading books...")
books = pd.read_csv(bx_directory + '/BX-Books.csv', sep=';', encoding='latin-1')[['ISBN']]

book_genre_dict = {}

print("Loading book genre dictionary...")

with open(processed_directory + '/goodreads_book_genre_dict.json', 'r') as f:
    book_genre_dict = json.load(f)
print(f"Loaded {len(book_genre_dict)} books with genres")

cluster_path = bx_directory + '/isbn-clusters.parquet'
clusters = pd.read_parquet(cluster_path)
isbn_cluster_dict = clusters[['isbn', 'cluster']].set_index('isbn').to_dict()['cluster']

books['cluster'] = books['ISBN'].map(isbn_cluster_dict)

cluster_genre_dict = {isbn_cluster_dict[isbn] : genre for isbn, genre in book_genre_dict.items() if isbn in isbn_cluster_dict.keys()}

books['genres'] = books['cluster'].map(cluster_genre_dict)
print(books['genres'])
genres = set()
for genre_list in books['genres']:
    if isinstance(genre_list, list):
        genres.update(genre_list)

with open('../utils/BX_genres.txt', 'w') as f:
    for genre in genres:
        f.write(genre + '\n')

print(f'Number of books in Bookcrossing: {len(books)}')
books = books[books['genres'].notna()]
print(f'Number of books in filtered Bookcrossing: {len(books)}')

books['genres'] = books['genres'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
books = books[['ISBN', 'genres']]
books.rename(columns={'ISBN': 'item_id'}, inplace=True)

books.to_csv(books_save_path, sep='\t', index=False, header=True)
print(f"Saved books to {books_save_path}")