######################################################################

# This script creates a mapping of book ISBNs to their genres based on the Goodreads dataset.
# The mapping is saved as a JSON file for further processing.

######################################################################

import os
import gzip
import json

from dotenv import load_dotenv
from pathlib import Path
env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")

goodreads_directory = dataset_dir + '/raw/Goodreads'
processed_directory = dataset_dir + '/processed/Book-Crossing'



if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)
    
save_path = os.path.join(processed_directory, 'goodreads_book_genre_dict.json')
if os.path.exists(save_path):
    os.remove(save_path)


files = [f for f in os.listdir(goodreads_directory) if f.startswith("goodreads_books")]

files = [f for f in files if not (f.startswith("goodreads_book_genres") or f.startswith("goodreads_books."))]

book_genre_dict = {}

books_without_isbn = 0
for file in files:
    genre = file.split(".")[0].split("goodreads_books_")[-1]
    print(f"Processing file: {file} with genre: {genre}")
    with gzip.open(os.path.join(goodreads_directory, file), 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            isbn = data.get("isbn")
            if isbn and isbn != '':
                book_genre_dict[isbn] = book_genre_dict.get(isbn, []) + [genre]
            else:
                books_without_isbn += 1
                
print(f"Total books processed: {len(book_genre_dict)}")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(book_genre_dict, f, ensure_ascii=False, indent=4)
print(f"Saved book genre dictionary to {save_path}")
print(f"Total books without ISBN: {books_without_isbn}")