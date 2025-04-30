import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from pathlib import Path
env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")


sample_directory = dataset_dir + '/processed/MLHD_sampled'


#listening_events_path = os.path.join(sample_directory, 'listening_events.tsv.bz2')

artists_MB_path = os.path.join(sample_directory, 'artist_MB_genres.tsv')
artists_genres_path = os.path.join(sample_directory, 'artists_genres.tsv')
artists_allmusic_path = os.path.join(sample_directory, 'artists_AM_genres.tsv')


artists = pd.read_csv(artists_MB_path, sep='\t')

am_genres = []
with open('AM_genres.txt', 'r') as f:
    for line in f:
        am_genres.append(line.strip())
        

with open('../utils/MLHD_genres.txt', 'w') as f:
    for genre in am_genres:
        f.write(genre + '\n')

def match_allmusic_genres(genres):
    if genres is None or genres == '' or genres is np.nan:
        return ''
    genres = genres.replace("&", "n")
    genres = genres.split(",")
    matched_genres = []
    for genre in genres:
        for am_genre in am_genres:
            if genre.lower() in am_genre.lower() or am_genre.lower() in genre.lower():
                matched_genres.append(am_genre)
    return ",".join(set(matched_genres))



artists['genres'] = artists['genres'].apply(lambda x: x.replace(", ", ",") if x is not np.nan else x)
print(f"Matching {len(artists)} artists with AllMusic genres...")
artists['am_genres'] = artists['genres'].apply(match_allmusic_genres)
print(f"matched; now saving...")
artists.to_csv(artists_genres_path, sep="\t", index=False)

artists_allmusic = artists[(artists['am_genres'] != '')] 

artists_allmusic = artists_allmusic[['artist_id', 'am_genres']]
artists_allmusic.to_csv(artists_allmusic_path, sep="\t", index=False)