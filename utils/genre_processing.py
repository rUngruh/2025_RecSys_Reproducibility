ML_genres = []
MLHD_genres = []
BX_genres = []

import os
this_file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_file_path)

with open(f'{this_file_path}/ML_genres.txt', 'r') as f:
    for line in f:
        ML_genres.append(line.strip())


with open(f'{this_file_path}/MLHD_genres.txt', 'r') as f:
    for line in f:
        MLHD_genres.append(line.strip())


with open(f'{this_file_path}/BX_genres.txt', 'r') as f:
    for line in f:
        BX_genres.append(line.strip())
       
def get_genres(dataset):
    if dataset == 'ml':
        return ML_genres
    elif dataset == 'mlhd':
        return MLHD_genres
    elif dataset == 'bx':
        return BX_genres
    else:
        raise ValueError("Invalid dataset. Choose from 'ml', 'mlhd', or 'bx'.") 
        
def genre_dict_to_list(genre_dict, dataset):
    if dataset == 'ml':
        return [genre_dict.get(genre, 0) for genre in ML_genres]
    elif dataset == 'mlhd':
        return [genre_dict.get(genre, 0) for genre in MLHD_genres]
    elif dataset == 'bx':
        return [genre_dict.get(genre, 0) for genre in BX_genres]
    else:
        return None
    
def genre_list_to_dict(genre_list, dataset):
    if dataset == 'ml':
        return {genre: genre_list[i] for i, genre in enumerate(ML_genres) if genre_list[i] != 0}
    elif dataset == 'mlhd':
        return {genre: genre_list[i] for i, genre in enumerate(MLHD_genres) if genre_list[i] != 0}
    elif dataset == 'bx':
        return {genre: genre_list[i] for i, genre in enumerate(BX_genres) if genre_list[i] != 0}
    else:
        return None