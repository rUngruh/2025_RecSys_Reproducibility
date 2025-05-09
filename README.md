#

As this work replicates the work by (Ungruh et al.)[], a lot of the code is based on their Code presented in their (repository)[]. 

## Set up environment
```
conda install -n 2025_RecSys_repro PYTHON=3.12
conda activate 2025_RecSys_repro
pip install -e .
```




Add all necessary data to a directory `data`: [MLHD+](https://musicbrainz.org/doc/MLHD+) and the respective demographical data from the original [MLHD](https://ddmal.music.mcgill.ca/research/The_Music_Listening_Histories_Dataset_(MLHD)/), [Book-Crossing](https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset), [Goodreads](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) subsets, [ml-1m](https://grouplens.org/datasets/movielens/1m/)
```
├── data
│   ├── raw
│   │   ├── MLHD+
│   │   │   ├── MLHD_demographics
│   │   │   ├── mlhdplus-complete-0.tar
│   │   │   ├── mlhdplus-complete-1.tar
│   │   │   ├── ...
│   │   │   ├── mlhdplus-complete-f.tar
│   │   ├── Book-Crossing
│   │   │   ├── BX-Book-Ratings
│   │   │   ├── BX-Books
│   │   │   ├── BX-Users
│   │   ├── Goodreads
│   │   │   ├── goodreads_books_children.json.gz
│   │   │   ├── goodreads_books_comics_graphic.json.gz
│   │   │   ├── ...
│   │   │   ├── goodreads_books_young_adult.json.gz
│   │   ├── ml-1m
│   │   │   ├── movies.dat
│   │   │   ├── ratings.dat
│   │   │   ├── users.dat
```
Add the path of `data` to `config.env`.


## Data Preprocessing
```
cd Pre-Processing
```

### MLHD+
- For initial insights regarding age distribution etc, run the script `MLHD_testing.ipynb`
- Run `MLHD_sampling.py`
- Run `MLHD_sample_LEs.py`
- Run `MLHD_gather_artists.py`
- Run `MLHD_MB_genre_annotation.py`
- Run `MLHD_allmusic_matching.py`
- Run `MLHD_filter_LEs_by_genre.py`; This creates the sets used for the experiments.
- Run `MLHD_simplify_IDs`; This generates a simplified version with IDs instead of MBIDs for easier processing



### BookCrossing
We utilize the [Book-Data](https://bookdata.piret.info/) tool to extract isbn's that are related. For this, we install the tool, and extract the file `book-links/isbn-clusters.parquet` and place it in the `data` directory. After that run the following scripts:

- Run `BX_create_goodreads_genre_dict.py`
- Run `BX_Goodreads_Matching.py`
- Run `BX_filter_ratings.py`

### MovieLens-1m
- Run `ML_process_ratings.py`

## Experiment 1
```
cd Experiment_1
```

### Popularity Extensions
To prepare information about popularity of songs analyzed for the extension of the first experiment, run the following scripts
```
python compute_item_popularity --dataset ml
python compute_item_popularity --dataset bx
python compute_item_popularity --dataset mlhd
```


### Book Crossing
```
python create_user_profiles_in_batches.py --dataset ml --weighted True
python create_user_profiles_in_batches.py --dataset bx --weighted True
python create_user_profiles_in_batches.py --dataset mlhd --weighted True
```


## Experiment 2

### Book Crossing

```
python split_set.py --dataset bx --k_core_filtering_user 5 --k_core_filtering_item 5 --train_split_size 0.6 --validation_split_size 0.2 -binarize
python split_set.py --dataset ml  --k_core_filtering_user 10 --k_core_filtering_item 10 --train_split_size 0.6 --validation_split_size 0.2 --min_rating 4 -binarize


python filter_year.py --chunksize 10000000 --start_date 2009-06-01 --end_date 2009-10-30
python split_set.py --dataset mlhd --year 2009 -remove_missing_profiles --k_core_filtering_user 5 --k_core_filtering_item 10 --validation_start 2009-09-01 --test_start 2009-10-01 --min_playcount 2 -binarize -sample_users --sample_size 13000
```




### Hyperparameter Tuning
- Install Elliot
```
conda create --yes --name elliot-env python=3.8
conda activate elliot-env
git clone https://github.com//sisinflab/elliot.git && cd elliot
pip install --upgrade pip
pip install -e . --verbose
pip install protobuf==3.20.3
```

- Add processed dataset directories to `elliot/data`, i.e. `elliot/data/bx_rec_filtered`, `elliot/data/ml_rec_filtered`, and `elliot/data/mlhd_rec_filtered`
- Add config files in `Experiment_2/configs` to `elliot/config_files`

- Run Elliot
```
python start_experiments.py --config all_user_config_bx
python start_experiments.py --config all_user_config_ml
python start_experiments.py --config all_user_config_mlhd
python start_experiments.py --config child_config_bx
python start_experiments.py --config child_config_ml
python start_experiments.py --config child_config_mlhd
```

## Post-processing
```
cd Experiment_2/Preprocessing
python compute_item_popularity.py --dataset ml
python compute_item_popularity.py --dataset bx
python compute_item_popularity.py --dataset mlhd
```

- For simplicity, add the identifying names of the best models in the `utils/best_models.json`.

```
cd Experiment_2/Result_Analysis
Process_Results.py --dataset ml --age_type defined_ages -filtered --models "Random" "MostPop" "RP3beta" "iALS"
--child_models "Random" "MostPop" "RP3beta" "iALS"

cd Experiment_2/Result_Analysis
Process_Results.py --dataset bx --age_type defined_ages -filtered 
--models "Random" "MostPop" "RP3beta" "iALS"
--child_models "Random" "MostPop" "RP3beta" "iALS"

cd Experiment_2/Result_Analysis
Process_Results.py --dataset mlhd --age_type defined_ages -filtered 
--models "Random" "MostPop" "RP3beta" "iALS"
--child_models "Random" "MostPop" "RP3beta" "iALS"
```