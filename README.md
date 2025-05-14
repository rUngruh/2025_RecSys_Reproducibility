# Impacts of Mainstream-Driven Algorithms on Recommendations for Children Across Domains: A Reproducibility Study

This repository contains the code to reproduce the experiments from the paper: "Impacts of Mainstream-Driven Algorithms on Recommendations for Children Across Domains: A Reproducibility Study", submitted to RecSys 2025 Reproducibility track. 

As this work reproduces the work by [Ungruh et al.](https://link.springer.com/chapter/10.1007/978-3-031-88714-7_50), a lot of the code is based on the [repository of the reference work](https://github.com/rUngruh/PreferenceAnalysis).

Additional Results not included in the main manuscript can be found in `Results/Additional_Results.md`.


# Reproducibility Code

## Set up environment
```
conda env create  -n 2025_RecSys_repro -f environment.yml
conda activate 2025_RecSys_repro
pip install -e .
```

## Data gathering
Gather all publicly available datasets and add the data to a directory `data`: [MLHD+](https://musicbrainz.org/doc/MLHD+) and the respective demographical data from the original [MLHD](https://ddmal.music.mcgill.ca/research/The_Music_Listening_Histories_Dataset_(MLHD)/), [Book-Crossing](https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset), [Goodreads](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) subsets, [MovieLens-1m](https://grouplens.org/datasets/movielens/1m/). 

To ease reproducibility, we provide our user and item samples, so that the sampling and genre gathering steps for MLHD as well as genre gathering steps for Book-Crossing can be skipped. With this, our experiments can be reproduced easily with our used samples and the same data samples can be easily used for future studies. If these should be used, extract them from the `Samples` directory (or [download the MLHD samples](https://zenodo.org/records/15394228)) and place them at `processed/MLHD_processed` and `processed/Book-Crossing`.

The data directory should look like this:
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
│   ├── processed
│   │   ├── MLHD_processed
│   │   │   ├── artists.tsv
│   │   │   ├── users.tsv
│   │   ├── Book-Crossing
│   │   │   ├── books_w_genre.tsv
├── ...
```
Add the path of `data` to `config.env`.

## Data Preprocessing
Here, we provide the required steps to process the data used for our experiments. In case you want to use the provided data samples, pay attention to the additional _notes_.

Navigate to the Pre-processing directory: `cd Pre-Processing`

### MLHD+
- For initial insights regarding age distribution etc, run the script `MLHD_testing.ipynb`.
- For Genre gathering from the MusicBrainz API, set the environment variables for the MusicBrainz app name and email in the `config.env` file.
- Then, the following scripts are used to sample the data, gather genres from the MusicBrainz API, and create the files required for the experiments:

```
# Sampling
MLHD_sampling.py
MLHD_sample_LEs.py
MLHD_gather_artists.py

# Genre Annotation and Filtering
MLHD_MB_genre_annotation.py
MLHD_allmusic_matching.py

# Create files used for experiments
MLHD_filter_LEs_by_genre.py

# Create a simplified version with custom IDs instead of MusicBrainz identifiers
MLHD_simplify_IDs.py
```
> **_NOTE:_**  In case, the published samples are used, the first steps can be skipped. Read below for more information

#### Usage of MLHD user and artist samples
When using the published samples, add them to the data directory as explained before and run the following script:

```
python MLHD_filter_LEs_by_user_and_artist.py -start_fresh -sample_les -save_users -save_artists
```

Afterward, the first five steps of the preprocessing can be skipped. Resume further processing with `MLHD_filter_LEs_by_genre.py`.

In case, you don't want to use the `config.env` to set the dataset directory for these steps (e.g., if the data sample is used for another work), set the argument `--dataset_dir` for the scripts used to processes MLHD with the data sample.


### BookCrossing
We utilize the [Book-Data](https://github.com/PIReTship/bookdata-tools/tree/main) tool to extract isbn's that are related across the two datasets. 
For this, we follow the processing steps as laid out in the [docs](https://github.com/PIReTship/bookdata-tools/tree/main/docs) (unfortunately, at time of submission, the tool's main documentation is not available since it goes through changes. Thus, we refer to the docs in the GitHub repository.) Clone the repository, follow the steps explained in the README to install the dependencies, download the relevant datasets (Bookcrossing and Goodreads) and add them to their respecitive directories in the `data` directory. Then, update the `config.yaml` to only include the necessary datasets, and run the tool with `dvc repro`.

Extract the file `book-links/isbn-clusters.parquet` from the tool and place it in the `data` directory. After that run the following scripts to do the matching and prepare data for the experiments:

```
BX_create_goodreads_genre_dict.py
BX_Goodreads_Matching.py
BX_filter_ratings.py
```

> **_NOTE:_**  In case, the published genre samples are used, the first steps and usage of the bookdata tool can be skipped. Read below for more information
#### Usage of BX book samples
When using the published samples, the combination of BX with Goodreads is not necessary and the first steps of the processing can be skipped. 


The first twp steps of the preprocessing can be skipped. Resume further processing with `BX_filter_ratings.py`.

In case, you don't want to use the `config.env` to set the dataset directory for these steps (e.g., if the data sample is used for another work), set the argument `--dataset_dir` of `BX_filter_ratings.py`.


### MovieLens-1m
To prepare the data for the experiments, run:
```
ML_process_ratings.py
```

## Experiment 1: Preference Deviation Exploration
To analyze user profiles and preferences, the processed data needs to be processed (to compute genre distributions/interest in popularity etc.).

First, navigate to the directory for the first experiment: `cd Experiment_1`

### Popularity Extensions
To prepare information about popularity of songs analyzed for the extension of the first experiment, run the following scripts
```
python compute_item_popularity --dataset ml
python compute_item_popularity --dataset bx
python compute_item_popularity --dataset mlhd
```

### User Profile Analysis
```
python create_user_profiles_in_batches.py --dataset ml --weighted True
python create_user_profiles_in_batches.py --dataset bx --weighted True
python create_user_profiles_in_batches.py --dataset mlhd --weighted True
```

### Analysis
Afterward, preferences can be analyzed with the scripts: 
- `get_subset_stats.ipynb`: Profile sizes etc.
- `age_centered_analysis.ipynb`: Genre distributions and popularity of items in profiles

## Experiment 2: RS Experiment

### Preprocessing
To preprocess the sets for the RS experiments, run the following scripts:
```
cd Experiment_2/Preprocessing

# Book-Crossing
python split_set.py --dataset bx --k_core_filtering_user 5 --k_core_filtering_item 5 --train_split_size 0.6 --validation_split_size 0.2 -binarize

# Movielens
python split_set.py --dataset ml  --k_core_filtering_user 10 --k_core_filtering_item 10 --train_split_size 0.6 --validation_split_size 0.2 --min_rating 4 -binarize

# MLHD
python filter_year.py --chunksize 10000000 --start_date 2009-06-01 --end_date 2009-10-30

python split_set.py --dataset mlhd --year 2009 -remove_missing_profiles --k_core_filtering_user 5 --k_core_filtering_item 10 --validation_start 2009-09-01 --test_start 2009-10-01 --min_playcount 2 -binarize -sample_users --sample_size 13000
```

The filtered and split sets can be analyzed with `compare_user_stats_to_filtered_set.ipynb` and `analyze_experimental_set.ipynb` to test their viability and the split sizes.

### Install Elliot
```
cd 2025_RecSys_Reproducibility

conda create --yes --name elliot-env python=3.8
conda activate elliot-env
git clone https://github.com//sisinflab/elliot.git && cd elliot
pip install --upgrade pip
pip install -e . --verbose

# The following steps were accomplished to align the environment with our requirements.
pip install protobuf==3.20.3
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install cudnn=7.6.5
```

### Prepare Data
- Add processed dataset directories from the `data/processed` directory to `elliot/data`, i.e. `elliot/data/bx_rec_filtered`, `elliot/data/ml_rec_filtered`, and `elliot/data/mlhd_rec_filtered`
- Add config files in `Experiment_2/configs` to `elliot/config_files`

### Start Experiments
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
conda activate 2025_RecSys_repro 
cd Experiment_2/Preprocessing

python compute_item_popularity.py --dataset ml
python compute_item_popularity.py --dataset bx
python compute_item_popularity.py --dataset mlhd
```

- For simplicity, add the identifying names of the best models in the `utils/best_models.json`.

To compute genre distributions/popularity of profiles, run:
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

## Gather Results
- To gather results, adapt the second block of the following Jupyter Notebooks and run the tests/experiments.
    - `Performance_Analysis.ipynb`
    - `Miscalibration_Analysis.ipynb`
    - `Mainstream_Child_AGP_Deviations.ipynb`
    - `Plot_AGPs.ipynb`
    - `Popularity_Analysis.ipynb`