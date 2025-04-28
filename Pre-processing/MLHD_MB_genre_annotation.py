import pandas as pd
import requests
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")
app_name = os.getenv("MusicBrainz_app_name")
app_email = os.getenv("MusicBrainz_app_email")
mlhd_directory = dataset_dir + '/processed/MLHD_sampled'
save_directory = dataset_dir + '/processed/MLHD_sampled'

INPUT_TSV = mlhd_directory + "/artists.tsv"
OUTPUT_TSV = mlhd_directory + "/artist_MB_genres.tsv"
BATCH_SIZE = 10
SLEEP_TIME = 1  # seconds between requests



def fetch_genre(artist_id):
    url = f"https://musicbrainz.org/ws/2/artist/{artist_id}?inc=genres&fmt=json"
    headers = {"User-Agent": f"{app_name} ({app_email})"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            genres = [g["name"] for g in data.get("genres", [])]
            return ", ".join(genres)
        else:
            print(f"[{response.status_code}] Failed to fetch {artist_id}")
            return None
    except Exception as e:
        print(f"[ERROR] {artist_id}: {e}")
        return None

def run_batch_fetch(input_path, output_path):
    df = pd.read_csv(input_path, sep="\t", names=["artist_id"])
    artist_ids = df["artist_id"].dropna().astype(str).unique()

    try:
        existing = pd.read_csv(output_path, sep="\t")
        processed_ids = set(existing["artist_id"].astype(str))
        print(f"Resuming: {len(processed_ids)} out of {len(artist_ids)} artists already processed.")
    except FileNotFoundError:
        existing = pd.DataFrame(columns=["artist_id", "genres"])
        processed_ids = set()

    results = []
    total_written = len(processed_ids)

    for artist_id in tqdm(artist_ids):
        if artist_id in processed_ids:
            continue

        genres = fetch_genre(artist_id)
        if genres is not None:
            results.append({"artist_id": artist_id, "genres": genres})
        time.sleep(SLEEP_TIME)

        # Save every BATCH_SIZE
        if len(results) >= BATCH_SIZE:
            pd.DataFrame(results).to_csv(output_path, sep="\t", index=False, mode='a', header=not total_written)
            total_written += len(results)
            results = []
    # Save any remaining results
    if results:
        pd.DataFrame(results).to_csv(output_path, sep="\t", index=False, mode='a', header=not total_written)

    print("Processed")

if __name__ == "__main__":
    run_batch_fetch(INPUT_TSV, OUTPUT_TSV)
