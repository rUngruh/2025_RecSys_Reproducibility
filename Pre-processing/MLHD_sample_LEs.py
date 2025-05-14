######################################################################

# This scripts takes the sampled users from the MLHD_sampling.py script and extracts their listening events from the MLHD dataset.

######################################################################


import os
import pandas as pd
import tarfile
from dotenv import load_dotenv
from pathlib import Path
import zstandard as zstd
import io


env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")

mlhd_directory = dataset_dir + '/raw/MLHD+'
sample_directory = dataset_dir + '/processed/MLHD_sampled'


savestate_path = os.path.join(sample_directory, 'processing_savestate.txt')

sampled_users = set(pd.read_csv(os.path.join(sample_directory, 'users.tsv'), sep="\t")['user_id'].unique())


mlhd_datasets = [os.path.join(mlhd_directory, f'mlhdplus-complete-{i}.tar') for i in list(range(0, 10)) + list(map(chr, range(ord('a'), ord('f')+1)))]
save_paths = [os.path.join(sample_directory, f'listening_events-{i}.tsv.bz2') for i in list(range(0, 10)) + list(map(chr, range(ord('a'), ord('f')+1)))]


found_users = 0

processed = set()

if os.path.exists(savestate_path):
    with open(savestate_path, 'r') as f:
        processed = set(line.strip() for line in f if line.strip())

def process_tar_file(tar_path, save_path):
    tar_name = os.path.basename(tar_path)
    global found_users
    
    print(f"Processing {tar_name}...")

    with tarfile.open(tar_path, 'r') as tar:
        write_lines = []
        savestate_members = []

        
        for member in tar:
            
            if member.name in processed:
                continue
            
            if not member.name.endswith('.txt.zst'):
                continue
            
            savestate_members.append(member)
            user_id = member.name.split('.')[0].split('/')[-1]
            
            if user_id in sampled_users:
                #print(f"  -> {user_id}")
                found_users += 1
                if found_users % 100 == 0:
                    print(f"    {found_users} users out of {len(sampled_users)} found, i.e. {found_users/len(sampled_users)*100:.2f}%")
                f = tar.extractfile(member)
                if f is None:
                    print(f"    Error extracting {member.name}")
                    continue

                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    for i, line in enumerate(text_stream):   
                        item = [user_id] + line.strip().split('\t')                   
                        write_lines.append(item[:3] + item[4:]) # Saved columns are: user_id, timestamp, artist_id, item_id

            
            if len(savestate_members) >= 10000:
                print(f'Saving {len(savestate_members)} members to savestate...')
                
                with open(savestate_path, 'a') as log:
                    for member in savestate_members:
                        log.write(member.name + '\n')
                savestate_members = []

                print(f'Saving {len(write_lines)} lines to {save_path}...')
                if write_lines:
                    save_batch(write_lines, save_path, compressed=True)
                    write_lines = []
                

        print(f'Saving {len(savestate_members)} members to savestate...')
        if savestate_members:
            with open(savestate_path, 'a') as log:
                for member in savestate_members:
                    log.write(member.name + '\n')
                    
        print(f'Saving {len(write_lines)} lines to {save_path}...')
        if write_lines:
            save_batch(write_lines, save_path, compressed=True)
            write_lines = []
        savestate_members = []
                
def save_batch(lines, save_path, compressed=True):
    # mode = 'at' if compressed else 'a'
    # open_func = bz2.open if compressed else open
    
    
    lines_df = pd.DataFrame(lines)
    lines_df.to_csv(save_path, mode='a', sep='\t', index=False, header=False, compression='bz2' if compressed else None)

    # with open_func(listening_events_path, mode, encoding='utf-8') as out_file:
    #     for line in lines:
    #         out_file.write(line + '\n')
            
    print('Saved batch.')

def main():
    print("Starting batch processing")
    for tar_file, save_file in zip(mlhd_datasets, save_paths):
        process_tar_file(tar_file, save_file)
        print(f"Finished processing {tar_file}.")
    print("Processed all datasets.")

    
if __name__ == "__main__":
    main()
