import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta
from pathlib import Path

env_path = Path('..') / 'config.env'
load_dotenv(dotenv_path=env_path)
dataset_dir = os.getenv("dataset_directory")
mlhd_directory = dataset_dir + '/raw/MLHD+'
save_directory = dataset_dir + '/processed/MLHD_sampled'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

num_sampled_users = 45000 # Number of users in the sampled set
base_date = pd.to_datetime('2009-01-01')
end_date = pd.to_datetime('2013-12-31')
data_collection_date = pd.to_datetime('2014-01-01') # This date is the day we assume each user turned the reported age


users = pd.read_csv(mlhd_directory + '/MLHD_demographics.csv', sep='\t')
# users.columns = 'uuid', 'age', 'country', 'gender', 'playcount', 'age_scrobbles', 'user_type', 'registered', 'firstscrobble', 'lastscrobble

users['age_at_signup'] = users['age'] - (data_collection_date - pd.to_datetime(users['registered'], unit='s')).dt.days // 365.25
users['age_at_base'] = users['age'] - (data_collection_date.year - base_date.year) # This only works if both dates are at the same day (different years)

users = users[(pd.to_datetime(users['firstscrobble'], unit='s') < (base_date + relativedelta(years=1))) &
                       (pd.to_datetime(users['lastscrobble'], unit='s') > (end_date - relativedelta(years=1))) & 
                       (users['age_at_base'] > 11) &
                       (users['age'] < 65)].copy()

listening_count_mean = users['playcount'].mean()
listening_count_std = users['playcount'].std()


users = users[(users['playcount'] > (listening_count_mean - 2 * listening_count_std)) &
              (users['playcount'] < (listening_count_mean + 2 * listening_count_std))]

sampled_users = users.groupby('age', group_keys=False).apply(lambda x: x.sample(frac=num_sampled_users/len(users), random_state=42))
sampled_users = sampled_users[['uuid', 'age', 'country', 'gender']]

sampled_users.rename(columns={'uuid':'user_id'}, inplace=True)

print('sampled users shape:', sampled_users.shape)
sampled_users.to_csv(save_directory + '/users.tsv', sep='\t', index=False)

print('Saved sampled users to:', save_directory + '/users.tsv')