import pandas as pd
import numpy as np
import os
from urllib.request import urlretrieve
from zipfile import ZipFile

# Constants
ML_100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_100K_FILENAME = ML_100K_URL.rsplit('/', 1)[1]
ML_100K_FOLDER = 'ml-100k'

# Download the dataset if not already downloaded
if not os.path.exists(ML_100K_FILENAME):
    print('Downloading %s to %s...' % (ML_100K_URL, ML_100K_FILENAME))
    urlretrieve(ML_100K_URL, ML_100K_FILENAME)

# Extract the dataset if not already extracted
if not os.path.exists(ML_100K_FOLDER):
    print('Extracting %s to %s...' % (ML_100K_FILENAME, ML_100K_FOLDER))
    with ZipFile(ML_100K_FILENAME, 'r') as zip_ref:
        zip_ref.extractall('.')

# Load the dataset
df = pd.read_csv(os.path.join(ML_100K_FOLDER, 'u.data'), sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Load the movies data
movies_df = pd.read_csv(os.path.join(ML_100K_FOLDER, 'u.item'), sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movie_id', 'title'])

# Merge the ratings dataframe with the movies dataframe to get the movie titles
df_merged = pd.merge(df, movies_df, left_on='item_id', right_on='movie_id')

# Handle duplicate entries by calculating the mean rating for each user-movie pair
df_aggregated = df_merged.groupby(['user_id', 'title']).agg({'rating': 'mean'}).reset_index()

# Create the user-item matrix with movie titles instead of IDs
user_item_matrix_named = df_aggregated.pivot(index='user_id', columns='title', values='rating')

# Display the first 5 rows of the updated user-item matrix
print(user_item_matrix_named.head())

# Save the updated user-item matrix to a CSV file
user_item_matrix_named.to_csv('user_item_matrix_with_names.csv')
