import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sps
import seaborn as sns
from sklearn.cluster import KMeans

ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
movies = pd.read_csv('data/ml-latest-small/movies.csv')
movie_titles = movies.set_index('movieId')['title']
item_ratings = ratings.set_index(['movieId','userId'])

item_means = ratings.groupby('item').rating.mean()
item_means.name = 'item_mean'
norm_ratings = ratings.join(item_means, on='item')
norm_ratings['nrating'] = norm_ratings['rating'] - norm_ratings['item_mean']

item_index = item_means.index
user_index = pd.Index(norm_ratings['user'].unique())

norm_mat = sps.csr_matrix((norm_ratings['rating'].values,
	(user_index.get_indexer(norm_ratings['user']), item_index.get_indexer(norm_ratings['item']))))

g = np.full(user_index.shape, 0.5)
g_prime = 1-g

def get_pu(mat_to_cluster, num_clusters = 5):
    k_means = KMeans(n_clusters=num_clusters)
    k_means.fit(mat_to_cluster)
    return [user_index[k_means.labels_ == i] for i in np.unique(k_means.labels_)]
pu = get_pu(norm_mat, 5)

