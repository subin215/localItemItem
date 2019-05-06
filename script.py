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

beta_g = 5
beta_l = 5


def get_pu(mat_to_cluster, num_clusters = 5):
    k_means = KMeans(n_clusters=num_clusters)
    k_means.fit(mat_to_cluster)
    return [user_index[k_means.labels_ == i] for i in np.unique(k_means.labels_)]
pu = get_pu(norm_mat, 5)

def comp_gu(u, cluster):
  for i_idx, i in enumerate(item_index.values):
    i = i_idx  # Convert to index from item_id
    sum_l, sum_g = sum_all(u, i, cluster)
    r_r = R[u,i] - sum_l
    r_l = sum_g - sum_l
    result_bottom = pow((sum_g - sum_l), 2)
    result_top = r_l * r_r
  g[u] = float(result_top)/float(result_bottom)
  g_prime[u] = float(1 - g[u])
  

def sum_all(u, i, cluster):
  temp_sum_1 = 0
  temp_sum_2 = 0
  for l in R[:, u]:
    temp_sum_2 = temp_sum_2 + S.A[l,i]
    temp_sum_1 = temp_sum_1 + S_pu[cluster].A[l,i]
  return temp_sum_1, temp_sum_2

def comp_gu(u, cluster):
  for i_idx, i in enumerate(item_index.values):
    i = i_idx  # Convert to index from item_id
    sum_l, sum_g = sum_all(u, i, cluster)
    r_r = R[u,i] - sum_l
    r_l = sum_g - sum_l
    result_bottom = pow((sum_g - sum_l), 2)
    result_top = r_l * r_r
  g[u] = float(result_top)/float(result_bottom)
  g_prime[u] = float(1 - g[u])
  

def sum_all(u, i, cluster):
  temp_sum_1 = 0
  temp_sum_2 = 0
  for l in R[:, u]:
    temp_sum_2 = temp_sum_2 + S.A[l,i]
    temp_sum_1 = temp_sum_1 + S_pu[cluster].A[l,i]
  return temp_sum_1, temp_sum_2

users_switching_cluster = 0
idx = 0
while True:
  idx += 1
  compute_similarity()
  for u_idx, u in tqdm(enumerate(user_index.values)):
    user_cluster = 0
    err = np.array(pu.shape[0])
    for cluster in range(pu.shape[0]):
      if u in pu[cluster]:
        user_cluster = cluster
        u = u_idx # Convert to index from user_id
        comp_gu(u, cluster)
        # Compute training error
        pred = get_rating_for_user(u, cluster)
        err[cluster] = rmse(R[u, :], pred)
    #END CLUSTER FOR LOOP
    new_cluster_for_u = np.where(err == np.min(err))
    if new_cluster_for_u != user_cluster:
      np.delete(pu[user_cluster], user_index.values[u])
      np.append(pu[new_cluster_for_u], user_index.values[u])
      users_switching_cluster += 1
      comp_gu(u, new_cluster_for_u)
  if users_switching_cluser < (0.10*user_index.values.shape[0]):
    break