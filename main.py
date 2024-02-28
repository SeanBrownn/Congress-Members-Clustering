import math

from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.spatial.distance import hamming
from kmodes.kmodes import KModes
from scipy.cluster.hierarchy import linkage, cophenet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.metrics import jaccard_score



def plot_eigenvalues(pca):
    plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker='o',
             linestyle='-', color='b')
    plt.title('Eigenvalues of Each Principal Component')
    plt.xlabel('Principal Component #')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.show()

def plot_cumulative_variance(pca):
    explained_variance_ratios = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratios)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o',
             linestyle='-', color='b')
    plt.title('Cumulative Variance Explained by Top k Principal Components')
    plt.xlabel('Number of Principal Components (k)')
    plt.ylabel('Cumulative Variance Explained')
    plt.grid(True)
    plt.show()

def party_color():
    return ['red' if affiliation == 'Republican' else 'blue' for affiliation in
            party_affiliations['Party']]

# takes numpy array from transformation as parameter. i and j are the components we want to plot
# (e.g. 1st component, 2nd component)
def project_data(votes_pca, i, j):
    plt.scatter(
        votes_pca[:, i-1], # do -1 b/c it starts at index 0, while the smallest input would be 1
        votes_pca[:, j-1],
        c=party_color(),
        marker='o',
        edgecolors='k',
        s=50,
    )
    plt.xlabel('Principal Component ' + str(i))
    plt.ylabel('Principal Component ' + str(j))
    plt.title('Congress Members Plotted by Principal Components and Colored by Party Affiliation')
    plt.show()

def votes_clusters():
    hamming_distances = pairwise_distances(votes, metric='hamming')
    clustering = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='complete')
    return clustering.fit_predict(hamming_distances)

# also takes numpy array from transformation
def plot_hamming_clusters(votes_pca,clusters):
    plt.scatter(
        votes_pca[:, 0],
        votes_pca[:, 1],
        c=clusters,
        marker='o',
        edgecolors='k',
        s=50,
    )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Congress Members Plotted by Principal Components and Colored by Cluster')
    plt.show()

    return clusters

# uses votes_dataset instead of votes so that we can find cophenetic distance on votes or permuted_votes
def cophenetic_distances(votes_dataset):
    hamming_distances = pdist(votes_dataset, metric='hamming')
    linkage_matrix = linkage(hamming_distances, method='complete')
    c, coph_dists = cophenet(linkage_matrix, hamming_distances)
    return c

# plots histogram of permuted distances and returns p value of original_distance
def permuted_votes_cophenetic_distances(num_iterations, original_distance):
    distances=np.zeros(num_iterations)
    for i in range(num_iterations):
        column_names = votes.columns.tolist()
        np.random.shuffle(column_names)
        permuted_votes = votes.apply(np.random.permutation, axis=0)
        distances[i]=cophenetic_distances(permuted_votes)
    plt.hist(distances, bins=20, edgecolor='black')
    plt.xlabel('Cophenetic Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Histogram of CCC Scores on Permuted Votes Dataset')
    plt.show()

    return np.mean(distances >= original_distance)

# takes numpy array from transformation
def clusters_using_top_2_principal_components(votes_pca):
    top_2_principal_components = pd.DataFrame(votes_pca[:, :2], columns=['PC1', 'PC2'])
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(top_2_principal_components)
    return kmeans.labels_

def jaccard(cluster_assignments):
    party_labels, party_levels = pd.factorize(party_affiliations['Party'])  # converts affiliations to numerical labels
    party_labels = 1 - party_labels
    return jaccard_score(cluster_assignments, party_labels, average='binary')

votes=pd.read_csv('C:/Users/Sean Brown/Documents/College/Case/2023-24/CSDS 313/assignment 3 '
                      'task 1/votes.csv', header=None)
votes.columns=["Issue " + str(i+1) for i in range(votes.shape[1])]

party_affiliations=pd.read_csv('C:/Users/Sean Brown/Documents/College/Case/2023-24/CSDS 313/assignment '
                               '3 task 1/party_affiliations.csv', header=None)
party_affiliations.columns=["Party"]

pca = PCA()
pca.fit_transform(votes)

#plot_eigenvalues(pca)
#plot_cumulative_variance(pca)
#project_data(pca.fit_transform(votes), 1, 2)
#project_data(pca.fit_transform(votes), 1, 3)
#project_data(pca.fit_transform(votes), 2, 3)
#plot_hamming_clusters(pca.fit_transform(votes))

#print(permuted_votes_cophenetic_distances(1000, cophenetic_distances(votes)))
#plot_hamming_clusters(pca.fit_transform(votes), clusters_using_top_2_principal_components(pca.fit_transform(votes)))