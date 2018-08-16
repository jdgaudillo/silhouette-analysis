import matplotlib as mpl
mpl.use('Tkagg')

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster.unsupervised import check_number_of_labels


def analysis(X):
    range_n_clusters = [3, 4, 5, 6]
    silhouette_avg=[]

    for n_clusters in range_n_clusters:
        
        clusterer = KMeans(n_clusters=n_clusters) 

        cluster_labels = clusterer.fit_predict(X)

        
       	print('Solving silhouette_score of cluster', n_clusters)
        silhouette_avg.append(silhouette_score(X, cluster_labels))
    np.savetxt('silhouette_score.csv', silhouette_avg, delimiter=',')

file = 'preprocessed_final.csv'
dcols = {'Province':np.str}
cols = ['CustomerID', 'AccountType', 'CustomerType', 'NoOfProducts', 
		'NumOfServicesAvailed', 'GC', 'CD', 'R', 'CS',
		'NumOfServicesAvailedOutofWarranty', 'AverageBillAmountOutOfWarranty',
		'AverageBillAmount', 'Class']
X = pd.read_csv(file, encoding="ISO-8859-1", dtype = dcols, usecols= cols,nrows=50000)
print('Read file!')
X = X.set_index('CustomerID')

X = X.values
analysis(X)