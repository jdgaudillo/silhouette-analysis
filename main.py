import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering

from silhoutte import analysis
import numpy as np
import random
random.seed(32)



file = 'preprocessed_final.csv'
dcols = {'Province':np.str}
cols = ['CustomerID', 'AccountType', 'CustomerType', 'NoOfProducts', 
		'NumOfServicesAvailed', 'GC', 'CD', 'R', 'CS',
		'NumOfServicesAvailedOutofWarranty', 'AverageBillAmountOutOfWarranty',
		'AverageBillAmount', 'Class']
X = pd.read_csv(file, encoding="ISO-8859-1", dtype = dcols, usecols= cols)
X = X.set_index('CustomerID')

X = X.values

'''np.random.seed(4711)  # for repeatability of this tutorial
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
X = np.concatenate((a, b),)'''

algo = ['kmeans']
analysis(X, algo)


#analysis(X,algo)

'''for i in algo:
	analysis(X, i)
	optimal_clusters = input("Enter optimal clusters: ")
	if i == 'kmeans':
		clusterer = KMeans(n_clusters = optimal_clusters, random_state = 42)
	else:
		clusterer = AgglomerativeClustering(n_clusters = optimal_clusters, linkage = 'ward')

	cluster_labels = clusterer.fit_predict(X)
	X_new = X
	X_new['Cluster'] = pd.Series(cluster_labels, index = X_new.index)
	#enter choice of filename labeling you want
	filename = 
	X_new.to_csv(filename, sep=',', encoding='utf-8', index='True')'''