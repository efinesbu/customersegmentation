'''

This script applied customer segmentation to a small mall data sample using the silhouette method and k-means

Uncomment section by section, and run them individually

'''
import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# IMPORT DATA
location = 'Mall_Customers.csv' # Identify data location
data = pd.read_csv(location)    # Read data into script

# # SEE PAIR PLOT
# sns.pairplot(data)
# plt.title('Pairplot for the Data', fontsize = 20)
# plt.show()

# # CALCULATE SILHOUETTE SCORE
# k_clusters = range(3,10)
# y = []
# for k in k_clusters:
#     kmeans = KMeans(n_clusters=k)
#     y_pred = kmeans.fit_predict(data)
#     score = silhouette_score(data, kmeans.labels_)
#     y.append(score)
#
# plt.plot(k_clusters, y)
# plt.title('Silhouette Scores')
# plt.xlabel('Number of clusters, k')
# plt.ylabel('Silhouette Score')
# plt.show()

# # GRAPHING SEGMENTS INCOME VS SPENDING
# kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
# x = data.iloc[:, [3, 4]].values
# y_means = kmeans.fit_predict(x)
#
# plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Careful')
# plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'blue', label = 'Standard')
# plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'green', label = 'Target')
# plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'orange', label = 'Careless')
# plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'red', label = 'Sensible')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='black', label='Centeroid')
#
# plt.style.use('fivethirtyeight')
# plt.title('Customer Segments | K Means Clustering', fontsize=20)
# plt.xlabel(" 'Annual Income 000's ")
# plt.ylabel('Spending Score')
# plt.legend()
# plt.grid()
# plt.show()

# # GRAPHING SEGEMENTS AGE VS SPENDING
# kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
# x = data.iloc[:, [2, 4]].values
# y_means = kmeans.fit_predict(x)

# plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Target Young')
# plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'green', label = 'Target')
# plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'blue', label = 'Usual')
# plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'orange', label = 'Target Old')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='black', label='Centeroid')

# plt.style.use('fivethirtyeight')
# plt.title('Customer Segments | K Means Clustering', fontsize=20)
# plt.xlabel('Age')
# plt.ylabel('Spending Score')
# plt.legend()
# plt.grid()
# plt.show()
