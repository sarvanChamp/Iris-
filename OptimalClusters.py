# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:08:29 2021

@author: ELCOT
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
sns.set()

dataset=pd.read_csv("D:\datasets\iris.csv")
d=dataset.head()
print(d)

plt.scatter(dataset['SepalLengthCm'],dataset['SepalWidthCm'])
plt.xlabel(['SEPAL LENGTH CM'])
plt.ylabel(['SEPAL WIDTH CM'])
plt.show

#Features Selcting

x=dataset.iloc[:,1:5]
print(x.head())

#Number of Cluster Selecting

wcs=[]
for i in range(1,11):
    kmeans=KMeans(i)
    kmeans.fit(x)
    wcs_iterator=kmeans.inertia_
    wcs.append(wcs_iterator)
    
print(wcs)

#Using elbow method to finding the appropriate number of cluster in a dataset

no_of_cluster=range(1,11)
plt.plot(no_of_cluster,wcs)
plt.title('USING ELBOW METHOD')
plt.xlabel('NUMBER OF CLUSTER')
plt.show

#SELECCTING 3 AS OPTIMAL NUMBER OF CLUSTERS

kmeans=KMeans(3)
kmeans.fit(x)

identified_clusters=kmeans.fit_predict(x)
identified_clusters
#Add identified clusters into the dataset

dataset_with_cluster=dataset.copy()
dataset_with_cluster['clusters']=identified_clusters

#Visualizing Clusters
plt.scatter(dataset_with_cluster['SepalLengthCm'],dataset_with_cluster['SepalWidthCm'],
            c=dataset_with_cluster['clusters'],cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',label='centroid')
plt.xlabel('Sepal Length CM')
plt.ylabel('Sepal Width CM')
plt.show

plt.scatter(dataset_with_cluster['PetalLengthCm'],dataset_with_cluster['PetalWidthCm'],
            c=dataset_with_cluster['clusters'],cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,2],kmeans.cluster_centers_[:,3],c='yellow',label='centroid')
plt.xlabel('Petal Length CM')
plt.ylabel('Petal Width CM')
plt.show
