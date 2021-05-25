# import necessary libraries
from pathlib import Path
from PIL import Image
import os, shutil
from os import listdir


# image resizing
from PIL import Image
import numpy as np


# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot


# Mac OS created a hidden file called .DS_Store which interfered with my data processing. I had to delete this hidden file.
for root, dirs, files in os.walk('/Users/jinchoi725/Desktop/fashionimages'):
    i = 0
    for file in files:
        if file.endswith('.DS_Store'):
            path = os.path.join(root, file)
            print("Deleting: %s" % (path))
            if os.remove(path):
                print("Unable to delete!")
            else:
                print("Deleted...")
                i += 1
print("Files Deleted: %d" % (i))


# import in a list format the fashion image RGB files
Input_dir = '/Users/jinchoi725/Desktop/fashionimages/'
Out_dir = '/Users/jinchoi725/Desktop/gr_fashionimages/'
a = os.listdir('/Users/jinchoi725/Desktop/fashionimages/')


# convert all the RGB files into grayscale files. In cases of errors, skip them.
for i in a:
    try:
        print(i)
        I = Image.open('/Users/jinchoi725/Desktop/fashionimages/'+i)
        L = I.convert('L')
        L.save('/Users/jinchoi725/Desktop/gr_fashionimages/'+i)
    except:
        pass

    
# Mac OS created a hidden file called .DS_Store which interfered with my data processing. I had to delete this hidden file, AGAIN.
for root, dirs, files in os.walk('/Users/jinchoi725/Desktop/gr_fashionimages'):
    i = 0
    for file in files:
        if file.endswith('.DS_Store'):
            path = os.path.join(root, file)

            print("Deleting: %s" % (path))

            if os.remove(path):
                print("Unable to delete!")
            else:
                print("Deleted...")
                i += 1
print("Files Deleted: %d" % (i))


# The grayscaled images have to be resized to the same dimension for K-Means clustering.
dim = (80, 60) # This is the dimension that I chose.
X_image_train = []
for fname in listdir(Out_dir):
    fpath = os.path.join(Out_dir, fname)
    im = Image.open(fpath)
    im_resized = im.resize(dim) # This is where the image files are resized
    X_image_train.append(im_resized) # The resized image files are appended to a list object X_image_train.
   

# Using the numpy library, I converted each image file to a 2 dimentional numpy array. 
X_image_array=[]
for x in range(len(X_image_train)):
    X_image=np.array(X_image_train[x],dtype='uint8')
    X_image_array.append(X_image) # The numpy arrays are appended to a list object X_image_array.


# Checking the size of a single numpy array
X_image_array[0].shape
X_image_array[15].shape


# Using np.stack, I stacked the numpy arrays along a new axis. 
# While X_image_array was simply a squence of 2D numpy arrays, the stacked one is one 3D numpy array with three axes.
X_final = np.stack(X_image_array)
X_final


# As a result the shape of X_final is now (44441, 60, 80) with one more value which indicates the added axis (Z).
X_final.shape #(44441, 60, 80)


# A 2D numpy array in the 3D stacked numply array can still be displayed as a visual image.
pyplot.imshow(X_final[0])
pyplot.show()


# Now we are ready to perform K-Means clustering. Renamed X_final to X_train
X_train = X_final

# But before performing K-Means clustering, the images are too large in their dimensions and need to be shrinked in the dimensions
# Import necessary libraries for K-Means clustering
import keras
from keras.datasets import fashion_mnist 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Reshapeing X to a 2D array for PCA and then k-means
X = X_train.reshape(-1,X_train.shape[1]*X_train.shape[2]) #We will only be using X for clustering
X.shape


#Visualise an image 
n= 2
plt.imshow(X[n].reshape(X_train.shape[1], X_train.shape[2]), cmap = plt.cm.binary)
plt.show()


# To perform PCA we must first change the mean to 0 and variance to 1 for X using StandardScalar
Clus_dataSet = StandardScaler().fit_transform(X) #(mean = 0 and variance = 1)


# Make an instance of the Model
variance = 0.98 #The higher the explained variance the more accurate the model will remain
pca = PCA(variance)


#fit the data according to our PCA instance
pca.fit(Clus_dataSet)


print("Number of components before PCA  = " + str(X.shape[1]))
print("Number of components after PCA 0.98 = " + str(pca.n_components_)) # Components reduced from 4800 to 773


#Transform our data according to our PCA instance
Clus_dataSet = pca.transform(Clus_dataSet)
print("Dimension of our data after PCA  = " + str(Clus_dataSet.shape)) # Dimension of our data after PCA  = (44441, 773)


#To visualise the data inversed from PCA
approximation = pca.inverse_transform(Clus_dataSet)
print("Dimension of our data after inverse transforming the PCA  = " + str(approximation.shape))
# Dimension of our data after inverse transforming the PCA  = (44441, 4800)


#image reconstruction using the less dimensioned data
plt.figure(figsize=(8,4));
n = 6177 #index value, change to view different data. In this case,, #6177 is an image of wrist watch.


# Original Image with 4800 components
plt.subplot(1, 2, 1);
plt.imshow(X[n].reshape(X_train.shape[1], X_train.shape[2]),
              cmap = plt.cm.gray,);
plt.xlabel(str(X.shape[1])+' components', fontsize = 14)
plt.title('Original Image', fontsize = 20);


# Check an image with shrinked 773 principal components
plt.subplot(1, 2, 2);
plt.imshow(approximation[n].reshape(X_train.shape[1], X_train.shape[2]),
              cmap = plt.cm.gray,);
plt.xlabel(str(Clus_dataSet.shape[1]) +' components', fontsize = 14)
plt.title(str(variance * 100) + '% of Variance Retained', fontsize = 20);


# Install tqdm to visualize processing time left for K-Means clustering.
# This is not necessary, but I just like to visualize everything.
!pip install tqdm
import time
from tqdm import tqdm

# The number of initial clusters is very important in performing K-Means clustering.
# In cases where one knows the exact number of categories of given images to cluster, that number can be used.
# However, in reality, most datasets are not labeled as it is very expensive to do so.
# Therefore, I am going to assume that I do not know the number of categories of my fashion images.
# I calculated Within Cluster Sum of Squares (WCSS) and used the elbow method to deduce the best combination of the number of clusters and WCSS.
# K-Means Clustering의 경우 클러스터 갯수 설정이 매우 중요함
# 카테고리 수를 알고 있는 경우는 클러스터 갯수 설정을 동일하게 해주면 되지만, 이번 분석에서는 모른다고 가정하고 진행
# Within Cluster Sum of Squares (WCSS)를 계산하여, 클러스터 갯수와 WCSS의 여러 조합중에서 최적의 클러스터 갯수를 도출
wcss=[]
for i in tqdm(range(1,20)): 
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=30,random_state=0 )
    kmeans.fit(Clus_dataSet)
    wcss.append(kmeans.inertia_)
    

# A graph visualization of the elbow method.
plt.plot(range(1,20),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.locator_params(axis="x", nbins=20)
plt.show()


# Because the initial states of centroids (the center point of each cluster) are randomly set, the ultimate result of each K-Means depends heavily 
# on the initial state. Therefore, initial states are set several times to minimize variances due to the randomness.
# As derived from the above elbow method, I am going to use set 3 as the number of clusters for my data.
# and repeatedly calculate inertia for those three clusters to deduce the best number of times for repeated initialization.
# As this takes a long time to process, I am going to limit the number of repetitions to max 30.
inertia = []
for k in tqdm(range(1,30)):
    kmeans = KMeans(init = "k-means++",n_clusters=3, max_iter=300, n_init = k, random_state=1).fit(Clus_dataSet)
    inertia.append(np.sqrt(kmeans.inertia_))

    
# This is the graph of the inertia by the number of initializations.
plt.plot(range(1,30), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');
plt.locator_params(axis="x", nbins=30)


# Now I finally, have all the necessary parameters to perform K-Means clustering.
# The number of clusters (centroids) is set to 3 and there will be 4 initializations to account for the randomness of the initial states of the centroids.
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 4)


# Fit the data to our k_means model
k_means.fit(Clus_dataSet)








k_means_labels = k_means.labels_ #List of labels of each dataset
print("The list of labels of the clusters are " + str(np.unique(k_means_labels)))


G = len(np.unique(k_means_labels)) #Number of labels
#2D matrix  for an array of indexes of the given label
cluster_index= [[] for i in range(G)]
for i, label in enumerate(k_means_labels,0):
    for n in range(G):
        if label == n:
            cluster_index[n].append(i)
        else:
            continue
            
            
k_means_cluster_centers = k_means.cluster_centers_ #numpy array of cluster centers
k_means_cluster_centers.shape #comes from 10 clusters and 420 features


#cluster visualisation
my_members = (k_means_labels == 0) #Enter different Cluster number to view its 3D plot
my_members.shape
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1,projection='3d')
#Clus_dataSet.shape
#Clus_dataSet[my_members,300].shape
ax.plot(Clus_dataSet[my_members, 0], Clus_dataSet[my_members,1],Clus_dataSet[my_members,2], 'w', markerfacecolor="blue", marker='.',markersize=10)


#cluster visualisation
my_members = (k_means_labels == 1) #Enter different Cluster number to view its 3D plot
my_members.shape
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1,projection='3d')
#Clus_dataSet.shape
#Clus_dataSet[my_members,300].shape
ax.plot(Clus_dataSet[my_members, 0], Clus_dataSet[my_members,1],Clus_dataSet[my_members,2], 'w', markerfacecolor="blue", marker='.',markersize=10)


#cluster visualisation
my_members = (k_means_labels == 2) #Enter different Cluster number to view its 3D plot
my_members.shape
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1,projection='3d')
#Clus_dataSet.shape
#Clus_dataSet[my_members,300].shape
ax.plot(Clus_dataSet[my_members, 0], Clus_dataSet[my_members,1],Clus_dataSet[my_members,2], 'w', markerfacecolor="blue", marker='.',markersize=10)


# !pip install chart_studio 
# !pip install plotly


import plotly as py
import plotly.graph_objs as go
import plotly.express as px


#3D Plotly Visualisation of Clusters using go

layout = go.Layout(
    title='<b>Cluster Visualisation</b>',
    yaxis=dict(
        title='<i>Y</i>'
    ),
    xaxis=dict(
        title='<i>X</i>'
    )
)
colors = ['red','green' ,'blue',]
trace = [ go.Scatter3d() for _ in range(11)]
for i in range(0,3):
    my_members = (k_means_labels == i)
    index = [h for h, g in enumerate(my_members) if g]
    trace[i] = go.Scatter3d(
            x=Clus_dataSet[my_members, 0],
            y=Clus_dataSet[my_members, 1],
            z=Clus_dataSet[my_members, 2],
            mode='markers',
            marker = dict(size = 2,color = colors[i]),
            hovertext=index,
            name='Cluster'+str(i),
            )
fig = go.Figure(data=[trace[0],trace[1],trace[2]], layout=layout)
py.offline.iplot(fig)


#If you hover over the points in the above plots you get an index value
n = 3264 #Use that value here to visualise the selected data
plt.imshow(X[n].reshape(60, 80), cmap = plt.cm.binary)
plt.show()


#Visualisation for clusters = clust
plt.figure(figsize=(20,20));
clust = 0 #enter label number to visualise
num = 100 #num of data to visualize from the cluster
for i in range(1,num): 
    plt.subplot(10, 10, i); #(Number of rows, Number of column per row, item number)
    plt.imshow(X[cluster_index[clust][i+500]].reshape(X_train.shape[1], X_train.shape[2]), cmap = plt.cm.binary);
plt.show()









