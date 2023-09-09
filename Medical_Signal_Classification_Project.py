
# 1. Import related libraries  #########################################################################################

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import  accuracy_score, confusion_matrix,classification_report

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)



# 2.  Read the data set  ##############################################################################################

data= pd.read_csv(r"C:\Users\erkan\Desktop\datasets\example.TXT", sep=" ", header=None,
                  usecols =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
df = data.copy()

df.head()

## give names to columns

df.columns=["v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12"]
df.head()

df.shape
#  (5000, 12)  we have 5000 obversation (rows) and 12 variable (columns)
df.isnull().sum()
# We have not NA value
df.info()
# All variables  dtype are int


# 3. K-Means ( Unsupervised Learning Algorithm ) ######################################################################

## 3.1  We determine the optimal number of clusters with elbow method

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_
# Number of cluster is 6.


## 3.2 Creation of Final Clusters

kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=34).fit(df)


kmeans.n_clusters               # number of cluster
kmeans.cluster_centers_         # cluster centers

kmeans.labels_[0:15]             # cluster labels

# add the cluster to data set

clusters_kmeans = kmeans.labels_
df["cluster"] = clusters_kmeans

df.head()

# number of observations for each cluster

pd.DataFrame({"Count": df["cluster"].value_counts(),
              "Ratio": 100 * df["cluster"].value_counts() / len(df)})



# Calculate the counts for each cluster
cluster_counts = df["cluster"].value_counts()

# Plot the cluster counts as a bar chart
ax = cluster_counts.plot(kind="bar")  # Use 'barh' for horizontal bar chart
# Add multiple horizontal lines at different thresholds
thresholds = [100, 750, 1250]
colors = ['red', 'orange', 'green']
linestyles = ['--', '-.', ':']
labels = ['Threshold (100)', 'Threshold (750)', 'Threshold (1250)']

for threshold, color, linestyle, label in zip(thresholds, colors, linestyles, labels):
    plt.axhline(y=threshold, color=color, linestyle=linestyle, label=label)

plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Cluster Counts")
plt.legend()
plt.show()



# observation average of clusters

df.groupby("cluster").agg("mean")
df.groupby("cluster").agg("sum")

# Calculate groups by average values
grouped_means = df.groupby("cluster").agg("mean")

# Plot the cluster counts as a bar chart
ax=grouped_means.plot(kind="bar",width=0.8)
plt.xlabel("Cluster")
plt.ylabel("Mean Value")
plt.title("Cluster Means")
# Change the legend position to upper right
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#plt.legend(loc='lower right', bbox_to_anchor=(0, 1),title='Legend')
# ax.get_legend().set_visible(False)
# Adjust the figure size to accommodate the legend
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()




## 3.3 Visualize the clusters


#  v1 & v2
plt.figure(figsize=(10, 6))

# Customize colors for each cluster (you can change this as per your preference)
colors = ['r', 'g', 'b', 'c', 'm', 'y']


for cluster in range(6):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['v1'], cluster_data['v2'], c=colors[cluster], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='x', label='Centroids')
plt.xlabel('v1')
plt.ylabel('v2')
plt.title('K-Means Clustering Results')
plt.legend()
plt.show()



#  v1 & v2 bar chart
plt.figure(figsize=(10, 6))


for cluster in range(6):
    cluster_data = df[df['cluster'] == cluster]
    plt.bar(cluster_data['v1'], cluster_data['v2'], label=f'Cluster {cluster}')


plt.xlabel('v1')
plt.ylabel('v2')
plt.title('K-Means Clustering Results')
plt.legend()
plt.show()






#  v1 & v3
plt.figure(figsize=(10, 6))

# Customize colors for each cluster (you can change this as per your preference)
colors = ['r', 'g', 'b', 'c', 'm', 'y']


for cluster in range(6):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['v1'], cluster_data['v3'], c=colors[cluster], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='x', label='Centroids')
plt.xlabel('v1')
plt.ylabel('v3')
plt.title('K-Means Clustering Results')
plt.legend()
plt.show()


#  v2 & v3
plt.figure(figsize=(10, 6))

# Customize colors for each cluster (you can change this as per your preference)
colors = ['r', 'g', 'b', 'c', 'm', 'y']


for cluster in range(6):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['v2'], cluster_data['v3'], c=colors[cluster], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=100, c='black', marker='x', label='Centroids')
plt.xlabel('v2')
plt.ylabel('v3')
plt.title('K-Means Clustering Results')
plt.legend()
plt.show()



#  v1 & v12
plt.figure(figsize=(10, 6))

# Customize colors for each cluster (you can change this as per your preference)
colors = ['r', 'g', 'b', 'c', 'm', 'y']


for cluster in range(6):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['v1'], cluster_data['v12'], c=colors[cluster], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 11], s=100, c='black', marker='x', label='Centroids')
plt.xlabel('v1')
plt.ylabel('v12')
plt.title('K-Means Clustering Results')
plt.legend()
plt.show()



#  v11 & v12
plt.figure(figsize=(10, 6))

# Customize colors for each cluster (you can change this as per your preference)
colors = ['r', 'g', 'b', 'c', 'm', 'y']


for cluster in range(6):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['v11'], cluster_data['v12'], c=colors[cluster], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 10], kmeans.cluster_centers_[:, 11], s=100, c='black', marker='x', label='Centroids')
plt.xlabel('v11')
plt.ylabel('v12')
plt.title('K-Means Clustering Results')
plt.legend()
plt.show()



## 3d visualize

# v1 & v2 & v3

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# Customize colors for each cluster (you can change this as per your preference)
colors = ['r', 'g', 'b', 'c', 'm', 'y']


for cluster in range(6):
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data['v1'], cluster_data['v2'], cluster_data['v3'], c=colors[cluster], label=f'Cluster {cluster}')

ax.set_xlabel('v1')
ax.set_ylabel('v2')
ax.set_zlabel('v3')
ax.set_title('K-Means Clustering Results in 3D')
ax.legend()
plt.show()



# 4. Hierarchical Clustering  ( Unsupervised Learning Algorithm )  ####################################################


## 4.1 avarage method

hc_average = linkage(df, method="average")


## 4.2 Determining the Number of Clusters ( Dendrograms )



plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=100, color='r', linestyle='--')
plt.axhline(y=125, color='b', linestyle='--')
plt.show()


plt.figure(figsize=(7, 5))
plt.title("Hierarchical Clustering Dendograms")
plt.xlabel("Data Points")
plt.ylabel("Distances")
plt.axhline(y=100, color='r', linestyle='--')
plt.axhline(y=125, color='b', linestyle='--')
plt.axhline(y=200, color='y', linestyle='--')
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()



## 4.3 Create  model


cluster_hc = AgglomerativeClustering(n_clusters=6, linkage="average")

clusters_hc = cluster_hc.fit_predict(df)

# add the cluster to data set
df["hc_cluster"] = clusters_hc

df.head()


# Calculate and print the cluster centroids (means)
cluster_centroids = []
for cluster_id in np.unique(clusters_hc):
    cluster_data = data[clusters_hc == cluster_id]
    centroid = np.mean(cluster_data, axis=0)
    cluster_centroids.append(centroid)

cluster_centroids = np.array(cluster_centroids)
print("Cluster Centroids:")
print(cluster_centroids)




# 5. Modeling    ######################################################################################################

X =df.drop(["cluster","hc_cluster"], axis=1)   # features
y = df["cluster"]               # target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.45, random_state=34)

## 5.1 Create the Multinomial Logistic Regression  object

multi_log_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)

## 5.2  Fit the model to the training data

multi_log_model.fit(X_train, y_train)

y_pred = multi_log_model.predict(X_test)


## 5.3 Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Accuracy: 0.9938

## 5.4 Generate a classification report

print(classification_report(y_test, y_pred))

## 5.5 Generate a confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

## 5.6 K-fold cross validation

Accuracy =cross_val_score(multi_log_model, X_test, y_test, scoring="accuracy",cv=10).mean()
Accuracy

# Accuracy : 0.989777



## 5.6 Hyperparameter optimization

param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Regularization penalty ('l1' for L1 regularization, 'l2' for L2 regularization)
}

grid_search = GridSearchCV(multi_log_model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Best Hyperparameters: {'C': 1, 'penalty': 'l2'}
# Best Score: 0.994909090909091

final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Accuracy: 0.9938

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)



# 6. Model results check   ############################################################################################

feature_cols = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12']   # models features

# We randomly select observations from the data set and check the results

random_sample = df.sample(n=5)
random_sample

random_sample_mod = random_sample[feature_cols]
final_model.predict(random_sample_mod)



