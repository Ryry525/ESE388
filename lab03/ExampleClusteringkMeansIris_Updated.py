"""
ExampleClusteringkMeansIris_Updated
"""

#******************************************************************************
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import warnings
warnings.filterwarnings("ignore")

"""#Load dataset"""

# Load iris data into a pandas DataFrame, from the built-in Seaborn Data Set
iris = sns.load_dataset("iris")
print(iris.head())

"""#Pairplot using Species as Hue"""

#****************************************
# In the following plots, we use plt.figure() and plt.show(),
# before and after each plot, to avoid the plots from overlapping.

#****************************************
# Scatter Plots with "pairplot"
plt.figure(1)
sns.pairplot(data=iris, hue="species", palette="husl")
plt.show()

"""#K-means Clustering Implementation"""

#****************************************
# Separate X (features) and Y (target variable)
irisX = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
irisY = iris[["species"]]

#****************************************
# Perform k-means clustering, with the X features; note that the target variable
# is not utilized in clustering; n_clusters is an input parameter.
km = KMeans(n_clusters=3)
km.fit(irisX)

#****************************************
# Extract cluster centers
centers = km.cluster_centers_
print(centers)

"""#Extract Cluster Labels"""

# Extract cluster labels
iris["new_labels"] = km.labels_
print(iris.head())

# # Map species names to numbers for scatter plots
#Commenting out the next line as we are only comparing the new labels.
# iris["species"] = iris["species"].map({"setosa":0, "versicolor":1, "virginica":2})

"""# Calculating Davies-Bouldin Index(DBI)"""

# Compute the Davies-Bouldin Index (DBI)
dbi = davies_bouldin_score(irisX, km.labels_)
print(f"Davies-Bouldin Index: {dbi:.4f}")

"""# Pairplots after the K-means implementation using new labels as hue"""

# Pairplot using the new clustering labels as hue
plt.figure(2)
sns.pairplot(data=iris, hue="new_labels", palette="coolwarm")
plt.show()

"""#Repeating the above process for different values of k"""

# List of values for k (number of clusters)
k_values = [2, 3, 4, 5]

# Loop through different values of k
for k in k_values:
    print(f"\nRunning K-Means for k = {k}")

    # Perform k-means clustering with k clusters
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(irisX)

    # Add the cluster labels as a new column "new_labels"
    iris["new_labels"] = km.labels_

    # Compute the Davies-Bouldin Index (DBI)
    dbi = davies_bouldin_score(irisX, km.labels_)
    print(f"Davies-Bouldin Index for k = {k}: {dbi:.4f}")

    # Pairplot using the new clustering labels as hue
    plt.figure(figsize=(10, 8))
    sns.pairplot(data=iris, hue="new_labels", palette="coolwarm")
    plt.suptitle(f"Pairplot for k = {k}", y=1.02, fontsize=16)
    plt.show()