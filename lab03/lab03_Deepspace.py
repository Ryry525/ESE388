import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
warnings.filterwarnings("ignore")

deep_df = pd.read_csv("lab03/DeepSpaceData.csv")
print(deep_df.head())

# Scatterplot using Class as Hue
sns.pairplot(data=deep_df, palette="husl")
plt.suptitle("Deep Data: Scatterplot by Class", y=1.02)
plt.show()

X_raisin = deep_df.drop(columns=["X1"])

# K-means clustering
kmeans_raisin = KMeans(n_clusters=2)
kmeans_raisin.fit(X_raisin)

# Extract cluster labels
deep_df["new_labels"] = kmeans_raisin.labels_
print(deep_df.head())

# Calculate DBI
dbi = davies_bouldin_score(X_raisin, kmeans_raisin.labels_)
print(f"Davies-Bouldin Index: {dbi:.4f}")

# Pairplot using the new clustering labels as hue
sns.pairplot(data=deep_df, hue="new_labels", palette="coolwarm")
plt.show()

# Visualize clusters for different values of k
k_values = [2, 3, 4, 5, 6, 7, 8, 9]

for k in k_values:
    print(f"\nRunning K-Means for k = {k}")

    # Perform k-means clustering with k clusters on features only
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_raisin)

    # Add the cluster labels as a new column "new_labels"
    deep_df["new_labels"] = km.labels_

    # Compute the Davies-Bouldin Index (DBI)
    dbi = davies_bouldin_score(X_raisin, km.labels_)
    print(f"Davies-Bouldin Index for k = {k}: {dbi:.4f}")

    # Pairplot using the new clustering labels as hue
    sns.pairplot(data=deep_df, hue="new_labels", palette="coolwarm")
    plt.suptitle(f"Pairplot for k = {k}", y=1.02, fontsize=16)
    plt.show()


