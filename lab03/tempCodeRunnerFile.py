X_raisin = raisin_df.drop(columns=["Class"])
# K-means clustering
kmeans_raisin = KMeans(n_clusters=2)
raisin_df['Cluster'] = kmeans_raisin.fit_predict(X_raisin)
