import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Apply K-Means clustering (3 clusters for 3 species)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[iris.feature_names])

# Reduce dimensions with PCA for visualization
pca = PCA(n_components=2)
pc = pca.fit_transform(df[iris.feature_names])
df['PC1'] = pc[:,0]
df['PC2'] = pc[:,1]

# Visualization
plt.figure(figsize=(8,6))
for cluster in range(3):
    subset = df[df['cluster'] == cluster]
    plt.scatter(subset['PC1'], subset['PC2'], label=f"Cluster {cluster}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.title("K-Means Clustering on Iris Dataset")
plt.show()
