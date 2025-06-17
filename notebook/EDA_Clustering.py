import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

df = pd.read_csv("marketing_campaign.csv", sep='\t')
df = df.dropna()

df['TotalChildren'] = df['Kidhome'] + df['Teenhome']
df['Age'] = 2025 - df['Year_Birth']
df['Spending'] = df[['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)

features = [
    'Income', 'Recency', 'Age', 'TotalChildren', 'Spending',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth'
]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig("elbow_method.png")

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)
df['PCA1'] = pca_components[:,0]
df['PCA2'] = pca_components[:,1]

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2")
plt.title("Customer Segments Visualization")
plt.savefig("cluster_visualization.png")

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Cluster Centers:")
print(kmeans.cluster_centers_)

cluster_summary = df.groupby('Cluster')[features].mean().round(2)
print("\nCluster-wise Feature Averages:")
print(cluster_summary)

