import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
data = pd.read_csv("hepatitis.csv")
data = data.dropna()  # Remove rows with missing values
data = data.select_dtypes(['number'])
print(data.head())
features = data.iloc[:, 1:]
labels = data.iloc[:, 0]
tsne = TSNE(n_components=2, perplexity=30, random_state=32)
embeddings = tsne.fit_transform(features)
plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of Hepatitis Dataset')
plt.colorbar()
plt.show()