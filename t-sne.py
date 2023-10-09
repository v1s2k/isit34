import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


data = pd.read_csv('hepatitis.csv')


data = data.dropna()

# Нормализация числовых признаков
numeric_features = data.select_dtypes(include=[np.number])


#Применение алгоритма t-SNE
tsne = TSNE(n_components=2,perplexity=30, random_state=42)
data_tsne = tsne.fit_transform(numeric_features)

#  Визуализация результатов
plt.scatter(data_tsne[:, 0], data_tsne[:, 1],c=data['age'] ,cmap='viridis')
plt.title('t-SNE Визуализация')
plt.xlabel('Параметр 1')
plt.ylabel('Параметр 2')
plt.colorbar()
plt.show()
print(numeric_features)
