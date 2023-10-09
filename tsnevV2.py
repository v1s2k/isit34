import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


data = pd.read_csv('hepatitis.csv')


data = data.dropna()
numeric_features = data.select_dtypes(include=[np.number])
print(numeric_features)
# Отделите признаки от меток классов
X = numeric_features.drop('age', axis=1)
#  пропущенные значения
X = X.fillna(X.mean())

# Применение алгоритма t-sne
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)


plt.scatter(X_tsne [:, 0], X_tsne [:, 1],c=data['age'] ,cmap='viridis')
plt.title('t-SNE Визуализация')
plt.xlabel('Параметр 1')
plt.ylabel('Параметр 2')
plt.colorbar()
plt.show()