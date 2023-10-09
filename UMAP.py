import pandas as pd
import umap
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('hepatitis.csv')
data = data.dropna()
# Удаление столбца с целевой переменной (для визуализации)
X = data.drop('age', axis=1)
numeric_features = data.select_dtypes(include=[np.number])
print(numeric_features)
# Выполнение UMAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(numeric_features)

# Визуализация
plt.scatter(embedding[:, 0], embedding[:, 1], c=data['age'], cmap='viridis')
plt.title('UMAP Visualization of Hepatitis Dataset')

plt.colorbar()
plt.show()