import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import umap

# 1) Дані
data = load_wine()
X = data.data
y = data.target

# 2) Scaling
X_scaled = StandardScaler().fit_transform(X)

# 3) UMAP -> 2D
reducer = umap.UMAP(
    n_components=2,   # 2D
    n_neighbors=15,   # “скільки сусідів” (як perplexity у t-SNE)
    min_dist=0.1,     # наскільки щільно можна “зліплювати” точки в кластері
    random_state=42
)

X_2d = reducer.fit_transform(X_scaled)

# 4) Малюємо
plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.title("Wine: UMAP 2D")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()
 