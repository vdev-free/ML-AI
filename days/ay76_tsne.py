import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1) Дані
data = load_wine()
X = data.data
y = data.target

# 2) Scaling (обовʼязково)
X_scaled = StandardScaler().fit_transform(X)

# 3) (Рекомендовано) Спочатку PCA до 10 компонент
# t-SNE краще працює і швидше, якщо перед ним прибрати шум PCA’шкою
X_pca10 = PCA(n_components=10).fit_transform(X_scaled)

# 4) t-SNE робить 2D
tsne = TSNE(
    n_components=2,      # хочемо 2 координати (для графіка)
    perplexity=30,       # “скільки сусідів” враховувати (локальність)
    learning_rate="auto",
    init="pca",          # стартові точки беремо з PCA (стабільніше)
    random_state=42      # щоб результат повторювався
)

X_2d = tsne.fit_transform(X_pca10)

# 5) Малюємо
plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.title("Wine: t-SNE 2D")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")
plt.show()
