import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1) Дані (твої)
data = pd.DataFrame({
    'time':[1,2,2,4,4,1,3,3],
    'classed':[1,1,3,2,3,1,2,2],
    'reviews':[0,0,3,6,4,7,9,1]
})

# 2) Масштабування (важливо: приводимо всі ознаки до співставних шкал)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[['time','classed','reviews']])

# 3) Кластеризація (на масштабованих даних)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X_scaled)
data['cluster'] = labels

# 4) PCA до 2 компонентів (щоб намалювати у 2D)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Пояснена варіація по компонентах:", np.round(pca.explained_variance_ratio_, 3))
print("Сума поясненої варіації:", np.round(pca.explained_variance_ratio_.sum(), 3))

# 5) Візуалізація у 2D з кольорами кластерів
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', s=80, edgecolor='k')
plt.xlabel('PC1 (головна компонента 1)')
plt.ylabel('PC2 (головна компонента 2)')
plt.title('PCA (2D) + KMeans кластери')
plt.grid(True, alpha=0.3)

# (опц.) підписи точок їхнім індексом
for i, (x, y) in enumerate(X_pca):
    plt.text(x+0.02, y+0.02, str(i), fontsize=9)

plt.tight_layout()
plt.show()

