import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

data = load_wine()
X = data.data
y = data.target

# print("X shape:", X.shape)
# print("classes (real, just for reference):", np.unique(y, return_counts=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# print("Scaled X: mean ~", np.round(X_scaled.mean(), 3), "std ~", np.round(X_scaled.std(), 3))

for k in [2,3,4,5,6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    # print(f"k={k} silhouette={score:.3f}")




from sklearn.cluster import DBSCAN
import numpy as np

db = DBSCAN(eps=0.9, min_samples=5)
labels_db = db.fit_predict(X_scaled)

# print("labels:", np.unique(labels_db, return_counts=True))
# print("noise points:", np.sum(labels_db == -1))

from sklearn.cluster import DBSCAN

for eps in [0.5, 0.7, 0.9, 1.1]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels_db = db.fit_predict(X_scaled)

    n_noise = np.sum(labels_db == -1)
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)

    # print(f"\n=== DBSCAN eps={eps} ===")
    # print("clusters:", n_clusters, "noise:", n_noise)
    # print("labels counts:", np.unique(labels_db, return_counts=True))

min_samples = 5

nn = NearestNeighbors(n_neighbors=min_samples)
nn.fit(X_scaled)

distances, _ = nn.kneighbors(X_scaled)

# distance to the 5th nearest neighbor for each point
kdist = np.sort(distances[:, -1])

# print("kdist min:", kdist[0])
# print("kdist 50%:", kdist[int(len(kdist)*0.50)])
# print("kdist 75%:", kdist[int(len(kdist)*0.75)])
# print("kdist 90%:", kdist[int(len(kdist)*0.90)])
# print("kdist 95%:", kdist[int(len(kdist)*0.95)])
# print("kdist max:", kdist[-1])

from sklearn.cluster import DBSCAN
import numpy as np

for eps in [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels_db = db.fit_predict(X_scaled)

    n_noise = np.sum(labels_db == -1)
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)

    # print(f"\neps={eps}")
    # print("clusters:", n_clusters, "noise:", n_noise)
    # print("counts:", np.unique(labels_db, return_counts=True))


from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 1) KMeans k=3
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = km.fit_predict(X_scaled)

print("\nKMeans k=3")
print("ARI:", adjusted_rand_score(y, km_labels))
print("NMI:", normalized_mutual_info_score(y, km_labels))

# 2) DBSCAN eps=2.4
db = DBSCAN(eps=2.4, min_samples=5)
db_labels = db.fit_predict(X_scaled)

# для ARI/NMI можна залишити noise як є (-1) — це норм
print("\nDBSCAN eps=2.4")
print("ARI:", adjusted_rand_score(y, db_labels))
print("NMI:", normalized_mutual_info_score(y, db_labels))
print("noise:", (db_labels == -1).sum())
