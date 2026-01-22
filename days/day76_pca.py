import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_wine()
X = data.data
y = data.target

# print("X shape:", X.shape)
# print("y classes:", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print("\ntrain:", X_train.shape, y_train.shape)
# print("test:", X_test.shape, y_test.shape)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.fit_transform(X_test)

# print("\nScaled train mean ~", np.round(X_train_sc.mean(), 3),
#       "std ~", np.round(X_train_sc.std(), 3))


pca_full = PCA()
pca_full.fit(X_train_sc)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

# print("\nExplained variance ratio (per PC):")
# for i, r in enumerate(explained, start=1):
#     print(f"PC{i}: {r:.4f}")

# print("\nCumulative explained variance:")
# for i, c in enumerate(cumulative, start=1):
#     print(f"PC1..PC{i}: {c:.4f}")

k_95 = int(np.argmax(cumulative >=0.95) + 1)
# print("\nComponents for >=95% variance:", k_95)

pca_2 = PCA(n_components=2)
X_train_pca2 = pca_2.fit_transform(X_train_sc)

# plt.figure()
# plt.scatter(X_train_pca2[:, 0], X_train_pca2[:, 1], c=y_train)
# plt.title("Wine dataset: PCA 2D (train)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

# print("PCA2 explained:", pca_2.explained_variance_ratio_)
# print("PCA2 total explained:", pca_2.explained_variance_ratio_.sum())

baseline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000))
]
)
baseline.fit(X_train_sc, y_train)
base_pred = baseline.predict(X_test_sc)

# print("\n=== Baseline (Scaler + LogisticRegression) ===")
# print("test acc:", accuracy_score(y_test, base_pred))
# print("cm:\n", confusion_matrix(y_test, base_pred))

pca_model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=k_95)),
    ("clf", LogisticRegression(max_iter=2000))
])
pca_model.fit(X_train, y_train)
pca_pred = pca_model.predict(X_test)
# print(f"\n=== PCA model (Scaler + PCA({k_95}) + LogisticRegression) ===")
# print("test acc:", accuracy_score(y_test, pca_pred))
# print("cm:\n", confusion_matrix(y_test, pca_pred))