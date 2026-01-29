import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

os.makedirs('artifacts/day83', exist_ok=True)

data = load_breast_cancer()
X = data['data']
y = data['target']
feature_names = data['feature_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# print("Shapes:", X_train.shape, X_test.shape)
# print("Positive rate (train):", round(float(y_train.mean()), 3))
# print("Positive rate (test):", round(float(y_test.mean()), 3))

model = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1
)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)

# print("Test AUC:", round(float(auc), 4))

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1,
)

importances = result.importances_mean
idx = np.argsort(importances)[::-1]

# print("\nTop-10 permutation importances:")
# for i in idx[:10]:
#     print(f"{feature_names[i]}: {importances[i]:.5f}")

topk = 10
top_idx = idx[:topk]

# plt.figure(figsize=(8, 5))
# plt.barh([feature_names[i] for i in top_idx][::-1], importances[top_idx][::-1])
# plt.tight_layout()
# plt.savefig("artifacts/day83/permutation_importance_top10.png")
# plt.close()

# print("\nSaved: artifacts/day83/permutation_importance_top10.png")

top_feature_index = int(idx[0])

# fig, ax = plt.subplots(figsize=(8, 5))
# PartialDependenceDisplay.from_estimator(
#     model,
#     X_test,
#     [top_feature_index],
#     feature_names=feature_names,
#     kind="average",
#     ax=ax,
# )
# plt.tight_layout()
# plt.savefig("artifacts/day83/pdp_top1.png")
# plt.close(fig)

# print("Saved: artifacts/day83/pdp_top1.png")

# print("Top-1 feature:", feature_names[top_feature_index])


fig, ax = plt.subplots(figsize=(8, 5))
PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    [top_feature_index],
    feature_names=feature_names,
    kind="individual",
    subsample=60,
    ax=ax,
)
plt.tight_layout()
plt.savefig("artifacts/day83/ice_top1.png")
plt.close(fig)

print("Saved: artifacts/day83/ice_top1.png")
