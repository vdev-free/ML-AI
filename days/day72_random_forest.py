import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("train:", X_train.shape, y_train.shape)
print("test:", X_test.shape, y_test.shape)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred_tree  = tree.predict(X_test)

print("\n=== Decision Tree (max_depth=3) ===")
print("accuracy:", accuracy_score(y_test, y_pred_tree))
print("cm:\n", confusion_matrix(y_test, y_pred_tree))

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest (200 trees) ===")
print("accuracy:", accuracy_score(y_test, y_pred_rf))
print("precision:", precision_score(y_test, y_pred_rf))
print("recall:", recall_score(y_test, y_pred_rf))
print("f1:", f1_score(y_test, y_pred_rf))
print("cm:\n", confusion_matrix(y_test, y_pred_rf))

print("\nRF train accuracy:", rf.score(X_train, y_train))
print("RF test accuracy:", rf.score(X_test, y_test))

# 3) Які фічі найважливіші (топ-10)
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1][:10]

print("\nTop-10 important features:")
for i in idx:
    print(f"{data.feature_names[i]}: {importances[i]:.4f}")