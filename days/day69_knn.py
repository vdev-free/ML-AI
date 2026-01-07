import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print("train:", X_train.shape, y_train.shape)
# print("test:", X_test.shape, y_test.shape)

knn_no_scaler = KNeighborsClassifier(n_neighbors=5)
knn_no_scaler.fit(X_train, y_train)

y_pred_no = knn_no_scaler.predict(X_test)

# print("\n=== KNN WITHOUT scaling (k=5) ===")
# print("accuracy:", accuracy_score(y_test, y_pred_no))
# print("precision:", precision_score(y_test, y_pred_no))
# print("recall:", recall_score(y_test, y_pred_no))
# print("f1:", f1_score(y_test, y_pred_no))
# print("cm:\n", confusion_matrix(y_test, y_pred_no))

knn_scaled = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=3))
])

knn_scaled.fit(X_train, y_train)
y_pred_sc = knn_scaled.predict(X_test)

# print("\n=== KNN WITH scaling (k=3) ===")
# print("accuracy:", accuracy_score(y_test, y_pred_sc))
# print("precision:", precision_score(y_test, y_pred_sc))
# print("recall:", recall_score(y_test, y_pred_sc))
# print("f1:", f1_score(y_test, y_pred_sc))
# print("cm:\n", confusion_matrix(y_test, y_pred_sc))

best_k = None
best_acc = 0

for k in range(1, 26):
    knn = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if acc > best_acc:
        best_acc = acc
        best_k = k

# print("Best k:", best_k)
# print("Best accuracy:", best_acc)
        
print("\n=== KNN sweep over k (WITH scaling) ===")

for k in [1, 3, 5, 15, 30]:
    knn = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nk =", k, "acc =", acc)
    print("cm:\n", cm)

