import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target

# print('X shape', X.shape)
# print('y classes', np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print('train', X_train.shape, y_train.shape)
# print('test', X_test.shape, y_test.shape)

for C in [0.01, 0.1, 1, 10, 100]:
    svm = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=C))
    ])
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    print(f"\n=== SVM linear C={C} ===")
    print("train acc:", svm.score(X_train, y_train))
    print("test acc:", svm.score(X_test, y_test))
    print("cm:\n", confusion_matrix(y_test, y_pred))


