import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

# print('X shape:', X.shape)
# print('Y shape:', y.shape)
# print('classes:', np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# print('train:', X_train.shape, y_train.shape)
# print('test:', X_test.shape, y_test.shape)

# print('train class counts:', np.unique(y_train, return_counts=True))
# print('test class counts:', np.unique(y_test, return_counts=True))

model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
# print('Model trained!')

y_pred = model.predict(X_test)
# print('First 20 predictions:', y_pred[:20])
# print('First 20 true labels:', y_test[:20])

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# print("\naccuracy:", acc)
# print("precision:", prec)
# print("recall:", rec)
# print("f1:", f1)

# print("\nconfusion matrix:\n", confusion_matrix(y_test, y_pred))

proba = model.predict_proba(X_test)[:, 1]

def eval_threshold(t: float):
    y_pred_t = (proba >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    return t, prec, rec, f1, cm

for t in [0.3, 0.5, 0.7]:
    t, prec, rec, f1, cm = eval_threshold(t)
    print("\nTH:", t)
    print("precision:", prec)
    print("recall:", rec)
    print("f1:", f1)
    print("cm:\n", cm)
