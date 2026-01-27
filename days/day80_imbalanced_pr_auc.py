import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)

rng = np.random.RandomState(42)
n = 800

df = pd.DataFrame({
    "age": rng.randint(18, 70, size=n),
    "income": rng.normal(45000, 15000, size=n).clip(5000),
    "city": rng.choice(["Helsinki", "Espoo", "Tampere"], size=n),
    "device": rng.choice(["ios", "android", "desktop"], size=n),
})

# Робимо "1" рідкісним: зсуваємо logit вниз
logit = (
    0.00005 * df["income"]
    + 0.03 * (df["age"] - 35)
    + (df["city"] == "Helsinki") * 0.6
    + (df["device"] == "ios") * 0.4
    - 5.0  # <-- нижче, ніж було: 1-клас стане рідкіснішим
)

proba = 1 / (1 + np.exp(-logit))
y = (rng.rand(n) < proba).astype(int)

# print("First 3 rows:\n", df.head(3))
# print("\nClass balance y (0/1):", np.unique(y, return_counts=True))

X = df.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# print("\nShapes:")
# print("X_train:", X_train.shape, "y_train:", y_train.shape)
# print("X_test :", X_test.shape, "y_test :", y_test.shape)

# print("\nClass balance:")
# print("train (0/1):", np.unique(y_train, return_counts=True))
# print("test  (0/1):", np.unique(y_test, return_counts=True))

num_cols = ["age", "income"]
cat_cols = ["city", "device"]

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ]
)

pipe = Pipeline(steps=[
    ('prep', preprocess),
    ('clf', LogisticRegression(max_iter=2000))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# print("\n=== Baseline (t=0.5) ===")
# print("accuracy:", accuracy_score(y_test, y_pred))
# print("precision:", precision_score(y_test, y_pred, zero_division=0))
# print("recall:", recall_score(y_test, y_pred, zero_division=0))
# print("f1:", f1_score(y_test, y_pred, zero_division=0))
# print("cm:\n", confusion_matrix(y_test, y_pred))


proba_pos = pipe.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, proba_pos)

# print("\nPR-AUC (Average Precision):", pr_auc)

def report_threshold(t: float):
    y_pred_t = (proba_pos >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    acc = accuracy_score(y_test, y_pred_t)
    print(f"\n--- threshold={t} ---")
    print("accuracy:", acc)
    print("precision:", prec)
    print("recall:", rec)
    print("f1:", f1)
    print("cm:\n", cm)

# for t in [0.3, 0.5, 0.7]:
#     report_threshold(t)

pipe_bal = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

pipe_bal.fit(X_train, y_train)

proba_bal = pipe_bal.predict_proba(X_test)[:, 1]
y_pred_bal = (proba_bal >= 0.5).astype(int)

# print("\n=== Balanced (class_weight) threshold=0.5 ===")
# print("accuracy:", accuracy_score(y_test, y_pred_bal))
# print("precision:", precision_score(y_test, y_pred_bal, zero_division=0))
# print("recall:", recall_score(y_test, y_pred_bal, zero_division=0))
# print("f1:", f1_score(y_test, y_pred_bal, zero_division=0))
# print("cm:\n", confusion_matrix(y_test, y_pred_bal))

# print("\nPR-AUC balanced:", average_precision_score(y_test, proba_bal))


def report_threshold(t: float):
    y_pred_t = (proba_bal >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    acc = accuracy_score(y_test, y_pred_t)
    print(f"\n--- threshold={t} ---")
    print("accuracy:", acc)
    print("precision:", prec)
    print("recall:", rec)
    print("f1:", f1)
    print("cm:\n", cm)

# for t in [0.3, 0.5, 0.7]:
#     report_threshold(t)

from sklearn.metrics import average_precision_score

# print("\nPR-AUC baseline:", average_precision_score(y_test, proba_pos))   # з твого baseline
# print("PR-AUC balanced:", average_precision_score(y_test, proba_bal))    # з balanced

# беремо ймовірності (візьми baseline або balanced — наприклад baseline proba_pos)
proba = proba_pos

quota = 0.20  # 20% отримають email
t_quota = np.quantile(proba, 1 - quota)  # поріг так, щоб 20% були >= t

# print("\nQuota:", quota)
# print("Threshold for quota:", t_quota)

# y_pred_q = (proba >= t_quota).astype(int)
# print("Predicted 1 rate:", y_pred_q.mean())
# print("cm:\n", confusion_matrix(y_test, y_pred_q))
# print("precision:", precision_score(y_test, y_pred_q, zero_division=0))
# print("recall:", recall_score(y_test, y_pred_q, zero_division=0))

for quota in [0.10, 0.20, 0.30]:
    t = np.quantile(proba_pos, 1 - quota)
    y_pred_q = (proba_pos >= t).astype(int)

    cm = confusion_matrix(y_test, y_pred_q)
    prec = precision_score(y_test, y_pred_q, zero_division=0)
    rec = recall_score(y_test, y_pred_q, zero_division=0)

    print("\nquota:", quota, "threshold:", round(t, 3), "pred1_rate:", round(y_pred_q.mean(), 3))
    print("cm:\n", cm)
    print("precision:", round(prec, 3), "recall:", round(rec, 3))

