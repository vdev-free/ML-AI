import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# 1) Зробимо маленький "реальний" табличний датасет
# -----------------------
rng = np.random.RandomState(42)
n = 800

df = pd.DataFrame({
    "age": rng.randint(18, 70, size=n),                     # число
    "income": rng.normal(45000, 15000, size=n).clip(5000),  # число (різний масштаб!)
    "city": rng.choice(["Helsinki", "Espoo", "Tampere"], size=n),  # категорія
    "device": rng.choice(["ios", "android", "desktop"], size=n),   # категорія
})

# Ціль: купив підписку (1/0)
# Робимо залежність: вищий дохід + деякі міста/девайси → більше шансів
logit = (
    0.00005 * df["income"]
    + 0.03 * (df["age"] - 35)
    + (df["city"] == "Helsinki") * 0.6
    + (df["device"] == "ios") * 0.4
    - 4.0
)

proba = 1 / (1 + np.exp(-logit))
y = (rng.rand(n) < proba).astype(int)

X = df.copy()

print("Target balance (0/1):", np.unique(y, return_counts=True))

# -----------------------
# 2) Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# 3) Функція звіту
# -----------------------
def report(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("accuracy:", accuracy_score(y_true, y_pred))
    print("precision:", precision_score(y_true, y_pred, zero_division=0))
    print("recall:", recall_score(y_true, y_pred, zero_division=0))
    print("f1:", f1_score(y_true, y_pred, zero_division=0))
    print("cm:\n", confusion_matrix(y_true, y_pred))

# -----------------------
# 4) ВАРІАНТ A — "Погано": LogisticRegression без preprocessing
# (це спеціально: воно впаде або дасть погано, бо є строки)
# -----------------------
try:
    bad_model = LogisticRegression(max_iter=2000)
    bad_model.fit(X_train, y_train)
    pred_bad = bad_model.predict(X_test)
    report("A) BAD: LogisticRegression without preprocessing", y_test, pred_bad)
except Exception as e:
    print("\nA) BAD: LogisticRegression without preprocessing -> ERROR (і це нормально)")
    print("Error:", e)

# -----------------------
# 5) ВАРІАНТ B — "Напів-погано": викинемо категорії і зробимо тільки scaling чисел
# -----------------------
num_features = ["age", "income"]
X_train_num = X_train[num_features]
X_test_num = X_test[num_features]

model_num = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

model_num.fit(X_train_num, y_train)
pred_num = model_num.predict(X_test_num)
report("B) OK-ish: only numeric + scaling + LogisticRegression", y_test, pred_num)

# -----------------------
# 6) ВАРІАНТ C — "Правильно": ColumnTransformer (числа scaled, категорії one-hot) + модель
# -----------------------
cat_features = ["city", "device"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

good_model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000))
])

good_model.fit(X_train, y_train)
pred_good = good_model.predict(X_test)
report("C) GOOD: ColumnTransformer + LogisticRegression", y_test, pred_good)

# -----------------------
# 7) Бонус: Tree model (RF) — encoding потрібно, scaling майже ні
# -----------------------
rf_model = Pipeline(steps=[
    ("prep", ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
])

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)
report("D) RandomForest + OneHot (no scaling)", y_test, pred_rf)

good_model_bal = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
good_model_bal.fit(X_train, y_train)
pred = good_model_bal.predict(X_test)
report("C2) GOOD + class_weight=balanced", y_test, pred)
