import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import joblib

rng = np.random.RandomState(42)
n = 800                

df = pd.DataFrame({
    "age": rng.randint(18, 70, size=n),                
    "income": rng.normal(45000, 15000, size=n).clip(5000),  
    "city": rng.choice(["Helsinki", "Espoo", "Tampere"], size=n),  
    "device": rng.choice(["ios", "android", "desktop"], size=n),   
})

logit = (
    0.00005 * df["income"]                 # більший дохід → більша ймовірність
    + 0.03 * (df["age"] - 35)              # старший вік → трохи більша ймовірність
    + (df["city"] == "Helsinki") * 0.6     # Helsinki → бонус
    + (df["device"] == "ios") * 0.4        # ios → бонус
    - 4.0                                  # зсув, щоб "1" було менше ніж "0"
)

# 3) Перетворюємо logit → ймовірність [0..1]
proba = 1 / (1 + np.exp(-logit))

# 4) Генеруємо ціль y: купив(1) / не купив(0)
y = (rng.rand(n) < proba).astype(int)

# 5) X — це наші фічі (табличка)
X = df.copy()

print("Перші 3 рядки X:")
print(X.head(3))
print("\nБаланс класів y (0/1):", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

print("\nShapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

print("\nClass balance:")
print("train (0/1):", np.unique(y_train, return_counts=True))
print("test  (0/1):", np.unique(y_test, return_counts=True))

num_cols = ['age', 'income']
cat_cols = ['city', 'device']

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ]
)

X_train_prep = preprocess.fit_transform(X_train)
X_test_prep = preprocess.transform(X_test)

print("\nAfter preprocess:")
print("X_train_prep shape:", X_train_prep.shape)
print("X_test_prep shape:", X_test_prep.shape)

# 10) Дістаємо назви one-hot колонок (щоб бачити що реально створилось)
ohe = preprocess.named_transformers_["cat"]
ohe_feature_names = ohe.get_feature_names_out(cat_cols)

print("\nOne-hot features:")
for name in ohe_feature_names:
    print(" -", name)

def report(name: str, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("accuracy:", accuracy_score(y_true, y_pred))
    print("precision:", precision_score(y_true, y_pred, zero_division=0))
    print("recall:", recall_score(y_true, y_pred, zero_division=0))
    print("f1:", f1_score(y_true, y_pred, zero_division=0))
    print("cm:\n", confusion_matrix(y_true, y_pred))

pipe = Pipeline(steps=[
    ('preprocessing', preprocess),
    ('model', LogisticRegression(max_iter=2000, class_weight='balanced'))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
report("Pipeline + LogisticRegression (predict, t=0.5)", y_test, y_pred)

# 13) Ймовірність класу 1 (купив)
proba_test = pipe.predict_proba(X_test)[:, 1]

# 14) Перевіряємо різні threshold
for t in [0.3, 0.5, 0.7]:
    y_pred_t = (proba_test >= t).astype(int)
    report(f"Pipeline threshold={t}", y_test, y_pred_t)

MODEL_PATH = 'day78_subscription_pipe.joblib'
joblib.dump(pipe, MODEL_PATH)
print("\nSaved model to:", MODEL_PATH)

# 16) Завантажуємо як у проді
loaded = joblib.load(MODEL_PATH)
print("Loaded model OK!")

# 17) Створимо одного нового юзера (ВАЖЛИВО: колонки ті ж самі!)
new_user = pd.DataFrame([{
    "age": 42,
    "income": 52000,
    "city": "Helsinki",
    "device": "ios"
}])

p1 = loaded.predict_proba(new_user)[0, 1]   # ймовірність класу 1
pred = loaded.predict(new_user)[0]          # 0 або 1 при threshold=0.5

print("\nNew user:", new_user.to_dict(orient="records")[0])
print(f"P(buy)= {p1:.3f}")
print("predict(t=0.5):", int(pred))

p = float(p1)

if p >= 0.7:
    action = "CALL (дорого/ручна дія)"
elif p >= 0.3:
    action = "BANNER/EMAIL (дешева дія)"
else:
    action = "NO ACTION"

print("\nAction by thresholds:", action)
