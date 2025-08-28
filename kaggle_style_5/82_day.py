import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. Дані
data = pd.DataFrame({
    "age": [22, 38, 26, 35, np.nan, 54, 2, 27, 14, np.nan],
    "sex": ["male", "female", "female", "female", "male", "male", "male", "female", "female", "male"],
    "pclass": [3, 1, 3, 1, 3, 1, 3, 2, 3, 3],
    "fare": [7.25, 71.28, 7.92, 53.1, 8.05, 51.86, 21.07, 11.13, 30.07, 7.23],
    "embarked": ["S", "C", "S", "S", "S", "S", "S", "C", "Q", "S"],
    "survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
})

X = data[["age", "sex", "pclass", "fare", "embarked"]]
y = data["survived"]

# 2. Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Препроцесинг
numeric = ["age", "fare"]
categorical = ["sex", "pclass", "embarked"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric),
    ("cat", categorical_pipeline, categorical),
])

# 4. Модель
model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])

# 5. Навчання
model.fit(X_train, y_train)

# 6. Оцінка
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
