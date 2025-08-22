import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1) Дані
df = pd.DataFrame({
    "age": [22, 30, 26, 28, 35, 40],
    "experience_years": [1, 5, 2, 3, 10, 15],
    "city": ["Tallinn", "Kyiv", "Warsaw", "Tallinn", "Kyiv", "Berlin"],
    "hired": [0, 1, 0, 1, 1, 1]
})

X = df[["age", "experience_years", "city"]]
y = df["hired"]

# 2) Трен/тест
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# 3) Преобробка: скейл числових + one-hot для city
numeric_features = ["age", "experience_years"]
categorical_features = ["city"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 4) Пайплайн: преобробка -> логістична регресія
model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

# 5) Навчання
model.fit(X_train, y_train)

# 6) Прогноз + звіт
y_pred = model.predict(X_test)
print("classification_report:\n", classification_report(y_test, y_pred, digits=3))

# 7) Новий кандидат (ті ж самі колонки!)
new_candidate = pd.DataFrame(
    [[29, 4, "Kyiv"]],
    columns=["age", "experience_years", "city"]
)

print("prediction_new_candidate:", model.predict(new_candidate))

