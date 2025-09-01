import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Дані
data = pd.DataFrame({
    "age": [22, 38, 26, 35, 54, 2, 27, 14],
    "income": [3000, 8000, 5000, 7000, 10000, 1500, 4000, 2000],
    "loan": [0, 1, 0, 1, 0, 1, 0, 1],
    "default": [0, 0, 1, 0, 1, 1, 0, 1]
})

X = data[["age", "income", "loan"]]
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Пайплайн: спочатку скейлінг, потім логістична регресія
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),  # масштабує всі числові ознаки
    ("clf", LogisticRegression(max_iter=1000))  # сама модель
])

# Параметри для перебору
param_grid = {
    "clf__max_iter": [100, 500, 1000],
    "clf__C": [0.1, 1, 10]
}

# GridSearchCV
grid = RandomizedSearchCV(pipeline, param_grid, cv=3, scoring="accuracy")
grid.fit(X_train, y_train)

print("Найкращі параметри:", grid.best_params_)
print("Точність на тесті:", grid.score(X_test, y_test))

y_pred = grid.predict(X_test)
print("Report:\n", classification_report(y_test, y_pred))

