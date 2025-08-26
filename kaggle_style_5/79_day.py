import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    "age": [25, 40, 50, 23, 35, 29, 45, 60, 28, 38],
    "income": [3000, 8000, 10000, 2500, 6000, 4000, 9000, 12000, 3500, 7200],
    "loan_amount": [500, 2000, 3000, 300, 1500, 1000, 2500, 4000, 700, 2200],
    "education": ["high", "uni", "uni", "school", "high", "uni", "uni", "high", "school", "uni"],
    "job_type": ["worker", "manager", "manager", "worker", "worker", "worker", "manager", "retired", "worker", "manager"],
    "default": [0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
})


X = df[["age", "income", "loan_amount", "education", "job_type"]]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

numeric = ["age", "income", "loan_amount"]
categorical = ["education", "job_type"]

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),  numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ]
)

model = Pipeline(steps=[
    ('prep', preprocess),
    ('clf', LogisticRegression(max_iter=200))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

new_client = pd.DataFrame([{
    "age": 32,
    "income": 5500,
    "loan_amount": 1200,
    "education": "uni",
    "job_type": "worker"
}])

new_predict = model.predict(new_client)
print("Прогноз:", new_predict)
 

new_predict = model.predict(new_client)

print(new_predict)

