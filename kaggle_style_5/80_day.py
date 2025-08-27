import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.DataFrame({
     "age": [25, 40, np.nan, 23, 35, 29, 45, np.nan],
    "income": [3000, 8000, 10000, np.nan, 6000, 4000, 9000, 12000],
    "loan_amount": [500, 2000, np.nan, 300, 1500, 1000, 2500, 4000],
    "education": ["high", "uni", "uni", "school", np.nan, "uni", "uni", "high"],
    "default": [0, 0, 1, 0, 0, 1, 1, 0] 
})

X = data[["age", "income", "loan_amount", "education"]]
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

numeric = ["age", "income", "loan_amount"]
categorical = ["education"]

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(transformers=[
    ('nun', numeric_pipeline, numeric),
    ('cat',categorical_pipeline ,categorical)
])

model = Pipeline(steps=[
    ('prep', preprocess),
    ('clf', RandomForestClassifier(n_estimators=150, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('report', classification_report(y_test, y_pred, digits=3))

new_client = pd.DataFrame([{
    "age": 32,
    "income": 7000,
    "loan_amount": 1200,
    "education": "uni"
}])

print('new_client', model.predict(new_client))
print('proba', model.predict_proba(new_client)[:, 1])