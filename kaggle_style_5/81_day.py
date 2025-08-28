import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score

data = pd.DataFrame({
    "age": [25, 32, 40, 28, np.nan, 45, 36, 29, 50, 38],
    "experience_years": [1, 5, 10, 3, 7, 15, np.nan, 4, 20, 12],
    "education": ["bachelor", "master", "phd", "bachelor", "master", np.nan, "bachelor", "bachelor", "phd", "master"],
    "english_level": ["basic", "advanced", "fluent", "intermediate", "advanced", "fluent", "basic", "intermediate", "fluent", np.nan],
    "got_job": [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
})

X = data[["age", "experience_years", "education", "english_level"]]
y = data["got_job"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

numeric = ["age", "experience_years"]
categorical = ["education", "english_level"]

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(transformers=[
('num', numeric_pipeline, numeric),
('cat', categorical_pipeline, categorical)
])

model = Pipeline(steps=[
    ('prep', preprocess),
    ('mod', LogisticRegression())
])

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("classification_report:\n", classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


new_candidate = pd.DataFrame([{
    "age": 30,
    "experience_years": 6,
    "education": "master",
    "english_level": "advanced"
}])

prob = model.predict_proba(new_candidate)[:, 1][0]
pred = int(prob >= 0.5)

print(f"\nЙмовірність отримати оффер: {prob:.2f}")
print("Клас (0/1):", pred)




