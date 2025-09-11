import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

num_features = ['Age', 'Fare', 'SibSp', "Parch"]
cat_features = ['Pclass', 'Sex', 'Embarked']

X = data[num_features + cat_features]
y = data['Survived']

X_test_final = test[num_features + cat_features]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

num_transformer_tree = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler(with_mean=True, with_std=True))
])

try:
    onehot_dense = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    onehot_dense = OneHotEncoder(handle_unknown='ignore', sparse=False)

cat_transformer_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot_dense),
])

preprocessor_tree = ColumnTransformer(
    transformers=[
        ('num', num_transformer_tree, num_features),
        ('cat', cat_transformer_tree, cat_features)
    ],
    remainder='drop'
)

rf_base_clf = Pipeline(steps=[
    ('prep', preprocessor_tree),
    ('clf', RandomForestClassifier(
        n_estimators=400,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

gb_clf = Pipeline(steps=[
    ('prep', preprocessor_tree),
    ('clf', GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

hgb_clf = Pipeline(steps=[
    ('prep', preprocessor_tree),
    ('clf', HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.45,
        max_depth=3,
        random_state=42,
        l2_regularization=0.0
    ))
])

lr_clf = Pipeline(steps=[
    ('prep', preprocessor_tree),
    ('clf', LogisticRegression(
        max_iter=1000,
        random_state=42,
    ))
])

voting_clf = VotingClassifier(
    estimators=[
        ("rf_base", rf_base_clf),
        ("gb", gb_clf),
        ("hgb", hgb_clf),
        ("lr", lr_clf)
    ],
    voting='soft'
)

voting_clf.fit(X, y)

pred_y = voting_clf.predict(X_test_final)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': pred_y
})

submission.to_csv('submission.csv', index=False)

print("✅ Файл 'submission.csv' збережено.")
print(submission.head())
print("Розмір:", submission.shape)
