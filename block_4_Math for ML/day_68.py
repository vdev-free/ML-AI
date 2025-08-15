import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# df = pd.DataFrame({
#     "monthly_spend":   [40, 25, 70, 15, 90, 55, 20, 110, 35, 60],
#     "months_active":   [24,  6, 36,  3, 48, 30,  5,  60, 10, 28],
#     "support_tickets": [ 0,  3,  1,  5,  0,  2,  4,   0,  3,  1],
#     "churn":           [ 0,  1,  0,  1,  0,  0,  1,   0,  1,  0]
# })

# X = df[["monthly_spend", "months_active", "support_tickets"]].values
# y = df[["churn"]].values

# X_train, X_test, y_train, y_test = train_test_split(
#  X, y, test_size=0.3, random_state=42, stratify=y   
# )

# scaler = StandardScaler()
# X_train_sc = scaler.fit_transform(X_train)
# X_test_sc = scaler.transform(X_test)

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_sc, y_train)

# y_prob = model.predict_proba(X_test_sc)[:,1]
# y_pred = (y_prob >= 0.5).astype(int)

# print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# print("\nReport:\n", classification_report(y_test, y_pred, digits=3))

# print("ROC-AUC:", roc_auc_score(y_test, y_prob))

df = pd.DataFrame({
    "hours_studied": [10, 2, 15, 1, 25, 12, 3, 30, 5, 18],
    "courses_started": [2, 1, 3, 1, 4, 2, 1, 5, 1, 3],
    "feedbacks_given": [1, 0, 2, 0, 3, 1, 0, 4, 1, 2],
    "churn": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
})

X = df[["hours_studied", "courses_started", "feedbacks_given"]].values
y = df["churn"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_sc, y_train)

y_prob = model.predict_proba(X_test_sc)[:,1]
y_pred = (y_prob>=0.5).astype(int)

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

