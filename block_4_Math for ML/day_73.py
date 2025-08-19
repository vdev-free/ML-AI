import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

data = pd.DataFrame({
    "hours_studied": [5, 1, 10, 2, 7, 6, 3, 12],
    "attendance":    [90, 50, 95, 40, 85, 80, 60, 100],
    "internet_usage": [2, 8, 1, 10, 3, 4, 6, 1],
    "pass_exam":     [1, 0, 1, 0, 1, 1, 0, 1]
})

X = data[[ "hours_studied", "attendance", "internet_usage"]].values
y = data[ "pass_exam"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42 )

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)

print('classification_report', classification_report(y_test, y_pred))

new_student = [[3, 70, 7]]

print(model.predict(scaler.transform(new_student)))