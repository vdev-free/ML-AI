import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'age':[27,30,40,55,18,37],
    'income':[1000, 2000, 5000, 1500, 700, 4000],
    'has_debt':[0, 1, 1, 0, 0, 1]
})

X = data[['age','income']].values
y = data['has_debt'].values

X_trayn, X_test, y_trayn, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X_trayn_scaler = sc.fit_transform(X_trayn)
X_test_scaler = sc.transform(X_test)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_trayn_scaler, y_trayn)

y_prediction = model.predict(X_test_scaler)

print('report', classification_report(y_test, y_prediction))
print('prediction', model.predict(sc.transform([[32, 2500]])))