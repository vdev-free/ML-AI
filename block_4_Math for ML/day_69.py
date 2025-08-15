import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

df = pd.DataFrame({
   "hours_studied": [8, 1, 10, 2, 6, 5, 3, 12],
    "attendance":    [90, 50, 95, 40, 85, 80, 60, 100],
    "late":          [0, 5, 0, 7, 1, 2, 4, 0],
    "diploma":       [1, 0, 1, 0, 1, 1, 0, 1] 
})

X = df[["hours_studied", "attendance", "late"]].values
y = df["diploma"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler();
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)

print('Результат класифікації:')
print(classification_report(y_test, y_pred))

X_new = [[4, 88, 5]]

X_new_sc = scaler.transform(X_new)

print('Результат передбачення:', model.predict(X_new_sc))

