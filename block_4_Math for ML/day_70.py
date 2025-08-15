import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.DataFrame({
    "hours_studied": [2, 4, 6, 1, 8, 10, 3, 7, 5, 9, 11, 12],
    "attendance":    [50, 60, 70, 40, 80, 95, 55, 85, 75, 90, 96, 98],
    "quizzes_avg":   [40, 55, 65, 30, 75, 85, 50, 70, 68, 80, 88, 92],
    "pass_exam":     [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]
})

X = df[["hours_studied", "attendance", "quizzes_avg"]].values
y = df["pass_exam"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Звіт:\n", classification_report(y_test, y_pred))

candidates = [
    [3, 60, 55],   # мало вчився, середня відвідуваність, квізи посередні
    [8, 90, 80],   # добре вчився, висока відвідуваність, хороші квізи
    [6, 50, 40],   # норм години, але низькі квізи
]

print(model.predict(candidates))