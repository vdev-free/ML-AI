import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Дані: студенти і результат здачі екзамену
df = pd.DataFrame({
    "hours_studied": [5, 1, 10, 2, 7, 6, 3, 12],
    "attendance":    [90, 50, 95, 40, 85, 80, 60, 100],
    "pass_exam":     [1, 0, 1, 0, 1, 1, 0, 1]
})

X = df[["hours_studied", "attendance"]].values
y = df["pass_exam"].values

# 1. Розбиваємо на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Масштабуємо
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Створюємо SVM модель
model = SVC(kernel="rbf", C=1.0)
model.fit(X_train_scaled, y_train)

# 4. Прогноз
y_pred = model.predict(X_test_scaled)

# 5. Оцінка
print("Звіт:")
print(classification_report(y_test, y_pred))

# 6. Передбачення нового студента
new_student = [[4, 70]]  # годин навчання, відвідуваність
new_student_scaled = scaler.transform(new_student)
print("Прогноз для нового студента:", model.predict(new_student_scaled))
