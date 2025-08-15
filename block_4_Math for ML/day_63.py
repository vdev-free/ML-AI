import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ✅ Візьми будь-які дані (наприклад, вага людини → час пробіжки).
# ✅ Розділи дані на train/test через train_test_split.
# ✅ Створи Pipeline з масштабуванням і лінійною регресією.
# ✅ Виведи якість моделі.

X = np.array([
    [65, 2, 2],
    [70, 2, 5],
    [95, 3, 4],
    [100, 4, 2],
    [110, 4, 1]
])

y = np.array([70000, 85000, 110000, 115000, 120000])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

pipeline = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

new = np.array([
    [65, 2, 3],
])



pipeline.fit(X_train, y_train)
score_pipeline = pipeline.score(X_test, y_test)

prediction = pipeline.predict(new)

print(f'Якість моделі: {score_pipeline:.2f}')
print(f'Прогнозована ціна для квартири 65м², 2 поверх, 2 кімнати: {prediction[0]:.2f} грн')
