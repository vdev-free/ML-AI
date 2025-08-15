import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'experience_years': [1, 3, 5, 4, 2],
    'english_level':    [0, 1, 2, 2, 1],
    'certs':            [0, 1, 2, 1, 0],
    'location':         [1, 2, 3, 3, 2],
    'salary':           [1200, 2000, 4500, 4200, 1700]
})

X = data[['experience_years', 'english_level', 'certs', 'location']]
y = data['salary']

model = LinearRegression()
model.fit(X,y)

new_candidate = np.array([[2, 1, 1, 2]])
predicted_salary = model.predict(new_candidate)[0]
print(f"Прогнозована зарплата: {predicted_salary:.2f} $")

similar = data[data['experience_years'] <= 3]['salary']
mean_similar = similar.mean()

if predicted_salary > 1.3 * mean_similar or predicted_salary < 0.7 * mean_similar:
    print("⚠️ Можливий аномальний запит по зарплаті (outlier)")
else:
    print("✅ Зарплатне очікування в межах норми")

