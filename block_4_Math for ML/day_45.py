import numpy as np
import pandas as pd

# A = np.array([
#     [55,2,2],
#     [85,5,3],
#     [70,3,2]
# ])

# b = np.array([65_000, 101_000, 80_000])

# solution = np.linalg.solve(A, b)

# print(f'ціна кв.м: {solution[0]:.2f}')
# print(f'доплата за поверх: {solution[1]:.2f}')
# print(f'доплата за кімнату: {solution[2]:.2f}')

# prise_fl = (60*solution[0])+(solution[1]*4)+solution[2]*2

# print(prise_fl)

# A = np.array([
#     [1,0,0],
#     [3,1,1],
#     [5,2,2]
# ])

# b = np.array([1000, 2000, 3300])

# solution = np.linalg.solve(A,b)

# print(f'Вага років досвіду: {solution[0]:.2f}')
# print(f'Вага рівня англ мови: {solution[1]:.2f}')
# print(f'Вага к-ті сертифікатів: {solution[2]:.2f}')

# salary_junior_py = solution[0]*2+solution[1]*1+solution[2]*1
# print('salary_junior_py', salary_junior_py)

data = pd.DataFrame({
    'years': [1,3,5],
    'english': [0,1,2],
    'certs': [0,1,2],
    'salary': [1000,2000,3300]
})

X = data[['years', 'english', 'certs']]
y = data['salary']

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X,y)

print(f"Вага років досвіду: {model.coef_[0]:.2f}")
print("Вага рівня англ мови:", model.coef_[1])
print("Вага сертифікатів:", model.coef_[2])
print("Зсув (intercept):", model.intercept_)

pred_salary = model.predict([[2,1,1]])
print(f"Прогнозована зарплата: {pred_salary[0]:.2f} $")

