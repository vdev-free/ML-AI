# from sklearn.linear_model import LinearRegression
# import numpy as np

# Дані: площа і кількість кімнат
# X = np.array([
#     [50, 2],
#     [60, 3],
#     [80, 3],
#     [100, 4]
# ])
# y = np.array([100000, 120000, 160000, 200000])

# Модель
# model = LinearRegression()
# model.fit(X, y)

# print("Ваги:", model.coef_)
# print("Зсув:", model.intercept_)

# Прогноз
# pred = model.predict([[70, 3]])
# print(f"Ціна квартири 70м² / 3 кімнати: {pred[0]:.2f}")
# import matplotlib.pyplot as plt

# X = np.array([[50], [60], [80], [100]])
# y = np.array([100000, 120000, 160000, 200000])

# model = LinearRegression()
# model.fit(X, y)

# plt.scatter(X, y, color='blue', label='Дані')
# plt.plot(X, model.predict(X), color='red', label='Лінія регресії')
# plt.xlabel('Площа')
# plt.ylabel('Ціна')
# plt.title('Лінійна регресія: Площа → Ціна')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import 

X=np.array([
    [8, 2, 5],
    [16, 4, 8],
    [8, 4, 6],
    [32, 8, 10],
    [16, 6, 9]
])

y=np.array([600, 1200, 900, 2000, 1600])

model=LinearRegression()
model.fit(X,y)

new_pc = np.array([[12,4,7]]) 

prediction = model.predict(new_pc)
print(f'ціна нового PC {prediction[0]:.2f}')

score = model.score(X,y)
print(f'R² score: {score:.2f}')