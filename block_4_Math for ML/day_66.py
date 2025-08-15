# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# import numpy as np

# X = np.array([
#     [1],
#     [2],
#     [3],
#     [4],
#     [5]
# ])

# y = np.array([10,17,30,45,100])

# linear_model = LinearRegression()
# linear_model.fit(X,y)

# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)

# model_poly = LinearRegression()
# model_poly.fit(X_poly, y)

# x_range = np.linspace(1,5,100).reshape(-1,1)
# x_range_poly = poly.transform

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# # 1. Дані (тренування → час на 5 км)
# X = np.array([[1], [2], [3], [4], [5], [6], [7]])
# y = np.array([30, 25, 20, 18, 19, 22, 26])

# # 2. Лінійна регресія (для порівняння)
# lin_model = LinearRegression()
# lin_model.fit(X, y)

# # 3. Поліноміальна регресія (додаємо X²)
# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)

# poly_model = LinearRegression()
# poly_model.fit(X_poly, y)

# # 4. Побудуємо гладку лінію для графіка
# X_range = np.linspace(1, 7, 100).reshape(-1, 1)
# X_range_poly = poly.transform(X_range)

# # 5. Графік
# plt.scatter(X, y, color='blue', label='Дані')
# plt.plot(X_range, lin_model.predict(X_range), color='green', label='Лінійна')
# plt.plot(X_range, poly_model.predict(X_range_poly), color='red', label='Поліноміальна (парабола)')
# plt.xlabel('Тренування на тиждень')
# plt.ylabel('Час (хв)')
# plt.title('Поліноміальна регресія у спорті')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# X = np.array([[0], [1], [2], [3], [4], [5], [6]])
# y = np.array([30,50,65,75,70,60,45])

# linear_model = LinearRegression()
# linear_model.fit(X,y)

# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)

# poly_model = LinearRegression()
# poly_model.fit(X_poly,y)

# X_range = np.linspace(0, 6, 100).reshape(-1, 1)
# X_range_poly = poly.transform(X_range)

# plt.scatter(X, y, color='blue', label='Дані')
# plt.plot(X_range, linear_model.predict(X_range), color='green', label='Лінійна')
# plt.plot(X_range, poly_model.predict(X_range_poly), color='red', label='Поліноміальна (парабола)')
# plt.xlabel('Кава (чашки/день)')
# plt.ylabel('Продуктивність (бали)')
# plt.title('ідеальна кількість кави для продуктивності')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# X_pred = np.array([[2.5], [5.5]])
# X_pred_ploy = poly.transform(X_pred)

# pred_linear = linear_model.predict(X_pred)
# pred_poly = poly_model.predict(X_pred_ploy)

# print(f'Лінійна регресія: при 2.5 чашки = {pred_linear[0]:.1f}, при 5.5 = {pred_linear[1]:.1f}')
# print(f'Поліноміальна: при 2.5 чашки = {pred_poly[0]:.1f}, при 5.5 = {pred_poly[1]:.1f}')


# from sklearn.linear_model import LinearRegression, Ridge, Lasso

# X = [[50, 2, 3], [70, 3, 2], [100, 4, 5], [120, 5, 4]]
# y = [100000, 150000, 200000, 250000]

# # Без регуляризації
# lr = LinearRegression().fit(X, y)

# # Ridge
# ridge = Ridge(alpha=1.0).fit(X, y)

# # Lasso
# lasso = Lasso(alpha=0.1).fit(X, y)

# print("Linear:", lr.coef_)
# print("Ridge:", ridge.coef_)
# print("Lasso:", lasso.coef_)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np

X = np.array([
    [50, 2, 1],
    [65, 3, 2],
    [80, 5, 3],
    [120, 10, 4],
    [100, 7, 3]
])

y = np.array([70000, 85000, 105000, 180000, 150000])

lg = LinearRegression().fit(X, y)
r = Ridge(alpha=1.0).fit(X, y)
l = Lasso(alpha=1.0).fit(X, y)

print('Linear:', lg.coef_)
print('Ridge:', r.coef_)
print('Lasso:', l.coef_)

new_flat = np.array([[90, 4, 2]])

print('new_flat_price_Linear:', lg.predict(new_flat))
print('new_flat_price_Ridge:', r.predict(new_flat))
print('new_flat_price_Lasso:', l.predict(new_flat))

