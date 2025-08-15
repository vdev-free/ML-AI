import numpy as np

X = np.array([
    [2.0, 2018, 40],
    [1.6, 2016, 70],
    [3.0, 2020, 20]
])

X[:, 0] /= 3.0
X[:, 1] -= 2015
X[:, 1] /= 10
X[:, 2] /= 100.0

y = np.array([15000, 12000, 25000])

w = np.random.randn(3)
b = np.random.randn()

learning_rate = 0.01

for epoch in range(1000):
    predictions = X @ w + b
    errors = predictions - y

    grad_w = 2 * X.T @ errors / len(X)
    grad_b = 2 * np.mean(errors)

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

print("Фінальні ваги:", w)
print("Фінальний зсув:", b)

new_car = np.array([2.5, 2019, 30])
new_car[0] /= 3.0
new_car[1] -= 2015
new_car[1] /= 10
new_car[2] /= 100.0

predicted_price = new_car @ w + b
print(f"Прогнозована ціна нового авто: {predicted_price:.2f} $")