import numpy as np

X = np.array([
    [10,0.5,1000],
    [5,0.3,500],
    [20,1.0,2000]
])

X[:, 0] /= 20.0    # ділимо вагу на максимум 20 кг
X[:, 1] /= 1.0     # об'єм вже в межах 0–1, можна залишити
X[:, 2] /= 2000.0  # ділимо ціну на максимум 2000 $

y = np.array([35, 20, 60])

w = np.random.randn(3)
b = np.random.randn()

learning_rate = 0.0001
for epoch in range(1000):
    predictions = X @ w + b
    errors = predictions - y
    loss = np.mean(errors ** 2)

    grad_w = 2 * X.T @ errors / len(X)
    grad_b = 2 * np.mean(errors)

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    if epoch % 100 == 0:
        print(f"Крок {epoch}: loss = {loss:.2f}, ваги = {w}, зсув = {b:.2f}")

        print("Фінальні ваги:", w)
        print("Фінальний зсув:", b)

