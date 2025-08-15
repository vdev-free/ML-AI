import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 4*x + 3

def df(x):
    return 2*x - 4

x = 5
learning_rate = 0.1

x_values = [x]

for _ in range(10):
    grad = df(x)
    x = x - learning_rate * grad
    x_values.append(x)

xs = np.linspace(-6, 8, 100)
ys = f(xs)

plt.plot(xs, ys, label='x**2 - 4*x + 3')
plt.scatter(x_values, [f(i) for i in x_values], color='red', label='Кроки градієнта')
plt.legend()
plt.grid(True)
plt.title('Градієнтний спуск')
plt.show()
