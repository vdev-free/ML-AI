import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x + 2)**2 + 1

def df(x):
    return 2*(x + 2)

x = 0

learning_rate = 0.1
history = [x]

for i in range(15):
    grad = df(x)
    x = x - learning_rate * grad
    history.append(x)

xs = np.linspace(-1, 6, 100)
ys = f(xs)

plt.plot(xs, ys, label="Функція помилки")
plt.scatter(history, [f(i) for i in history], color='red', label='Кроки')
plt.title("Градієнтний спуск")
plt.grid(True)
plt.legend()
plt.show()

print(f"Наближене мінімум: x = {x:.2f}")