import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data = np.random.normal(loc=0, scale=1, size=1000)

# plt.hist(data, bins=30, color='skyblue', alpha=0.7, label='Гістограма')
# plt.title('Нормальний розподіл (mu=0, sigma=1)')
# plt.xlabel=('Значення')
# plt.ylabel=('Частота')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# n=10
# p=0.5
# size =1000

# data = np.random.binomial(n, p, size)

# plt.hist(data, bins=range(n+2), color='orange', alpha=0.7, label='Гістограма')
# plt.title('Біноміальний розподіл (n=10, p=0.5)')
# plt.xlabel('Кількість "успіхів" (орлів за 10 підкидань)')
# plt.ylabel('Частота')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

n=20
p=0.3
size = 1000

data = np.random.binomial(n, p, size)

plt.hist(data, color='red', label='Гістограма')
plt.xlabel('К-ть успіхів')
plt.ylabel('Частота')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

print("Найчастіша кількість кліків:", pd.Series(data).mode()[0])
prob_8 = np.sum(data == 8) / size
print("Ймовірність рівно 8 кліків:", prob_8)