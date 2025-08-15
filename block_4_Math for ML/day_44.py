import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'weight': [60, 65, 62, 72, 68, 63, 80, 76, 74, 82],
    'hight': [165, 170, 168, 175, 172, 169, 180, 178, 176, 181],
})

cov = data['weight'].cov(data['hight'])
print(f'Коваріація: {cov:.2f}')

corr = data['weight'].corr(data['hight'])
print(f'Кореляція: {corr:.2f}')

plt.scatter(data['weight'], data['hight'])
plt.title('Зв\'язок між вагою та зростом')
plt.xlabel('weight')
plt.ylabel('hight')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()