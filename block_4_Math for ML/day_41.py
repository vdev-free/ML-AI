import pandas as pd
import matplotlib.pyplot as plt

# 1. Створюємо DataFrame з даними
data = pd.DataFrame({
    'sales': [100, 110, 130, 120, 140, 115, 300, 125, 135, 118, 122, 117]
})

mean_val = data['sales'].mean()
median_val = data['sales'].median()
std_val = data['sales'].std()

plt.figure(figsize=(8, 4))

plt.bar(data.index, data['sales'], label='Продажі')
plt.axhline(mean_val, label='Середній показник продаж')
plt.axhline(mean_val + std_val, label='Верхній показник')
plt.axhline(mean_val - std_val, label='Нижній показник')

plt.title('Аналіз продай 12 днів')
plt.xlabel('Дні')
plt.ylabel('Продажі')
plt.legend()
plt.tight_layout()
plt.show()
