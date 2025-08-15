import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
    'visits': [120, 135, 150, 160, 170, 185, 190, 200, 210, 220, 215, 210, 205, 200, 195],
    'conversions': [10, 12, 14, 13, 15, 17, 18, 20, 21, 22, 20, 19, 18, 17, 15]
})

df['conversion_rate'] = df['conversions'] / df['visits']

plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['visits'], label='Відвідування', marker='o')
plt.plot(df['date'], df['conversions'], label='Конверсії', marker='s')
plt.plot(df['date'], df['conversion_rate'], label='Коефіцієнт конверсії', linestyle='--', marker='x')

plt.title('Динаміка відвідувань, конверсій і коефіцієнта конверсії')
plt.xlabel('Дата')
plt.ylabel('Значення')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()