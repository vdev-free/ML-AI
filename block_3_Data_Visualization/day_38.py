# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.DataFrame({
#     'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
#     'sales': [100, 120, 130, 125, 145, 160, 150, 170, 165, 180]
# })

# df.set_index('date', inplace=True)

# df['rolling_mean'] = df['sales'].rolling(window=3).mean()

# plt.plot(df['date'], df['sales'], label='Щоденні продажі', marker='o')
# plt.plot(df['date'], df['rolling_mean'], label='Згладжені (3 дні)', linestyle='--', marker='x')

# plt.title('Продажі за часом: оригінальні vs згладжені')
# plt.xlabel('Дата')
# plt.ylabel('Продажі')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
    'shop_a_sales': [100, 110, 120, 130, 140, 150, 160, 170, 160, 150, 140, 130, 120, 110, 100],
    'shop_b_sales': [90, 100, 95, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]
})

df['shop_a_smooth'] = df['shop_a_sales'].rolling(window=3).mean()
df['shop_b_smooth'] = df['shop_b_sales'].rolling(window=3).mean()

plt.plot(df['date'], df['shop_a_sales'], label='Shop A - Оригінал', color='blue', marker='o')
plt.plot(df['date'], df['shop_b_sales'], label='Shop B - Оригінал', color='green', marker='s')

plt.plot(df['date'], df['shop_a_smooth'], label='Shop A - Згладжено', color='cyan', linestyle='--', marker='x')
plt.plot(df['date'], df['shop_b_smooth'], label='Shop B - Згладжено', color='lime', linestyle='--', marker='d')

plt.title('Продажі магазинів з 3-денним згладженням')
plt.xlabel('Дата')
plt.ylabel('Продажі')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()