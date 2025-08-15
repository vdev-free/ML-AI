import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.DataFrame({
#     'height': [160, 170, 175, 180, 185, 190, 195],
#     'weight': [55, 65, 68, 75, 85, 90, 95],
#     'age':    [25, 32, 40, 50, 28, 33, 48]
# })

# corr = df.corr()

# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title('Кореляційна матриця')
# plt.show()

# df = pd.DataFrame({
#     'height': [160, 170, 180, 190],
#     'weight': [55, 65, 75, 85],
#     'age':    [60, 50, 40, 30]
# })

# cleaned = pd.DataFrame({
#     'product': ['apple', 'banana', 'banana', 'apple', 'orange', 'orange'],
#     'region': ['North', 'North', 'South', 'South', 'North', 'South'],
#     'sales': [120, 150, 200, 130, 160, 140],
#     'profit': [20, 30, 50, 25, 35, 32]
# })

# sns.pairplot(cleaned, hue='region')
# plt.show()


df = pd.DataFrame({
    'product': ['apple', 'banana', 'banana', 'apple', 'orange', 'orange', 'banana', 'apple'],
    'region': ['North', 'North', 'South', 'South', 'North', 'South', 'South', 'North'],
    'sales': [120, 150, 200, 130, 160, 140, 180, 125],
    'profit': [20, 30, 50, 25, 35, 32, 45, 22]
})

sns.pairplot(df, hue='region')

corr = df[['sales', 'profit']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Кореляція між числовими змінними')
plt.show()

