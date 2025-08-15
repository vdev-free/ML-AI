import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'product': ['apple', 'banana', 'banana', 'apple', 'banana', 'orange', 'orange', 'apple', 'banana'],
    'region': ['North', 'North', 'South', 'South', 'North', 'North', 'South', 'North', 'South'],
    'sales': [120, 150, 200, 180, 180, 160, 140, 100, 190]
})

# sns.barplot(data=df, x='product', y='sales', estimator=sum)
# plt.title("Загальні продажі по продуктах")
# plt.show()

# sns.countplot(data=df, x='region')
# plt.title("Кількість продажів у кожному регіоні (без суми)")
# plt.show()

# sns.boxplot(data=df, x='region', y='sales')
# plt.title('Розподіл продажів по регіонах')
# plt.show()

sns.barplot(data=df, x='product', y='sales', estimator=sum)
plt.title('product sales')
plt.show()

sns.boxplot(data=df, x='region', y='sales')
plt.title('product sales')
plt.show()