import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    'product': ['apple', 'banana', 'banana', 'apple', 'orange', 'orange', 'banana', 'apple'],
    'region': ['North', 'North', 'South', 'South', 'North', 'South', 'South', 'North'],
    'sales': [120, 150, 200, 130, 160, 140, 180, 125],
    'profit': [20, 30, 50, 25, 35, 32, 45, 22]
})

numeric_df = df[['sales', 'profit']]

corr = numeric_df.corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='YlGnBu',
    fmt='.2f',
    linewidths=0.5,
    square=True
)

sns.pairplot(
    data=df,
    vars=['sales', 'profit'],
    hue='region',
    kind='scatter',
    plot_kws={'alpha':0.7}
)


plt.suptitle('Взаємозв’язки між змінними по регіонах', y=1.02)
plt.show()