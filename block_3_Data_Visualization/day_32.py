import pandas as pd
import matplotlib.pyplot as plt

cleaned = pd.DataFrame({
    'product': ['apple', 'banana', 'banana', 'apple', 'banana', 'orange', 'orange', 'apple', 'banana'],
    'region': ['North', 'North', 'South', 'South', 'North', 'North', 'South', 'North', 'South'],
    'sales': [120, 150, 200, 180, 180, 160, 140, 100, 190]
})

# sales_by_product = cleaned.groupby('product')['sales'].sum()
# sales_by_region = cleaned.groupby('region')['sales'].sum()

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# sales_by_product.plot(
#     kind='bar',
#     ax=axes[0],
#     title='Продажі по продуктах',
#     color='green'
# )

# axes[0].set_xlabel('Продукт')
# axes[0].set_ylabel('Продажі')

# sales_by_region.plot(
#     kind='pie',
#     ax=axes[1],
#     autopct='%1.1f%%',
#     title='Частка продажів по регіонах'
# )

# axes[1].set_ylabel('')

# plt.tight_layout()
# plt.show()

# fig.savefig('sales_report.png')

group_by_region = cleaned.groupby('region')['sales'].sum()
group_by_product = cleaned.groupby('product')['sales'].sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

group_by_region.plot(
    kind='bar',
    ax=axes[0],
    title='Продажі по продуктах',
    color='green'
)

axes[0].set_xlabel('Продукт')
axes[0].set_ylabel('Продажі')

group_by_product.plot(
   kind='pie',
   ax=axes[1],
   autopct='%1.1f%%',
   title='Продажі по регіонах'
)

axes[1].set_ylabel('')

plt.tight_layout()
plt.style.use('ggplot')
plt.show()

fig.savefig('new.png')
