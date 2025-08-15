import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({
    'product': ['apple', 'banana', 'banana', 'apple', 'banana', 'orange', 'orange'],
    'region': ['North', 'North', 'South', 'South', 'North', 'North', 'South'],
    'sales': [120, 150, 200, 180, 180, 160, 140]
})

# sales_by_product = data.groupby('product')['sales'].sum()
# sales_by_product.plot(
#     kind='bar',
#     title='Продажі по продуктах',
#     ylabel='Продажі',
#     xlabel='Продукт',
#     color='orange')

# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# sales_by_region = data.groupby('region')['sales'].sum()


# sales_by_region.plot(
#     kind='pie',
#     autopct='%1.1f%%',     
#     title='Частка продажів по регіонах'
# )

# plt.ylabel('')  
# plt.show()

# sales_by_region = data.groupby('product')['sales'].sum()

pivot = pd.pivot_table(
    data,
    values='sales',
    index='product',
    columns='region',
    aggfunc='sum'
)

pivot.plot(
    kind='bar',
    stacked=True,
    title='Продажі по продуктах і регіонах'
)

plt.ylabel('Обсяг продажів')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.savefig('chart.png')


# cleaned = pd.DataFrame({
#     'product': ['apple', 'banana', 'banana', 'apple', 'banana', 'orange', 'orange', 'apple', 'banana'],
#     'region': ['North', 'North', 'South', 'South', 'North', 'North', 'South', 'North', 'South'],
#     'sales': [120, 150, 200, 180, 180, 160, 140, 100, 190]
# })

# cleaned_groupby_product = cleaned.groupby('product')['sales'].sum()
# cleaned_groupby_product.plot(
#     kind='bar',
#     title='Продажі по продуктах',
#     xlabel='product',
#     ylabel='sales',
# )

# cleaned_groupby_region = cleaned.groupby('region')['sales'].sum()
# cleaned_groupby_region.plot(
#     kind='pie',
#     autopct='%1.1f%%', 
#     title='Частка регіонів',
# )

# plt.ylabel('')
# plt.tight_layout()
