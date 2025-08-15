import pandas as pd

data = pd.DataFrame({
    'product': ['apple', 'Banana', 'banana', 'Apples', 'banan', 'orange', 'Orange', None],
    'region': ['North', 'North', 'South', 'South', 'North', 'North', 'South', 'South'],
    'sales': [120, 150, 200, None, 180, 160, 140, 170]
})

cleaned = data.dropna(subset=['product'])
cleaned['sales'] = cleaned['sales'].fillna(0)
cleaned['product'] = cleaned['product'].str.lower().str.strip()
cleaned['product'] = cleaned['product'].replace({
    'apples': 'apple',
    'banan': 'banana'
})

sales_by_product = cleaned.groupby('product')['sales'].sum()

pivot = pd.pivot_table(
    cleaned,
    values='sales',
    index='product',
    columns='region',
    aggfunc='sum'
)

# print(sales_by_product)

best_product = sales_by_product.idxmax()
best_product_sales = sales_by_product.max()
sales_by_region = cleaned.groupby('region')['sales'].sum()
best_region = sales_by_region.idxmax()
best_region_sales = sales_by_region.max()




print(pivot)
print(pivot.stack().idxmax())
print(f'найбільше продаж було у {pivot.stack().idxmax()[1]} товару {pivot.stack().idxmax()[0]} на суму {pivot.values.max()}')