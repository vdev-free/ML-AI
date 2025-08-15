import pandas as pd
pd.set_option('display.max_columns', None)  # показати всі колонки
pd.set_option('display.width', None)        # не обмежувати ширину виводу
pd.set_option('display.max_colwidth', None) # повна ширина для вмісту клітинок


# df = ({
#     'category': ['a', 'a', 'b', 'b', 'c', 'c'],
#     'region': ['North', 'South', 'North', 'South', 'North', 'South'],
#     'sales': [100, 150, 200, 250, 300, 350]
# })

# data = pd.DataFrame(df)

# td =  pd.pivot_table(
#     data,               
#     values='sales',        
#     index=['category', 'region'],    
#     # columns='region',     
#     aggfunc=['sum', 'mean']         
# )

# print(td)

df = pd.DataFrame({
    'product': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C'],
    'region': ['North', 'South', 'North', 'South', 'North', 'South', 'North', 'South', 'North'],
    'sales': [100, 200, 150, 300, 250, 350, 120, 280, 270],
    'orders': [5, 8, 6, 10, 9, 11, 4, 9, 7]
})

nt = pd.pivot_table(
    df,
    index='product',
    columns='region',
    values=['sales', 'orders'],
    aggfunc=['count', 'mean', 'sum']
)

nt.to_csv('product.csv')

print(nt)