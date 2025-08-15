import pandas as pd

# df = pd.DataFrame({
#     'team': ['a', 'a', 'b', 'b'],
#     'points': [10, 20, 30, 40],
#     'goals': [1, 2, 3, 4]
# })

# grouped = df.groupby('team').agg({
#     'points': ['sum', 'mean'],
#     'goals': ['min', 'max']
# })

# print(df[['points', 'goals']].agg(['sum', 'mean', 'max']))

# grouped = df.groupby('team').agg(
#     total_points=('points', 'sum'),
#     avg_points=('points', 'mean'),
#     max_goals=('goals', 'max')
# )

# print(grouped)

df = pd.DataFrame({
    'category': ['A', 'A', "B", 'B', 'C'],
    'sales': [100, 200, 150, 300, 400],
    'profit': [20, 30, 15, 40, 60]
})

groupby = df.groupby('category').agg({
'sales': ['sum', 'mean'],
'profit': ['min', 'max']
})

name_groupby = df.groupby('category').agg(
    total_sales_sum=('sales', 'sum'),
    total_sales_mean=('sales', 'mean'),
    min_profit=('profit', 'min'),
    max_profit=('profit', 'max')
)

name_groupby.to_csv('name_groupby.csv')

print(name_groupby)