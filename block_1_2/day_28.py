import pandas as pd

# df = pd.DataFrame({
#     'date': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20'],
#     'sales': [100, 150, 200, 130, 170]
# })

# df['date'] = pd.to_datetime(df['date'])
# # print(df['date'].dt.day_name())

# filtered = df[df['date'] > '2023-01-10']
# # print(filtered)

# df_sort = df.sort_values('date')
# # print(df_sort)

# df.set_index('date', inplace=True)

# print(df.resample('W').mean())

df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20'],
    'sales': [100, 150, 200, 130, 170]
})

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

print(df['date'])

print(df[df['date'] > '2023-01-10']['sales'])

df.set_index('date', inplace=True)

print(df.resample('W').mean())