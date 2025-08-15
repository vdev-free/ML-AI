# import pandas as pd

# data = {
#     'name': ['Anna', 'Den', 'Lilia', 'Rob', 'Susi'],
#     'group': ['A', 'B', 'A', 'B', 'A'],
#     'grade': [90, 80, 95, 85, 88]
# }

# df = pd.DataFrame(data)

# new_data = df.groupby('group')['grade'].mean().reset_index()

# new_data.to_csv('data_group.csv', index=False)

# print(pd.read_csv('data_group.csv'))

import pandas as pd

data = {
    'name': ['Anna', 'Den', 'Lilia', 'Rob', 'Susi'],
    'group': ['A', 'B', 'A', 'B', 'A'],
    'grade': [90, 80, 95, 85, 88]
}

df = pd.DataFrame(data)

print(df.groupby('group')['grade'].mean())
print(df.groupby('group')['grade'].max())
print(df.groupby('group').count())
print(df.groupby('group')['grade'].mean().reset_index())

