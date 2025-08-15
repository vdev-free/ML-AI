import pandas as pd
import numpy as np

data = {
    'name': ['Anna', 'Den', 'Lilia', 'Rob'],
    'grade': [90, np.nan, 85, None],
    'group': ['A', 'B', 'A', 'B']
}

df = pd.DataFrame(data)

# print(df.isnull())
# print(df.dropna())
# print(df.fillna(0))
# print(df['grade'].fillna(df['grade'].mean()))
df['grade'] = df['grade'].fillna(df['grade'].mean())

for idx, row in df.iterrows():
    print(row)