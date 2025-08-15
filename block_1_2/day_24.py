import pandas as pd
import numpy as np

# data = {
#     'name': ['Anna', 'Rob', 'Den', np.nan, 'Diana'],
#     'age': [ None, 18, 24, 20, 19],
#     'grade': [90, 86, np.nan, 95, 90]
# }

# df = pd.DataFrame(data)
# df.to_csv('students.csv', index=False)

# nst = pd.read_csv('students.csv')

# nst_cleaned = nst.dropna()
# nst_filled =  nst.fillna(0)
# nst['age'] = nst['age'].fillna(nst['age'].mean())
# nst['name'] = nst['name'].replace("Anna", 'Bob')


# print(nst[nst['grade'] >= 90])

data = {
    'name': ['Anna', 'Rob', 'Den', np.nan, 'Diana'],
    'age': [ None, 18, 24, 20, 19],
    'grade': [90, 86, np.nan, 95, 90]
}

df = pd.DataFrame(data)

print(df.isnull())
print(df.isnull().sum())
print(df.dropna())
print(df.fillna(0))
print(df.fillna(0)['grade'].mean())

fst = df.fillna(0)
print(fst[fst['grade'] >= 90])

print(fst['name'].replace('Rob', 'Nick'))