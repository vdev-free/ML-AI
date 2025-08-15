import pandas as pd

data = {
    'name': ['Den', 'Bob', 'Anna', "Lilia", 'Sten', 'Lusija'],
    'age': [18, 19, 20, 21, 22, 23],
    'gender': ['m', 'm', 'w', 'w', 'm', 'w'],
    'grade': [87, 88, 95, 90, 96, 88]
}

dt = pd.DataFrame(data)

dt_sorted = dt.sort_values(['grade'], ascending=False)

print(dt_sorted.head(3))

groupby = dt.groupby(['gender'])

print(groupby['grade'].mean())
print(groupby['grade'].max())