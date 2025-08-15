import pandas as pd

data = {
    'name': ['Den', 'Stiv', 'Olga', 'Klara'],
    'age': [18, 19, 25, 21],
    'grade': [85, 88, 90, 95]
}

st = pd.DataFrame(data)

print(st['name'])
print(st.iloc[0])
print(st.loc[2, 'grade'])
print(st[st['age'] > 90])
print(st.sort_values('age'))
print(st.describe())
