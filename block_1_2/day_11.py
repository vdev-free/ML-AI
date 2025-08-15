import pandas as pd

data = {
    'name': ['Anna', 'Den', 'Oleh'],
    'age': [18, 20, 21],
    'grade': [90, 85, 92]
}

df = pd.DataFrame(data)

print(df)

students = {'name': ['Rob', 'Cali', 'Lilia', 'Bob', 'Joni'],
            'color': ['Red', 'Blue', 'Green', 'Braun', 'Dark'],
            'auto': [True, False, True, True, False],
            'grade': [90, 70, 80, 94, 87]}

df = pd.DataFrame(students)

print(df)

print(df['name'])

print(df[df['grade'] >= 90])

print(df['grade'].mean())