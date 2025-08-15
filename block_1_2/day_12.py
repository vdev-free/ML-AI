import pandas as pd

data = {
    'name': ['Anna', 'Rob', "Can", 'Jira', 'Susanna'],
    'age': [18, 19, 20, 21, 22],
    'grade': [87, 90, 93, 91, 82],
}

df = pd.DataFrame(data)
df.to_csv('students.csv', index=False)

db = pd.read_csv('students.csv')
print(db)

top = db[db['grade'] > 90]
print(top)

top.to_csv('top_students.csv', index=False)