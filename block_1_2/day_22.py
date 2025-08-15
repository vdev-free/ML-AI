import pandas as pd

data = {
    'name': ['Georg', 'Lilu', 'Marko', 'Susi'],
    'age': [18, 19, 20, 21],
    'grade': [85, 90, 88, 96]
}

ts = pd.DataFrame(data)

ts.to_csv('students.csv', index=False)

df = pd.read_csv('students.csv')

print(df.head(2))
print(df.tail(2))
print(df.columns)
print(df['grade'].mean())