import pandas as pd

# left = pd.DataFrame({
#     'id': [1, 2, 3],
#     'name': ['Alice', 'Bob', 'Den']
# })

# right = pd.DataFrame({
#     'id': [1, 2, 3],
#     'score': [85, 90, 88]
# })

# result = pd.merge(left, right, on='id', how='inner')
# print(result)

# df1 = pd.DataFrame({'a': [1, 2], 
#                     'b': [3, 4]})

# df2 = pd.DataFrame({'a': [5, 6], 
#                     'b': [7, 8]})

# vertical = pd.concat([df1, df2], axis=0)
# print(vertical)

# horizontal = pd.concat([df1, df2], axis=1)
# print(horizontal)

# df1 = pd.DataFrame({
#     'a': [1, 2]
# }, index=['x', 'y'])

# df2 = pd.DataFrame({
#     'b': [3, 4]
# }, index=['x', 'y'])

# joined = df1.join(df2)
# print(joined)

df1 = pd.DataFrame({'user_id': [1,2,3], 'age': [18,19,20]})

df2 = pd.DataFrame({'user_id': [1,2,3], 'grade': [21,None,23]})

users = pd.merge(df1, df2, on='user_id', how='outer')
print(users)

st1 = pd.DataFrame({
'a': [1, 2],
'b': [3, 4]
})
st2 = pd.DataFrame({
'a': [5, 6],
'b': [7, 8]
})
st3 = pd.DataFrame({
'a': [9, 12],
'b': [13, 14]
})

print(pd.concat([st1, st2, st3], axis=1))

s1 = pd.DataFrame({'a': [1, 2]}, index=['x', 'y'])
s2 = pd.DataFrame({'b': [5, 6]},  index=['x', 'y'])

print(s1.join(s2))