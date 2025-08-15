import numpy as np
# from sklearn.linear_model import LinearRegression

# X = np.array([
#     [19, 7, 1],
#     [30, 8, 2],
#     [40, 8, 3]
# ])

# y = np.array([70, 60, 55])

# model = LinearRegression()

# model.fit(X,y)

# new_member_50 = np.array([[50, 9, 2]])

# prediction = model.predict(new_member_50)

# print(prediction)

from sklearn.linear_model import LogisticRegression

# X = np.array([
#     [19, 7, 1],
#     [30, 8, 2],
#     [40, 8, 3],
#     [50, 6, 1],
#     [60, 5, 2]
# ])

# y = np.array([0,0,1,1,1])

# model = LogisticRegression()

# model.fit(X, y)

# new_person = np.array([[33, 7, 1]])
# prediction = model.predict(new_person)

# print(f'Клас нової людини: {prediction[0]}')

X = np.array([
    [20,20,35],
    [25,23,33],
    [40,40,55],
    [120,2000,2200]
    ])

y = np.array([0,0,1,1])

model = LogisticRegression()

model.fit(X,y)

humster = np.array([[1,5,7]])

prediction = model.predict(humster)

print(f'класифікація humster - {prediction}')