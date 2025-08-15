import numpy as np

A = np.array([
    [4,7,9],
    [2,6,8],
    [1,2,3],
])

det = np.linalg.det(A)

print(det)

if det != 0:
    inv = np.linalg.inv(A)
    print(inv)

    result = A @ inv
    print("Перевірка (A @ inv):")
    print(np.round(result, 2))
else:
    print('вироджена')