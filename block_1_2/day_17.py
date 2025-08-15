import numpy as np

# a = np.array([10, 20, 30, 40, 50])

# print(a[0])
# print(a[-2])

# print(a[1:3])
# print(a[:3])
# print(a[2:])

# b = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])

# print(b[0:,1:])

a = [10, 20, 30, 40, 50]

print(a[0])
print(a[-1])
print(a[1:3])
print(a[2:])

b = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

print(b[1,1])
print(b[2:])
print(b[0:,0])
print(b[0:2, 0:2])

