import numpy as np

# prices = np.array([100,200,150,300,250])

# mask = prices > 200
# filtered = prices[mask]
# print(filtered)

# data = np.array([
#     [50,2,100000],
#     [75,3,150000],
#     [100,4,250000]
# ])

# mask = data[:,0] > 60
# filtered = data[mask]
# print(filtered)

# arr = np.random.randint(0, 100, size=10)
# mask = arr > 50
# filtered = arr[mask]
# print(filtered)

candidates = np.array([
    [1, 0, 0],
    [3, 1, 1],
    [5, 2, 2],
    [2, 1, 1],
    [4, 0, 2]
])

mask_year = candidates[:,0] > 2
mask_level_eng = candidates[:,1] >= 1
filtered = candidates[mask_year & mask_level_eng]

print(candidates)
print(filtered)
