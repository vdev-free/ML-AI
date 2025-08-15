import numpy as np

# arr = np.array([True, False, True, True])

# nums = np.array([5, 10, 15, 20])

# result = np.where(nums > 10, 99, nums)

# print(result)

# probs = np.array([0.1, 0.8, 0.4, 0.95])

# np.any(probs > 0.9)
# np.all(probs < 0.5)
# np.where(probs > 0.7)

# print(np.any(probs > 0.9),
# np.all(probs < 0.5),
# np.where(probs > 0.7))

# arr = np.random.randint(0, 100, size=5)

# print(arr)

# print(np.any(arr > 90) )
# print(np.all(arr > 10))
# print(np.where(arr > 50, 999, arr))

temps = np.array([
    [22,24,27],
    [25,30,35],
    [20,17,21]
])

print(np.any(temps > 30))
print(np.all(temps[0] > 20))
print(np.where(temps>28, '🔥', temps))