# import numpy as np

# a = np.array([1,2,3,4,5,7])

# print(np.where(a % 2 == 0, '0', '1'))

import numpy as np

a = np.array([1,2,3,4,5])

print((a == 4).any())
print((a == 0).all())
print(np.where(a > 2, 'OK', 'BAD'))
print(np.where(a % 2 == 0, 'парне', 'непарне'))