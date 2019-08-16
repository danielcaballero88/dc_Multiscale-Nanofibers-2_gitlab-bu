import numpy as np
import time

a = np.array(
    [
        [1,2],
        [3,4],
        [5,6],
        [7,8],
        [9,0]
    ]
)

b = a.reshape(-1,)
print b

m = np.logical_or(b<4, b>8)
print m

c = b[m]
print c