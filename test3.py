import numpy as np

a= np.array(
    [
        [1,2],
        [3,4]
    ]
)

b = a[0].copy()

print b

b[0] = 20

print a