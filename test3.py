import numpy as np

a = np.array(
    [
        [1,2],
        [3,4],
        [5,6],
        [7,8],
        [9,0]
    ]
)

con = [1,2,4]

b = a[con]
b_in = b[1:-1]

c = np.array([10, 11])

b_in = c.reshape(-1,2)

a[con[1:-1]] = b_in

print a