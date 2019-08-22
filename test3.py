import numpy as np

a= np.array(
    [
        [1,2],
        [3,4],
        [5,6],
        [7,8],
        [9,0]
    ]
)

a = a.tolist()
b = a[0].copy()

print b

for item in a:
    if item == [5,6]:
        a.append("foo")
    print item
b[0] = 20

print a
