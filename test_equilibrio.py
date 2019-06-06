from Malla_equilibrio_2 import Malla
from matplotlib import pyplot as plt

coors, subfibs = m = Malla.leer_de_archivo("Malla.txt")

fig = plt.figure()
ax = fig.add_subplot(111)

# grafico las subfibras
for i in range(len(subfibs)):
    xx = list() # valores x
    yy = list() # valores y
    # son dos nodos por subfibra
    n0 = subfibs[i][0]
    n1 = subfibs[i][1]
    r0 = coors[n0]
    r1 = coors[n1]
    xx = [r0[0], r1[0]]
    yy = [r0[1], r1[1]]
    p = ax.plot(xx, yy, label=str(i))

ax.legend(loc="upper left", numpoints=1, prop={"size":6})
plt.show()