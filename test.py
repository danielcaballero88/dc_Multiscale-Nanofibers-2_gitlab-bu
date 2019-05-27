from Malla import Nodos, Conectividad, Malla

nod_coors = [
    [0, 0],
    [0, 2],
    [2, 2],
    [0, 2],
    [1, 1]
]

nod_tipos = [1,1,1,1,2]

n = Nodos(nod_coors,nod_tipos) 

print "num nodos: ", n.num
print "tipo nodos: ", n.tipos 
print "coordenadas: ", n.x
print "mask fronteras: ", n.mask_fr 
print "mask intersecciones: ", n.mask_in 
print "num fron y num in: ", n.num_fr, n.num_in
print "---"

conec = [
    [0,4],
    [4,1],
    [2,4],
    [4,3]
]

c = Conectividad(conec) # conec indica los nodos de cada subfibra

print "ne: ", c.ne 
print "ie. ", c.ie 
print "je: ", c.je 
print "---"

ctr = c.get_traspuesta() # la traspuesta indica las subfibras de cada nodo

print "neT: ", ctr.ne 
print "ieT: ", ctr.ie 
print "jeT: ", ctr.je
print "---"

def EcCon_lineal(lam, param):
    k = param[0]
    return k*(lam-1.0)

m = Malla(n,c, EcCon_lineal)
print m.dl0

m.calcular_tensiones()
print m.a 
print m.t