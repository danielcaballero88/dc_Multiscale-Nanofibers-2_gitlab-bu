"""
Malla de fibras discretas compuesta de nodos y subfibras
2 dimensiones
Coordenadas y conectividad
"""
import numpy as np 

class Nodos(object):
    def __init__(self, coors=[], tipos=[]):
        """
        x es una lista de python (num_nodos, 2) y aca lo convierto en array de numpy
        primero vienen los nodos de frontera 
        luego vienen los nodos interseccion
        """
        self.open = True # estado open significa que se pueden agregar nodos, false que no (estado operativo)
        self.num_nod = 0 
        self.num_nod_fr = 0 
        self.num_nod_in = 0 
        self.x0 = []  # coordenadas iniciales de los nodos 
        self.x = [] # coordenadas actuales
        self.tipos = [] 
        self.mask_fr = [] 
        self.mask_in = []
        for coor, tipo in zip(coors,tipos):
            self.num_nod += 1 
            self.x0.append(coor) 
            self.x.append(coor) 
            if tipo == 1:
                self.num_nod_fr += 1 
                self.tipos.append(1)
            elif tipo == 2:
                self.num_nod_in += 1 
                self.tipos.append(2)
            else: 
                raise ValueError("tipo solo puede ser 1 (frontera) o 2 (interseccion)")
        self.cerrar()

    def abrir(self): 
        """ es necesario para agregar mas nodos """
        self.x0 = [val for val in self.x0]
        self.x = [val for val in self.x]
        self.tipo = [val for val in self.tipo]
        self.mask_fr = [val for val in self.mask_fr]
        self.mask_in = [val for val in self.mask_in]

    def add_nodo(self, xnod, tipo):
        if self.open:
            self.num_nod += 1 
            self.x0.append(xnod) 
            self.x.append(xnod)
            self.tipo.append(tipo) 
        else:
            raise RuntimeError("nodos closed, no se pueden agregar nodos")

    def add_nodos(self, n, x, tipo):
        if self.open:
            for i in range(n): 
                self.add_nodo(x[i], tipo)
        else: 
            raise RuntimeError("nodos closed, no se pueden agregar nodos")

    def cerrar(self):
        """ convertir todo a numpy arrays y crear masks"""
        self.x0 = np.array(self.x0, dtype=float) 
        self.x = np.array(self.x, dtype=float) 
        self.tipos = np.array(self.tipos, dtype=int)
        self.mask_fr = self.tipos == 1 
        self.mask_in = self.tipos == 2 
        self.num_nod_fr = np.sum(self.mask_fr) 
        self.num_nod_in = np.sum(self.mask_in)
        self.open = False

class Conectividad(object):
    def __init__(self, conec_in=[]):
        """
        conectividad conecta dos listas (o arrays) de objetos: elems0 con elems1
        es para conectar nodos con subfibras por lo pronto:
        cada item de conec_in es un elem0
        elems0 son indices de subfibras y estan todos del 0 al n0-1
        cada elem0 se compone de los indices de los nodos a los que esta conectado (elems1) 
        no es necesario que figuren todos los elems1, pero si es recomdendable
        """
        self.n0 = len(conec_in)
        self.ne = np.zeros(self.n0, dtype=int) # numero de elem1 por cada elem0
        self.je = [] # aca pongo la conectividad en un array chorizo
        self.len_je = 0
        for i0, elem0 in enumerate(conec_in):
            for item1 in elem0:
                self.ne[i0] += 1 # sumo un elem1 conectado a elem0
                self.je.append(item1) # lo agrego a la conectividad
                self.len_je += 1
        # convierte el self.je a numpy array
        self.je = np.array(self.je, dtype=int)
        # ahora ensamblo el self.ie a partir del self.ne 
        self.ie = np.zeros(self.n0+1, dtype=int)
        self.ie[self.n0] = self.len_je # ultimo indice (len=self.n0+1, pero como es base-0, el ultimo tiene indice self.n0) 
        for i0 in range(self.n0-1, -1, -1): # recorro desde el ultimo elemento (indice n0-1) hasta el primer elemento (indice 0)
            self.ie[i0] = self.ie[i0+1] - self.ne[i0]

    def get_con_elem0(self, elem0):
        """ 
        devuelve los elementos conectados al elemento "elem0"
        por lo pronto elem0 es un indice (los elems0 son indices de subfibras) 
        y sus conectados elems1 son indices tambien 
        """
        return self.je[self.ie[elem0] : self.ie[elem0+1]]

    def calcular_traspuesta(self):
        """ 
        la conectividad traspuesta indica para cada nodo
        cuales son las subfibras que estan en contacto
        puede pasar que algun indice de nodo no este presente en je
        pero es recomendable que esten todos


        OJO, necesito calcular las orientaciones de las subfibras respecto de los nodos
        """
        # supongo que el nodo de mayor indice en je es el maximo nodo que hay 
        n1 = np.max(self.je) + 1 # el +1 es por la base-0 de python
        # ahora para cada elem1, recorro los elem0 para ver las conexiones
        # notar que supondre que los elems0 son range(self.n0) y los elems1 son range(n1)
        jeT = [] 
        len_jeT = 0
        neT = np.zeros(n1, dtype=int)
        for i1 in range(n1): # aca supongo que los elems1 son range(n1), o sea una lista de indices
            for i0 in range(self.n0): # idem para elems0, son una lista de indices
                elem0 = self.get_con_elem0(i0)
                if i1 in elem0:
                    jeT.append(i0)
                    len_jeT += 1
                    neT[i1] += 1
        # convierto el jeT a numpy 
        jeT = np.array(jeT, dtype=int)
        # ensamblo el ieT 
        ieT = np.zeros(n1+1, dtype=int)
        ieT[n1] = len_jeT # empiezo con el indice de un elemento extra que no existe
        for i1 in range(n1-1, -1, -1):
            ieT[i1] = ieT[i1+1] - neT[i1] # de ahi calculo el indice donde empieza cada elemento
        return n1, len_jeT, neT, ieT, jeT

    def get_traspuesta(self):
        """ 
        la conectividad traspuesta indica para cada nodo
        cuales son las subfibras que estan en contacto
        puede pasar que algun indice de nodo no este presente en je
        pero es recomendable que esten todos


        OJO, necesito calcular las orientaciones de las subfibras respecto de los nodos
        """
        # creo una conectividad vacia 
        cotr = Conectividad([])
        # calculo los arrays de la conectividad traspuesta 
        n1, len_jeT, neT, ieT, jeT = self.calcular_traspuesta() 
        cotr.n0 = n1 
        cotr.len_je = len_jeT 
        cotr.ne = neT 
        cotr.ie = ieT 
        cotr.je = jeT 
        return cotr

    # def get_con_elem1(self, elem1):
    #     """ 
    #     devuelve los elementos conectados al elemento "elem1"
    #     por lo pronto elem1 es un indice (los elems1 son indices de subfibras) 
    #     y sus conectados elems1 son indices tambien 
    #     """
    #     return self.jeT[self.ieT[elem1] : self.ieT[elem1+1]]


class Malla(object):
    def __init__(self, nodos, con):
        self.nodos = nodos 
        self.con = con
        self.conT = con.get_traspuesta() 
        # calculo las longitudes iniciales de las subfibras
        self.dl0 = []
        for jsf in range(con.n0):
            nod_ini, nod_fin = con.get_con_elem0(jsf)
            x0_ini = self.nodos.x0[nod_ini]
            x0_fin = self.nodos.x0[nod_fin]
            dr0 = x0_fin - x0_ini 
            dl0 = np.sqrt( np.dot(dr0,dr0) )
            self.dl0.append(dl0)
        self.dl0 = np.array(self.dl0, dtype=float)
