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
        self.num = 0 
        self.num_fr = 0 
        self.num_in = 0 
        self.x0 = []  # coordenadas iniciales de los nodos 
        self.x = [] # coordenadas actuales
        self.tipos = [] 
        self.mask_fr = [] 
        self.mask_in = []
        for coor, tipo in zip(coors,tipos):
            self.num += 1 
            self.x0.append(coor) 
            self.x.append(coor) 
            if tipo == 1:
                self.num_fr += 1 
                self.tipos.append(1)
            elif tipo == 2:
                self.num_in += 1 
                self.tipos.append(2)
            else: 
                raise ValueError("tipo solo puede ser 1 (frontera) o 2 (interseccion)")
        self.cerrar()

    def get_nodos_fr(self):
        return self.x0[self.mask_fr]

    def get_nodos_in(self):
        return self.x0[self.mask_in]

    def abrir(self): 
        """ es necesario para agregar mas nodos """
        self.x0 = [val for val in self.x0]
        self.x = [val for val in self.x]
        self.tipo = [val for val in self.tipo]
        self.mask_fr = [val for val in self.mask_fr]
        self.mask_in = [val for val in self.mask_in]

    def add_nodo(self, xnod, tipo):
        if self.open:
            self.num += 1 
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
        self.num_fr = np.sum(self.mask_fr) 
        self.num_in = np.sum(self.mask_in)
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

class Iterador(object):
    def __init__(self, n, x, ecuacion, ref_small, ref_big, ref_div, maxiter, tol):
        self.n = n # tamano de la solcion (len(x))
        self.x = x # solucion actualizada o iterada
        self.dl1 = np.zeros(self.n, dtype=float) # necesario para evaluar convergencia
        self.ecuacion = ecuacion # ecuacion iterable a resolver (dx = x-x1 = f(x1))
        self.ref_small = ref_small # integer, referencia para desplazamientos pequenos
        self.ref_big = ref_big # integer, referencia para desplazamientos grandes
        self.ref_div = ref_div # integer, referencia para desplazamientos divergentes (<1.0)
        self.maxiter = maxiter  # integer, maximo numero de iteraciones no lineales
        self.tol = tol # integer, tolerancia para converger
        self.flag_primera_iteracion = True
        self.flag_big = np.zeros(self.n, dtype=bool)
        self.flag_div = np.zeros(self.n, dtype=bool)
        self.it = 0
        self.convergencia = False
        self.maxiter_alcanzado = False


    def iterar1(self):
        # incremento el numero de iteraciones
        self.it += 1
        # calculo el incremento de x
        dx = self.ecuacion(self.x) # solucion nueva
        # calculo las magnitudes de las variaciones y en base a ellas 
        # inicializo flags para evaluar inestabilidad
        self.flag_big[:] = False
        self.flag_div[:] = False
        # inicializo array de magnitud de desplazamientos
        dl = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            # calculo magnitud de desplazamiento [i]
            dl[i] = np.sqrt( np.dot(dx[i],dx[i]) )
            # evaluo relacion entre desplazamiento actual y desplazamiento previo
            if self.flag_primera_iteracion: 
                # si es la primera iteracion no puedo evaluar relacion con desplazamiento anterior
                self.flag_primera_iteracion = False 
                rel_dl = 0.0
            else:
                rel_dl = dl[i]/self.dl1[i]
            # evaluo si la solucion es estable o no
            if dl[i]<self.ref_small: 
                pass 
            elif dl[i]>self.ref_big:
                self.flag_big[i] = True 
            elif rel_dl>self.ref_div:
                self.flag_div[i] = True
        # si el incremento fue estable, modifico los arrays 
        inestable = np.any(self.flag_big) or np.any(self.flag_div)
        if not inestable:
            self.x = self.x + dx # incremento la solucion
            self.dl1 = dl
            # calculo errores 
            self.err = dl
            # evaluo convergencia 
            if ( np.max(self.err) < self.tol):
                self.convergencia = True 
            # evaluo que no haya superado el maximo de iteraciones
            if self.it >= self.maxiter:
                self.maxiter_alcanzado = True
        else:
            # si es inestable no se modifica la solucion, hay que reintentar
            self.it -= 1
            self.ecuacion.solventar_inestabilidad(self.flag_big, self.flag_div)   
  
    def iterar(self):
        while True:
            self.iterar1()
            if self.convergencia or self.maxiter_alcanzado:
                break


class Malla(object):
    def __init__(self, nodos, con, ec_con):
        self.nodos = nodos 
        self.con = con
        self.ec_con = ec_con
        self.conT = con.get_traspuesta() 
        self.dl0 = [] 
        self.calcular_dl0()
        self.a = np.zeros( (self.num_sfs, 2), dtype=float )
        self.t = np.zeros( self.num_sfs, dtype=float )

    def calcular_dl0(self):
        """ calcular las longitudes iniciales de todas las subfibras """
        for jsf in range(self.num_sfs):
            nod_ini, nod_fin = self.con.get_con_elem0(jsf)
            x0_ini = self.nodos.x0[nod_ini]
            x0_fin = self.nodos.x0[nod_fin]
            dr0 = x0_fin - x0_ini 
            dl0 = np.sqrt( np.dot(dr0,dr0) )
            self.dl0.append(dl0)
        self.dl0 = np.array(self.dl0, dtype=float)

    @property 
    def num_sfs(self):
        """ getter: numero de subfibras """
        return self.con.n0

    def get_x(self):
        return self.nodos.x

    def set_x(self, x):
        self.nodos.x = x

    def ec_constitutiva(self, k, lam):
        return k*(lam-1.0)

    def calcular_tensiones(self):
        """ calcula las tensiones de las subfibras en base a 
        las coordenadas de los nodos y la conectividad """
        for jsf in range(self.num_sfs):
            nod_ini, nod_fin = self.con.get_con_elem0(jsf)
            x_ini = self.nodos.x[nod_ini]
            x_fin = self.nodos.x[nod_fin]
            dr = x_fin - x_ini 
            dl = np.sqrt(np.dot(dr,dr))
            lam = dl / self.dl0[jsf]
            self.a[jsf] = dr/dl
            self.t[jsf] = self.ec_con(lam, [0.1])
    
    def mover_nodos_frontera(self, F):
        xf0 = self.nodos.get_nodos_fr()
        xf = np.matmul( xf0, np.transpose(F) )
        self.nodos.x[self.nodos.mask_fr] = xf

    def mover_nodos(self):
        """ mueve los nodos segun la tension resultante
        sobre cada nodo y aplicando una pseudoviscosidad
        los nodos frontera van con deformacion afin 
        esto representa una sola iteracion """ 
        TenRes = np.zeros(2, dtype=float)
        dx = np.zeros((self.nodos.num,2), dtype=float)
        for n in range(self.nodos.num):
            if self.nodos.tipos[n] == 1:
                # nodo frontera
                dx[n] = 0.0
            else: 
                # nodo interseccion
                # tengo que obtener cuales son las subfibras correspondientes
                # eso lo hago con la conectividad traspuesta
                sfs = self.conT.get_con_elem0[n]
                # la tension resultante es la suma
                TenRes = np.sum( self.t[sfs] , 0 )
                
                    
