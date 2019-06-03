"""
Malla de fibras discretas compuesta de nodos y subfibras
2 dimensiones
Coordenadas y conectividad
"""
import numpy as np 

class Nodos(object):
    def __init__(self):
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


    @classmethod 
    def from_coordenadas(cls, coors, tipos):
        instance = cls()
        for coor, tipo in zip(coors,tipos):
            instance.num += 1 
            instance.x0.append(coor) 
            instance.x.append(coor) 
            if tipo == 1:
                instance.num_fr += 1 
                instance.tipos.append(1)
            elif tipo == 2:
                instance.num_in += 1 
                instance.tipos.append(2)
            else: 
                raise ValueError("tipo solo puede ser 1 (frontera) o 2 (interseccion)")
        instance.cerrar()
        return instance

    def get_nodos_fr(self):
        return self.x0[self.mask_fr]

    def get_nodos_in(self):
        return self.x0[self.mask_in]

    def abrir(self): 
        """ es necesario para agregar mas nodos """
        self.x0 = [val for val in self.x0]
        self.x = [val for val in self.x]
        self.tipos = [val for val in self.tipos]
        self.mask_fr = [val for val in self.mask_fr]
        self.mask_in = [val for val in self.mask_in]
        self.open = True

    def add_nodo(self, xnod, tipo):
        if self.open:
            self.num += 1 
            self.x0.append(xnod) 
            self.x.append(xnod)
            self.tipos.append(tipo) 
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
    """
    conectividad conecta dos listas (o arrays) de objetos: elems0 con elems1
    es para conectar nodos con subfibras por lo pronto:
    cada item de conec_in es un elem0
    elems0 son indices de subfibras y estan todos del 0 al n0-1
    cada elem0 se compone de los indices de los nodos a los que esta conectado (elems1) 
    no es necesario que figuren todos los elems1, pero si es recomdendable
    """
    def __init__(self):
        self.open = True 
        self.num = 0 # cantidad de elem0
        self.len_je = 0 # cantidad de elem1 repetidos en la conectividad (si aparecen mas de una vez se cuentan todas)
        self.ne = [] 
        self.je = [] 
        self.ie = [] 

    @classmethod 
    def from_listoflists(cls, conec):
        instance = cls()
        instance.add_conec_listoflists(conec)
        instance.cerrar()  # se calcula num, len_je, y el array ie
        return instance

    def cerrar(self):
        """ se cierra la conectividad
        mejora el uso
        mas dificil agregar elementos 
        se pasan los arrays a numpy por eficiencia
        se calcula el ie = array que apunta a donde comienza cada elem0 en el je """
        assert self.open == True
        # convierto a numpy
        self.ne = np.array(self.ne, dtype=int) 
        self.je = np.array(self.je, dtype=int) 
        # calculo el ie (apunta a donde comienza cada elem0 en je)
        self.ie = np.zeros(self.num+1, dtype=int) # necesito tener num+1 porque el ultimo elem0 termina donde comenzaria un elem0 extra inexistente en num+1
        self.ie[self.num] = self.len_je # ultimo indice (len=instance.num+1, pero como es base-0, el ultimo tiene indice instance.num)
        for i0 in range(self.num-1, -1, -1): # recorro desde el ultimo elem0 (indice num-1) hasta el primer elem0 (indice 0)
            self.ie[i0] = self.ie[i0+1] - self.ne[i0] # para saber donde empieza cada elem0 voy restando el numero de elem1 de cada uno
        self.open = False

    def abrir(self):
        """ se abre la conectividad
        para agregar nuevos elementos
        imposibilita operar con ella """
        assert self.open == False 
        self.ne = [n for n in self.ne]
        self.je = [e1 for e1 in self.je] 
        self.ie = [] 
        self.open = True

    def add_conec_listoflists(self, conec):
        """ agrega la conectividad dada en una lista de listas
        a los arrays ne y je
        ademas calcula num y len_je """
        self.num = len(conec) # numero de elementos en conec
        self.ne = [] # numero de elem1 por cada elem0]
        self.je = [] # aca pongo la conectividad en un array chorizo
        self.len_je = 0
        for elem0 in conec:
            num_elem1 = 0 # numero de elem1 en elem0
            for item1 in elem0:
                num_elem1 += 1 # sumo un elem1 conectado a elem0
                self.je.append(item1) # lo agrego a la conectividad
                self.len_je += 1
            self.ne.append(num_elem1) # agrego el numero de elem1 de elem0 a la lista

    def add_elem0(self, con_elem0):
        assert self.open==True
        num_elem1 = 0 
        for elem1 in con_elem0:
            num_elem1 += 1
            self.je.append(elem1)
            self.len_je += 1
        self.ne.append(num_elem1)

    def get_con_elem0(self, j0):
        """ 
        devuelve los elem1 conectados al elem0 "j0"
        j0 es el indice de elem0 (que por ahora son indices asi que se corresponde con el)
        y sus conectados elem1 son indices tambien 
        """
        return self.je[ self.ie[j0] : self.ie[j0+1] ]

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
        # notar que supondre que los elems0 son range(self.num) y los elems1 son range(n1)
        jeT = [] 
        len_jeT = 0
        neT = np.zeros(n1, dtype=int)
        for i1 in range(n1): # aca supongo que los elems1 son range(n1), o sea una lista de indices
            for i0 in range(self.num): # idem para elems0, son una lista de indices
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
        cotr = Conectividad()
        # calculo los arrays de la conectividad traspuesta 
        n1, len_jeT, neT, ieT, jeT = self.calcular_traspuesta() 
        cotr.num = n1 
        cotr.len_je = len_jeT 
        cotr.ne = neT 
        cotr.ie = ieT 
        cotr.je = jeT 
        return cotr


class Subfibras(Conectividad):
    """ es una conectividad
    con algunos atributos y metodos particulares """ 

    def __init__(self):
        Conectividad.__init__()
        self.longs0 = np.zeros(self.num, dtype=float) # longitudes iniciales de las fibras
        self.ecuacion_constitutiva = None 
        self.param_con = None # va a tener que ser un array de (num_subfibras, num_paramcon) presumiendo que cada subfibra pueda tener diferentes valores de parametros        

    @classmethod 
    def closed(cls, conec, coors0, param_con, ec_con):
        instance = cls()
        instance.add_conec_listoflists(conec)
        instance.calcular_longs0(coors0) 
        instance.param_con = param_con
        instance.ecuacion_constitutiva = ec_con 
        return instance

    def get_con_sf(self, j):
        """ obtener la conectividad de una subfibra (copia de get_con_elem0)"""
        return self.je[ self.ie[j] : self.ie[j+1] ]

    def calcular_long_j(self, xnods, j):
        nod_ini, nod_fin = self.get_con_sf(j) 
        x_ini = xnods[nod_ini] 
        x_fin = xnods[nod_fin] 
        dr = x_fin - x_ini 
        long = np.sqrt ( np.dot(dr,dr) )
        return long

    def calcular_longs0(self, xnods0):
        """ calcular las longitudes iniciales de todas las subfibras """
        self.longs0 = self.calcular_longitudes(xnods0, self.longs0)

    def calcular_longitudes(self, xnods, longs=None):
        """ calcular las longitudes de las fibras """ 
        if longs is None:
            longs = np.zeros(self.num, dtype=float) # seria mejor tenerlo preadjudicado
        for jsf in range(self.num):
            nod_ini, nod_fin = self.get_con_sf(jsf)
            x_ini = xnods[nod_ini]
            x_fin = xnods[nod_fin]
            dr = x_fin - x_ini 
            longs[jsf] = np.sqrt( np.dot(dr,dr) )
        return longs

    def calcular_tension_j(self, lam, j):
        return self.ecuacion_constitutiva(lam, self.param_con[j])

    def calcular_elongaciones(self, xnods):
        longs = self.calcular_longitudes(xnods)
        lams = longs/self.longs0 
        return lams

    def calcular_tensiones(self, xnods):
        lams = self.calcular_elongaciones(xnods) 
        tens = self.ecuacion_constitutiva(lams, self.param_con)
        return tens


class Iterador(object):
    def __init__(self, n, x, sistema, ref_small, ref_big, ref_div, maxiter, tol):
        self.n = n # tamano de la solcion (len(x))
        self.x = x # solucion (comienza como semilla, luego es la iterada o actualizada)
        self.dl1 = np.zeros(self.n, dtype=float) # necesario para evaluar convergencia
        self.sistema = sistema # ecuacion iterable a resolver (dx = x-x1 = f(x1))
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
        dx = self.sistema.calcular_incremento(self.x) # solucion nueva
        # ===
        # ahora voy a chequear la estabilidad
        # ===
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
                # si es la primera iteracion no tengo desplazamiento previo
                # entonces seteo un valor que no de divergencia en desplazamientos
                rel_dl = self.ref_div 
                if i==self.n-1:
                    # cuando estoy en el ultimo nodo de la primera iteracion
                    # desmarco el flag de primera iteracion
                    self.flag_primera_iteracion = False
            else:
                # si no es la primera iteracion puedo calcular la relacion de desplazamientos
                # pero puede pasar que dl1[i] = 0 y entonces hay error
                if self.dl1[i] < self.tol:
                    rel_dl = self.ref_div 
                else:
                    rel_dl = dl[i]/self.dl1[i]
            # evaluo si la solucion es estable o no
            # si el desplazamiento es pequeno lo dejo pasar
            if dl[i]<self.ref_small: 
                pass 
            # si es muy grande lo marco como inestable
            elif dl[i]>self.ref_big:
                self.flag_big[i] = True
            # al igual que si es divergente
            elif rel_dl>self.ref_div:
                self.flag_div[i] = True
        # si el incremento fue estable, modifico los arrays 
        inestable = np.any(self.flag_big) or np.any(self.flag_div)
        if not inestable:
            print "estable: ", self.it, dx[1], self.x[1], self.x[1]+dx[1]
            self.x = self.x + dx # incremento la solucion
            self.dl1[:] = dl
            # calculo error 
            self.err = np.max(dl)
            # evaluo convergencia 
            if ( self.err < self.tol):
                self.convergencia = True 
            # evaluo que no haya superado el maximo de iteraciones
            if self.it >= self.maxiter:
                self.maxiter_alcanzado = True
        else:
            # si es inestable no se modifica la solucion, hay que reintentar
            print "inestable: ", dl[1], self.dl1[1]
            self.it -= 1
            self.sistema.solventar_inestabilidad(self.flag_big, self.flag_div)   
  
    def iterar(self):
        while True:
            self.iterar1()
            if self.convergencia or self.maxiter_alcanzado:
                return self.x


class Malla(object):
    def __init__(self):
        self.open = True
        self.nodos = Nodos() 
        self.sfs = Subfibras() # las subfibras las conecto con los nodos que las componen
        self.psv = None 

    @classmethod
    def closed(cls, nodos, subfibras, psv):
        instance = cls()
        instance.nodos = nodos 
        instance.sfs = subfibras
        # para resolver el sistema uso pseudoviscosidad
        instance.psv = np.array(psv, dtype=float)
        instance.open = False 
        
    def get_x(self):
        return self.nodos.x

    def set_x(self, x):
        self.nodos.x = x

    def calcular_tracciones_de_subfibras(self, x1=None):
        """ calcula las tensiones de las subfibras en base a 
        las coordenadas de los nodos x1 y la conectividad """
        # si no mando valor significa que quiero averiguar las tracciones 
        # con las coordenadas que tengan los nodos de la malla
        if x1 is None:
            x1 = self.nodos.x 
        tracciones = np.zeros( (self.sfs.num,2), dtype=float )
        for jsf in range(self.sfs.num):
            nod_ini, nod_fin = self.sfs.get_con_sf(jsf)
            x_ini = x1[nod_ini]
            x_fin = x1[nod_fin]
            dr = x_fin - x_ini 
            dl = np.sqrt(np.dot(dr,dr))
            lam = dl / self.sfs.longs0[jsf]
            a = dr/dl
            t = self.sfs.tension_subfibra(jsf, lam)
            tracciones[jsf] = t*a
        return tracciones

    def calcular_tracciones_sobre_nodos(self, tracciones_subfibras):
        """ calcula las tensiones resultantes sobre los nodos
        recorriendo las subfibras y para cada subfibra sumando
        la traccion correspondiente a sus nodos, con el signo
        segun si es el nodo inicial o el nodo final """ 
        TraRes = np.zeros( (self.nodos.num,2), dtype=float)
        # recorro las fibras para saber las tensiones sobre los nodos
        for jsf in range(self.sfs.num):
            # tengo que sumar la tension de la fibra a los nodos
            traccion_j = tracciones_subfibras[jsf]
            # sobre el primer nodo va asi y sobre el segundo en sentido contrario
            nod_ini, nod_fin = self.sfs.get_con_sf(jsf)
            TraRes[nod_ini] += traccion_j 
            TraRes[nod_fin] -= traccion_j
        return TraRes

    def mover_nodos_frontera(self, F):
        """ mueve los nodos de la frontera de manera afin segun el 
        gradiente de deformaciones (tensor F de 2x2) """
        xf0 = self.nodos.get_nodos_fr()
        xf = np.matmul( xf0, np.transpose(F) )
        self.nodos.x[self.nodos.mask_fr] = xf

    def deformar_afin(self, F):
        """ mueve todos los nodos de manera afin segun el F """
        x0 = self.nodos.x0 
        x = np.matmul( x0, np.transpose(F)) 
        self.nodos.x = x

    def mover_nodos(self, tracciones_nodos):
        """ mueve los nodos segun la tension resultante
        sobre cada nodo y aplicando una pseudoviscosidad
        los nodos frontera van con deformacion afin 
        esto representa una sola iteracion 
        (calculo de dx segun x: dx=f(x) ) """ 
        dx = np.zeros((self.nodos.num,2), dtype=float)
        for n in range(self.nodos.num):
            if self.nodos.mask_fr[n]:
                dx[n] = 0.0 # nodo de dirichlet
            else:
                dx[n] = tracciones_nodos[n] / self.psv[n]
        return dx 

    def calcular_incremento(self, x1=None):
        """ metodo para sobrecargar los parentersis () 
        la idea es que este metodo sea la funcion dx=f(x)
        donde f se evalua en un valor x1 """ 
        # si no mando valor significa que quiero averiguar las tracciones 
        # con las coordenadas que tengan los nodos de la malla
        if x1 is None:
            x1 = self.nodos.x 
        # primero calculo segun las coordenadas, las tracciones de las subfibras
        trac_sfs = self.calcular_tracciones_de_subfibras(x1) 
        trac_nod = self.calcular_tracciones_sobre_nodos(trac_sfs) 
        dx = self.mover_nodos(trac_nod) 
        return dx

    def solventar_inestabilidad(self, flag_big_dx, flag_div_dx):
        """ es necesario tener esta subrutina para solventar situaciones
        en que, durante las iteraciones, haya desplazamiento exagerados
        o desplazamientos crecientes en iteraciones, en ese caso lo que
        se hace es aumentar la pseudoviscosidad del nodo en cuestion """ 
        nodos_criticos = flag_big_dx + flag_div_dx
        self.psv[nodos_criticos] = 2.0*self.psv[nodos_criticos]