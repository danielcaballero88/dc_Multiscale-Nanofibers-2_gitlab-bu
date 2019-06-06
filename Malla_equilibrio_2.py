"""
Malla de fibras discretas compuesta de nodos y subfibras
2 dimensiones
Coordenadas y conectividad
"""
import numpy as np
from matplotlib import pyplot as plt
from Aux import find_string_in_file, calcular_largo_de_segmento

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
            instance.check_xnod_tipo(coor, tipo)
            instance.num += 1
            instance.x0.append(coor)
            instance.x.append(coor)
            instance.tipos.append(tipo)
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

    def check_xnod_tipo(self, xnod, tipo):
        assert isinstance(xnod, list)
        assert len(xnod) == 2
        assert isinstance(xnod[0], float)
        assert isinstance(xnod[1], float)

    def add_nodo(self, xnod, tipo):
        if self.open:
            self.check_xnod_tipo(xnod, tipo)
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
        self.pregraficado = False
        self.fig = None
        self.ax = None

    @classmethod
    def closed(cls, nodos, subfibras, psv):
        instance = cls()
        instance.nodos = nodos
        instance.sfs = subfibras
        # para resolver el sistema uso pseudoviscosidad
        instance.psv = np.array(psv, dtype=float)
        instance.open = False

    @classmethod
    def leer_de_archivo(cls, archivo):
        fid = open(archivo, "r")
        # leo los nodos
        target = "*coordenadas"
        ierr = find_string_in_file(fid, target, True)
        num_r = int(fid.next())
        coors = list()
        tipos = list()
        for i in range(num_r):
            j, t, x, y = (float(val) for val in fid.next().split())
            tipos.append(int(t))
            coors.append([x,y])
        # luego los segmentos
        target = "*segmentos"
        ierr = find_string_in_file(fid, target, True)
        num_s = int(fid.next())
        segs = list()
        for i in range(num_s):
            j, n0, n1 = (int(val) for val in fid.next().split())
            segs.append([n0,n1])
        # luego las fibras
        target = "*fibras"
        ierr = find_string_in_file(fid, target, True)
        num_f = int(fid.next())
        fibs = list()
        for i in range(num_f):
            out = [int(val) for val in fid.next().split()]
            j = out[0]
            fcon = out[1:]
            fibs.append(fcon)
        # ---
        # ahora puedo armar la malla simplificada:
        # nodos interseccion y subfibras
        # ---
        ms_nods_r = list() # coordenadas de la malla simplificada
        ms_nods_t = list() # tipos de los nodos de la malla simplificada
        ms_nods_n = list() # indices originales de los nodos
        ms_sfbs_c = list() # conectividad de subfibras de la malla simplificada
        ms_sfbs_l = list() # largo de las siguiendo el contorno de los segmentos
        # recorro cada fibra:
        for f in range(len(fibs)): # f es el indice de cada fibra en la malla completa
            # tengo una nueva subfibra
            new_sfb = [0, 0] # por ahora esta vacia
            # agrego el primer nodo
            s = fibs[f][0] # segmento 0 de la fibra f
            n0 = segs[s][0] # nodo 0 del segmento s
            ms_nods_r.append( coors[n0] )
            ms_nods_t.append( tipos[n0] ) # deberia ser un 1
            ms_nods_n.append( n0 )
            # lo agrego a la nueva subfibra como su primer nodo
            new_sfb[0] = len( ms_nods_r ) - 1 # es el ultimo nodo agregado, tambien podria hacer ms_nods_n.index(n0) que es lo mismo pero mas lento
            assert ms_nods_t[-1] == 1
            # recorro el resto de los nodos de la fibra para agregar los nodos intereccion
            l = 0. # aca voy sumando el largo de los segmentos que componen la subfibra
            for js in range(len(fibs[f])): # js es el indice de cada segmento en la fibra f (numeracion local a la fibra)
                # voy viendo los nodos finales de cada segmento
                s = fibs[f][js] # s es el indice de cada segmento en la malla original (numeracion global)
                n0 = segs[s][0] # primer nodo del segmento
                n1 = segs[s][1] # nodo final del segmento
                r0 = coors[n0]
                r1 = coors[n1]
                dx = r1[0] - r0[0]
                dy = r1[1] - r0[1]
                l += np.sqrt( dx*dx + dy*dy ) # largo del segmento (lo sumo al largo de la subfibra)
                if tipos[n1] in (1,2): # nodo interseccion (2) o nodo final (1)
                    # tengo que fijarme si el nodo no esta ya presente
                    # (ya que los nodos interseccion pertenecen a dos fibras)
                    if not n1 in ms_nods_n:
                        # el nodo no esta listado,
                        # tengo que agregarlo a la lista de nodos
                        ms_nods_r.append( coors[n1] )
                        ms_nods_t.append( tipos[n1] )
                        ms_nods_n.append( n1 )
                    # me fijo en la nueva numeracion cual es el indice del ultimo nodo de la subfibra
                    new_sfb[1] = ms_nods_n.index(n1)
                    # y agrega la conectividad de la subfibra a la lista
                    ms_sfbs_c.append( new_sfb )
                    ms_sfbs_l.append( l )
                    # si no llegue a un nodo frontera, debo empezar una nueva subfibra
                    if tipos[n1] == 2:
                        l = 0.
                        new_sfb = [0, 0]
                        new_sfb[0] = ms_nods_n.index(n1) # el primer nodo de la siguiente subfibra sera el ultimo nodo de la anterior
        # debug
        print ms_nods_n
        print ms_nods_r
        print ms_nods_t
        print ms_sfbs_c
        print ms_sfbs_l
        return ms_nods_r, ms_sfbs_c


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

    def pre_graficar_bordes(self, L):
        # seteo
        if not self.pregraficado:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            margen = 0.1*L
            self.ax.set_xlim(left=0.-margen, right=L+margen)
            self.ax.set_ylim(bottom=0.-margen, top=L+margen)
            self.pregraficado = True
        # dibujo los bordes del rve
        fron = []
        fron.append( [[0.,L], [0.,0.]] )
        fron.append( [[0.,0.], [L,0.]] )
        fron.append( [[0.,L], [L,L]] )
        fron.append( [[L,L], [L,0.]] )
        plt_fron0 = self.ax.plot(fron[0][0], fron[0][1], linestyle=":")
        plt_fron1 = self.ax.plot(fron[1][0], fron[1][1], linestyle=":")
        plt_fron2 = self.ax.plot(fron[2][0], fron[2][1], linestyle=":")
        plt_fron3 = self.ax.plot(fron[3][0], fron[3][1], linestyle=":")

    def pre_graficar_subfibras(self):
        # seteo
        if not self.pregraficado:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            margen = 0.1*L
            self.ax.set_xlim(left=0.-margen, right=L+margen)
            self.ax.set_ylim(bottom=0.-margen, top=L+margen)
            self.pregraficado = True
        # grafico las subfibras
        for i in range(self.sfs.num):
            xx = list() # valores x
            yy = list() # valores y
            # son dos nodos por subfibra
            n0, n1 = self.sfs.get_con_sf(i)
            r0 = self.nodos.x[n0]
            r1 = self.nodos.x[n1]
            xx = [r0[0], r1[0]]
            yy = [r0[1], r1[1]]
            p = self.ax.plot(xx, yy, label=str(i))