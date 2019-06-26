"""
Malla de fibras discretas compuesta de nodos y subfibras
2 dimensiones
Coordenadas y conectividad
"""
import numpy as np
from matplotlib import cm, pyplot as plt
from Aux import find_string_in_file, calcular_longitud_de_segmento, Conectividad

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
    def from_coordenadas_y_tipos(cls, coors, tipos):
        instance = cls()
        for coor, tipo in zip(coors,tipos):
            instance.check_coor_tipo(coor, tipo)
            instance.num += 1
            instance.x0.append(coor)
            instance.x.append(coor)
            instance.tipos.append(tipo)
        instance.cerrar()
        return instance

    def abrir(self):
        """ es necesario para agregar mas nodos """
        self.x0 = [val for val in self.x0]
        self.x = [val for val in self.x]
        self.tipos = [val for val in self.tipos]
        self.mask_fr = [val for val in self.mask_fr]
        self.mask_in = [val for val in self.mask_in]
        self.open = True

    def get_coors0_fr(self):
        return self.x0[self.mask_fr]

    def set_coors_fr(self, xf):
        self.x[self.mask_fr] = xf

    # ---
    # Ecuaciones de chequeo
    def check_coors_tipos(self, coors, tipos, exhaustivo=False):
        assert isinstance(coors, list)
        assert isinstance(tipos, list)
        assert len(coors) == len(tipos)
        if exhaustivo:
            for coor, tipo in zip(coors, tipos):
                self.check_coor(coor)
                self.check_tipo(tipo)

    def check_coor_tipo(self, coor, tipo):
        self.check_coor(coor)
        self.check_tipo(tipo)

    def check_coor(self, coor):
        assert isinstance(coor, list)
        assert len(coor) == 2
        assert isinstance(coor[0], float)
        assert isinstance(coor[1], float)

    def check_tipo(self, tipo):
        assert isinstance(tipo, int)
    # ---

    def add_nodo(self, coor, tipo, chequear=True):
        if self.open:
            if chequear:
                self.check_coor_tipo(coor, tipo)
            self.num += 1
            self.x0.append(coor)
            self.x.append(coor)
            self.tipos.append(tipo)
        else:
            raise RuntimeError("nodos closed, no se pueden agregar nodos")

    def add_nodos(self, coors, tipos, chequear=True):
        if chequear:
            self.check_coors_tipos(coors, tipos, exhaustivo=True)
        n = len(coors)
        if self.open:
            for i in range(n):
                self.add_nodo(coors[i], tipos[i], chequear=False)
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


class Subfibras(Conectividad):
    """ es una conectividad
    con algunos atributos y metodos particulares """

    def __init__(self):
        Conectividad.__init__(self)
        self.letes0 = None # va a ser un array de numpy luego, longitudes end-to-end iniciales
        self.locos0 = None # tambien va a ser un array de numpy, longitudes de contorno iniciales
        self.lamsr = None # loco0/lete0
        self.ecucon_id = None
        self.ecuacion_constitutiva = None
        self.param_con = None # va a tener que ser un array de (num_subfibras, num_paramcon) presumiendo que cada subfibra pueda tener diferentes valores de parametros

    def set_conectividad(self, conec):
        """ asigna conectividad al objeto de subfibras
        por ahora queda abierta (solo tengo ne y je, como listas) """
        self.add_conec_listoflists(conec) # calcula el ne y el je

    def set_ecuacion_constitutiva(self, param_con, ec_con_id):
        """ asigna parametros constitutivos y ecuacion constitutiva
        param_con tiene que ser un array de (nsubfibras, nparamcon) """
        self.param_con = param_con
        self.ecucon_id = ec_con_id
        self.ecuacion_constitutiva = self.ecuaciones_constitutivas(ec_con_id)

    def cerrar(self, locos0,  coors0):
        Conectividad.cerrar(self)
        self.locos0 = locos0
        self.calcular_letes0(coors0)
        self.lamsr = self.locos0 / self.letes0

    @classmethod
    def closed(cls, conec, locos0, coors0, param_con, ec_con):
        instance = cls()
        instance.add_conec_listoflists(conec) # calcula ne y je
        instance.set_ecuacion_constitutiva(param_con, ec_con)
        instance.cerrar(locos0, coors0) # calcula el ie y se pasan los arrays a numpy, ademas asigna locos0 y calcula letes0 y lamsr
        instance.ecuacion_constitutiva = instance.ecuaciones_constitutivas(ec_con)
        return instance

    @staticmethod
    def ecuaciones_constitutivas(ecucon):
        """ de una lista de funciones devuelve una sola segun el indice ecucon
        consigo un comportamiento similar a un select case usando un diccionario """
        d = dict()
        # ---
        def lineal_con_reclutamiento(lam, lamr, paramcon):
            Et = paramcon[0]
            Eb = paramcon[1]
            if lam<=lamr:
                return Et*(lam-1.)
            else:
                return Eb*(lamr-1.) + Et*(lam/lamr - 1.)
        d[0] = lineal_con_reclutamiento
        # ---
        return d[ecucon]

    def get_con_sf(self, j):
        """ obtener la conectividad de una subfibra (copia de get_con_elem0)"""
        return self.je[ self.ie[j] : self.ie[j+1] ]

    def calcular_lete_j(self, j, coors):
        nod_ini, nod_fin = self.get_con_sf(j)
        x_ini = coors[nod_ini]
        x_fin = coors[nod_fin]
        dr = x_fin - x_ini
        lete = np.sqrt ( np.dot(dr,dr) )
        return lete

    def calcular_letes0(self, coors0):
        """ calcular las longitudes iniciales de todas las subfibras """
        self.letes0 = self.calcular_letes(coors0, self.letes0)

    def calcular_letes(self, coors, longs=None):
        """ calcular las longitudes de extremo a extremo (letes) de las subfibras """
        if longs is None:
            longs = np.zeros(self.num, dtype=float) # seria mejor tenerlo preadjudicado
        for jsf in range(self.num):
            nod_ini, nod_fin = self.get_con_sf(jsf)
            x_ini = coors[nod_ini]
            x_fin = coors[nod_fin]
            dr = x_fin - x_ini
            longs[jsf] = np.sqrt( np.dot(dr,dr) )
        return longs

    def calcular_tension_j(self, j, lam):
        return self.ecuacion_constitutiva(lam, self.lamsr[j], self.param_con[j])

    def calcular_elongacion_j(self, j, coors):
        lete = self.calcular_lete_j(j, coors)
        lam = lete/self.letes0[j]
        return lam

    def calcular_elongaciones(self, coors):
        letes = self.calcular_letes(coors)
        lams = letes/self.letes0
        return lams

    def calcular_tensiones(self, coors):
        tens = np.array( self.num, dtype=float)
        lams = self.calcular_elongaciones(coors)
        tens = self.ecuacion_constitutiva(lams, self.param_con)
        return tens


class Iterador(object):
    def __init__(self, n, sistema, ref_small, ref_big, ref_div, maxiter, tol):
        self.n = n # tamano de la solcion (len(x))
        self.sistema = sistema # ecuacion iterable a resolver (dx = x-x1 = f(x1))
        self.x = np.zeros( np.shape(self.sistema.nodos.x), dtype=float ) # solucion (comienza como semilla, luego es la iterada o actualizada)
        self.dl1 = np.zeros(self.n, dtype=float) # necesario para evaluar convergencia
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
        inestable_big = np.any(self.flag_big)
        inestable_div = np.any(self.flag_div)
        inestable = inestable_big or inestable_div
        if not inestable:
            print "estable: ", self.it
            self.x = self.x + dx # actualizo la solucion
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
            print "inestable: ", self.it, inestable_big, inestable_div
            self.it -= 1
            self.sistema.solventar_inestabilidad(self.flag_big, self.flag_div)

    def iterar(self):
        self.x[:] = self.sistema.nodos.x # esta va a ser mi semilla
        self.it = 0
        self.flag_primera_iteracion = True
        self.convergencia = False
        self.maxiter_alcanzado = False
        while True:
            self.iterar1()
            if self.convergencia or self.maxiter_alcanzado:
                return self.x


class Malla(object):
    def __init__(self):
        self.nodos = Nodos()
        self.sfs = Subfibras() # las subfibras las conecto con los nodos que las componen
        self.psv = None
        self.L = None # tamano del rve
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
        return instance

    @classmethod
    def from_malla_completa(cls, mc, par_con_IN, ecu_con, psvis):
        ms = cls()
        ms.simplificar_malla_completa(mc, par_con_IN, ecu_con, psvis)
        return ms


    def setear_nodos(self, coors, tipos):
        self.nodos.add_nodos(coors, tipos)
        self.nodos.cerrar()

    def setear_subfibras(self, conec_listoflists, locos_0, coors_0, par_con, euc_con):
        self.sfs.set_conectividad(conec_listoflists)
        self.sfs.set_ecuacion_constitutiva(par_con, euc_con)
        self.sfs.cerrar(locos_0, coors_0)

    def setear_pseudoviscosidad(self, psv):
        self.psv = np.array(psv, dtype=float)

    # @classmethod
    # def leer_de_archivo_malla_completa(cls, archivo, par_con_IN, ecu_con, psvis):
    #     fid = open(archivo, "r")
    #     # primero leo los parametros
    #     target = "*parametros"
    #     ierr = find_string_in_file(fid, target, True)
    #     #L, dl, dtheta = (float(val) for val in fid.next().split())
    #     L = float(fid.next())
    #     # leo los nodos
    #     target = "*coordenadas"
    #     ierr = find_string_in_file(fid, target, True)
    #     num_r = int(fid.next())
    #     coors = list()
    #     tipos = list()
    #     for i in range(num_r):
    #         j, t, x, y = (float(val) for val in fid.next().split())
    #         tipos.append(int(t))
    #         coors.append([x,y])
    #     # luego los segmentos
    #     target = "*segmentos"
    #     ierr = find_string_in_file(fid, target, True)
    #     num_s = int(fid.next())
    #     segs = list()
    #     for i in range(num_s):
    #         j, n0, n1 = (int(val) for val in fid.next().split())
    #         segs.append([n0,n1])
    #     # luego las fibras
    #     target = "*fibras"
    #     ierr = find_string_in_file(fid, target, True)
    #     num_f = int(fid.next())
    #     fibs = list()
    #     for i in range(num_f):
    #         out = [int(val) for val in fid.next().split()]
    #         j = out[0]
    #         fcon = out[1:]
    #         fibs.append(fcon)
    #     # ---
    #     # ahora puedo armar la malla simplificada:
    #     # nodos interseccion y subfibras
    #     # ---
    #     ms_nods_r = list() # coordenadas de la malla simplificada
    #     ms_nods_t = list() # tipos de los nodos de la malla simplificada
    #     ms_nods_n = list() # indices originales de los nodos
    #     ms_sfbs_c = list() # conectividad de subfibras de la malla simplificada
    #     ms_sfbs_l = list() # largo de las siguiendo el contorno de los segmentos
    #     # recorro cada fibra:
    #     for f in range(len(fibs)): # f es el indice de cada fibra en la malla completa
    #         # tengo una nueva subfibra
    #         new_sfb = [0, 0] # por ahora esta vacia
    #         # agrego el primer nodo
    #         s = fibs[f][0] # segmento 0 de la fibra f
    #         n0 = segs[s][0] # nodo 0 del segmento s
    #         ms_nods_r.append( coors[n0] )
    #         ms_nods_t.append( tipos[n0] ) # deberia ser un 1
    #         ms_nods_n.append( n0 )
    #         # lo agrego a la nueva subfibra como su primer nodo
    #         new_sfb[0] = len( ms_nods_r ) - 1 # es el ultimo nodo agregado, tambien podria hacer ms_nods_n.index(n0) que es lo mismo pero mas lento
    #         assert ms_nods_t[-1] == 1
    #         # recorro el resto de los nodos de la fibra para agregar los nodos intereccion
    #         l = 0. # aca voy sumando el largo de los segmentos que componen la subfibra
    #         for js in range(len(fibs[f])): # js es el indice de cada segmento en la fibra f (numeracion local a la fibra)
    #             # voy viendo los nodos finales de cada segmento
    #             s = fibs[f][js] # s es el indice de cada segmento en la malla original (numeracion global)
    #             n0 = segs[s][0] # primer nodo del segmento
    #             n1 = segs[s][1] # nodo final del segmento
    #             r0 = coors[n0]
    #             r1 = coors[n1]
    #             dx = r1[0] - r0[0]
    #             dy = r1[1] - r0[1]
    #             l += np.sqrt( dx*dx + dy*dy ) # largo del segmento (lo sumo al largo de la subfibra)
    #             if tipos[n1] in (1,2): # nodo interseccion (2) o nodo final (1)
    #                 # tengo que fijarme si el nodo no esta ya presente
    #                 # (ya que los nodos interseccion pertenecen a dos fibras)
    #                 if not n1 in ms_nods_n:
    #                     # el nodo no esta listado,
    #                     # tengo que agregarlo a la lista de nodos
    #                     ms_nods_r.append( coors[n1] )
    #                     ms_nods_t.append( tipos[n1] )
    #                     ms_nods_n.append( n1 )
    #                 # me fijo en la nueva numeracion cual es el indice del ultimo nodo de la subfibra
    #                 new_sfb[1] = ms_nods_n.index(n1)
    #                 # y agrega la conectividad de la subfibra a la lista
    #                 ms_sfbs_c.append( new_sfb )
    #                 ms_sfbs_l.append( l )
    #                 # si no llegue a un nodo frontera, debo empezar una nueva subfibra
    #                 if tipos[n1] == 2:
    #                     l = 0.
    #                     new_sfb = [0, 0]
    #                     new_sfb[0] = ms_nods_n.index(n1) # el primer nodo de la siguiente subfibra sera el ultimo nodo de la anterior
    #     # # debug
    #     # print ms_nods_n
    #     # print ms_nods_r
    #     # print ms_nods_t
    #     # print ms_sfbs_c
    #     # print ms_sfbs_l
    #     # return ms_nods_r, ms_sfbs_c
    #     # ---
    #     # armo los objetos
    #     # los nodos a partir de las coordenadas y los tipos
    #     nodos = Nodos.from_coordenadas_y_tipos(ms_nods_r, ms_nods_t)
    #     # las subfibras con los parmetros constitutivos para cada subfibra
    #     # calculo el lamr por subfibra
    #     par_con = np.zeros( (len(ms_sfbs_c), len(par_con_IN)), dtype=float )
    #     for i in range(len(ms_sfbs_c)):
    #         r0_i = ms_nods_r[ ms_sfbs_c[i][0] ]
    #         r1_i = ms_nods_r[ ms_sfbs_c[i][1] ]
    #         lete_i = calcular_longitud_de_segmento(r0_i, r1_i)
    #         lamr_i = ms_sfbs_l[i] / lete_i
    #         par_con_i = par_con_IN
    #         par_con_i[2] = lamr_i
    #         par_con[i,:] = par_con_i
    #     sfbs = Subfibras.closed(ms_sfbs_c, ms_sfbs_l, nodos.x, par_con, ecu_con)
    #     psv = [psvis]*nodos.num
    #     malla = Malla.closed(nodos, sfbs, psv)
    #     malla.L = L
    #     return malla

    def simplificar_malla_completa(self, mc, par_con_IN, ecu_con, psvis):
        """ toma una malla completa (mc)
        y construye una malla simplifiada
        es necesario dar ecuacion y parametros constitutivos
        mc es una instancia de malla completa
        par_con_IN es un array con los parametros generales que se van a copiar para todas las fibras
        ecu_con es un entero que indica cual ecuacion usar
        psvis es un float que se va a copiar a todos los nodos """
        # obtengo lo que necesito de la malla completa
        # recordar que en la malla completa los objetos suelen ser listas (no arrays de numpy)
        L = mc.L
        coors = mc.nods.r
        tipos = mc.nods.tipos
        segs = mc.segs.con
        fibs = mc.fibs.con
        # los voy a mapear a otras listas propias de la malla simplificada
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
        # ---
        # ahora coloco las variables en mi objeto malla simplificada
        # nodos con coordenadas y tipos
        self.L = L
        self.setear_nodos(ms_nods_r, ms_nods_t)
        # subfibras con parametros constitutivos y ecuacion constitutiva
        # ademas tengo que pasar las longitudes de contorno y las coordenadas para calcular los enrulamientos
        par_con = np.zeros( (len(ms_sfbs_c), len(par_con_IN)), dtype=float )
        for par_con_i in par_con:
            par_con_i[:] = par_con_IN
        self.setear_subfibras(ms_sfbs_c, ms_sfbs_l, self.nodos.x0, par_con, ecu_con)
        # pseudoviscosidad
        psv = [psvis] * len(ms_nods_r)
        self.setear_pseudoviscosidad(psv)

    def guardar_en_archivo(self, archivo):
        fid = open(archivo, "w")
        # ---
        # escribo parametros: L (por lo pronto solo ese)
        dString = "*Parametros \n"
        dString += "{:17.8f}".format(self.L) + "\n"
        fid.write(dString)
        # escribo los nodos: indice, tipo, y coordenadas
        dString = "*Coordenadas \n" + str(self.nodos.num) + "\n"
        fid.write(dString)
        for n in range( self.nodos.num ):
            dString = "{:6d}".format(n)
            dString += "{:2d}".format(self.nodos.tipos[n])
            dString += "".join( "{:+17.8e}".format(val) for val in self.nodos.x0[n] ) + "\n"
            fid.write(dString)
        # ---
        # sigo con las subfibras: indice, nodo inicial y nodo final, long contorno, long end-to-end, enrulamiento
        dString = "*Subfibras \n" + str( self.sfs.num ) + "\n"
        fid.write(dString)
        for s in range( self.sfs.num ):
            n0, n1 = self.sfs.get_con_sf(s)
            loco = self.sfs.locos0[s]
            lete = self.sfs.letes0[s]
            lamr = self.sfs.lamsr[s]
            fmt = "{:6d}"*3 + "{:17.8e}"*3
            dString = fmt.format(s, n0, n1, loco, lete, lamr) + "\n"
            fid.write(dString)
        # ---
        # sigo con los parametros constitutivos para las subfibras
        nParam = np.shape( self.sfs.param_con )[1]
        dString = "*Constitutivos \n"
        dString += "{:6d}".format(nParam) + "\n"
        dString += "{:6d}".format(self.sfs.ecucon_id) + "\n"
        fid.write(dString)
        for s in range( self.sfs.num ):
            dString = "".join( "{:17.8e}".format(val) for val in self.sfs.param_con[s] ) + "\n"
            fid.write(dString)
        # ---
        dString = "*End \n"
        fid.write(dString)
        # termine
        fid.close()

    @classmethod
    def leer_de_archivo(self, archivo="Malla_simplificada.txt"):
        fid = open(archivo, "r")
        # primero leo los parametros
        target = "*parametros"
        ierr = find_string_in_file(fid, target, True)
        linea = fid.next().split()
        L = float(linea[0])
        # luego busco coordenadas
        target = "*coordenadas"
        ierr = find_string_in_file(fid, target, True)
        num_r = int(fid.next())
        nods_coors = list()
        nods_tipos = list()
        for i in range(num_r):
            # j, t, x, y = (float(val) for val in fid.next().split())
            svals = fid.next().split() # valores como strings
            j = int(svals[0]) # deberia coincidir con i, es el indice
            t = int(svals[1])
            x = float(svals[2])
            y = float(svals[3])
            nods_tipos.append(t)
            nods_coors.append([x,y])
        # luego las subfibras
        target = "*subfibras"
        ierr = find_string_in_file(fid, target, True)
        num_sfs = int(fid.next())
        sfbs_conec = list()
        sfbs_locos = list()
        sfbs_letes = list()
        sfbs_lamsr = list()
        for s in range(num_sfs):
            svals = fid.next().split()
            j = int(svals[0]) # deberia coincidir con s, es el indice
            n0 = int(svals[1])
            n1 = int(svals[2])
            loco = float(svals[3])
            lete = float(svals[4])
            lamr = float(svals[5])
            sfbs_conec.append([n0,n1])
            sfbs_locos.append(loco)
            sfbs_letes.append(lete)
            sfbs_lamsr.append(lamr)
        # luego los parametros constitutivos
        target = "*constitutivos"
        ierr = find_string_in_file(fid, target, True)
        nParam = int(fid.next())
        ecucon_id = int(fid.next())
        sfbs_paramcon = list()
        for s in range(num_sfs):
            svals = fid.next().split()
            paramcon_s = [float(val) for val in svals]
            sfbs_paramcon.append(paramcon_s)
        # termine

    def get_x(self):
        return self.nodos.x

    def set_x(self, x):
        self.nodos.x[:] = x

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
            lam = dl / self.sfs.letes0[jsf]
            a = dr/dl
            t = self.sfs.calcular_tension_j(jsf, lam)
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

    def deformar_afin_frontera(self, F):
        """ mueve los nodos de la frontera de manera afin segun el
        gradiente de deformaciones (tensor F de 2x2) """
        xf0 = self.nodos.get_coors0_fr()
        xf = np.matmul( xf0, np.transpose(F) )
        self.nodos.set_coors_fr(xf)

    def deformar_afin(self, F):
        """ mueve todos los nodos de manera afin segun el F """
        x0 = self.nodos.x0
        x = np.matmul( x0, np.transpose(F))
        self.nodos.x[:] = x

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
        la idea es que este metodo sea la funcion dx=f(x1) """
        # si no mando valor significa que quiero averiguar las tracciones
        # con las coordenadas que tengan los nodos de la malla
        if x1 is None:
            dx = np.zeros( np.shape(self.nodos.x), dtype=float )
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
        self.psv[nodos_criticos] = 1.05*self.psv[nodos_criticos]

    def pre_graficar_bordes0(self, color_borde="gray"):
        # seteo
        if not self.pregraficado:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            margen = 0.1*self.L
            self.ax.set_xlim(left=0.-margen, right=self.L+margen)
            self.ax.set_ylim(bottom=0.-margen, top=self.L+margen)
            self.pregraficado = True
        # dibujo los bordes del rve
        x0 = 0.
        x1 = self.L
        x2 = self.L
        x3 = 0.
        y0 = 0.
        y1 = 0.
        y2 = self.L
        y3 = self.L
        xx = [x0, x1, x2, x3, x0]
        yy = [y0, y1, y2, y3, y0]
        plt_fron = self.ax.plot(xx, yy, linestyle=":", color=color_borde)

    def pre_graficar_bordes(self, F, color_borde="k"):
        # seteo
        if not self.pregraficado:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.pregraficado = True
        margen = 0.1*self.L
        izq = 0. - margen * np.maximum(F[0,0], F[1,1])
        der = (self.L + margen) * np.maximum(F[0,0], F[1,1])
        aba = izq
        arr = der
        self.ax.set_xlim(left=izq, right=der)
        self.ax.set_ylim(bottom=aba, top=arr)

        # dibujo los bordes del rve
        x0 = 0.
        x1 = self.L*F[0,0]
        x2 = self.L*(F[0,0]+F[0,1])
        x3 = self.L*F[0,1]
        y0 = 0.
        y1 = self.L*F[1,0]
        y2 = self.L*(F[1,1]+F[1,0])
        y3 = self.L*F[1,1]
        xx = [x0, x1, x2, x3, x0]
        yy = [y0, y1, y2, y3, y0]
        plt_fron = self.ax.plot(xx, yy, linestyle=":", color=color_borde)

    def pre_graficar_subfibras0(self, linestyle=":", colorbar=False):
        # previamente hay que haber llamado a pre_graficar_bordes
        assert self.pregraficado == True
        # preparo los colores segun el enrulamiento
        mi_colormap = cm.rainbow
        sm = cm.ScalarMappable(cmap=mi_colormap, norm=plt.Normalize(vmin=np.min(self.sfs.lamsr), vmax=np.max(self.sfs.lamsr)))
        # grafico las subfibras
        for i in range(self.sfs.num):
            xx = list() # valores x
            yy = list() # valores y
            # son dos nodos por subfibra
            n0, n1 = self.sfs.get_con_sf(i)
            r0 = self.nodos.x0[n0]
            r1 = self.nodos.x0[n1]
            xx = [r0[0], r1[0]]
            yy = [r0[1], r1[1]]
            col = sm.to_rgba(self.sfs.lamsr[i])
            p = self.ax.plot(xx, yy, label=str(i), linestyle=linestyle, color=col)
        if colorbar:
            sm._A = []
            self.fig.colorbar(sm)

    def pre_graficar_subfibras(self, colores=False, colorbar=False):
        # previamente hay que haber llamado a pre_graficar_bordes
        assert self.pregraficado == True
        # preparo los colores segun el estiramiento (relativo a lamr)
        mi_colormap = cm.rainbow
        lams = self.sfs.calcular_elongaciones(self.nodos.x)
        sm = cm.ScalarMappable(cmap=mi_colormap, norm=plt.Normalize(vmin=np.min(lams), vmax=np.max(lams)))
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
            col = sm.to_rgba( lams[i] / self.sfs.lamsr[i])
            p = self.ax.plot(xx, yy, label=str(i), linestyle="-", color=col)
            if not colores:
                p[0].set(color="gray")
        if colorbar:
            sm._A = []
            self.fig.colorbar(sm)

    def pre_graficar0(self, linestyle=":", colorbar=False):
        self.pre_graficar_bordes0()
        self.pre_graficar_subfibras0(linestyle, colorbar=colorbar)

    def pre_graficar(self, F, colores=False):
        self.pre_graficar_bordes(F)
        self.pre_graficar_subfibras(colores=colores, colorbar=True)

    def graficar0(self):
        self.pre_graficar0(linestyle="-", colorbar=True)
        self.ax.legend(loc="upper left", numpoints=1, prop={"size":6})
        plt.show()

    def graficar(self, F, inicial=True, colores=True):
        if inicial:
            self.pre_graficar0()
        self.pre_graficar(F, colores=colores)
        plt.show()