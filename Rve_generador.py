import numpy as np 
from matplotlib import pyplot as plt

def iguales(x, y, tol=1.0e-8):
    """ returns True if x is equal to y, both floats """
    if np.abs(x-y) < tol:
        return True 
    else: 
        return False

def calcular_interseccion(seg1,seg2):
    # paralelismo
    paralelos = np.abs(seg1.get_theta()-seg2.get_theta())<np.pi*1.0e-8 
    if paralelos:
        return None # no hay chance de interseccion porque son paralelos los segmentos
    # lejania
    minRight = np.minimum(seg1.get_right(), seg2.get_right()) # nodo de la derecha mas a la izquierda entre los dos segmentos
    maxLeft = np.maximum(seg1.get_left(), seg2.get_left()) # nodo de la izquierda mas a la derecha entre los dos segmentos
    minTop = np.minimum(seg1.get_top(), seg2.get_top()) # nodo de la derecha mas a la izquierda entre los dos segmentos
    maxBottom = np.maximum(seg1.get_bottom(), seg2.get_bottom()) # nodo de la derecha mas a la izquierda entre los dos segmentos
    # ahora chequeo
    lejos = ( (maxLeft-minRight) > 1.0e-8 ) or ( (maxBottom-minTop) > 1.0e-8 ) 
    if lejos: 
        return None # no hay chance de interseccion porque estan lejos los segmentos
    # si no son paralelos y si tienen cercania entonces encuentro la interseccion
    # usando un sistema de referencia intrinseco al segmento 1 (chi, eta)
    theta_rel = seg2.get_theta() - seg1.get_theta() # angulo que forma el segmento 2 con el eje chi (intrinseco al segmento 1)
    # me fijo que no sean verticales en el sistema intrinseco (para no manejar pendientes infinitas)
    m_rel_inf = ( np.abs(theta_rel-np.pi*0.5)<np.pi*1.0e-8 ) or (np.abs(theta_rel-np.pi*1.5)<np.pi*1.0e-8)
    # coordenadas en (chi,eta) de los nodos del segmento 2
    # previamente variables auxiliares
    dx0 = seg2.n0.get_x()-seg1.n0.get_x()
    dy0 = seg2.n0.get_y()-seg1.n0.get_y()
    dx1 = seg2.n1.get_x()-seg1.n1.get_x()
    dy1 = seg2.n1.get_y()-seg1.n1.get_y()
    C = np.cos(seg1.get_theta())
    S = np.sin(seg1.get_theta())
    # ahora si coordenadas relativas del segmento 2
    chi0 = dx0*C + dy0*S
    eta0 = -dx0*S + dy0*C
    chi1 = dx1*C + dy1*S
    eta1 = -dx1*S + dy1*C
    # chequeo que el segmento 1 realmente corte al eje del 2 
    if np.sign(eta0) == np.sign(eta1):
        return None # si eta no cambia de signo entonces no corta al eje
    # supere varios chequeos
    # calculo valor de chi de la interseccion
    if m_rel_inf:
        chi_in = chi0 
    else: 
        m_rel = np.tan(theta_rel)
        chi_in = chi0 - eta0/m_rel
    # si la interseccion fue por fuera del segmento 1 no es real
    fuera_de_seg1 = (chi_in<0) or (chi_in>seg1.get_dl())
    if fuera_de_seg1:
        return None
    else:
        # la interseccion se da dentro del segmento
        x_in = seg1.get_x0() + chi_in*np.cos(seg1.get_theta()) 
        y_in = seg1.get_y0() + chi_in*np.sin(seg1.get_theta())
        r = np.array([x_in, y_in])
        return r

class Nodo(object):
    """
    El objeto nodo tiene su coordenada r (2d)
    tipo: 0 (nodo comun conectando dos segmentos)
          1 (nodo de frontera)
          2 (nodo de interseccion de dos segmentos que se tocan)
    """
    def __init__(self):
        self.r = None 
        self.__tipo = 0 
        self.segmentos = []
        self.fibras = []

    def set_as_frontera(self):
        self.__tipo = 1

    def get_if_frontera(self):
        return self.__tipo == 1

    def get_x(self):
        # devuelve la coordenada x 
        return self.r[0]

    def get_y(self):
        # devuelve la coordenada y 
        return self.r[1]

    @classmethod
    def from_r(cls, r): 
        instance = cls() 
        instance.r = r
        return instance

    def add_segmento(self, segmento):
        self.segmentos.append(segmento) 

    def add_fibra(self, fibra):
        self.fibras.append(fibra)

    def mover(self, new_r):
        # cambio la posicion
        self.r = new_r
        # y actualizo los segmentos que comparten al nodo
        for seg in self.segmentos: 
            seg.update_from_Nodos()


class Segmento(object):
    def __init__(self):
        self.fibra = None
        self._n0 = None 
        self._n1 = None 
        self.intersectado = False # indica si el segmento sufrio intersecciones con otros
        self.ni = [] # nodos interseccion
        self.__theta = 0
        self.__dl = 0
        self.__dr = None
        self.__m = 0 
        self.__m_inf = [False, False] # [+inf, -inf]
        self.__bottom = 0
        self.__right = 0
        self.__top = 0
        self.__left = 0

    @property
    def n0(self):
        return self._n0 

    @n0.setter
    def n0(self, nodo):
        if self._n0 == None:
            self._n0 = nodo
            nodo.add_segmento(self)
        else: 
            raise AttributeError("No se puede setear un nodo de un segmento mas de una vez, hay que moverlo!")

    @property
    def n1(self):
        return self._n1

    @n1.setter
    def n1(self, nodo):
        if self._n1 == None:
            self._n1 = nodo
            nodo.add_segmento(self)
        else: 
            raise AttributeError("No se puede setear un nodo de un segmento mas de una vez, hay que moverlo!")

    def set_as_intersectado(self):
        self.intersectado = True 

    def set_as_no_intersectado(self):
        self.intersectado = False

    def get_if_intersectado(self):
        return self.intersectado

    def add_nodo_interseccion(self, nodo_i):
        self.ni.append(nodo_i)
        nodo_i.add_segmento(self)
        self.set_as_intersectado()

    def get_r0(self):
        return self.n0.r

    def get_r1(self):
        return self.n1.r

    def get_x0(self):
        return self.n0.get_x()

    def get_y0(self):
        return self.n0.get_y()

    def get_x1(self):
        return self.n1.get_x()

    def get_y1(self):
        return self.n1.get_y()

    def get_theta(self):
        return self.__theta 

    def get_dl(self):
        return self.__dl 

    def get_dr(self):
        return self.__dr 

    def get_dx(self):
        return self.get_dr()[0] 

    def get_dy(self):
        return self.get_dr()[1]

    def get_m(self):
        return self.__m 

    def set_vertical_arriba(self):
        self.__m_inf = [True, False]

    def set_vertical_abajo(self):
        self.__m_inf = [False, True]

    def set_no_vertical(self):
        self.__m_inf = [False, False]

    def get_vertical_arriba(self):
        return self.__m_inf[0]

    def get_vertical_abajo(self):
        return self.__m_inf[0]
    
    def get_vertical(self):
        return any(self.__m_inf)

    def get_bottom(self):
        return self.__bottom 

    def get_right(self):
        return self.__right

    def get_top(self):
        return self.__top

    def get_left(self):
        return self.__left

    @classmethod
    def from_walk(cls, nodo_ini, theta, dl):
        instance = cls()
        instance.n0 = nodo_ini
        instance.__theta = theta
        dr = dl * np.array([np.cos(theta), np.sin(theta)])
        r1 = nodo_ini.r + dr 
        instance.n1 = Nodo.from_r(r1)
        instance.update_from_Nodos()
        return instance

    @classmethod 
    def from_coordenadas(cls, r0, r1):
        instance = cls()
        instance.n0 = Nodo.from_r(r0)
        instance.n1 = Nodo.from_r(r1)
        instance.update_from_Nodos() 
        return instance

    @classmethod 
    def from_nodos(cls, nodo_ini, nodo_fin):
        instance = cls()
        instance.n0 = nodo_ini
        instance.n1 = nodo_fin
        instance.update_from_Nodos()
        return instance

    def update_from_Nodos(self):
        """ con los nodos calcula el resto de los parametros del segmento """
        # vector nodo (dr)
        r0 = self.n0.r
        r1 = self.n1.r
        self.__dr = r1-r0
        # longitud, angulo y pendiente 
        self.calcular_dl_theta_y_m()
        # left right top bottom
        self.__left =  np.minimum(r0[0], r1[0])
        self.__right = np.maximum(r0[0], r1[0])
        self.__bottom =  np.minimum(r0[1], r1[1])
        self.__top = np.maximum(r0[1], r1[1])

    def calcular_dl_theta_y_m(self):
        """ a partir del vector de segmento dr
        calcula la longitud, el angulo y la pendiente del segmento """
        # variables
        dx = self.get_dx()
        dy = self.get_dy()
        # longitud
        dl = np.sqrt(dx**2 + dy**2)
        self.__dl = dl
        # angulo y pendiente juntos segun cuadrante
        self.set_no_vertical()
        self.__m = None
        if iguales(dx,0):
            # segmento vertical
            if iguales(dy,0):
                raise ValueError("Error, segmento de longitud nula!!")
            elif dy>0:
                self.__theta = np.pi*.5
                self.set_vertical_arriba()
            else:
                self.__theta = -np.pi*.5
                self.set_vertical_abajo()
        elif iguales(dy,0):
            # segmento horizontal
            self.__m = 0.0
            if dx>0:
                self.__theta = 0.0
            else:
                self.__theta = np.pi
        else:
            # segmento oblicuo
            self.__m = dy/dx
            if dx<0:
                # segundo o tercer cuadrante
                self.__theta = np.pi + np.arctan(dy/dx)
            elif dy>0:
                # primer cuadrante (dx>0)
                self.__theta = np.arctan(dy/dx)
            else:
                # dx>0 and dy<0
                # cuarto cuadrante
                self.__theta = 2.0*np.pi + np.arctan(dy/dx)


class Fibra(object):
    def __init__(self):
        self.nodos = []
        self.nodos_in = []
        self.segmentos = []

    def add_nodo(self, newNodo):
        self.nodos.append(newNodo)
        newNodo.add_fibra(self)

    def add_nodo_interseccion(self, newNodo):
        self.nodos_in.append(newNodo)
        newNodo.add_fibra(self)

    def add_segmento(self, newSegmento):
        self.segmentos.append(newSegmento) 
        newSegmento.fibra = self

    def get_last_segmento(self):
        return self.segmentos[-1]

    def make_primer_segmento(self, n0, theta, dl):
        # construyo el primer segmento a partir de un nodo
        primerSeg = Segmento.from_walk(n0, theta, dl)
        # agrego nodos y segmento a la conectividad
        self.add_nodo(n0)
        self.add_segmento(primerSeg) 
        self.add_nodo(primerSeg.n1)

    def make_next_segmento(self, dtheta, dl):
        # calculo un segmento extra de una fibra existente
        # uso el segmento previo para continuar con variacion en angulo
        segPrevio = self.get_last_segmento()
        newTheta = segPrevio.get_theta() + dtheta * (2.0*np.random.rand() - 1.0) 
        nodPrevio = segPrevio.n1 
        # construyo el segmento nuevo
        newSegmento = Segmento.from_walk(nodPrevio, newTheta, dl)
        # agrego el segmento y el nodo nuevo a la conectividad
        self.add_segmento(newSegmento) 
        self.add_nodo(newSegmento.n1)


class Layer(object):
    def __init__(self, L, dl, dtheta):
        self.L = L
        self.dl = dl
        self.dtheta = dtheta
        self.fibras = [] 
        self.segmentos = []
        self.nodos = []
        self.nodos_in = [] # nodos interseccion

    def set_L(self, value):
        self.L = value

    def get_L(self):
        return self.L

    def set_dl(self, value):
        self.dl = value

    def get_dl(self):
        return self.dl

    def set_dtheta(self, value):
        self.dtheta = value 

    def get_dtheta(self):
        return self.dtheta 

    def add_fibra(self, fibra):
        self.fibras.append(fibra) 

    def add_segmento(self, segmento):
        self.segmentos.append(segmento) 

    def add_nodo(self, nodo):
        self.nodos.append(nodo)

    def add_nodo_interseccion(self, nodo):
        self.nodos_in.append(nodo)

    def get_punto_sobre_frontera(self):
        boundary = np.random.randint(4)
        d = np.random.rand() * self.get_L()
        if boundary==0:
            x = d
            y = 0
        elif boundary==1:
            x = self.get_L()
            y = d
        elif boundary==2:
            x = self.get_L() - d 
            y = self.get_L() 
        elif boundary==3:
            x = 0
            y = self.get_L() - d
        return np.array([x,y]), boundary

    def check_fuera_del_RVE(self, r):
        x = r[0] 
        y = r[1] 
        L = self.get_L()
        if x<0 or x>L or y<0 or y>L: 
            return True 
        else:
            return False 

    def make_fibra(self):
        # construyo el objeto fibra vacio
        f = Fibra()
        # obtengo coordenadas de un punto en la frontera del rve
        r0, b0 = self.get_punto_sobre_frontera()
        # creo un objeto nodo
        n0 = Nodo.from_r(r0)
        n0.set_as_frontera()
        # agrego el nodo inicial a la conectividad
        self.add_nodo(n0)
        # angulo inicial
        theta0 = np.random.rand() * np.pi + float(b0)*0.5*np.pi
        # primer segmento de fibra
        f.make_primer_segmento(n0, theta0, self.dl)
        # agrego a la conectividad el segmento nuevo y nodo nuevo
        newSegmento = f.get_last_segmento()
        newNodo = newSegmento.n1 
        self.add_segmento(newSegmento)
        self.add_nodo(newNodo)
        # construyo el resto de los segmentos
        while True:
            lastNodo = f.get_last_segmento().n1
            # si el segmento cae fuera del rve ya se termino la fibra
            arafue = self.check_fuera_del_RVE(lastNodo.r)
            if arafue:
                lastNodo.set_as_frontera()
                break
            # si cae dentro entonces hay un nuevo segmento
            f.make_next_segmento(self.dtheta, self.dl)
            # agrego a la conectividad el segmento nuevo y nodo nuevo
            newSegmento = f.get_last_segmento()
            newNodo = newSegmento.n1 
            self.add_segmento(newSegmento)
            self.add_nodo(newNodo)
            # # chequeo por intersecciones con las demas fibras de la capa
            # for fibra2 in self.fibras:
            #     num_ins, mask_ins, r_ins = compute_intersecciones(nuevoSeg,fibra2.segmentos)
            #     if num_ins>0:
            #         for r_in in r_ins[mask_ins]:
            #             interseccion = Interseccion(f,fibra2,r_in)
            #             self.intersecciones.append(interseccion)
        # finalmente agrego la fibra nueva a la lista
        self.add_fibra(f)

    def calcular_interecciones(self):
        # recorro fibra a fibra, segmento a segmento
        # cada fibra solo puede intersectar con las demas
        num_fibras = len(self.fibras) 
        for jf in range(num_fibras):
            fibra1 = self.fibras[jf] # fibra de la que quiero chequear intersecciones
            fibras2 = self.fibras[jf+1:] # fibra contra las que quiero chequear intersecciones
            # tengo que chequear intersecciones de cada segmento de fibra1
            # contra cada segmento de cada fibra de fibras2 
            for seg1 in fibra1.segmentos:
                    # me traigo todos los segmentos de las demas fibras
                    segs2 = []
                    for fibra2 in fibras2:
                        segs2 += fibra2.segmentos
                    # chequeo intersecciones con cada uno
                    for seg2 in segs2:
                        if seg1.get_if_intersectado() or seg2.get_if_intersectado():
                            continue # estoy admitiendo solo una interseccion por segmento 
                        else:
                            # ahora si calculo intersecciones
                            r_in = calcular_interseccion(seg1, seg2)
                            if r_in is not None:
                                # debo crear nodos interseccion y agregarlos a la lista
                                nuevoNodo_in = Nodo.from_r(r_in) 
                                self.add_nodo_interseccion(nuevoNodo_in)
                                # agrego el nodo a las fibras y los segmentos
                                # ademas seteo los segmentos como intersectados
                                seg1.fibra.add_nodo_interseccion(nuevoNodo_in)
                                seg2.fibra.add_nodo_interseccion(nuevoNodo_in)
                                seg1.add_nodo_interseccion(nuevoNodo_in)
                                seg2.add_nodo_interseccion(nuevoNodo_in)



    def graficar(self):
        # seteo
        fig = plt.figure()
        ax = fig.add_subplot(111)
        margen = 0.1*self.L
        ax.set_xlim(left=0-margen, right=self.L+margen)
        ax.set_ylim(bottom=0-margen, top=self.L+margen)
        # preparar el grafico de las fronteras
        fron = []
        fron.append( [[0,self.L], [0,0]] )
        fron.append( [[0,0], [self.L,0]] )
        fron.append( [[0,self.L], [self.L,self.L]] )
        fron.append( [[self.L,self.L], [self.L,0]] )
        plt_fron0 = ax.plot(fron[0][0], fron[0][1], linestyle=":")
        plt_fron1 = ax.plot(fron[1][0], fron[1][1], linestyle=":")
        plt_fron2 = ax.plot(fron[2][0], fron[2][1], linestyle=":")
        plt_fron3 = ax.plot(fron[3][0], fron[3][1], linestyle=":")
        # preparar el grafico de fibras
        for f in self.fibras:
            xx = [] 
            yy = []
            for n in f.nodos:
                xx.append(n.r[0])
                yy.append(n.r[1])
            plt_f = ax.plot(xx, yy, linestyle="-")
        # preparar el grafico de todos los putos nodos
        xx_n = []
        yy_n = []
        xx_f = [] # frontera
        yy_f = [] # frontera
        for n in self.nodos:
            if n.get_if_frontera():
                xx_f.append(n.r[0])
                yy_f.append(n.r[1])
            xx_n.append(n.r[0])
            yy_n.append(n.r[1])
        plt_n = ax.plot(xx_n, yy_n, linewidth=0, marker="x")                
        plt_nf = ax.plot(xx_f, yy_f, linewidth=0, marker="o", mfc="none", mec="k")
        # preparar el grafico de intersecciones 
        xx_in = []
        yy_in = []
        for n in self.nodos_in:
            xx_in.append(n.r[0])
            yy_in.append(n.r[1])
        plt_in = ax.plot(xx_in, yy_in, linewidth=0, marker="s", mec="k")
        
        # graficar
        plt.show()

    def get_simplified_connectivity(self):
        """ devuelve la conectividad de la siguiente manera
        numero de nodos
        numero de nodos frontera
        numero de nodos interseccion
        coordenadas (lista de listas)
        numero de subfibras
        conectividad (subfibras, lista de listas)
        """
        # calculo el numero efectivo de nodos (fronteras e intersecciones)
        num_nodos = 0 
        num_nodos_fr = 0
        for n in self.nodos:
            if n.get_if_frontera():
                num_nodos += 1 
                num_nodos_fr += 1
        num_nodos_in = len(self.nodos_in)
        num_nodos += num_nodos_in
        # armo el array de coordenadas de nodos
        coordenadas = np.zeros((num_nodos, 2), dtype=float)
        # lo lleno con las coordenadas
        j_n = 0
        for n in self.nodos:
            if n.get_if_frontera():
                coordenadas[j_n,:] = n.r 
                n.id_number = j_n
                j_n += 1 
        for n_in in self.nodos_in: 
            coordenadas[j_n,:] = n_in.r 
            n_in.id_number = j_n
            j_n += 1
        # ahora calculo la conectividad de las fibras
        # primero el numero de nodos de cada fibra
        # teniendo en cuenta los fronterizos y los interseccion
        num_fibras = len(self.fibras)
        num_nod_x_fibra = np.zeros(num_fibras, dtype=int)
        for j_f, f in enumerate(self.fibras):
            # numero de nodos por fibra de la fibra
            num_nod_x_fibra[j_f] = 2 + len(f.nodos_in)
        # ahora si la conectividad
        max_nodos_por_fibra = np.max(num_nod_x_fibra)
        conec_fibras = np.zeros( (num_fibras, max_nodos_por_fibra), dtype=int)
        for j_f, f in enumerate(self.fibras):
            num_nodos_f = num_nod_x_fibra[j_f]
            conec_fibras[j_f,0] = f.nodos[0].id_number
            for j_n_in, n_in in enumerate(f.nodos_in):
                conec_fibras[j_f, j_n_in+1] = f.nodos_in[j_n_in].id_number
            conec_fibras[j_f, num_nodos_f-1] = f.nodos[-1].id_number
        # ahora conectividad de subfibras (entre dos nodos siempre)
        # primero calculo la cantidad de subfibras
        # en total y por fibra
        num_sf_x_fibra = np.zeros(num_fibras, dtype=int)
        for j_f in range(num_fibras):
            num_nodos_f = num_nod_x_fibra[j_f]
            num_sf_x_fibra[j_f] = num_nodos_f - 1
        num_sf = np.sum(num_sf_x_fibra)
        # ahora la conectividad de las sf
        # y la conectividad de fibras en sf
        max_sf_x_fibra = np.max(num_sf_x_fibra)
        conec_sf = np.zeros( (num_sf, 2), dtype=int)
        conec_f_sf = np.zeros( (num_fibras, max_sf_x_fibra), dtype=int)
        j_sf = 0
        for j_f in range(num_fibras):
            for j_n_f in range(num_nod_x_fibra[j_f] - 1):
                conec_f_sf[j_f, j_n_f] = j_sf
                n0_id = conec_fibras[j_f, j_n_f]
                n1_id = conec_fibras[j_f, j_n_f+1]
                conec_sf[j_sf,:] = [n0_id, n1_id]
                j_sf += 1
        # # print to debug 
        # for j_n in range(num_nodos):
        #     print coordenadas[j_n,:]
        # for j_f in range(num_fibras):
        #     print conec_fibras[j_f,:]
        # for j_sf in range(num_sf):
        #     print conec_sf[j_sf,:]
        # for j_f in range(num_fibras):
        #     print conec_f_sf[j_f,:]
        return num_nodos, num_nodos_fr, num_nodos_in, coordenadas, num_sf, conec_sf



class Rve(object):
    def __init__(self):
        pass 






# # rve instance
# layer1 = Layer(L=1.0, dl=0.1, dtheta=np.pi*0.1)

# for i in range(5):
#     layer1.make_fibra()
# layer1.calcular_interecciones()

# layer1.get_simplified_connectivity()

# layer1.graficar()