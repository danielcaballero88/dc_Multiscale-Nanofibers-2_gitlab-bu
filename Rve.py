import numpy as np 
from matplotlib import pyplot as plt

def iguales(x, y, tol=1.0e-8):
    """ returns True if x is equal to y, both floats """
    if np.abs(x-y) < tol:
        return True 
    else: 
        return False

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
        self.fibras = []
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

    def add_fibra(self, fibra):
        # agrego la fibra a la conectividad del segmento
        self.fibras.append(fibra)

    @property
    def n0(self):
        return self._n0 

    @n0.setter
    def n0(self, nodo):
        self._n0 = nodo
        nodo.add_segmento(self)

    @property
    def n1(self):
        return self._n1

    @n1.setter
    def n1(self, nodo):
        self._n1 = nodo
        nodo.add_segmento(self)

    def set_as_intersectado(self):
        self.intersectado = True 

    def set_as_no_intersectado(self):
        self.intersectado = False

    def get_if_intersectado(self):
        return self.intersectado

    def add_nodo_interseccion(self, nodo_i):
        self.ni.append(nodo_i)
        nodo_i.add_segmento(self)

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
        self.segmentos = []

    def add_nodo(self, newNodo):
        self.nodos.append(newNodo)
        newNodo.add_fibra(self)

    def add_segmento(self, newSegmento):
        self.segmentos.append(newSegmento) 
        newSegmento.add_fibra(self)

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
        # angulo inicial
        theta0 = np.random.rand() * np.pi + float(b0)*0.5*np.pi
        # primer segmento de fibra
        f.make_primer_segmento(n0, theta0, self.dl)
        # resto de segmentos
        while True:
            lastNodo = f.get_last_segmento().n1
            # si el segmento cae fuera del rve ya se termino la fibra
            arafue = self.check_fuera_del_RVE(lastNodo.r)
            if arafue:
                lastNodo.set_as_frontera()
                break
            # si cae dentro entonces hay un nuevo segmento
            f.make_next_segmento(self.dtheta, self.dl)
            # # chequeo por intersecciones con las demas fibras de la capa
            # for fibra2 in self.fibras:
            #     num_ins, mask_ins, r_ins = compute_intersecciones(nuevoSeg,fibra2.segmentos)
            #     if num_ins>0:
            #         for r_in in r_ins[mask_ins]:
            #             interseccion = Interseccion(f,fibra2,r_in)
            #             self.intersecciones.append(interseccion)
        # finalmente agrego la fibra nueva a la lista
        self.add_fibra(f)

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
        # # preparar el grafico de todos los putos nodos
        # xx_n = []
        # yy_n = []
        # for n in self.nodos:
        #     xx_n.append(n.r[0])
        #     yy_n.append(n.r[1])
        # #plt_n = ax.plot(xx_n, yy_n, linewidth=0, marker="x")
        # # preparar el grafico de nodos frontera
        # xx_f = []
        # yy_f = []        
        # for n in self.nodos:    
        #     if n.tipo==1:
        #         xx_f.append(n.r[0])
        #         yy_f.append(n.r[1])
        # # preparar el grafico de intersecciones 
        # xx_in = []
        # yy_in = []
        # for i in self.intersecciones:
        #     xx_in.append(i.r[0])
        #     yy_in.append(i.r[1])
        # plt_in = ax.plot(xx_in, yy_in, linewidth=0, marker="x", mec="k")
        #plt_nf = ax.plot(xx_f, yy_f, linewidth=0, marker="o", mfc="none", mec="k")
        # graficar
        plt.show()

class Rve(object):
    def __init__(self):
        pass 






# rve instance
layer1 = Layer(L=1.0, dl=0.1, dtheta=np.pi*0.1)

for i in range(20):
    layer1.make_fibra()

layer1.graficar()