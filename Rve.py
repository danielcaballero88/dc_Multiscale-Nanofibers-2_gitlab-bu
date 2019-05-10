import numpy as np 

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
        self.tipo = 0 
        self.segmentos = []
        self.fibras = []

    def set_r(self, r):
        self.r = r

    def get_r(self):
        """ devuelve un array con las coordenadas 2d """
        return self.r 

    def set_tipo(self, tipo):
        self.tipo = tipo

    def get_tipo(self):
        return self.tipo

    def set_as_frontera(self):
        self.tipo = 1

    def get_if_frontera(self):
        return self.tipo == 1

    def get_x(self):
        """ devuelve la coordenada x """
        return self.get_r()[0]

    def get_y(self):
        """ devuelve la coordenada y """
        return self.get_r()[1]

    @classmethod
    def from_r(cls, r, tipo=0): 
        instance = cls() 
        instance.set_r(r)
        instance.set_tipo(tipo)
        return instance

    def mover(self, new_r):
        # cambio la posicion
        self.set_r(new_r)
        # y actualizo los segmentos que comparten al nodo
        for seg in self.segmentos: 
            seg.update_from_Nodos()


class Segmento(object):
    def __init__(self):
        self.fibras = []
        self.nodos = [None, None]
        self.theta = 0
        self.dl = 0
        self.dr = None
        self.m = 0 
        self.m_inf = [False, False] # [+inf, -inf]
        self.bottom = 0
        self.right = 0
        self.top = 0
        self.left = 0

    def add_fibra(self, fibra):
        # agrego la fibra a la conectividad del segmento
        self.fibras.append(fibra)
        # y tambien a la conectividad de los nodos del segmento
        for nodo in self.nodos:
            nodo.add_fibra(fibra)

    def set_n0(self, n0):
        self.nodos[0] = n0

    def get_n0(self):
        return self.nodos[0]

    def set_n1(self, n1):
        self.nodos[1] = n1

    def get_n1(self):
        return self.nodos[1]

    def get_r0(self):
        return self.nodos[0].get_r()

    def get_r1(self):
        return self.nodos[1].get_r()

    def get_x0(self):
        return self.nodos[0].get_x()

    def get_y0(self):
        return self.nodos[0].get_y()

    def get_x1(self):
        return self.nodos[1].get_x()

    def get_y1(self):
        return self.nodos[1].get_y()

    def set_theta(self, theta):
        self.theta = theta 

    def get_theta(self, theta):
        return self.theta

    def set_dr(self, dr):
        self.dr = dr

    def get_dr(self):
        return self.dr 

    def set_dl(self, dl):
        self.dl = dl 

    def get_dl(self):
        return self.dl

    def get_dx(self):
        return self.get_dr()[0] 

    def get_dy(self):
        return self.get_dr()[1]

    def set_m(self, m):
        self.m = m 

    def get_m(self):
        return self.m

    def set_vertical_arriba(self):
        self.m_inf = [True, False]

    def set_vertical_abajo(self):
        self.m_inf = [False, True]

    def set_no_vertical(self):
        self.m_inf = [False, False]

    def get_vertical_arriba(self):
        return self.m_inf[0]

    def get_vertical_abajo(self):
        return self.m_inf[0]
    
    def get_vertical(self):
        return any(self.m_inf)

    def set_bottom(self, bottom):
        self.bottom = bottom 

    def get_bottom(self):
        return self.bottom 

    def set_right(self, right):
        self.right = right 

    def get_right(self):
        return self.right

    def set_top(self, top):
        self.top = top 

    def get_top(self):
        return self.top

    def set_left(self, left):
        self.left = left 

    def get_left(self):
        return self.left

    @classmethod
    def from_walk(cls, r0, theta, dl, fibra_madre):
        instance = cls()
        instance.add_fibra(fibra_madre)
        instance.set_n0(Nodo.from_r(r0))
        instance.set_theta(theta)
        dr = dl * np.array([np.cos(theta), np.sin(theta)])
        r1 = r0 + dr 
        instance.set_n1(Nodo.from_r(r1))
        instance.update_from_Nodos()
        return instance

    @classmethod 
    def from_coordenadas(cls, r0, r1, fibra_madre):
        instance = cls()
        instance.add_fibra(fibra_madre)
        instance.set_n0(Nodo.from_r(r0))
        instance.set_n1(Nodo.from_r(r1))
        instance.update_from_Nodos() 
        return instance

    @classmethod 
    def from_nodos(cls, n0, n1):
        instance = cls()
        instance.set_n0(n0)
        instance.set_n1(n1)
        instance.update_from_Nodos()
        return instance

    def update_from_Nodos(self):
        """ con los nodos calcula el resto de los parametros del segmento """
        # vector nodo (dr)
        r0 = self.get_n0().get_r() 
        r1 = self.get_n1().get_r() 
        self.set_dr(r1-r0)
        # longitud, angulo y pendiente 
        self.calcular_dl_theta_y_m()
        # left right top bottom
        self.set_left( np.minimum(r0[0], r1[0]) )
        self.set_right( np.maximum(r0[0], r1[0]) )
        self.set_bottom( np.minimum(r0[1], r1[1]) )
        self.set_top( np.maximum(r0[1], r1[1]) )

    def calcular_dl_theta_y_m(self):
        """ a partir del vector de segmento dr
        calcula la longitud, el angulo y la pendiente del segmento """
        # variables
        dx = self.get_dx()
        dy = self.get_dy()
        # longitud
        dl = np.sqrt(dx**2 + dy**2)
        self.set_dl(dl)
        # angulo y pendiente juntos segun cuadrante
        self.set_no_vertical()
        self.set_m(None )
        if iguales(dx,0):
            # segmento vertical
            if iguales(dy,0):
                raise ValueError("Error, segmento de longitud nula!!")
            elif dy>0:
                self.set_theta(np.pi*.5)
                self.set_vertical_arriba()
            else:
                self.set_theta(-np.pi*.5)
                self.set_vertical_abajo()
        elif iguales(dy,0):
            # segmento horizontal
            self.set_m(0.0)
            if dx>0:
                self.set_theta(0.0)
            else:
                self.set_theta(np.pi)
        else:
            # segmento oblicuo
            self.set_m(dy/dx)
            if dx<0:
                # segundo o tercer cuadrante
                self.set_theta(np.pi + np.arctan(dy/dx))
            elif dy>0:
                # primer cuadrante (dx>0)
                self.set_theta(np.arctan(dy/dx))
            else:
                # dx>0 and dy<0
                # cuarto cuadrante
                self.set_theta(2.0*np.pi + np.arctan(dy/dx))


class Fibra(object):
    def __init__(self):
        pass 

class Layer(object):
    def __init__(self):
        pass 

class Rve(object):
    def __init__(self):
        pass 

