import numpy as np 

def iguales(x, y, tol=1.0e-8):
    """ returns True if x is equal to y, both floats """
    if np.abs(x-y) < tol:
        return True 
    else: 
        return False

class Nodo(object):
    def __init__(self):
        self.r = None 
        self.tipo = 0 
        self.segmentos = []
        self.fibras = []

    @classmethod
    def from_r(cls, r, tipo=0): 
        instance = cls() 
        instance.r = r 
        instance.tipo = tipo
        return instance

    def set_r(self, r):
        self.r = r

    def mover(self, r, segs):
        # cambio la posicion
        self.r = r
        # y actualizo los segmentos que comparten al nodo
        for seg in seg: 
            seg.calcular_extras()

class Segmento(object):
    def __init__(self):
        self.nodos = []
        self.fibras = []
        self.theta = 0
        self.dl = 0
        self.dr = None
        self.m = 0 
        self.m_inf = [False, False] # [+inf, -inf]
        self.bottom = 0
        self.right = 0
        self.top = 0
        self.left = 0

    @classmethod
    def from_walk(cls, r0, theta, dl, fibra_madre):
        instance = cls()
        instance.add_fibra(fibra_madre)
        instance.nodos.append(Nodo.from_r(r0)) 
        instance.theta = theta
        instance.dl = dl 
        instance.dr = dl * np.array([np.cos(theta), np.sin(theta)])
        r1 = r0 + dr 
        intance.nodos.append(Nodo.from_r(r1))
        return instance

    @classmethod 
    def from_coordenadas(cls, r0, r1, fibra_madre):
        instance = cls()
        instance.add_fibra(fibra_madre)
        instance.nodos.append(Nodo.from_r(r0))
        instance.nodos.append(Nodo.from_r(r1))
        instance.update_from_Nodos() 
        return instance

    @classmethod 
    def from_nodos(cls, n0, n1):
        instance = cls()
        instance.nodos.append(n0)
        instance.nodos.append(n1)
        instance.update_from_Nodos()
        return instance

    def update_from_Nodos(self):
        """ con los nodos calcula el resto de los parametros del segmento """
        # vector nodo (dr)
        r0 = self.nodos[0].r 
        r1 = self.nodos[1].r 
        self.dr = r1 - r0
        # angulo y pendiente 
        self.theta = self.calcular_theta_m()
        # left right top bottom
        self.left = np.minimum(r0[0], r1[0])
        self.right = np.maximum(r0[0], r1[0])
        self.bottom = np.minimum(r0[1], r1[1])
        self.top = np.maximum(r0[1], r1[1])

    def calcular_theta_m(self):
        """ a partir del vector de segmento dr
        calcula el angulo y la pendiente del segmento """
        self.m_inf = [False, False]
        self.m = None 
        dx = self.dr[0] 
        dy = self.dr[1]
        if iguales(dx,0):
            # segmento vertical
            if iguales(dy,0):
                raise ValueError("Error, segmento de longitud nula!!")
            elif dy>0:
                self.theta = np.pi*.5
                self.m_inf = [True, False]
            else:
                self.theta = -np.pi*.5
                self.m_inf = [False, True]
        elif iguales(dy,0):
            # segmento horizontal
            self.m = 0.0
            if dx>0:
                self.theta = 0.0 
            else:
                self.theta = np.pi
        else:
            # segmento oblicuo
            self.m = dy/dx
            if dx<0:
                # segundo o tercer cuadrante
                self.theta = np.pi + np.arctan(dy/dx)
            elif dy>0:
                # primer cuadrante (dx>0)
                self.theta np.arctan(dy/dx)
            else:
                # dx>0 and dy<0
                # cuarto cuadrante
                self.theta 2.0*np.pi + np.arctan(dy/dx)

    def add_fibra(self, fibra):
        # agrego la fibra a la conectividad del segmento
        self.fibras.append(fibra)
        # y tambien a la conectividad de los nodos del segmento
        for nodo in self.nodos:
            nodo.add_fibra(fibra)

    def get_r0(self):
        return self.nodos[0].r

    def get_r1(self):
        return self.nodos[1].r

    def get_x0(self):
        return self.nodos[0].r[0]

    def get_y0(self):
        return self.nodos[0].r[1]

    def get_x1(self):
        return self.nodos[1].r[0]

    def get_y1(self):
        return self.nodos[1].r[1]

class Fibra(object):
    def __init__(self):
        pass 

class Layer(object):
    def __init__(self):
        pass 

class Rve(object):
    def __init__(self):
        pass 

