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
    def get_if_frontera(self):
    def get_x(self): # devuelve la coordenada x
    def get_y(self): # devuelve la coordenada y
    @classmethod
    def from_r(cls, r): 
    def add_segmento(self, segmento):
    def add_fibra(self, fibra):
    def mover(self, new_r): # cambio la posicion y actualizo los segmentos que comparten al nodo
        



class Segmento(object):
    def __init__(self):
        self.fibras = []
        self.n0 = None # detalle: tiene un metodo setter que agrega al segmento a los segmentos del nodo
        self.n1 = None # mismo detalle que n0
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

    def add_fibra(self, fibra):  # agrego la fibra a la conectividad del segmento
    def set_as_intersectado(self):
    def set_as_no_intersectado(self):
    def get_if_intersectado(self):
    def add_nodo_interseccion(self, nodo_i): # mismo detalle que n0 y n1
    def get_r0(self):
    def get_r1(self):
    def get_x0(self):
    def get_y0(self):
    def get_x1(self):
    def get_y1(self):
    def get_theta(self):
    def get_dl(self):
    def get_dr(self):
    def get_dx(self):
    def get_dy(self):
    def get_m(self):
    def set_vertical_arriba(self):
    def set_vertical_abajo(self):
    def set_no_vertical(self):
    def get_vertical_arriba(self):
    def get_vertical_abajo(self):
    def get_vertical(self):
    def get_bottom(self):
    def get_right(self):
    def get_top(self):
    def get_left(self):
    @classmethod
    def from_walk(cls, nodo_ini, theta, dl): # construye una instancia y la devuelve
    @classmethod 
    def from_coordenadas(cls, r0, r1): # idem
    @classmethod 
    def from_nodos(cls, nodo_ini, nodo_fin): # idem
    def update_from_Nodos(self): # con los nodos calcula el resto de las variables dependientes del segmento
    def calcular_dl_theta_y_m(self): # a partir del vector de segmento dr calcula la longitud, el angulo y la pendiente del segmento




class Fibra(object):
    def __init__(self):
        self.nodos = []
        self.segmentos = []

    def add_nodo(self, newNodo): # ademas agrega a la propia fibra a la lista del nodo
    def add_segmento(self, newSegmento): # ademas agrega a la propia fibra a la lista del segmento
    def get_last_segmento(self):
    def make_primer_segmento(self, n0, theta, dl): # construyo el primer segmento a partir de un nodo
    def make_next_segmento(self, dtheta, dl): # calculo un segmento extra de una fibra existente, uso el segmento previo para continuar con variacion en angulo
  