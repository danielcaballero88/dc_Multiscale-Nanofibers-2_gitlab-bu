"""
Modulo para ensamblar una malla de fibras con intersecciones
La malla tiene tres especies: fibras, segmentos y nodos.
Los segmentos se componen de dos nodos
Las fibras se componen de muchos segmentos (random walk)
Cada especie tiene una numeracion global
Las fibras tienen una conectividad que son los indices de los segmentos que la componen
Los segmentos tienen una conectividad dada por los indices de los dos nodos que lo componen
Los nodos tienen coordenadas y tipo (0=continuacion, 1=frontera, 2=interseccion)
"""


import numpy as np
from matplotlib import pyplot as plt
# import collections
import TypedLists
from Aux import iguales, calcular_interseccion_entre_segmentos as calcular_interseccion, find_string_in_file
from Aux import calcular_longitud_de_segmento


class Nodos(object):
    def __init__(self):
        self.r = TypedLists.Lista_de_listas_de_dos_floats()  # coordenadas de los nodos
        self.tipos = TypedLists.Lista_de_algunos_enteros((0,1,2)) # lista de tipos (0=cont, 1=fron, 2=inter)

    def add_nodo(self, r_nodo, tipo):
        self.r.append(r_nodo)
        self.tipos.append(tipo)

    def get_r(self, i_nodo):
        return self.r[i_nodo]

    def __len__(self):
        if not len(self.r) == len(self.tipos):
            raise ValueError, "longitudes de coordenadas y tipos no concuerdan"
        else:
            return len(self.r)


class Segmentos(object):
    def __init__(self):
        self.con = TypedLists.Lista_de_listas_de_dos_enteros() # lista de listas de dos nodos (indices)
        self.thetas = TypedLists.Lista_de_floats()
        self.longs = TypedLists.Lista_de_floats()

    def add_segmento(self, seg_con, coors):
        """
        aca las coordenadas las necesito para calcularle a cada segmento su longitud y angulo
        seg_con es la conectividad (2 nodos) del segmento
        coors son las coordenadas (lista de listas de a dos floats) de todos los nodos
        (con todos los nodos hasta el momento de crear este segmento esta bien,
        alcanza con que esten presentes en la lista los dos nodos de seg_con) """
        self.con.append(seg_con)
        longitud, angulo = self.calcular_long_y_theta(seg_con, coors)
        self.thetas.append(angulo)
        self.longs.append(longitud)

    def actualizar_segmento(self, j, coors):
        """ en caso de que se mueva un nodo y haya que actualizar theta y longitud """
        long, ang = self.calcular_long_y_theta(self.con[j], coors)
        self.thetas[j] = ang
        self.longs[j] = long

    def mover_nodo(self, j, n, coors, new_r):
        """ mueve un nodo del segmento
        coors es una lista, es un objeto mutable
        por lo que al salir de este metodo se va ver modificada
        es decir, es un puntero
        j es el indice del segmento a moverle un nodo
        n es el indice del nodo para el segmento: 0 es inicial, 1 es final """
        assert n in (0,1)
        nglobal = self.con[j][n]
        coors[nglobal] = new_r # se lo modifica resida donde resida (normalmente en un objeto nodos)
        self.actualizar_segmento(j, coors)

    def cambiar_conectividad(self, j, new_con, coors):
        """ se modifica la conectividad de un segmento (j) de la lista
        se le da la nueva conectividad new_con
        y por lo tanto se vuelve a calcular su angulo y longitud
        (util para dividir segmentos en 2) """
        self.con[j] = new_con
        longitud, angulo = self.calcular_long_y_theta(new_con, coors)
        self.thetas[j] = angulo
        self.longs[j] = longitud

    @staticmethod
    def calcular_long_y_theta(seg, coors):
        n0 = seg[0]
        n1 = seg[1]
        dx = coors[n1][0] - coors[n0][0]
        dy = coors[n1][1] - coors[n0][1]
        long = np.sqrt( dx*dx + dy*dy )
        # ahora theta
        if iguales(dx,0.0):
            # segmento vertical
            if iguales(dy,0.0,1.0e-12):
                raise ValueError("Error, segmento de longitud nula!!")
            elif dy>0:
                theta = np.pi*.5
            else:
                theta = -np.pi*.5
        elif iguales(dy,0):
            # segmento horizontal
            if dx>0:
                theta = 0.0
            else:
                theta = np.pi
        else:
            # segmento oblicuo
            if dx<0:
                # segundo o tercer cuadrante
                theta = np.pi + np.arctan(dy/dx)
            elif dy>0:
                # primer cuadrante (dx>0)
                theta = np.arctan(dy/dx)
            else:
                # dx>0 and dy<0
                # cuarto cuadrante
                theta = 2.0*np.pi + np.arctan(dy/dx)
        return long, theta

    def get_right(self, j, coors):
        n0 = self.con[j][0]
        n1 = self.con[j][1]
        x0 = coors[n0][0]
        x1 = coors[n1][0]
        return np.maximum(x0,x1)

    def get_left(self, j, coors):
        n0 = self.con[j][0]
        n1 = self.con[j][1]
        x0 = coors[n0][0]
        x1 = coors[n1][0]
        return np.minimum(x0,x1)

    def get_top(self, j, coors):
        n0 = self.con[j][0]
        n1 = self.con[j][1]
        y0 = coors[n0][1]
        y1 = coors[n1][1]
        return np.maximum(y0,y1)

    def get_bottom(self, j, coors):
        n0 = self.con[j][0]
        n1 = self.con[j][1]
        y0 = coors[n0][1]
        y1 = coors[n1][1]
        return np.minimum(y0,y1)

    def get_dx(self, j, coors):
        n0 = self.con[j][0]
        n1 = self.con[j][1]
        x0 = coors[n0][0]
        x1 = coors[n1][0]
        return x1-x0

    def get_dy(self, j, coors):
        n0 = self.con[j][0]
        n1 = self.con[j][1]
        y0 = coors[n0][1]
        y1 = coors[n1][1]
        return y1-y0

    def get_dx_dy_brtl(self, j, coors):
        n0 = self.con[j][0]
        n1 = self.con[j][1]
        x0 = coors[n0][0]
        y0 = coors[n0][1]
        x1 = coors[n1][0]
        y1 = coors[n1][1]
        return x1-x0, y1-y0, np.minimum(y0,y1), np.maximum(x0,x1), np.maximum(y0,y1), np.minimum(x0,x1)


class Fibras(object):
    """ es algo como una lista con algunas funciones particulares
    tiene un atributo con (conectividad) que es una lista
    pero la propia instancia se comporta como la lista """
    def __init__(self):
        self.con = TypedLists.Lista_de_listas_de_enteros() # conectividad: va a ser una lista de listas de segmentos (sus indices nada mas), cada segmento debe ser una lista de 2 nodos
        self.dls = TypedLists.Lista_de_floats()
        self.dthetas = TypedLists.Lista_de_floats()

    def add_fibra(self, fib_con, dl, dtheta):
        self.con.append(fib_con)
        self.dls.append(dl)
        self.dthetas.append(dtheta)

    def add_seg_a_fibra(self, j, seg):
        # agrego seg
        assert isinstance(seg, int)
        self.con[j].append(seg)

    def nueva_fibra_vacia(self, dl, dtheta):
        # agrego una nueva fibra, vacia por ahora
        self.con.append( list() )
        self.dls.append(dl)
        self.dthetas.append(dtheta)

    def add_seg_a_fibra_actual(self, seg):
        # argrego seg a la ultima fibra
        assert isinstance(seg, int)
        n = len(self.con)
        assert n>=1
        self.con[n-1].append(seg)

    def insertar_segmento(self, j, k, s):
        """ inserta un segmento en la conectividad de una fibra
        j: indice de la fibra
        k: indice donde se inserta el nuevo segmento
        s: indice del nuevo segmento para agregar a la conectividad """
        self.con[j].insert(k,s)

class Capas(object):
    """ es como una lista de fibras que compone cada capa """
    def __init__(self):
        self.con = TypedLists.Lista_de_listas_de_enteros()

    def set_capas_listoflists(self, capas_con):
        self.__init__()
        for capa_con in capas_con:
            self.add_capa(capa_con)

    def add_capa(self, cap_con):
        self.con.append(cap_con)

    def add_fibra_a_capa(self, capa, fibra):
        """ capa y fibra son los indices """
        self.con[capa].append[fibra]


class Malla(object):
    def __init__(self, L):
        self.L = L
        self.caps = Capas() # lista vacia
        self.fibs = Fibras() # lista vacia
        self.segs = Segmentos() # lista vacia
        self.nods = Nodos() # tiene dos listas vacias
        self.bordes_n = Nodos() # lista de coordenadas con los 4 nodos del borde
        self.bordes_s = Segmentos() # lista con los segmentos con los 4 nodos del borde
        self.calcular_marco()
        self.pregraficado = False
        self.fig = None
        self.ax = None

    def calcular_marco(self):
        # agrego los 4 nodos
        self.bordes_n.add_nodo([0., 0.], 1)
        self.bordes_n.add_nodo([self.L, 0.], 1)
        self.bordes_n.add_nodo([self.L, self.L], 1)
        self.bordes_n.add_nodo([0., self.L], 1)
        # agrego los 4 segmentos
        self.bordes_s.add_segmento([0,1], self.bordes_n.r)
        self.bordes_s.add_segmento([1,2], self.bordes_n.r)
        self.bordes_s.add_segmento([2,3], self.bordes_n.r)
        self.bordes_s.add_segmento([3,0], self.bordes_n.r)

    def make_capa(self, dl, dtheta, nfibs):
        """
        armo una capa con nfibs fibras, todas van a armarse con los
        mismos parmetros dl y dtheta (se debe modificar para usar distribuciones)
        """
        ncapas = len(self.caps.con)
        capa_con = list()
        for i in range(nfibs):
            j = self.make_fibra(dl, dtheta)
            capa_con.append(j)
        self.caps.add_capa(capa_con)

    def make_fibra(self, dl, dtheta):
        """ tengo que armar una lista de segmentos
        nota: todos los indices (de nodos, segmentos y fibras)
        son globales en la malla, cada nodo nuevo tiene un indice +1 del anterior
        idem para segmentos y fibras
        los indices de los nodos, de los segmentos y de las fibras van por separado
        es decir que hay un nodo 1, un segmento 1 y una fibra 1
        pero no hay dos de misma especie que compartan indice """
        # ---
        # primero hago un segmento solo
        # para eso pongo un punto sobre la frontera del rve y el otro lo armo con un desplazamiento recto
        # tomo un angulo random entre 0 y pi, saliente del borde hacia adentro del rve
        # eso me da un nuevo segmento
        # agrego todas las conectividades
        # ---
        # preparo la conectividad de una nueva fibra
        f_con = list() # por lo pronto es una lista vacia
        # primero busco un nodo en el contorno
        x0, y0, b0 = self.get_punto_sobre_frontera()
        self.nods.add_nodo([x0,y0], 1) # agrego el nodo
        # ahora armo el primer segmento de la fibra
        theta = np.random.rand() * np.pi + b0*0.5*np.pi # angulo inicial
        dx = dl * np.cos(theta)
        dy = dl * np.sin(theta)
        # tengo el nuevo nodo, lo agrego
        self.nods.add_nodo([x0+dx, y0+dy], 0)
        # puedo armar el primer segmento
        nnods = len(self.nods)
        s0 = [nnods-2, nnods-1] # estos son los indices de los nodos nuevos
        # agrego el segmento (que es la conexion con los 2 nodos)
        self.segs.add_segmento(s0, self.nods.r) # paso las coordenadas para que calcule la longitud y el angulo
        # lo agrego a la fibra
        f_con.append( len(self.segs.con) -1 )
        # ---
        # ya tengo dos nodos nuevos, con sus coordenadas y tipos agregados a self.nods
        # un segmento nuevo, con los indices de esos dos nodos, agregado a self.segs.con y su longitud y angulo en self.segs.longs y self.segs.thetas
        # una fibra nueva, que por ahora tiene solamente el indice de ese nuevo segmento
        # ---
        # ahora agrego nuevos segmentos en un bucle
        while True:
            # si el nodo anterior ha caido fuera del rve ya esta la fibra
            if self.check_fuera_del_RVE(self.nods.r[-1]):
                self.nods.tipos[-1] = 1
                self.trim_fibra_at_frontera(f_con)
                break
            # de lo contrario armo un nuevo segmento a partir del ultimo nodo
            # el angulo puede sufrir variacion
            theta = theta + dtheta * (2.0*np.random.rand() - 1.0)
            # desplazamiento:
            dx = dl * np.cos(theta)
            dy = dl * np.sin(theta)
            # nuevo nodo
            x = self.nods.r[-1][0] + dx
            y = self.nods.r[-1][1] + dy
            self.nods.add_nodo([x,y], 0)
            # nuevo segmento
            s = [len(self.nods)-2, len(self.nods)-1]
            self.segs.add_segmento(s, self.nods.r)
            # lo agrego a la fibra
            f_con.append( len(self.segs.con) -1 )
        # al terminar agrego la conectividad de la fibra a las fibras
        self.fibs.add_fibra(f_con, dl, dtheta)
        return len(self.fibs.con) - 1 # devuelvo el indice de la fibra


    def get_punto_sobre_frontera(self):
        boundary = np.random.randint(4)
        d = np.random.rand() * self.L
        if boundary==0:
            x = d
            y = 0.0
        elif boundary==1:
            x = self.L
            y = d
        elif boundary==2:
            x = self.L - d
            y = self.L
        elif boundary==3:
            x = 0.0
            y = self.L - d
        return x, y, float(boundary)

    def check_fuera_del_RVE(self, r):
        x = r[0]
        y = r[1]
        if x<=0 or x>=self.L or y<=0 or y>=self.L:
            return True
        else:
            return False

    def trim_fibra_at_frontera(self, fib_con):
        """ subrutina para cortar la fibra que ha salido del rve """
        # debo cortar la ultima fibra en su interseccion por el rve
        # para eso calculo las intersecciones de los nodos con los bordes
        # coordenadas del ultimo segmento de la fibra de conectividad fib_con
        s = fib_con[-1]
        rs0 = self.nods.r[ self.segs.con[s][0] ] # coordenadas xy del nodo 0 del segmento s
        rs1 = self.nods.r[ self.segs.con[s][1] ] # coordenadas xy del nodo 1 del segmento s
        # pruebo con cada borde
        for b in range( len(self.bordes_s.con) ): # recorro los 4 bordes
            # puntos del borde en cuestion
            rb0 = self.bordes_n.r[ self.bordes_s.con[b][0] ] # coordenadas xy del nodo 0 del borde b
            rb1 = self.bordes_n.r[ self.bordes_s.con[b][1] ] # coordenadas xy del nodo 1 del borde b
            interseccion = calcular_interseccion(rs0, rs1, rb0, rb1)
            if interseccion is None: # no hubo interseccion
                continue # con este borde no hay interseccion, paso al que sigue
            else: # hubo interseccion
                in_r, in_tipo, in_seg0, in_seg1 = interseccion
                if in_tipo==2: # interseccion en el medio
                    try: # tengo que mover el ultimo nodo y por lo tanto cambia el segmento
                        self.segs.mover_nodo(s, 1, self.nods.r, in_r)
                        rs1 = in_r
                    except ValueError as e:
                        print "error"
                        print fib_con, b, interseccion
                        quit()
                else: # interseccion coincide con uno o dos extremos
                    # en este caso solo me importa el segundo nodo del primer segmento (seg 0)
                    # porque el segmento 1 es el borde, y el primer nodo del seg 0 siempre deberia estar dentro del rve
                    # (o en el borde a lo sumo si se trata de una fibra de un solo segmento)
                    # y en ese caso no hay nada que hacer! puesto que el nodo ya esta en el borde
                    pass

    def cambiar_capas(self, new_ncapas):
        """ un mapeo de las fibras en un numero de capas diferente """
        # me fijo cuantas fibras van a entrar en cada capa
        nfibras = len(self.fibs.con)
        nf_x_capa = int(nfibras / new_ncapas)
        # armo una nueva conectividad de capas
        capas_con = list()
        for c in range(new_ncapas-1): # -1 porque la ultima capa la hare aparte
            print c
            capa_con = range(c*nf_x_capa, (c+1)*nf_x_capa)
            capas_con.append(capa_con)
        # la ultima capa puede tener alguna fibra de mas
        c = new_ncapas - 1
        capa_con = range(c*nf_x_capa, (c+1)*nf_x_capa)
        capas_con.append(capa_con)
        self.caps.set_capas_listoflists(capas_con)

    def intersectar_fibras(self):
        """ recorro las capas y voy intersectando fibras dentro de la misma capa
        y con las capas vecinas """
        print "intersectando fibras"
        for c, cap_con in enumerate(self.caps.con): # recorro las capas
            print ""
            print "capa: ", c
            for fc, f0 in enumerate(cap_con): # recorro las fibras de la capa
                print ""
                print "f0 ", f0
                f0_con = self.fibs.con[f0]
                # chequeo interseccion con las demas fibras de la misma capa
                # dejo fuera las fibras con las que ya he chequeado (y la misma fibra, obviamente)
                if c == len(self.caps.con)-1: # es la ultima capa, solamente tengo que chequear fibras con ella misma
                    fibras1 = cap_con[fc+1:]
                else: # no estoy en la ultima capa, chequeo con la misma capa y con la capa siguiente
                    fibras1 = cap_con[fc+1:] + self.caps.con[c+1]
                for f1 in fibras1:
                    print f1,
                    f1_con = self.fibs.con[f1]
                    for j0, s0 in enumerate(f0_con): # recorro los segmentos de la fibra f0
                        n0_s0, n1_s0 = self.segs.con[s0]
                        r0_s0 = self.nods.get_r(n0_s0)
                        r1_s0 = self.nods.get_r(n1_s0)
                        for j1, s1 in enumerate(f1_con): # recorro los segmentos de la fibra f1
                            n0_s1, n1_s1 = self.segs.con[s1]
                            r0_s1 = self.nods.get_r(n0_s1)
                            r1_s1 = self.nods.get_r(n1_s1)
                            interseccion = calcular_interseccion(r0_s0, r1_s0, r0_s1, r1_s1)
                            if interseccion is None: # no ha habido interseccion
                                continue
                            else:
                                # hubo interseccion
                                in_r, in_tipo, in_e_j0, in_e_j1 = interseccion
                                # dependiendo del tipo tendre un nodo nuevo o no
                                if in_tipo==2: # el nodo interseccion es nuevo
                                    self.nods.add_nodo(in_r, 2)
                                    new_node_index = len(self.nods.r) - 1
                                    # parto en dos los dos segmentos
                                    subseg_j0_0 = [n0_s0, new_node_index]
                                    subseg_j0_1 = [new_node_index , n1_s0]
                                    subseg_j1_0 = [n0_s1, new_node_index]
                                    subseg_j1_1 = [new_node_index , n1_s1]
                                    # para cada segmento, cambio la conectividad de la primera mitad
                                    # y agrego la segunda mitad a la lista como un nuevo segmento
                                    self.segs.cambiar_conectividad(s0, subseg_j0_0, self.nods.r)
                                    self.segs.cambiar_conectividad(s1, subseg_j1_0, self.nods.r)
                                    self.segs.add_segmento(subseg_j0_1, self.nods.r)
                                    self.segs.add_segmento(subseg_j1_1, self.nods.r)
                                    # ahora debo cambiar la conectividad de las fibra
                                    # insertando un segmento en cada fibra en la posicion correcta
                                    index_newseg_f0 = len( self.segs.con ) - 2 # indice del nuevo segmento de la fibra f0 (subseg_j0_1)
                                    index_newseg_f1 = len( self.segs.con ) - 1 # indice del nuevo segmento de la fibra i1 (subseg_j1_1)
                                    self.fibs.insertar_segmento(f0, j0+1, index_newseg_f0)
                                    self.fibs.insertar_segmento(f1, j1+1, index_newseg_f1)

    def calcular_conectividad_de_interfibras(self):
        """ ojo son diferentes a las subfibras de una malla simplificada
        aqui las interfibras son concatenaciones de segmentos entre nodos interseccion
        en una ms las subfibras son una simplificacion de una fibra dando solamente los nodos extremos y enrulamiento """
        infbs_con = list() # conectividad: lista de listas de segmentos
        for f, fcon in enumerate(self.fibs.con): # recorro las fibras
            # cada fibra que empieza implica una nueva interfibra
            infb = list() # conectividad de la interfibra: lista de segmentos
            # tengo que ir agregando segmentos hasta toparme con un nodo interseccion o frontera
            for s in fcon: # recorro los segmentos de la fibra f
                scon = self.segs.con[s]
                n0, n1 = scon
                # agrego el segmento s a la interfibra
                infb.append(s)
                # si el ultimo nodo de s es interseccion o frontera aqui termina la interfibra
                if self.nods.tipos[n1] in (1,2):
                    infbs_con.append(infb) # agrego la interfibra a la conectividad
                    infb = list() # preparo una nueva interfibra vacia para continuar agregando segmentos
        # aqui ya deberia tener una conectividad terminada
        return infbs_con

    def calcular_enrulamientos(self):
        """ calcular para todas las fibras sus longitudes de contorno y
        sus longitudes extremo a extremos (loco y lete)
        y calcula el enrulamiento como lamr=loco/lete """
        lamsr = []
        for fcon in self.fibs.con: # recorro las fibras del rve
            loco = 0.
            for s in fcon: # recorro los segmentos de cada fibra
                scon = self.segs.con[s]
                n0, n1 = scon
                r0 = self.nods.r[n0]
                r1 = self.nods.r[n1]
                loco += calcular_longitud_de_segmento(r0, r1)
            n_ini = self.segs.con[fcon[0]][0]
            n_fin = self.segs.con[fcon[-1]][1]
            r_ini = self.nods.r[n_ini]
            r_fin = self.nods.r[n_fin]
            lete = calcular_longitud_de_segmento(r_ini, r_fin)
            lamsr.append( loco/lete )
        return lamsr

    def calcular_distribucion_de_enrulamiento(self, n=10):
        """ calcular la distribucion de enrulamientos
        para eso subdivido el intervalo total en n subintervalos
        y cuento cuantas fibras caen dentro de cada subintervalo,
        obtengo asi una distribucion discreta (historiograma?) """
        lamsr = self.calcular_enrulamientos()
        lamsr = np.array(lamsr, dtype=float)
        lamr_min = np.min(lamsr)
        lamr_max = np.max(lamsr)
        delta = (lamr_max - lamr_min) / n
        x = list()
        frec = list()
        for i in range(n):
            lamr_ini = lamr_min + i*delta
            lamr_fin = lamr_ini + delta
            x.append(0.5*(lamr_ini+lamr_fin))
            mask = np.logical_and( lamr_ini <= lamsr, lamsr < lamr_fin ) # HOJALDRE creo que el ultimo no entra nunca en intervalo
            frec_i = np.sum( mask )
            frec.append(frec_i)
        return x, delta, frec

    def calcular_enrulamientos_de_interfibras(self):
        """ calcular para todas las interfibras sus longitudes de contorno y
        sus longitudes extremo a extremos (loco y lete)
        y calcula el enrulamiento como lamr=loco/lete """
        lamsr = []
        infbs_con = self.calcular_conectividad_de_interfibras()
        for infb_con in infbs_con: # recorro las interfibras (fibras interectadas) del rve
            loco = 0.
            for s in infb_con: # recorro los segmentos de cada interfibra
                scon = self.segs.con[s]
                n0, n1 = scon
                r0 = self.nods.r[n0]
                r1 = self.nods.r[n1]
                loco += calcular_longitud_de_segmento(r0, r1)
            n_ini = self.segs.con[infb_con[0]][0]
            n_fin = self.segs.con[infb_con[-1]][1]
            r_ini = self.nods.r[n_ini]
            r_fin = self.nods.r[n_fin]
            lete = calcular_longitud_de_segmento(r_ini, r_fin)
            lamsr.append( loco/lete )
        return lamsr

    def calcular_distribucion_de_enrulamiento_de_interfibras(self, lamr_min=None, lamr_max=None, n=10):
        """ calcular la distribucion de enrulamientos
        para eso subdivido el intervalo total en n subintervalos
        y cuento cuantas interfibras caen dentro de cada subintervalo,
        obtengo asi una distribucion discreta (historiograma?) """
        lamsr = self.calcular_enrulamientos_de_interfibras()
        lamsr = np.array(lamsr, dtype=float)
        if lamr_min is None:
            lamr_min = np.min(lamsr)
        if lamr_max is None:
            lamr_max = np.max(lamsr)
        delta = (lamr_max - lamr_min) / n
        x = list()
        frec = list()
        for i in range(n):
            lamr_ini = lamr_min + i*delta
            lamr_fin = lamr_ini + delta
            x.append(0.5*(lamr_ini+lamr_fin))
            mask = np.logical_and( lamr_ini <= lamsr, lamsr < lamr_fin )
            frec_i = np.sum( mask )
            frec.append(frec_i)
        return x, delta, frec

    def guardar_en_archivo(self, archivo="Malla.txt"):
        fid = open(archivo, "w")
        # ---
        # primero escribo L, dl y dtheta
        dString = "*Parametros (L) \n"
        fmt = "{:17.8e}"
        dString += fmt.format(self.L) + "\n"
        fid.write(dString)
        # ---
        # escribo los nodos: indice, tipo, y coordenadas
        dString = "*Coordenadas \n" + str(len(self.nods.r)) + "\n"
        fid.write(dString)
        for n in range( len(self.nods.r) ):
            dString = "{:6d}".format(n)
            dString += "{:2d}".format(self.nods.tipos[n])
            dString += "".join( "{:+17.8e}".format(val) for val in self.nods.r[n] ) + "\n"
            fid.write(dString)
        # ---
        # sigo con los segmentos: indice, nodo inicial y nodo final
        dString = "*Segmentos \n" + str( len(self.segs.con) ) + "\n"
        fid.write(dString)
        for s in range( len(self.segs.con) ):
            n0, n1 = self.segs.con[s]
            fmt = "{:6d}"*3
            dString = fmt.format(s, n0, n1) +"\n"
            fid.write(dString)
        # ---
        # sigo con las fibras: indice, dl, dtheta, y segmentos (conectividad)
        dString = "*Fibras \n" + str( len(self.fibs.con) ) + "\n"
        fid.write(dString)
        for f, fcon in enumerate(self.fibs.con):
            dString = "{:6d}".format(f) # indice
            dString += "{:17.8e}{:+17.8e}".format(self.fibs.dls[f], self.fibs.dthetas[f]) # dl y dtheta
            dString += "".join( "{:6d}".format(val) for val in fcon ) + "\n" # conectividad
            fid.write(dString)
        # termino con las capas: indice y fibras (conectividad):
        dString = "*Capas \n" + str( len(self.caps.con) ) + "\n"
        fid.write(dString)
        for c, ccon in enumerate(self.caps.con):
            dString = "{:+6d}".format(c) # indice
            dString += "".join( "{:6d}".format(val) for val in ccon ) + "\n" # conectividad
            fid.write(dString)
        # ---
        # termine
        fid.close()

    @classmethod
    def leer_de_archivo(cls, archivo="Malla.txt"):
        fid = open(archivo, "r")
        # primero leo los parametros
        target = "*parametros"
        ierr = find_string_in_file(fid, target, True)
        L = float( fid.next() )
        # luego busco coordenadas
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
        dls = list()
        dthetas = list()
        for i in range(num_f):
            svals = fid.next().split()
            j = int(svals[0])
            dl = float(svals[1])
            dtheta = float(svals[2])
            fcon = [int(val) for val in svals[3:]]
            fibs.append(fcon)
            dls.append(dl)
            dthetas.append(dtheta)
        # luego la capas
        target = "*capas"
        ierr = find_string_in_file(fid, target, True)
        num_c = int(fid.next())
        caps = list()
        for c in range(num_c):
            svals = fid.next().split()
            j = int(svals[0])
            ccon = [int(val) for val in svals[1:]]
            caps.append(ccon)
        # ahora que tengo todo armo el objeto
        malla = cls(L)
        # le asigno los nodos
        for i in range(num_r):
            malla.nods.add_nodo(coors[i], tipos[i])
        # le asigno los segmentos
        for i in range(num_s):
            s_con = segs[i]
            malla.segs.add_segmento(s_con, coors)
        # le asigno las fibras
        for i in range(num_f):
            f_con = fibs[i]
            dl = dls[i]
            dtheta = dthetas[i]
            malla.fibs.add_fibra(f_con, dl, dtheta)
        # le asigno las capas
        for c in range(num_c):
            c_con = caps[c]
            malla.caps.add_capa(c_con)
        # listo
        return malla

    def pre_graficar_bordes(self, fig, ax):
        # seteo
        margen = 0.1*self.L
        ax.set_xlim(left=0-margen, right=self.L+margen)
        ax.set_ylim(bottom=0-margen, top=self.L+margen)
        self.pregraficado = True
        # dibujo los bordes del rve
        fron = []
        fron.append( [[0,self.L], [0,0]] )
        fron.append( [[0,0], [self.L,0]] )
        fron.append( [[0,self.L], [self.L,self.L]] )
        fron.append( [[self.L,self.L], [self.L,0]] )
        plt_fron0 = ax.plot(fron[0][0], fron[0][1], linestyle=":")
        plt_fron1 = ax.plot(fron[1][0], fron[1][1], linestyle=":")
        plt_fron2 = ax.plot(fron[2][0], fron[2][1], linestyle=":")
        plt_fron3 = ax.plot(fron[3][0], fron[3][1], linestyle=":")


    def pre_graficar_capas(self, fig, ax):
        nc = len(self.caps.con)
        mi_colormap = plt.cm.raingbow
        sm = plt.cm.ScalarMappable(cmap=mi_colormap, norm=plt.Normalize(vmin=0, vmax=nc-1))
        # dibujo las fibras (los segmentos)
        # preparo las listas, una lista para cada fibra
        xx = [ list() for f in  self.fibs.con ]
        yy = [ list() for f in  self.fibs.con ]
        grafs = list()
        for c, c_con in enumerate(self.caps.con): # recorro las capas
            for f in c_con: # recorro las fibra de la capa
                f_con = self.fibs.con[f]
                # antes de recorrer los segmentos de cada fibra
                # el primer nodo del primer segmento lo agrego antes del bucle
                s = f_con[0] # primer segmento de la fibra f
                n = self.segs.con[s][0] # primer nodo del segmento s
                r = self.nods.r[n] # coordenadas de ese nodo
                xx[f].append(r[0])
                yy[f].append(r[1])
                for s in f_con: # recorro los segmentos de la fibra f
                    s_con = self.segs.con[s]
                    n = s_con[1] # ultimo nodo del segmento s
                    r = self.nods.r[n] # coordenadas de ese nodo
                    xx[f].append(r[0])
                    yy[f].append(r[1])
                grafs.append( ax.plot(xx[f], yy[f], linestyle="-", marker="", label=str(f), color=sm.to_rgba[c]) )
        sm._A = []
        fig.colorbar(sm)

    def pre_graficar_fibras(self, fig, ax, lamr_min=None, lamr_max=None):
        # preparo un mapa de colores mapeable por escalar
        lamsr = self.calcular_enrulamientos()
        mi_colormap = plt.cm.rainbow
        if lamr_min is None:
            lamr_min = np.min(lamsr)
        if lamr_max is None:
            lamr_max = np.max(lamsr)
        sm = plt.cm.ScalarMappable(cmap=mi_colormap, norm=plt.Normalize(vmin=lamr_min, vmax=lamr_max))
        # dibujo las fibras (los segmentos)
        # preparo las listas, una lista para cada fibra
        xx = [ list() for f in  self.fibs.con ]
        yy = [ list() for f in  self.fibs.con ]
        grafs = list()
        for c, c_con in enumerate(self.caps.con): # recorro las capas
            for f in c_con: # recorro las fibra de la capa
                f_con = self.fibs.con[f]
                # antes de recorrer los segmentos de cada fibra
                # el primer nodo del primer segmento lo agrego antes del bucle
                s = f_con[0] # primer segmento de la fibra f
                n = self.segs.con[s][0] # primer nodo del segmento s
                r = self.nods.r[n] # coordenadas de ese nodo
                xx[f].append(r[0])
                yy[f].append(r[1])
                for s in f_con: # recorro los segmentos de la fibra f
                    s_con = self.segs.con[s]
                    n = s_con[1] # ultimo nodo del segmento s
                    r = self.nods.r[n] # coordenadas de ese nodo
                    xx[f].append(r[0])
                    yy[f].append(r[1])
                col = sm.to_rgba(lamsr[f])
                grafs.append( ax.plot(xx[f], yy[f], linestyle="-", marker="", label=str(f), color=col) )
        sm._A = []
        fig.colorbar(sm)


    def pre_graficar_interfibras(self, fig, ax, lamr_min=None, lamr_max=None):
        # preparo un mapa de colores mapeable por escalar
        infbs_con = self.calcular_conectividad_de_interfibras()
        lamsr = self.calcular_enrulamientos_de_interfibras()
        mi_colormap = plt.cm.rainbow
        if lamr_min is None:
            lamr_min = np.min(lamsr)
        if lamr_max is None:
            lamr_max = np.max(lamsr)
        sm = plt.cm.ScalarMappable(cmap=mi_colormap, norm=plt.Normalize(vmin=lamr_min, vmax=lamr_max))
        # dibujo las fibras (los segmentos)
        # preparo las listas, una lista para cada fibra
        xx = [ list() for infb_con in  infbs_con ]
        yy = [ list() for infb_con in  infbs_con ]
        grafs = list()
        for i, infb_con in enumerate(infbs_con): # recorro las interfibras
            # antes de recorrer los segmentos de cada interfibra
            # el primer nodo del primer segmento lo agrego antes del bucle
            s = infb_con[0] # primer segmento de la interfibra i
            n = self.segs.con[s][0] # primer nodo del segmento s
            r = self.nods.r[n] # coordenadas de ese nodo
            xx[i].append(r[0])
            yy[i].append(r[1])
            for s in infb_con: # recorro los segmentos de la interfibra i
                s_con = self.segs.con[s]
                n = s_con[1] # ultimo nodo del segmento s
                r = self.nods.r[n] # coordenadas de ese nodo
                xx[i].append(r[0])
                yy[i].append(r[1])
            col = sm.to_rgba(lamsr[i])
            grafs.append( ax.plot(xx[i], yy[i], linestyle="-", marker="", label=str(i), color=col) )
        sm._A = []
        fig.colorbar(sm)

    def pre_graficar_nodos_frontera(self, fig, ax):
        # dibujo las fibras (los segmentos)
        # preparo las listas, una lista para cada fibra
        xx = [ list() for f in  self.fibs.con ]
        yy = [ list() for f in  self.fibs.con ]
        grafs = list() # un plot para cada fibra
        for f in range(len(self.fibs.con)):  # f es un indice
            # el primer nodo y el ultimo de cada fibra son fronteras
            s = self.fibs.con[f][0] # obtengo el indice del primer segmento de la fibra numero f
            n = self.segs.con[s][0] # obtengo el indice del primer nodo del segmento numero s
            r = self.nods.r[n] # obtengo las coordenadas del nodo numero n
            xx[f].append(r[0])
            yy[f].append(r[1])
            s = self.fibs.con[f][-1] # obtengo el indice del ultimo segmento de la fibra numero f
            n = self.segs.con[s][1] # obtengo el indice del segundo nodo del ultimo numero s
            r = self.nods.r[n] # obtengo las coordenadas del nodo numero n
            xx[f].append(r[0])
            yy[f].append(r[1])
            grafs.append( ax.plot(xx[f], yy[f], linewidth=0, marker="x", mec="k") )

    def pre_graficar_nodos_interseccion(self, fig, ax):
        # dibujo las fibras (los segmentos)
        # preparo las listas, una lista para cada fibra
        xx = list()
        yy = list()
        grafs = list() # un plot para cada fibra
        for n in range(len(self.nods.r)):
            if self.nods.tipos[n] == 2:
                xx.append(self.nods.r[n][0])
                yy.append(self.nods.r[n][1])
        ax.plot(xx, yy, linewidth=0, marker="s", mec="k")

    def pre_graficar_nodos_internos(self, fig, ax):
        # dibujo las fibras (los segmentos)
        # preparo las listas, una lista para cada fibra
        xx = list()
        yy = list()
        grafs = list() # un plot para cada fibra
        for n in range(len(self.nods.r)):
            if self.nods.tipos[n] == 0:
                xx.append(self.nods.r[n][0])
                yy.append(self.nods.r[n][1])
        ax.plot(xx, yy, linewidth=0, marker=".", markersize=1)

    def pre_graficar(self, fig, ax, lamr_min = None, lamr_max = None):
        self.pre_graficar_bordes(fig, ax)
        self.pre_graficar_nodos_frontera(fig, ax)
        self.pre_graficar_nodos_interseccion(fig, ax)
        self.pre_graficar_nodos_internos(fig, ax)
        self.pre_graficar_interfibras(fig, ax, lamr_min=lamr_min, lamr_max=lamr_max)
        ax.legend(loc="upper left", numpoints=1, prop={"size":6})

    def graficar(self, fig=None, ax=None, lamr_min=None, lamr_max=None):
        if ax is None:
            fig, ax = plt.subplots()
        self.pre_graficar(fig, ax, lamr_min, lamr_max)
        plt.show()