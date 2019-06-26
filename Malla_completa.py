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
from Aux import calcular_longitud_de_segmento as calcular_lete
from Aux import calcular_longitud_y_angulo_de_segmento as calcular_long_y_ang
from Aux import Conectividad


class Nodos(object):
    def __init__(self):
        self.n = 0
        self.r = np.zeros( (0,2), dtype=float )
        self.tipos = np.zeros( 0, dtype=int )

    def add_nodo(self, r_nodo, tipo):
        self.n += 1
        self.r = np.append(self.r, r_nodo, axis=0)
        self.tipos = np.append(self.tipos, tipo)

    def add_nodos(self, n_nodos, r_nodos, tipos):
        self.n += n_nodos
        self.r = np.append(self.r, r_nodos, axis=0)
        self.tipos = np.append(self.tipos, tipos)

    def __len__(self):
        return self.n


class Segmentos(object):
    def __init__(self):
        self.con = np.zeros( (0,2), dtype=int ) # lista de a dos nodos
        self.thetas = np.zeros( 0, dtype=float )
        self.longs = np.zeros( 0, dtype=float )

    def add_segmento(self, seg_con, coors):
        """
        aca las coordenadas las necesito para calcularle a cada segmento su longitud y angulo
        seg_con es la conectividad (2 nodos) del segmento
        coors son las coordenadas (lista de listas de a dos floats) de todos los nodos
        (con todos los nodos hasta el momento de crear este segmento esta bien,
        alcanza con que esten presentes en la lista los dos nodos de seg_con) """
        self.con = np.append(self.con, seg_con, axis=0)
        r0, r1 = coors[seg_con, :]
        long, ang = calcular_long_y_ang(r0, r1)
        self.thetas = np.append(self.thetas, ang)
        self.longs = np.append(self.longs, long)

    def add_segmentos(self, segs_con, coors):
        self.con = np.append(self.con, segs_con, axis=0)
        longs = list()
        thetas = list()
        for seg_con in segs_con:
            r0, r1 = coors[seg_con, :]
            long, ang = calcular_long_y_ang(r0, r1)
            longs.append(long)
            thetas.append(ang)
        self.thetas = np.append(self.thetas, thetas)
        self.longs = np.append(self.longs, longs)

    def get_con_seg(self, j):
        """ get conectividad de un segmento segun el indice """
        return self.con[j] # devuelve la fila j

    def set_con_seg(self, j, new_con):
        self.con[j,:] = new_con

    def upd_seg(self, j, coors):
        """ update segmento j:
        en caso de que se mueva un nodo y haya que actualizar theta y longitud """
        seg_con = self.get_con_seg(j)
        r0, r1 = coors[seg_con, :]
        long, ang = calcular_long_y_ang(r0, r1)
        self.thetas[j] = ang
        self.longs[j] = long

    def mover_nodo(self, j, n, coors, new_r):
        """ mueve un nodo del segmento
        coors es un array de numpy, es un objeto mutable
        por lo que al salir de este metodo se va ver modificado
        es decir, es un puntero
        j es el indice del segmento a moverle un nodo
        n es el indice del nodo para el segmento: 0 es inicial, 1 es final """
        nglobal = self.get_con_seg(j)[n]
        coors[nglobal] = new_r # se lo modifica resida donde resida (normalmente en un objeto nodos)
        self.upd_seg(j, coors)

    def cambiar_conectividad_seg_j(self, j, new_con, coors):
        """ se modifica la conectividad de un segmento (j) de la lista
        se le da la nueva conectividad new_con
        y por lo tanto se vuelve a calcular su angulo y longitud
        (util para dividir segmentos en 2) """
        self.set_con_seg(j, new_con)
        r0, r1 = coors[new_con, :]
        long, ang = calcular_long_y_ang(r0, r1)
        self.thetas[j] = ang
        self.longs[j] = long

    def get_right(self, j, coors):
        r0, r1 = coors[self.get_con_seg(j), :]
        return np.maximum(r0[0],r1[0])

    def get_left(self, j, coors):
        r0, r1 = coors[self.get_con_seg(j), :]
        return np.minimum(r0[0],r1[0])

    def get_top(self, j, coors):
        r0, r1 = coors[self.get_con_seg(j), :]
        return np.maximum(r0[1],r1[1])

    def get_bottom(self, j, coors):
        r0, r1 = coors[self.get_con_seg(j), :]
        return np.minimum(r0[1],r1[1])

    def get_dx(self, j, coors):
        r0, r1 = coors[self.get_con_seg(j), :]
        return r1[0]-r0[0]

    def get_dy(self, j, coors):
        r0, r1 = coors[self.get_con_seg(j), :]
        return r1[1]-r1[0]

    def get_dx_dy_brtl(self, j, coors):
        r0, r1 = coors[self.get_con_seg(j), :]
        dx, dy = r1 - r0
        return dx, dy, np.minimum(r0[1],r1[1]), np.maximum(r0[0],r1[0]), np.maximum(r0[1],r1[1]), np.minimum(r0[0],r1[0])


class Fibras(object):
    """ es algo como una lista con algunas funciones particulares
    tiene un atributo con (conectividad) que es una lista
    pero la propia instancia se comporta como la lista """
    def __init__(self):
        self.n = 0
        self.con = Conectividad()
        self.capas = np.array( 0, dtype=int )
        self.dls = np.array( 0, dtype=float )
        self.dthetas = np.array( 0, dtype=float )

    def set_fibras(self, fibs_con, dls, dthetas, capas):
        self.n = len(fibs_con)
        self.con = Conectividad()
        self.con.set_conec_listoflists(fibs_con)
        self.dls = np.array( dls, dtype=float )
        self.dthetas = np.array( dthetas, dtype=float )
        self.capas = np.array( capas, dtype=int )

    def add_fibra(self, fib_con, dl, dtheta, capa):
        self.con.add_elem0(fib_con)
        self.dls = np.append(self.dls, dl)
        self.dthetas = np.append(self.dthetas, dtheta)
        self.capas = np.append(self.capas, capa)

    def add_fibras(self, fibs_con, dls, dthetas, capas):
        self.con.add_elems0(fibs_con)
        self.dls = np.append(self.dls, dls)
        self.dthetas = np.append(self.dthetas, dthetas)
        self.capas = np.append(self.capas, capas)

    def add_seg_a_fibra(self, j, seg):
        """
        agrego seg a la conectividad de la fibra j
        seg debe ser un entero (indice del segmento)
        """
        self.con.add_elem1_to_elem0(j, seg)

    def insertar_segmento_en_fibra(self, j, k, s):
        """ inserta un segmento en la conectividad de una fibra
        j: indice de la fibra
        k: indice donde se inserta el nuevo segmento
        s: indice del nuevo segmento para agregar a la conectividad """
        self.con.insert_elem1_in_elem0(j,k,s)


class Malla(object):
    def __init__(self, L):
        self.L = L
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


    def make_fibra(self, dl, dtheta, capa):
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
        self.fibs.add_fibra(f_con, dl, dtheta, capa)


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

    def intersectar_fibras(self):
        """ recorro las fibras y me fijo las intersecciones entre ellas
        donde se produce una interseccion se crea un nuevo nodo (si no se intersecta en un nodo)
        y se subdividen los segmentos que queden partidos en dos por la interseccion
        finalmente se debe acomodar la conectividad por la aparicion de nuevos segmentos """
        # primero tengo que recorrer todas las fibras y contrastarlas con las demas
        # tomo la primera fibra y me fijo si intersecta a todas las demas
        # despues la segunda fibra y me fijo si intersecta a todas menos a la primera (porque ya me fije)
        # asi sucesivamente, la ultima fibra no necesito
        print "intersectando fibras"
        num_f = len( self.fibs.con )
        for f0 in range(num_f): # recorro todas las fibras
            print "f0: ", f0
            capa0 = self.fibs.capas[f0]
            fibcon0 = self.fibs.con[f0] # tengo la lista de segmentos
            num_s0 = len(fibcon0)
            for f1 in range(f0+1,num_f): # para cada fibra recorro las demas fibras (excepto las previas)
                print f1,
                capa1 = self.fibs.capas[f1]
                if not capa1 in (capa0-1, capa0, capa0+1):
                    continue # paso a la sigueinte fibra, este no interseca con f0 por estar en capas alejadas
                fibcon1 = self.fibs.con[f1]
                num_s1 = len(fibcon1)
                for j0 in range(num_s0): # recorro los segmentos de la fibra 0
                    s0 = fibcon0[j0]
                    n_j0_0, n_j0_1 = self.segs.con[s0]
                    for j1 in range(num_s1):
                        s1 = fibcon1[j1]
                        n_j1_0, n_j1_1 = self.segs.con[s1]
                        # ya tengo los indices de los nodos de los dos segmentos,
                        # ahora puedo obtener sus coordenadas y chequear interseccion
                        r_j0_0 = self.nods.r[n_j0_0]
                        r_j0_1 = self.nods.r[n_j0_1]
                        r_j1_0 = self.nods.r[n_j1_0]
                        r_j1_1 = self.nods.r[n_j1_1]
                        interseccion = calcular_interseccion(r_j0_0, r_j0_1, r_j1_0, r_j1_1)
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
                                subseg_j0_0 = [n_j0_0, new_node_index]
                                subseg_j0_1 = [new_node_index , n_j0_1]
                                subseg_j1_0 = [n_j1_0, new_node_index]
                                subseg_j1_1 = [new_node_index , n_j1_1]
                                # para cada segmento, cambio la conectividad de la primera mitad
                                # y agrego la segunda mitad a la lista como un nuevo segmento
                                self.segs.cambiar_conectividad_seg_j(s0, subseg_j0_0, self.nods.r)
                                self.segs.cambiar_conectividad_seg_j(s1, subseg_j1_0, self.nods.r)
                                self.segs.add_segmento(subseg_j0_1, self.nods.r)
                                self.segs.add_segmento(subseg_j1_1, self.nods.r)
                                # ahora debo cambiar la conectividad de las fibra
                                # insertando un segmento en cada fibra en la posicion correcta
                                index_newseg_f0 = len( self.segs.con ) - 2 # indice del nuevo segmento de la fibra f0 (subseg_j0_1)
                                index_newseg_f1 = len( self.segs.con ) - 1 # indice del nuevo segmento de la fibra i1 (subseg_j1_1)
                                self.fibs.insertar_segmento_en_fibra(f0, j0+1, index_newseg_f0)
                                self.fibs.insertar_segmento_en_fibra(f1, j1+1, index_newseg_f1)
            print ""

    def calcular_enrulamientos(self):
        lamsr = list()
        for f in range(len(self.fibs.con)):
            # tengo que calcular la longitud de contorno (loco) y la longitud extremo-extremo (lete)
            n_ini = self.segs.con[self.fibs.con[f][0]][0]
            n_fin = self.segs.con[self.fibs.con[f][-1]][1]
            r_ini = self.nods.r[n_ini]
            r_fin = self.nods.r[n_fin]
            lete = calcular_lete(r_ini, r_fin)
            # para la loco recorro los segmentos
            loco = 0.
            for sf in range(len(self.fibs.con[f])):
                s = self.fibs.con[f][sf]
                n_ini = self.segs.con[sf][0]
                n_fin = self.segs.con[sf][1]
                r_ini = self.nods.r[n_ini]
                r_fin = self.nods.r[n_fin]
                loco += calcular_lete(r_ini, r_fin)
            # finalmente calculo el valor de enrulamiento
            lamsr.append( lete / loco )

    def calcular_distribucion_de_enrulamientos(self, n=10):
        """ se calcula una distribucion discreta
        para eso divido el rango de enrulamientos en n intervalos """
        raise NotImplementedError

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
        # termino con las fibras: indice, dl, dtheta, capa y segmentos
        dString = "*Fibras \n" + str( len(self.fibs.con) ) + "\n"
        fid.write(dString)
        for f in range( len(self.fibs.con) ):
            nsegs = len(self.fibs.con[f])
            dString = "{:6d}".format(f) # indice
            dString += "{:6d}".format(self.fibs.capas[f]) # capa
            dString += "{:17.8e}{:+17.8e}".format(self.fibs.dls[f], self.fibs.dthetas[f]) # dl y dtheta
            dString += "".join( "{:6d}".format(val) for val in self.fibs.con[f] ) + "\n" # conectividad
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
        capas = list()
        for i in range(num_f):
            svals = fid.next().split()
            j = int(svals[0])
            dl = float(svals[1])
            dtheta = float(svals[2])
            capa = int(svals[3])
            fcon = [int(val) for val in svals[4:]]
            fibs.append(fcon)
            dls.append(dl)
            dthetas.append(dtheta)
            capas.append(capa)
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
            capa = capas[i]
            malla.fibs.add_fibra(f_con, dl, dtheta, capa)
        # listo
        return malla

    def pre_graficar_bordes(self):
        # seteo
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        margen = 0.1*self.L
        self.ax.set_xlim(left=0-margen, right=self.L+margen)
        self.ax.set_ylim(bottom=0-margen, top=self.L+margen)
        self.pregraficado = True
        # dibujo los bordes del rve
        fron = []
        fron.append( [[0,self.L], [0,0]] )
        fron.append( [[0,0], [self.L,0]] )
        fron.append( [[0,self.L], [self.L,self.L]] )
        fron.append( [[self.L,self.L], [self.L,0]] )
        plt_fron0 = self.ax.plot(fron[0][0], fron[0][1], linestyle=":")
        plt_fron1 = self.ax.plot(fron[1][0], fron[1][1], linestyle=":")
        plt_fron2 = self.ax.plot(fron[2][0], fron[2][1], linestyle=":")
        plt_fron3 = self.ax.plot(fron[3][0], fron[3][1], linestyle=":")


    def pre_graficar_fibras(self):
        nc = np.max(self.fibs.capas) + 1
        colores = plt.cm.rainbow(np.linspace(0,1,nc))
        # dibujo las fibras (los segmentos)
        # preparo las listas, una lista para cada fibra
        xx_fibs = [ list() for f in  self.fibs.con ]
        yy_fibs = [ list() for f in  self.fibs.con ]
        grafs_fibs = list() # un plot para cada fibra
        for f in range(len(self.fibs.con)):  # f es un indice
            # el primer nodo del primer segmento lo agrego antes del bucle
            s = self.fibs.con[f][0] # obtengo el indice del primer segmento de la fibra numero f
            n = self.segs.con[s][0] # obtengo el indice del primer nodo del segmento numero s
            r = self.nods.r[n] # obtengo las coordenadas del nodo numero n
            xx_fibs[f].append(r[0])
            yy_fibs[f].append(r[1])
            for s in self.fibs.con[f]:
                # s es un indice de un segmento
                # voy agregando los nodos finales
                n = self.segs.con[s][1]
                r = self.nods.r[n]
                xx_fibs[f].append(r[0])
                yy_fibs[f].append(r[1])
            grafs_fibs.append( self.ax.plot(xx_fibs[f], yy_fibs[f], linestyle="-", marker="", label=str(f), color=colores[self.fibs.capas[f]]) )

    def pre_graficar_nodos_frontera(self):
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
            grafs.append( self.ax.plot(xx[f], yy[f], linewidth=0, marker="x", mec="k") )

    def pre_graficar_nodos_interseccion(self):
        # dibujo las fibras (los segmentos)
        # preparo las listas, una lista para cada fibra
        xx = list()
        yy = list()
        grafs = list() # un plot para cada fibra
        for n in range(len(self.nods.r)):
            if self.nods.tipos[n] == 2:
                xx.append(self.nods.r[n][0])
                yy.append(self.nods.r[n][1])
        self.ax.plot(xx, yy, linewidth=0, marker="s", mec="k")

    def graficar(self):
        if not self.pregraficado:
            self.pre_graficar_bordes()
            self.pre_graficar_nodos_frontera()
            self.pre_graficar_nodos_interseccion()
            self.pre_graficar_fibras()
        self.ax.legend(loc="upper left", numpoints=1, prop={"size":6})
        plt.show()