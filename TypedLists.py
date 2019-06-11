import collections

class Lista_de_listas_de_dos_floats(collections.MutableSequence):
    def __init__(self):
        self.lista = list()

    def check(self, v):
        # cada item debe ser una lista de dos floats (coordenadas)
        if not isinstance(v, list):
            raise TypeError, v
        else:
            if len(v)!=2:
                raise ValueError, len(v)
            if not isinstance(v[0],float) or not isinstance(v[1],float):
                raise TypeError, v

    def __len__(self): return len(self.lista)

    def __getitem__(self, i): return self.lista[i]

    def __delitem__(self, i): del self.lista[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.lista[i] = v

    def insert(self, i, v):
        self.check(v)
        self.lista.insert(i, v)

    def __str__(self):
        return str(self.lista)


class Lista_de_algunos_enteros(collections.MutableSequence):
    def __init__(self, algunos_enteros):
        self.lista = list()
        self.algunos_enteros = algunos_enteros

    def check(self, v):
        # cada item debe ser una lista de integers 0, 1 o 2 (continuacion, frontera o interseccion)
        if not v in self.algunos_enteros:
            raise ValueError, v

    def __len__(self): return len(self.lista)

    def __getitem__(self, i): return self.lista[i]

    def __delitem__(self, i): del self.lista[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.lista[i] = v

    def insert(self, i, v):
        self.check(v)
        self.lista.insert(i, v)

    def __str__(self):
        return str(self.lista)



class Lista_de_listas_de_dos_enteros(collections.MutableSequence):
    def __init__(self):
        self.lista = list()

    def check(self, v):
        # cada item debe ser una lista de dos nodos (indices), integers
        if not isinstance(v, list):
            raise TypeError, v
        else:
            if len(v)!=2:
                raise ValueError, len(v)
            if not isinstance(v[0],int) or not isinstance(v[1],int):
                raise TypeError, v

    def __len__(self): return len(self.lista)

    def __getitem__(self, i): return self.lista[i]

    def __delitem__(self, i): del self.lista[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.lista[i] = v

    def insert(self, i, v):
        self.check(v)
        self.lista.insert(i, v)

    def __str__(self):
        return str(self.lista)


class Lista_de_enteros(collections.MutableSequence):
    def __init__(self):
        self.lista = list()

    def check(self, v):
        # cada item debe ser una lista de integers 0, 1 o 2 (continuacion, frontera o interseccion)
        if not isinstance(v,int):
            raise ValueError, v

    def __len__(self): return len(self.lista)

    def __getitem__(self, i): return self.lista[i]

    def __delitem__(self, i): del self.lista[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.lista[i] = v

    def insert(self, i, v):
        self.check(v)
        self.lista.insert(i, v)

    def __str__(self):
        return str(self.lista)


class Lista_de_floats(collections.MutableSequence):
    def __init__(self):
        self.lista = list()

    def check(self, v):
        # cada item debe ser una lista de integers 0, 1 o 2 (continuacion, frontera o interseccion)
        if not isinstance(v,float):
            raise ValueError, v

    def __len__(self): return len(self.lista)

    def __getitem__(self, i): return self.lista[i]

    def __delitem__(self, i): del self.lista[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.lista[i] = v

    def insert(self, i, v):
        self.check(v)
        self.lista.insert(i, v)

    def __str__(self):
        return str(self.lista)


class Lista_de_listas_de_enteros(collections.MutableSequence):
    """ es algo como una lista con algunas funciones particulares
    tiene un atributo lista que es una lista
    pero la propia instancia se comporta como la lista """
    def __init__(self):
        self.lista = list() # conectividad: va a ser una lista de listas de segmentos (sus indices nada mas), cada segmento debe ser una lista de 2 nodos

    def check(self, v):
        # cada item debe ser una lista de segmentos (indices), integers
        if not isinstance(v, list):
            raise TypeError, v
        else:
            for item in v:
                if not isinstance(item,int):
                    raise TypeError, item

    def __len__(self): return len(self.lista)

    def __getitem__(self, i): return self.lista[i]

    def __delitem__(self, i): del self.lista[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.lista[i] = v

    def insert(self, i, v):
        self.check(v)
        self.lista.insert(i, v)

    def __str__(self):
        return str(self.lista)
