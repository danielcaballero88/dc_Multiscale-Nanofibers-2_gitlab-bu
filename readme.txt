1. crear objeto Nodos con las coordenadas (lista de listas de shape (numnodos,2)) y los tipos (lista de len numnodos). Tipo=1 es frontera, =2 es interseccion.

n = Nodos(coors, tipos) 

internamente el objeto pasa las coordenadas a numpy y calcula las masks (mask_fr y mask_in son arrays booleanos que indican si un nodo es fr o es in) 

2. crear objeto subfibras que es una conectividad con algunos atributos y metodos particulares mas. Se deben dar las coordenadas iniciales (array de numpy) para calcular las longitudes iniciales. Tambien los parametros constitutivos para que ya queden almacenados.

s = Subfibras(conec_in, coors0, paramcon)


3. crear malla a partir de los nodos y subfibras, ademas indico la pseudoviscosidad (float) para resolver iterativamente el equilibrio (quizas deberia ser una propiedad de los nodos pero no estoy seguro)

m = Malla(n,s,psv)

4. crear iterador para resolver iterativamente el equilibrio de la malla segun el movimiento de la frontera
(para lo cual es importante mover la frontera!, para eso hay un metodo mover_nodos_frontera())
Aca hay que dar varios parametros para las iteraciones
n= tamano de la solucion (en este caso numero de nodos, notar que dejo los nodos frontera aunque no seria necesario)
x = solucion  (al principio es la semilla, despues se actualiza)
ecuacion = ecuacion a resolver (en este caso el objeto malla que tiene los parentesis sobrecargados para resolver una iteracion)
ref_small = referencia de pequenos desplazamientos
ref_big = referencia de grandes desplazamientos 
ref_div = referencia de desplazamientos divergentes (es una relacion entre el desplazamiento previo y el actual)
maxiter = numero maximo de iteraciones para alcanzar el equilibrio
tol = tolerancia necesaria para obtener el equilibrio


i = Iterador(


codecombat.com
