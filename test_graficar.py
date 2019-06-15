import Rve_generador2
import Malla_equilibrio_2
from matplotlib import pyplot as plt
import numpy as np
import os.path

fdir = os.path.join("mallas", "mc_00007.txt")

mc = Rve_generador2.Malla.leer_de_archivo(fdir)

mc.graficar()