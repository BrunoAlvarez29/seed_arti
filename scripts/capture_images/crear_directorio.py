
'''
Escribir un código que se encarge de crear una carpeta tipo "/[FECHA]" (eg. 24_03_08 - 2024 Marzo 08). 
Esta tiene que tener dos carpetas más, horizontal, vertical, params y un archivo altura.csv.

/[FECHA]
- /horizontal/rgb
- /vertical/rgb
- /params
- alturas.csv

Antes de crear el conjunto de carpetas se debe verificar si existe una carpeta con el mismo nombre 
en el directorio actual. Usar la libreria OS
'''
import os
from datetime import datetime

path = "/home/bruno29/catkin_ws/seed-arti/gallery"

if not os.path.isdir(path):
    print("La ruta ingresada no es un directorio válido.")
    exit()

fecha_actual = datetime.now().strftime("%y_%m_%d")

carpeta_principal = os.path.join(path, fecha_actual)

if os.path.exists(carpeta_principal):
    indice = 2
    while os.path.exists(carpeta_principal + f"_{indice}"):
        indice += 1
    carpeta_principal += f"_{indice}"

os.makedirs(carpeta_principal, exist_ok=True)

for subcarpeta in ["horizontal", "vertical"]:
    carpeta_subprincipal = os.path.join(carpeta_principal, subcarpeta)
    os.makedirs(carpeta_subprincipal, exist_ok=True)
    
    for subcarpeta_interna in ["rgb", "mask", "depth"]:
        os.makedirs(os.path.join(carpeta_subprincipal, subcarpeta_interna), exist_ok=True)

os.makedirs(os.path.join(carpeta_principal, "params"), exist_ok=True)

with open(os.path.join(carpeta_principal, "altura.csv"), "w") as f:
    pass

print("Estructura de carpetas creada:", carpeta_principal)


