import argparse
from shutil import copyfile
import os

def exportar(archivo_origen):
    if not os.path.exists(archivo_origen): raise Exception(f"El archivo a importar {archivo_origen} no existe.")
    copyfile(archivo_origen, 'training/cp.h5')
    print(f"Modelo {archivo_origen} importado exitosamente.")

parser = argparse.ArgumentParser()
parser.add_argument('archivo_origen', 
                    help='Archivo de modelo en formato h5 a importar.')
args = parser.parse_args()

exportar(args.archivo_origen)