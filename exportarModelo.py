import argparse
from shutil import copyfile
import os

def exportar(carpeta_destino):
    if not os.path.exists(carpeta_destino): raise Exception(f"La carpeta destino {carpeta_destino} no existe.")
    path = os.path.join(carpeta_destino, "modelo.h5")
    copyfile('training/cp.h5', path)
    print(f"Modelo exportado a {path}")

parser = argparse.ArgumentParser()
parser.add_argument('carpeta_destino', 
                    help='Carpeta destino donde se exportará el modelo.')
args = parser.parse_args()

exportar(args.carpeta_destino)