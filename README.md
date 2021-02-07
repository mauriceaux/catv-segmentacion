# catv-segmentacion
Guía de instalación

Se recomienda encarecidamente el uso de virtualenv (https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
El proyecto usa python3.

En la raíz del proyecto, instalar las dependencias usando:

$ pip install -r requirements.txt

El proyecto tiene compatibilidad con GPU NVidia. En caso de contar con una, se debe intalar el driver CUDA en la máquina (https://www.tensorflow.org/install/gpu?hl=es-419)

Guía de uso

El proyecto tiene 4 funciones:

1.- Entrenamiento

Para entrenar el segmentador se invoca el archivo segmentar.py usando el interprete de python. El archivo recibe como parámetro la ruta en la que se almacenan los datos de entrenamiento y el número de épocas por las que se quiere entrenar (opcional). El archivo muestra uan descripción de los parámetros al invocarlo con la opción -h o --help:

$ python entrenar.py -h
2021-02-06 22:08:34.580633: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
usage: entrenar.py [-h] [--epochs EPOCHS] carpeta_raiz

positional arguments:
  carpeta_raiz     Carpeta raiz donde se encuentran los datos de
                   entrenamiento. Debe contener una carpeta llamada "images"
                   con las imagenes jpg a segmentar y una carpeta "labels" con
                   las imagenes png de ejemplos de segmentación

optional arguments:
  -h, --help       show this help message and exit
  --epochs EPOCHS  Cantidad de epocas a entrenar
  
La carpeta raiz donde se encuentran los datos de entrenamiento debe tener 2 subcarpetas, "images" y "labels". La estructura es la siguiente:
  
  raiz
    |
    +-images
    +-labels

La carpeta "images" contiene las imágenes a segmentar, en formato jpg y de tamaño 640x480. La carpeta "labels" contiene las imagenes de ejemplo (ya segmentadas), en formato png, en blanco y negro y de tamaño 640x480.
La asociación entre la imagen y el ejemplo se hace por nombre, el algoritmo asociará la imagen de ejemplo a segmentar imagen1.jpg en la carpeta "images" con la segmentación imagen1.png en la carpeta "labels".

2.- Segmentación

Una vez que el modelo ha sido entrenado, 
