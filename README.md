# Segmentación de imágenes
## Guía de instalación

Se recomienda encarecidamente el uso de virtualenv (https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
El proyecto usa python3.

En la raíz del proyecto, instalar las dependencias usando:

$ pip install -r requirements.txt

El proyecto tiene compatibilidad con GPU NVidia. En caso de contar con una, se debe intalar el driver CUDA en la máquina (https://www.tensorflow.org/install/gpu?hl=es-419)

## Guía de uso

El proyecto tiene 4 funciones:

### 1.- Entrenamiento

Para entrenar el segmentador se invoca el archivo entrenar.py usando el interprete de python. El archivo recibe como parámetro la ruta en la que se almacenan los datos de entrenamiento y el número de épocas por las que se quiere entrenar (opcional). El archivo muestra una descripción de los parámetros al invocarlo con la opción -h o --help:

```
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
```  

La carpeta raiz donde se encuentran los datos de entrenamiento debe tener 2 subcarpetas, "images" y "labels".
La carpeta "images" contiene las imágenes a segmentar, en formato jpg y de tamaño 640x480. La carpeta "labels" contiene las imagenes de ejemplo (ya segmentadas), en formato png, en blanco y negro y de tamaño 640x480.
La asociación entre la imagen y el ejemplo se hace por nombre, el algoritmo asociará la imagen de ejemplo a segmentar imagen1.jpg en la carpeta "images" con la segmentación imagen1.png en la carpeta "labels".

Ejemplo:

```
$ python entrenar.py ~/ruta/dataset --epochs 10
```

### 2.- Segmentación

Una vez que el modelo ha sido entrenado, se invoca el archivo segmentar.py. El archivo recibe como parámetro la ruta en la que se encuentran las imágenes a segmentar y la ruta en la que se guardará el resultado de la segmentación. El archivo muestra una descripción de los parámetros al invocarlo con la opción -h o --help:

```
$ python segmentar.py -h
2021-02-06 22:21:13.877339: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
usage: segmentar.py [-h] input output

positional arguments:
  input       Carpeta donde se encuentran las imágenes jpg a segmentar.
  output      Carpeta donde se almacenan las imágenes segmentadas

optional arguments:
  -h, --help  show this help message and exit
```

La carpeta contenedora de las imágenes a segmentar debe poder ser leída desde el programa. Las imágenes a segmentar deben tener formato jpg y un tamaño de 640x480. La carpeta en la que se guardará el resultado de la segmentación debe tener permisos de escritura. Las imágenes segmentadas se almacenan en formato png, en blanco y negro y con un tamaño de 640x480.

Ejemplo:
```
$ python segmentar.py ~/ruta/imagenes/a/segmentar ~/ruta/resultado/segmentacion
```
### 3.- Importar/Exportar modelo

Para facilitar el entrenamiento en una máquina diferente, el programa permite importar y exportar un modelo que luego se usará para la segmentación. El modelo debe estar en formato h5. 
Para importar un modelo se invoca el archivo importarModelo.py. El archivo recibe como parámetro la ruta en la que se encuentra el modelo a importar. El archivo muestra una descripción de los parámetros al invocarlo con la opción -h o --help:
```
$ python importarModelo.py -h
usage: importarModelo.py [-h] archivo_origen

positional arguments:
  archivo_origen  Archivo de modelo en formato h5 a importar.

optional arguments:
  -h, --help      show this help message and exit
```
Ejemplo:
```
$ python importarModelo.py ~/ruta/al/modelo.h5
```
Para exportar el modelo que el segmentador usa, se invoca el archivo exportarModelo.py. El archivo recibe como parámetro la ruta en la que se guardará el modelo. El archivo muestra una descripción de los parámetros al invocarlo con la opción -h o --help:
```
$ python exportarModelo.py -h
usage: exportarModelo.py [-h] carpeta_destino

positional arguments:
  carpeta_destino  Carpeta destino donde se exportará el modelo.

optional arguments:
  -h, --help       show this help message and exit
```
Ejemplo:
```
$ python importarModelo.py ~/ruta/almacenamiento/modelo
```
La carpeta donde se almacenará el modelo debe existir y contar con permisos de escritura.
