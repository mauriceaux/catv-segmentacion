import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime, os
from lib import utils
import matplotlib.pyplot as plt
import gc
import glob



def mostrarSegmentacion(data_dir):
    tf.keras.backend.clear_session()
    img_height = 480
    img_width = 640

    test_dataset, image_count = utils.loadImages(data_dir, img_height, img_width)

    checkpoint_path = "training/cp.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path)
    else:
        model = utils.crearModelo(img_width, img_height)

    utils.show_new_predictions(test_dataset, model, 10)

def segmentar(input_dir, output_dir):
    gc.collect()
    img_height = 480
    img_width = 640

    if input_dir[-1] != '/': input_dir+='/'
    if output_dir[-1] != '/': output_dir+='/'
    jpgFilenamesList = glob.glob(f'{input_dir}*.jpg')

    checkpoint_path = "training/cp.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path)
    else:
        model = utils.crearModelo(img_width, img_height)


    for img in jpgFilenamesList:
        segmentedFilename = img.split('/')[-1].replace("jpg","png")
        print(f"Segmentando imagen {segmentedFilename}")
        one_img_batch = utils.load_image(utils.parse_new_image(img), img_width, img_height)[tf.newaxis, ...]
        pred_mask = model.predict(one_img_batch)
        segmented = utils.create_mask(pred_mask)
        tf.keras.preprocessing.image.save_img(f"{output_dir}{segmentedFilename}", segmented)
    print(f"Terminado.")


parser = argparse.ArgumentParser()
parser.add_argument('input', 
                    help='Carpeta donde se encuentran las imágenes jpg a segmentar.')
parser.add_argument('output',
                    help='Carpeta donde se almacenan las imágenes segmentadas')
args = parser.parse_args()
if not os.path.exists(args.input):
    raise Exception(f"La carpeta de origen de imagenes {args.input} no existe.")

if not os.path.exists(args.output):
    raise Exception(f"La carpeta de destino de imagenes {args.output} no existe.")

segmentar(args.input, args.output)