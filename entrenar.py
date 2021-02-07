import datetime, os
from tensorflow.keras.models import load_model
import tensorflow as tf
from lib import utils
import argparse

def entrenar(data_dir, EPOCHS = 1):
    if EPOCHS is None: EPOCHS = 1
    img_height = 480
    img_width = 640

    if data_dir[-1] != '/': data_dir+='/'
    train_dataset, test_dataset, image_count = utils.loadTrainingDataset(data_dir, img_height, img_width)

    

    checkpoint_path = "training/cp.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path)
    else:
        model = utils.crearModelo(img_width, img_height)

    BATCH_SIZE = 10
    
    VAL_SUBSPLITS = 5
    TRAIN_LENGTH = (image_count*4)//5
    TEST_LENGTH = image_count-TRAIN_LENGTH
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    VALIDATION_STEPS = TEST_LENGTH//BATCH_SIZE//VAL_SUBSPLITS

    tf.keras.backend.clear_session()
    test_dataset = test_dataset.batch(BATCH_SIZE)

    loss, acc = model.evaluate(test_dataset, verbose=2, batch_size=BATCH_SIZE)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


    callbacks = [
        tensorboard_callback,
        tf.keras.callbacks.EarlyStopping(patience=20, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False)
    ]

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset,
                            callbacks=callbacks)
    
    
    tf.keras.backend.clear_session()


parser = argparse.ArgumentParser()
parser.add_argument('carpeta_raiz', 
                    help='Carpeta raiz donde se encuentran los datos de entrenamiento. Debe contener una carpeta llamada "images" con las imagenes jpg a segmentar y una carpeta "labels" con las imagenes png de ejemplos de segmentación')
parser.add_argument('--epochs', type=int,
                    help='Cantidad de epocas a entrenar')
args = parser.parse_args()

if not os.path.exists(args.carpeta_raiz):
    raise Exception(f"La carpeta raiz {args.carpeta_raiz} no existe.")

entrenar(args.carpeta_raiz, args.epochs)