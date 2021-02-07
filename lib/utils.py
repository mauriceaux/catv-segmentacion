import tensorflow as tf
import pathlib
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def create_mask_batch(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask

def display_sample(display_list):
    
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def show_predictions(dataset=None, model=None, num=1):
    
    if dataset:
        for image, true_mask in dataset.take(num):
            one_img_batch = image[tf.newaxis, ...]
            pred_mask = model.predict(one_img_batch)
            display_sample([image, true_mask, create_mask(pred_mask)])
    else:
        one_img_batch = sample_image[tf.newaxis, ...]
        inference = model.predict(one_img_batch)
        pred_mask = create_mask(inference)
        display_sample([sample_image, pred_mask])

def show_new_predictions(dataset=None, model=None, num=1):
    if dataset:
        for image in dataset.take(num):
            one_img_batch = image[tf.newaxis, ...]
            pred_mask = model.predict(one_img_batch)
            display_sample([image, create_mask(pred_mask)])

def unet_model(output_channels, imgSize, down_stack, up_stack):
    img_width = imgSize['width']
    img_height = imgSize['height']
    inputs = tf.keras.layers.Input(shape=[img_height, img_width, 3])
    x = inputs
    
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 2, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    mask_path = tf.strings.regex_replace(img_path, "images", "labels")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    return {'image': image, 'segmentation_mask': mask}

def parse_new_image(img_path: str):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return image


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict, img_width, img_height) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (img_height, img_width))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (img_height, img_width))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict, img_width, img_height) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (img_height, img_width))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (img_height, img_width))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image(data, img_width, img_height):
    input_image = tf.image.resize(data, (img_height, img_width))
    input_image = tf.cast(input_image, tf.float32) / 255.0

    return input_image

def loadTrainingDataset(data_dir, img_height, img_width, train_percentage = 0.75):
    data_dir = pathlib.Path(f"{data_dir}images")
    image_count = len(list(data_dir.glob("*.jpg")))
    SEED = 234

    all_dataset = tf.data.Dataset.list_files(f"{data_dir}/*.jpg", seed=SEED)
    all_dataset = all_dataset.map(parse_image)

    trainSize = int(image_count*train_percentage)
    testSize = image_count-trainSize

    train_files = all_dataset.take(trainSize)
    test_files = all_dataset.skip(trainSize) 

    BATCH_SIZE = 4

    BUFFER_SIZE = 50

    dataset = {"train": train_files, "test": test_files}

    train = dataset['train'].map(lambda x:load_image_train(x, img_width, img_height), num_parallel_calls=tf.data.AUTOTUNE)

    test = dataset['test'].map(lambda x:load_image_test(x, img_width, img_height))

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
   
    return train_dataset, test, image_count

def loadImages(dataDir, img_height, img_width):
    data_dir = pathlib.Path(f"{dataDir}")
    image_count = len(list(data_dir.glob("*.jpg")))

    fileList = tf.data.Dataset.list_files(f"{data_dir}/*.jpg", shuffle=False)
    dataset = fileList.map(parse_new_image)
    dataset = dataset.map(lambda x:load_image(x, img_width, img_height))
    return dataset, image_count, fileList

def crearModelo(img_width, img_height):
    OUTPUT_CHANNELS = 2

    base_model = tf.keras.applications.MobileNetV2(input_shape=[img_height, img_width, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 2),  # 4x4 -> 8x8
        pix2pix.upsample(256, 2),  # 8x8 -> 16x16
        pix2pix.upsample(128, 2),  # 16x16 -> 32x32
        pix2pix.upsample(64, 2),   # 32x32 -> 64x64
    ]

    imgSize = {"width": img_width, "height": img_height}

    model = unet_model(OUTPUT_CHANNELS, imgSize, down_stack, up_stack)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model