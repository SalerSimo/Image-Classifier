import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image, ImageOps
import numpy as np



def LoadData(imgHeight, imgWidth, batchSize, directory):
    colorMode = 'grayscale'
    train = keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        labels='inferred',
        label_mode='int',
        color_mode=colorMode,
        batch_size=batchSize,
        image_size=(imgHeight, imgWidth),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset='training'
    )
    validation = keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        labels='inferred',
        label_mode='int',
        color_mode=colorMode,
        batch_size=batchSize,
        image_size=(imgHeight, imgWidth),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset='validation'
    )

    def augment(x, y):
        image = tf.image.random_brightness(x, max_delta=0.05)
        return image, y

    train = train.map(augment)
    return train, validation


def TrainModel(train, validation, imgHeight, imgWidth):
    model = keras.Sequential([
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(imgHeight, imgWidth)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128),
        layers.Dense(units=3)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
        metrics=['accuracy']
    )

    model.fit(train, validation_data=validation, epochs=10, verbose=1)
    return model


def LoadModel(modelPath):
    return keras.models.load_model(modelPath)


def ClassifieImage(imgHeight, imgWidth, imagePath, classNames, newModel):
    try:
        image = Image.open(imagePath)
    except:
        print("Issue with image opening")
        return
    image = image.resize((imgHeight, imgWidth))
    image = ImageOps.grayscale(image)
    image = np.asarray(image)
    prediction = newModel.predict(np.array([image]))
    print(f'\nThe image {os.path.split(imagePath)[1]} is', classNames[np.argmax(prediction)], '\n')
