import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image, ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    imgHeight = 256
    imgWidth = 256
    batchSize = 2
    directory = 'data'
    train, validation, classNames = LoadData(imgHeight, imgWidth, batchSize, directory)
    TrainModel(train, validation, imgHeight, imgWidth)
    newModel = keras.models.load_model('model/imageclassifier.keras')
    ClassifieImage(imgHeight, imgWidth, 'catTest.jpg', ['cat', 'dog', 'horse'], newModel)
    '''TestData(imgHeight, imgWidth, 'catTest.jpg', classNames, newModel)
    TestData(imgHeight, imgWidth, 'dogTest.jpg', ['cat', 'dog', 'horse'], newModel)
    TestData(imgHeight, imgWidth, 'horseTest.jpg', ['cat', 'dog', 'horse'], newModel)
    TestData(imgHeight, imgWidth, 'deerTest.jpg', ['cat', 'dog', 'horse'], newModel)
    TestData(imgHeight, imgWidth, 'axel.jpg', ['cat', 'dog', 'horse'], newModel)
    TestData(imgHeight, imgWidth, 'dog.jpg', ['cat', 'dog', 'horse'], newModel)'''
    return


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
    # saveModel = int(input("Do you want to save this model?"))
    # model.save('model/imageclassifier.keras')
    # print('Model saved in path model/imageclassifier.keras')


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
    print(f'The image {os.path.split(imagePath)[1]} is', classNames[np.argmax(prediction)])


if __name__ == '__main__':
    main()
