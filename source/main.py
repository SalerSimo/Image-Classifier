import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import image_classifier
import manage_dataFolder
from tkinter.filedialog import askopenfilename
import tkinter as tk
import pyautogui
import time
import sys

workingDirectory = os.getcwd()
imgHeight = 128
imgWidth = 128
def main():
    modelsPath = 'models'
    '''modelsList = os.listdir(modelsPath)
    print('\nAvailable models:\n')
    for i, model in enumerate(modelsList):
        print(f' {i}:', model.replace('_ImageClassifier.keras', ''))
    print('\n')'''
    while True:
        choice = input("Insert 'train' to train a new model \nInsert 'model' to use an already existing model \nInsert 'exit' to exit \n\n")
        if choice == 'train':
            #imgHeight = int(input('Insert image dimension: '))
            #imgWidth = imgHeight
            model, classes = TrainAndSaveNewModel(imgHeight, imgWidth, modelsPath)
            Prediction(model, classes, imgHeight, imgWidth)
            break
        elif choice == 'model':
            print('Select model')
            time.sleep(0.5)
            modelPath = askopenfilename(title="Select model", initialdir='models')
            try:
                model = image_classifier.LoadModel(modelPath)
                classes = os.path.split(modelPath)[1]
                classes = classes.split('_')
                Prediction(model, classes[0:len(classes) - 1], imgHeight, imgWidth)
            except:
                print('Issue with model loading')
            break
        elif choice == 'exit':
            return
        print("Invalid input")
    return


def TrainAndSaveNewModel(imgHeight, imgWidth, modelsPath):
    dataPath = 'images'
    minDimension = 10000
    classes = input("Insert classes names: ").upper()
    classes = classes.replace(',', '')
    classes = classes.split(' ')
    classes.sort()
    print("\n")
    modelName = ''
    manage_dataFolder.removeFolder(dataPath)
    for className in classes:
        modelName = modelName + className + '_'
        manage_dataFolder.downloadImagesFromGoogle(keyword=className, limit=5, dataPath=dataPath)
    modelName += '.keras'
    manage_dataFolder.removeWrongImages(dataPath, minDimenion=minDimension)
    train, validation = image_classifier.LoadData(imgHeight, imgWidth, batchSize=2, directory=dataPath)
    model = image_classifier.TrainModel(train, validation, imgHeight, imgWidth)
    model.save(os.path.join(modelsPath, modelName))
    manage_dataFolder.removeFolder(dataPath)
    return model, classes


def Prediction(model, classes, imgHeight, imgWidth):
    print('\nNow select an image\n')
    time.sleep(0.5)
    imagePath = askopenfilename(title='Select image')
    while not imagePath:
        input("You have to select a image, press enter to try again")
        imagePath = askopenfilename(title='Select image')
    image_classifier.ClassifieImage(imgHeight, imgWidth, imagePath, classes, model)
    main()


if __name__ == '__main__':
    print("\n")
    main()