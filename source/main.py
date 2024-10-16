import image_classifier
import manage_dataFolder
from tkinter.filedialog import askopenfilename
import tkinter as tk
import os
import pyautogui
import time

workingDirectory = os.getcwd()

def main():
    modelsPath = 'models'
    modelsList = os.listdir(modelsPath)
    print('\nAvailable models:\n')
    for i, model in enumerate(modelsList):
        print(f' {i}:', model.replace('_ImageClassifier.keras', ''))
    print('\n')
    while True:
        choice = input("Insert 'train' to train a new model \nInsert 'model' to use an already existing model \nInsert 'exit' to exit \n")
        if choice == 'train':
            imgHeight = int(input('Insert height: '))
            imgWidth = int(input('Insert width: '))
            model, classes = TrainAndSaveNewModel(imgHeight, imgWidth, modelsPath)
            Prediction(model, classes, imgHeight, imgWidth)
            break
        elif choice == 'model':
            print('Select model')
            time.sleep(0.75)
            modelPath = askopenfilename(title="Select model", initialdir='models')
            try:
                model = image_classifier.LoadModel(modelPath)
                classes = os.path.split(modelPath)[1]
                classes = classes.replace('.keras', '')
                classes = classes.split('_')
                imgHeight = int(classes[len(classes) - 1].split('x')[0])
                imgWidth = int(classes[len(classes) - 1].split('x')[1])
                Prediction(model, classes[:len(classes) - 1], imgHeight, imgWidth)
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
    print(classes)
    modelName = ''
    manage_dataFolder.removeFolder(dataPath)
    for className in classes:
        modelName = modelName + className + '_'
        manage_dataFolder.downloadImagesFromGoogle(keyword=className, limit=50, dataPath=dataPath)
    modelName += str(imgHeight) + 'x' + str(imgWidth) + '.keras'
    print(modelName)
    manage_dataFolder.removeWrongImages(dataPath, minDimenion=minDimension)
    train, validation = image_classifier.LoadData(imgHeight, imgWidth, batchSize=2, directory=dataPath)
    model = image_classifier.TrainModel(train, validation, imgHeight, imgWidth)
    model.save(os.path.join(modelsPath, modelName))
    manage_dataFolder.removeFolder(dataPath)
    return model, classes


def Prediction(model, classes, imgHeight, imgWidth):
    print('\nNow select an image\n')
    time.sleep(0.75)
    imagePath = askopenfilename(title='Select image')
    while not imagePath:
        input("You have to select a image, press enter to try again")
        imagePath = askopenfilename(title='Select image')
    image_classifier.ClassifieImage(imgHeight, imgWidth, imagePath, classes, model)
    main()


if __name__ == '__main__':
    main()