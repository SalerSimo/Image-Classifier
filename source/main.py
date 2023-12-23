import image_classifier
import manage_dataFolder

import os

workingDirectory = os.path.split(os.getcwd())[0]


def main():
    modelsPath = input("Insert models' directory name: ")
    modelsPath = os.path.join(workingDirectory, modelsPath)
    try:
        modelsList = os.listdir(modelsPath)
    except:
        print("Invalid name")
        main()
        return
    if len(modelsList) == 0:
        print('No models available')
    else:
        print('\nAvailable models:\n')
        for i, model in enumerate(modelsList):
            print(f'{i}:', model.replace('_ImageClassifier.keras', ''))
        print('\n')
    modelsSize = len(modelsList)
    while True:
        choice = input("Insert 'train' to train a new model, or insert an already existing model's index")
        if choice == 'train':
            imgHeight = int(input('Insert height'))
            imgWidth = int(input('Insert width'))
            model, classes = TrainAndSaveNewModel(imgHeight, imgWidth, modelsPath)
            Prediction(model, classes, imgHeight, imgWidth)
            break
        else:
            try:
                index = int(choice)
                if index < modelsSize:
                    try:
                        model = image_classifier.LoadModel(os.path.join(modelsPath, modelsList[index]))
                        classes = modelsList[index].replace('_ImageClassifier.keras', '')
                        classes = classes.split('_')
                        imgHeight = int(classes[len(classes) - 1].split('x')[0])
                        imgWidth = int(classes[len(classes) - 1].split('x')[1])
                        Prediction(model, classes[:len(classes) - 1], imgHeight, imgWidth)
                    except:
                        print('Issue with model loading')
                    break
            except:
                pass
        print("Invalid input")
    return


def TrainAndSaveNewModel(imgHeight, imgWidth, modelsPath):
    dataPath = 'images'
    minDimension = 10000
    classes = input("Insert classes names").upper()
    classes = classes.replace(',', '')
    classes = classes.split(' ')
    classes.sort()
    print(classes)
    modelName = ''
    manage_dataFolder.removeFolder(dataPath)
    for className in classes:
        modelName = modelName + className + '_'
        manage_dataFolder.downloadImagesFromGoogle(keyword=className, limit=10, dataPath=dataPath)
    modelName += str(imgHeight) + 'x' + str(imgWidth) + '_ImageClassifier.keras'
    print(modelName)
    manage_dataFolder.removeWrongImages(dataPath, minDimenion=minDimension)
    train, validation = image_classifier.LoadData(imgHeight, imgWidth, batchSize=2, directory=dataPath)
    model = image_classifier.TrainModel(train, validation, imgHeight, imgWidth)
    model.save(os.path.join(modelsPath, modelName))
    manage_dataFolder.removeFolder(dataPath)
    return model, classes


def Prediction(model, classes, imgHeight, imgWidth):
    imageDirectory = 'test images'
    imageName = input('Insert image name')
    imageName = imageName.strip(' ')
    imagePath = os.path.join(workingDirectory, imageDirectory, imageName)
    image_classifier.ClassifieImage(imgHeight, imgWidth, imagePath, classes, model)


if __name__ == '__main__':
    main()
