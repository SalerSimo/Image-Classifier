import os

import neural_network
from neural_network import ConvNeuralNetwork
import manage_dataFolder
from tkinter.filedialog import askopenfilename
import time
import pyautogui

def main():
    models_path = os.path.join(working_directory, 'models')
    while True:
        choice = input("Insert 'train' to train a new model \nInsert 'model' to use an already existing model \nInsert 'exit' to exit \n\n")
        if choice == 'train':
            model, class_names = TrainAndSaveNewModel(models_path)
            Prediction(model, class_names)
            break
        elif choice == 'model':
            print('Select model')
            time.sleep(0.5)
            modelPath = askopenfilename(title="Select model", initialdir=models_path)
            try:
                model = neural_network.LoadModel(modelPath)
                class_names = os.path.split(modelPath)[1]
                class_names = class_names.replace('.pth', '')
                class_names = class_names.split('_')
                Prediction(model, class_names)
            except:
                print('Issue with model loading')
            break
        elif choice == 'exit':
            return
        print("Invalid input")
    return


def TrainAndSaveNewModel(models_path: str) -> tuple[ConvNeuralNetwork, list[str]]:
    data_path = os.path.join(working_directory, 'images')
    min_dimension = 10000
    class_names = input("Insert categories names: ").upper()
    class_names = class_names.replace(',', '')
    class_names = class_names.split(' ')
    class_names.sort()
    print("\n")
    model_name = ''
    manage_dataFolder.removeFolder(data_path)
    limit = int(input("Insert the number of images to download for each category: "))
    for className in class_names:
        model_name = model_name + className + '_'
        manage_dataFolder.downloadImagesFromGoogle(keyword=className, limit=limit)
    print("\n")
    model_name = model_name[:-1]
    model_name += '.pth'
    manage_dataFolder.removeWrongImages(data_path, minDimenion=min_dimension)
    train_loader, test_loader = neural_network.LoadData(image_height, image_width, batchSize=2, directory=data_path)
    model = neural_network.TrainModel(train_loader, test_loader)
    neural_network.SaveModel(model, os.path.join(models_path, model_name))
    manage_dataFolder.removeFolder(data_path)
    return model, class_names


def Prediction(model: ConvNeuralNetwork, class_names: list[str]) -> None:
    print('\nNow select an image\n')
    time.sleep(0.5)
    image_path = askopenfilename(title='Select image')
    while not image_path:
        input("You have to select a image, press enter to try again")
        image_path = askopenfilename(title='Select image')

    prediction = neural_network.MakePrediction(model, image_path, class_names, (image_height, image_width))
    print(f'The image {os.path.split(image_path)[1]} is: ', prediction)


if __name__ == '__main__':
    print("\n")
    working_directory = os.path.dirname(__file__)
    working_directory = os.path.split(working_directory)[0]
    image_height = 128
    image_width = 128
    main()