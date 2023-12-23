import os
import shutil
from pygoogle_image import image


def downloadImagesFromGoogle(keyword, limit, dataPath):
    print(f'downloading {keyword} images...')
    image.download(keyword, limit)
    print('download complete')
    downloadPath = 'images'  # image.download automatically create a new folder named "images"
    # it is possible to directly download into a specific folder, but it really increases the duration
    '''try:
        renameFile(downloadPath, dataPath)
    except:
        # means that the folder "dataPath" already exists, inserting subfolder 'downloadPath/keyword" into "dataPath"
        renameFile(os.path.join(downloadPath, keyword), os.path.join(dataPath, keyword))
        removeFolder(downloadPath)'''


def renameFile(oldPath, newPath):
    os.rename(oldPath, newPath)


def removeWrongImages(dataPath, minDimenion):
    imageExtensions = ['jpeg', 'jpg', 'png', 'bmp']
    for imageClass in os.listdir(dataPath):
        for image in os.listdir(os.path.join(dataPath, imageClass)):
            imagePath = os.path.join(dataPath, imageClass, image)
            try:
                ext = image.split('.')  # image is in format imageName.ext
                ext = ext[len(ext) - 1]
                if ext not in imageExtensions:
                    print(f'Image not in extension list {image}')
                    os.remove(imagePath)
                elif os.stat(imagePath).st_size < minDimenion:
                    print(f"Image too small {image}")
                    os.remove(imagePath)
            except:
                print(f"Issue with image {image}")


def removeFolder(folderPath):
    try:
        shutil.rmtree(folderPath)
        print(f'Folder {folderPath} successfully removed')
    except:
        print(f'Folder {folderPath} does not exists')

