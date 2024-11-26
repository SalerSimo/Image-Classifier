import os
import shutil
from pygoogle_image import image

def downloadImagesFromGoogle(keyword: str, limit: int) -> None:
    print(f'downloading {keyword} images...')
    image.download(keyword, limit)
    print('download complete')

def removeWrongImages(dataPath: str, minDimenion: int) -> None:
    imageExtensions = ['jpeg', 'jpg', 'png', 'bmp']
    for imageClass in os.listdir(dataPath):
        for image in os.listdir(os.path.join(dataPath, imageClass)):
            imagePath = os.path.join(dataPath, imageClass, image)
            try:
                ext = image.split('.')  #image is in format imageName.ext
                ext = ext[len(ext) - 1]
                if ext not in imageExtensions:
                    os.remove(imagePath)
                elif os.stat(imagePath).st_size < minDimenion:
                    os.remove(imagePath)
            except:
                print(f"Issue with image {image}")


def removeFolder(folderPath: str) -> None:
    try:
        shutil.rmtree(folderPath)
    except:
        return

