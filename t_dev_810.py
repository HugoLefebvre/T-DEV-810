import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import asarray
from PIL import Image

mainDIR = './chest_xray/'

train_folder= mainDIR + '/train/'
val_folder = mainDIR + '/val/'
test_folder = mainDIR + '/test/'

def getImageValue(img):
    return_tab = []
    for line_pixel in img:
        for pixel in line_pixel:
            return_tab.append(pixel)
    return return_tab

def loadDataset(folder):
    x_array = []
    y_array = []
    listDir = os.listdir(folder)
    for dir in listDir:
        if dir[0] != ".":
            for file in os.listdir(folder + dir):
                if(file[0] != "."):
                    img = Image.open(folder + dir + '/' + file).convert('L')

                    resize_img = img.resize((50, 50))
                    img = asarray(resize_img)
                    img_array = getImageValue(img)

                    x_array.append(img_array)
                    y_array.append(dir)

    return {'data': x_array, 'labels': y_array}