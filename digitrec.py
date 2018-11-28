import keras as kr
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.preprocessing as pre
import gzip
import PIL
from PIL import Image
import os.path



def inputImage(userImg):

    ## Read in an image
    im = Image.open(userImg).convert('L')

    ## Resize the image
    img = im.resize((28, 28), Image.BICUBIC)



## Ask user for the image they want to use.
print("Please enter the path of the image you want the model to use")
userInput = input("")

## Pass this image into the inputImage function.
inputImage(userInput)