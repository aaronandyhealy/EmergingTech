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


## Ask user for the image they want to use.
print("Please enter the path of the image you want the model to use")
userInput = input("")