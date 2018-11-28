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


## The neural 
def nueralNet(userImg):

    model = kr.models.Sequential()

    ## Add Dense, Activation and Dropout to the model
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    ## Used adam optimizer as seemed to get best results.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ## Read in images for training the model using gzip.
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()

    ## Reshape the images and labels.
    train_img =  np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8)/ 255.0
    train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

    ## Binarize labels, Reshape the images
    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)
    inputs = train_img.reshape(60000, 784)

    ## Train the model with our images and Labels. 3 epochs ran.
    print("Building neural network - May take a few mins!")
    model.fit(inputs, outputs, epochs=3, batch_size=100)

    ## Print the neural networks prediction for the entered image.
    print("According to my network your number is: ")
    print(encoder.inverse_transform(model.predict(userImg)))

def inputImage(userImg):

    ## Read in an image
    im = Image.open(userImg).convert('L')

    ## Resize the image
    img = im.resize((28, 28), Image.BICUBIC)

    ## Pass the image into the imagePrepare function
    img = imageprepare(im)

    ## Run the neural network
    nueralNet(img)


def imageprepare(im):

    ## Get the data from the image
    imd = list(im.getdata())

    ## Normalize the pixels to 0 and 1.
    imd = [(255 - x) * 1.0 / 255.0 for x in imd]
    
    ## Reshape the image
    imd = np.array(list(imd)).reshape(1, 784)

    ## Return the image data
    return imd


## Ask user for the image they want to use.
print("Please enter the path of the image you want the model to use")
userInput = input("")

## Pass this image into the inputImage function.
inputImage(userInput)