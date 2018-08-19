# Convolutional Neural Network
# Image Classification

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

"""
We will be using Keras so we need to create a specific folder structure

data_set
    :
    :
    :----> training_set
            :
            :---->cats
                    :---->label is not important as long as the os can read the file
            :---->dogs
                :---->label is not important as long as the os can read the file
    :----> test_set
            :---->cats
                :---->label is not important as long as the os can read the file
            :---->dogs
                :---->label is not important as long as the os can read the file

There will NOT be any data pre-processing
"""

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
# Sequential is used to inizialize the ANN
from keras.models import Sequential
# In order to create the convulutian step, since images are 2d we use Convolution2D, if it was videos then we use Convolution3D
from keras.layers import Convolution2D
# This is used to create the pooling step to add the pooling layers
from keras.layers import MaxPooling2D
# Used to create the Flattening step, convert all the pooling feature maps which was created from convultion of our images and then converted to a large
# feature layer with one column
from keras.layers import Flatten
# Add the fully connected layer of the CNN
from keras.layers import Dense

# Initialising the CNN
# Creating the object of the Sequential class to initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
"""
Convert the image to an array of all the pixels and convert to an array of 0 and 1
Then using a feature detector and create feature maps that contain numbers in which the heighest number on th feature map 
shows the location of a specific feature
A layer of all the feature maps will be created with a stride of one
This will create the convultional layer

"""
# .add() will add layer
# we will be adding a convultion2d layer aka.convolution layer
#     nb_filter : number of feature maps we need to create
#     nb_row : number of rows in the feature detector
#     nb_col : number of columns in the feature detector
#     input_shape : the size of the input image
#         since we are using CPU, we select a 64X64 for the 2D array and 3 channels for color images(if black and white channels == 1)
#         the order matters, this order is for tensor flow backend
#     activation : the activation function use (rectifeier == "relu")
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
"""
The size of the feature map will reduce after max pooling, we use a stride of two
the size of the feature map will devide by two
We reduce the size two reduce time complexity
"""
# MaxPooling2D :
#     pool_size : the size of the reduced feature size which is efficent and will retain the features of the image
classifier.add(MaxPooling2D(pool_size = (2, 2)))

"""
Deeper CNN
Adding another Convolutional layer
"""
# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
"""
A huge single vector is created with the multiple feature maps
"""
classifier.add(Flatten())

# Step 4 - Full connection
"""
Create the full connection layer
"""
# Dense function is used to create the hidden layer
#     units : the number of hidden nodes on the hidden layer
#     activation : the activation function
classifier.add(Dense(units = 128, activation = 'relu'))
# Dense function is used to create the output hidden layer
#     units : the number of hidden nodes on the hidden layer
#     activation : the activation function (sigmoid because of probability)
classifier.add(Dense(units = 1, activation = 'sigmoid'))

"""
Compiling means adding the sochastic gradient descent on the whole CNN
"""
# Compiling the CNN
# .compile():
    # optimizer: add an algorithm for the stochastic gradient descent
        # there are many types of sgd and a very effiecent one is "adam"
    # loss: corresponds to the loss function of the sgd
        # if the dependent varibale has a binary outcome then the loss function will be binary_crossentropy
        # if the dependent variable has a categorical outcome then the loss function will be catergorical_crossentropy
    # metrics: critation used to evaluate the model
        # algorithm uses this model to imporve the model
        # we will be using the accuracy model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])




# Part 2 - Fitting the CNN to the images

"""
Fitting the CNN to the training set using th keras documentaion
"""
# Import the ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

# Image Augmentation for training set
train_datagen = ImageDataGenerator(rescale = 1./255,        # all pixel values will be 0 and 1
                                   shear_range = 0.2,       # sheering
                                   zoom_range = 0.2,        # random zoom
                                   horizontal_flip = True)  # flips the image horizontally

# Image Augmentation for test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creation of the training set
training_set = train_datagen.flow_from_directory('dataset/training_set',    # Where we extract the images from
                                                 target_size = (64, 64),    # size of the images expected in the CNN model
                                                 batch_size = 32,           # number of images that will go through after which the weights will be updated
                                                 class_mode = 'binary')     # dependent varibale is binary or has more than two catergories

test_set = test_datagen.flow_from_directory('dataset/test_set',             # Where we extract the images from
                                            target_size = (64, 64),         # size of the images expected in the CNN model
                                            batch_size = 32,                # number of images that will go through after which the weights will be updated
                                            class_mode = 'binary')          # dependent varibale is binary or has more than two catergories
# Fit the CNN to the images which also testing the performance of the model
classifier.fit_generator(training_set,                  # the training set
                         steps_per_epoch = 8000,        # number of images in the network
                         epochs = 25,                   # number of epochs
                         validation_data = test_set,    # test set which the testing is done
                         validation_steps = 2000)       # number of images in the test set