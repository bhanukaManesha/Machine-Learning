# Artificial Neural Network
# Using stochastic gradient descent ANN
# Solving a classification business problem

"""
Neural Network with one neuron is called a perceptron neural network
"""

# What is Theano?
# Open source numerical computation libaray based on numpy syntax
# Can run on both CPU and GPU
# GPU is more powerful and can run more floating point calculations
# In the neural network, calling the ctivation functions involve Parallel Computation, which the GPU is better at than CPU
# GPU better for Neural Network
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# What is Tensorflow?
# Same as Theano, it can run on GPU and CPU
# It is also a fast computational libarary
# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# What is Keras?
# Is used to build neural networks with a few lines of code
# This wraps around the Tensorflow and Theano libraries which are mainly used for research purpose
# Build Deep learning models
# Installing Keras
# pip install --upgrade keras





# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# Exited column is the dependant variable
# All other columns are independent variables
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# since there are labels in the independent variables
    # country
    # gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Replacng the catergorical data with numerical values
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Create dummy varibles, for the countries
# Creating new columns for all the variables in country
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Then removing the first created column to eliminate the dummy variable trap
# Therefore leaving 00 as one country, 01 as another and 10 as the third
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# To avoid one variable dominating another one we appy feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# First fit the x_train to the StandradScaler object
X_train = sc.fit_transform(X_train)
# Then fit the test set to the StandardScaler object which will scale it accordingly
X_test = sc.transform(X_test)




# Part 2 - Now let's make the ANN!


# Importing the Keras libraries and packages
import keras
# Sequential is used to inizialize the ANN
from keras.models import Sequential
# Dense is used to Create the layers of the ANN
from keras.layers import Dense

# Initialising the ANN
# No arguments since we are defining the layers afterwards
classifier = Sequential()

"""
When choosing the function the rectifier function is know to be more accurate and relaible (for hidden layers),
But we use the sigmoid function to the output layer so that we can get a probability of the output
"""

# Adding the input layer and the first hidden layer

# Classifer:
    # .add() : method is used to add the layers
# Dense function :
    # units : Number of nodes we need to add for the hidden layer
"""
    Choose this using Parameter Tuning such as K-fold validation model
    Tip - Choose the number of node using the average of input and output nodes (eg 11+1 //2)
"""
    # kernel_initializer : Randomly initizie the weghts (uniform)
        # This will take care of step 1; which will randomly initize the weights to a value close to zero
    # activation : Select the function which needed to be used
        # We will be using rectifier function for this node (relu)
    # input_dim :
        # Since the ANN is still initilized, we need to create the input layer and create the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer= 'uniform', activation = 'relu', input_dim = 11))


# Adding the second hidden layer
# We will remove the input_dim since we already defined the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
# We will remove the input_dim since we already defined the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
# We will remove the input_dim since we already defined the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# We will define the activation function as a sigmoid function so that we can get the probability
    # Sigmoid function is the heart of the probabilistic approach
# Output_dim will be 1
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
"""
If we have three categories for the dependent variables, we need to use onehotencoder and encode it to three variables,
then we have to change the activiation function to "softmax",
Softmax is the sigmoid function to multiple variables
"""

"""
Compiling means adding the sochastic gradient descent on the whole ANN
"""
# Compiling the ANN
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

# Fitting the ANN to the Training set
    # data_set : (X_train,y_train)
    # batch_size : number of items after which we need to update the weights
    # epochs : number of epochs
        # Epoch is a round when the whole training set passes through the ANN
classifier.fit(X_train, y_train, batch_size = 100, epochs = 1000)




# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
# Probability will be received using the sigmoid model so it will give a value between 0 and 1
y_pred = classifier.predict(X_test)
# Convert the probability to False or True
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# printing the confusion matrix
print(cm)