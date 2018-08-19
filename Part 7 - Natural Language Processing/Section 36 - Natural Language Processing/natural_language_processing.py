"""
@author:superdatascience
@updated: 13042018
@updatedby: bhanuka
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# We use the tsv(tab seperated) files since the reviews will contain commas so we cannot seperate with commas
# delimiter = /t since it is a tab seperated
# quoting = code "3" is used to ignore double quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# Libraries
    # re = Regular Expression
    # nltk = Natural Language Toolkit
import re
import nltk

# Tools needed for the nltk library
# Import the stopwords (this,a,are)
nltk.download('stopwords')
# Importing the stopwords into the code from the nltk.corpus
from nltk.corpus import stopwords
# Importing the stemming class from the ntlk.stem.porter (PorterStemmer)
from nltk.stem.porter import PorterStemmer

# Corpus is a collection of text ( eg- Article, HTML, Book)
# This list is used to collect all the texts
corpus = []

# Looping through all the items in the pandas data frame
for i in range(dataset.shape[0]):
    # Removing everything except letters, eg - !@# and 0-9
        # 1 - ^ remove characters except the values after this
        # 2 - Replace the removing character to a space " "
        # 3 - The string
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

    # Convert the string into a lower case string
    review = review.lower()

    ## Change the irrelevant words (stopwords) such as the,a,is,this
    # Split the review into an array
    review = review.split()

    # Stemming is taking the root of the word (eg:- loved = love, sleeping = sleep)
    # Initializing an instance of the PorterStemmer class to stem the words
    ps = PorterStemmer()

    # Traverse through the words in the review array and if the word is not in stopwords set; then add it to the review list
    # Stemming will be done while filtering the stopwords from the reviews the list
    # Set will improve the efficiency of the algorithm; since indexing is fast in set
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    # Join the review array back into a string with a space when joining
    review = ' '.join(review)

    # Cleaned review sentence will be appended to the corpus
    # This will be a cleaned list of all the cleaned review sentences
    corpus.append(review)


# Creating the Bag of Words model [array = (number of reviews , number of the unique words in the bag of word)]
# Bag of words model is to simplify the number of words(by getting the unique words) in the review and create a sparse matrix using tokenization
# sparse matrix is a matrix with a lot of zeros, (**Reduce Spacity as much as possible)
# Tokenization is the process of taking all the words in the review and take one column for each word
# This model is used to predict whether the review is positive or negative
# Machine learning classification model needs to be trained using the reviews
# If column in sparse matrix is one, the review is good when the specific word is present
# If column in sparse matrix is zero, the review is bad when the specific word is present
# import the CountVerctorizer class from the sklearn.feature.extrsction.text
from sklearn.feature_extraction.text import CountVectorizer

# Create an instace of the CountVectorizer class
# max_features = it will keep the most frequent words and keep only the relavant from the matrix (1500 = keep the 1500 relavant words)
# We have done these, but we can also do this using this class:(But its better to use nltk)
# other_parameters = stopwords = nltk.corpus.stopwords, lowercase = .lower() ,token_pattern = re.sub()
cv = CountVectorizer(max_features = 1500)

# Apply the fit_transform so that it will create the huge sparse matrix from the corpus matrix
# Convert to an array at the end
X = cv.fit_transform(corpus).toarray()

# Create the dependant variable matrix
# Take the good and bad (0,1) from the dataset and assign it to y
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
# From the sklearn.cross_validation import the train_test_split to seperate the test and train data
from sklearn.cross_validation import train_test_split
# Splitting the dataset into two tran and test sets
# test_size = 80% to train (800 training) : 20% going to the test set (200 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# we can use the confusion matrix to find the best modal
# Most common models
    # Naives Bayes
    # Decision Tree
    # Random Forest

# Using Naves Bayes Classification
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
# Creating the instance of the GuassianNB() class
classifier = GaussianNB()
# Fitting the Naives Bay to the training data
classifier.fit(X_train, y_train)

# Predicting the Test set results given the test values
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
# We can test the accuracy of the model and get the best modal for natural processing
# Check whether the prediction of y is correct
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Calculate the accuracy and print it
accuracy = (cm[0][0]+cm[1][1])/200
print(str(accuracy*100)+"%")
# Print the confusion matrix
print(cm)