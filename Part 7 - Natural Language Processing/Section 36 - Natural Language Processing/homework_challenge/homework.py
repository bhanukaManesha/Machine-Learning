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


model_names = ["LogisticReg","KNN","SVM","KernelSVM","NaiveBayes","DecisionTree","RandomForest"]
accuracy = []
precision = []
recall = []
f1_score = []

from EvaluateConfusionMatrix import EvaluateConfusionMatrix

from models.logistic_regression import logistic_regression
logistic_regression_cm = EvaluateConfusionMatrix(logistic_regression(X,y))
accuracy.append(logistic_regression_cm.get_accuracy())
precision.append(logistic_regression_cm.get_precision())
recall.append(logistic_regression_cm.get_recall())
f1_score.append(logistic_regression_cm.get_f1_score())

from models.kn_neighbour import knn
knn_cm = EvaluateConfusionMatrix(knn(X,y))
accuracy.append(knn_cm.get_accuracy())
precision.append(knn_cm.get_precision())
recall.append(knn_cm.get_recall())
f1_score.append(knn_cm.get_f1_score())

from models.svm import svm
svm_cm = EvaluateConfusionMatrix(svm(X,y))
accuracy.append(svm_cm.get_accuracy())
precision.append(svm_cm.get_precision())
recall.append(svm_cm.get_recall())
f1_score.append(svm_cm.get_f1_score())

from models.kernel_svm import kernel_svm
kernel_svm_cm = EvaluateConfusionMatrix(kernel_svm(X,y))
print(kernel_svm(X,y))
accuracy.append(kernel_svm_cm.get_accuracy())
precision.append(kernel_svm_cm.get_precision())
recall.append(kernel_svm_cm.get_recall())
f1_score.append(kernel_svm_cm.get_f1_score())

from models.naive_bayes import naive_bayes
naive_bayes_cm = EvaluateConfusionMatrix(naive_bayes(X,y))
accuracy.append(naive_bayes_cm.get_accuracy())
precision.append(naive_bayes_cm.get_precision())
recall.append(naive_bayes_cm.get_recall())
f1_score.append(naive_bayes_cm.get_f1_score())

from models.decision_tree import decision_tree
decision_tree_cm = EvaluateConfusionMatrix(decision_tree(X,y))
accuracy.append(decision_tree_cm.get_accuracy())
precision.append(decision_tree_cm.get_precision())
recall.append(decision_tree_cm.get_recall())
f1_score.append(decision_tree_cm.get_f1_score())

from models.random_forest import random_forest
random_forest_cm = EvaluateConfusionMatrix(random_forest(X,y))
accuracy.append(random_forest_cm.get_accuracy())
precision.append(random_forest_cm.get_precision())
recall.append(random_forest_cm.get_recall())
f1_score.append(random_forest_cm.get_f1_score())


# print(logistic_regression_cm)
# print(knn_cm)
# print(svm_cm)
# print(kernel_svm_cm)
# print(naive_bayes_cm)
# print(decision_tree_cm)
# print(random_forest_cm)



print(accuracy)
print(precision)
print(recall)
print(f1_score)



n_groups = 7

# means_men = (20, 35, 30, 35, 27)
# std_men = (2, 3, 4, 1, 2)
#
# means_women = (25, 32, 34, 20, 25)
# std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, accuracy, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label="Accuracy")

rects2 = ax.bar(index+bar_width, precision, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label="Precision")

rects3 = ax.bar(index+2*bar_width, recall, bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label="Recall")

rects4 = ax.bar(index+3*bar_width, f1_score, bar_width,
                alpha=opacity, color='y',
                error_kw=error_config,
                label="F1 Score")

ax.set_xlabel('Models')
ax.set_title('Evaluating performance pf Models')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels((model_names[0],model_names[1],model_names[2],model_names[3],model_names[4],model_names[5],model_names[6],))
ax.legend()

fig.tight_layout()
plt.show()





