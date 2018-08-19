# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

print("======Results========")
for item in range(0,10):
    print("Item " + str(item))
    print(str(results[item][2][0][0]) + "==>" + str(results[item][2][0][1]))
    print("Confidence = " + str(results[item][2][0][2]*100) +"%")
    print("Lift = " + str(results[item][2][0][3]))