# Activity 2 - DATA MINING (Wine Dataset)
# Author: Matheus Aparecido do Carmo Alves (9791114)
# Importing the packages
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels

import pandas as pd

# 1. Getting the Iris data
wine = load_wine()

# 2. Splitting the data set into random train and test subsets
train_set, test_set, train_class, test_class = \
	train_test_split(wine['data'],wine['target'],train_size=0.75,test_size=0.25)
print('Train set size:',len(train_set))
print('Test set size:',len(test_set),'\n')

# 3. Classification steps
# a. building the KNN space
knn_classifier = KNeighborsClassifier(n_neighbors=5,weights='distance')
knn_classifier.fit(train_set, train_class)

# b. classifing the test set (predicting the test set classes)
predict_class = knn_classifier.predict(test_set)

# c. calculating the confusion matrix
result = confusion_matrix(test_class,predict_class)
print('Unnormalized Confusion Matrix\n',result,'\n')

# d. normalizing the confusion matrix
result = result.astype('float') / result.sum(axis=1)[:, np.newaxis]
print('Normalized Confusion Matrix\n',result) 

# e. getting the confusion matrix labels
classes = wine.target_names[unique_labels(train_class, predict_class)]

# 4. Plotting the results
# a. configuring the plot
fig, ax = plt.subplots()
im = ax.imshow(result, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(result.shape[1]),yticks=np.arange(result.shape[0]),\
    xticklabels=wine.target_names, yticklabels=wine.target_names,\
    title='Confusion Matrix', ylabel='True class', xlabel='Predicted class')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
fmt = '.2f'
thresh = result.max() / 2.
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        ax.text(j, i, format(result[i, j], fmt),\
            ha="center", va="center",\
            color="white" if result[i, j] > thresh else "black")
fig.tight_layout()

# b. plotting
plt.show()

# 4. That's all folks... :)