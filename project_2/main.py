import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import preprocessing as pp
from agent import Agent

import time
# 1. Preprocessing
# a. Extracting the enrolement and
# the performance information for
# each learner in the dataset
enrolement_info = pp.enrolments()
questions_info, question_set  = pp.questions()

# b. Splitting the quit learners and
# the no quit learners in the dataset
noquit_learners,quit_learners =\
	pp.split_learners(enrolement_info,\
		questions_info)

# c. Cleaning the extracted data and 
# finishing the preprocessing step
preprocessed_dataset = pp.transform_nn_input(noquit_learners,quit_learners,question_set)
#print(preprocessed_dataset)
print('--------------------------------------------------------------------------------')



# 2. Split dataset in train and evaluate
dataset = []
labels = []
for learner in preprocessed_dataset:
	dataset.append(preprocessed_dataset[learner][0])
	labels.append(preprocessed_dataset[learner][1])

dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels)

train = []
length = len(dataset_train)
for i in range (0, length):
	data_label_train = [dataset_train[i], labels_train[i]]
	train.append(data_label_train)

test = []
length = len(dataset_test)
for i in range (0, length):
	data_label_test = [dataset_test[i], labels_test[i]]
	test.append(data_label_test)

# 3. Train the network
# a. Make an instance of the agent(structure that have a neural network and methods to its trainment)
agent = Agent()

# b. Train the network
print('train')
for transaction in train:	
	predict, loss = agent.optimize(transaction[1], transaction[0])

# c. Save network
torch.save(agent.network.state_dict(), 'checkpoint.pth')

# 4. evaluate
print('evaluate')
hit = 0
miss = 0
real = []
predito = []
for transaction in test:
	length = len(transaction[0])
	for i in range(0, length):
		values = torch.tensor(transaction[0][i]).float()
		pred = agent.network(values).argmax()
		
		predito.append(pred.float().item())
		real.append(transaction[1][i])
		
		if transaction[1][i] == pred.float().item():
			hit += 1
		else:
			miss += 1

print('hits: ', hit)
print('hit rate: ', hit/(hit+miss))
print('miss: ', miss)
print('miss rate: ', miss/(hit+miss))

tn, fp, fn, tp = confusion_matrix(real, predito).ravel()
print('true negative: ' + str(tn) + ' | false positive: ' + str(fp) + ' | false nagative: ' + str(fn) + ' | true positive: ' + str(tp))


confusion = confusion_matrix(real, predito)/(hit+miss)
df_cm = pd.DataFrame(confusion, range(2),
                  range(2))
sn.set(font_scale=1)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
plt.show()

""""print('trainning')
for learner in preprocessed_dataset:
	predict, loss = agent.optimize(preprocessed_dataset[learner][1], preprocessed_dataset[learner][0])

print('evaluate')
for learner in preprocessed_dataset:
	length = len(preprocessed_dataset[learner][0])
	for i in range(0, length):
		values = torch.tensor(preprocessed_dataset[learner][0][i]).float()
		pred = agent.network(values).argmax()
		print(preprocessed_dataset[learner][1][i], pred)"""