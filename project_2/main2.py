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

# 3. Load the network
# a. Make an instance of the agent(structure that have a neural network and methods to its trainment)
agent = Agent()

# b. load
agent.network.load_state_dict(torch.load('checkpoint.pth'))
agent.network.eval()

option = -1

while True :
	print('Existem', len(test), 'alunos no conjunto de teste.')

	option = input('selecione um dos ' + str(len(test)) + ' alunos para prever se irá desistir\n')
	question = input('selecione uma das 35 semanas do curso\n')

	values = torch.tensor(test[int(option)][0][int(question)]).float()
	pred = agent.network(values).argmax()

	print('\n')

	if pred.float().item() == 0.0:
		print('a predição é que o aluno NÃO DESISTA')
	elif pred.float().item() == 1.0:
		print('a predição é que o aluno DESISTA')

	if test[int(option)][1][int(question)] == 0:
		print('o que realmente aconteceu: NÃO DESISTIU')
	elif test[int(option)][1][int(question)] == 1:
		print('o que realmente aconteceu: DESISTIU')
	print('--------------------------------------------------------------------------------\n')

