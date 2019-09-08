import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import time

from config import Config
from model import Net


class Agent():
	def __init__(self):
		# Build network
		self.network = Net()

		# Create optimizer
		self.optimizer = optim.SGD(self.network.parameters(), lr=Config.LEARNING_RATE)

	def optimize(self, target, values):
		
		# Perform feedforward
		values = torch.tensor(values, requires_grad=True).float()
		target = torch.tensor(target, requires_grad=True)

		predict = self.network(values)
		predict = predict.argmax(1)

		# Compute Huber loss
		criterion = nn.MSELoss()
		#print('predict.size', predict.size())
		#print('target.size', target.size())

		if target.size() != predict.size():
			print(target)
			time.sleep(5) 
		
		loss = criterion(predict.float(), target.float())	

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return predict, loss.item()