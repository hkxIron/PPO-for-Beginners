"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim:int, out_dim:int, hidden_dim=64):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()
		# 一个三层的网络
		self.layer1 = nn.Linear(in_dim, hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, hidden_dim)
		self.layer3 = nn.Linear(hidden_dim, out_dim)

	def forward(self, observation:Union[np.ndarray, torch.Tensor]):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(observation, np.ndarray):
			observation = torch.tensor(observation, dtype=torch.float)

		activation1 = F.relu(self.layer1(observation))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
