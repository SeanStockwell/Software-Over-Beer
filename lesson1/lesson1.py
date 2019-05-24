import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class FeedForwardNetwork(nn.Module):
	"""
	Describe the network a bit here...
	"""
	def __init__(self, input_size=1, output_size=1):
		super(FeedForwardNetwork, self).__init__()     # This is calling the __init__() method of the parent class (nn.module)
		self.fc1 = nn.Linear(input_size, output_size)  # Only Fully-Connected Layer: x (input data) -> y (hidden node)
	
	def forward(self, x):                              # Forward pass: stacking each layer together (in this case, we just have one)
		out = self.fc1(x)
		return out


def simple_linear_calculation():
	'''
	This is a demonstration of how you can compute a simple linear
	equation using a linear neural network with PyTorch.

	In this demo, we will be computing the equation y = 2x + 5
	'''

	# This will be our input to the FeedForwardNetwork			#
	# Note that this is a 2d Array. The Neural Network expects 	#
	# an array for input. 
	x = [[2.0], [8.0], [32.0]]

	x = torch.tensor(x)

	# Declare the FeedForwardNetwork
	model = FeedForwardNetwork()

	# We are going to hardcode the values for the parameters of	
	# our model. 

	# Later, we will learn how these values can be 'learned'	#
	# using data, which is what machine learning and deep		#
	# learning is all about. For now, this is the simplest demo	#
	# of how a neural network works.							#

	# Returns list with tensors for values of W and b
	model_params = list(model.parameters())

	# Set parameter values so that linear network is equivalent
	# to 2x + 5
	model_params[0].data = torch.tensor([[2.0]])
	model_params[1].data = torch.tensor([5.0])

	# Now we can pass our input (x) into the model, getting our output (y)
	output = model(x)
	# If you're curious why we can just call model(x) and not 	#
	# model.forward(x), check out the link below:				#
	# https://discuss.pytorch.org/t/why-can-you-call-model-without-specifying-forward-method/24762 #

	# You can verify that the y is correctly calculated for 2, 8, and 32
	print(output.data)


def main():
	print("Running Lesson 1\n")
	simple_linear_calculation()


if __name__ == '__main__':
	main()