import json
from net_nodes import *


class NetworkDiagram:
	'''
	Strictly used to create diagrams of fully connected neural networks
	'''

	def __init__(self, n_layers=3, n_layer_sizes=[1,1,1], 
		activation_types = ['relu']):
		'''
		Initialize the network diagram

		Inputs:
		- n_layers: Number of layers in this neural network
			INCLUDES input and output layer
		- n_layer_sizes: List of the sizes for each layer.
			INCLUDES input and output layer
			Should be a list of size (n_layers)
		- activation_types: List of strings, declaring what activation
			function to use following each hidden layer
		'''
		print("creating network diagram")
		self.n_layers = n_layers
		self.n_layer_sizes = n_layer_sizes
		self.activation_types = activation_types

		# Create a network full of default nodes
		# TODO: incorporate activation types
		self.nodes = []
		for layer_size in n_layer_sizes:
			self.nodes.append([NetworkNode() for i in range(layer_size)])
		


	def set_parameters_from_json(self, json_file_name):
		try:
			with open(json_file_name) as json_file:
				data = json.load(json_file)
				self.n_layers = data['n_layers']
				self.n_layer_sizes = data['n_layer_sizes']
				self.activation_types = data['activation_types']

		except FileNotFoundError:
			print("The specified json file was not found\nExiting Program")
			exit(1)
		except KeyError as e:
			print("This JSON file was not formatted correctly. "
				+ "The key value %s was not found. " % str(e)
				+ "Please review the readme for proper formatting.\n"
				+ "Exiting Program")
			exit(1)
		except json.JSONDecodeError:
			print("JSON file not formatted correctly. Not valid JSON.\n"
				+ "Exiting Program")
			exit(1)


