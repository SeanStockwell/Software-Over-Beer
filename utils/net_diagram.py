import json
from net_nodes import *


class NetworkDiagram:
	'''
	Strictly used to create diagrams of fully connected neural networks
	'''

	def __init__(self, n_layers=3, n_layer_sizes=[1,1,1],
		node_default_color='w', node_default_edge_color='k',
		activation_fns = {}, torch_mod = None, tensorflow_mod = None,
		json_file_name=None):
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
		- node_default_color: String specifying the default color for
			the network's nodes
		- node_default_edge_color: String specifying the default color
			for the edges of the network's nodes
		'''
		print("Creating network diagram")
		if torch_mod:
			print("Constructing from pytorch model")
		elif tensorflow_mod:
			print("Constructing from tensorflow model")
		elif json_file_name:
			print("Constructing from json file")
			self.set_parameters_from_json(json_file_name)
		else:
			print("Constructing Neural Network directly from parameters")
			self.n_layers = n_layers
			self.n_layer_sizes = n_layer_sizes
			self.default_color = node_default_color
			self.default_edge_color = node_default_edge_color
			self.activation_fns = activation_fns
			self.initialize_nodes()


	def initialize_nodes(self):
		# Create a network full of default nodes
		# TODO: incorporate activation types
		self.nodes = []
		for i, layer_size in enumerate(self.n_layer_sizes):
			act_fn = None
			if str(i) in self.activation_fns and self.activation_fns[str(i)]:
				act_fn = self.activation_fns[str(i)]

			self.nodes.append([NetworkNode(color=self.default_color,
				edge_color=self.default_edge_color, activation_f=act_fn)
				for i in range(layer_size)]
				)


	def set_parameters_from_json(self, json_file_name):
		try:
			with open(json_file_name) as json_file:
				data = json.load(json_file)
				self.n_layers = data['n_layers']
				self.n_layer_sizes = data['n_layer_sizes']
				self.default_color = data['default_color']
				self.default_edge_color = data['default_edge_color']
				self.activation_fns = data['activation_fns']
				self.initialize_nodes()

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

	def set_parameters_from_pytorch(self):
		'''
		TODO
		'''
		pass

	def set_parameters_from_tensorflow(self):
		'''
		TODO
		'''
		pass

