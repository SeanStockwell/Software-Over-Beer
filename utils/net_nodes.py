class NetworkNode:
	'''
	Used to enable customization of individual nodes within
	network diagram.
	'''
	def __init__(self, color='w', edge_color="k", description="", activation_f=None):
		'''
		Initialize the NetworkNode

		Inputs:
		color: String that represents the node's color (see readme.txt for color codes)
		edge_color: String that represents the node's edge color
		description: String that may describe something about node
		activation_f: If this node represents an activation function, then the
			constructor should be passed a string specifying the activation func.
		'''
		self.color = color
		self.edge_color = edge_color
		self.description = description
		self.activation_f = activation_f