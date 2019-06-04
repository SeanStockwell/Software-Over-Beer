# Network Node

class NetworkNode:
	'''
	Used to enable customization of individual nodes within
	network diagram.
	'''
	def __init__(self, color='w', edge_color="k"):
		self.color = color
		self.edge_color = edge_color