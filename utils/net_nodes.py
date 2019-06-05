import matplotlib.pyplot as plt

class NetworkNode:
	'''
	Used to enable customization of individual nodes within
	network diagram.
	'''
	def __init__(self, color='w', edge_color="k", description=None, activation_f=None):
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


	def get_figure(self, x_center, y_center, radius):
		if self.activation_f:
			return self.get_rhombus(x_center, y_center, radius)
		else:
			return self.get_circle(x_center, y_center, radius)


	def get_rhombus(self, x_center, y_center, radius):
		'''
		Returns a plt.Polygon shape as a rhombus
		'''
		left_pt = (x_center - radius, y_center)
		bottom_pt = (x_center, y_center - radius)
		right_pt = (x_center + radius, y_center)
		top_pt = (x_center, y_center + radius)
		# Polygon coordinate order: bottom, right, top, left
		return plt.Polygon((bottom_pt, right_pt, top_pt, left_pt),
			color=self.color, ec=self.edge_color, zorder=4)


	def get_circle(self, x_center, y_center, radius):
		'''
		Returns a plt.Circle
		'''
		return plt.Circle((x_center, y_center), radius, color=self.color,
			ec=self.edge_color, zorder=4)
