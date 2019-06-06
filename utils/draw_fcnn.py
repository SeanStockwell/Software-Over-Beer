#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
import math
from net_diagram import *

"""
Created by @author: craffel
Modified on Sun Jan 15, 2017 by anbrjohn
Modifications: 
	-Changed xrange to range for python 3
	-Added functionality to annotate nodes

File can be found here:
https://gist.github.com/anbrjohn/7116fa0b59248375cd0c0371d6107a59

Further modified on Mon Jun 3, 2017 by software-over-beer
Modifications:
	- Added methods for input
"""    

rad2deg = 180 / (math.pi)

def draw_neural_net(ax, network_diagram, edge_text=None, left=0.1, right=0.90, bottom=0.1, top=0.9):
	'''
	Draw a neural network cartoon using matplotilb.
	
	:usage:
		>>> fig = plt.figure(figsize=(12, 12))
		>>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2], ['x1', 'x2','x3','x4'])
	
	:parameters:
		- ax : matplotlib.axes.AxesSubplot
			The axes on which to plot the cartoon (get e.g. by plt.gca())
		- left : float
			The center of the leftmost node(s) will be placed here
		- right : float
			The center of the rightmost node(s) will be placed here
		- bottom : float
			The center of the bottommost node(s) will be placed here
		- top : float
			The center of the topmost node(s) will be placed here
		- layer_sizes : list of int
			List of layer sizes, including input and output dimensionality
		- layer_text : list of str
			List of node annotations in top-down left-right order
	'''
	n_layers = network_diagram.n_layers
	v_spacing = (top - bottom)/float(max(network_diagram.n_layer_sizes))
	radius = v_spacing / 4.

	space_for_activation_fns = 2.25*radius*float(len(network_diagram.activation_fns))
	h_spacing = (right - left - space_for_activation_fns)/float(n_layers - 1)
	ax.axis('off')
	# Center of the activation node (or non-activation if no activation fn)
	node_x_centers_right = []
	# Center of the non-activation node
	node_x_centers_left = []

	# Nodes
	for n, layer_size in enumerate(network_diagram.n_layer_sizes):
		layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.

		x_center = left + radius

		if n > 0:
			# new x coordinate is one horizontal space unit right from the
			# right of the last node
			x_center = h_spacing + node_x_centers_right[n-1]
		
		# Here we add what would be the center of the non-activation node
		node_x_centers_left.append(x_center)

		layer_includes_activation = False
		for m in range(layer_size):
			
			y = layer_top - m*v_spacing

			### Can simplify some stuff here (unnecessary local vars)
			node = network_diagram.nodes[n][m]
			node_ec = network_diagram.nodes[n][m].edge_color

			# For now, layer_text will only be activation functions
			layer_text = network_diagram.nodes[n][m].activation_f
			###

			if layer_text:
				plt.annotate(layer_text, xy=(x_center, y), zorder=5, ha='center', va='center')
				
			plt_figures = node.get_figure(x_center,y, radius)
			for plt_figure in plt_figures:
				ax.add_artist(plt_figure)

			# Add a line between node and activation function
			if len(plt_figures) > 1:
				line = plt.Line2D([x_center + radius, x_center + 1.25*radius],[y, y],c='k')
				ax.add_artist(line)
				layer_includes_activation = True

		if layer_includes_activation:
			x_center = x_center + 2.25*radius

		node_x_centers_right.append(x_center)


	# Edges
	for n in range(1, network_diagram.n_layers):
	#for n, (layer_size_a, layer_size_b) in enumerate(
	#	zip(network_diagram.n_layer_sizes[:-1],network_diagram.n_layer_sizes[1:])):

		layer_size_a = network_diagram.n_layer_sizes[n-1]
		layer_size_b = network_diagram.n_layer_sizes[n]
		
		layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
		layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.

		# Include string on top of top edge between layers
		if edge_text:
			x_center = (node_x_centers_right[n-1] + node_x_centers_left[n]) / 2.0
			y_center = (layer_top_a + layer_top_b) / 2.0

			x_delta = node_x_centers_left[n] -node_x_centers_right[n-1]
			y_delta = layer_top_b - layer_top_a
			rotation_deg = rad2deg * math.atan2(y_delta, x_delta)
			plt.text(x_center, y_center, edge_text,
				rotation=rotation_deg, horizontalalignment='center')


		for m in range(layer_size_a):
			for o in range(layer_size_b):
				line = plt.Line2D([node_x_centers_right[n-1], node_x_centers_left[n]],
								  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
					# print(x_center, y_center)
					# print(rotation_deg)
				ax.add_artist(line)


def main():
	# Need empty strings for unlabeled nodes at start, but not at end
	
	# node_text = ['','','','h1','h2','h3','h4','h5']

	fig = plt.figure(figsize=(12, 12))
	ax = fig.gca()
	# draw_neural_net(ax, .1, .9, .1, .9, [3, 5, 2], node_text)
	
	
	#test = NetworkDiagram()
	test2 = NetworkDiagram(json_file_name="test.json")

	### How to hard code colors... will have to change this later
	test2.nodes[1][0].color = 'g'
	###
	draw_neural_net(ax, test2, edge_text="w*x + b")

	plt.show()

if __name__ == '__main__':
	main()