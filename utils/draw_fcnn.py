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

def draw_neural_net(ax, network_diagram, edge_text=None, left=0.1, right=0.9, bottom=0.1, top=0.9):
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
	h_spacing = (right - left)/float(n_layers - 1)
	ax.axis('off')
	# Nodes
	for n, layer_size in enumerate(network_diagram.n_layer_sizes):
		layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
		for m in range(layer_size):
			x = n*h_spacing + left
			y = layer_top - m*v_spacing

			### Can simplify some stuff here (unnecessary local vars)
			node_color = network_diagram.nodes[n][m].color
			node_ec = network_diagram.nodes[n][m].edge_color

			# For now, layer_text will only be activation functions
			layer_text = network_diagram.nodes[n][m].activation_f
			###

			circle = plt.Circle((x,y), v_spacing/4.,
								color=node_color, ec=node_ec, zorder=4)
			ax.add_artist(circle)
			# Node annotations
			if layer_text:
				plt.annotate(layer_text, xy=(x, y), zorder=5, ha='center', va='center')


	# Edges
	for n, (layer_size_a, layer_size_b) in enumerate(
		zip(network_diagram.n_layer_sizes[:-1],network_diagram.n_layer_sizes[1:])):
		layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
		layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.

		if edge_text:
			x_center = (n*h_spacing + left + (n + 1)*h_spacing + left) / 2.0
			y_center = (layer_top_a + layer_top_b) / 2.0

			x_delta = ((n + 1)*h_spacing + left) - (n*h_spacing + left)
			y_delta = layer_top_b - layer_top_a
			rotation_deg = rad2deg * math.atan2(y_delta, x_delta)
			plt.text(x_center, y_center, edge_text,
				rotation=rotation_deg, horizontalalignment='center')

		for m in range(layer_size_a):
			for o in range(layer_size_b):
				line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
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