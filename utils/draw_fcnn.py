#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
from net_diagram import *

"""
Created by @author: craffel
Modified on Sun Jan 15, 2017 by anbrjohn
Modifications: 
    -Changed xrange to range for python 3
    -Added functionality to annotate nodes

File can be found here:
https://gist.github.com/anbrjohn/7116fa0b59248375cd0c0371d6107a59

Further modified on Mon Jun 3, 2017
Modifications:
	- Added methods for input
"""    



def create_fcnn_dims():
	number_of_layers = 2 + num_hidden_layers
	# TODO finish this ... 
	# Want to be able to do this from command line or json file



def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_text=None):
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
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    ax.axis('off')
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            x = n*h_spacing + left
            y = layer_top - m*v_spacing

            ### Some stuff I've added as proof-of-concept
            if m == 0:
            	node_color = 'b'
            else:
            	node_color = 'w'
            ###

            circle = plt.Circle((x,y), v_spacing/4.,
                                color=node_color, ec='k', zorder=4)
            ax.add_artist(circle)
            # Node annotations
            if layer_text:
                text = layer_text.pop(0)
                plt.annotate(text, xy=(x, y), zorder=5, ha='center', va='center')


    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)


def main():
	
	# print("slip away")
	# x = input("Type in name: ")
	# print(x)
	# y = input("type in last name: ")
	# print(y)

	# Need empty strings for unlabeled nodes at start, but not at end
	
	node_text = ['','','','h1','h2','h3','h4','h5']

	fig = plt.figure(figsize=(12, 12))
	ax = fig.gca()
	draw_neural_net(ax, .1, .9, .1, .9, [3, 5, 2], node_text)
	plt.show()
	test = NetworkDiagram()
	test.set_parameters_from_json("test.json")


if __name__ == '__main__':
	main()