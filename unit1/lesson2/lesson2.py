import torch
import torch.nn as nn
import numpy as np

################################################
############ Lesson 1 - Universal approximation and NN basics ############
 # TODOs: 
 # 1. Decide everything this lesson should contain
 # 2. Implement things this lesson should contain
 # 3. Create a lesson1_problems.py for people to implement (like a hw assignment)
 #		- need to brainstorm what type of problems would be appropriate this early on
 # 4. Create a lesson1_soln.py to check problems for valid solutions
 # 		- may or may not release
 # 5. Have a beer and clean code up for release
 # 		- may include a lesson1_doc/pytorch_doc (for pytorch documentation)
 #
 # REMINDER: Don't reinvent the wheel. Add new + relevant info
 #
 # What this lesson contains:
 # i. An intro to pytorch
 # 		- Basics of what comprises the neural net. I will get into training and all of that later
 # ii. Demonstration of a neural network's ability to approximate
 # a few different functions
 # 		- Simple linear function(s) y = wx + b, or y = wx1 + wx2 + b
 #		- A more complicated function: y = cos(x) + x^2 or y = cos(x1) + x2^2
 #		- A function with multiple outputs... y,z = TODO
 # 
 #		- Neural network might be pretrained (by me) and included in git repo
 #		- Can plot error and all of that through matplotlib ... good chance to learn more about that
 #
 # Might be neat: Have one big, custom neural network that I build/have throughout series
 # 	- Each part of network will be color coded (line by line) that details which lesson
 #	covers that material.
 #  - Will have to decide if I want to custom build forward and backward propagation methods
 #
 ################################################ 


def main():
    print("Running Lesson 1")

if __name__ == "__main__":
    main()