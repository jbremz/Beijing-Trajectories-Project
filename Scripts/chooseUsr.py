# To select a random user for testing

import os
import random

def randUsr(root):
	'''
	Takes the root path and returns the path of a random user

	'''
	dirs = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
	return root + '/' + random.sample(dirs, 1)[0]
