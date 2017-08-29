# To select a random trajectory for testing

import os
import random

def randTraj(root, labelled=True, either=False):
	'''
	Takes the root path and returns the path of a random user

	'''
	found = False

	while found == False:
		dirs = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
		user = root + '/' + random.sample(dirs, 1)[0] + '/Trajectory/'
		subdirs = [name for name in os.listdir(user) if os.path.isdir(os.path.join(user, name))]
		if either == True:
			trajsPath = user + random.sample(subdirs, 1)[0]
			trajs =  [name for name in os.listdir(trajsPath) if os.path.isfile(os.path.join(trajsPath, name))]
			return trajsPath + '/' + random.sample(trajs, 1)[0]
		if 'Labelled' in subdirs and labelled == True:
			trajsPath = user + 'Labelled/'
			trajs =  [name for name in os.listdir(trajsPath) if os.path.isfile(os.path.join(trajsPath, name))]
			return trajsPath + random.sample(trajs, 1)[0]
		elif labelled == False:
			trajsPath = user + 'Unlabelled/'
			trajs =  [name for name in os.listdir(trajsPath) if os.path.isfile(os.path.join(trajsPath, name))]
			if len(trajs) != 0:
				return trajsPath + random.sample(trajs, 1)[0]