import csv, os, sys
import pandas as pd
import numpy as np

def meanTime(root, direc, traj):

	df = pd.read_csv(root + '/' + direc + '/Trajectory/' + traj)
	dtBar = df.dayNo.diff().mean() * 86400 #average time-step in seconds
	return dtBar

def masterMeanTime(root):

	dirs = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
	meanTimes = []

	for direc in dirs:
		trajs = os.listdir(root + '/' + direc + '/Trajectory')
		for traj in trajs:
			meanTimes.append(meanTime(root, direc, traj))

	return np.mean(meanTimes)

thedir = sys.argv[1]

print(masterMeanTime(thedir))



