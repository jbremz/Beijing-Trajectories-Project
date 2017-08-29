# Creates an inventory of all the trajectories and puts it into a csv file

from trajAnal import trajectory
import csv, os, sys
import pandas as pd

thedir = sys.argv[1]

dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]

theTrajectories = []

for direc in dirs:
	print('User:', direc)
	trajdir = thedir + '/' + direc + '/Trajectory/'
	labelStates = [name for name in os.listdir(trajdir) if os.path.isdir(os.path.join(trajdir, name))]
	for labelState in labelStates:
		trajs = os.listdir(thedir + '/' + direc + '/Trajectory/' + labelState + '/')
		for traj in trajs:
			path = thedir + '/' + direc + '/Trajectory/' + labelState + '/' + traj
			t = trajectory(path)
			theTrajectories.append([direc + '/Trajectory/' + labelState + '/' + traj, labelState, t.time, t.len, len(t.points)])

paths = [traj[0] for traj in theTrajectories]
labelStates = [traj[1] for traj in theTrajectories]
times = [traj[2] for traj in theTrajectories]
lengths = [traj[3] for traj in theTrajectories]
pointCount = [traj[4] for traj in theTrajectories]

df = pd.DataFrame({'Path':paths, 'Label-state':labelStates, 'Duration':times, 'Length':lengths, 'Point Count':pointCount})