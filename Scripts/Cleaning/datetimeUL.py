# Converts the Date and Time cols to datetime trajectories for users with only unlabelled trajectories


import csv, os, sys
from datetime import datetime
import pandas as pd
import numpy as np

thedir = sys.argv[1]

dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]	

for direc in dirs:
	print('Converting User:', direc)
	trajdir = thedir + '/' + direc + '/Trajectory/'
	labelStates = [name for name in os.listdir(trajdir) if os.path.isdir(os.path.join(trajdir, name))]
	if 'Labelled' not in labelStates:
		for labelState in labelStates:
			trajs = os.listdir(thedir + '/' + direc + '/Trajectory/' + labelState + '/')
			for traj in trajs:
				df = pd.read_csv(thedir + '/' + direc + '/Trajectory/' + labelState + '/' + traj)

				for column in df.columns:
					if 'Unnamed' in column:
						df.drop(column, axis=1, inplace=True)
							
				df['datetime'] = df.Date + ' ' + df.Time
				df['datetime'] = pd.to_datetime(df.datetime)
				df.drop(['Date', 'Time', '-'], axis=1, inplace=True)
				df.to_csv(thedir + '/' + direc + '/Trajectory/' + labelState + '/' + traj)

