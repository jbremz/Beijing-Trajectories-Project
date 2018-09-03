# Creates an inventory of all the trajectories and puts it into a csv file

from trajAnalysis import trajectory
import csv, os, sys
import pandas as pd
from tqdm import tqdm

thedir = sys.argv[1]

dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]

theTrajectories = []

for direc in tqdm(dirs):
	print('User:', direc)
	trajdir = thedir + '/' + direc + '/Trajectory/'
	labelStates = [name for name in os.listdir(trajdir) if os.path.isdir(os.path.join(trajdir, name))]
	for labelState in tqdm(labelStates):
		trajs = os.listdir(thedir + '/' + direc + '/Trajectory/' + labelState + '/')
		for traj in tqdm(trajs):
			if traj != '.DS_Store':
				print(traj)
				path = thedir + '/' + direc + '/Trajectory/' + labelState + '/' + traj
				if os.path.getsize(path) > 1e6: # File-size criterion
					print('Too large')
					continue
				t = trajectory(path)
				if (len(t.points)<20) or (t.time<0.5) or (t.time>90) or (t.len < 20): # Trajectory length/duration criterion
					print('Too short/long')
					continue
				t.removeNoise()
				if t.trashy: # Trajectory quality criterion
					print('Utter trash')
					continue
				theTrajectories.append([direc + '/Trajectory/' + labelState + '/' + traj, labelState, t.time, t.len, len(t.points), t.crowLength(), t.pathCrowRatio(), t.coveredArea(), t.windowArea(), t.areaPerUnitL(), t.areaPerUnitT(), t.hurst(), t.angleDensS(), t.angleDensT(), t.transMode(), t.meanSpeed])

# TODO make this neater
paths = [traj[0] for traj in theTrajectories]
labelStates = [traj[1] for traj in theTrajectories]
times = [traj[2] for traj in theTrajectories]
lengths = [traj[3] for traj in theTrajectories]
pointCount = [traj[4] for traj in theTrajectories]
crowLength = [traj[5] for traj in theTrajectories]
pathCrowRatio = [traj[6] for traj in theTrajectories]
coveredArea = [traj[7] for traj in theTrajectories]
windowArea = [traj[8] for traj in theTrajectories]
areaPerUnitL = [traj[9] for traj in theTrajectories]
areaPerUnitT = [traj[10] for traj in theTrajectories]
hurst = [traj[11] for traj in theTrajectories]
angleDensS = [traj[12] for traj in theTrajectories]
angleDensT = [traj[13] for traj in theTrajectories]
transMode = [traj[14] for traj in theTrajectories]
meanSpeed = [traj[15] for traj in theTrajectories]

df = pd.DataFrame({'Path':paths, 'Label-state':labelStates, 'Duration':times, 'Length':lengths, 'Point Count':pointCount, 'Crow Length':crowLength, 'Path-Crow Ratio':pathCrowRatio, 'Covered Area':coveredArea, 'Window Area':windowArea, 'Area/Length':areaPerUnitL, 'Area/Time':areaPerUnitT, 'Hurst Exponent':hurst, 'Turning-angle/Length':angleDensS, 'Turning-angle/Time':angleDensT, 'Mean Speed':meanSpeed, 'Mode of Transport':transMode})
df.to_csv('../Metadata/trajFeatures.csv')