# Functions for resampling the trajectory at a constant time-step

import csv, os, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def sRate(dataFrame, percentile):
	'''
	Returns the sample rate given a certain 0 < rateFraction < 1.0 of the smallest timesteps to ignore. 
	E.G. timesteps = [1,1,2,3,4], rateFraction = 0.4 --> [1,1] are ignored returning an sRate of 2

	'''

	df = dataFrame
	dts = np.sort(np.array(df['datetime'].diff()[1:])/10**6) # average time-step in milliseconds
	dts = np.array([dt.item() for dt in dts])
	rate = int(np.percentile(dts,percentile))
	return rate

def resampleTraj(dataFrame, percentile):
	'''
	Takes the dataFrame of a trajectory and a resampling percentile and overwrites the original trajectory with the resampled one

	'''

	df = dataFrame

	if len(df) < 3:
		return

	for column in df.columns:
		if 'Unnamed' in column:
			df.drop(column, axis=1, inplace=True)

	if 'datetime' not in df.columns:
		df['datetime'] = df.Date + ' ' + df.Time
		df['datetime'] = pd.to_datetime(df.datetime)
		df.drop(['Date', 'Time', '-'], axis=1, inplace=True)
	else:
		df['datetime'] = pd.to_datetime(df.datetime)

	# interpolate in the time domain when there are multiple space points per time (second)
	for n, row in df.iterrows():
		time = row['datetime']
		mask = (df['datetime'] == time)
		sameSec = df[mask]
		l = len(sameSec)
		if l > 1:
			dt = 1./float(l)
			for i in range(len(sameSec)):
				index = sameSec.index[i]
				df['datetime'].iloc[index] = time + timedelta(0,dt*i)

	# calculate the resampling step size
	step = sRate(df, percentile)

	if 'Transportation Mode' in df.columns:
		transMode = df['Transportation Mode'].iloc[0]
	elif 'Transportation Mode' not in df.columns:
		transMode = '-'

	df = df.set_index('datetime')
	finalTime = df.index[-1]

	# calculate offset so that the resampling starts on the first time
	date = df.index[0].date()
	start = datetime.strptime(str(date) + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
	dt = (df.index[0] - start).seconds * 1000
	offset = dt % step

	# resample the trajectory
	df = df.resample(str(step)+'L', base=offset, closed='right').mean()
	df = df.interpolate()

	# extrapolate over the final point simply using the difference from the last point - perhaps not valid
	if (finalTime != df.index[-1]) & ((finalTime - df.index[-1]).total_seconds()*1000 > step/2):
		df = pd.DataFrame(data=df, index=pd.date_range(start=df.index[0], periods=len(df.index) + 1, freq=df.index.freq))
		df.iloc[-1] =  df.iloc[-2]+df.diff().iloc[-2] 

	df['Transportation Mode'] = transMode
	df.index.name = 'datetime'
	df['datetime'] = df.index
	df.index = range(len(df))

	return df

# def masterMeanTime(root):

# 	dirs = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
# 	meanTimes = []

# 	for direc in dirs:
# 		trajs = os.listdir(root + '/' + direc + '/Trajectory')
# 		for traj in trajs:
# 			meanTimes.append(meanTime(root, direc, traj))

# 	return np.mean(meanTimes)

# thedir = sys.argv[1]
