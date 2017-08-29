import csv, os, sys
from os.path import isfile, join
import pandas as pd
import numpy as np

def labelTraj(root, direc):
	'''
	Adds 'Transportation Mode' columns to trajectories and adds labels accordingly, splitting up the trajectories into appropriate 'subtrajectories'
	Also sorts the resulting trajectories into Labelled and Unlabelled folders

	'''

	# labels dataframe
	ldf = pd.read_csv(root + '/' + direc + '/labels.csv')
	ldf['Start Time'], ldf['End Time']  = pd.to_datetime(ldf['Start Time']), pd.to_datetime(ldf['End Time'])
	
	# get list of trajectories
	trajs = os.listdir(root + '/' + direc + '/Trajectory')
	trajs = [traj for traj in trajs if traj != '.DS_Store']

	os.makedirs(root + '/' + direc + '/Trajectory/Labelled/')
	os.makedirs(root + '/' + direc + '/Trajectory/Unlabelled/')

	labelled = []
	unlabelled= []

	i = 0

	for index, row in ldf.iterrows():

		if i == len(trajs):
			i = 0

		while i < len(trajs):

			traj = trajs[i]

			# trajectories dataframe
			tdf = pd.read_csv(root + '/' + direc + '/Trajectory/' + traj)
			tdf['datetime'] = tdf.Date + ' ' + tdf.Time
			tdf['datetime'] = pd.to_datetime(tdf.datetime)

			# filter for subtrajectory starting before the label
			premask = (tdf['datetime'] <= ldf['Start Time'].iloc[index])
			preSubTraj = tdf[premask].copy()

			# filter between start and end times of current label
			mask = (tdf['datetime'] <= ldf['End Time'].iloc[index]) & (tdf['datetime'] >= ldf['Start Time'].iloc[index])
			subTraj = tdf[mask].copy()

			# if matched sub trajectory: save as labelled trajectory
			if len(subTraj) != 0:
				subTraj['Transportation Mode'] = str(ldf['Transportation Mode'].iloc[index])
				name = str(subTraj['datetime'].iloc[0]).replace('-','').replace(' ','').replace(':','') + '.csv'
				subTraj.to_csv(root + '/' + direc + '/Trajectory/Labelled/' + name)
				labelled.append(name)
				# print('Labelled: ', name)
				print('Labelled: ', name, end='\r')

				# if the pre-subtrajectory isn't already labelled (or unlabelled)
				if len(preSubTraj) != 0:
					name = str(preSubTraj['datetime'].iloc[0]).replace('-','').replace(' ','').replace(':','') + '.csv'
					if name not in labelled and name not in unlabelled:
						# print('Presub: ', name)
						print('Presub: ', name, end='\r')
						preSubTraj['Transportation Mode'] = '-'
						preSubTraj.to_csv(root + '/' + direc + '/Trajectory/Unlabelled/' + name)
						unlabelled.append(name)

				# if at end of indexes - TAG - what if the rest of the trajectory needs to be exported as unlabelled?
				if index + 1 == len(ldf):
					startIndex = subTraj.index[-1] + 1
					if startIndex > tdf.index[-1]:
						break
					nolabelSubTraj = tdf[startIndex:].copy()
					nolabelSubTraj['Transportation Mode'] = '-'
					name = str(tdf['datetime'].iloc[startIndex]).replace('-','').replace(' ','').replace(':','') + '.csv'
					nolabelSubTraj.to_csv(root + '/' + direc + '/Trajectory/Unlabelled/' + name)
					unlabelled.append(name)
					# print('PostUnlabelled: ', name)
					print('PostUnlabelled: ', name, end='\r')
					break

				# if the next label is in the same trajectory: move to the next label
				if ldf['Start Time'].iloc[index+1] <= tdf['datetime'].iloc[-1]:
					break

				# otherwise: save the rest of the trajectory as unlabelled
				else:
					# os.remove(root + '/' + direc + '/Trajectory/' + traj)
					startIndex = subTraj.index[-1] + 1
					if startIndex > tdf.index[-1]:
						i += 1
						continue # tag - break?
					nolabelSubTraj = tdf[startIndex:].copy()
					nolabelSubTraj['Transportation Mode'] = '-'
					name = str(tdf['datetime'].iloc[startIndex]).replace('-','').replace(' ','').replace(':','') + '.csv'
					nolabelSubTraj.to_csv(root + '/' + direc + '/Trajectory/Unlabelled/' + name)
					unlabelled.append(name)
					# print('PostUnlabelled: ', name)
					print('PostUnlabelled: ', name, end='\r')
					i += 1
					continue

			# if at end of indexes
			if index + 1 == len(ldf):
				break			

			# if no match: save as unlabelled trajectory
			elif ldf['Start Time'].iloc[index + 1] > tdf['datetime'].iloc[-1] and traj not in unlabelled:
				tdf['Transportation Mode'] = '-'
				tdf.to_csv(root + '/' + direc + '/Trajectory/Unlabelled/' + traj)
				unlabelled.append(traj)
				# print('Unlabelled: ', traj)
				print('Unlabelled: ', traj, end='\r')
				i += 1
				continue

			# if at end of trajectory: move to next one
			if len(subTraj) != 0:
				if subTraj.index[-1] == tdf.index[-1]:
					i += 1

			# otherwise: move onto next label but keep the same trajectory
			else:
				break

		else:
			continue

def removeOriginals(root, direc):
	'''
	Removes the original trajectory .csv files
	'''
	files = [name for name in os.listdir(thedir + '/' + direc + '/Trajectory') if os.path.isfile(os.path.join(thedir + '/' + direc + '/Trajectory', name))]
	files = [file for file in files if file != '.DS_Store']

	for file in files:
		os.remove(root + '/' + direc + '/Trajectory/' + file)

thedir = sys.argv[1]

dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]

for direc in dirs:
	print('User:', direc)
	if 'labels.csv' in os.listdir(thedir + '/' + direc):
		labelTraj(thedir, direc)
		removeOriginals(thedir, direc)
	else:
		print('No Labels Present')
		os.makedirs(thedir + '/' + direc + '/Trajectory/Unlabelled/')
		trajs = [name for name in os.listdir(thedir + '/' + direc + '/Trajectory/') if os.path.isfile(os.path.join(thedir + '/' + direc + '/Trajectory/', name))]
		trajs = [traj for traj in trajs if traj != '.DS_Store']
		for traj in trajs:
			os.rename(thedir + '/' + direc + '/Trajectory/' + traj, thedir + '/' + direc + '/Trajectory/Unlabelled/' + traj)
	print('\n')

