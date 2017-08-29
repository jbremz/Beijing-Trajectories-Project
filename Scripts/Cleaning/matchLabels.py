#Matches tags in the labels.csv file to the trajectory files and keeps only those where both the tag and file are present

import csv, os, sys

def cleanLabels(root, direc):

	#strips out the start times from the labels.csv
	with open(root + '/' + direc + '/labels.csv','rt') as fin:
		cr = csv.reader(fin)
		labelContents = [line for line in cr]
		startTimes = [entry[0] for entry in labelContents[1:]]
		startLabels = [time.replace('/','').replace(' ','').replace(':','') + '.csv' for time in startTimes]

	trajs = os.listdir(root + '/' + direc + '/Trajectory')

	#deletes all trajectories which don't have associated labels
	for traj in trajs:
		if traj not in startLabels:
			os.remove(root + '/' + direc + '/Trajectory/' + traj)

	filecontents = [labelContents[0]] #the new labels.csv to be written

	#finds index of labels which have corresponding trajectories available
	for i in range(len(startLabels)):
		if startLabels[i] in trajs:
			filecontents.append(labelContents[i+1])	

	os.remove(root + '/' + direc + '/labels.csv')

	#writes trimmed labels.csv file
	with open(root + '/' + direc + '/labels.csv','wt') as fou:
		cw = csv.writer(fou, quotechar='', quoting=csv.QUOTE_NONE)
		cw.writerows(filecontents)

thedir = sys.argv[1]

dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]

for direc in dirs:
	cleanLabels(thedir, direc)