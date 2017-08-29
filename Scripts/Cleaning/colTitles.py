#Adds column names to the .csv Trajectory files

import csv, os, sys

def colLabels(root, direc, traj):

	#Read trajectory -> add column titles
	with open(root + '/' + direc + '/Trajectory/' + traj,'rt') as fin:
		cr = csv.reader(fin)
		trajContents = [line for line in cr]
		colNames = ['Lat','Long','-','Alt','dayNo','Date','Time']
		trajContents.insert(0, colNames)

	os.remove(root + '/' + direc + '/Trajectory/' + traj)

	#writes new trajectory with column titles to a new file
	with open(root + '/' + direc + '/Trajectory/' + traj,'wt') as fou:
		cw = csv.writer(fou, quotechar='', quoting=csv.QUOTE_NONE)
		cw.writerows(trajContents)

thedir = sys.argv[1]

dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]

for direc in dirs:
	trajs = os.listdir(thedir + '/' + direc + '/Trajectory')
	for traj in trajs:
		colLabels(thedir, direc, traj)
