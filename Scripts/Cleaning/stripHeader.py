#Strips header from trajectories, adds column titles and converts to csv

import csv, os, sys

colNames = ['Lat','Long','-','Alt','dayNo','Date','Time']

def cleanPlt(root, direc1, direc2, colNames):

	# read plt file
	with open(root + '/' + direc1 + '/Trajectory/' + direc2,'rt') as fin:
		cr = csv.reader(fin)
		filecontents = [line for line in cr][6:]
		filecontents.insert(0, colNames)

	# write csv file without header
	with open(root +'/' + direc1 + '/Trajectory/' + direc2[:-4] + '.csv','wt') as fou:
		cw = csv.writer(fou, quotechar='', quoting=csv.QUOTE_NONE)
		cw.writerows(filecontents)

	os.remove(root + '/' + direc1 + '/Trajectory/' + direc2)

thedir = sys.argv[1]

dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]

for direc in dirs:
	print('Cleaning:', direc)
	tempdirs = os.listdir(thedir + '/' + direc + '/Trajectory')
	subdirs = []
	for item in tempdirs:
		if not item.endswith('.DS_Store'):
			subdirs.append(item)
	for subdir in subdirs:
		cleanPlt(thedir, direc, subdir, colNames)