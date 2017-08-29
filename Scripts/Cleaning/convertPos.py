# Creates new columns with x, y coordinates in metres (converted from the latitude and longitude coordinates)

import csv, os, sys
import pandas as pd
import numpy as np

#The median of the mean position of all the trajectories. Here used as the origin of the new Cartesian system
startPos = [39.976483585105079, 116.337387332238023]

def haversine(startPos, endPos):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """

    radius = earthRad(endPos[0])

    lon1, lat1, lon2, lat2 = map(np.radians, [startPos[1], startPos[0], endPos[1], endPos[0]])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = radius * c
    return distance

def gimmeCoords(origin, endPos):
	'''
	Uses haversine to map the lat/long coords to (pseudo-)cartesian coordinates in metres with a specified origin

	'''
	x = haversine(origin, [origin[0], endPos[1]])
	y = haversine(origin, [endPos[0], origin[1]])

	if endPos[1] - origin[1] < 0:
		x = x*(-1)
	if endPos[0] - origin[0] < 0:
		y = y*(-1)

	return [x,y]

def earthRad(Lat):
	'''
	Calculates the Earth's radius (in m) at a given latitude using an ellipsoidal model. Major/minor axes from NASA

	'''
	a = 6378137
	b = 6356752
	Lat = np.radians(Lat)
	g = (a**2*np.cos(Lat))**2 + (b**2*np.sin(Lat))**2
	f = (a*np.cos(Lat))**2 + (b*np.sin(Lat))**2
	R = np.sqrt(g/f)
	return R

def addCart(root, direc, labelState, traj):
	'''
	Adds x, y coordinate columns to trajectories from lat/long columns, using startPos (in this case the median mean position of all trajectories) as the origin.

	'''

	df = pd.read_csv(root + '/' + direc + '/Trajectory/' + labelState + '/' +traj)
	df[['x','y']] = df[['Lat','Long']].apply(lambda x: gimmeCoords(startPos, [x[0],x[1]]),axis=1).copy()
	for column in df.columns:
		if 'Unnamed' in column:
			df.drop(column, axis=1, inplace=True)
	df.to_csv(root + '/' + direc + '/Trajectory/' + labelState + '/' + traj)

thedir = sys.argv[1]

dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]	

for direc in dirs:
	print('Converting User:', direc)
	trajdir = thedir + '/' + direc + '/Trajectory/'
	labelStates = [name for name in os.listdir(trajdir) if os.path.isdir(os.path.join(trajdir, name))]
	for labelState in labelStates:
		trajs = os.listdir(thedir + '/' + direc + '/Trajectory/' + labelState + '/')
		for traj in trajs:
			print(traj)
			addCart(thedir, direc, labelState, traj)

