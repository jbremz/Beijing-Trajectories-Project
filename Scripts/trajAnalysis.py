# Contains trajectory class which has methods to analyse a single trajectory

import csv, os, sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nolds import corr_dim, hurst_rs, dfa
from Scripts.resample import resampleTraj
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from Scripts.stdbscan import st_dbscan, retrieve_neighbors
from Scripts.mathsFuncs import geometric_median
from shapely.geometry import LineString, box, Point
import shapely.ops as shlops # to use in finding buffer
from shapely import strtree
from math import ceil, isnan
from Scripts.chooseTraj import randTraj

class trajectory:
	'''
	Contains routines for single trajectory analysis

	'''

	def __init__(self, path):
		# self.df = resampleTraj(pd.read_csv(path))
		self.df = pd.read_csv(path)
		self.df.loc[:,'datetime'] = pd.to_datetime(self.df.datetime)
		self.path = path

		for column in self.df.columns:
			if 'Unnamed' in column:
				self.df.drop(column, axis=1, inplace=True)

		# Initial Values
		self.points = self.locs()

		if len(self.points) > 1:
			self.len = LineString(self.points).length
		else:
			self.len = 'NaN'

		self.mode = self.transMode()
		self.time = self.timeSpent()/60 # in MINUTES
		self.meanSpeed = self.averageSpeed() 
		self.maxSpeeds = {'walk':12.4, 'run':12.4, 'car':121, 'train':120,'airplane':313, 'taxi':121, 'bus':121, 'subway':120, 'Unlabelled':313, 'bike':75, 'boat':142, 'motorcycle':112}
		self.lingerIndices = False
		self.lingerLocs = False
		self.lingerTimes = False
		self.corr_dim = False

		# Bools
		self.resampled = False
		self.cleaned = False # has the noise been removed?
		self.trashy = False # does the trajectory have more than ten 'noisy' datapoints in a row?

	# ~~~~~~~~~~~~~~~~~~~~ SPATIAL ROUTINES ~~~~~~~~~~~~~~~~~~~~

	def crowLength(self):
		'''
		Calculates the straight line distance between the start and end of the trajectory

		'''
		df = self.df
		start  = np.array([df['x'].iloc[0], df['y'].iloc[0]])
		end = np.array([df['x'].iloc[-1], df['y'].iloc[-1]])
		ds = end - start
		return np.sqrt(np.dot(ds,ds))

	def pathCrowRatio(self):
		'''
		Returns the ratio between the path legnth of a trajectory and the straight-line distace between the start and end points

		'''
		return float(self.len/self.crowLength())

	def locs(self):
		'''
		Returns an array of coordinate pairs for all the points in the trajectory

		'''
		locs = []

		points = np.array([self.df.loc[:,'x'], self.df.loc[:,'y']]).T

		return points

	def coveredArea(self, radius=30, RES=16):
		'''
		Returns the area covered by the trajectory using the shapely buffer - need to think about an appropriate radius here.

		'''
		COOR = self.points
		circ=[]

		# for i in range(len(COOR)):
		# 	circ.append(Point(COOR[i,0],COOR[i,1]).buffer(radius,RES))
		# 	# 20 is the resolution, default is 16, 20 is good enough for coverage problem
		# 	return shlops.unary_union(circ).area

		line = LineString(self.points)
		buff = line.buffer(radius)
		return buff.area

	def windowArea(self):
		'''
		Returns the area of the smallest rectangular window containing the trajectory

		'''
		bounds = [np.min(self.points[:,0]), np.min(self.points[:,1]), np.max(self.points[:,0]), np.max(self.points[:,1])]
		area = (bounds[2]-bounds[0])*(bounds[3]-bounds[1])

		return area

	def areaPerUnitL(self):
		'''
		Returns the area covered per unit length

		'''
		if not self.cleaned:
			self.removeNoise()

		return self.coveredArea()/self.len

	def areaPerUnitT(self):
		'''
		Returns the area covered per unit length

		'''
		if not self.cleaned:
			self.removeNoise()

		return self.coveredArea()/(self.time*60)

	def hurst(self, xy=False):
		'''
		Returns the Hurst exponent of the data
		TODO Only seems to work one dimensionally
		if xy=True function returns an array for the x and the y value of the exponent, otherwise it is averaged across both dimensions

		'''
		if not self.cleaned:
			self.removeNoise()

		hurstExp = np.array([hurst_rs(self.points[:,0]), hurst_rs(self.points[:,1])])

		self.hurstExp = np.mean(hurstExp)

		if xy:
			return hurstExp
		else:
			return self.hurstExp

	def DFA(self):
		'''	
		Returns the H exponent from detrended fluctuation analysis 
		Seems like the number of points needs to be over 70 or so for good results

		'''
		if not self.cleaned:
			self.removeNoise()

		return dfa(self.points)

	def angles(self):
		'''
		Returns the turning angle magnitudes for a trajectory at each sample point

		'''

		df = self.df
		df['dx'] = df.loc[:,'x'].diff()
		df['dy'] = df.loc[:,'y'].diff()
		angles = []

		for i, row in df[1:-1].iterrows():
			v1 = np.array([row['dx'],row['dy']])
			v2 = np.array([df.loc[i+1,'dx'],df.loc[i+1,'dy']])
			if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
				continue
			angles.append(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))

		angles = np.array(angles)
		angles = angles[~np.isnan(angles)]

		angles = abs(np.rad2deg(np.array(angles)))

		return angles

	def angleDensS(self):
		'''
		Returns the magnitude of the turning angle per unit length of the trajectory

		'''	
		return np.sum(self.angles())/self.len

	def angleDensT(self):
		'''
		Returns the magnitude of the turning angle per unit time of the trajectory

		'''
		return np.sum(self.angles())/(self.time*60)

	def averageSpeed(self):
		'''
		Returns the average speed of the trajectory

		'''
		if type(self.len) == float and self.time>0:
			return self.len/(self.time*60)
		else:
			return 'NaN'

	# ~~~~~~~~~~~~~~~~~~~~ PLOT ROUTINES ~~~~~~~~~~~~~~~~~~~~	

	def plotTraj(self, clusters=False):
		'''
		Plots the trajectory

		'''
		plt.clf()

		if not self.cleaned:
			self.removeNoise()

		if self.trashy:
			print('It\'s a trashy trajectory, I\'m not plotting this shit')
			return

		locx = self.points[:,0]
		locy = self.points[:,1]

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(locx, locy, 'wo', markersize=1)

		ax.plot(locx[-1], locy[-1],'r^', fillstyle='none')
		ax.plot(locx[0], locy[0],'g^', fillstyle='none')

		if clusters == True:
			# Show the clusters

			if type(self.lingerLocs) == bool:
				lingerSpots = self.findLingerLocs()
			else:
				lingerSpots = self.lingerLocs

			if len(lingerSpots) != 0:
				clx = lingerSpots[:,0]
				cly = lingerSpots[:,1]
				ax.plot(clx, cly,'m^', fillstyle='none')

		# if type(self.corr_dim) == bool:
		# 	self.corrDim()
		# plt.title('Correlation Dimension: ' + str(self.corr_dim))

		plt.title('Mode of Transport: ' + self.mode)
		ax.set_ylabel('y')
		ax.set_xlabel('x')
		ax.set_facecolor('black')
		# plt.show()

	# ~~~~~~~~~~~~~~~~~~~~ TEMPORAL ROUTINES ~~~~~~~~~~~~~~~~~~~~

	def timeSpent(self):
		'''
		Calculates the overall length (in time) of the trajectory in seconds
		
		'''
		df = self.df
		return (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).seconds

	def timeSteps(self):
		'''
		Returns a sorted array with the timesteps (in seconds) from the trajectory

		'''
		dts = self.df['datetime'].diff()
		dts = np.array(dts.astype(int)[1:])/10**9
		return dts

	# ~~~~~~~~~~~~~~~~~~~~ CLUSTERING ROUTINES ~~~~~~~~~~~~~~~~~~~~

	def nnDist(self):
		'''
		Takes an array of coordinates and returns an array of the n-n distances 

		'''
		nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(self.points)
		distances, indices = nbrs.kneighbors(self.points)

		return distances[:,1]

	def lingering(self):
		'''
		Sets the self.df to a dataframe containing a column with cluster index for locations of suspected lingering using the st_dbscan()
		Also saves the lingering spot indices to self.lingerIndices
		
		'''
		df = self.df
		self.df = st_dbscan(df,20,300,2,300)
		self.lingerIndices = df['cluster'].unique()

	def findLingerLocs(self):
		'''
		Returns an array of the suspected linger locations

		'''
		if not self.cleaned:
			self.removeNoise()

		if type(self.lingerIndices) == bool:
			self.lingering()

		lingers = [n for n in self.lingerIndices if n != -1]

		lingerLocs = []
		lingerTimes = []

		for lingerIndex in lingers:
			cdf = self.df[self.df['cluster'] == lingerIndex]
			points = np.array([np.array(cdf['x']),np.array(cdf['y'])]).T
			centre = np.median(points, axis=0)
			timeSpent = (cdf['datetime'].iloc[-1] - cdf['datetime'].iloc[0]).seconds
			lingerTimes.append(timeSpent)
			lingerLocs.append(centre)

		self.lingerLocs = np.array(lingerLocs)
		self.lingerTimes = np.array(lingerTimes)

		return self.lingerLocs

	# ~~~~~~~~~~~~~~~~~~~~ FRACTAL ROUTINES ~~~~~~~~~~~~~~~~~

	def corrDim(self, emb_dim=5):
		'''
		Returns the correlation dimension of trajectory

		'''
		if not self.cleaned:
			self.removeNoise()

		self.corr_dim = corr_dim(self.points, emb_dim)

		return self.corr_dim

	# ~~~~~~~~~~~~~~~~~~~~ NOISE ROUTINES ~~~~~~~~~~~~~~~~~~

	def removeNoise(self):
		'''
		Finds points with extreme velocities and removes them.
		If 10 or more points in a row are detected as noise, then the trajectory is marked as too noisy

		'''

		if not self.resampled:
			self.df = resampleTraj(self.df, 10)
			self.resampled = True

		self.df.loc[:,'Speed'] = None

		vThresh = self.maxSpeeds[self.mode]

		trashCount = 0

		while not self.trashy:

			if trashCount == 0:
				self.df.loc[1:,'Speed'] = np.sqrt(self.df.loc[:,'x'].diff()[1:]**2+self.df.loc[:,'y'].diff()[1:]**2)/self.df['datetime'].diff()[1:].apply(lambda x: x.seconds)
				
				# Pad initial velocity value with second one
				self.df.loc[0,'Speed'] = self.df.loc[1,'Speed']

			else:
				self.df.loc[:,'Speed'] = np.sqrt(self.df.loc[:,'x'].diff()[1:]**2+self.df.loc[:,'y'].diff()[1:]**2)/self.df['datetime'].diff()[1:].apply(lambda x: x.seconds)
				self.df.loc[0,'Speed'] = initSpeed

			mask = (self.df['Speed'] <= vThresh)

			if len(self.df[~mask]) == 0:
				break

			self.df = self.df[mask]
			trashCount += 1

			self.df = self.df.reset_index(drop=True)

			if len(self.df) == 0:
				self.trashy = True
				break

			initSpeed = self.df.loc[:,'Speed'].iloc[0]

			if trashCount >= 10:
				self.trashy = True

		self.points = self.locs()
		self.cleaned = True

		return

	# ~~~~~~~~~~~~~~~~~~~~ MISC ROUTINES ~~~~~~~~~~~~~~~~~~~~

	def transMode(self):
		'''	
		Returns the mode of transport

		'''
		if 'Transportation Mode' in self.df.columns:

			mode = self.df['Transportation Mode'].iloc[0]

			if mode == '-':
				return 'Unlabelled'

			else:
				return mode
		else:
			return 'Unlabelled'



