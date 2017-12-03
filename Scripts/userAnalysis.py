# Contains user class which has methods to analyse a user's trajectories

# u = user(randUsr('/Users/JBremner/Desktop/Beijing Trajectories/Geolife Trajectories 1.3/Data'))

import csv, os, sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import radius_neighbors_graph 
from Scripts.mathsFuncs import geometric_median
from Scripts.chooseUsr import randUsr
from nolds import corr_dim

class user:
	'''
	Contains methods to analyse a user's trajectories

	'''

	def __init__(self, path):
		self.folder = path
		self.path = path + '/Trajectory'
		self.dirs = [name for name in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, name))]
		self.len = self.trajNo() 

		# Initial values to be filled on request but then saved to cut processing time.
		self.epsilon = False
		self.clusters = False
		self.points = False

	# ~~~~~~~~~~~~~~~~~~~~ SPATIAL ROUTINES ~~~~~~~~~~~~~~~~~~~~

	def startLocs(self):
		'''
		Returns array of start locations for the user's trajectories

		'''
		locs = []
		for direc in self.dirs:
			trajs = os.listdir(self.path + '/' + direc)
			for traj in trajs:
				trajPath = self.path + '/' + direc + '/' + traj
				df = pd.read_csv(trajPath)
				locs.append([df['x'].iloc[0], df['y'].iloc[0]])

		return np.array(locs)

	def endLocs(self):
		'''
		Returns array of end locations for the user's trajectories

		'''
		locs = []
		for direc in self.dirs:
			trajs = os.listdir(self.path + '/' + direc)
			for traj in trajs:
				trajPath = self.path + '/' + direc + '/' + traj
				df = pd.read_csv(trajPath)
				locs.append([df['x'].iloc[-1], df['y'].iloc[-1]])

		return np.array(locs)

	def startEndLocs(self):
		'''
		Returns commbined array of start and end locations for the user's trajectories

		'''

		locs = np.concatenate([self.startLocs(),self.endLocs()], axis=0)
		return locs

	def locs(self):
		'''
		Returns array with every location point from every trajectory - could be used to create heatmap.

		'''

		locs = []
		for direc in self.dirs:
			trajs = os.listdir(self.path + '/' + direc)
			for traj in trajs:
				trajPath = self.path + '/' + direc + '/' + traj
				df = pd.read_csv(trajPath)
				s = np.array([df['x'], df['y']]).T
				for pos in s:
					locs.append(pos)

		self.points = np.array(locs)

		return np.array(locs)

	def corrDim(self):
		'''
		Returns the correlation dimension of the start/end locations

		'''
		return corr_dim(self.startEndLocs(), emb_dim=5)
	# ~~~~~~~~~~~~~~~~~~~~ CLUSTERING ROUTINES ~~~~~~~~~~~~~~~~~~~~

	def nnDist(self, theLocs):
		'''
		Takes an array of coordinates and returns an array of the n-n distances 

		'''
		nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(theLocs)
		distances, indices = nbrs.kneighbors(theLocs)

		return distances[:,1]

	def nnDistDisp(self, theLocs):
		'''	
		Takes an array of coordinate pairs and produces a histogram of the nearest neighbour distance distribution

		'''
		distances = self.nnDist(theLocs)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.grid(True,which='both')
		ax.hist(distances,100)
		ax.set_ylabel('Frequency')
		plt.show()

	def findEps(self, theLocs):
		'''
		Returns the appropriate value of epsilon (95th percentile of n-n distances) to use with DBSCAN from given locations - there are more sophisticated ways to do this.

		'''
		self.epsilon = np.percentile(self.nnDist(theLocs),95)
		return self.epsilon

	def clusterLabels(self):
		'''
		Uses the start/end locations to find clusters of popular destinations or start locations.
		Returns the cluster labels, the array of original start/end coordinates and the core_samples_mask

		'''
		s = self.startEndLocs()
		if not self.epsilon:
			self.findEps(self.startEndLocs())

		eps = self.epsilon

		# e = 60
		# if eps < e:
		# 	eps = e

		db = DBSCAN(eps=eps, min_samples=10).fit(s)
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_

		self.n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

		return labels, s, core_samples_mask

	def clusterLocs(self):
		'''
		Returns an array of cluster positions (calulated from the mean position of all sites belonging to the cluster in question)

		'''
		labels, s = self.clusterLabels()[:2]
		labels = np.array(labels)

		unique_labels = set(labels)
		unique_labels = [lbl for lbl in unique_labels if lbl != -1]

		centres = []

		for lbl in unique_labels:
			indices = np.where(labels==lbl)[0]
			locs = []
			for i in indices:
				locs.append(s[i])
			centre = geometric_median(np.array(locs))
			centres.append(centre)

		self.clusters = np.array(centres)

		return np.array(centres)

	def plotClusters(self):
		'''
		Plots the clusters and noise points found from the DBSCAN algorithm

		'''
		labels, s, core_samples_mask = self.clusterLabels()

		# Black removed and is used for noise instead.
		unique_labels = set(labels)
		colors = [plt.cm.Spectral(each)
		          for each in np.linspace(0, 1, len(unique_labels))]
		for k, col in zip(unique_labels, colors):
		    if k == -1:
		        # Black used for noise.
		        col = [0, 0, 0, 1]

		    class_member_mask = (labels == k)

		    xy = s[class_member_mask & core_samples_mask]
		    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
		             markeredgecolor='k', markersize=10)

		    xy = s[class_member_mask & ~core_samples_mask]
		    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
		             markeredgecolor='k', markersize=2)

		plt.title('Estimated number of clusters: %d' % self.n_clusters_)
		plt.show()

	# ~~~~~~~~~~~~~~~~~~~~ TEMPORAL ROUTINES ~~~~~~~~~~~~~~~~~~~~

	def startTs(self):
		'''
		Returns an array of start datetimes from a user in datetime format

		'''
		Ts = []

		for direc in self.dirs:
			trajs = os.listdir(self.path + '/' + direc)
			for traj in trajs:
				trajPath = self.path + '/' + direc + '/' + traj
				df = pd.read_csv(trajPath)
				Ts.append(datetime.strptime(df['datetime'].iloc[0], '%Y-%m-%d %H:%M:%S'))

		return np.array(Ts)

	def endTs(self):
		'''
		Returns an array of end datetimes from a user in datetime format

		'''
		Ts = []

		for direc in self.dirs:
			trajs = os.listdir(self.path + '/' + direc)
			for traj in trajs:
				trajPath = self.path + '/' + direc + '/' + traj
				df = pd.read_csv(trajPath)
				Ts.append(datetime.strptime(df['datetime'].iloc[-1], '%Y-%m-%d %H:%M:%S'))

		return np.array(Ts)		

	def times_hist(self):
		'''
		Plots a histogram of the start times

		'''
		# Ts = np.concatenate((self.startTs(),self.endTs()), axis=0)
		Ts = self.startTs()
		Ts = np.array([t.time() for t in Ts])

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.grid(True,which='both')
		ax.hist(Ts,48)
		ax.set_ylabel('Frequency')
		# plt.show()

	def dates_hist(self):
		'''
		Plots a histogram of the start dates

		'''
		Ts = self.startTs()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.grid(True,which='both')
		ax.hist(Ts,50)
		ax.set_ylabel('Frequency')
		ax.set_xlabel('Date')
		# plt.show()

	# ~~~~~~~~~~~~~~~~~~~~ HEATMAP ROUTINES ~~~~~~~~~~~~~~~~~~~~

	def plotHeatmap(self, coords):
		'''
		Takes an array of coordinate pairs and returns a heatmap with cluster locations

		'''
		x = coords[:,0]
		y = coords[:,1]

		heatmap, xedges, yedges = np.histogram2d(x, y, bins=(100,100))
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

		plt.clf()
		plt.imshow(heatmap.T, cmap='hot', extent=extent, origin='lower', interpolation='bilinear')

		if type(self.clusters) == bool:
			clusters = self.clusterLocs()
		else:
			clusters = self.clusters

		if len(clusters) != 0:
			clx = clusters[:,0]
			cly = clusters[:,1]
			plt.plot(clx, cly,'g^', fillstyle='none')
			plt.title('Estimated number of clusters: %d' % self.n_clusters_)

		# plt.show()

	def plotAllTrajs(self):
		'''
		Plots all of the user's trajectories with cluster locations

		'''
		plt.clf()

		if type(self.points) == bool:
			locs = self.locs()
		else:
			locs = self.points

		locx = locs[:,0]
		locy = locs[:,1]

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(locx, locy, 'wo', markersize=0.1)

		if type(self.clusters) == bool:
			clusters = self.clusterLocs()
		else:
			clusters = self.clusters

		if len(clusters) != 0:
			clx = clusters[:,0]
			cly = clusters[:,1]
			ax.plot(clx, cly,'r^', fillstyle='none')

		ax.set_title('Estimated number of clusters: ' + str(self.n_clusters_))
		ax.set_ylabel('y')
		ax.set_xlabel('x')
		ax.set_facecolor('black')
		# plt.show()

	def heatmap(self):
		'''
		Produces heatmap from all user's trajectories

		'''
		if type(self.points) == bool:
			locs = self.locs()
		else:
			locs = self.points

		self.plotHeatmap(locs)

	def start_heatmap(self):
		'''
		Produces heatmap from the start positions of the user's trajectories

		'''
		self.plotHeatmap(self.startLocs())

	def end_heatmap(self):
		'''
		Produces heatmap from the end positions of the user's trajectories

		'''
		self.plotHeatmap(self.endLocs())	

	def startEnd_heatmap(self):
		'''
		Produces heatmap from the start and end positions of the user's trajectories

		'''
		self.plotHeatmap(self.startEndLocs())

	# ~~~~~~~~~~~~~~~~~~~~ MISC ROUTINES ~~~~~~~~~~~~~~~~~~~~

	def trajNo(self):
		'''
		Returns the number of trajectories a user has

		'''
		tot = 0

		for direc in self.dirs:
			trajs = os.listdir(self.path + '/' + direc)
			tot += len(trajs)

		return tot

	def modeInfo(self):
		'''	
		Produces info on the common mode of transport for a user

		'''
		files = os.listdir(self.folder)

		# WARNING - Not strictly correct since labels.csv contains labels which don't refer to any trajectory

		if 'labels.csv' in files:
			df = pd.read_csv(self.folder + '/labels.csv')
			print('Number of Labelled Trajectories:', str(len(df)) + '/' + str(self.len))
			return df['Transportation Mode']\
											.groupby(df['Transportation Mode'])\
											.count()\
											.sort_values(ascending=False)
			
		else:
			print('No labels provided')



