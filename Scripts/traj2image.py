from PIL import Image
import numpy as np
from Scripts.trajAnalysis import trajectory
from tqdm import tqdm

def scaleMat(M):
	'''
	Takes a matrix M and returns a matrix that has been normalised to 255 prior to conversion to an image
	'''                                
	return M*(256/M.max())

def makeImg(trajectory,size):
	'''
	Takes trajectory object and returns an image of dimensions (size,size) from the constituent points

	'''
	mat = makeMat(trajectory, size)
	mat = scaleMat(mat)
	return Image.fromarray(np.uint8(mat), 'L')

def makeMat(trajectory,range,size):
	'''
	Returns (unscaled) 2D histograms of trajectories' points of dimensions (size,size) in a given x and y range

	'''
	return np.histogram2d(trajectory.points[:,0],trajectory.points[:,1],range=range,bins=size)[0]

def imgCentre(traj):
	'''
	Takes a trajectory object and returns its middle point coordinates.

	TODO: Return the geometric median instead
	''' 
	return traj.points[len(traj.points)//2]

def windowRange(centre, side):
	'''
	Takes the centre and window side length and returns the x and y extents of the window
	(to be fed into the makeMat function)

	Returns: numpy array [[xmin, xmax],[ymin, ymax]]
	'''
	d = side/2
	xmin, xmax, ymin, ymax = centre[0] - d, centre[0] + d, centre[1] - d, centre[1] + d

	return np.array([[xmin, xmax],[ymin, ymax]])

def batchTraj2Image(df, root, size, side):
	'''
	Takes the inventory dataframe (containing N trajectories) and the root path of data

	Returns: 
	- An (N,size,size) array of image data
	- An array of Transportation Mode labels of length N

	'''
	df = df.loc[df['Point Count']>20][df['Label-state']!='Unlabelled'].copy()
	df = df.reset_index(drop=True)
	imgs = np.zeros((len(df),size,size))
	labels = []
	for i in tqdm(range(len(df))):
		t = trajectory(root + '/' + df.loc[i]['Path'])
		imgs[i] = makeMat(t, windowRange(imgCentre(t),side),size)
		labels.append(t.mode)

	return imgs, np.array(labels)

