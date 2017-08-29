# Different analysis functions for analysis of the trajectories

from trajAnal import trajectory
import pandas as pd
import time
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import iqr

# An inventory of all the trajectories with their path, label-state, duration and length
idf = pd.read_csv('/Users/JBremner/Desktop/Beijing Trajectories/Geolife Trajectories 1.3/Metadata/Inventory.csv')

def length_area(root):
	'''
	Creates plots of path length vs area per unit length

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] > 20][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(1000)
	df = df.reset_index(drop=True)	
	lengths = []
	areaPerUnitLs = []
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		lengths.append(t.len)
		areaPerUnitLs.append(t.areaPerUnitL())

	l = np.array([lengths, areaPerUnitLs])

	plt.scatter(l[0],l[1], s=1)
	plt.xlabel('Length (m)')
	plt.ylabel('Area per unit length (m)')
	plt.show()

	return l

def time_area(root):
	'''
	Creates plots of path length vs area per unit length

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] > 20][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(1000)
	df = df.reset_index(drop=True)	
	times = []
	areaPerUnitLs = []
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		times.append(t.time)
		areaPerUnitLs.append(t.areaPerUnitL())

	l = np.array([areaPerUnitLs, times])

	plt.scatter(l[0],l[1], s=1)
	plt.ylabel('Duration (minutes)')
	plt.xlabel('Area per unit length (m)')
	plt.show()

	return l

def time_corrDim(root):
	'''
	Creates plots of path length vs area per unit length

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] > 20][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(1000)
	df = df.reset_index(drop=True)	
	times = []
	corrDims = []
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		times.append(t.time)
		corrDims.append(t.corrDim())

	l = np.array([corrDims, times])

	plt.scatter(l[0],l[1], s=1)
	plt.ylabel('Duration (minutes)')
	plt.xlabel('Correlation Dimension')
	plt.show()

	return l

def length_corrDim(root):
	'''
	Creates plots of length vs correlation dimension

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] > 50][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(500)
	df = df.reset_index(drop=True)
	lengths = []
	corrDims = []
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		lengths.append(t.len)
		corrDims.append(t.corrDim())

	l = np.array([lengths, corrDims])

	plt.scatter(l[0],l[1], s=1)
	plt.xlabel('Length (m)')
	plt.ylabel('Correlation Dimension')
	plt.show()

	return l

def hurst_XY(root):
	'''
	Creates plots of path the hurst exponent in the y-direction vs x-direction

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 60][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(1000)
	df = df.reset_index(drop=True)	
	hurst_x = []
	hurst_y = []
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		if len(t.points) < 41:
			continue
		hurst = t.hurst(xy=True)
		hurst_x.append(hurst[0])
		hurst_y.append(hurst[1])

	l = np.array([hurst_x, hurst_y])

	plt.scatter(l[0],l[1], s=1)
	plt.xlabel('Hurst exponent in x-direction')
	plt.ylabel('Hurst exponent in y-direction')
	plt.show()

	return l

def find_lowHurst(root):
	'''
	Returns a list with the paths of trajectories with small hurst exponents

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] > 20][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(1000)
	df = df.reset_index(drop=True)	
	odd_hurst = []
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		if len(t.points) < 41:
			continue
		if t.hurst() < 0.5:
			odd_hurst.append(t.path)

	return odd_hurst


def hurst_length(root):
	'''
	Creates plots of hurst exponent vs length of trajectory

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 60][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(1000)
	df = df.reset_index(drop=True)	
	hursts = []
	lengths = []
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		if len(t.points) < 41:
			continue
		hursts.append(t.hurst())
		lengths.append(t.len)

	l = np.array([lengths, hursts])

	plt.scatter(l[0],l[1], s=1)
	plt.xlabel('Length (m)')
	plt.ylabel('Mean Hurst exponent')
	plt.show()

	return l

def hurst_area(root, samples=500):
	'''
	Creates plot of hurst exponent vs area covered per unit length by a trajectory

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 60][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(samples)
	df = df.reset_index(drop=True)	
	hursts = []
	areas = []
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		if len(t.points) < 41:
			continue
		hursts.append(t.hurst())
		areas.append(t.areaPerUnitL())

	l = np.array([areas, hursts])

	plt.scatter(l[0],l[1], s=1)
	plt.xlabel('Area Covered per unit length (m)')
	plt.ylabel('Mean Hurst exponent')
	plt.show()

	return l

def angleS_mode(root, samples=500):
	'''
	Creates plot of turning angle per unit length vs mode of transport 

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 20][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(samples)
	df = df.reset_index(drop=True)	
	angles = [[], [], [], [], [], [], [], [], [], [], [], []]
	modes = ['walk', 'run', 'car', 'train','airplane', 'taxi', 'bus', 'subway', 'bike', 'boat', 'motorcycle', 'Unlabelled']
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		i = modes.index(t.mode)
		angles[i].append(t.angleDensS())

	l = np.array([[np.median(x), iqr(x)] for x in angles])

	N = len(modes)
	fig, ax = plt.subplots()

	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	errors = ax.errorbar(ind, l[:,0], yerr=l[:,1], fmt='x', capsize=2)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Turning Angle Density (degrees/metre)')
	ax.set_xticks(ind)
	ax.set_xticklabels(modes)

	plt.xlabel('Mode of Transport')
	plt.show()

	return l

def angleT_mode(root, samples=500):
	'''
	Creates plot of turning angle per unit time vs mode of transport 

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 20][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(samples)
	df = df.reset_index(drop=True)	
	angles = [[], [], [], [], [], [], [], [], [], [], [], []]
	modes = ['walk', 'run', 'car', 'train','airplane', 'taxi', 'bus', 'subway', 'bike', 'boat', 'motorcycle', 'Unlabelled']
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		i = modes.index(t.mode)
		angles[i].append(t.angleDensT())

	l = np.array([[np.median(x), iqr(x)] for x in angles])

	N = len(modes)
	fig, ax = plt.subplots()

	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	errors = ax.errorbar(ind, l[:,0], yerr=l[:,1], fmt='x', capsize=2)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Turning Angle Density (degrees/second)')
	ax.set_xticks(ind)
	ax.set_xticklabels(modes)

	plt.xlabel('Mode of Transport')
	plt.show()

	return l

def areaT_mode(root, samples=500):
	'''
	Creates plot of area covered per unit time vs mode of transport 

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 20][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(samples)
	df = df.reset_index(drop=True)	
	areas = [[], [], [], [], [], [], [], [], [], [], [], []]
	modes = ['walk', 'run', 'car', 'train','airplane', 'taxi', 'bus', 'subway', 'bike', 'boat', 'motorcycle', 'Unlabelled']
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		i = modes.index(t.mode)
		areas[i].append(t.areaPerUnitT())

	l = np.array([[np.median(x), iqr(x)] for x in areas])

	N = len(modes)
	fig, ax = plt.subplots()

	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	errors = ax.errorbar(ind, l[:,0], yerr=l[:,1], fmt='x', capsize=2)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Area covered per unit time (m^2/second)')
	ax.set_xticks(ind)
	ax.set_xticklabels(modes)

	plt.xlabel('Mode of Transport')
	plt.show()

	return l

def area_time(root, samples=500):
	'''
	Creates plot of hurst exponent vs area covered per unit length by a trajectory

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 60][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(samples)
	df = df.reset_index(drop=True)	
	areas = []
	times = []
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		if len(t.points) < 41:
			continue
		times.append(t.time*60)
		areas.append(t.coveredArea())

	l = np.array([times, areas])

	slope, intercept, r_value, p_value, std_err = stats.linregress(l[0], l[1])

	plt.scatter(l[0],l[1], s=1)
	plt.plot(l[0], intercept + slope*l[0], 'r', label='fitted line')
	plt.xlabel('Duration (s)')
	plt.ylabel('Area Covered (m^2)')
	plt.show()

	return l



