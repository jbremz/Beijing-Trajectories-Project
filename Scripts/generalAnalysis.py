# Different analysis functions for analysis of the trajectories

from Scripts.trajAnalysis import trajectory
import pandas as pd
import time
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import iqr

# An inventory of all the trajectories with their path, label-state, duration and length
idf = pd.read_csv('/Users/JBremner/Documents/Docs/Imperial/Physics /UROP/Beijing Trajectories/Beijing Trajectories Project/Metadata/Inventory.csv')

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
	ax.boxplot(angles)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Turning Angle Density (degrees/metre)')
	ax.set_xticks(ind+1)
	ax.set_xticklabels(modes)

	# Rotate mode labels to a slant
	for label in ax.get_xmajorticklabels():
		label.set_rotation(50)
		label.set_horizontalalignment("right")

	plt.xlabel('Mode of Transport')
	# plt.show()

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
	ax.boxplot(angles)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Turning Angle Density (degrees/second)')
	ax.set_xticks(ind+1)
	ax.set_xticklabels(modes, rotation='vertical')

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
	ax.boxplot(areas)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Area covered per unit time (m^2/second)')
	ax.set_xticks(ind+1)
	ax.set_xticklabels(modes, rotation='vertical')

	plt.xlabel('Mode of Transport')
	plt.show()

	return l

def area_time(root, samples=500):
	'''
	Creates plot of area covered vs duration of a trajectory

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
	# plt.show()

	return l

def area_length(root, samples=500):
	'''
	Creates plot of window area vs length of a trajectory

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 60][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(samples)
	df = df.reset_index(drop=True)	
	areas = []
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
		lengths.append(t.len)
		areas.append(t.windowArea())
		# areas.append(t.coveredArea(radius=500))

	l = np.array([lengths, areas])

	coeffs, cov = np.polyfit(np.log(lengths), np.log(areas), 1, cov=True)
	error = np.sqrt(np.diag(cov))

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	ax.scatter(l[0],l[1], s=1)

	ax.set_yscale('log')
	ax.set_xscale('log')

	x = np.geomspace(np.min(l[0]), np.max(l[0]), 100)
	ax.plot(x, np.exp(coeffs[1])*x**coeffs[0], 'r', linewidth=0.7)

	ax.set_xlabel('Path Length (m)')
	ax.set_ylabel('Window Area ($m^2$)')
	ax.set_title('Exponent: ' + str(coeffs[0]) +'$\pm$'+str(error[0]))
	# plt.show()

	return [coeffs[0], error[0]]
	# return [coeffs, error]

def corrDim_mode(root, samples=500):
	'''
	Creates plot of correlation dimension vs mode of transport 

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 60][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(samples)
	df = df.reset_index(drop=True)	
	dims = [[], [], [], [], [], [], [], [], [], [], [], []]
	modes = ['walk', 'run', 'car', 'train','airplane', 'taxi', 'bus', 'subway', 'bike', 'boat', 'motorcycle', 'Unlabelled']
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		i = modes.index(t.mode)
		dims[i].append(t.corrDim())

	dims =  [[dim for dim in mode if not np.isnan(dim)] for mode in dims]

	l = np.array([[np.median(x), iqr(x)] for x in dims])

	N = len(modes)
	ind = np.arange(N)  # the x locations for the groups

	fig, ax = plt.subplots()
	ax.boxplot(dims)
	# errors = ax.errorbar(ind, l[:,0], yerr=l[:,1], fmt='x', capsize=2)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Correlation dimension')
	ax.set_xticks(ind+1)
	ax.set_xticklabels(modes, rotation='vertical')

	plt.xlabel('Mode of Transport')
	plt.show()

	return dims

def hurst_mode(root, samples=500):
	'''
	Creates plot of hurst exponent vs mode of transport 

	'''
	df = idf[idf['Length'] > 20][idf['Point Count'] >= 60][idf['Duration'] > 0.5][idf['Duration'] < 60].sample(samples)
	df = df.reset_index(drop=True)	
	hursts = [[], [], [], [], [], [], [], [], [], [], [], []]
	modes = ['walk', 'run', 'car', 'train','airplane', 'taxi', 'bus', 'subway', 'bike', 'boat', 'motorcycle', 'Unlabelled']
	bar = progressbar.ProgressBar(max_value=len(df))

	for index, row in df.iterrows():
		bar.update(index)
		t = trajectory(root + '/' + row['Path'])
		t.removeNoise()
		if t.trashy:
			continue
		i = modes.index(t.mode)
		hursts[i].append(t.hurst())

	l = np.array([[np.median(x), iqr(x)] for x in hursts])

	N = len(modes)

	ind = np.arange(N)  # the x locations for the groups

	fig, ax = plt.subplots()
	ax.boxplot(hursts)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Hurst Exponent')
	ax.set_xticks(ind+1)
	ax.set_xticklabels(modes)

	# Rotate mode labels to a slant
	for label in ax.get_xmajorticklabels():
		label.set_rotation(50)
		label.set_horizontalalignment("right")

	plt.xlabel('Mode of Transport')
	plt.show()

	return l


