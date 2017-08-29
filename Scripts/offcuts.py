# A file containing offcuts of code that became obsolete (for safe-keeping) 

# ~~~~~~~~~~~~~~~~~~~~ 1D FRACTAL ROUTINES ~~~~~~~~~~~~~~~~~

def boxCount1D(self, x, y, length):
	'''
	Returns the number of boxes of a certain length required to cover the trajectory

	'''
	xedges = np.arange(x.min()-length, x.max() + length, length)
	yedges = np.arange(y.min()-length, y.max() + length, length)

	H, xedges, yedges = np.histogram2d(x, y, bins=[xedges, yedges])

	return np.sum(H>0)

def fractalDim1D(self):
	'''
	Returns a plot to find the fractal dimension of the trajectory using the boxCount() method

	'''
	x = self.points[:,0]
	y = self.points[:,1]

	Ns = []

	start = min(self.nnDist())
	end = max((self.nnDist()))

	scales=np.logspace(start/2, end/10, num=20, endpoint=False, base=2)

	for scale in scales:
		Ns.append(self.boxCount1D(x,y,scale))

	coeffs = np.polyfit(np.log(1/scales), np.log(Ns), 1)

	plt.plot(np.log(1/scales),np.log(Ns), 'o', mfc='none')
	plt.plot(np.log(1/scales), np.polyval(coeffs,np.log(1/scales)))
	plt.xlabel('log 1/$\epsilon$')
	plt.ylabel('log N')
	plt.title('Fractal Dimension: ' + str(coeffs[0]))
	plt.show()

# ~~~~~~~~~~~~~~~~~~~~ 2D FRACTAL ROUTINES ~~~~~~~~~~~~~~~~~

def boxCount(self, length):
	'''
	Returns the number of boxes required to cover the 2D line created by the trajectory

	'''
	line = LineString(self.points)
	bounds = line.bounds
	boxnx = ceil((bounds[2] - bounds[0])/length)
	boxny = ceil((bounds[3] - bounds[1])/length)
	boxes = []
	count = 0

	for ix in range(boxnx):
		for iy in range(boxny):
			b = box(bounds[0]+ix*length,bounds[1]+iy*length,bounds[2]+(ix+1)*length,bounds[3]+(iy+1)*length)
			if line.intersects(b):
				count += 1

	return count

def fractalDim(self):
	'''
	Returns a plot to find the fractal dimension of the trajectory using the boxCount() method

	'''

	Ns = []

	ds = self.nnDist()
	ds = [d for d in ds if d != 0]
	# start = np.percentile(ds, 10)
	start = np.mean(ds)
	end = np.percentile(ds, 90)

	start = 2
	end = 5

	scales=np.geomspace(start/np.sqrt(2), end/np.sqrt(2), num=8, endpoint=False)

	for scale in scales:
		print('Scale:', scale)
		Ns.append(self.boxCount(scale))

	coeffs = np.polyfit(np.log(1/scales), np.log(Ns), 1)

	plt.plot(np.log(1/scales),np.log(Ns), 'o', mfc='none')
	plt.plot(np.log(1/scales), np.polyval(coeffs,np.log(1/scales)))
	plt.xlabel('log 1/$\epsilon$')
	plt.ylabel('log N')
	plt.title('Fractal Dimension: ' + str(coeffs[0]))
	plt.show()