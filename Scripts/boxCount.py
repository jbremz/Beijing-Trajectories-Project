def boxCount(self, length):
	'''
	Returns the number of boxes required to cover the 2D line created by the trajectory

	'''
	line = LineString(self.points)
	bounds = line.bounds
	boxnx = ceil((bounds[2] - bounds[0])/2.)
	boxny = ceil((bounds[3] - bounds[1])/2.)
	boxes = []
	count = 0

	# It's these nested for loops creating the boxes and checking for an intersection that are obviously slowing things down
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
	start = min(ds)
	end = max(ds)

	# I wasn't sure in what range I should scale the box size to find the fractal dimension, so I scaled it as the side length of a square whose diagonal went from the smallest nearest neigbour distance to the largest n-n distance.
	# I could probably save a lot of computational time if I determined where the scaling region was before, but I'm not sure quite how to do this.
	scales=np.logspace(start/np.sqrt(2), end/np.sqrt(2), num=10, endpoint=False, base=2)

	for scale in scales:
		Ns.append(self.boxCount(scale))

	coeffs = np.polyfit(np.log(1/scales), np.log(Ns), 1)

	plt.plot(np.log(1/scales),np.log(Ns), 'o', mfc='none')
	plt.plot(np.log(1/scales), np.polyval(coeffs,np.log(1/scales)))
	plt.xlabel('log 1/$\epsilon$')
	plt.ylabel('log N')
	plt.title('Fractal Dimension: ' + str(coeffs[0]))
	plt.show()
