# Contains useful maths functions

import numpy as np
from scipy.spatial.distance import cdist, euclidean
import math
import warnings

def geometric_median(X, eps=1e-5):
    '''
    From - http://bit.ly/2uoh7Ut
    
    '''

    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


# def geometric_median(X, numIter = 200):
#     """
#     [Same source, different answer - http://bit.ly/2uoh7Ut]

#     Compute the geometric median of a point sample.
#     The geometric median coordinates will be expressed in the Spatial Image reference system (not in real world metrics).
#     We use the Weiszfeld's algorithm (http://en.wikipedia.org/wiki/Geometric_median)

#     :Parameters:
#      - `X` (list|np.array) - voxels coordinate (3xN matrix)
#      - `numIter` (int) - limit the length of the search for global optimum

#     :Return:
#      - np.array((x,y,z)): geometric median of the coordinates;
#     """
#     # -- Initialising 'median' to the centroid
#     y = np.mean(X,1)
#     # -- If the init point is in the set of points, we shift it:
#     while (y[0] in X[0]) and (y[1] in X[1]) and (y[2] in X[2]):
#         y+=0.1

#     convergence=False # boolean testing the convergence toward a global optimum
#     dist=[] # list recording the distance evolution

#     # -- Minimizing the sum of the squares of the distances between each points in 'X' and the median.
#     i=0
#     while ( (not convergence) and (i < numIter) ):
#         num_x, num_y, num_z = 0.0, 0.0, 0.0
#         denum = 0.0
#         m = X.shape[1]
#         d = 0
#         for j in range(0,m):
#             div = math.sqrt( (X[0,j]-y[0])**2 + (X[1,j]-y[1])**2 + (X[2,j]-y[2])**2 )
#             num_x += X[0,j] / div
#             num_y += X[1,j] / div
#             num_z += X[2,j] / div
#             denum += 1./div
#             d += div**2 # distance (to the median) to miminize
#         dist.append(d) # update of the distance evolution

#         if denum == 0.:
#             warnings.warn( "Couldn't compute a geometric median, please check your data!" )
#             return [0,0,0]

#         y = [num_x/denum, num_y/denum, num_z/denum] # update to the new value of the median
#         if i > 3:
#             convergence=(abs(dist[i]-dist[i-2])<0.1) # we test the convergence over three steps for stability
#             #~ print abs(dist[i]-dist[i-2]), convergence
#         i += 1
#     if i == numIter:
#         raise ValueError( "The Weiszfeld's algoritm did not converged after"+str(numIter)+"iterations !!!!!!!!!" )
#     # -- When convergence or iterations limit is reached we assume that we found the median.

#     return np.array(y)