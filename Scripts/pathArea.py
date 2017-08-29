import shapely.geometry as shlgeo # to use in finding buffer
import shapely.ops as shlops # to use in finding buffer

def covered_area(COOR,radius,RES):
    circ=[]

    for i in xrange(0,len(COOR)):
        circ.append(shlgeo.Point(COOR[i,0],COOR[i,1]).buffer(radius,RES))
        # 20 is the resolution, default is 16, 20 is good enough for coverage problem
    return shlops.unary_union(circ).area