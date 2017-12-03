# Modified from: https://github.com/eubr-bigsea/py-st-dbscan

from datetime import timedelta
import numpy as np

def st_dbscan(df, spatial_threshold, temporal_threshold, min_neighbors, min_time):
    """
    Python st-dbscan implementation.
    INPUTS:
        df={o1,o2,...,on} Set of objects
        spatial_threshold = Maximum geographical coordinate (spatial) distance
        value
        temporal_threshold = Maximum non-spatial distance value
        min_neighbors = Minimun number of points within Eps1 and Eps2 distance
        min_time = is the minimum time the user must stay in the same place for the cluster to be considered a cluster
    OUTPUT:
        C = {c1,c2,...,ck} Set of clusters
    """
    cluster_label = 0
    noise = -1
    unmarked = 777777
    stack = []

    # initialize each point with unmarked
    df['cluster'] = unmarked

    # for each point in database
    for index, point in df.iterrows():
        if df.loc[index]['cluster'] == unmarked:
            neighborhood = retrieve_neighbors(index, df, spatial_threshold,
                                              temporal_threshold)

            if len(neighborhood) < min_neighbors:
                df.set_value(index, 'cluster', noise)
            else:  # found a core point
                cluster_label += 1
                # assign a label to core point
                df.set_value(index, 'cluster', cluster_label)

                # assign core's label to its neighborhood
                for neig_index in neighborhood:
                    df.set_value(neig_index, 'cluster', cluster_label)
                    stack.append(neig_index)  # append neighborhood to stack

                # find new neighbors from core point neighborhood
                while len(stack) > 0:
                    current_point_index = stack.pop()
                    new_neighborhood = retrieve_neighbors(
                        current_point_index, df, spatial_threshold,
                        temporal_threshold)

                    # current_point is a new core
                    if len(new_neighborhood) >= min_neighbors:
                        for neig_index in new_neighborhood:
                            neig_cluster = df.loc[neig_index]['cluster']
                            if all([neig_cluster != noise,
                                    neig_cluster == unmarked]):
                                # TODO: verify cluster average
                                # before add new point
                                df.set_value(neig_index, 'cluster',
                                             cluster_label)
                                stack.append(neig_index)

    clusters = df['cluster'].unique()

    # Need to check whether this works
    for cluster in clusters:
        cdf = df[df['cluster'] == cluster]
        timeSpent = (cdf['datetime'].iloc[-1] - cdf['datetime'].iloc[0]).seconds
        if timeSpent < min_time:
            df.loc[df['cluster'] == cluster, 'cluster'] = -1

    return df


def retrieve_neighbors(index_center, df, spatial_threshold, temporal_threshold):
    neigborhood = []

    center_point = df.loc[index_center]
    start = np.array([center_point['x'],center_point['y']])

    # filter by time 
    min_time = center_point['datetime'] - timedelta(minutes=temporal_threshold)
    max_time = center_point['datetime'] + timedelta(minutes=temporal_threshold)
    df = df[(df['datetime'] >= min_time) & (df['datetime'] <= max_time)]

    points = np.array([df['x'], df['y']]).T
    # centres filled with the same centre point 
    centres = np.zeros((len(points),2)) + start
    disps = points - centres
    distances = np.sqrt(disps[:,0]**2 + disps[:,1]**2)
    isNeighbor = distances <= spatial_threshold
    neigborhood = np.where(isNeighbor)[0]

    # # filter by distance - TAG - could vectorise this I think
    # for index, point in df.iterrows():
    #     if index != index_center:
    #         end = np.array([point['x'],point['y']])
    #         disp = end - start
    #         distance = np.sqrt(np.dot(disp, disp))
    #         if distance <= spatial_threshold:
    #             neigborhood.append(index)

    return neigborhood
