# Beijing-Trajectories-Project
Analysis of the GeoLife GPS Trajectories Dataset (which can be found here: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)

Scripts contained in the 'Cleaning' directory are to convert the raw data files into a useable format (e.g. converting lat/long to cartesian metres, splitting up trajectories by their labels).

The most important scripts for analysis:

**userAnal.py** - contains a class which represents a single user. Has methods for analysis of that user's trajectories

**trajAnal.py** - contains a class which represents a single trajectory. Has methods for analysis of that trajectory

**generalAnal.py** - contains functions for creating plots of different quantities using the user and trajectory classes

The Jupyter notebook contains a few justifications for initial decisions such as deciding the position of the origin and the sampling rate.
