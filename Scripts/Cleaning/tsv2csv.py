# Goes into the separate directories in the root directory and converts the labels.txt files into labels.csv files

import csv, os, sys

def convertToCSV(root, direc):

	# read tab-delimited file
	with open(root + '/' + direc + '/' + 'labels.txt','rt') as fin:
		cr = csv.reader(fin, delimiter='\t')
		filecontents = [line for line in cr]

	# write comma-delimited file (comma is the default delimiter)
	with open(root + '/' + direc + '/' + 'labels.csv','wt') as fou:
		cw = csv.writer(fou, quotechar='', quoting=csv.QUOTE_NONE)
		cw.writerows(filecontents)

	os.remove(root + '/' + direc + '/' + 'labels.txt')

thedir = sys.argv[1]

dirs = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]

for direc in dirs:
	if 'labels.txt' in os.listdir(thedir + '/' + direc):
		convertToCSV(thedir, direc)