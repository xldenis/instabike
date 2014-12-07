import csv
import operator
from datetime import datetime
import pickle
import pylab as pl
import numpy as np
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter


###############################################################################
# Grab pickled data 
###############################################################################


srtd = pickle.load(open('sorted.pk', 'rb'))
fnl = pickle.load(open('fnl.pk', 'rb'))

start = datetime(1995, 1, 1)
s = int(start.strftime("%s")) / 86400

def convDate(date):
	t = int(date.strftime("%s")) / 86400
	return t - s

end = datetime(2009, 12, 31)
e = convDate(end)
all_days = range(1, e)


train_set = fnl[:4930]
test_set = fnl[4930:]
