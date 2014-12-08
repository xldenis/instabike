import csv
import matplotlib.pyplot as plt

def drawMap(crshs_by_ctr, kmeans):
	x, y = zip(*crshs_by_ctr[0])
	plt.plot(x, y, marker='.', linestyle='None', color='b')

	x, y = zip(*crshs_by_ctr[1])
	plt.plot(x, y, marker='.', linestyle='None', color='g')

	x, y = zip(*crshs_by_ctr[2])
	plt.plot(x, y, marker='.', linestyle='None', color='y')

	x, y = zip(*crshs_by_ctr[3])
	plt.plot(x, y, marker='.', linestyle='None', color='k')

	x, y = zip(*crshs_by_ctr[4])
	plt.plot(x, y, marker='.', linestyle='None', color='c')

	x, y = zip(*crshs_by_ctr[5])
	plt.plot(x, y, marker='.', linestyle='None', color='m')

	cnt = kmeans.cluster_centers_
	x, y = zip(*cnt)

	plt.plot(x, y, marker='o', linestyle='None', color='r')
	plt.show()


with open('mtl_data2.csv', 'rb') as csvfile:
	rdr = csv.reader(csvfile, delimiter='|', skipinitialspace=True)
	crshs = []
	ignore = 2
	for row in rdr:
		if ignore is 0:
			crshs += [[row[2]] + row[7:9]]
		else: ignore -= 1

locs = [x[1:] for x in crshs]

# kmeans clustering to find neighborhouds
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6)
kmeans.fit(locs)

cnt = kmeans.cluster_centers_
x, y = zip(*cnt)

cntrs = kmeans.predict(locs)
crshs_by_ctr = {key:[] for key in range(0,6)}

for idx, c in enumerate(cntrs):
	crshs_by_ctr[c] += [locs[idx]]


from datetime import datetime

start = datetime(2006, 1, 13)
s = int(start.strftime("%s")) / 86400

def convDate(date):
	t = int(date.strftime("%s")) / 86400
	return t - s

e = datetime(2010, 12, 24)
end = convDate(e)
crshs_by_day = {key:[0]*6 for key in range(0, end+1)}

dates = [x[0] for x in crshs]

for (dt, c) in zip(dates, cntrs):
	y, m, d = dt.split('-')
	date = datetime(int(y), int(m), int(d))
	day = convDate(date)

	crshs_by_day[day][c] += 1


import operator
# sort by day
srtd = sorted(crshs_by_day.items(), key=operator.itemgetter(0))
fnl = [s[1] for s in srtd]

