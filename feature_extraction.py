import csv
import operator
from datetime import datetime
from sklearn import cross_validation
import pickle

start = datetime(1995, 1, 1)
s = int(start.strftime("%s")) / 86400

def convDate(date):
	t = int(date.strftime("%s")) / 86400
	return t - s

# find all accidents involving a bycicle 
with open('/Users/samijaber/Downloads/CrashStat3_Ped-Bike-Crashes_1995-2009_20111020.csv/TBL_VEHICLE_1995_2009_20111020.csv', 'rb') as csvfile1:
	rdr1 = csv.reader(csvfile1)
	bike_crsh = []
	for row in rdr1:
		if row[5] == "35":
			bike_crsh += [row[1]]

with open('/Users/samijaber/Downloads/CrashStat3_Ped-Bike-Crashes_1995-2009_20111020.csv/TBL_CRASHES_1995_2009_20111020.csv', 'rb') as csvfile2:
	rdr2 = csv.reader(csvfile2)

	# BRONX, KINGS, NEW YORK, QUEENS, RICHMOND
	crsh_by_day = {}

	for row in rdr2:
		if row[1] in bike_crsh:
			month, day, year = row[4].split('/')
			date = datetime(int(year), int(month), int(day))
			day_num = convDate(date)

			if not day_num in crsh_by_day:
				crsh_by_day[day_num] = [0, 0, 0, 0, 0]

			x = crsh_by_day[day_num]
			idx = int(row[25][1]) - 1
			x[idx] += 1

# last day
end = datetime(2009, 12, 31)
e = convDate(end)
all_days = range(1, e)

# add missing days
for day in all_days:
	if not day in crsh_by_day:
		crsh_by_day[day] = [0, 0, 0, 0, 0]


# sort by day
srtd = sorted(crsh_by_day.items(), key=operator.itemgetter(0))
fnl = [s[1] for s in srtd]

train_set = fnl[:4930]
test_set = fnl[4930:]
