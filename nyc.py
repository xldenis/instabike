import csv
import operator
from datetime import datetime
from sklearn import cross_validation

start = datetime(1995, 1, 1)
s = int(start.strftime("%s")) / 86400

def convDate(date):
	t = int(date.strftime("%s")) / 86400
	return t - s

# find all accidents involving a bycicle 
with open('crashstatdata/TBL_VEHICLE_1995_2009_20111020.csv', 'rb') as csvfile1:
	rdr1 = csv.reader(csvfile1)
	bike_crsh = []
	for row in rdr1:
		if row[5] == "35":
			bike_crsh += [row[1]]

with open('crashstatdata/TBL_CRASHES_1995_2009_20111020.csv', 'rb') as csvfile2:
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

x_train = train_set[:-1]
y_train = train_set[1:]
x_test = test_set[:-1]
y_test = test_set[1:]


# HMMMLearn
####################################################################################
####################################################################################
####################################################################################

import numpy as np
from hmmlearn.hmm import GaussianHMM

new_x = np.asarray(x_train)

n_comps = 6
model = GaussianHMM(n_comps)
model.fit([new_x])
hidden_states = model.predict(new_x)


###############################################################################
# print trained parameters and plot

import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

print("Transition matrix")
print(model.transmat_)
print()

print("means and vars of each hidden state")
for i in range(n_comps):
    print("%dth hidden state" % i)
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')
fig = pl.figure()
ax = fig.add_subplot(111)

ald = np.asarray(all_days)
summed_fnl = [sum(x) for x in fnl]
smdf = np.asarray(summed_fnl)

for i in range(n_comps):
    # use fancy indexing to plot data in each state
    idx = (hidden_states == i)
	# print(idx, all_days[idx], summed_fnl[idx], sep=',')
    ax.plot_date(ald[idx], smdf[idx], 'o', label="%dth hidden state" % i)
ax.legend()

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()

# format the coords message box
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.fmt_ydata = lambda x: '$%1.2f' % x
ax.grid(True)

fig.autofmt_xdate()
pl.show()
