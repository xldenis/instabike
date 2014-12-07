import csv
import operator
from datetime import datetime
import pickle
import pylab as pl
import numpy as np
from hmmlearn.hmm import GaussianHMM
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter


###############################################################################
# Grab pickled data 
###############################################################################


srtd = pickle.load(open('/Users/samijaber/sorted.pk', 'rb'))
fnl = pickle.load(open('/Users/samijaber/fnl.pk', 'rb'))

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


###############################################################################
# print trained parameters and plot
###############################################################################

new_x = np.asarray(train_set)

n_comps = 6
model = GaussianHMM(n_comps)
model.fit([new_x])
hidden_states = model.predict(new_x)

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
summed_fnl = [x[3] for x in fnl]
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


############################################################################################
# Prediction step
############################################################################################

new_x = np.asarray(train_set)

n_comps = 6
model = GaussianHMM(n_comps)
model.fit([new_x])
hidden_states = model.predict(new_x)


new_test = np.asarray(test_set)

predictions = []

chunk = train_set[2500:]
'''find prob for each test point, compare to expected, then re-fit HMM with it'''
for idx, x in enumerate(chunk):
	_, pst_prob = model.score_samples([x])
	max_ind = pst_prob.argmax()
	trn = model._get_transmat()[max_ind]

	'''Get the max one for now. Maybe use some other method later one'''
	max_trn = trn.argmax()

	cov = model._get_covars()[max_trn]
	mns = model._get_means()[max_trn]
	rd = np.random.multivariate_normal(mns, cov)

	int_rd = [int(x) for x in rd]
	predictions += [int_rd]

	# retrain HMM with new data point
	moving_idx = 30-idx
	mov_train_set = []
	if moving_idx < 1:
		mov_train_set = []
	else: mov_train_set = train_set[-moving_idx:]

	mov_new_set_idx = 0
	if idx >= 30:
		mov_new_set_idx = idx-30

	new_set = mov_train_set + test_set[mov_new_set_idx:idx]
	new_x_train = np.asarray(new_set)
	print idx, int_rd, len(new_x_train)
	
	n_comps = 6
	model = GaussianHMM(n_comps)
	model.fit([new_x_train])
	hidden_states = model.predict(new_x_train)


def mov_avg(data, k):
	avgs = []
	for i in range(0, len(data)):
		subset = data[i:i+k]
		avg = float(sum(subset))/len(subset)
		avgs += [avg]
	return avgs


all_days2 = range(5478, 5478+len(predictions))

fnl_1        = [x[0] for x in fnl]
pred_1       = [x[0] for x in predictions]
y1           = [x[0] for x in test_set]

summed_train = [sum(x) for x in fnl]
summed_pred  = [sum(x) for x in predictions]
summed_test  = [sum(x) for x in test_set]


train_avg    = mov_avg(summed_train + summed_test, 10)
pred_avg     = mov_avg(summed_pred, 10)
test_avg     = mov_avg(summed_test, 10)

plt.plot(train_avg, marker='.', linestyle='None', color='b')
plt.plot(all_days[2500:-547], pred_avg, marker='.', linestyle='None', color='g')

# plt.plot(all_days2, pred_avg, marker='.', linestyle='None', color='g')
# plt.plot(all_days2, test_avg, marker='.', linestyle='None', color='r')

# pred_diff                  = [x - y for (x,y) in zip(summed_pred, summed_test)]
# plt.plot(pred_diff, marker = '.', linestyle                                     = 'None', color = 'b')

# plt.plot(fnl, marker='.', linestyle='None', color='b')
# plt.plot(all_days2, predictions, marker='.', linestyle='None', color='g')
# plt.plot(all_days2, test_set, marker='.', linestyle='None', color='r')

plt.show()