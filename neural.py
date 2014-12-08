from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure import FullConnection
import pandas
from collections import deque
from pandas.stats.moments import rolling_mean

from nyc import *
from avr import avr_p

import matplotlib.pyplot as plt

frame = pandas.DataFrame(fnl)
# frame = (frame - frame.mean()) / (frame.max() - frame.min())
frame = (frame - frame.mean()) / (frame.var())
for k in [1, 2, 7, 10, 30]:
  for i in frame.columns:

    fnn = buildNetwork(k,10,1)

    DS = SupervisedDataSet(k, 1)
    dta = frame[i][:5000]
    for j in xrange(0, len(dta) - (k+1)):
      DS.appendLinked(dta[j:j+k], [dta[j+k+1]])

    test = frame[i][5000:]
    testDS = SupervisedDataSet(k, 1)
    for j in xrange(0, len(test) - (k+1)):
      testDS.appendLinked(test[j:j+k], [test[5000+i+k+1]])

    trainer = BackpropTrainer(fnn, dataset=DS, momentum=0.1, verbose=False, weightdecay=0.01)

    for ep in range(0, 5):
      trainer.trainEpochs(5)

      # trainer.trainUntilConvergence()

      res = fnn.activateOnDataset(testDS)

      res = pandas.DataFrame(res)
      res.index = test[k+1:].index
      # plt.figure()
      # ax = rolling_mean(test,30).plot()
      # rolling_mean(res,30).plot(ax=ax, style='r--')
      print 'AVR SCORE %f trace %d k %d epo %d' % (avr_p(test[k+1:],res), i, k, 5*ep)
# ax = rolling_mean(frame.iloc[5000:],30).plot()  
# plt.show()
