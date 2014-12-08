from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure import FullConnection
import pandas
from collections import deque
from pandas.stats.moments import rolling_mean
from itertools import chain
from nyc import *
from avr import avr_p

import matplotlib.pyplot as plt

frame, weather = load_full()

# frame = pandas.DataFrame(fnl)
# frame = (frame - frame.mean()) / (frame.max() - frame.min())
frame = (frame - frame.mean()) / (frame.var())
for k in [30]:
  for i in frame.columns:

    fnn = buildNetwork(2*k,10,1)

    DS = SupervisedDataSet(2*k, 1)
    dta = frame[i][:int(len(frame)*0.8)]
    wea = weather[0][:int(len(frame)*0.8)]

    for j in xrange(0, len(dta) - (k+1)):
      DS.appendLinked(list(chain(*zip(dta[j:j+k], wea[j:j+k]))), [dta[j+k+1]])

    test = frame[i][int(len(frame)*0.8):]
    wea = weather[0][int(len(frame)*0.8):]
    testDS = SupervisedDataSet(2*k, 1)

    for j in xrange(0, len(test) - (k+1)):
      testDS.appendLinked(list(chain(*zip(test[j:j+k], wea[j:j+k]))), [test[5000+i+k+1]])

    trainer = BackpropTrainer(fnn, dataset=DS, momentum=0.1, verbose=True, weightdecay=0.01)

    # for ep in range(0, 5):
      # trainer.trainEpochs()

    trainer.trainUntilConvergence(maxEpochs=20)

    res = fnn.activateOnDataset(testDS)

    res = pandas.DataFrame(res)
    res.index = test[k+1:].index
    plt.figure()
    ax = rolling_mean(test,30).plot()
    rolling_mean(res,30).plot(ax=ax, style='r--')
    print 'AVR SCORE %f trace %d k %d' % (avr_p(test[k+1:],res), i, k)
# ax = rolling_mean(frame.iloc[5000:],30).plot()  
plt.show()
