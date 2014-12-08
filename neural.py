from pybrain.structure import LinearLayer, SigmoidLayer, FeedForwardNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure import FullConnection
import pandas
from collections import deque
from pandas.stats.moments import rolling_mean

from nyc import *

import matplotlib.pyplot as plt

frame = pandas.DataFrame(fnl)
frame = (frame - frame.mean()) / (frame.max() - frame.min())
k = 30

for i in [0,1,2]:

  fnn = FeedForwardNetwork()
  inLayer = LinearLayer(k)
  hiddenLayer = SigmoidLayer(10)
  outLayer = LinearLayer(1)

  fnn.addInputModule(inLayer)
  fnn.addModule(hiddenLayer)
  fnn.addOutputModule(outLayer)

  in_to_hidden = FullConnection(inLayer, hiddenLayer)
  hidden_to_out = FullConnection(hiddenLayer, outLayer)

  fnn.addConnection(in_to_hidden)
  fnn.addConnection(hidden_to_out)

  fnn.sortModules()

  DS = SupervisedDataSet(k, 1)
  dta = frame[i][:5000]
  for j in xrange(0, len(dta) - (k+1)):
    DS.appendLinked(dta[j:j+k], [dta[j+k+1]])

  test = frame[i][5000:]
  testDS = SupervisedDataSet(k, 1)
  for j in xrange(0, len(test) - (k+1)):
    testDS.appendLinked(test[j:j+k], [test[5000+i+k+1]])

  trainer = BackpropTrainer(fnn, dataset=DS, momentum=0.1, verbose=True, weightdecay=0.01)

  # trainer.trainEpochs(15)

  trainer.trainUntilConvergence(maxEpochs=5)

  res = fnn.activateOnDataset(testDS)

  res = pandas.DataFrame(res)
  res.index = test[k+1:].index
  plt.figure()
  ax = rolling_mean(test,30).plot()
  rolling_mean(res,30).plot(ax=ax, style='r--')
ax = rolling_mean(frame.iloc[5000:],30).plot()  
plt.show()
