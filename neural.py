from pybrain.structure import LinearLayer, SigmoidLayer, FeedForwardNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure import FullConnection
import pandas
from collections import deque

import nyc

fnn = FeedForwardNetwork()
k = 30
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
dta = pandas.DataFrame(fnl)[0][:5000]
for i in xrange(0, len(dta) - (k+1)):
  DS.appendLinked(dta[i:i+k], [dta[i+k+1]])

test = pandas.DataFrame(fnl)[0][5000:]
testDS = SupervisedDataSet(k, 1)

for i in xrange(0, len(test) - (k+1)):
  testDS.appendLinked(test[i:i+k], [test[5000+i+k+1]])

trainer = BackpropTrainer(fnn, dataset=DS, momentum=0.1, verbose=True, weightdecay=0.01)

# trainer.trainEpochs(15)

trainer.trainUntilConvergence(maxEpochs=50)

res = fnn.activateOnDataset(testDS)

res = pandas.DataFrame(res)
res.index = test[k+1:].index

ax = rolling_mean(test,30).plot()
rolling_mean(res,30).plot(ax=ax, style='r--')
plt.show()
