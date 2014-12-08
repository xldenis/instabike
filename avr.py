import numpy as np

def avr(true_y, predicted_y):
    true_y = np.array(true_y)
    predicted_y = np.array(predicted_y)

    mean = true_y.mean()
    diff = true_y - predicted_y
    diff_sq = diff ** 2
    denom = true_y - mean
    denom_sq = denom ** 2

    return diff_sq.sum() / denom_sq.sum()

def avr_p(true,pred):
  diff = true.rsub(pred.squeeze(),axis=0)
  diff_sq = diff ** 2
  denom = true - true.mean()
  denom_sq = denom ** 2

  return diff_sq.sum() / denom_sq.sum()