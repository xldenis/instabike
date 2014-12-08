import numpy as np

def avr(true_y, predicted_y):
    true_y = np.array(true_y)
    predicted_y = np.array(predicted_y)

    mean = true_y.mean()
    diff = true_y - pred_y
    diff_sq = diff ** 2
    denom = true_y - mean
    denom_sq = denom ** 2

    return diff_sq.sum() / denom_sq.sum()