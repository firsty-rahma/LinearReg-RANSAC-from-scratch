import numpy as np

def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def absolute_error_loss(y_true, y_pred):
  return np.abs(y_true - y_pred)

def mean_square_error(y_true, y_pred):
  return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]

def mean_absolute_error(y_true, y_pred):
  return np.sum(absolute_error_loss(y_true, y_pred)) / y_true.shape[0]