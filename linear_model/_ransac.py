import numpy as np
import math
from numpy.random import default_rng
rng = default_rng()

from ._linreg import LinearRegression
from ._metric import square_error_loss, mean_absolute_error, mean_square_error, absolute_error_loss

"""
RANSAC
--------------------------
digunakan untuk mengestimasi 
parameter dari sebuah model dengan 
menggunakan sampel data acak yang memiliki outlier.
--------------------------
Input :
- n : jumlah data point minimal dalam satu sampel data
- t : batas data
"""
class RANSAC:
  def __init__(self, n=3, t=0.1, z=0.95, loss='square_error', metric='l2'):
    self.n = n
    self.t = t
    self.d = 0
    self.loss = loss
    self.metric = metric
    
    self.best_fit = None
    self.best_error = np.inf

    self.k = math.inf
    self.k_done = 0

    self.z = z
    self.prob_outlier = 1 - self.z

  def fit(self, x, y):
    x = np.array(x).copy()
    y = np.array(y).copy()

    while self.k > self.k_done:
      ids = rng.permutation(x.shape[0])
      initial_inliers = ids[: self.n]

      initial_model = LinearRegression()
      initial_model.fit(x[initial_inliers], y[initial_inliers])

      if self.loss == 'square_error':
        thresholded = (square_error_loss(y[ids][self.n :], initial_model.predict(x[ids][self.n :])) < self.t)
      else:
        thresholded = (absolute_error_loss(y[ids][self.n :], initial_model.predict(x[ids][self.n :])) < self.t)
      
      inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

      if len(inlier_ids) == 0:
        print("No inliers found")
      else:
        if inlier_ids.size > self.d:
          self.d = inlier_ids.size
          inlier_points = np.hstack([initial_inliers, inlier_ids])
          
          better_model = LinearRegression()
          better_model.fit(x[inlier_points], y[inlier_points])

          if self.metric == 'l2':
            this_error = mean_square_error(y[inlier_points], better_model.predict(x[inlier_points]))
          else:
            this_error = mean_absolute_error(y[inlier_points], better_model.predict(x[inlier_points]))

          if this_error < self.best_error:
            self.best_error = this_error
            self.best_fit = better_model
        
        w = inlier_ids.size/len(x)
        prob_outlier = 1 - w
        b = (1 - prob_outlier)**self.n

        if math.log(1 - b) == 0.0:
          break
        else:
          self.k = math.log(1 - self.z)/math.log(1 - b)
          self.k_done += 1

        print('# iteration done:', self.k_done)
        print('# n:', self.k)
        print('# max_inlier_count: ', self.d)

  def predict(self, x):
    if self.best_fit == None:
      print("No best model's parameter found to predict")
    else:
      return self.best_fit.predict(x)