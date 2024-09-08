import numpy as np

class LinearRegression :
  def __init__(self, learning_rate = 0.2, iterations = 1000, tol=1e-4):
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.tol = tol

  def fit(self, x, y):
    # Prepare x (input) and y (target) data
    x = np.array(x).copy()
    y = np.array(y).copy()

    self.n_samples, self.n_feature = x.shape
    
    self.coef_ = np.zeros(1)
    self.intercept_ = 0
    self.cost = 0

    for i in range(self.iterations):
      y_pred = np.dot(x, self.coef_) + self.intercept_

      grad_coef_ = (y - y_pred).dot(x) / self.n_samples
      grad_intercept_ = (y - y_pred).dot(np.ones(self.n_samples)) / self.n_samples
      
      # Update parameters
      self.coef_ += np.round((self.learning_rate * grad_coef_),20)
      self.intercept_ += np.round(self.learning_rate * grad_intercept_,20)

      cost = np.round((1/2) * sum([val ** 2 for val in (y - y_pred)]), 30)

      # Break the iteration
      grad_stack_ = np.hstack((grad_coef_, grad_intercept_))


      if all(np.abs(grad_stack_) < self.tol):
        break
      #print("coefs {}, intercept {}, iteration {}, cost {}".format(self.coef_, self.intercept_, i, cost ))

  def predict(self, x):
    y_pred = np.dot(x, self.coef_) + self.intercept_
    return y_pred