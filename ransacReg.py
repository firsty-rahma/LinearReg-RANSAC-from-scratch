import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.random import default_rng
rng = default_rng()

from sklearn.model_selection import train_test_split

from linear_model._ransac import RANSAC

def read_dataset(path, sep, id = None):
  df = pd.read_csv(path, sep=sep)

  if id == None:
    return df
  else:
    df_with_id = df.set_index(id)
    return df_with_id

df = read_dataset('data/car_data.csv', ',', 'id')

x = df[['car_engine_capacity']]
y = df['car_engine_hp']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)

y_train = np.round((y_train + rng.normal(np.mean(y_train)/100, np.std(y_train)/100, size=y_train.shape[0])))
y_test = np.round((y_test + rng.normal(np.mean(y_test)/100, np.std(y_test)/100, size=y_test.shape[0])))

rnsc1 = RANSAC(n=3, t=0.05, z=0.95)
rnsc1.fit(x_train, y_train)

y_pred_rnsc1 = rnsc1.predict(x_test)

plt.scatter(x_test, y_test, c="grey")

plt.plot(x_test, y_pred_rnsc1, c="peru")
plt.show()

