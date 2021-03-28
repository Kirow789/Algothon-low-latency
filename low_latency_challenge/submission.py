import pandas as pd
import numpy as np
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from sys import stdin

data = list(pd.read_csv('LatencyTraining.csv')['LogReturns'])

# convert time series into training examples
windowsize=500
x = []
y = []
for i in range(len(data)-windowsize):
    x.append(data[i:i+windowsize])
    if data[i + windowsize] > data[i + windowsize - 1]:
        y.append(1)
    else:
        y.append(0)

# train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

train_set = lgbm.Dataset(x_train, label = y_train)
valid_set = lgbm.Dataset(x_test, label = y_test)
param = {'num_leaves': 30, 'max_depth': 5, 'min_data_in_leaf':10, 'objective': 'binary', 'metric': 'auc', 'verbose': 0}
num_round = 30
lgbm_model = lgbm.train(param, train_set, num_round, valid_sets=[valid_set])


for line in stdin:
    if line == '': 
        break
    d=[float(x) for x in line.split(',')]
    print(round(lgbm_model.predict([d])[0]))
