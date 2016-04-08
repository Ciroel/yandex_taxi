import os,sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
from sklearn.externals import joblib
import seaborn as sns

from datanight import download_datasets

# <editor-fold desc="Reading data">
train_raw = pd.DataFrame.from_csv("./train.csv")
test = pd.read_csv('test.csv')
train_raw = train_raw[::1].reset_index(drop=True)
target = train_raw['burned'].values

x_col = ['due', 'dist','lat','lon']
train = train_raw[x_col]

# Забыл сказать - был добавлен признак часов со сдвинутыми нулями

def cycle_it(data_raw, min, max, n, name):
    if n == 1: return data_raw
    data = np.array([data_raw]*n).T
    names = np.zeros(n).astype('object'); names[0] = name
    for k in range(1,n):
        bound = float(min )+ (float(max)-float(min))*k/n
        # data[data_raw > bound,k]+=max# + data[data_raw > bound,k]
        data[data_raw <= bound,k]+=max
        names[k] = name + '_' + str(round(bound,2))
    return pd.DataFrame(data, columns=names)

holidays_list = "1.01,2.01,3.01,4.01,5.01,6.01,7.01,8.01,"\
"23.02,8.03,9.03,10.03,1.05,2.05,3.05,4.05,9.05,10.05,11.05,12.06,13.06".split(',')
holidays_list = map(lambda s: tuple(map(int,s.split('.'))),holidays_list)

from dateutil import parser

# data = train

def preprocess_data(data):
    datetimes = list(data.due.apply(lambda x: parser.parse(x)))
    print 'One'
    rel_times = map(lambda dt: (dt.hour*60 + dt.minute)/(24.*60), datetimes)

    rel_times = cycle_it(np.array(rel_times),0.,1.,10,'time')

    is_holiday_1 = map(lambda dt: (dt.day,dt.month) in holidays_list,# or dt.isoweekday()>=6,
                 datetimes)
    is_holiday_2 = map(lambda dt: (dt.day, dt.month) in holidays_list or dt.isoweekday()>=6,
                       datetimes)
    is_holiday_3 = map(lambda dt: (dt.day, dt.month) in holidays_list or dt.isoweekday() >= 5,
                       datetimes)
    holidays = pd.DataFrame(np.vstack([is_holiday_1,is_holiday_2,is_holiday_3]).T, columns=['hol_1','hol_2','hol_3'])

    data_new = pd.concat([rel_times,holidays,data[["dist","lat","lon"]]],axis=1)

    print 'Done'
    return data_new

train_new = preprocess_data(train)
test_new = preprocess_data(test)

# Для сохранения
joblib.dump(train_new, 'train_new_geo_0')
joblib.dump(train_new, 'test_new_geo_0')

# Для загрузки
# train_new = joblib.load('train_new_geo_0')
# test_new = joblib.load('test_new_geo_0')

bound = int(train_new.shape[0]*0.9)
x_train, x_val = train_new[:bound], train_new[bound:]
y_train, y_val = target[:bound], target[bound:]

dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val, y_val)
dtest = xgb.DMatrix(test_new)
watchlist = [(dval,'eval')]

params = {"booster": "gbtree",
          "objective": "binary:logistic",
          "eta": 0.1,
          "max_depth": 7,
          "subsample": 0.95,
          "colsample_bytree": 0.95,
          "silent": 1,
          "seed": 0,
          "eval_metric": "auc",
          }
num_trees = 5000
esr = 12
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, verbose_eval=True, early_stopping_rounds = esr)

# Смотрим важности
importance = gbm.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
featp = df.iloc[-50:,].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))

prediction = gbm.predict(dtest, ntree_limit=gbm.best_ntree_limit)

response = pd.DataFrame()
response["Ids"] = np.arange(test.shape[0])
response["Y_prob"] = prediction

response.to_csv('Subs/sub_geo_2.csv',index=None)