
# Import Library
import numpy as np
import pandas as pd


# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# make a log of the target
Y = train.target
Y = np.log(Y+1)
Y.to_csv('train_target.csv',index=False)


# Remove 'ID' and 'target' features
train = train.drop(['ID','target'], axis=1)
test = test.drop('ID', axis=1)
print(train.shape)
print(test.shape)


# Remove Constant Features
constant_feat = train.nunique().reset_index()
constant_feat.columns = ['column', 'count']
constant_feat = constant_feat[constant_feat['count']==1]
print(constant_feat.shape)
print(list(constant_feat.column))

train = train.drop(constant_feat['column'], axis=1)
test = test.drop(constant_feat['column'], axis=1)
print(train.shape)
print(test.shape)

train.to_csv('train_remove_constant.csv', index=False)
test.to_csv('test_remove_constant.csv', index=False)


# Check the percent of zero per column
total = (train == 0).sum().sort_values(ascending=False)
percent = ((train == 0).sum() / (train == 0).count() * 100).sort_values(ascending=False)
feat = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
print(feat.head(50))
print(feat.tail(50))


#
use_feat = feat[feat.Percent < 90]
print(use_feat.shape)

train2 = train[use_feat.index]
test2 = test[use_feat.index]
print(train2.columns)
print(test2.columns)

train2.to_csv('train_use_feat_90.csv', index=False)
test2.to_csv('test_use_feat_90.csv', index=False)


#
use_feat = feat[feat.Percent < 80]
print(use_feat.shape)

train2 = train[use_feat.index]
test2 = test[use_feat.index]
print(train2.columns)
print(test2.columns)

train2.to_csv('train_use_feat_80.csv', index=False)
test2.to_csv('test_use_feat_80.csv', index=False)


#
use_feat = feat[feat.Percent < 70]
print(use_feat.shape)

train2 = train[use_feat.index]
test2 = test[use_feat.index]
print(train2.columns)
print(test2.columns)

train2.to_csv('train_use_feat_70.csv', index=False)
test2.to_csv('test_use_feat_70.csv', index=False)