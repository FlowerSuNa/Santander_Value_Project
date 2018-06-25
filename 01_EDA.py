
# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# View Data Size
print('Train Record Size : {}'.format(train.shape[0]))      # 4459
print('Train Feature Size : {}'.format(train.shape[1]))     # 4993

print('Test Record Size : {}'.format(test.shape[0]))        # 49342
print('Test Feature Size : {}'.format(test.shape[1]))       # 4992


# View Data Columns
print('Train data columns : ', train.columns)
print('Test data columns : ', test.columns)


# View Data
print(train.head())
print(train.tail())

print(test.head())
print(test.tail())


# View Data Describe
print(train.describe())
print(test.describe())


# View Data Info
print(train.info())
print(test.info())


# Check for Missing Value
print('Total Train Feature with Missing Values : ', train.columns[train.isnull().sum() != 0].size)  # 0
print('Total Test Feature with Missing Values : ', test.columns[test.isnull().sum() != 0].size)  # 0


# Check for Target Variable
print(train.target.min())
print(train.target.max())


# Check for correlation
feature_pearson = x.index for x in train.corr(method='pearson').target if x.abs() > 0.5

train.corr(method='pearson')[]
print(train.corr(method='pearson'))
print(train.corr(method='spearman'))


# Find Important Features
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def RMSLE(real, pred):
    r = [np.log(x+1) for x in real]
    p = [np.log(x+1) for x in pred]
    
    rmsle = (np.sum(np.subtract(p, r) ** 2) / len(pred)) ** 0.5
    return rmsle
    

 
# 
train['target'] = np.log(train['target'])

# Divid Train Data
features = train.columns[2:]
X_train, X_test, y_train, y_test = train_test_split(train[features], train['target'], random_state=0)


# Train the Model Using Decision Tree
tree = DecisionTreeRegressor(random_state=0)
tree.fit(X_train, y_train)

print(tree.score(X_train, y_train))
print(tree.score(X_test, y_test))


# Train the Model Using Random Forest
forest = RandomForestRegressor(random_state=0)
forest.fit(X_train, y_train)

print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))


# Train the Model Using Gradient Boosting
boost = GradientBoostingRegressor(random_state=0)
boost.fit(X_train, y_train)

print(boost.score(X_train, y_train))
print(boost.score(X_test, y_test))

important_features_index = [i for i, x in enumerate(boost.feature_importances_) if x > 0.01]
important_features_index

important_features_value = [x for x in boost.feature_importances_ if x > 0.01]
important_features_value

important_features = [train.columns[i+2] for i in important_features_index]
important_features

pred = boost.predict(X_test)

rmsle = RMSLE(y_test, pred)
rmsle

print(boost.feature_importances_)


a = [1, 2, 3]
enumerate(a)

tree.feature_importances_


def find_important_features(train, features):
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], train['target'], random_state=0)


    # Train the Model Using Decision Tree
    print('Train the Model Using Decision Tree')
    tree = DecisionTreeRegressor(random_state=0)
    tree.fit(X_train, y_train)
    
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)
    
    print('RMSLE of Train Set : ', RMSLE(y_train, train_pred))
    print('RMSLE of Test Set', RMSLE(y_test, test_pred))
    
    
    # Train the Model Using Random Forest
    print('Train the Model Using Random Forest')
    forest = RandomForestRegressor(random_state=0)
    forest.fit(X_train, y_train)
    
    train_pred = forest.predict(X_train)
    test_pred = forest.predict(X_test)
    
    print('RMSLE of Train Set : ', RMSLE(y_train, train_pred))
    print('RMSLE of Test Set', RMSLE(y_test, test_pred))
    
    
    # Train the Model Using Gradient Boosting
    print('Train the Model Using Gradient Boosting')
    boost = GradientBoostingRegressor(random_state=0)
    boost.fit(X_train, y_train)
    
    train_pred = boost.predict(X_train)
    test_pred = boost.predict(X_test)
    
    print('RMSLE of Train Set : ', RMSLE(y_train, train_pred))
    print('RMSLE of Test Set', RMSLE(y_test, test_pred))
    

find_important_features(train, features)


pred = forest.predict(test[features])
pred = np.exp(pred)

submission = pd.read_csv('sample_submission.csv')
submission.columns
submission['target'] = pred

submission.to_csv('sub1.csv', index=False)







pred = boost.predict(test[features])
pred = np.exp(pred)

submission['target'] = pred

submission.to_csv('sub2.csv', index=False)
