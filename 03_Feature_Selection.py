
# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb


# Define Functions
def RMSLE(real, pred):
    r = [np.log(x+1) for x in real]
    p = [np.log(x+1) for x in pred]
    
    rmsle = (np.sum(np.subtract(p, r) ** 2) / len(pred)) ** 0.5
    return rmsle


# Load Data
train = pd.read_csv('train_use_feat_90.csv')
target = pd.read_csv('train_target.csv', header=None)


# Divid Data
X_train, X_val, y_train, y_val = train_test_split(train, target[0], test_size=0.2, random_state=0)
print('X_train size : ', X_train.shape)
print('X_val size : ', X_val.shape)
print('y_train size : ', y_train.shape)
print('y_val size : ', y_val.shape)


# ------------------------------------ Feature Scoring using LightGBM ------------------------------------
# Set a Parameter
params = {'objective': 'regression',
          'metric': 'rmse',
          'num_leaves': 40,
          'learning_rate': 0.005,
          'bagging_fraction': 0.6,
          'featrue_fraction': 0.6,
          'bagging_frequency': 6,
          'bagging_seed': 0,
          'verbosity': -1,
          'seed':0}

scores = []


for feat in train.columns:
    # Make Train dataset and Valid dataset
    train_set = lgb.Dataset(X_train[[feat]].values, label=y_train.values, feature_name=[feat])
    valid_set = lgb.Dataset(X_val[[feat]].values, label=y_val.values, feature_name=[feat])
    
    # Train the model
    model = lgb.train(params, train_set, 5000,
                      valid_sets=[train_set,valid_set],
                      early_stopping_rounds=100,
                      verbose_eval=500)
    
    # Add Scores
    real = np.exp(y_train) - 1
    pred = model.predict(X_train[[feat]], num_iteration=model.best_iteration)
    pred = np.exp(pred) - 1    
    train_rmsle = RMSLE(real, pred)
    
    real = np.exp(y_val) - 1
    pred = model.predict(X_val[[feat]], num_iteration=model.best_iteration)
    pred = np.exp(pred) - 1    
    test_rmsle = RMSLE(real, pred)

    scores.append((feat, round(test_rmsle, 5), round(train_rmsle - test_rmsle, 5)))


# Make the Result Dataset
features = pd.DataFrame(scores, columns=['feature','rmsle','sub']).set_index('feature')
features = features.sort_values(ascending=True, by='rmsle')


# Check the Result
print(features.head(10))
print(features.tail(10))

print(features.loc[features['sub'] > 0.03])
print(features.loc[features['sub'] < 0.03])
print(features.loc[features['sub'] < 0.001])
print(features.loc[features['sub'] < 0])


# Save the Result
features.to_csv('feature_scoring/feature_scoring_LGBM.csv')


# ------------------------------------ Feature Scoring using XGBoost ------------------------------------
# Set a Parameter
params = {'objective':'reg:linear',
          'eval_metric':'rmse',
          'eta':0.001,
          'max_depth':10,
          'subsample':0.6,
          'colsample_bytree':0.6,
          'alpha':0.001,
          'random_state':0,
          'silent':True}

scores = []


for feat in train.columns:
    # Make Train dataset and Valid dataset
    train_set = xgb.DMatrix(X_train[[feat]].values, y_train.values)
    valid_set = xgb.DMatrix(X_val[[feat]].values, y_val.values)
    
    
    # Train the model
    model = xgb.train(params, train_set, 5000,
                      [(train_set, 'train', valid_set, 'valid')],
                      maximize=False,
                      early_stopping_rounds=100,
                      verbose_eval=1000)
    
    
    # Add Scores
    test_set = xgb.DMatrix(X_train[[feat]])
    
    real = np.exp(y_train) - 1
    pred = model.predict(test_set, ntree_limit=model.best_ntree_limit)
    pred = np.exp(pred) - 1    
    train_rmsle = RMSLE(real, pred)
    
    test_set = xgb.DMatrix(X_val[[feat]])
    
    real = np.exp(y_val) - 1
    pred = model.predict(test_set, ntree_limit=model.best_ntree_limit)
    pred = np.exp(pred) - 1    
    test_rmsle = RMSLE(real, pred)

    scores.append((feat, round(test_rmsle, 5), round(train_rmsle - test_rmsle, 5)))

