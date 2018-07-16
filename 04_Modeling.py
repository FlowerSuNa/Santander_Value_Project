
# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb


# Load Data
target = pd.read_csv('train_target.csv', header=None)
submission = pd.read_csv('sample_submission.csv')


# Define Functions
def RMSLE(real, pred):
    r = [np.log(x+1) for x in real]
    p = [np.log(x+1) for x in pred]
    
    rmsle = (np.sum(np.subtract(p, r) ** 2) / len(pred)) ** 0.5
    return rmsle


def important_features(model):
    gain = model.feature_importance('gain')
    features = pd.DataFrame({'feature':model.feature_name(),
                             'split':model.feature_importance('split'),
                             'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    
    return features


def lgbm(train_data, test_data, target):
    # Divid the Data
    X_train, X_val, y_train, y_val = train_test_split(train_data, target, test_size=0.2, random_state=0)
    print('X_train size : ', X_train.shape)
    print('X_val size : ', X_val.shape)
    print('y_train size : ', y_train.shape)
    print('y_val size : ', y_val.shape)
    
    
    #
    params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 40,
            'learning_rate': 0.005,
            'bagging_fraction': 0.6,
            'featrue_fraction': 0.6,
            'bagging_frequency': 6,
            'bagging_seed': 0,
            'verbosity': -1,
            'seed':0
            }
    
    train = lgb.Dataset(X_train, label=y_train)
    valid = lgb.Dataset(X_val, label=y_val)
    
    evals_result = {}
    
    model = lgb.train(params, train, 5000,
                      valid_sets=[train,valid],
                      early_stopping_rounds=100,
                      verbose_eval=200,
                      evals_result=evals_result)
    
    
    #
    real = np.exp(y_train) - 1
    pred = model.predict(X_train, num_iteration=model.best_iteration)
    pred = np.exp(pred) - 1
    
    rmsle = RMSLE(real, pred)
    print('RMSLE of Train data : ', rmsle)
    
    real = np.exp(y_val) - 1
    pred = model.predict(X_val, num_iteration=model.best_iteration)
    pred = np.exp(pred) - 1
    
    rmsle = RMSLE(real, pred)
    print('RMSLE of Test data : ', rmsle) 
    
    
    #
    pred = model.predict(test_data, num_iteration=model.best_iteration)
    pred = np.exp(pred) - 1
    
    return pred, evals_result, model


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_remove_constant.csv')
test = pd.read_csv('test_remove_constant.csv')


# Train a Model
lgb_pred, result, LGBM = lgbm(train, test, target[0])   # Overfitting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))    

features.to_csv('feature_importance/LGBM_remove_constant.csv', index=False)


# ------------------------------------------ #
# Remove 'f190486d6' Feature to decrease Overftiting
train2 = train.drop(features['feature'].iloc[0], axis=1)
test2 = test.drop(features['feature'].iloc[0], axis=1)


# Train a Model
lgb_pred, result, LGBM = lgbm(train2, test2, target[0])     # Decrease Overftiting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))    

features.to_csv('feature_importance/LGBM_remove_constant_remove1.csv', index=False)


# ------------------------------------------ #
# Remove '15ace8c9f' Feature
train3 = train2.drop(features['feature'].iloc[2], axis=1)
test3 = test2.drop(features['feature'].iloc[2], axis=1)


# Train a Model
lgb_pred, result, LGBM = lgbm(train3, test3, target[0])     # Decrease Overftiting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))    

features.to_csv('feature_importance/LGBM_remove_constant_remove1.csv', index=False)



# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_use_feat_90.csv')
test = pd.read_csv('test_use_feat_90.csv')


# Train a Model
lgb_pred, result, LGBM = lgbm(train, test, target[0])   # Overfitting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))    

features.to_csv('feature_importance/LGBM_use_feat_90.csv', index=False)


# ------------------------------------------ #
# Remove 'f190486d6' Feature to decrease Overftiting
train2 = train.drop(features['feature'].iloc[0], axis=1)
test2 = test.drop(features['feature'].iloc[0], axis=1)


# Train a Model
lgb_pred, result, LGBM = lgbm(train2, test2, target[0])     # Decrease Overftiting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))    

features.to_csv('feature_importance/LGBM_use_feat_90_remove1.csv', index=False)




submission['target'] = lgb_pred
submission.to_csv('lgb_90_pred.csv', index=False)


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_use_feat_80.csv')
test = pd.read_csv('test_use_feat_80.csv')



# Divid Data
X_train, X_val, y_train, y_val = train_test_split(train, target[0], test_size=0.2, random_state=0)
print('X_train size : ', X_train.shape)
print('X_val size : ', X_val.shape)
print('y_train size : ', y_train.shape)
print('y_val size : ', y_val.shape)


# Train the Model
lgb_pred, result, boost = lgbm(X_train, X_val, y_train, y_val, test)
submission['target'] = lgb_pred
submission.to_csv('lgb_80_pred.csv', index=False)


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_use_feat_70.csv')
test = pd.read_csv('test_use_feat_70.csv')



# Divid Data
X_train, X_val, y_train, y_val = train_test_split(train, target[0], test_size=0.2, random_state=0)
print('X_train size : ', X_train.shape)
print('X_val size : ', X_val.shape)
print('y_train size : ', y_train.shape)
print('y_val size : ', y_val.shape)


# Train the Model
lgb_pred, result, boost = lgbm(X_train, X_val, y_train, y_val, test)
submission['target'] = lgb_pred
submission.to_csv('lgb_70_pred.csv', index=False)


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_feature_scoring_LGBM_log.csv')
test = pd.read_csv('test_use_feat_70.csv')



# Divid Data
X_train, X_val, y_train, y_val = train_test_split(train, target[0], test_size=0.2, random_state=0)
print('X_train size : ', X_train.shape)
print('X_val size : ', X_val.shape)
print('y_train size : ', y_train.shape)
print('y_val size : ', y_val.shape)


# Train the Model
lgb_pred, result, boost = lgbm(X_train, X_val, y_train, y_val, test)
submission['target'] = lgb_pred
submission.to_csv('lgb_70_pred.csv', index=False)


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_feature_scoring_LGBM_log.csv')
test = pd.read_csv('test_use_feat_90.csv')

test = test[train.columns]
test = np.log1p(test)



# Divid Data
X_train, X_val, y_train, y_val = train_test_split(train, target[0], test_size=0.2, random_state=0)
print('X_train size : ', X_train.shape)
print('X_val size : ', X_val.shape)
print('y_train size : ', y_train.shape)
print('y_val size : ', y_val.shape)


# Train the Model
lgb_pred, result, boost = lgbm(X_train, X_val, y_train, y_val, test)
submission['target'] = lgb_pred
submission.to_csv('feature_scoring_lgb_pred.csv', index=False)



