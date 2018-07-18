
# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import lightgbm as lgb


# Load Data
target = pd.read_csv('train_target.csv', header=None)
submission = pd.read_csv('sample_submission.csv')
feature_scoring = pd.read_csv('feature_scoring/feature_scoring.csv')


# Define Functions
def RMSLE(real, pred):
    r = np.log1p(real)
    p = np.log1p(pred)
    
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
# Select Features
# --- 'adb64ff71', '70feb1494', '1db387535', '58e056e12', '26fc93eb7', 
# --- '5c6487af1', '703885424', 'c47340d97', '963a49cdc', '9fd594eec'
feat = list(feature_scoring['feature'].iloc[:10])
print(feat)


# Train a Model
# train : 1.33393
# valid : 1.48510
lgb_pred, result, LGBM = lgbm(train[feat], test[feat], target[0])


# Find Important Features
features = important_features(LGBM)
print(features.head(20))


# ------------------------------------------ #
# Select Features
# --- 'adb64ff71', '70feb1494', '1db387535', '58e056e12', '26fc93eb7', 
# --- '5c6487af1', '703885424', 'c47340d97', '963a49cdc', '9fd594eec', 
# --- '1931ccfdd', '15ace8c9f', 'c5a231d81', 'e176a204a', 'eeb9cd3aa', 
# --- '66ace2992', '024c577b9', 'f74e8f13d', '277ef93fc', '6619d81fc', 
# --- '58e2e02e6', '9306da53f', 'f190486d6', '91f701ba2', '0572565c2', 
# --- 'd6bb78916', '62e59a501', '1702b5bf0', 'b43a7cfd5', '20aa07010', 
# --- 'fb49e4212', '23310aa6f', 'e78e3031b', 'fb0f5dbfe', 'fc99f9426', 
# --- '491b9ee45', '324921c7b', '3c8a3ced0', '1e6306c7c', 'e7c0cfd0f', 
# --- '36a9a8479', 'b6c0969a2', '32174174c', '9a07d7b1f', '58232a6fb', 
# --- '5831f4c76', '4d2671746', 'a93118262', '13bdd610a', '241f0f867'
feat = list(feature_scoring['feature'].iloc[:50])
print(feat)


# Train a Model
# --- train : 1.12886
# --- valid : 1.43411
lgb_pred, result, LGBM = lgbm(train[feat], test[feat], target[0])


# Find Important Features
features = important_features(LGBM)
print(features.head(20))


# ------------------------------------------ #
# Select Features
# --- Add : 'c4972742d', '95742c2bf'
# --- Delete : '277ef93fc', 'e7c0cfd0f'
feat_ = feature_scoring.loc[feature_scoring['sub_mean'] < 0.03]
feat = list(feat_['feature'].iloc[:50])
print(feat)


# Train a Model
# --- train : 1.09132
# --- valid : 1.43409
lgb_pred, result, LGBM = lgbm(train[feat], test[feat], target[0])   # Overfitting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))


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


# Train the Model
lgb_pred, result, boost = lgbm(X_train, X_val, y_train, y_val, test)
submission['target'] = lgb_pred
submission.to_csv('feature_scoring_lgb_pred.csv', index=False)


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_pca_100.csv')
test = pd.read_csv('test_pca_100.csv')


# Train a Model
# --- 1.09743
# --- 1.48375
lgb_pred, result, LGBM = lgbm(train, test, target[0])   # Overfitting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_pca_500.csv')
test = pd.read_csv('test_pca_500.csv')


# Train a Model
# --- 0.72555
# --- 1.47753
lgb_pred, result, LGBM = lgbm(train, test, target[0])   # Overfitting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_use_feat_90_pca_100.csv')
test = pd.read_csv('test_use_feat_90_pca_100.csv')


# Train a Model
# --- 1.05823
# --- 1.49530
lgb_pred, result, LGBM = lgbm(train, test, target[0])   # Overfitting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_use_feat_90_pca_50.csv')
test = pd.read_csv('test_use_feat_90_pca_50.csv')


# Train a Model
# --- 1.06939
# --- 1.47835
lgb_pred, result, LGBM = lgbm(train, test, target[0])   # Overfitting


# Find Important Features
features = important_features(LGBM)
print(features.head(20))


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_use_feat_90_pca_30.csv')
test = pd.read_csv('test_use_feat_90_pca_30.csv')


# Train a Model
# --- 1.10258
# --- 1.48029
lgb_pred, result, LGBM = lgbm(train, test, target[0])


# Find Important Features
features = important_features(LGBM)
print(features.head(20)) 