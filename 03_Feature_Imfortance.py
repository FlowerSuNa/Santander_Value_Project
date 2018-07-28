
# Import Library
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import lightgbm as lgb


# Load Data
target = pd.read_csv('train_target_log.csv', header=None)
submission = pd.read_csv('sample_submission.csv')


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
    
    
    # Set a Parameter
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
    
    
    # Train a model
    train = lgb.Dataset(X_train, label=y_train)
    valid = lgb.Dataset(X_val, label=y_val)
    
    evals_result = {}
    
    model = lgb.train(params, train, 5000,
                      valid_sets=[train,valid],
                      early_stopping_rounds=100,
                      verbose_eval=200,
                      evals_result=evals_result)
    
    
    # Evaluate the model
    real = np.expm1(y_train)
    pred = np.expm1(model.predict(X_train, num_iteration=model.best_iteration))
    rmsle = RMSLE(real, pred)
    print('RMSLE of Train data : ', rmsle)
    
    real = np.expm1(y_val)
    pred = np.expm1(model.predict(X_val, num_iteration=model.best_iteration))
    rmsle = RMSLE(real, pred)
    print('RMSLE of Valid data : ', rmsle) 
    
    
    # Predict Target
    pred = model.predict(test_data, num_iteration=model.best_iteration)
    pred = np.expm1(pred)
    
    return pred, evals_result, model


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_remove_constant.csv')
test = pd.read_csv('test_remove_constant.csv')


# Train a Model
lgb_pred, result, LGBM = lgbm(train, test, target[0])


# Find Important Features
features = important_features(LGBM)
features.drop(features[features['split'] == 0].index, inplace=True, axis=0)

print(features.head(20))    
print(features.tail(20))

features.to_csv('feature_importance/LGBM_remove_constant.csv', index=False)


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_sparsity_90.csv')
test = pd.read_csv('test_sparsity_90.csv')


# Train a Model
# --- train : 0.98985
# --- valid : 1.40980
lgb_pred, result, LGBM = lgbm(train, test, target[0])


# Find Important Features
features = important_features(LGBM)
features.drop(features[features['split'] == 0].index, inplace=True, axis=0)

print(features.head(20))    
print(features.tail(20))  

features.to_csv('feature_importance/LGBM_sparsity_90.csv', index=False)


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_sparsity_80.csv')
test = pd.read_csv('test_sparsity_80.csv')


# Train a Model
# --- train : 1.13595
# --- valid : 1.43989
lgb_pred, result, LGBM = lgbm(train, test, target[0])


# Find Important Features
features = important_features(LGBM)
features.drop(features[features['split'] == 0].index, inplace=True, axis=0)

print(features.head(20))    
print(features.tail(20))  

features.to_csv('feature_importance/LGBM_sparsity_80.csv', index=False)


# ----------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_sparsity_70.csv')
test = pd.read_csv('test_sparsity_70.csv')


# Train a Model
# --- train : 1.10235
# --- valid : 1.44254
lgb_pred, result, LGBM = lgbm(train, test, target[0])


# Find Important Features
features = important_features(LGBM)
features.drop(features[features['split'] == 0].index, inplace=True, axis=0)

print(features.head(20))    
print(features.tail(20))  

features.to_csv('feature_importance/LGBM_sparsity_70.csv', index=False)


# ----------------------------------------------------------------------------------------------
#
f1 = pd.read_csv('feature_importance/LGBM_remove_constant.csv')
f2 = pd.read_csv('feature_importance/LGBM_sparsity_90.csv')
f3 = pd.read_csv('feature_importance/LGBM_sparsity_80.csv')
f4 = pd.read_csv('feature_importance/LGBM_sparsity_70.csv')


#
print(f1.head(10))
print(f2.head(10))
print(f3.head(10))
print(f4.head(10))
