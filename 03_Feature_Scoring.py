
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
                      verbose_eval=1000)
    
    # Add Scores
    real = np.expm1(y_train)
    pred = np.expm1(model.predict(X_train[[feat]], num_iteration=model.best_iteration))   
    train_rmsle = RMSLE(real, pred)
    
    real = np.expm1(y_val)
    pred = np.expm1(model.predict(X_val[[feat]], num_iteration=model.best_iteration))
    valid_rmsle = RMSLE(real, pred)

    scores.append((feat, round(valid_rmsle, 5), round(train_rmsle - valid_rmsle, 5)))


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
features.to_csv('feature_scoring/feature_scoring_LGB.csv')


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
    dset = xgb.DMatrix(X_train[[feat]].values)
    
    real = np.expm1(y_train)
    pred = np.expm1(model.predict(dset, ntree_limit=model.best_ntree_limit))   
    train_rmsle = RMSLE(real, pred)
    
    dset = xgb.DMatrix(X_val[[feat]].values)
    
    real = np.expm1(y_val)
    pred = np.expm1(model.predict(dset, ntree_limit=model.best_ntree_limit))
    valid_rmsle = RMSLE(real, pred)

    scores.append((feat, round(valid_rmsle, 5), round(train_rmsle - valid_rmsle, 5)))
    

# Make the Result Dataset
features = pd.DataFrame(scores, columns=['feature','rmsle','sub']).set_index('feature')
features = features.sort_values(ascending=True, by='rmsle')


# Check the Result
print(features.head(10))
print(features.tail(10))


# Save the Result
features.to_csv('feature_scoring/feature_scoring_XGB.csv')


# ------------------------------------ Feature Scoring using CatBoost ------------------------------------
scores = []


for feat in train.columns:
    model = ctb.CatBoostRegressor(iterations = 500,
                                  learning_rate=0.05,
                                  depth=10,
                                  eval_metric='RMSE',
                                  random_seed=0,
                                  bagging_temperature=0.2,
                                  od_type='Iter',
                                  metric_period=50,
                                  od_wait=20)
    
    model.fit(X_train[[feat]], y_train,
              eval_set=(X_val[[feat]], y_val),
              use_best_model=True,
              verbose=True)
    
    
    # Add Scores
    real = np.expm1(y_train)
    pred = np.expm1(model.predict(X_train[[feat]]))
    train_rmsle = RMSLE(real, pred)
    
    real = np.expm1(y_val)
    pred = np.expm1(model.predict(X_val[[feat]]))
    valid_rmsle = RMSLE(real, pred)
    
    scores.append((feat, round(valid_rmsle, 5), round(train_rmsle - valid_rmsle, 5)))


# Make the Result Dataset
features = pd.DataFrame(scores, columns=['feature','rmsle','sub']).set_index('feature')
features = features.sort_values(ascending=True, by='rmsle')


# Check the Result
print(features.head(10))
print(features.tail(10))


# Save the Result
features.to_csv('feature_scoring/feature_scoring_CTB.csv')


# ------------------------------------ Feature Selection ------------------------------------
LGB = pd.read_csv('feature_scoring/feature_scoring_LGB.csv')
XGB = pd.read_csv('feature_scoring/feature_scoring_XGB.csv')
CTB = pd.read_csv('feature_scoring/feature_scoring_CTB.csv')


features = pd.merge(LGB, XGB, on='feature', how='outer')
features = features.merge(CTB, on='feature', how='outer')


features.columns = ['feature','rmsle_lgb','sub_lgb','rmsle_xgb','sub_xgb','rmsle_ctb','sub_ctb']

print(features.head(10))
print(features.tail(10))


features['rmsle_mean'] = features[['rmsle_lgb','rmsle_xgb','rmsle_ctb']].mean(axis=1)
features['sub_mean'] = features[['sub_lgb','sub_xgb','sub_ctb']].mean(axis=1)

print(features.head(10))
print(features.tail(10))


features = features.sort_values(ascending=True, by='rmsle_mean')
print(features.head(20))


features.to_csv('feature_scoring/feature_scoring.csv', index=False)
features = pd.read_csv('feature_scoring/feature_scoring.csv')


# ------------------------------------ Modeling ------------------------------------
# Load Data
test = pd.read_csv('test_use_feat_90.csv')
target = pd.read_csv('train_target.csv', header=None)
submission = pd.read_csv('sample_submission.csv')


# Select Features
feat = list(features.head(30).feature)
print(feat)


# Train a Model
# --- 1.17512
# --- 1.43715
lgb_pred, result, LGBM = lgbm(train[feat], test[feat], target[0])


# Find Important Features
feature_ = important_features(LGBM)
print(feature_.head(20))


# Predict target
submission['target'] = lgb_pred
submission.to_csv('submission_feature_scoring.csv', index=False)


# -------------------------------------------------
# Select Features
feat = features.loc[features['sub_mean'] < 0.01]
feat = list(feat.head(30).feature)
print(feat)


# Train a Model
# --- 1.17637
# --- 1.44734
lgb_pred, result, LGBM = lgbm(train[feat], test[feat], target[0])


# Find Important Features
feature_ = important_features(LGBM)
print(feature_.head(20))


# -------------------------------------------------
# Select Features
feat = features.loc[features['sub_mean'] < 0.02]
feat = list(feat.head(30).feature)
print(feat)


# Train a Model
# --- 1.20505
# --- 1.44417
lgb_pred, result, LGBM = lgbm(train[feat], test[feat], target[0])


# Find Important Features
feature_ = important_features(LGBM)
print(feature_.head(20))







from multiprocessing import Pool
CPU_CORES = 1
def _get_leak(df, cols, lag=0):
    """ To get leak value, we do following:
       1. Get string of all values after removing first two time steps
       2. For all rows we shift the row by two steps and again make a string
       3. Just find rows where string from 2 matches string from 1
       4. Get 1st time step of row in 3 (Currently, there is additional condition to only fetch value if we got exactly one match in step 3)"""
    series_str = df[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    series_shifted_str = df[cols].shift(lag+2, axis=1)[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    target_rows = series_shifted_str.progress_apply(lambda x: np.where(x == series_str)[0])
    target_vals = target_rows.apply(lambda x: df.loc[x[0], cols[lag]] if len(x)==1 else 0)
    return target_vals

def get_all_leak(df, cols=None, nlags=15):
    """
    We just recursively fetch target value for different lags
    """
    df =  df.copy()
    #with Pool(processes=CPU_CORES) as p:
    #    res = [p.apply_async(_get_leak, args=(df, cols, i)) for i in range(nlags)]
    #    res = [r.get() for r in res]
    
    for i in range(nlags):
        print("Processing lag {}".format(i))
        df["leaked_target_"+str(i)] = _get_leak(df, cols, i)
    return df


series_str = train.apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
series_shifted_str = train.shift(0, axis=1).apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
target_rows = series_shifted_str.progress_apply(lambda x: np.where(x == series_str)[0])
target_vals = target_rows.apply(lambda x: train.loc[x[0]] if len(x)==1 else 0)