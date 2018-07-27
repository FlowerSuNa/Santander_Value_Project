
# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso



# Define Functions
def RMSLE(real, pred):
    r = np.log1p(real)
    p = np.log1p(pred)
    
    rmsle = (np.sum(np.subtract(p, r) ** 2) / len(pred)) ** 0.5
    return rmsle

def plotting(data1, data2):
    plt.scatter(data1, data2)
    plt.ylabel('target')
    plt.show()


# Load Data
train = pd.read_csv('train_remove_constant.csv')
test = pd.read_csv('test_remove_constant.csv')
target = pd.read_csv('train_target.csv', header=None)
submission = pd.read_csv('sample_submission.csv')


# Divid Data
X_train, X_val, y_train, y_val = train_test_split(train, target[0], test_size=0.2, random_state=0)
print('X_train size : ', X_train.shape)
print('X_val size : ', X_val.shape)
print('y_train size : ', y_train.shape)
print('y_val size : ', y_val.shape)


#
for a in [1, 10, 100]:
    lasso = Lasso(alpha=a, max_iter=1000000)
    lasso.fit(X_train, y_train)

    print('train score : {:.2f}'.format(lasso.score(X_train, y_train)))
    print('test score : {:.2f}'.format(lasso.score(X_val, y_val)))
    print('feature : {}'.format(np.sum(lasso.coef_ != 0)))
    print(train.columns[[c for c in lasso.coef_ != 0]])
    
    
    #
    real = np.expm1(y_train)
    pred = np.expm1(lasso.predict(X_train))
    rmsle = RMSLE(real, pred)
    print('train RMSLE : {:.5f}'.format(rmsle))
    
    
    #
    real = np.expm1(y_val)
    pred = np.expm1(lasso.predict(X_val))
    rmsle = RMSLE(real, pred)
    print('valid RMSLE : {:.5f}'.format(rmsle))
    
    
    #
    pred = np.expm1(lasso.predict(test))
    submission['target'] = pred
    submission['target'] = submission['target'].apply(lambda x: 0 if x < 0 else x)
    submission['target'] = submission['target'].apply(lambda x: 10000000 if x > 10000000 else x)
    submission.to_csv('submission_lasso_{}.csv'.format(str(a)), index=False)



# ----------------------------------------------------------------------------------------------------------
# Load Data
train = pd.read_csv('train_use_feat_90.csv')
test = pd.read_csv('test_use_feat_90.csv')
target = pd.read_csv('train_target.csv', header=None)
submission = pd.read_csv('sample_submission.csv')

target[0] = np.expm1(target[0])


# Divid Data
X_train, X_val, y_train, y_val = train_test_split(train, target[0], test_size=0.2, random_state=0)
print('X_train size : ', X_train.shape)
print('X_val size : ', X_val.shape)
print('y_train size : ', y_train.shape)
print('y_val size : ', y_val.shape)


#
lasso = Lasso(max_iter=10000)
lasso.fit(X_train, y_train)

print('train score : {:.2f}'.format(lasso.score(X_train, y_train)))
print('test score : {:.2f}'.format(lasso.score(X_val, y_val)))
print('feature : {}'.format(np.sum(lasso.coef_ != 0)))
print(train.columns[[c for c in lasso.coef_ != 0]])
print(lasso.coef_)


#
pred = lasso.predict(X_train)
plotting(pred, y_train)


pred = lasso.predict(X_val)
plotting(pred, y_val)





rmsle = RMSLE(real, pred)
print('train RMSLE : {:.5f}'.format(rmsle))


#
real = np.expm1(y_val)
pred = np.expm1(lasso.predict(X_val))
rmsle = RMSLE(real, pred)
print('valid RMSLE : {:.5f}'.format(rmsle))


#
for a in [1, 10, 100]:
    lasso = Lasso(alpha=a, max_iter=1000000)
    lasso.fit(X_train, y_train)

    print('train score : {:.2f}'.format(lasso.score(X_train, y_train)))
    print('test score : {:.2f}'.format(lasso.score(X_val, y_val)))
    print('feature : {}'.format(np.sum(lasso.coef_ != 0)))
    print(train.columns[[c for c in lasso.coef_ != 0]])
    
    
    #
    real = np.expm1(y_train)
    pred = np.expm1(lasso.predict(X_train))
    rmsle = RMSLE(real, pred)
    print('train RMSLE : {:.5f}'.format(rmsle))
    
    
    #
    real = np.expm1(y_val)
    pred = np.expm1(lasso.predict(X_val))
    rmsle = RMSLE(real, pred)
    print('valid RMSLE : {:.5f}'.format(rmsle))
    
    
    #
    pred = np.expm1(lasso.predict(test))
    submission['target'] = pred
    submission['target'] = submission['target'].apply(lambda x: 0 if x < 0 else x)
    submission['target'] = submission['target'].apply(lambda x: 10000000 if x > 10000000 else x)
    submission.to_csv('submission_lasso_{}.csv'.format(str(a)), index=False)



