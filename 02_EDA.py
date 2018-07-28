
# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#
train = pd.read_csv('train_sparsity_80.csv')
test = pd.read_csv('test_sparsity_80.csv')
target = pd.read_csv('train_target.csv', header=None)
target_log = pd.read_csv('train_target_log.csv', header=None)


#
print('train size : ', train.shape)
print('test size: ', test.shape)


#
def plotting(data1, data2, t, tl):
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.scatter(data1, t)
    plt.scatter(data2, np.full(len(data2), t.min()-10))
    plt.ylabel('target')
    
    plt.subplot(1,2,2)
    plt.scatter(data1, tl)
    plt.scatter(data2, np.full(len(data2), tl.min()-1))
    plt.ylabel('log(target)+1')
    
    plt.title(data1.name)
    plt.show()
    
    
# --------------------------------------------------- 1 ---------------------------------------------------
#
train['sum'] = train.sum(axis=1)
test['sum'] = test.sum(axis=1)


#
train['target'] = target[0]
train['target_log'] = target_log[0]


#
print(train.corr(method='pearson')['target'])
print(train.corr(method='spearman')['target'])


#
print(train.corr(method='pearson')['target_log'])
print(train.corr(method='spearman')['target_log'])


#
del train['target']
del train['target_log']


#
for feat in train.columns:
    plotting(train[feat], test[feat], target[0], target_log[0])
    

# --------------------------------------------------- 2 ---------------------------------------------------
#
train_log = np.log1p(train)
test_log = np.log1p(test)


#
train_log['target'] = target[0]
train_log['target_log'] = target_log[0]


#
print(train_log.corr(method='pearson')['target'])
print(train_log.corr(method='spearman')['target'])


#
print(train_log.corr(method='pearson')['target_log'])
print(train_log.corr(method='spearman')['target_log'])


for feat in train_log.columns:
    plotting(train_log[feat], test_log[feat], target[0], target_log[0])
    

# --------------------------------------------------- 3 ---------------------------------------------------
#
train_root = round(np.power(train, 0.5),2)
test_root = round(np.power(test, 0.5), 2)


#
train_root['target'] = target[0]
train_root['target_log'] = target_log[0]


#
print(train_root.corr(method='pearson')['target'])
print(train_root.corr(method='spearman')['target'])


#
print(train_root.corr(method='pearson')['target_log'])
print(train_root.corr(method='spearman')['target_log'])


#
del train_root['target']
del train_root['target_log']


#
for feat in train_root.columns:
    plotting(train_root[feat], test_root[feat], target[0], target_log[0])
    
    
# --------------------------------------------------- 4 ---------------------------------------------------
#
train_pre = train.copy()
train_pre[train_pre != 0] = 1

test_pre = test.copy()
test_pre[test_pre != 0] = 1

train_pre['sum'] = train_pre.sum(axis=1)
test_pre['sum'] = test_pre.sum(axis=1)


train_pre['target'] = target[0]
train_pre['target_log'] = target_log[0]


print(train_pre.corr(method='pearson')['target'])
print(train_pre.corr(method='spearman')['target'])

print(train_pre.corr(method='pearson')['target_log'])
print(train_pre.corr(method='spearman')['target_log'])


#
for feat in train_pre.columns:
    plotting(train_pre[feat], test_pre[feat], target[0], target_log[0])




