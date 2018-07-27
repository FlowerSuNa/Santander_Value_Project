
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
def plotting(data1, data2, t, label):
    plt.scatter(data1, t)
    plt.scatter(data2, np.full(len(data2), t.min()-1))
    plt.title(data1.name)
    plt.ylabel(label)
    plt.show()
    

for feat in train.columns:
    plotting(train[feat], test[feat], target[0], 'target')
    plotting(train[feat], test[feat], target_log[0], 'log(target) +1')
    

#
train_log = np.log1p(train)
test_log = np.log1p(test)

for feat in train_log.columns:
    plotting(train_log[feat], test_log[feat], target[0], 'target')
    plotting(train_log[feat], test_log[feat], target_log[0], 'log(target) +1')
    
    
#
train['sum'] = train.sum(axis=1)
test['sum'] = test.sum(axis=1)

plotting(train['sum'], test['sum'], target[0], 'target')
plotting(train['sum'], test['sum'], target_log[0], 'log(target) +1')




