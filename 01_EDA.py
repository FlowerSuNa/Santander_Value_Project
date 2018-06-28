
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


# Check Missing Value
def check_missing(data):
    nulls = data.isnull().sum(axis=0).reset_index()
    nulls.columns = ['columns','missing']
    nulls = nulls[nulls['missing'] > 0]
    
    return nulls

print('Missing Values of Train set\n', check_missing(train))    # Empty DataFrame
print('Missing Values of Test set\n', check_missing(test))      # Empty DataFrame


# Check Data Sparsity
def check_sparsity(data):
    non_zeros = (data.ne(0).sum(axis=1)).sum()
    total = data.shape[1] * data.shape[0]
    zeros = total - non_zeros
    sparsity = round(zeros / total * 100, 2)
    density = round(non_zeros / total * 100, 2)
    
    print('Total : ', total)
    print('Zeros : ', zeros)
    print('Sparsity : ', sparsity)
    print('Density : ', density)
    
    return density
    
print('Train Data Sparsity')
train_density = check_sparsity(train)

print('Test Data Sparsity')
test_density = check_sparsity(test)


# Check Features Type
def check_type(data):
    dtype = data.dtypes.reset_index()
    dtype.columns = ['count', 'column type']
    dtype = dtype.groupby('column type').aggregate('count').reset_index()
    
    return dtype

dtype_train = check_type(train)
print('Features Type of Train data\n', dtype_train)

dtype_test = check_type(test)
print('Features Type of Test data\n', dtype_test)


# Make Metadata
column = pd.Series(train.columns)
dtype = pd.Series([train[x].dtype for x in train.columns])
metadata = pd.DataFrame({'column':column, 'dtype':dtype})
print(metadata)
print(metadata.groupby('dtype')['dtype'].count().reset_index(name='count'))


# Check Data Sparsity per Feature Type
feat = metadata[(metadata.dtype == 'int64') & (metadata.column != 'ID') & (metadata.column != 'target')].column
train_int_density = check_sparsity(train[feat])
test_int_density = check_sparsity(test[feat])

feat = metadata[(metadata.dtype == 'float64') & (metadata.column != 'ID') & (metadata.column != 'target')].column
train_float_density = check_sparsity(train[feat])
test_float_density = check_sparsity(test[feat])

density_data = {'data':['train','test'], 'all':[train_density,test_density], 
                'integer':[train_int_density, test_int_density], 'float':[train_float_density, test_float_density]}


# Check for Target Variable
print(train.target.min())
print(train.target.max())


# Check for Correlation
print(train.corr(method='pearson'))
print(train.corr(method='spearman'))
