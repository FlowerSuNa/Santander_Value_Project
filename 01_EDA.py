
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
print('Total Test Feature with Missing Values : ', test.columns[test.isnull().sum() != 0].size)     # 0


# Check for Target Variable
print(train.target.min())
print(train.target.max())


# Check for Correlation
print(train.corr(method='pearson'))
print(train.corr(method='spearman'))
