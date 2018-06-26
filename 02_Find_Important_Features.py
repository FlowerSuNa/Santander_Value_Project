
# Import Library
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Define Functions
def RMSLE(real, pred):
    r = [np.log(x+1) for x in real]
    p = [np.log(x+1) for x in pred]
    
    rmsle = (np.sum(np.subtract(p, r) ** 2) / len(pred)) ** 0.5
    return rmsle


def check_important_features(feature_importances_, featrues):
    important_features = pd.Series(feature_importances_, index=features)
    important_features = important_features.nlargest(100)
        
    return important_features


# Create Result DataFrame
idx = 0
columns = ['model', 'train_RMSLE', 'test_RMSLE']
result = pd.DataFrame(columns=columns)


# Find Important Features Using Decision Tree
train2 = train.copy()    

for i in range(3):
    # Divid Train Data
    features = train2.columns[2:]
    X_train, X_test, y_train, y_test = train_test_split(train2[features], np.log(train2['target']), random_state=0)
    
    
    # Train the model
    tree = DecisionTreeRegressor(random_state=0)
    tree.fit(X_train, y_train)
    
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)
    
    
    # Calculate RMSLE
    train_rmsle = RMSLE(np.exp(y_train), np.exp(train_pred))
    test_rmsle = RMSLE(np.exp(y_test), np.exp(test_pred))
    
    print('RMSLE of Train Set : ', train_rmsle)
    print('RMSLE of Test Set : {}\n'.format(test_rmsle))
    
    
    # Check for Important Features
    tree_feat = check_important_features(tree.feature_importances_, X_train.columns)
    print('Important Features Size : ', tree_feat.shape)
    print('Important Features :\n', tree_feat)
    
    
    # Save Result
    tree_feat.to_csv('tree_feat' + str(i+1) + '.csv')
    
    r = pd.DataFrame({'model':'tree' + str(i+1), 'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle}, 
                     columns=columns, index=[idx])
    result = pd.concat([result, r], ignore_index=True)
    idx += 1
    
    
    train2 = train2.drop(list(tree_feat.index), axis=1)
    

# Find Important Features Using Random Forest
train2 = train.copy()    

for i in range(3):
    # Divid Train Data
    features = train2.columns[2:]
    X_train, X_test, y_train, y_test = train_test_split(train2[features], np.log(train2['target']), random_state=0)
    
    
    # Train the model
    forest = RandomForestRegressor(random_state=0)
    forest.fit(X_train, y_train)
    
    train_pred = forest.predict(X_train)
    test_pred = forest.predict(X_test)
    
    
    # Calculate RMSLE
    train_rmsle = RMSLE(np.exp(y_train), np.exp(train_pred))
    test_rmsle = RMSLE(np.exp(y_test), np.exp(test_pred))
    
    print('RMSLE of Train Set : ', train_rmsle)
    print('RMSLE of Test Set : {}\n'.format(test_rmsle))
    
    
    # Check for Important Features
    forest_feat = check_important_features(forest.feature_importances_, X_train.columns)
    print('Important Features Size : ', forest_feat.shape)
    print('Important Features :\n', forest_feat)
    
    
    # Save Result
    forest_feat.to_csv('forest_feat' + str(i+1) + '.csv')
    
    r = pd.DataFrame({'model':'forest' + str(i+1), 'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle}, 
                     columns=columns, index=[idx])
    result = pd.concat([result, r], ignore_index=True)
    idx += 1
    
    
    train2 = train2.drop(list(forest_feat.index), axis=1)
    

# Find Important Features Using Gradient Boosting
train2 = train.copy()    

for i in range(3):
    # Divid Train Data
    features = train2.columns[2:]
    X_train, X_test, y_train, y_test = train_test_split(train2[features], np.log(train2['target']), random_state=0)
    
    
    # Train the model
    boost = GradientBoostingRegressor(random_state=0)
    boost.fit(X_train, y_train)
    
    train_pred = boost.predict(X_train)
    test_pred = boost.predict(X_test)
    
    
    # Calculate RMSLE
    train_rmsle = RMSLE(np.exp(y_train), np.exp(train_pred))
    test_rmsle = RMSLE(np.exp(y_test), np.exp(test_pred))
    
    print('RMSLE of Train Set : ', train_rmsle)
    print('RMSLE of Test Set : {}\n'.format(test_rmsle))
    
    
    # Check for Important Features
    boost_feat = check_important_features(boost.feature_importances_, X_train.columns)
    print('Important Features Size : ', boost_feat.shape)
    print('Important Features :\n', boost_feat)
    
    
    # Save Result
    boost_feat.to_csv('boost_feat' + str(i+1) + '.csv')
    
    r = pd.DataFrame({'model':'boost' + str(i+1), 'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle}, 
                     columns=columns, index=[idx])
    result = pd.concat([result, r], ignore_index=True)
    idx += 1
    
    
    train2 = train2.drop(list(boost_feat.index), axis=1)
    

print(result)
result.to_csv('result1.csv', index=False)


# Compare Important Features