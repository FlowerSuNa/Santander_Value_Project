
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


def check_important_features(feature_importances_, train_columns):
    important_features_index = [i for i, x in enumerate(feature_importances_) 
                                if x > np.mean(feature_importances_)]
    
    important_features_value = [x for x in feature_importances_ 
                                if x > np.mean(feature_importances_)]
    
    important_features = [train_columns[i+2] for i in important_features_index]
    
    for i in range(len(important_features)):
        print('{} = {}'.format(important_features[i], important_features_value[i]))
        
    return important_features


def tree(train, features, min_depth, max_depth):
    tree_importance = []
    i = 0
    columns = ['max_depth', 'train_RMSLE', 'test_RMSLE', 'important_features']
    result = pd.DataFrame(columns=columns)
    
    
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], np.log(train['target']), random_state=0)


    # Train the Model Using Decision Tree
    print('Train the Model Using Decision Tree')
    
    for d in range(min_depth, max_depth+1):
        print('\nmax_depth = {}'.format(d))
        
        tree = DecisionTreeRegressor(max_depth=d, random_state=0)
        tree.fit(X_train, y_train)
        
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        
        
        # Calculate RMSLE
        train_rmsle = RMSLE(np.exp(y_train), np.exp(train_pred))
        test_rmsle = RMSLE(np.exp(y_test), np.exp(test_pred))
        
        print('RMSLE of Train Set : ', train_rmsle)
        print('RMSLE of Test Set : {}\n'.format(test_rmsle))
        
        
        # Check for Important Features
        feat = check_important_features(tree.feature_importances_, train.columns)
        tree_importance.append(feat)

        
        # Save Result
        r = pd.DataFrame({'max_depth':d, 'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle,
                          'important_features':', '.join(feat)}, 
                         columns=columns, index=[i])
        result = pd.concat([result, r], ignore_index=True)
        i += 1
        
    return tree_importance, result
    

def forest(train, features, max_depth, max_features, n_estimators):
    forest_importance = []
    i = 0
    columns = ['max_depth', 'max_features', 'n_estimators', 'train_RMSLE', 'test_RMSLE', 'important_features']
    result = pd.DataFrame(columns=columns)
    
    
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], np.log(train['target']), random_state=0)
    
    
    # Train the Model Using Random Forest
    print('Train the Model Using Random Forest')
    
    for d in range(1, max_depth+1):
        for f in range(1, max_features+1):
            for n in range(25, n_estimators+1, 25):
                print('\nmax_depth = {}, max_features = {}, n_estimators = {}'.format(d, f, n))
                
                forest = RandomForestRegressor(max_depth = d, max_features = f, n_estimators = n, random_state=0)
                forest.fit(X_train, y_train)
                
                train_pred = forest.predict(X_train)
                test_pred = forest.predict(X_test)
                
                
                # Calculate RMSLE
                train_rmsle = RMSLE(np.exp(y_train), np.exp(train_pred))
                test_rmsle = RMSLE(np.exp(y_test), np.exp(test_pred))
                
                print('RMSLE of Train Set : ', train_rmsle)
                print('RMSLE of Test Set : {}\n'.format(test_rmsle))
                
                
                # Check for Important Features
                feat = check_important_features(forest.feature_importances_, train.columns)
                forest_importance.append(feat)
                
                
                # Save Result
                r = pd.DataFrame({'max_depth':d, 'max_features':f, 'n_estimators':n, 
                                  'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle, 
                                  'important_features':', '.join(feat)}, 
                                 columns=columns, index=[i])
                result = pd.concat([result, r], ignore_index=True)
                i += 1
                
    return forest_importance, result
    

def boost(train, features, max_depth, n_estimators, learning_rate):
    boost_importance = []
    i = 0
    columns = ['max_depth', 'n_estimators', 'learning_rate', 'train_RMSLE', 'test_RMSLE', 'important_features']
    result = pd.DataFrame(columns=columns)
    
    
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], np.log(train['target']), random_state=0)
    
    
    # Train the Model Using Gradient Boosting
    print('Train the Model Using Gradient Boosting')
    
    for d in range(1, max_depth+1):
        for n in range(25, n_estimators+1, 25):
            for l in [learning_rate/10000, learning_rate/1000, learning_rate/100, learning_rate/10, learning_rate]:
                print('\nmax_depth = {}, n_estimators = {}, learning_rate = {}'.format(d, n, l))
                
                boost = GradientBoostingRegressor(max_depth=d, n_estimators=n, learning_rate=l, random_state=0)
                boost.fit(X_train, y_train)
                
                train_pred = boost.predict(X_train)
                test_pred = boost.predict(X_test)
                
                
                # Calculate RMSLE
                train_rmsle = RMSLE(np.exp(y_train), np.exp(train_pred))
                test_rmsle = RMSLE(np.exp(y_test), np.exp(test_pred))
                
                print('RMSLE of Train Set : ', train_rmsle)
                print('RMSLE of Test Set : {}\n'.format(test_rmsle))
                
                
                # Check for Important Features
                feat = check_important_features(boost.feature_importances_, train.columns)
                boost_importance.append(feat)
                
                
                # Save Result
                r = pd.DataFrame({'max_depth':d, 'n_estimators':n, 'learning_rate':l,
                                  'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle,
                                  'important_features':', '.join(feat)}, 
                                 columns=columns, index=[i])
                result = pd.concat([result, r], ignore_index=True)
                i += 1                
                
    return boost_importance, result


# Find Important Features Using Decision Tree
# -----------------------------------------------------------------------

features = train.columns[2:]

tree_importance1, tree_result1 = tree(train, features, 1, 5)
tree_importance1 = list(set(itertools.chain(*tree_importance1)))
print(len(tree_importance1))

print(train[['target'] + tree_importance1].corr(method='pearson'))
print(train[['target'] + tree_importance1].corr(method='spearman'))

print(tree_importance1)
print(tree_result1)

tree_result1.to_csv('tree_result1.csv', index=False)

# -----------------------------------------------------------------------

train2 = train.drop(tree_importance1, axis=1)
features2 = train2.columns[2:]

tree_importance2, tree_result2 = tree(train2, features2, 1, 5)
tree_importance2 = list(set(itertools.chain(*tree_importance2)))
print(len(tree_importance2))

print(train2[['target'] + tree_importance1].corr(method='pearson'))
print(train2[['target'] + tree_importance1].corr(method='spearman'))

print(tree_importance2)
print(tree_result2)

tree_result2.to_csv('tree_result2.csv', index=False)
train2.to_csv('train2.csv', index=False)

# -----------------------------------------------------------------------

train3 = train2.drop(tree_importance2, axis=1)
features3 = train3.columns[2:]

tree_importance3, tree_result3 = tree(train3, features3, 1, 5)
tree_importance3 = list(set(itertools.chain(*tree_importance3)))
print(len(tree_importance3))

print(train3[['target'] + tree_importance1].corr(method='pearson'))
print(train3[['target'] + tree_importance1].corr(method='spearman'))

print(tree_importance3)
print(tree_result3)

tree_result3.to_csv('tree_result3.csv', index=False)
train3.to_csv('train3.csv', index=False)

# -----------------------------------------------------------------------


# Find Important Features Using Random Forest
# -----------------------------------------------------------------------

features = train.columns[2:]

forest_importance1, forest_result1 = forest(train, features, 5, 5, 100)
forest_importance1 = list(set(itertools.chain(*forest_importance1)))
print(len(forest_importance1))

print(train[['target'] + forest_importance1].corr(method='pearson'))
print(train[['target'] + forest_importance1].corr(method='spearman'))

print(forest_importance1)
print(forest_result1)

forest_result1.to_csv('forest_result1.csv', index=False)

# -----------------------------------------------------------------------

train2 = train.drop(forest_importance1, axis=1)
features2 = train2.columns[2:]

forest_importance2, forest_result2 = forest(train2, features2, 5, 5, 100)
forest_importance2 = list(set(itertools.chain(*forest_importance2)))
print(len(forest_importance2))

print(train2[['target'] + forest_importance1].corr(method='pearson'))
print(train2[['target'] + forest_importance1].corr(method='spearman'))

print(forest_importance2)
print(forest_result2)

forest_result2.to_csv('forest_result2.csv', index=False)

# -----------------------------------------------------------------------


# Find Important Features Using Gradient Boost
# -----------------------------------------------------------------------

features = train.columns[2:]

tree_importance1, tree_result1 = tree(train, features, 1, 5)
tree_importance1 = list(set(itertools.chain(*tree_importance1)))
print(len(tree_importance1))

print(train[['target'] + tree_importance1].corr(method='pearson'))
print(train[['target'] + tree_importance1].corr(method='spearman'))

print(tree_importance1)
print(tree_result1)

tree_result1.to_csv('tree_result1.csv', index=False)

# -----------------------------------------------------------------------

train2 = train.drop(tree_importance1, axis=1)
features2 = train2.columns[2:]

tree_importance2, tree_result2 = tree(train2, features2, 1, 5)
tree_importance2 = list(set(itertools.chain(*tree_importance2)))
print(len(tree_importance2))

print(train2[['target'] + tree_importance1].corr(method='pearson'))
print(train2[['target'] + tree_importance1].corr(method='spearman'))

print(tree_importance2)
print(tree_result2)

tree_result2.to_csv('tree_result2.csv', index=False)
train2.to_csv('train2.csv', index=False)

# -----------------------------------------------------------------------

train3 = train2.drop(tree_importance2, axis=1)
features3 = train3.columns[2:]

tree_importance3, tree_result3 = tree(train3, features3, 1, 5)
tree_importance3 = list(set(itertools.chain(*tree_importance3)))
print(len(tree_importance3))

print(train3[['target'] + tree_importance1].corr(method='pearson'))
print(train3[['target'] + tree_importance1].corr(method='spearman'))

print(tree_importance3)
print(tree_result3)

tree_result3.to_csv('tree_result3.csv', index=False)
train3.to_csv('train3.csv', index=False)

# -----------------------------------------------------------------------