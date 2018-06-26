
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
submission = pd.read_csv('sample_submission.csv')


# Define
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
    columns = ['max_depth','train_RMSLE','test_RMSLE']
    result = pd.DataFrame(columns=columns)
    
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], train['target'], random_state=0)


    # Train the Model Using Decision Tree
    print('Train the Model Using Decision Tree')
    
    for d in range(min_depth, max_depth+1):
        print('\nmax_depth = {}'.format(d))
        
        tree = DecisionTreeRegressor(max_depth=d, random_state=0)
        tree.fit(X_train, y_train)
        
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        
        train_rmsle = RMSLE(np.exp(y_train), np.exp(train_pred))
        test_rmsle = RMSLE(np.exp(y_test), np.exp(test_pred))
        
        r = pd.DataFrame({'max_depth':d,'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle}, 
                         columns=columns, index=[i])
        result = pd.concat([result, r], ignore_index=True)
        i += 1
        
        print('RMSLE of Train Set : ', train_rmsle)
        print('RMSLE of Test Set : {}\n'.format(test_rmsle))
        
        
        # Predict Target
        pred = tree.predict(test[features])
        pred = np.exp(pred)
        
        
        # Save predicted Target
        submission['target'] = pred
        submission.to_csv('tree_' + str(d) + '.csv', index=False)
        
    return tree_importance, result
    

def forest(train, features, max_depth, max_features, n_estimators):
    forest_importance = []
    
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], train['target'], random_state=0)
    
    
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
                
                print('RMSLE of Train Set : ', RMSLE(np.exp(y_train), np.exp(train_pred)))
                print('RMSLE of Test Set : {}\n'.format(RMSLE(np.exp(y_test), np.exp(test_pred))))
                
                
                # Check for Important Features
                forest_importance.append(check_important_features(forest.feature_importances_, train.columns))
                
                
                # Predict Target
                pred = forest.predict(test[features])
                pred = np.exp(pred)
                
                
                # Save Predicted Target
                submission['target'] = pred
                submission.to_csv('forest_{}_{}_{}.csv'.format(str(d),str(f),str(n)), index=False)
                
    return forest_importance
    

def boost(train, features, max_depth, n_estimators, learning_rate):
    boost_importance = []
    
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], train['target'], random_state=0)
    
    
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
                
                print('RMSLE of Train Set : ', RMSLE(np.exp(y_train), np.exp(train_pred)))
                print('RMSLE of Test Set : {}\n'.format(RMSLE(np.exp(y_test), np.exp(test_pred))))
                
                
                # Check for Important Features
                boost_importance.append(check_important_features(boost.feature_importances_, train.columns))
                
                
                # Predict Target
                pred = boost.predict(test[features])
                pred = np.exp(pred)
                
                
                # Save Predicted Target
                submission['target'] = pred
                submission.to_csv('boost_{}_{}_{}.csv'.format(str(d),str(n),str(l)), index=False)
                
    return boost_importance



#  
train['target'] = np.log(train['target'])


#     
features = train.columns[2:]

tree_importance = tree(train, features, 10)
tree_importance1 = list(set(itertools.chain(*tree_importance)))
print(len(tree_importance1))

tree_importance = tree(train, features, 100000, 100000)
tree_importance2 = list(set(itertools.chain(*tree_importance)))
print(len(tree_importance2))
print(tree_importance2)


forest_importance1 = forest(train, tree_importance1, 5, 5, 100)

