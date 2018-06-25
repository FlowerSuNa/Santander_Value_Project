
# Import Library
import numpy as np
import pandas as pd
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


def check_important_features(model, train_columns):
    important_features_index = [i for i, x in enumerate(model.feature_importances_) 
                                if x > np.mean(model.feature_importances_)]
    important_features_value = [x for x in model.feature_importances_ 
                                if x > np.mean(model.feature_importances_)]
    important_features = [train_columns[i+2] for i in important_features_index]
    
    for i in range(len(important_features)):
        print('{} = {}'.format(important_features, important_features_value))
    important_features


def training(train, features, max_depth, max_features, n_estimators, learning_rate):
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], train['target'], random_state=0)


    # Train the Model Using Decision Tree
    print('Train the Model Using Decision Tree')
    
    for d in range(1, max_depth+1):
        print('max_depth = {}'.format(d))
        
        tree = DecisionTreeRegressor(max_depth=d, random_state=0)
        tree.fit(X_train, y_train)
        
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        
        print('RMSLE of Train Set : ', RMSLE(y_train, train_pred))
        print('RMSLE of Test Set : {}\n'.format(RMSLE(y_test, test_pred)))
        
        
        # Check for Important Features
        
        
        
        pred = tree.predict(test[features])
        pred = np.exp(pred)
        
        submission['target'] = pred
        submission.to_csv('tree_' + str(d) + '.csv', index=False)
    
    
    # Train the Model Using Random Forest
    print('Train the Model Using Random Forest')
    
    for d in range(1, max_depth+1):
        for f in range(1, max_features+1):
            for n in range(25, n_estimators+1, 25):
                print('max_depth = {}, max_features = {}, n_estimators = {}'.format(d, f, n))
                
                forest = RandomForestRegressor(max_depth = d, max_features = f, n_estimators = n, random_state=0)
                forest.fit(X_train, y_train)
                
                train_pred = forest.predict(X_train)
                test_pred = forest.predict(X_test)
                
                print('RMSLE of Train Set : ', RMSLE(y_train, train_pred))
                print('RMSLE of Test Set : {}\n'.format(RMSLE(y_test, test_pred)))
                
                pred = forest.predict(test[features])
                pred = np.exp(pred)
                
                submission['target'] = pred
                submission.to_csv('forest_{}_{}_{}.csv'.format(str(d),str(f),str(n)), index=False)
    
    
    # Train the Model Using Gradient Boosting
    print('Train the Model Using Gradient Boosting')
    
    for d in range(1, max_depth+1):
        for n in range(25, n_estimators+1, 25):
            for l in [learning_rate/10000, learning_rate/1000, learning_rate/100, learning_rate/10, learning_rate]:
                print('max_depth = {}, n_estimators = {}, learning_rate = {}'.format(d, n, l))
                
                boost = GradientBoostingRegressor(random_state=0)
                boost.fit(X_train, y_train)
                
                train_pred = boost.predict(X_train)
                test_pred = boost.predict(X_test)
                
                print('RMSLE of Train Set : ', RMSLE(y_train, train_pred))
                print('RMSLE of Test Set : {}\n'.format(RMSLE(y_test, test_pred)))
                
                pred = boost.predict(test[features])
                pred = np.exp(pred)
                
                submission['target'] = pred
                submission.to_csv('boost_{}_{}_{}.csv'.format(str(d),str(n),str(l)), index=False)
    
    
    
features = train.columns[2:]
training(train, features, 5, 5, 100, 10)




important_features_index = [i for i, x in enumerate(boost.feature_importances_) if x > 0.01]
important_features_index

important_features_value = [x for x in boost.feature_importances_ if x > 0.01]
important_features_value

important_features = [train.columns[i+2] for i in important_features_index]
important_features
