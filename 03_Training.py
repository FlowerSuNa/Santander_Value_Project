
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

merge_feat1 = pd.read_csv('merge_feat1.csv')
merge_feat2 = pd.read_csv('merge_feat2.csv')


# Define
def RMSLE(real, pred):
    r = [np.log(x+1) for x in real]
    p = [np.log(x+1) for x in pred]
    
    rmsle = (np.sum(np.subtract(p, r) ** 2) / len(pred)) ** 0.5
    return rmsle


def check_important_features(feature_importances_, features):
    important_features = pd.Series(feature_importances_, index=features)
    important_features = important_features.loc[important_features > 0].nlargest(100)
        
    return important_features

    
def tree(train, features, max_depth, start_depth=1, index=0):
    # Crete Result DataFrame
    i = 0
    columns = ['max_depth', 'train_RMSLE', 'test_RMSLE', 'important_features']
    result = pd.DataFrame(columns=columns)
    
    
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], np.log(train['target']), random_state=0)


    # Train the Model Using Decision Tree
    print('Train the Model Using Decision Tree')
    
    for d in range(start_depth, max_depth+1):
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
        feat = check_important_features(tree.feature_importances_, X_train.columns)
        feat.to_csv('tree_result/tree' + str(index) + '_' + str(d) + '.csv')

        
        # Save Result
        r = pd.DataFrame({'max_depth':d, 'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle,
                          'important_features':', '.join(feat.index)}, 
                         columns=columns, index=[i])
        result = pd.concat([result, r], ignore_index=True)
        i += 1
        
        
        # Predict Target
        # pred = tree.predict(test[features])
        # pred = np.exp(pred)
        
        
        # Save predicted Target
        # submission['target'] = pred
        # submission.to_csv('tree/tree_' + str(d) + '.csv', index=False)
        
    return result
    

def forest(train, features, max_depth, max_features, n_estimators, start_depth=1, start_features=1, unit_estimators=25):
    forest_importance = []
    i = 0
    columns = ['max_depth', 'max_features', 'n_estimators', 'train_RMSLE', 'test_RMSLE', 'important_features']
    result = pd.DataFrame(columns=columns)
    
    
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], np.log(train['target']), random_state=0)
    
    
    # Train the Model Using Random Forest
    print('Train the Model Using Random Forest')
    
    for d in range(start_depth, max_depth+1):
        for f in range(start_features, max_features+1):
            for n in range(25, n_estimators+1, unit_estimators):
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
                feat = check_important_features(forest.feature_importances_, X_train.columns)
                forest_importance.append(feat)
                
                
                # Save Result
                r = pd.DataFrame({'max_depth':d, 'max_features':f, 'n_estimators':n, 
                                  'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle, 
                                  'important_features':', '.join(feat.index)}, 
                                 columns=columns, index=[i])
                result = pd.concat([result, r], ignore_index=True)
                i += 1
                
                
                # Predict Target
                # pred = forest.predict(test[features])
                # pred = np.exp(pred)
                
                
                # Save Predicted Target
                # submission['target'] = pred
                # submission.to_csv('forest/forest_{}_{}_{}.csv'.format(str(d),str(f),str(n)), index=False)
                
    return forest_importance, result
    

def boost(train, features, max_depth, n_estimators, learning_rate, start_depth=1, unit_estimators=25, unit_rate=0.1):
    boost_importance = []
    i = 0
    columns = ['max_depth', 'n_estimators', 'learning_rate', 'train_RMSLE', 'test_RMSLE', 'important_features']
    result = pd.DataFrame(columns=columns)
    
    
    # Divid Train Data
    X_train, X_test, y_train, y_test = train_test_split(train[features], np.log(train['target']), random_state=0)
    
    
    # Train the Model Using Gradient Boosting
    print('Train the Model Using Gradient Boosting')
    
    for d in range(start_depth, max_depth+1):
        for n in range(25, n_estimators+1, unit_estimators):
            l = learning_rate
            for x in range(3):
                
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
                feat = check_important_features(boost.feature_importances_, X_train.columns)
                boost_importance.append(feat)
                
                
                # Save Result
                r = pd.DataFrame({'max_depth':d, 'n_estimators':n, 'learning_rate':l,
                                  'train_RMSLE':train_rmsle, 'test_RMSLE':test_rmsle,
                                  'important_features':', '.join(feat.index)}, 
                                 columns=columns, index=[i])
                result = pd.concat([result, r], ignore_index=True)
                i += 1 
                
                
                # Predict Target
                # pred = boost.predict(test[features])
                # pred = np.exp(pred)
                
                
                # Save Predicted Target
                # submission['target'] = pred
                # submission.to_csv('boost/boost_{}_{}_{}.csv'.format(str(d),str(n),str(l)), index=False)
                
                l *= unit_rate
                
    return boost_importance, result


features = train.columns[2:]

train1 = train.drop(merge_feat1.feature, axis=1)
features1 = train1.columns[2:]

train2 = train.drop(merge_feat2.feature, axis=1)
features2 = train2.columns[2:]

#
tree_result = tree(train, features, 10)
tree_result1 = tree(train, merge_feat1.feature, 10, index=1)
tree_result2 = tree(train, merge_feat2.feature, 10, index=2)
tree_result3 = tree(train, features1, 10, index=3)
tree_result4 = tree(train, features2, 10, index=4)

print(tree_result)
print(tree_result1)
print(tree_result2)
print(tree_result3)
print(tree_result4)

tree_result.to_csv('tree_result/tree_result0.csv', index=False)
tree_result1.to_csv('tree_result/tree_result1.csv', index=False)
tree_result2.to_csv('tree_result/tree_result2.csv', index=False)
tree_result3.to_csv('tree_result/tree_result3.csv', index=False)
tree_result4.to_csv('tree_result/tree_result4.csv', index=False)


#
forest_important1, forest_result1 = forest(train, merge_feat1.feature, 5, 5, 100)
forest_important2, forest_result2 = forest(train, merge_feat2.feature, 5, 5, 100)
forest_important3, forest_result3 = forest(train, features, 5, 5, 100)

print(forest_result1)
print(forest_result2)
print(forest_result3)

print(forest_important1)
print(forest_important2)
print(forest_important3)

forest_result1.to_csv('forest_result1.csv', index=False)
forest_result2.to_csv('forest_result2.csv', index=False)
forest_result3.to_csv('forest_result3.csv', index=False)


#
boost_important1, boost_result1 = boost(train, merge_feat1.feature, 5, 100, 1)

boost_important2, boost_result2 = boost(train, features, 5, 100, 1)

print(boost_result1)
print(boost_result2)

print(boost_important1)
print(boost_important2)

boost_result1.to_csv('boost_result1.csv', index=False)
boost_result2.to_csv('boost_result2.csv', index=False)



# tree_importance1 = list(set(itertools.chain(*tree_importance1)))
# print(len(tree_importance1))

# tree_importance = tree(train, features, 100000, 100000)
# tree_importance2 = list(set(itertools.chain(*tree_importance)))
# print(len(tree_importance2))
# print(tree_importance2)


# forest_importance1 = forest(train, tree_importance1, 5, 5, 100)









