
# Import Library
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')


# Define Functions
def RMSLE(real, pred):
    r = [np.log(x+1) for x in real]
    p = [np.log(x+1) for x in pred]
    
    rmsle = (np.sum(np.subtract(p, r) ** 2) / len(pred)) ** 0.5
    return rmsle


def check_important_features(feature_importances_, features):
    important_features = pd.Series(feature_importances_, index=features)
    important_features = important_features.loc[important_features > 0].nlargest(100)
        
    return important_features


for k in range(1,2):
    
    if k == 1:
        # Check Constant Features
        remove_feat = []
        for col in train.columns:
            if col != 'ID' and col != 'target':
                if train[col].std() == 0: 
                    remove_feat.append(col)
        
                
        # Remove Constant Features
        train.drop(remove_feat, axis=1, inplace=True)
        test.drop(remove_feat, axis=1, inplace=True) 
        
        print(len(remove_feat))
        print(remove_feat)
            

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
        
        
        # Predict Target
        pred = tree.predict(test[features])
        pred = np.exp(pred)
        
        
        # Save predicted Target
        submission['target'] = pred
        submission.to_csv('tree/tree' + str(i+1) + '.csv', index=False)
        
        
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
        
        
        # Predict Target
        pred = forest.predict(test[features])
        pred = np.exp(pred)
        
        
        # Save predicted Target
        submission['target'] = pred
        submission.to_csv('forest/forest' + str(i+1) + '.csv', index=False)
        
        
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
        
        
        # Predict Target
        pred = boost.predict(test[features])
        pred = np.exp(pred)
        
        
        # Save predicted Target
        submission['target'] = pred
        submission.to_csv('boost/boost' + str(i+1) + '.csv', index=False)
        
        
        train2 = train2.drop(list(boost_feat.index), axis=1)
        
    
    print(result)
    result.to_csv('result' + str(k+1) + '.csv', index=False)
    
    
    # Compare Important Features
    tree_feat1 = pd.read_csv('tree_feat1.csv', names=['feature','tree_feat1_value'])
    tree_feat2 = pd.read_csv('tree_feat2.csv', names=['feature','tree_feat2_value'])
    tree_feat3 = pd.read_csv('tree_feat3.csv', names=['feature','tree_feat3_value'])
    
    forest_feat1 = pd.read_csv('forest_feat1.csv', names=['feature','forest_feat1_value'])
    forest_feat2 = pd.read_csv('forest_feat2.csv', names=['feature','forest_feat2_value'])
    forest_feat3 = pd.read_csv('forest_feat3.csv', names=['feature','forest_feat3_value'])
    
    boost_feat1 = pd.read_csv('boost_feat1.csv', names=['feature','boost_feat1_value'])
    boost_feat2 = pd.read_csv('boost_feat2.csv', names=['feature','boost_feat2_value'])
    boost_feat3 = pd.read_csv('boost_feat3.csv', names=['feature','boost_feat3_value'])
    
    merge_feat = pd.merge(tree_feat1, tree_feat2, on='feature', how='outer')
    merge_feat = merge_feat.merge(tree_feat3, on='feature', how='outer')
    merge_feat = merge_feat.merge(forest_feat1, on='feature', how='outer')
    merge_feat = merge_feat.merge(forest_feat2, on='feature', how='outer')
    merge_feat = merge_feat.merge(forest_feat3, on='feature', how='outer')
    merge_feat = merge_feat.merge(boost_feat1, on='feature', how='outer')
    merge_feat = merge_feat.merge(boost_feat2, on='feature', how='outer')
    merge_feat = merge_feat.merge(boost_feat3, on='feature', how='outer')
    
    merge_feat['value_sum'] = merge_feat.sum(axis=1)
    merge_feat['value_mean'] = merge_feat['value_sum'] / 9
    merge_feat.to_csv('merge_feat' + str(k+1) + '.csv', index=False)


# Save Dataset
train.to_csv('train_pre.csv', index=False)
test.to_csv('test_pre.csv', index=False)
