
# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load Data
train = pd.read_csv('train_remove_constant.csv')
test = pd.read_csv('test_remove_constant.csv')
target = pd.read_csv('train_target.csv', header=None)


# ------------------------------------------------------ PCA ------------------------------------------------------

length = len(train)

data = pd.concat([train, test])
print(data.shape)       # (53801, 4735)


scaler = StandardScaler()
scaler.fit(data)
X_scaled = scaler.transform(data)


pca = PCA(n_components=100)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(X_pca.shape)

train_pca = pd.DataFrame(X_pca[:length])
test_pca = pd.DataFrame(X_pca[length:])

print(train_pca.shape)
print(test_pca.shape)

train_pca.to_csv('train_pca_100.csv', index=False)
test_pca.to_csv('test_pca_100.csv', index=False)


# ------------------------------------------------------- #
pca = PCA(n_components=500)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(X_pca.shape)

train_pca = pd.DataFrame(X_pca[:length])
test_pca = pd.DataFrame(X_pca[length:])

print(train_pca.shape)
print(test_pca.shape)

train_pca.to_csv('train_pca_500.csv', index=False)
test_pca.to_csv('test_pca_500.csv', index=False)


# ------------------------------------------------------- #
train = pd.read_csv('train_use_feat_90.csv')
test = pd.read_csv('test_use_feat_90.csv')

length = len(train)

data = pd.concat([train, test])
print(data.shape)       # (53801, 376)


scaler = StandardScaler()
scaler.fit(data)
X_scaled = scaler.transform(data)


pca = PCA(n_components=100)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(X_pca.shape)

train_pca = pd.DataFrame(X_pca[:length])
test_pca = pd.DataFrame(X_pca[length:])

print(train_pca.shape)
print(test_pca.shape)

train_pca.to_csv('train_use_feat_90_pca_100.csv', index=False)
test_pca.to_csv('test_use_feat_90_pca_100.csv', index=False)


# ------------------------------------------------------- #
pca = PCA(n_components=50)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(X_pca.shape)

train_pca = pd.DataFrame(X_pca[:length])
test_pca = pd.DataFrame(X_pca[length:])

print(train_pca.shape)
print(test_pca.shape)

train_pca.to_csv('train_use_feat_90_pca_50.csv', index=False)
test_pca.to_csv('test_use_feat_90_pca_50.csv', index=False)


# ------------------------------------------------------- #
pca = PCA(n_components=30)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(X_pca.shape)

train_pca = pd.DataFrame(X_pca[:length])
test_pca = pd.DataFrame(X_pca[length:])

print(train_pca.shape)
print(test_pca.shape)

train_pca.to_csv('train_use_feat_90_pca_30.csv', index=False)
test_pca.to_csv('test_use_feat_90_pca_30.csv', index=False)
