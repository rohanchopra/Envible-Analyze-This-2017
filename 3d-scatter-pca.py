from sklearn.decomposition import PCA as sklearnPCA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



data = pd.read_csv('./Dataset/Training_Dataset.csv')

data.drop('mvar1',inplace=True,axis=1)
data.drop('cm_key',inplace=True,axis=1)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(data['mvar12'])
data['mvar12'] = le.transform(data['mvar12'])

target = list()
for i in range(len(data)):
    if data['mvar49'][i] == 1:
        target.append(1)
    elif data['mvar50'][i] == 1:
        target.append(1)
    elif data['mvar51'][i] == 1:
        target.append(1)
    else:
        target.append(0)

data['target']= target
data.drop('mvar46',inplace=True,axis=1)
data.drop('mvar47',inplace=True,axis=1)
data.drop('mvar48',inplace=True,axis=1)
data.drop('mvar49',inplace=True,axis=1)
data.drop('mvar50',inplace=True,axis=1)
data.drop('mvar51',inplace=True,axis=1)

dataYear = data.copy()

dataYear['ElectronicsSpend'] = data['mvar16']+data['mvar17']+data['mvar18']+data['mvar19']
dataYear['TravelSpend'] = data['mvar20']+data['mvar21']+data['mvar22']+data['mvar23']
dataYear['HouseholdSpend'] = data['mvar27']+data['mvar26']+data['mvar25']+data['mvar24']
dataYear['CarSpend'] = data['mvar28']+data['mvar29']+data['mvar30']+data['mvar31']
dataYear['RetailSpend'] = data['mvar32']+data['mvar33']+data['mvar34']+data['mvar35']
dataYear['TotalSpend'] = data['mvar36']+data['mvar37']+data['mvar38']+data['mvar39']

dataYear.drop('mvar16',inplace=True,axis=1)
dataYear.drop('mvar17',inplace=True,axis=1)
dataYear.drop('mvar18',inplace=True,axis=1)
dataYear.drop('mvar19',inplace=True,axis=1)
dataYear.drop('mvar20',inplace=True,axis=1)
dataYear.drop('mvar21',inplace=True,axis=1)
dataYear.drop('mvar22',inplace=True,axis=1)
dataYear.drop('mvar23',inplace=True,axis=1)
dataYear.drop('mvar24',inplace=True,axis=1)
dataYear.drop('mvar25',inplace=True,axis=1)
dataYear.drop('mvar26',inplace=True,axis=1)
dataYear.drop('mvar27',inplace=True,axis=1)
dataYear.drop('mvar28',inplace=True,axis=1)
dataYear.drop('mvar29',inplace=True,axis=1)
dataYear.drop('mvar30',inplace=True,axis=1)
dataYear.drop('mvar31',inplace=True,axis=1)
dataYear.drop('mvar32',inplace=True,axis=1)
dataYear.drop('mvar33',inplace=True,axis=1)
dataYear.drop('mvar34',inplace=True,axis=1)
dataYear.drop('mvar35',inplace=True,axis=1)
dataYear.drop('mvar36',inplace=True,axis=1)
dataYear.drop('mvar37',inplace=True,axis=1)
dataYear.drop('mvar38',inplace=True,axis=1)
dataYear.drop('mvar39',inplace=True,axis=1)


X = dataYear.copy()
X.drop('target',inplace=True,axis=1)
y = dataYear['target']


X_norm = (X - X.min())/(X.max() - X.min())

pca = sklearnPCA(n_components=3) #3-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))

fig = plt.figure()
ax3D = Axes3D(fig)

ax3D.scatter(transformed[y==0][0], transformed[y==0][1], transformed[y==0][2], label='Class 0', c='red')
ax3D.scatter(transformed[y==1][0], transformed[y==1][1], transformed[y==1][2], label='Class 1', c='blue')

plt.legend()
plt.show()