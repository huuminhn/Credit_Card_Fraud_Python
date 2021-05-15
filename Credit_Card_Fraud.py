# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:48:12 2021

@author: Minh Nguyen
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import imblearn
 
data = pd.read_csv("creditcard.csv")     

#EDA:  
data.info()
data.isnull().sum()

 ## Drop the Time column as we will not need it for analyzation:
data = data.drop('Time',axis = 1)

 ## Standardize the Amount:
scaler = preprocessing.StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape (-1,1))

 ## Class : 1 = Fraud, 0 = non-fraud
data['Class'].value_counts()
sns.countplot( x= 'Class', data = data)

 ## Undersampling the data:
from imblearn.under_sampling import RandomUnderSampler 
undersample = RandomUnderSampler(sampling_strategy=0.5)

dataX = data.drop('Class', axis = 1)
X, Y = undersample.fit_sample(dataX, data['Class'])
test = pd.DataFrame(Y, columns = ['Class'])

fig, axs = plt.subplots(ncols=2, figsize=(13,4.5))
sns.countplot(x="Class", data=data, ax=axs[0])
sns.countplot(x="Class", data=test, ax=axs[1])

fig.suptitle("Class repartition before and after undersampling")
a1=fig.axes[0]
a1.set_title("Before")
a2=fig.axes[1]
a2.set_title("After")

# Tried to use gridspec, kidna ugly, so use the above.
#from matplotlib import gridspec
#fig = plt.figure(figsize=(20, 6))
#gs = gridspec.GridSpec(nrows=1, ncols=2,
#                       height_ratios=[6], 
#                      width_ratios=[10, 10])

#ax = plt.subplot(gs[0])
#sns.countplot(x="Class", data=data, ax=ax, palette="RdGy")
#ax.set_title('Original Dataset', fontsize=15, fontweight='bold')

#ax2 = plt.subplot(gs[1])
#sns.countplot(x="Class", data=test, ax=ax2, palette="RdGy")
#ax2.set_title('UnderSampled Dataset', fontsize=15, fontweight='bold')

#plt.show()

 ## Split the data:

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state= 41)

# Import the modules for modeling:

print('hello word')











