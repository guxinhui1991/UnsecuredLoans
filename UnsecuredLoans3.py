# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:06:43 2018

@author: xigu
"""

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import time

import datetime
from collections import Counter
from dateutil import relativedelta
from matplotlib import pyplot as plt


# In[]:

########################################################################
##
##
##  Utility Functions
##
##
########################################################################


##
## To print out the model's predictability
def print_score(model, x, y):
    print(model)
    print('Score : ', model.score(x, y))


## Check Delinquencies. Categorize delinquencies into Good and Serious Delinquency
##
## i.e.
## For Pattern 'C1C', dismiss the delinquency and consider as good payment
## For Pattern '1234', consider as serious Delinquency

def Modify_Delinquency(df, column_name):
    data = df[column_name].values
    for i in range(df.shape[0]):
        if (data[i].isdigit()):
            if (data[i] == '1'):
                data[i] = 'C'
            else:
                data[i] = 'DQ'
        elif (data[i] == 'O'):
            data[i] = 'D'
    return data


##
## Convert numerical to Categorical
def convert_to_categorical(df, columns, num_buckets):
    cols = []
    for col in columns:
        print(col)
        df[col] = df[col].astype(float)
        df[col] = pd.qcut(df[col].rank(method='first'), num_buckets, labels=False)
        col_dummy = pd.get_dummies(df[col], prefix=col)
        cols = cols + list(col_dummy.columns)
        df = pd.concat([df, col_dummy], axis=1)

    return df, cols


##
##
##
########################################################################
##
##
##  This function should be removed
##  Temporary fix because we start by reading in the data_regression.csv
##  Not the original dataset
##
##  The loans start with 'O' should be cleaned when first read the data
##  in stage 2.
##
##
##  Fix Done In Stage 4
##
########################################################################


## Remove Loans starting with 'O'
## Running Time : 25 minutes
##
def remove_charge_off(df, col, id_name):
    for i in df.index:
        print(i)
        if (df[col][i][0] == 'O'):
            loanid = df[id_name][i]
            drop_index = df[df[id_name] == loanid].index
            df = df.drop(drop_index)
    df = df.reset_index(drop=True)
    return df


# In[]:
########################################################################
##
##  Read Data
##
########################################################################

data_regression = pd.read_csv('data_regression_20180305.csv', index_col=0)
data_regression = data_regression.replace(['<null>'], 0)

# In[]:
#
# data_regression = remove_charge_off(data_regression, 'EOM Paystring', 'Loan ID')

########################################################################
##
##  Temporary Fix for Feb 7th, 2018
##
########################################################################

test = data_regression.sort_values('EOM Paystring')
todrop = test.iloc[3190434:3190891].index
data_regression = data_regression.drop(todrop)
data_regression = data_regression.reset_index(drop=True)

# In[]:
#

housing_status = data_regression['Housing Status']
housing_status_numerical = pd.get_dummies(housing_status)
grade = data_regression['Grade']
grade = pd.get_dummies(grade)
data_regression = pd.concat([data_regression, housing_status_numerical], axis=1)
data_regression = pd.concat([data_regression, grade], axis=1)

# In[]:

cols_numerical = [
    'Original Loan Balance',
    'Interest Rate',
    'Payment-To-Income',
    'Original FICO',
    'EHUPUS',
    'USSWAP3']
y_col = 'Status'

data_regression, columns_add = convert_to_categorical(data_regression, cols_numerical, 5)
x_cols_all_categories = list(housing_status_numerical.columns) + list(grade.columns) + columns_add

# In[]:
for col in x_cols_all_categories:
    data_regression[col] = data_regression[col].astype(float)

# In[]:
y_status = Modify_Delinquency(data_regression.copy(), y_col)

# In[]:
##  Count Occurences
##
print(Counter(y_status))

# In[]:
########################################################################
##
##  Logistic Regression
##  Setting Hyper Parameters
##
########################################################################

NUM_ENTRIES = data_regression.shape[0]
alpha = 0.75
ytrain = y_status[:int(NUM_ENTRIES * alpha)]
Xtrain = np.matrix(data_regression[x_cols_all_categories][:int(NUM_ENTRIES * alpha)])
x_test = data_regression[x_cols_all_categories][int(NUM_ENTRIES * alpha):]
y_test = y_status[int(NUM_ENTRIES * alpha):]

data_regression[y_col] = y_status
# In[]:
##
##  Categorize and split the dataset

Current = data_regression[data_regression[y_col] == 'C']
Default = data_regression[data_regression[y_col] == 'D']
Prepay = data_regression[data_regression[y_col] == 'P']
Delinquency = data_regression[data_regression[y_col] == 'DQ']

# In[]:
##
##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def logistic_regression(X_train, y_train, X_test, y_test):
    start = time.time()
    lr = LogisticRegression().fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start))
    print_score(lr, X_test, y_test)
    return lr


def split_test_train(X, y, train_size=0.75):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    return X_train, X_test, y_train, y_test


# In[]:
########################################################################
##
##
##  Logistic Regression 1
##  C-->D
##
##
########################################################################
dataset = Current.append(Default)
X_train, X_test, y_train, y_test = split_test_train(dataset[x_cols_all_categories], dataset[y_col], train_size=alpha)

lr_Default = logistic_regression(X_train, y_train, X_test, y_test)

# In[]:
########################################################################
##
##
##  Logistic Regression 2
##  C-->P
##
##
########################################################################
dataset = Current.append(Prepay)
X_train, X_test, y_train, y_test = split_test_train(dataset[x_cols_all_categories], dataset[y_col], train_size=alpha)

lr_Prepay = logistic_regression(X_train, y_train, X_test, y_test)

# In[]:
########################################################################
##
##
##  Logistic Regression
##  Results
##
##
########################################################################
Default = pd.DataFrame(np.exp(lr_Default.coef_))
Prepay = pd.DataFrame(np.exp(lr_Prepay.coef_))
Default.columns = x_cols_all_categories
Prepay.columns = x_cols_all_categories

# In[]:
Default.to_csv('Default.csv')
Prepay.to_csv('Prepay.csv')
