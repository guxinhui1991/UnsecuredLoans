# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 10:00:01 2018

@author: xigu
"""

########################################################################
##
##
##  Validation of the models
##
##
########################################################################


# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import time

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
            data[i] = 'DQ'
        elif (data[i] == 'O'):
            data[i] = 'D'
    return data


##  Deprecated Method
##  Buckets need to be economically reasonable
##
##  Date: Feb 7th, 2018
##
##  Convert numerical to Categorical

# =============================================================================
# def convert_to_categorical(df, columns, num_buckets):
#     cols=[]
#     for col in columns:
#         print(col)
#         df[col] = df[col].astype(float)
#         #df[col] = pd.qcut(df[col].rank(method='first'), num_buckets, labels=False)
#         df[col] = pd.qcut(df[col], num_buckets)
#         col_dummy = pd.get_dummies(df[col], prefix=col)
#         cols=cols+list(col_dummy.columns)
#         df = pd.concat([df, col_dummy], axis=1)
#
#     return df, cols
#
# =============================================================================


##  Latest Method
##

##
## Convert numerical to Categorical
def check_numerical(df, col):
    df[col] = df[col].astype(float)
    return df[col]


# In[]:
########################################################################
##
##  Read Data
##
##  <null> data has been cleaned
##
##
########################################################################
cols_regression = [
    'Status',
    'Housing Status',
    'Grade',
    'Original Loan Balance',
    'Interest Rate',
    'Payment-To-Income',
    'Original FICO',
    'EHUPUS',
    'Age',
    'Pre-Loan DTI',
    'Next Status'
]

data_regression = pd.read_csv('data_regression_20180305.csv', usecols=cols_regression)
data_regression = data_regression.replace(['<null>'], 0)

# In[]:
#

housing_status = data_regression['Housing Status']
housing_status_numerical = pd.get_dummies(housing_status)
grade = data_regression['Grade']
grade[grade == 'B'] = 'A'
grade_cols = pd.get_dummies(grade)
data_regression = pd.concat([data_regression, housing_status_numerical], axis=1)
data_regression = pd.concat([data_regression, grade_cols], axis=1)

# In[]:

########################################################################
##
##
##  Buckets Definition
##
##
########################################################################
cols_numerical = [
    'Original Loan Balance',
    'Interest Rate',
    'Payment-To-Income',
    'Original FICO',
    #    'EHUPUS'
    'Pre-Loan DTI',
    'Age'
]
y_col = 'Next Status'
status_col = 'Status'
for col in cols_numerical:
    data_regression[col] = check_numerical(data_regression.copy(), col)

buckets = {}
buckets['Original Loan Balance'] = [0, 8000, 15000, 20000, 1e10]
buckets['Interest Rate'] = [0, 0.2, 0.25, 0.3, 0.35, 1e10]
buckets['Payment-To-Income'] = [0, 0.05, 0.10, 0.15, 1e10]
buckets['Original FICO'] = [0, 600, 650, 700, 750, 1e10]
buckets['Pre-Loan DTI'] = [0, 0.25, 0.5, 0.75, 1e10]
buckets['Age'] = list(np.arange(24)) + [10e6]

data_regression_clean = data_regression.copy()

cols = list(housing_status_numerical.columns) + list(grade_cols.columns)
for col in cols_numerical:
    data_regression_clean[col] = pd.cut(data_regression[col], bins=buckets[col], include_lowest=True)
    col_dummy = pd.get_dummies(data_regression_clean[col], prefix=col)
    cols = cols + list(col_dummy.columns)
    data_regression_clean = pd.concat([data_regression_clean, col_dummy], axis=1)

y_status = Modify_Delinquency(data_regression_clean.copy(), y_col)
data_regression_clean[y_col] = y_status
x_status = Modify_Delinquency(data_regression_clean.copy(), status_col)
data_regression_clean[status_col] = x_status

current_name = 'Current'
curr_status_dummy = pd.get_dummies(data_regression_clean[status_col], prefix=current_name)
curr_status_cols = [current_name + '_DQ', current_name + '_C']
x_cols = cols + curr_status_cols
data_regression_clean = pd.concat([data_regression_clean, curr_status_dummy[curr_status_cols]], axis=1)

# In[]:
########################################################################
##
##
##  Count Occurences
##
##
########################################################################
print(Counter(y_status))


# In[]:
# def Classification(df, y_col, identifier, curr_status):
#    y_values = (df[y_col]==identifier).astype(int)
#    df[curr_status] = 0
#    Current_status = (df[y_col]==identifier).astype(int)
#    df[curr_status][:-1] = Current_status[1:]
#
#    index = df[df[y_col]==identifier].index
#    df[y_col] = y_values
#    df.loc[index][curr_status] = 1
#    return df


def Classification(df, col, identifier):
    indicator = (df[col].values == identifier).astype(int)

    return indicator


# In[]:
########################################################################
##
##  Logistic Regression
##  Setting Hyper Parameters
##
########################################################################

alpha = 0.75

status = ['C', 'P', 'D', 'DQ']
y_names = ['To_' + s for s in status]

# In[]:
for i, s in enumerate(status):
    data_regression_clean[y_names[i]] = Classification(data_regression_clean.copy(), y_col, s)
drop_index = data_regression_clean.loc[data_regression_clean['Status'].isin(['D', 'P'])].index
data_regression_clean = data_regression_clean.drop(drop_index)

# In[]:
########################################################################
##
##  Logistic Regression
##  Setting Hyper Functions
##
########################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def logistic_regression(X_train, y_train, X_test, y_test, intercept):
    start = time.time()
    lr = LogisticRegression(fit_intercept=intercept, C=1e15).fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start))
    print_score(lr, X_test, y_test)
    return lr


def split_test_train(X, y, train_size=0.75):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    return X_train, X_test, y_train, y_test


def regression_results(df, y_name, x_cols, alpha):
    X_train, X_test, y_train, y_test = split_test_train(df[x_cols], df[y_name], train_size=alpha)
    result = logistic_regression(X_train, y_train, X_test, y_test, True)

    return result


# In[]:
########################################################################
##
##  Logistic Regression
##
########################################################################

regression_ = {}
for y_name in y_names:
    regression_[y_name] = regression_results(data_regression_clean, y_name, x_cols, alpha)

# In[]:
########################################################################
##
##  Output Results
##
########################################################################

for y_name in y_names:
    result = regression_[y_name]
    intcpt = result.intercept_
    coefs = np.squeeze(result.coef_)
    coeffs = pd.DataFrame(np.concatenate((intcpt, coefs))).T
    coeffs.columns = ['Intercept'] + x_cols
    coeffs.to_csv(y_name + '.csv')

# In[]:
########################################################################
##
##
##  Validation On the Original Dataset
##
##
########################################################################
validation = {}
for y_name in y_names:
    validation[y_name] = regression_[y_name].score(data_regression_clean[x_cols], data_regression_clean[y_name])
