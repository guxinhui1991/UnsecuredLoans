# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:36:22 2018

@author: xigu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:46:56 2018

@author: xigu
"""

########################################################################
##
##  Assuming Data format same as before
##  This py file is used for forecasting, Coefficients calibrated from Stage 4
##  And Calculate CDR from SMM
##
##  Note:
##  Buckets should be EXACTLY same as stage 4
##
##
########################################################################


# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime

from collections import Counter
from dateutil import relativedelta
from matplotlib import pyplot as plt


# In[]:
def normalize(probs, NUM_Datapoints):
    sum_probs = np.zeros([NUM_Datapoints, 1])
    Probs_normalized = {}
    for key in probs.keys():
        sum_probs = sum_probs + probs[key]

    for key in probs.keys():
        Probs_normalized[key] = probs[key] / sum_probs
    return Probs_normalized


##
## Convert numerical to Categorical
def check_numerical(df, col):
    df[col] = df[col].astype(float)
    return df[col]


# In[]:

columns = ['Loan Origination Date',
           'Loan ID',
           'Original Loan Balance',
           'Current Loan Balance',
           'Loan Status',
           'Term',
           'Loan Age',
           'Grade',
           'Interest Rate',
           'Pre-Loan DTI',
           'Payment-To-Income',
           'Original FICO',
           'Housing Status']
data_path = 'Data/'

Balance = 'Current Loan Balance'
data = pd.read_csv(data_path + 'avant-platform-as-of-2017-09-30-with-paystring.csv', usecols=columns)
data = data[data['Term'] == 36]
data = data.loc[data['Loan Status'].isin(
    ['Late (16 - 29 DPD)', '30 - 59 Days Delinquent', '60 - 89 Days Delinquent', '>= 90 Days Delinquent', 'Current'])]

data = data.replace(['<null>'], 0)

# In[]:

columns_names = [
    'Loan ID',
    'Original Loan Balance',
    'Current Loan Balance',
    'Loan Status',
    'Date',
    'Term',
    'Age',
    'Grade',
    'Interest Rate',
    'Pre-Loan DTI',
    'Payment-To-Income',
    'Original FICO',
    'Housing Status']
data.columns = columns_names
Age_vec = data['Age']

# In[]:
########################################################################
##
##  Setting All Current Status to 1
##  E.g, all loans are current
##
########################################################################
loan_status = (data['Loan Status'] == 'Current').astype(int)
current_name = 'Current'
loan_status_dummy = pd.get_dummies(loan_status)
curr_status_cols = [current_name + '_DQ', current_name + '_C']
data[curr_status_cols] = loan_status_dummy

# In[]:

########################################################################
##
##
##  Buckets Definition
##
##
########################################################################
intecpt = 'Intercept'
cols_numerical = [
    'Original Loan Balance',
    'Interest Rate',
    'Payment-To-Income',
    'Original FICO',
    'Pre-Loan DTI',
    'Age']
for col in cols_numerical:
    data[col] = check_numerical(data.copy(), col)

housing_status = data['Housing Status']
housing_status_numerical = pd.get_dummies(housing_status)
grade = data['Grade']
grade[grade == 'B'] = 'A'
grade_cols = pd.get_dummies(grade)
data = pd.concat([data, housing_status_numerical], axis=1)
data = pd.concat([data, grade_cols], axis=1)

buckets = {}
buckets['Original Loan Balance'] = [0, 8000, 15000, 20000, 1e10]
buckets['Interest Rate'] = [0, 0.2, 0.25, 0.3, 0.35, 1e10]
buckets['Payment-To-Income'] = [0, 0.05, 0.10, 0.15, 1e10]
buckets['Original FICO'] = [0, 600, 650, 700, 750, 1e10]
buckets['Pre-Loan DTI'] = [0, 0.25, 0.5, 0.75, 1e10]
buckets['Age'] = list(np.arange(36)) + [10e6]

data_clean = data.copy()
data_clean[intecpt] = 1

cols = [intecpt] + list(housing_status_numerical.columns) + list(grade_cols.columns)
for col in cols_numerical:
    data_clean[col] = pd.cut(data[col], bins=buckets[col], include_lowest=True)
    col_dummy = pd.get_dummies(data_clean[col], prefix=col)
    cols = cols + list(col_dummy.columns)
    data_clean = pd.concat([data_clean, col_dummy], axis=1)
cols = cols + curr_status_cols

# In[]:
########################################################################
##
##
##  Forecast Balances
##  Setting Up Starting Balances
##
##  Test for Vintage Effects:
##
##
########################################################################

data_clean['Date'] = pd.to_datetime(data_clean['Date'])
data_clean['Origination Year'] = data_clean['Date'].dt.year
years = data_clean['Origination Year'].unique()

# In[]:
data_regression_vintage = {}
for year in years:
    df = data_clean[data_clean['Origination Year'] == year][cols + [Balance]].groupby(cols).sum().dropna()
    data_regression_vintage[year] = df.reset_index()


# In[]:
########################################################################
##
##
##  Forecast Balances for each time step
##
##
########################################################################

def probs_cal(df, x_col, Coeffs, status):
    Probs = {}
    for s in status:
        Probs[s] = 1 / (1 + np.exp(-df[x_col].dot(Coeffs[s].T)))
    Probs_normalized = normalize(Probs, len(data_regression))

    return Probs_normalized


def age_matrix_transform(df):
    last_group = df.iloc[:, -1]
    df = df.shift(1, axis=1)
    df = df.fillna(0)
    df.iloc[:, -1] = last_group
    return df


# In[]:
status = ['C', 'P', 'D', 'DQ']
status_dic = {"C": "Current",
              "P": "Prepaid",
              "D": "Default",
              "DQ": "Delinquent",
              }
T = 36
Coeffs = {}

for s in status:
    Coeffs[s] = pd.read_csv(data_path + 'to_' + s + '.csv', index_col=0)

# In[]:
########################################################################
##
##
##  Forecast Balances
##  Setting Up Starting Balances
##
##  Test for Vintage Effects:
##
##
########################################################################
Forecast_Bals_vintage = {}
for year in years:
    data_regression = data_regression_vintage[year]

    Forecast_Bals = pd.DataFrame(columns=status, index=range(T + 1)).fillna(0)
    bal_cols = ['Bal_' + s for s in status]
    Bal = pd.DataFrame(index=data_regression.index, columns=bal_cols)
    Bal['Bal_C'] = data_regression[data_regression['Current_C'] == 1][Balance]

    Bal['Bal_DQ'] = data_regression[data_regression['Current_DQ'] == 1][Balance]
    Bal = Bal.fillna(0)
    data_regression = pd.concat([data_regression, Bal], axis=1)
    Forecast_Bals.loc[0, :] = np.sum(Bal, axis=0).values
    Forecast_Bals_vintage[year] = Forecast_Bals
    data_regression_vintage[year] = data_regression

# In[]:
########################################################################
##
##
##  Forecast Balances
##  Dimemsion : NUM_Loans * 4 (4 status)
##
##
########################################################################

for year in years:

    data_regression = data_regression_vintage[year]
    Forecast_Bals = Forecast_Bals_vintage[year]

    age_cols = pd.get_dummies(pd.cut(Age_vec, bins=buckets['Age'], include_lowest=True), prefix='Age').columns

    for i in np.arange(1, T + 1):

        current_balance = data_regression[bal_cols].copy()
        current_age = data_regression[age_cols].copy()
        Probs_normalized = probs_cal(data_regression, cols, Coeffs, status)

        # Status : Default or Prepay
        # These two states are absorbing status, hence group together
        # C/DQ --> D/P
        for s in ['D', 'P']:
            current_balance['Bal_' + s] = data_regression[bal_cols]['Bal_' + s] + np.multiply(
                data_regression[bal_cols]['Bal_C'], np.squeeze(Probs_normalized[s].T.values)) + np.multiply(
                data_regression[bal_cols]['Bal_DQ'], np.squeeze(Probs_normalized[s].T.values))
            Forecast_Bals.loc[i, s] = sum(current_balance['Bal_' + s])

        # Status :
        # These two states are not absorbing status, hence needs balances from previous step
        # C/DQ --> C/DQ
        for s in ['C', 'DQ']:
            current_balance['Bal_' + s] = np.multiply(data_regression[bal_cols]['Bal_C'],
                                                      np.squeeze(Probs_normalized[s].T.values)) + np.multiply(
                data_regression[bal_cols]['Bal_DQ'], np.squeeze(Probs_normalized[s].T.values))
            Forecast_Bals.loc[i, s] = sum(current_balance['Bal_' + s])

        data_regression[age_cols] = age_matrix_transform(data_regression[age_cols])
        data_regression[bal_cols] = current_balance
        data_regression = data_regression[cols + [Balance] + bal_cols].groupby(cols).sum().dropna()
        data_regression = data_regression.reset_index()

    Forecast_Bals_vintage[year] = Forecast_Bals
    Forecast_Bals.to_csv(data_path + 'Forecast_' + str(year) + '.csv')

# In[]:
for s in status:
    print(s)
    fig = plt.figure(figsize=(20, 8))

    y = Forecast_Bals[s]
    x = np.arange(0, T + 1)
    width = .35

    plt.title('Status :' + status_dic[s])
    plt.bar(x, y, width=0.8)
    plt.xticks(x)
    fig.autofmt_xdate()

    plt.savefig(status_dic[s] + ".png")
