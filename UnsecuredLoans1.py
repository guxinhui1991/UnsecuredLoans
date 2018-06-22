# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime

from dateutil import relativedelta
from statsmodels.tsa.stattools import coint
from matplotlib import pyplot as plt

get_ipython().magic('matplotlib inline')

# In[58]:


columns = ['Loan ID',
           'Original Loan Balance',
           'Principal Payment (Cumulative)',
           'Interest Payment (Cumulative)',
           'Late Fees Paid (Cumulative)',
           'Loan Origination Date',
           'Loan Maturity Date',
           'Term',
           'Loan Age',
           'Grade',
           'Interest Rate',
           'APR',
           'Monthly Payment',
           'Annual Income',
           'Pre-Loan DTI',
           'Post-Loan DTI',
           'Payment-To-Income',
           'Original FICO',
           'Housing Status',
           'Employment Status',
           'Employment Length',
           'EOM Paystring']

rate_ticker, unemployment_ticker = 'USSWAP3', 'EHUPUS'
data_path = 'Data/'
data = pd.read_csv(data_path + 'avant-platform-as-of-2017-09-30-with-paystring.csv', usecols=columns)
usswap3 = pd.read_csv(data_path + rate_ticker + '.csv', index_col=0)
unemployment = pd.read_csv(data_path + unemployment_ticker + '.csv', index_col=0)

# In[72]:


data_36 = data[data.Term == 36].copy()
loan_id_mapping = pd.DataFrame(data_36['Loan ID'])
data_36['Loan ID'] = loan_id_mapping.index
data_36['Date'] = pd.to_datetime(data_36['Loan Origination Date'])

# Reindexing
data_36 = data_36.reset_index(drop=True)

num_loans_Total = len(data)
num_loans_36 = len(data_36)


# ## Cleaning the payment string
# All payment string with duplicate 'P' and 'O' are trimmed since we are only interested in the termination month.
# I.e, Sample payment string 'CCCCCPPPPP' will become 'CCCCCP'

# In[77]:


#
# Passed Data Sanity Check
#    if(paystring.find('P')>0):
#        print('ERROR')
#        print(paystring)
#

def clean_paystring(df):
    for i in df.index:
        paystring = df['EOM Paystring'][i]
        position_trim = len(paystring)
        if (paystring.find('O') > 0):
            position_trim = paystring.find('O')
        elif (paystring.find('P') > 0):
            position_trim = paystring.find('P')
        df.at[i, 'EOM Paystring'] = paystring[:position_trim + 1]
    return df


# In[78]:


data_36 = clean_paystring(data_36)


# ## Stacking Each Loan
# Converting each loan and stack up from the payment string and months.
# Adding age variable for each loan

# In[128]:


def convert_paystring(df, columns):
    df['Age'] = 0
    df['Status'] = 0
    df_converted = pd.DataFrame(columns=df.columns)

    loans = {}
    count = 0
    for ind in df.index:
        loan_i = df.loc[ind].to_dict()
        paystring = loan_i['EOM Paystring']
        length_loan = len(paystring)

        for i in range(length_loan):
            loan_i_copy = loan_i.copy()
            loan_i_copy['Age'] = i + 1
            loan_i_copy['Status'] = paystring[i]
            loan_i_copy['Date'] = loan_i_copy['Date'] + relativedelta.relativedelta(months=i)
            loans[count] = loan_i_copy
            count = count + 1
    df_converted = pd.DataFrame.from_dict(loans, orient='index')
    # df_converted = df_converted.reset_index(drop=True)
    return df_converted


# In[ ]:
# data_36_converted = convert_paystring(data_36, columns)
data_36_converted = pd.read_csv('data_36_converted.csv', index_col=0)

# In[ ]:
# data_36_converted.to_csv('data_36_converted.csv')


# # Step 3:
# ### Matching Macro Data
# Match the macro data with the loan-specific data by column 'Date

# In[117]:


usswap3.index = pd.to_datetime(usswap3.index)
unemployment.index = pd.to_datetime(unemployment.index)


# ### Assumption : Convert unemployment rate, interest rates
# Linear Interpolation for unemployment rates from quarterly to daily
#
# Linear Interpolation for interest rates for weekends holidays
#

# In[118]:


# Dataframe dates are ascending
def linear_interpolation(df, freq='D'):
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='D'), fill_value="NaN")
    df = df.astype(float)
    df = df.interpolate(method='linear', axis=0).ffill().bfill()
    return df


# In[119]:


usswap3 = linear_interpolation(usswap3, 'D')
unemployment = linear_interpolation(unemployment, 'D')


# In[130]:


def merge_macrodata(df, rate, unemployment, month_lag=3):
    df[rate_ticker] = float(0)
    df[unemployment_ticker] = float(0)

    for i in df.index:
        print(i)
        df.at[i, rate_ticker] = usswap3[
            usswap3.index == (df['Date'][i] + relativedelta.relativedelta(months=-month_lag))].values
        df.at[i, unemployment_ticker] = unemployment[
            unemployment.index == (df['Date'][i] + relativedelta.relativedelta(months=-month_lag))].values
    return df


# In[ ]:
# data_regression = merge_macrodata(data_36_converted, usswap3, unemployment)
data_regression = pd.read_csv('data_regression.csv', index_col=0)
# data_regression.to_csv('data_regression.csv')

# In[ ]:
housing_status = data_regression['Housing Status']
housing_status_numerical = pd.get_dummies(housing_status)
grade = data_regression['Grade']
grade = pd.get_dummies(grade)
data_regression = pd.concat([data_regression, housing_status_numerical], axis=1)
data_regression = pd.concat([data_regression, grade], axis=1)

# In[ ]:
x_cols = [
    'Original Loan Balance',
    'Interest Rate',
    'Payment-To-Income',
    'Original FICO',
    'EHUPUS',
    'USSWAP3']
x_cols = x_cols + list(housing_status_numerical.columns) + list(grade.columns)
y_col = ['Status']

# In[ ]:
data_regression = data_regression.replace(['<null>'], 0)
for col in x_cols:
    data_regression[col] = data_regression[col].astype(float)

# In[ ]:
NUM_ENTRIES = data_regression.shape[0]
alpha = 0.75

# In[ ]:
from sklearn.linear_model import LogisticRegression

ytrain = data_regression[y_col][:int(NUM_ENTRIES * alpha)]
Xtrain = np.matrix(data_regression[x_cols][:int(NUM_ENTRIES * alpha)])
lr = LogisticRegression().fit(Xtrain, ytrain)

# In[ ]:
Xpred = np.matrix(data_regression[x_cols][int(NUM_ENTRIES * alpha):])
y_test = np.squeeze(data_regression[y_col][int(NUM_ENTRIES * alpha):].T.values)
preds = lr.predict(Xpred)

test_accuracy = 0
for i in range(len(preds)):
    if (preds[i] == y_test[i]):
        test_accuracy = test_accuracy + 1
print('Test Accuracy : ', test_accuracy / len(preds))

################################################
##
##  Sklearn One
##
################################################
# In[ ]:
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()





