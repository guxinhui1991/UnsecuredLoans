# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

from collections import Counter
from dateutil import relativedelta
from matplotlib import pyplot as plt

# In[2]:


columns = ['Loan ID',
           'Original Loan Balance',
           'Current Loan Balance',
           'Loan Status',
           'Principal Payment (Cumulative)',
           'Interest Payment (Cumulative)',
           'Late Fees Paid (Cumulative)',
           'Loan Origination Date',
           'Loan Maturity Date',
           'Next Payment Date',
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
           'Current FICO',
           'Housing Status',
           'Employment Status',
           'Employment Length',
           'EOM Paystring']

rate_ticker, unemployment_ticker = 'USSWAP3', 'EHUPUS'
data_path = 'Data/'
data = pd.read_csv(data_path + 'avant-platform-as-of-2017-09-30-with-paystring.csv', usecols=columns)

# In[3]:


data_36 = data[data.Term == 36].copy()
loan_id_mapping = pd.DataFrame(data_36['Loan ID'])
data_36['Loan ID'] = loan_id_mapping.index
data_36['Date'] = pd.to_datetime(data_36['Loan Origination Date'])

# Reindexing
data_36 = data_36.reset_index(drop=True)

num_loans_Total = len(data)
num_loans_36 = len(data_36)


# # Step 1:
# ### Cleaning the payment string
# All payment string with duplicate 'P' and 'O' are trimmed since we are only interested in the termination month.
# I.e, Sample payment string 'CCCCCPPPPP' will become 'CCCCCP'
#
# ### Sanity Check:
# Passed Data Sanity Check. No 'P' appeared after 'O' or vice versa.
#
#

# In[135]:
#
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


# In[5]:


data_36 = clean_paystring(data_36)


# # Step 2:
# ### Stacking Each Loan
# Converting each loan and stack up from the payment string and months.
# Adding age variable for each loan

# In[14]:


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
            loan_i_copy['Next Status'] = paystring[i + 1]
            loans[count] = loan_i_copy
            count = count + 1
    df_converted = pd.DataFrame.from_dict(loans, orient='index')
    # df_converted = df_converted.reset_index(drop=True)
    return df_converted


## Remove Loans starting with 'O'
## Running Time : 3 minutes
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


##########################################################################################
##########################################################################################
##########################################################################################
###
### Location : 251732:251754
### Total 22 loans starting with 'O'
###
###
##########################################################################################
##########################################################################################
##########################################################################################

data_36 = remove_charge_off(data_36, 'EOM Paystring', 'Loan ID')

# In[]:
data_36_converted = convert_paystring(data_36, columns)
# data_36_converted = pd.read_csv('data_36_converted.csv', index_col=0)
data_36_converted['Date'] = pd.to_datetime(data_36_converted['Date'])

# ### Sanity Check for Payment Strings;
# Result as printed

# In[9]:


paystrings = data_36_converted['EOM Paystring']
Loans_first_payment_count = {'P': 0, 'O': 0, '1': 0}
for i in paystrings.index:
    if (paystrings[i][0] != 'C'):
        Loans_first_payment_count[paystrings[i][0]] = Loans_first_payment_count[paystrings[i][0]] + 1
print(Loans_first_payment_count)

# In[]:
## Step 3:
# ### Matching Macro Data
# Match the macro data with the loan-specific data by column Date


usswap3 = pd.read_csv(data_path + rate_ticker + '.csv', index_col=0)
unemployment = pd.read_csv(data_path + unemployment_ticker + '.csv', index_col=0)
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


usswap3 = linear_interpolation(usswap3, freq='D')
unemployment = linear_interpolation(unemployment, freq='D')

unemployment['Date'] = pd.to_datetime(unemployment.index)


# In[130]:


def merge_macrodata(df, rate, unemployment, month_lag=3):
    #    df[rate_ticker] = float(0)
    #    df[unemployment_ticker] = float(0)

    #    for i in df.index:
    #        df.at[i, rate_ticker] =  usswap3[df['Date'][i]+relativedelta.relativedelta(months=-month_lag))].values
    #        df.at[i, unemployment_ticker] =  unemployment[unemployment.index==(df['Date'][i]+relativedelta.relativedelta(months=-month_lag))].values
    #    return df

    df[unemployment_ticker] = float(0)
    for i in df.index:
        df.at[i, unemployment_ticker] = unemployment.loc[df['Date'][i]].values
    return df


# In[88]:

data_clean = data_36_converted.join(unemployment['EHUPUS'], on='Date', how='left')

# data_regression = merge_macrodata(data_36_converted, usswap3, unemployment)
# data_regression = pd.read_csv('data_regression.csv', index_col=0)


# # Step 4:
# ### Multinomial Logistic Regression
# Data: data_regression

# In[89]:
#
#
#

housing_status = data_clean['Housing Status']
housing_status_numerical = pd.get_dummies(housing_status)
grade = data_clean['Grade']
grade = pd.get_dummies(grade)
data_clean = pd.concat([data_clean, housing_status_numerical], axis=1)
data_clean = pd.concat([data_clean, grade], axis=1)

# In[90]:


x_cols = [
    'Original Loan Balance',
    'Interest Rate',
    'Payment-To-Income',
    'Original FICO',
    'EHUPUS']
x_cols = x_cols + list(housing_status_numerical.columns) + list(grade.columns)
y_col = ['Status']

# In[]:
# ### Detect NULL Entries
# Convert all values to numerical.
#
# Original Data will contain string $\text{'<null>'}$ which is a string but not a null value

# data_regression = data_regression.replace(['<null>'], 0)
null_index = data_clean[data_clean['Original FICO'] == '<null>'].index
data_clean = data_clean.drop(null_index)

# In[]:
##  Clean Data for regression
##  Continue here or import in stage 4
data_clean.to_csv('data_regression.csv')

##########################################################################################
##########################################################################################
##########################################################################################
###
###
### Code below
### Deprecated on Feb 14th, 2018
###
###
###
##########################################################################################
##########################################################################################
##########################################################################################
# In[93]:

for col in x_cols:
    data_regression[col] = data_regression[col].astype(float)


# ### Categorize Numerical Columns
# Convert numerical values into buckets.
#
# Columns converted {<br>
# 'Original Loan Balance',<br>
# 'Principal Payment (Cumulative)',<br>
# 'Interest Payment (Cumulative)',<br>
# 'Interest Rate',<br>
# 'APR',<br>
# 'Payment-To-Income',<br>
# 'Original FICO',<br>
# 'EHUPUS',<br>
# 'USSWAP3'<br>}

# In[100]:

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


# In[101]:
cols_numerical = [
    'Original Loan Balance',
    'Interest Rate',
    'Payment-To-Income',
    'Original FICO',
    'EHUPUS',
    'USSWAP3']

# In[102]:


data_regression, columns_add = convert_to_categorical(data_regression, cols_numerical, 5)
x_cols_all_categories = ['Other',
                         'Own - Mortgage',
                         'Rent',
                         'A',
                         'B',
                         'C',
                         'D'] + columns_add


# ###  Logistic Regression  Code
# Data: data_regression
# Columns Used {<br>
# 'Original Loan Balance',<br>
# 'Principal Payment (Cumulative)',<br>
# 'Interest Payment (Cumulative)',<br>
# 'Interest Rate',<br>
# 'APR',<br>
# 'Payment-To-Income',<br>
# 'Original FICO',<br>
# 'EHUPUS',<br>
# 'USSWAP3'<br>}

# In[107]:
# Clean data
# Combing delinquency categories

def combine_delinquencies(df, column_name):
    data = df[column_name].values
    for i in range(df.shape[0]):
        if (data[i, 0].isdigit()):
            if (data[i, 0] == '1'):
                data[i, 0] = 'C'
            else:
                data[i, 0] = 'DQ'
        elif (data[i, 0] == 'O'):
            data[i, 0] = 'D'
    return data


y_status = combine_delinquencies(data_regression, y_col)

# In[107]:

NUM_ENTRIES = data_regression.shape[0]
alpha = 0.75

ytrain = y_status[:int(NUM_ENTRIES * alpha)]
Xtrain = np.matrix(data_regression[x_cols_all_categories][:int(NUM_ENTRIES * alpha)])
x_test = data_regression[x_cols_all_categories][int(NUM_ENTRIES * alpha):]
y_test = y_status[int(NUM_ENTRIES * alpha):].T
