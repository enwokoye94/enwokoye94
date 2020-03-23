#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import numpy.random as nr

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn import linear_model
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC

import sklearn.metrics as sklm
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
from sklearn import feature_selection as fs
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

import scipy.stats as ss
from scipy.stats import boxcox, norm, skew
from scipy.special import boxcox1p


import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')


# In[67]:


# reading and observing the data

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
test.name = 'Testing Data'
train.name = 'Training Data'

merged = pd.concat([train, test], axis=0, sort=True)
# we see later taking log gives more linear relationship
merged['LogSalePrice'] = np.log(merged['SalePrice'])

num_merged = merged.select_dtypes(include=np.number)
cat_merged = merged.select_dtypes(include='object')

print('Shape of training data: ', train.shape)
print('Shape of testing data: ', test.shape)
print('Shape of merged data:', merged.shape)
# print(train.head())
print(merged.select_dtypes(include=np.number).columns.values)


# In[68]:


# visulazing the data
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(merged.SalePrice, color='red')
plt.title('Histogram of Sales Price', fontsize=17)
plt.ylabel('Frequency of Price', fontsize=17)

# hist of log of Sale Price
plt.subplot(1, 2, 2)
plt.hist(np.log(merged.SalePrice), color='green')
plt.title('Histogram of the Log of Sales Price', fontsize=17)
plt.ylabel('Frequency of Price', fontsize=17)


# In[69]:


# Function for creating scatter of numerical categories vs price
print(num_merged.shape)


def scatter_plot(df, variables, n_cols, n_rows):
    fig = plt.figure(figsize=(15, 45))
    for i, var_name in enumerate(variables):
        # print(var_name)
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        plt.scatter(x=df[var_name], y=num_merged['LogSalePrice'], alpha=.2)
        plt.subplots_adjust(hspace=.5)
        ax.set_title('{} vs. Log of SalePrice'.format(var_name))


scatter_plot(merged, num_merged, 3, 15)


# In[70]:


# we can see linear relationships in scatters above the ones that are are verticle should be categorized as categorical
# looking at the correlation matrix

corr = train.corr()
fig, ax = plt.subplots(figsize=(15, 9))
sns.heatmap(corr, linewidths=.5, vmin=0, vmax=1, square=True)


# In[71]:


# we can make a more consise corr matrix based on sales price

k = 10  # number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[72]:


'''From looking at heat map we see the top 10 correlations to Sale Pirce
further observation show GarageCars and GarageArea are closesly correlated and logically
that makes sense. large number cars more likely to have larger garage area so we can ommit it

'''


# In[73]:


# further observation of data to spot outliers

# From observing data taking the log of sales prics provides more normal distribution
def scatter_plot(x, y, Title, font_size, color):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.scatter(x, y, c=color)
    ax2.scatter(x, np.log(y), c=color)
    ax1.set_title(Title, fontsize=int(font_size))
    ax2.set_title('Garage Living Area vs. Log of Price', fontsize=17)
    plt.show()


scatter_plot(merged['GrLivArea'], merged['SalePrice'],
             'Garage Living Area vs. Price', 17, 'red')


# In[74]:


# dropping outliers from scatter plots above

train.drop(train[train['GrLivArea'] > 4000].index, inplace=True)
train.reset_index(drop=True, inplace=True)
scatter_plot(train['GrLivArea'], train['SalePrice'],
             'Garage Living Area vs. Price', 17, 'green')

train.drop(train[train['TotalBsmtSF'] > 3000].index, inplace=True)
train.reset_index(drop=True, inplace=True)
scatter_plot(train['TotalBsmtSF'], train['SalePrice'],
             'TotalBsmtSF vs. Price', 17, 'red')

train.drop(train[train['YearBuilt'] < 1900].index, inplace=True)
train.reset_index(drop=True, inplace=True)
scatter_plot(train['YearBuilt'], train['SalePrice'],
             'YearBuilt vs. Price', 17, 'grey')

y_train = train['SalePrice']


# In[75]:


# imputing missing variables
# for categorical variables mode-imputation is performed
# for numerical variable is usually mean-imputation when symmetric, and median for for skewed

for col in train.columns:
    if 'SalePrice'in col:
        train.drop('SalePrice', axis=1, inplace=True)
    else:
        train = train

merged = pd.concat([train, test], axis=0, sort=True)

# print(train.shape)
# print(test.shape)
print(merged.shape)

merged.loc[:, ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']] = merged.loc[:, [
    'MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']].astype('object')

# columns with missing numbers
missing_cols = merged.columns[merged.isnull().any()].values
print('There are {} features with missing falues'.format(
    missing_cols.shape[0]))
rotation = 90
# plot of missing features and their missing values
missing_col_val = len(
    merged)-merged.loc[:, np.sum(merged.isnull()) > 0].count()
x = missing_col_val.index
y = missing_col_val
title = 'Variables with Missing Values'
fig, ax = plt.subplots(figsize=(16, 6))
plt.yticks(fontsize=20)
ax.scatter(x, y, s=700, c='green')
plt.show()


# In[76]:


[print('Percentage of missing values: \n\n', (missing_col_val/(merged.shape[0])*100)
       .sort_values(ascending=False))
 ]
'''
## looking at missing values we usually drop cols with large ammounts
of missing data, however from data description we can treat many columns
as 0 or none instead of NaN. '''

# imputing (filling in) None for NaN
none_cols = merged[['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                    'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual',
                    'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrType']]

for i in none_cols.columns:
    merged[i].fillna('None', inplace=True)

# imputing categorical null cols for the mode
mode_cols = merged[['Electrical', 'MSZoning', 'Utilities', 'Exterior1st',
                    'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']]

for i in mode_cols.columns:
    merged[i].fillna(merged[i].mode()[0], inplace=True)

# imputing discrete or continuous numerical variables by their medians
median_cols = merged[['BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'MasVnrArea',
                      'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']]

for i in median_cols.columns:
    merged[i].fillna(merged[i].median(), inplace=True)


# In[77]:


# viewing merged df after dealing with most missing values
missing_col_val = len(
    merged)-merged.loc[:, np.sum(merged.isnull()) > 0].count()
[
    print('Percentage of missing values: \n\n', (missing_col_val/(merged.shape[0])*100)
          .sort_values(ascending=False))
]

# since missing % is close to 15 we will impute by median grouped by some other variable
# we need convert categorical var into numerical
df = merged.drop(['Id', 'LotFrontage'], axis=1)
le = LabelEncoder()
df = df.apply(le.fit_transform)
df.head(3)

# Inserting LotFrontage and setting it as index
df['LotFrontage'] = merged['LotFrontage']
df = df.set_index('LotFrontage').reset_index()
df.head(3)


# correlation of df
corr = df.corr()
display(corr['LotFrontage'].sort_values(ascending=False)[:5])
display(corr['LotFrontage'].sort_values(ascending=False)[-5:])

# since BldgType is most neg correlated we will impute median with groupby using BldgType

merged['LotFrontage'] = merged.groupby(
    ['BldgType'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
print('Missing variables left ', merged.columns[merged.isna().any()].values)


# In[135]:


'''
## Dealing with Skewedness of Distributions
## We already know our target (SalePrice is Skewed, we can account for that,
and other skewed variables)
'''


def hist_plot(x, title, xlabel, ylabel):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.hist(x, bins=20, color='orange')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.subplot(122)
    plt.hist(np.log(x), bins=20, color='magenta')
    title2 = title.replace('SalePrice', 'Log of SalePrice')
    plt.title(title2)
    plt.xlabel(xlabel.replace('SalePrice', 'Log of SalePrice'))
    plt.show()


hist_plot(y_train, 'Histogram of SalePrice', 'SalePrice', 'Abs Freq')

# Calculating the skewedness of variables in dataframe
skew_num = pd.DataFrame(data=merged.select_dtypes(
    include=['int64', 'float64']).skew(), columns=['Skewness'])
skew_num_sorted = skew_num.sort_values(by='Skewness', ascending=False)

print(skew_num_sorted)
# We can also visualize this in a box plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(skew_num_sorted.index, skew_num_sorted['Skewness'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


'''
## For loop for plotting mult hist of variables not done yet


def hist_plots(df, variables, n_cols, n_rows):
    for i, var_name in enumerate(variables[:3]):
        plt.figure(figsize=(15,5))
        # print(var_name)
        plt.subplot(121)
        plt.hist(x=df[var_name], bins=20,)
        plt.subplots_adjust(hspace = .5)

        plt.subplot(122)
        plt.hist(x=np.log(df[var_name]>0), bins = 20)

num_vars = merged.select_dtypes(include = np.number).astype('float64')
variables = num_vars.columns.values
print(num_vars.dtypes)
hist_plots(merged, variables, 5 ,5)'''


# In[ ]:


# In[ ]:


# In[ ]:


# https://www.kaggle.com/vikassingh1996/extensive-data-preprocessing-and-modeling
#
# https://www.kaggle.com/eiosifov/top-20-with-data-cleaning-only-elasticnet
#
# https://www.kaggle.com/eiosifov/top-8-without-feature-engineering
#
