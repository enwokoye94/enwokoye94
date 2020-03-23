import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
from sklearn import feature_selection as fs
from sklearn import metrics
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math


## reading and observing the data

test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')
test.name ='Testing Data'
train.name = 'Training Data'

print('Shape of training data: ', train.shape)
print ('Shape of testing data: ', test.shape)
# print(train.head())
##print(train.columns)
##print(train.isna().sum())
##print(test.isna().sum())

### cleaning the data
def clean_data(df):
    df.replace('?', np.nan, inplace=True)
    df.dropna(axis=1, inplace=True)
    print('Shape of', df.name, 'after dropping NaN: ', df.shape)
    return df 
clean_data(train)
clean_data(test)

labels = train['SalePrice']

# next need to create model matrix taking into account creating dummy variabls for categorical features
'''  this is a three step process:
1. Encode the categorical string variables as integers.
2. Transform the integer coded variables to dummy variables.
3. Append each dummy coded categorical variable to the model matrix.
'''

def encode_string(cat_features):
    # First encode the strings to numeric categories
    # Converts categorical strings variables in integers
    enc = preprocessing.LabelEncoder()
    # Fits user input 'cat_features' to labelencoder conversion
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    # Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder(categories='auto')
    encoded = ohe.fit(enc_cat_features.reshape(-1, 1))
    return encoded.transform(enc_cat_features.reshape(-1, 1)).toarray()

cat_columns = train.drop(columns='SalePrice').select_dtypes('object')


    

##Features = encode_string(cat_columns)

##tot_data = test.merge(train, on='Id', how='outer')
##tot_data.replace('?', np.nan, inplace=True)
##print('The shape of the combined dataframe: ', tot_data.shape)
####tot_data.dropna(thresh=len(tot_data)*0.5, inplace=True, axis=0)
