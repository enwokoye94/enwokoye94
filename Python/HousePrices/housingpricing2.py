import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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
# print(train.columns)

# visulazing the data
fig = plt.figure()
plt.subplot(1,2,1)
plt.hist(train.SalePrice, color='red')
plt.title('Histogram of Sales Price')
plt.ylabel('Frequency of Price')

# hist of log of Sale Price
plt.subplot(1,2,2)
plt.hist(np.log(train.SalePrice), color='blue')
plt.title('Histogram of the Log of Sales Price')
plt.ylabel('Frequency of Price')

# plt.show()

# organizing the data starting with nulls
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns =['Null Count']
nulls.index.name ='Feature'

### cleaning the data
def clean_data(df):
    df.replace('?', np.nan, inplace=True)
    df.dropna(axis=1, inplace=True)
    print('Shape of', df.name, 'after dropping NaN: ', df.shape)
    return df 
clean_data(train)
##clean_data(test)

# categorical features
categoricals = train.select_dtypes(exclude=[np.number])

cat_enc=pd.get_dummies(train, drop_first=True)
# print(type(cat_enc))

train_enc=pd.concat([train.select_dtypes(include=[np.number]), cat_enc], axis=1)
print('Shape after creating dummy variables: ', train_enc.shape)
# print(type(train_enc))

labels=train_enc['SalePrice']

# Feature Selection
sel=fs.VarianceThreshold(threshold=(.8*(1-.8)))
Features_reduced = sel.fit_transform(train_enc)
print('reduced features shape: ', Features_reduced.shape)
##
# Building a linear Model

y=np.log(train.SalePrice)
X=train_enc.drop(['SalePrice', 'Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

# fitting the model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

pred=model.predict(X_test)

actual_values = y_test
plt.figure()
plt.scatter(pred, actual_values, alpha=.25,

            color='b')  # alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()

