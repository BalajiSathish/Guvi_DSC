#!/usr/bin/env python
# coding: utf-8

# #### Load Packages

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# #### Load Data

# In[2]:


df = pd.read_csv("E:\\Python\\Dataset\\Jan_2019_ontime.csv")


# In[3]:


df


# #### Concating Arrival and depaarture in one column

# In[5]:


df['DELAYED'] = (df['ARR_DEL15'].astype(bool) | df['DEP_DEL15'].astype(bool)).astype(int)


# #### Removing unwanted columns to decrease dimensionality

# In[6]:


df.drop(['OP_CARRIER_AIRLINE_ID','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID',
         'DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','Unnamed: 21','OP_CARRIER',
         'ARR_DEL15','DEP_DEL15','CANCELLED', 'DIVERTED'],axis=1, inplace=True)


# #### Getting the object columns and float columns
# 

# In[7]:


str_columns =  list(df.dtypes[df.dtypes == 'object'].index)
print(f"The number of string columns is: {len(str_columns)}")
    
num_columns = list(df.drop(str_columns,axis=1))
print(f"The number of numeric columns is: {len(num_columns)}")


# #### Percentage of null values on the whole data

# In[8]:


(df.isna().sum() / df.shape[0]) * 100


# #### Drop Null values

# In[9]:


df.dropna(inplace=True)
df.isna().any()


# In[11]:


df.duplicated().sum()


# In[14]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# In[70]:


import numpy as np
from sklearn.metrics import accuracy_score,r2_score,confusion_matrix,classification_report,recall_score,confusion_matrix,mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split,KFold,cross_val_score,ShuffleSplit,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
import re
from pandas_profiling import ProfileReport
pd.set_option("display.precision", 6)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[18]:


df.hist(bins=100,figsize=(20,15))
plt.grid()
plt.show()


# In[27]:


def check_outliers(df):
    for col in num_columns:
        fig,ax = plt.subplots(figsize=(2,3))
        plt.grid()
        sns.boxplot(x=df[col])
        plt.show()


# In[28]:


check_outliers(df)


# In[33]:


def replace_outliers(df):
    for column in num_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        up_lim = q3 + 1.5 * iqr
        low_lim = q1 - 1.5 * iqr 
        col_mean = df[column].mean()
        out_up = (df[column] > up_lim)
        out_down = (df[column] < low_lim)
        df[column] = np.where((df[column] > up_lim)| (df[column] > low_lim) , col_mean , df[column] )
    return(df)
replace_outliers(df)


# In[44]:


def encode_categories(features):
    lb_make = preprocessing.LabelEncoder()
    for i in range(len(features)):
        df[features[i]] = lb_make.fit_transform(df[features[i]])


# In[36]:


check_outliers(df)


# In[45]:


encode_categories(['OP_UNIQUE_CARRIER' , 'ORIGIN' , 'DEST' , 'DEP_TIME_BLK'])


# In[46]:


fig,ax = plt.subplots(figsize=(15,9))
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
plt.show()


# In[64]:


x = df.drop('DELAYED',axis=1)
y = df['DELAYED']
lb_make = preprocessing.LabelEncoder()
y_trans = lb_make.fit_transform(y)


# In[65]:


x_train, x_test, y_train, y_test = train_test_split(x, y_trans, test_size = 0.3,random_state=42)


# In[66]:


models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))


# In[67]:


for name, model in models:
    
    print(name)

    
    trained_model = model.fit(x_train, y_train)
        
    predictions = trained_model.predict(x_test) 
    
    print(f"train score: {accuracy_score(y_train, trained_model.predict(x_train))}\n")
    
    print(f"test score: {accuracy_score(predictions,y_test)}\n\n")


# In[106]:


# old accuracy
lr = LogisticRegression()
lr.fit(x_train,y_train)
trained_model = lr.fit(x_train, y_train)
trained_model.fit(x_train, y_train)  
predictions = trained_model.predict(x_test) 
lr_old_train = accuracy_score(y_train, trained_model.predict(x_train))
pred = lr.predict(x_test)
lr_old_accur = accuracy_score(y_test, pred)

# new accuracy
lr = LogisticRegression(C = 10)
lr.fit(x_train,y_train)
trained_model = lr.fit(x_train, y_train)
trained_model.fit(x_train, y_train)  
predictions = trained_model.predict(x_test) 
lr_new_train = accuracy_score(y_train, trained_model.predict(x_train))
pred = lr.predict(x_test)
lr_new_accur = accuracy_score(y_test, pred)

print("The Training Accuracy of LogisticRegression Model before tuning: " + str(rf1_old_train))
print("The Testing Accuracy of LogisticRegression Model before tuning: " + str(rf1_old_accur))
print()
print("The Training Accuracy of LogisticRegressiont Model after tuning: " + str(rf1_new_train))
print("The Testing Accuracy of LogisticRegression Model after tuning: " + str(rf1_new_accur))
print()

