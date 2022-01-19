#!/usr/bin/env python
# coding: utf-8

# ### loading libraries

# In[1]:


#Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### loading dataset

# In[4]:


#Load Dataset
data=pd.read_csv(r'D:\Spring 2021\Data 1200\final project\illnessstudy.csv')
data


# ### data types

# In[5]:


#Using pandas "dtypes" function to get the data type of all columns in the dataframe

data.dtypes


# ### missing values

# In[6]:


#checking missing values in the dataset

data.isnull().sum()


# ### key characteristics

# In[7]:


#Overall Statistics of the Dataset

data.describe()


# ### correlation

# In[8]:


#Using pandas "corr" method to know correlation among variables in the dataset

datacorr = data.corr()
datacorr


# ### correlation visualization

# In[9]:


#Correlation Visualization

plt.figure(figsize=(22,16))
sns.heatmap(datacorr, annot = True, cmap = "Blues")


# ### data Preprocessing

# In[11]:


#Create x and y variables
x=data.drop('diagnosis', axis=1).to_numpy()
y=data['diagnosis'].to_numpy()

#Create Training and Test Datasets
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, stratify=y,test_size=0.2,random_state=100)

#Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)


# ### SVM & NB

# In[12]:


#Script for SVM and NB
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix  

for name,method in [('SVM', SVC(kernel='linear',random_state=100)),
                    ('Naive Bayes',GaussianNB())]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    target_names=['M','B']
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict,target_names=target_names))


# ### Decision Tree

# In[13]:


#Script for Decision Tree
from sklearn.tree import DecisionTreeClassifier  

for name,method in [('DT', DecisionTreeClassifier(random_state=100))]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    target_names=['M','B']
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict,target_names=target_names))  


# In[ ]:




