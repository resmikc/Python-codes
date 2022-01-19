#!/usr/bin/env python
# coding: utf-8

# ### Loading libraries

# In[1]:


#Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### loading dataset

# In[2]:


#Load Data
cancerdata = pd.read_csv(r'C:\Users\resmi\Downloads\cancer.csv')
cancerdata


# ### dropping 'id' column

# In[3]:


cancerdata2=cancerdata.drop('id',axis=1)
cancerdata2.head()


# ### Missing values

# In[4]:


cancerdata2.isnull().sum()


# ### Key Characteristics

# In[5]:


cancerdata2.describe()


# ### Correlation

# In[6]:


cancerdata2corr=cancerdata2.corr()
cancerdata2corr


# ### Data Processing

# In[7]:


#Create x and y variables
X = cancerdata2.drop('Class',axis=1).to_numpy()
y = cancerdata2['Class'].to_numpy()

#Create Train and Test datasets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size = 0.20,random_state=100)

#Scale the data
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train2 = sc.fit_transform(X_train)
x_test2 = sc.transform(X_test)


# ### Neural Network

# In[8]:


#Script for Neural Network
from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(9,4,2),
                    activation='relu',solver='adam',
                    max_iter=10000,random_state=100)  
mlp.fit(x_train2, y_train) 
predictions = mlp.predict(x_test2) 

#Evaluation Report and Matrix
from sklearn.metrics import classification_report, confusion_matrix  
target_names=['2(benign)','4(malignant)']
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions,target_names=target_names)) 


# In[ ]:




